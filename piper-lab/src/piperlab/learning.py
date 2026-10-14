"""ACT training and shadow replay for Piper Lab on LeRobot v0.6.1.

Deep-learning imports (torch / LeRobot) are kept lazy: training runs the
official ``lerobot.scripts.lerobot_train`` CLI as a subprocess, and shadow
replay runs as a ``python -m piperlab.learning --shadow-worker`` subprocess.
The parent process therefore never needs torch, and CUDA device selection by
GPU UUID happens in the child environment before torch is imported there.

No dataset downloads, no Hub pushes, no wandb, no WAN inference: the child
environment pins ``HF_HUB_OFFLINE=1`` and ``WANDB_MODE=disabled`` and the CLI
is invoked with ``--policy.push_to_hub=false --wandb.enable=false``. Shadow
replay only reads a dataset and a checkpoint; it never touches hardware.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

LEROBOT_TRAIN_MODULE = "piperlab.train_worker"
DEFAULT_JOB_NAME = "piperlab_act"
REPO_ROOT_FALLBACK_PREFIX = "local"


class LearningError(RuntimeError):
    """Raised when a train/shadow subprocess fails or dependencies are missing."""


# ---------------------------------------------------------------------------
# environment helpers (no torch here)


def _nvidia_smi_gpu_table() -> list:
    """Return [(index, uuid), ...] from nvidia-smi, or [] when unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    table = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "," not in line:
            continue
        index, uuid = (part.strip() for part in line.split(",", 1))
        if index.isdigit():
            table.append((index, uuid))
    return table


def _resolve_cuda_visible_devices(gpu_uuid: str) -> str:
    """Validate and return the UUID itself; CUDA and NVML indices may differ."""
    table = _nvidia_smi_gpu_table()
    if not table:
        raise LearningError("cannot select GPU by UUID: nvidia-smi is unavailable")
    wanted = gpu_uuid.strip().lower()
    for index, uuid in table:
        if uuid.lower() == wanted:
            return uuid
    known = ", ".join(f"{i}:{u}" for i, u in table)
    raise LearningError(f"GPU UUID {gpu_uuid!r} not found; visible GPUs: {known}")


def _child_env(gpu_uuid: str | None) -> tuple:
    """Build the subprocess environment; returns (env, CUDA_UUID_or_None)."""
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    cuda_index = None
    if gpu_uuid:
        cuda_index = _resolve_cuda_visible_devices(gpu_uuid)
        env["CUDA_VISIBLE_DEVICES"] = cuda_index
    return env, cuda_index


def _default_device() -> str:
    return "cuda" if _nvidia_smi_gpu_table() else "cpu"


def _dataset_repo_id(dataset: Path) -> str:
    sidecar = dataset / "piperlab_export.json"
    if sidecar.is_file():
        with open(sidecar, encoding="utf-8") as handle:
            repo_id = json.load(handle).get("repo_id")
        if repo_id:
            return str(repo_id)
    return f"{REPO_ROOT_FALLBACK_PREFIX}/{dataset.name}"


def _check_lerobot_dataset(dataset: Path) -> None:
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"not a LeRobot dataset root (missing meta/info.json): {dataset}. "
            "Run data.export_lerobot first."
        )


def _tail(text: str, max_lines: int = 40) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def dataset_split(dataset: Path, split: str) -> list[int]:
    meta = json.loads((dataset / "piperlab_export.json").read_text(encoding="utf-8"))
    train, val = meta["splits"]["train"], meta["splits"]["val"]
    if set(train) & set(val):
        raise LearningError("Training and validation episodes overlap")
    result = meta["splits"][split]
    if not result:
        raise LearningError(f"No episodes in {split} split")
    return result


def _fail_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists, refusing to overwrite: {path}")


# ---------------------------------------------------------------------------
# training via the official LeRobot v0.6.1 train CLI


def _build_train_command(
    dataset: Path,
    output_dir: Path,
    *,
    repo_id: str,
    steps: int,
    batch_size: int,
    device: str,
    job_name: str,
    seed: int,
    resume_from: Path | None,
    extra_args: list | None,
) -> list:
    cmd = [
        sys.executable,
        "-m",
        LEROBOT_TRAIN_MODULE,
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset}",
        f"--dataset.episodes={json.dumps(dataset_split(dataset, 'train'))}",
        "--policy.type=act",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
        # null avoids a torchvision weight download over the WAN; training
        # from scratch is what a 100-step smoke run wants anyway
        "--policy.pretrained_backbone_weights=null",
        "--policy.chunk_size=20",
        "--policy.n_action_steps=1",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        "--num_workers=0",
        "--log_freq=10",
        f"--seed={seed}",
        "--wandb.enable=false",
    ]
    if resume_from is not None:
        cmd += ["--resume=true", f"--config_path={resume_from}"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def train(
    dataset: str,
    output: str,
    steps: int = 100,
    batch_size: int = 4,
    gpu_uuid: str | None = None,
    *,
    resume_from: str | None = None,
    seed: int = 1000,
    job_name: str = DEFAULT_JOB_NAME,
    device: str | None = None,
    timeout_s: float | None = None,
    extra_args: list | None = None,
) -> dict:
    """Train ACT on a local LeRobot dataset via the official v0.6.1 CLI.

    Runs ``python -m lerobot.scripts.lerobot_train`` as a subprocess with
    wandb and Hub push disabled. ``resume_from`` may point at a checkpoint's
    ``train_config.json``, its ``pretrained_model/`` directory, or the
    checkpoint step directory; the run then continues from the saved
    optimizer/scheduler/RNG state (checkpoint resume).
    """
    ds = Path(dataset).resolve()
    _check_lerobot_dataset(ds)
    out = Path(output).resolve()
    _fail_if_exists(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    resume_path = (_resolve_pretrained_model_dir(resume_from) / "train_config.json") if resume_from else None
    if resume_path is not None and not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_from}")

    if gpu_uuid:
        device = "cuda"
    elif device is None:
        device = _default_device()

    env, cuda_index = _child_env(gpu_uuid)
    staging = out.parent / (out.name + f".tmp-{os.getpid()}")
    _fail_if_exists(staging)
    repo_id = _dataset_repo_id(ds)
    cmd = _build_train_command(
        ds,
        staging,
        repo_id=repo_id,
        steps=steps,
        batch_size=batch_size,
        device=device,
        job_name=job_name,
        seed=seed,
        resume_from=resume_path,
        extra_args=extra_args,
    )
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise LearningError(f"training exceeded timeout_s={timeout_s}") from exc
    duration_s = round(time.monotonic() - started, 3)
    if proc.returncode != 0:
        # Preserve partial checkpoints and logs for diagnosis; never delete a failed run.
        out.with_suffix('.failed.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        raise LearningError(
            f"lerobot train CLI exited with code {proc.returncode}.\n"
            f"command: {' '.join(cmd)}\n--- stdout/stderr tail ---\n"
            f"{_tail(proc.stdout)}\n{_tail(proc.stderr)}"
        )

    last_link = staging / "checkpoints" / "last"
    pointer = staging / 'checkpoints' / 'last.txt'
    if pointer.is_file():
        last_link = staging / 'checkpoints' / pointer.read_text(encoding='utf-8').strip()
    checkpoint = last_link / "pretrained_model"
    if not checkpoint.is_dir():
        raise LearningError(
            f"training finished but no checkpoint was found at {checkpoint}; "
            f"stdout tail:\n{_tail(proc.stdout)}"
        )
    step_name = None
    try:  # checkpoints/last is a directory copy or link to the step dir
        step_name = max(
            (p.name for p in (staging / "checkpoints").iterdir() if p.is_dir() and p.name.isdigit()),
            default=None,
        )
    except OSError:
        pass
    os.rename(staging, out)
    report = {
        "output": str(out),
        "checkpoint": str(out / "checkpoints" / (step_name or 'last') / "pretrained_model"),
        "checkpoint_step": step_name,
        "steps": steps,
        "batch_size": batch_size,
        "device": device,
        "gpu_uuid": gpu_uuid,
        "cuda_visible_devices": cuda_index,
        "resumed_from": str(resume_path) if resume_path else None,
        "dataset": str(ds),
        "repo_id": repo_id,
        "training_episodes": dataset_split(ds, "train"),
        "duration_s": duration_s,
        "command": cmd,
        "log_tail": _tail(proc.stdout + proc.stderr, 20),
    }
    (out / 'train.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
    (out / 'piperlab_train.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


# ---------------------------------------------------------------------------
# shadow replay (no hardware, metrics only)


def _resolve_pretrained_model_dir(checkpoint: str) -> Path:
    """Accept a run dir, a checkpoint step dir, or pretrained_model/ itself."""
    path = Path(checkpoint).resolve()
    if path.is_file() and path.name == 'train_config.json':
        path = path.parent
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints" / "last" / "pretrained_model",
    ]
    pointer = path / 'checkpoints' / 'last.txt'
    if pointer.is_file():
        candidates.append(path / 'checkpoints' / pointer.read_text(encoding='utf-8').strip() / 'pretrained_model')
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"no pretrained_model checkpoint (config.json + model.safetensors) under: {checkpoint}"
    )


def shadow(
    dataset: str,
    checkpoint: str,
    output: str,
    gpu_uuid: str | None = None,
    *,
    device: str | None = None,
    max_frames: int | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Replay a dataset against a checkpoint open-loop and report metrics.

    The policy predicts actions from recorded observations only; predicted
    actions are compared with the recorded targets. Nothing is published,
    no robot is connected, no hardware moves.
    """
    ds = Path(dataset).resolve()
    _check_lerobot_dataset(ds)
    pretrained = _resolve_pretrained_model_dir(checkpoint)
    out = Path(output).resolve()
    _fail_if_exists(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if gpu_uuid:
        device = "cuda"
    elif device is None:
        device = _default_device()
    env, cuda_index = _child_env(gpu_uuid)

    report_tmp = out.parent / (out.name + f".worker-{os.getpid()}.json")
    cmd = [
        sys.executable,
        "-m",
        "piperlab.learning",
        "--shadow-worker",
        f"--dataset={ds}",
        f"--checkpoint={pretrained}",
        f"--report={report_tmp}",
        f"--device={device}",
    ]
    if max_frames is not None:
        cmd.append(f"--max-frames={max_frames}")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise LearningError(f"shadow replay exceeded timeout_s={timeout_s}") from exc
    duration_s = round(time.monotonic() - started, 3)
    if proc.returncode != 0:
        report_tmp.unlink(missing_ok=True)
        raise LearningError(
            f"shadow worker exited with code {proc.returncode}.\n"
            f"--- stdout/stderr tail ---\n{_tail(proc.stdout)}\n{_tail(proc.stderr)}"
        )
    with open(report_tmp, encoding="utf-8") as handle:
        worker_report = json.load(handle)
    report_tmp.unlink()

    report = {
        "output": str(out),
        "dataset": str(ds),
        "checkpoint": str(pretrained),
        "device": device,
        "gpu_uuid": gpu_uuid,
        "cuda_visible_devices": cuda_index,
        "duration_s": duration_s,
        "hardware_interaction": "none (open-loop replay on recorded data)",
        **worker_report,
    }
    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, out)
    return report


# ---------------------------------------------------------------------------
# shadow worker entry point (torch / LeRobot imports live in the subprocess)


def _shadow_worker(dataset: Path, checkpoint: Path, device: str, max_frames: int | None) -> dict:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: PLC0415
    from lerobot.policies import make_pre_post_processors  # noqa: PLC0415
    from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: PLC0415

    repo_id = _dataset_repo_id(dataset)
    meta = LeRobotDatasetMetadata(repo_id, root=dataset)
    policy = ACTPolicy.from_pretrained(checkpoint)
    policy.to(device)
    policy.eval()

    processor_source = "checkpoint"
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=str(checkpoint)
    )

    val_episodes = dataset_split(dataset, 'val')
    ds = LeRobotDataset(repo_id, root=dataset, episodes=val_episodes)
    input_keys = set(policy.config.input_features)
    n_frames = len(ds) if max_frames is None else min(len(ds), int(max_frames))

    sum_abs = None
    sum_sq = None
    latencies_ms = []
    evaluated = 0
    skipped = 0
    last_episode = None
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index in range(n_frames):
            item = ds[index]
            episode = int(item['episode_index'])
            if episode != last_episode:
                policy.reset()
                last_episode = episode
            try:
                batch = {
                    key: item[key].unsqueeze(0).to(device)
                    for key in input_keys
                    if key in item and isinstance(item[key], torch.Tensor)
                }
                if len(batch) != len(input_keys):
                    skipped += 1
                    continue
                target = item["action"].detach().cpu().numpy().reshape(-1)[:7].astype("float64")
                obs = preprocessor(batch)
                if device.startswith('cuda'): torch.cuda.synchronize()
                start = time.perf_counter()
                action = policy.select_action(obs)
                action = postprocessor(action)
                predicted = action.detach().cpu().numpy().reshape(-1)[:7].astype("float64")
                if device.startswith('cuda'): torch.cuda.synchronize()
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            diff = predicted - target
            sum_abs = np.abs(diff) if sum_abs is None else sum_abs + np.abs(diff)
            sum_sq = diff**2 if sum_sq is None else sum_sq + diff**2
            evaluated += 1
    if evaluated == 0:
        raise LearningError("shadow replay evaluated zero frames")

    mae = (sum_abs / evaluated).tolist()
    mse = (sum_sq / evaluated).tolist()
    lat = sorted(latencies_ms) or [0.0]

    def percentile(values, pct):
        rank = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
        return values[rank]

    return {
        "policy": "act",
        "frames_evaluated": evaluated,
        "frames_skipped": skipped,
        "per_joint_mae": mae,
        "per_joint_mse": mse,
        "gripper_mae_m": mae[-1],
        "joint_mae_rad": mae[:-1],
        "latency_ms": {
            "mean": float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
            "p50": percentile(lat, 50),
            "p95": percentile(lat, 95),
            "max": lat[-1],
        },
        "processor_source": processor_source,
        "validation_episodes": val_episodes,
        "n_action_steps": policy.config.n_action_steps,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        "first_frame_ms": latencies_ms[0],
        "latency_note": "CUDA synchronized; includes action postprocessing and CPU transfer; first frame included",
        "episodes": int(getattr(meta, "total_episodes", 0)),
        "fps": int(getattr(meta, "fps", 0)),
    }


def main(argv: list | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m piperlab.learning")
    parser.add_argument("--shadow-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.shadow_worker:
        parser.error("piperlab.learning is a library; only the internal --shadow-worker mode is runnable")
    report = _shadow_worker(
        Path(args.dataset), Path(args.checkpoint), args.device, args.max_frames
    )
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
