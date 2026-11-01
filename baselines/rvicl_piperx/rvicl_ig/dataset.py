"""Resolve official task lists, human test episodes and optional scoring-only GT."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import save


def environment_name(env_id):
    match = re.fullmatch(r"(L[0-3])_(.+)-v(\d+)", env_id)
    if not match:
        raise ValueError(f"Invalid environment id: {env_id}")
    level, name, version = match.groups()
    return level, f"{name}{'L3' if level == 'L3' else ''}-v{version}"


def episode_seed(index):
    # Author's single CPU SyncVectorEnv: auto-reset, then explicit reset.
    return int(np.random.RandomState(2022).randint(2**31, size=2*index+1)[-1])


@lru_cache(maxsize=128)
def episode_metadata(repo, camera):
    import pyarrow.parquet as pq
    repo = Path(repo)
    info = json.loads((repo / "meta/info.json").read_text())
    key = f"observation.images.{camera}"
    if key not in info["features"]:
        raise ValueError(f"Camera {key} missing in {repo}")
    prefix = f"videos/{key}/"
    fields = [prefix + k for k in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    rows = [r for path in sorted((repo / "meta/episodes").glob("chunk-*/file-*.parquet"))
            for r in pq.read_table(path, columns=["episode_index", *fields]).to_pylist()]
    return info, key, fields, rows


def video_source(repo, episode, camera):
    repo = Path(repo)
    info, key, fields, metadata = episode_metadata(repo, camera)
    rows = [r for r in metadata if r["episode_index"] == episode]
    if len(rows) != 1:
        raise ValueError(f"Expected one episode metadata row: {repo}, {episode}")
    row = rows[0]
    path = repo / info["video_path"].format(video_key=key,
        chunk_index=int(row[fields[0]]), file_index=int(row[fields[1]]))
    stat = path.stat()
    start, end = float(row[fields[2]]), float(row[fields[3]])
    if not 0 <= start < end:
        raise ValueError("Invalid human video episode bounds")
    return {"path": str(path.resolve()), "start_s": start, "end_s": end,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_jobs(config, tasks=(), levels=(), limit=0):
    root = Path(config["game_root"]).resolve()
    resolve = lambda key: (root / config[key]).resolve()
    for key in ("episodes_per_env", "max_episode_steps", "max_turns", "max_candidates", "clip_budget"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["episode_start"]) is not int or config["episode_start"] < 0:
        raise ValueError("episode_start must be a nonnegative integer")
    if not 2 <= config["max_keyframes"] <= config["max_candidates"] or config["max_candidates"] < 2:
        raise ValueError("Require 2 <= max_keyframes <= max_candidates")
    if levels and set(levels) - {"L0", "L1", "L2", "L3"}:
        raise ValueError("Unknown difficulty level")
    envs = list(dict.fromkeys(line.split("#", 1)[0].strip()
        for line in resolve("eval_config").read_text().splitlines()))
    envs = [env for env in envs if env]
    if set(tasks) - set(envs):
        raise ValueError("Requested tasks absent from official evaluation list")
    mappings = json.loads(resolve("task_mapping").read_text())["task_mappings"]
    rows = json.loads(resolve("human_config").read_text())
    jobs = []
    for env in envs:
        level, _ = environment_name(env)
        if (tasks and env not in tasks) or (levels and level not in levels):
            continue
        matches = [m for m in mappings if env in m["sim_task_id"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one task mapping: {env}")
        mapping = matches[0]
        matches = [r for r in rows if r["repo_id"] == mapping["human_task_id"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one human evaluation config: {env}")
        row = matches[0]
        # The author's *evaluation* JSON stores its held-out range in 'train'.
        bounds = re.fullmatch(r"(\d+):(\d+)", row.get("train", ""))
        if bounds is None:
            raise ValueError("Human evaluation range must be start:end")
        low, high = map(int, bounds.groups())
        demo_offset = config.get("demo_episode_offset")
        if demo_offset is not None and (type(demo_offset) is not int or not 0 <= demo_offset < high-low):
            raise ValueError(f"Fixed demonstration outside evaluation range for {env}")
        if demo_offset is None and low + config["episode_start"] + config["episodes_per_env"] > high:
            raise ValueError(f"Requested episodes outside evaluation range for {env}")
        for offset in range(config["episode_start"], config["episode_start"] + config["episodes_per_env"]):
            job = dict(config, game_root=str(root), env_id=env, level=level,
                human_id=row["repo_id"], human_episode=low + (offset if demo_offset is None else demo_offset),
                reset_index=offset, seed=episode_seed(offset))
            repo = resolve("human_root") / row["root"]
            job["video"] = video_source(repo, job["human_episode"], config["human_camera"])
            job["robot_repo_id"] = mapping["robot_task_id"][mapping["sim_task_id"].index(env)]
            jobs.append(job)
            if limit and len(jobs) >= limit:
                break
        if limit and len(jobs) >= limit:
            break
    if not jobs:
        raise ValueError("No evaluation jobs selected")
    return jobs


def source_identity(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path in sorted(Path(__file__).parent.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".txt"}:
            digest.update(path.read_bytes())
    root = Path(config["game_root"])
    files = [root / config[key] for key in ("eval_config", "human_config", "task_mapping", "sim_config")]
    files += [root / "mani_skill/utils/wrappers/gymnasium.py",
              root / "mani_skill/envs/tasks/tabletop/utils/reward_utils.py",
              root / "mani_skill/envs/tasks/tabletop/utils/L0_L3_utils.py",
              root / "mani_skill/examples/motionplanning/panda/motionplanner.py",
              root / "examples/baselines/lerobot_dataset/trajectory_metrics.py"]
    files += sorted((root / "mani_skill/envs/tasks/tabletop/dual_tasks").glob("*.py"))
    files += [Path(config["rpent_root"]) / "robots/robotwin" / name for name in ("primitives.py", "tools.py")]
    files += [Path(config["rpent_root"]) / name for name in ("robots/libero/tools.py",
        "rpent/robots/components/sam3_client.py", "rpent/session/base.py",
        "rpent/planner/api_loop.py", "rpent/planner/base.py", "rpent/tools/toolkit.py",
        "robots/robotwin/prompts/system.py")]
    for path in files:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def ground_truth(job, directory):
    """Evaluator only. Reuse the official loader, action units and DTW implementation."""
    from examples.baselines.lerobot_dataset.trajectory_metrics import GTTrajectoryProvider
    root = Path(job["game_root"])
    rows = json.loads((root / job["sim_config"]).read_text())
    matches = [r for r in rows if r["repo_id"] == job["env_id"]]
    if len(matches) != 1:
        raise ValueError("Missing simulation evaluation config for TSS")
    row = dict(matches[0])
    sim_root = root / job["sim_root"]
    # Local downloads use robot_H*_L*, whereas upstream config roots use env ids.
    if not (sim_root / row["root"] / "data").is_dir():
        row["root"] = job["robot_repo_id"]
    if not (sim_root / row["root"] / "data").is_dir():
        raise FileNotFoundError(f"TSS reference dataset missing: {sim_root / row['root']}")
    path = Path(directory) / "scoring_gt_config.json"
    save(path, [row])
    provider = GTTrajectoryProvider(str(path), str(sim_root), action_key="action.qpos_gripper_actions")
    trajectory = provider.sample_gt_trajectory(job["env_id"], seed=job["seed"], normalize=False)
    if trajectory is None or len(trajectory) < 2 or trajectory.shape[1:] != (16,):
        raise ValueError("Official TSS loader did not find a valid 16D reference trajectory")
    return trajectory
