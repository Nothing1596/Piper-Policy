#!/usr/bin/env python3
"""Make unbound Piper jobs from public WatchAct rows, never from oracle plans."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", default="Imitation")
    parser.add_argument("--view", choices=["front", "side", "oblique"], default="front")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("limit must be positive")
    if Path(args.task).name != args.task:
        parser.error("task must be a dataset file stem")
    root = args.data_root.resolve()
    jobs = []
    with (root / "data" / (args.task + ".jsonl")).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["camera_perspective"] != args.view:
                continue
            relative = Path(row["video"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid dataset video path")
            jobs.append({"job_id": row["id"], "kind": "model", "arm_count": 1,
                "video": str(root / "videos" / args.task / relative),
                "language_instruction": row["language_instruction"],
                "camera_perspective": row["camera_perspective"],
                "spatial_reference": row["spatial_reference"],
                "source": {"dataset": "BaiqiL/WatchAct", "original_id": row["original_id"],
                           "bddl_file": row["bddl_file"]}})
            if len(jobs) == args.limit:
                break
    if not jobs:
        raise ValueError("No public rows for the requested view")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(jobs, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Prepared {len(jobs)} unbound single-arm tasks. Bind matching Piper scenes and independent goals before running.")


if __name__ == "__main__":
    main()
