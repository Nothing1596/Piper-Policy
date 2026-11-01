"""Official trajectory computation and baseline-compatible aggregation."""

import numpy as np


def trajectory_scores(actions, ground_truth, first_success):
    from examples.baselines.lerobot_dataset.trajectory_metrics import TrajectoryMetrics
    prediction = np.asarray(actions)
    # Same split and trimming as eval_dp_imitator.evaluate_with_task_encoder.
    successful = first_success is not None and first_success >= 2
    trimmed = prediction[:first_success] if successful else prediction[:len(ground_truth)]
    if len(trimmed) < 2:
        raise ValueError("TSS needs at least two predicted actions")
    result = TrajectoryMetrics(band_ratio=.15).compute(trimmed, ground_truth)
    suffix = "success" if successful else "fail"
    return {"tss_" + suffix: result["tss"], "ndtw_" + suffix: result["ndtw"]}


def summarize(rows):
    groups = {"overall": rows}
    for row in rows:
        groups.setdefault(row["env_id"], []).append(row)
        groups.setdefault(row["level"], []).append(row)
    output = {}
    for name, group in groups.items():
        valid = [r for r in group if r.get("status") == "complete" and r.get("scored")]
        record = {"expected_episodes": len(group), "scored_episodes": len(valid),
                  "invalid_episodes": sum(r.get("status") == "error" for r in group),
                  "complete": len(valid) == len(group)}
        keys = {k for r in valid for k in r.get("metrics", {})}
        values = {k: [r["metrics"][k] for r in valid if k in r["metrics"]] for k in keys}
        for k in ("native_phase_score", "tss_success", "ndtw_success", "tss_fail", "ndtw_fail"):
            values[k] = [r[k] for r in valid if r.get(k) is not None]
        for key, samples in values.items():
            record[key + "_mean"] = float(np.mean(samples)) if samples else None
            record[key + "_std"] = float(np.std(samples)) if samples else None
        record["SR"] = record.get("success_once_mean")
        # Author baseline: no successful TSS values => wTSS 0, but missing GT => unavailable.
        available = bool(valid) and all(r.get("tss_available") for r in valid)
        record["tss_available"] = available
        record["wtss_mean"] = round((record.get("success_once_mean") or 0.) *
            (record.get("tss_success_mean") or 0.), 6) if available else None
        record["sub_sr"] = None
        output[name] = record
    return output
