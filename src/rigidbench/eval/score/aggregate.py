from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from rigidbench.core.io import write_json
from rigidbench.core.paths import OutputPaths, model_output_dir
from rigidbench.core.result import Result

PRIMARY_METRICS = (
    "iou",
    "l2",
    "chamfer",
    "ate",
    "si_mse",
    "lpips",
    "ssim",
    "ate3d",
    "iddrift",
    "bgdrift",
)


def aggregate(results: list[Result]) -> dict:
    """Mean, std, and finite-count for every metric that appears in any result."""
    if not results:
        return {}
    out: dict = {"n_samples": len(results), "statistics": {}}
    keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r.metrics:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    for key in keys:
        vals = np.array([r.metrics[key] for r in results if key in r.metrics], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        out["statistics"][key] = {
            "std": float(vals.std()) if len(vals) else None,
            "n": int(len(vals)),
        }
        if len(vals) == 0:
            continue
        out[key] = float(vals.mean())
    return out


def aggregate_by_task(results: list[Result]) -> dict[str, dict]:
    """Group results by task_type and aggregate within each group."""
    by_task: dict[str, list[Result]] = {}
    for r in results:
        by_task.setdefault(r.task_type, []).append(r)
    return {task: aggregate(rs) for task, rs in by_task.items()}


def aggregate_metrics(
    model: str,
    output_dir: str | Path,
    checkpoint: str | None = None,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    official: bool = False,
) -> dict:
    """Read per-sample metrics, optionally validate an exact official result set, and aggregate."""
    paths = OutputPaths(model_output_dir(output_dir, model, checkpoint))
    if not paths.metrics_dir.exists():
        raise FileNotFoundError(f"No metrics dir at {paths.metrics_dir}")

    expected_ids = () if expected_sample_ids is None else expected_sample_ids
    expected = tuple(str(sample_id) for sample_id in expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("expected_sample_ids contains duplicates")
    if official and not expected:
        raise ValueError("Official aggregation requires expected_sample_ids")

    result_files = sorted(paths.metrics_dir.glob("*.json"))
    if not result_files:
        raise RuntimeError(f"No per-sample metrics files under {paths.metrics_dir}")

    by_id: dict[str, Result] = {}
    for path in result_files:
        result = Result.from_json(json.loads(path.read_text()))
        if result.sample_id in by_id:
            raise RuntimeError(f"Duplicate per-sample result for {result.sample_id}")
        by_id[result.sample_id] = result

    if expected:
        expected_set = set(expected)
        missing = [sample_id for sample_id in expected if sample_id not in by_id]
        unexpected = sorted(set(by_id) - expected_set)
        if official and (missing or unexpected):
            problems = []
            if missing:
                problems.append(f"missing {_format_ids(missing)}")
            if unexpected:
                problems.append(f"unexpected {_format_ids(unexpected)}")
            raise RuntimeError("Official results are incomplete: " + "; ".join(problems))
        results = [by_id[sample_id] for sample_id in expected if sample_id in by_id]
    else:
        results = [by_id[sample_id] for sample_id in sorted(by_id)]

    if not results:
        raise RuntimeError("None of the expected samples has a per-sample result")
    if expected:
        _validate_primary_metrics(results)

    agg = aggregate(results)
    agg["model"] = model
    agg["official"] = official
    agg["by_task"] = aggregate_by_task(results)
    payload = {"samples": [r.to_json() for r in results], "aggregated": agg}
    write_json(paths.results, payload)
    return agg


def _validate_primary_metrics(results: list[Result]) -> None:
    invalid: dict[str, list[str]] = {}
    for result in results:
        missing = [metric for metric in PRIMARY_METRICS if metric not in result.metrics]
        if missing:
            invalid[result.sample_id] = missing
    if not invalid:
        return
    preview = "; ".join(f"{sample_id}: {', '.join(metrics)}" for sample_id, metrics in list(invalid.items())[:10])
    if len(invalid) > 10:
        preview += "; ..."
    raise RuntimeError(f"Official results have missing primary metrics ({preview})")


def _format_ids(sample_ids: list[str]) -> str:
    preview = ", ".join(sample_ids[:10])
    if len(sample_ids) > 10:
        preview += ", ..."
    return f"{len(sample_ids)} sample(s) [{preview}]"
