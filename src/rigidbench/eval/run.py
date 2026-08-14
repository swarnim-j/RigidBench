from __future__ import annotations

import argparse
from pathlib import Path

from rigidbench import __version__
from rigidbench.benchmark import EVALUATION_SIZE, PROTOCOL
from rigidbench.core.io import write_json
from rigidbench.core.paths import OutputPaths, model_output_dir
from rigidbench.data import find_data_root

from .samples import filter_samples, load_samples
from .score.aggregate import aggregate_metrics

_SUMMARY_FORMATS = [
    ("iou", "{:.4f}"),
    ("l2", "{:.4f}"),
    ("chamfer", "{:.4f}"),
    ("ate", "{:.2f}"),
    ("si_mse", "{:.4f}"),
    ("lpips", "{:.4f}"),
    ("ssim", "{:.4f}"),
    ("ate3d", "{:.3f}"),
    ("iddrift", "{:.3f}"),
    ("bgdrift", "{:.6f}"),
]


def _format_summary(agg: dict, n: int) -> str:
    """Format aggregated metrics as a compact one-line summary."""
    parts = [f"{k.upper()}={fmt.format(agg[k])}" for k, fmt in _SUMMARY_FORMATS if k in agg]
    return " ".join(parts) + f" (n={n})"


def _select_samples(
    data_dir: str | Path,
    split: str,
    task_type: str | None = None,
    sample_ids: list[str] | None = None,
    max_samples: int | None = None,
):
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero")
    if sample_ids is not None:
        if not sample_ids:
            raise ValueError("The sample ID list is empty")
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("The sample ID list contains duplicates")
    available = load_samples(data_dir, split)
    if not available:
        raise RuntimeError(f"No valid samples found under {Path(data_dir) / split}")
    if sample_ids:
        unknown = sorted(set(sample_ids) - {sample.id for sample in available})
        if unknown:
            raise RuntimeError(f"Unknown sample IDs: {', '.join(unknown)}")
    candidates = [sample for sample in available if not task_type or sample.task_type == task_type]
    selected = filter_samples(candidates, sample_ids, max_samples)
    if not selected:
        raise RuntimeError("No samples match the requested selection")
    return available, selected


def run_eval(
    model: str,
    data_dir: str,
    output_dir: str,
    checkpoint: str | None = None,
    split: str = "eval",
    task_type: str | None = None,
    sample_ids: list[str] | None = None,
    max_samples: int | None = None,
    force: bool = False,
    generated_fps: float | None = None,
) -> dict:
    """Track, score, and aggregate one model run."""
    _, samples = _select_samples(data_dir, split, task_type, sample_ids, max_samples)

    from .pipeline import EvalPipeline

    pipeline = EvalPipeline(model, data_dir, output_dir, checkpoint, split, task_type, generated_fps)
    is_subset = bool(sample_ids or max_samples)

    print(f"Loaded {len(samples)} samples" + (f" ({task_type})" if task_type else ""))
    pipeline_result = pipeline.run(samples=samples, force=force)
    pipeline_result.raise_for_failures()

    official = split == "eval" and task_type is None and not is_subset and len(samples) == EVALUATION_SIZE
    agg = aggregate_metrics(
        model,
        output_dir,
        checkpoint=checkpoint,
        expected_sample_ids=(sample.id for sample in samples),
        official=official,
    )
    print(_format_summary(agg, agg.get("n_samples", 0)))
    if not official:
        print("Subset result (not an official full-benchmark score).")
    return agg


def evaluate_prediction_dir(
    predictions: str | Path,
    *,
    data_dir: str | Path = "data",
    output_dir: str | Path = "runs",
    name: str | None = None,
    split: str = "eval",
    task_type: str | None = None,
    sample_ids: list[str] | None = None,
    max_samples: int | None = None,
    fps: float | None = None,
    force: bool = False,
) -> dict:
    """Validate, prepare, and evaluate a directory of sample-ID videos."""
    source = Path(predictions)
    run_name = name or source.name
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("The run name must be a single directory-safe name")

    root = find_data_root(Path(data_dir))
    if root is None:
        raise FileNotFoundError(f"No RigidBench evaluation set found under {data_dir}")
    available, selected = _select_samples(root, split, task_type, sample_ids, max_samples)

    from .prepare import prepare_videos

    prepared = prepare_videos(
        source,
        model=run_name,
        output_dir=output_dir,
        fps=fps,
        force=force,
        expected_sample_ids=[sample.id for sample in available],
        selected_sample_ids=[sample.id for sample in selected],
    )
    agg = run_eval(
        run_name,
        str(root),
        str(output_dir),
        split=split,
        task_type=task_type,
        sample_ids=sample_ids,
        max_samples=max_samples,
        force=force,
        generated_fps=prepared.fps,
    )
    _write_run_metadata(
        run_name,
        source,
        root,
        output_dir,
        [sample.id for sample in selected],
        bool(agg.get("official")),
    )
    return agg


def _write_run_metadata(
    name: str,
    predictions: Path,
    data_dir: Path,
    output_dir: str | Path,
    sample_ids: list[str],
    official: bool,
) -> None:
    paths = OutputPaths(model_output_dir(output_dir, name))
    payload = {
        "protocol": PROTOCOL,
        "evaluator_version": __version__,
        "name": name,
        "predictions": str(predictions.resolve()),
        "data": str(data_dir.resolve()),
        "sample_ids": sample_ids,
        "official": official,
    }
    write_json(paths.run_metadata, payload)


def main():
    p = argparse.ArgumentParser(description="Evaluate videos named <sample_id>.mp4 with RigidBench.")
    p.add_argument("predictions", nargs="?", type=Path, help="Directory of generated videos")
    p.add_argument("--name", help="Run name; defaults to the prediction directory name")
    p.add_argument(
        "--data",
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=Path,
        help="Benchmark data directory (default: data)",
    )
    p.add_argument(
        "--output",
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        help="Run directory (default: runs)",
    )
    p.add_argument("--task", help="Evaluate one task only")
    p.add_argument(
        "--sample-ids",
        "--sample_ids",
        help="Comma-separated sample IDs to evaluate",
    )
    p.add_argument(
        "--max-samples",
        "--max_samples",
        type=int,
        help="Evaluate only the first N samples",
    )
    p.add_argument("--fps", type=float, help="Override the detected video frame rate")
    p.add_argument("--force", action="store_true", help="Recompute existing artifacts")

    # Retain the submitted prepared-frame interface for paper reproduction.
    hidden = argparse.SUPPRESS
    p.add_argument("--model", help=hidden)
    p.add_argument("--checkpoint", help=hidden)
    p.add_argument("--split", default="eval", choices=["train", "eval"], help=hidden)
    args = p.parse_args()
    sample_ids = None
    if args.sample_ids:
        sample_ids = [sample_id.strip() for sample_id in args.sample_ids.split(",")]
        if any(not sample_id for sample_id in sample_ids):
            p.error("--sample-ids must be a comma-separated list without empty values")

    if args.predictions:
        if args.model or args.checkpoint:
            p.error("A prediction directory cannot be combined with registered-model options")
        try:
            evaluate_prediction_dir(
                args.predictions,
                data_dir=args.data_dir or "data",
                output_dir=args.output_dir or "runs",
                name=args.name,
                split=args.split,
                task_type=args.task,
                sample_ids=sample_ids,
                max_samples=args.max_samples,
                fps=args.fps,
                force=args.force,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        run_name = args.name or args.predictions.name
        print(f"Results: {(args.output_dir or Path('runs')) / run_name / 'results.json'}")
        return 0

    if not args.data_dir:
        p.error("Pass a prediction directory, or use --data-dir with the legacy prepared-output interface")

    data_root = find_data_root(args.data_dir) or args.data_dir
    output_dir = args.output_dir or Path("outputs")

    if args.model:
        run_eval(
            args.model,
            data_dir=str(data_root),
            output_dir=str(output_dir),
            checkpoint=args.checkpoint,
            split=args.split,
            task_type=args.task,
            sample_ids=sample_ids,
            max_samples=args.max_samples,
            force=args.force,
            generated_fps=args.fps,
        )
    else:
        p.error("Pass a prediction directory or specify --model")
    return 0


if __name__ == "__main__":
    main()
