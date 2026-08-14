from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import lpips
import numpy as np
import torch
from tqdm import tqdm

from rigidbench.core.io import save_npz, write_json
from rigidbench.core.metric import METRICS
from rigidbench.core.paths import OutputPaths, model_output_dir
from rigidbench.core.result import Result
from rigidbench.core.sample import RenderedSample
from rigidbench.core.tracker import TRACKERS, Tracker
from rigidbench.eval.score import background, depth, frame, identity, mask, track, trajectory  # noqa: F401
from rigidbench.eval.score.context import ScoreContext
from rigidbench.eval.track import cotracker3, sam2, vda  # noqa: F401

from .generate.models import ALL_MODELS
from .samples import load_samples


@dataclass
class StageResult:
    completed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass
class PipelineResult:
    tracking: dict[str, StageResult] = field(default_factory=dict)
    evaluation: StageResult = field(default_factory=StageResult)

    @property
    def ok(self) -> bool:
        return self.evaluation.ok and all(stage.ok for stage in self.tracking.values())

    @property
    def failures(self) -> dict[str, list[str]]:
        failures = {f"track:{name}": stage.failed for name, stage in self.tracking.items() if stage.failed}
        if self.evaluation.failed:
            failures["score"] = self.evaluation.failed
        return failures

    def raise_for_failures(self) -> None:
        """Raise after a run in which any sample failed tracking or scoring."""
        if self.ok:
            return
        details = "; ".join(
            f"{stage}: {', '.join(sample_ids[:10])}" + (" ..." if len(sample_ids) > 10 else "")
            for stage, sample_ids in self.failures.items()
        )
        raise RuntimeError(f"Evaluation pipeline failed ({details})")


class EvalPipeline:
    """Run all registered trackers, then all registered metrics, over a set of samples."""

    def __init__(
        self,
        model: str,
        data_dir: str,
        output_dir: str,
        checkpoint: str | None = None,
        split: str = "eval",
        task_type: str | None = None,
        generated_fps: float | None = None,
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.split = split
        self.task_type = task_type
        self.generated_fps = generated_fps
        self.data_dir = data_dir
        self.paths = OutputPaths(model_output_dir(output_dir, model, checkpoint))
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

    def load_samples(self) -> list[RenderedSample]:
        return load_samples(self.data_dir, self.split, self.task_type)

    def run(
        self,
        samples: list[RenderedSample] | None = None,
        force: bool = False,
    ) -> PipelineResult:
        """Track every sample with every Tracker, then score with every Metric."""
        if samples is None:
            samples = self.load_samples()

        result = PipelineResult()
        for tracker_cls in TRACKERS.values():
            result.tracking[tracker_cls.name] = self._run_tracker(tracker_cls, samples, force)
        result.evaluation = self._run_evaluation(samples, force)
        return result

    def _run_tracker(
        self,
        tracker_cls: type[Tracker],
        samples: list[RenderedSample],
        force: bool,
    ) -> StageResult:
        """Run one tracker over every sample that isn't already done."""
        result = StageResult()
        pending = [
            s
            for s in samples
            if tracker_cls.can_track(s, self.paths) and (force or not tracker_cls.is_done(s, self.paths))
        ]
        if not pending:
            result.skipped = len(samples)
            print(f"[{tracker_cls.name}] skip (all {len(samples)} done)", flush=True)
            return result

        print(f"[{tracker_cls.name}] start: {len(pending)}/{len(samples)} pending", flush=True)
        t0 = time.time()
        with tracker_cls() as tracker:
            t_load = time.time() - t0
            print(f"[{tracker_cls.name}] model loaded in {t_load:.1f}s", flush=True)
            for s in tqdm(pending, desc=tracker_cls.name):
                try:
                    tracker.track(s, self.paths)
                    result.completed += 1
                except Exception as e:
                    print(f"[{tracker_cls.name}] {s.id} failed: {e}", flush=True)
                    result.failed.append(s.id)

        result.skipped = len(samples) - len(pending)
        print(
            f"[{tracker_cls.name}] done in {time.time() - t0:.1f}s "
            f"({result.completed} ok, {len(result.failed)} failed)",
            flush=True,
        )
        return result

    def _run_evaluation(self, samples: list[RenderedSample], force: bool) -> StageResult:
        """Score each sample with every applicable metric and write per-sample JSONs."""
        result = StageResult()
        self.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metrics_per_frame_dir.mkdir(parents=True, exist_ok=True)

        pending = [
            s
            for s in samples
            if (force or not self.paths.per_sample_metrics(s.id).exists()) and self.paths.mask(s.id).exists()
        ]
        if not pending:
            result.skipped = len(samples)
            print(f"[score] skip (all {len(samples)} done)", flush=True)
            return result

        print(f"[score] start: {len(pending)}/{len(samples)} pending", flush=True)
        t0 = time.time()
        gen_fps = self._generated_fps()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lpips_model = lpips.LPIPS(net="alex").to(device)
        dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()
        cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        print(f"[score] models loaded in {time.time() - t0:.1f}s", flush=True)

        t_loop = time.time()
        timings: dict[str, float] = {}
        for s in tqdm(pending, desc="score"):
            try:
                avg, per_frame, dt = _score_sample(
                    s,
                    self.paths,
                    gen_fps,
                    lpips_model,
                    dinov2_model,
                    cotracker_model,
                    device,
                )
                for k, v in dt.items():
                    timings[k] = timings.get(k, 0.0) + v
                r = Result(s.id, s.task_type, avg)
                save_npz(self.paths.per_sample_per_frame(s.id), **per_frame)
                write_json(self.paths.per_sample_metrics(s.id), r.to_json())
                result.completed += 1
            except Exception as e:
                print(f"[score] {s.id} failed: {e}", flush=True)
                result.failed.append(s.id)
        result.skipped = len(samples) - len(pending)
        total = time.time() - t_loop
        summary = " ".join(f"{k}={v:.1f}s" for k, v in sorted(timings.items(), key=lambda kv: -kv[1]))
        print(f"[score] done in {total:.1f}s ({result.completed} ok, {len(result.failed)} failed)", flush=True)
        print(f"[score] per-metric totals: {summary}", flush=True)
        return result

    def _generated_fps(self) -> float | None:
        if self.generated_fps is not None:
            return self.generated_fps
        if self.paths.generation_metadata.exists():
            metadata = json.loads(self.paths.generation_metadata.read_text())
            if metadata.get("fps") is not None:
                return float(metadata["fps"])
        registered = ALL_MODELS.get(self.model, {}).get("fps")
        if registered is not None:
            return float(registered)
        return None


def _score_sample(
    sample,
    paths,
    gen_fps,
    lpips_model,
    dinov2_model,
    cotracker_model,
    device,
) -> tuple[dict, dict, dict]:
    """Run every metric whose inputs are present for the sample, returning (scalars, per_frame, per_metric_seconds)."""
    ctx = ScoreContext(sample, paths, gen_fps, lpips_model, dinov2_model, cotracker_model, device)
    avg: dict[str, float] = {}
    per_frame: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for metric in METRICS.values():
        try:
            bundle = getattr(ctx, metric.requires)
        except (FileNotFoundError, KeyError):
            continue
        if bundle is None:
            continue
        t0 = time.time()
        result = metric.compute(ctx)
        timings[metric.name] = time.time() - t0
        avg.update(result.scalars)
        per_frame.update(result.per_frame)
    return avg, per_frame, timings
