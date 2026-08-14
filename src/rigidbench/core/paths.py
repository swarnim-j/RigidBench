from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SamplePaths:
    sample_dir: Path

    @property
    def frames_dir(self) -> Path:
        return self.sample_dir / "frames"

    @property
    def first_frame(self) -> Path:
        return self.frames_dir / "00000.png"

    @property
    def video(self) -> Path:
        return self.sample_dir / "video.mp4"

    @property
    def prompt(self) -> Path:
        return self.sample_dir / "prompt.txt"

    @property
    def masks(self) -> Path:
        return self.sample_dir / "masks.npz"

    @property
    def depth(self) -> Path:
        return self.sample_dir / "depth.npz"

    @property
    def trajectories(self) -> Path:
        return self.sample_dir / "trajectories.npz"

    @property
    def metadata(self) -> Path:
        return self.sample_dir / "metadata.json"


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path

    def generated_dir(self, sid: str) -> Path:
        return self.output_dir / "generated" / sid

    def mask(self, sid: str) -> Path:
        return self.output_dir / "masks" / sid / "mask.npz"

    def tracks(self, sid: str) -> Path:
        return self.output_dir / "tracks" / sid / "tracks.npz"

    def gt_tracks(self, sid: str) -> Path:
        return self.output_dir / "tracks" / sid / "gt_tracks.npz"

    def depth(self, sid: str) -> Path:
        return self.output_dir / "depth" / sid / "depth.npz"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def metrics_per_frame_dir(self) -> Path:
        return self.output_dir / "metrics_per_frame"

    def per_sample_metrics(self, sid: str) -> Path:
        return self.metrics_dir / f"{sid}.json"

    def per_sample_per_frame(self, sid: str) -> Path:
        return self.metrics_per_frame_dir / f"{sid}.npz"

    @property
    def results(self) -> Path:
        return self.output_dir / "results.json"

    @property
    def generation_metadata(self) -> Path:
        return self.output_dir / "generation.json"

    @property
    def run_metadata(self) -> Path:
        return self.output_dir / "run.json"


def model_output_dir(output_root: Path | str, model: str, checkpoint: str | Path | None = None) -> Path:
    stem = Path(checkpoint).stem if checkpoint else ""
    return Path(output_root) / model / stem


@dataclass(frozen=True)
class ProbePaths:
    variant_dir: Path

    @property
    def activations_dir(self) -> Path:
        return self.variant_dir / "activations"

    def activation(self, sid: str) -> Path:
        return self.activations_dir / f"{sid}.pt"

    @property
    def cells_dir(self) -> Path:
        return self.variant_dir / "cells"

    @property
    def probe_results(self) -> Path:
        return self.variant_dir / "probes.json"

    def inlp(self, target: str) -> Path:
        return self.variant_dir / "inlp" / f"{target}.pt"

    def amnesic(self, target: str) -> Path:
        return self.variant_dir / "amnesic" / f"{target}.json"


def probe_variant_dir(output_root: Path | str, variant: str) -> Path:
    """e.g. .../probe_outputs/base, .../probe_outputs/random_init, .../probe_outputs/ft2k."""
    return Path(output_root) / variant
