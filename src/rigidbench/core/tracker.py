from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .paths import OutputPaths
from .sample import RenderedSample


class Tracker(ABC):
    name: str

    @classmethod
    def can_track(cls, sample: RenderedSample, paths: OutputPaths) -> bool:
        """Whether this tracker's inputs are present for the sample."""
        return (paths.generated_dir(sample.id) / "00000.jpg").exists()

    @classmethod
    @abstractmethod
    def output_path(cls, paths: OutputPaths, sample_id: str) -> Path:
        """Where this tracker writes its output for a given sample."""

    @classmethod
    def is_done(cls, sample: RenderedSample, paths: OutputPaths) -> bool:
        """Whether this tracker has already produced its output for the sample."""
        return cls.output_path(paths, sample.id).exists()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    @abstractmethod
    def track(self, sample: RenderedSample, paths: OutputPaths) -> None:
        """Produce this tracker's output file(s) for the sample."""


TRACKERS: dict[str, type[Tracker]] = {}


def register_tracker(cls: type[Tracker]) -> type[Tracker]:
    """Add the tracker class to the global TRACKERS registry under its `name`."""
    TRACKERS[cls.name] = cls
    return cls
