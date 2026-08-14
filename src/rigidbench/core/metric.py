from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetricResult:
    scalars: dict[str, float] = field(default_factory=dict)
    per_frame: dict[str, np.ndarray] = field(default_factory=dict)


class Metric(ABC):
    name: str
    requires: str  # "masks" | "tracks" | "depth" | "frames"

    @abstractmethod
    def compute(self, ctx) -> MetricResult: ...


METRICS: dict[str, Metric] = {}


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Add the metric class to the global METRICS registry under its `name`."""
    METRICS[cls.name] = cls()
    return cls
