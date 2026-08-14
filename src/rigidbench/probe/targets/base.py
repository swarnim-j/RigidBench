from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import torch

LossType = Literal["mse", "bce"]
MetricType = Literal["r2", "auc"]


class Target(ABC):
    """A probe target. Maps a sample directory to a per-(actor, slot) tensor of ground-truth physical state."""

    name: str
    output_dim: int
    loss_type: LossType
    metric: MetricType

    @abstractmethod
    def load(self, sample_dir: Path, actor_names: list[str], T_lat: int) -> torch.Tensor:
        """Return (N_actors, T_lat, output_dim) or (N_actors, T_lat) for output_dim == 1."""


TARGETS: dict[str, Target] = {}


def register_target(cls: type[Target]) -> type[Target]:
    """Class decorator that instantiates the target and registers it under its `name`."""
    TARGETS[cls.name] = cls()
    return cls


def get_target(name: str) -> Target:
    return TARGETS[name]
