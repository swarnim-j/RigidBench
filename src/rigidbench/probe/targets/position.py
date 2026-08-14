from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..tokens import pixel_frames_for_slot
from .base import Target, register_target


@register_target
class PositionTarget(Target):
    """Per-actor mean 3D world position over each latent slot's pixel-frame range."""

    name = "position"
    output_dim = 3
    loss_type = "mse"
    metric = "r2"

    def load(self, sample_dir: Path, actor_names: list[str], T_lat: int) -> torch.Tensor:
        traj = np.load(sample_dir / "trajectories.npz")
        out = torch.full((len(actor_names), T_lat, self.output_dim), float("nan"))
        for n, name in enumerate(actor_names):
            pos = traj[f"{name}_positions"]
            for tau in range(T_lat):
                out[n, tau] = torch.from_numpy(pos[pixel_frames_for_slot(tau)].mean(axis=0))
        return out
