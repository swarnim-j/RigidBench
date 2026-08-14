from __future__ import annotations

import json
from pathlib import Path

import torch

from .base import Target, register_target


@register_target
class ContactTarget(Target):
    """One per (actor, slot), set if any contact event involving the actor falls in the slot's pixel-frame range."""

    name = "contact"
    output_dim = 1
    loss_type = "bce"
    metric = "auc"

    def load(self, sample_dir: Path, actor_names: list[str], T_lat: int) -> torch.Tensor:
        contacts = json.loads((sample_dir / "contacts.json").read_text())
        name_to_idx = {name: i for i, name in enumerate(actor_names)}
        out = torch.zeros(len(actor_names), T_lat, dtype=torch.float32)
        for event in contacts:
            tau = _frame_to_slot(event["frame"], T_lat)
            for actor in (event["obj_a"], event["obj_b"]):
                if actor in name_to_idx:
                    out[name_to_idx[actor], tau] = 1.0
        return out


def _frame_to_slot(frame: int, T_lat: int) -> int:
    """Inverse of pixel_frames_for_slot. Returns the latent slot index that covers the given pixel frame."""
    if frame == 0:
        return 0
    return min((frame - 1) // 4 + 1, T_lat - 1)
