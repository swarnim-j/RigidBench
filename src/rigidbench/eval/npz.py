from __future__ import annotations

from pathlib import Path

import numpy as np

from rigidbench.core.constants import DEPTH_KEY, MASK_KEY, TRACKS_KEY
from rigidbench.core.io import save_npz


def load_masks(path: Path | str) -> np.ndarray:
    data = np.load(path)
    if MASK_KEY in data:
        return data[MASK_KEY]
    if "mask" in data:
        return data["mask"]
    raise KeyError(f"No '{MASK_KEY}' or 'mask' key in {path}")


def load_tracks(path: Path | str) -> dict:
    data = np.load(path)
    out = {TRACKS_KEY: data[TRACKS_KEY], "actor_offsets": data["actor_offsets"]}
    if "visibility" in data.files:
        out["visibility"] = data["visibility"]
    return out


def save_tracks(
    path: Path | str,
    tracks: np.ndarray,
    actor_offsets: np.ndarray,
    visibility: np.ndarray | None = None,
) -> None:
    """Write (N, T, 2) tracks, per-actor offsets, and optional visibility to a compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        TRACKS_KEY: tracks,
        "actor_offsets": np.asarray(actor_offsets, dtype=np.int64),
    }
    if visibility is not None:
        payload["visibility"] = visibility
    save_npz(path, **payload)


def load_depth(path: Path | str) -> np.ndarray:
    data = np.load(path)
    if DEPTH_KEY in data:
        return data[DEPTH_KEY]
    raise KeyError(f"No '{DEPTH_KEY}' key in {path}")
