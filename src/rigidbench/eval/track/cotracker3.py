from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rigidbench.core.paths import OutputPaths
from rigidbench.core.sample import RenderedSample
from rigidbench.core.tracker import Tracker, register_tracker
from rigidbench.eval.npz import save_tracks

from .gt import compute_gt_trajectories


@register_tracker
class CoTracker3Tracker(Tracker):
    """Run CoTracker3 over generated frames seeded with GT actor query points."""

    name = "tracks"

    @classmethod
    def output_path(cls, paths: OutputPaths, sample_id: str) -> Path:
        return paths.tracks(sample_id)

    @classmethod
    def is_done(cls, sample: RenderedSample, paths: OutputPaths) -> bool:
        return paths.tracks(sample.id).exists() and paths.gt_tracks(sample.id).exists()

    @classmethod
    def can_track(cls, sample: RenderedSample, paths: OutputPaths) -> bool:
        return super().can_track(sample, paths) and sample.trajectories is not None

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def __enter__(self):
        self._model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self._model = self._model.to(self.device)
        return self

    def __exit__(self, *_):
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def track(self, sample: RenderedSample, paths: OutputPaths) -> None:
        """Seed CoTracker3 with first-frame GT query points and save pred + GT tracks side by side."""
        actors = [k for k, v in sample.metadata.get("actors", {}).items() if v.get("role") == "active"]
        if not actors:
            raise ValueError(f"{sample.id}: no active actors to track")

        gts = [compute_gt_trajectories(sample.trajectories.parent, a) for a in actors]
        actor_offsets = np.cumsum([0] + [len(g["query_points"]) for g in gts]).tolist()
        query_points = np.concatenate([g["query_points"] for g in gts], axis=0)
        gt_tracks = np.concatenate([g["trajectories"] for g in gts], axis=0)
        gt_visibility = np.concatenate([g["visibility"] for g in gts], axis=0)

        pred_tracks, pred_visibility = self._track_frames(
            paths.generated_dir(sample.id),
            query_points,
        )

        save_tracks(paths.gt_tracks(sample.id), gt_tracks, actor_offsets, visibility=gt_visibility)
        save_tracks(paths.tracks(sample.id), pred_tracks, actor_offsets, visibility=pred_visibility)

    def _track_frames(
        self,
        frames_dir: Path,
        query_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Push frames + query points through CoTracker3, return (tracks, visibility) as numpy."""
        frame_files = sorted(Path(frames_dir).glob("*.jpg"))
        frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files])
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(self.device)

        N = len(query_points)
        queries = np.zeros((1, N, 3), dtype=np.float32)
        queries[0, :, 1:] = query_points
        queries_t = torch.from_numpy(queries).to(self.device)

        with torch.no_grad():
            pred_tracks, pred_visibility = self._model(video, queries=queries_t)

        return (
            pred_tracks[0].cpu().numpy().transpose(1, 0, 2),
            pred_visibility[0].cpu().numpy().astype(bool).T,
        )
