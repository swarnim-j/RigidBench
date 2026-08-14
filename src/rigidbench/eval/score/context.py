from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from rigidbench.core.constants import GT_FPS
from rigidbench.core.paths import OutputPaths
from rigidbench.core.sample import RenderedSample
from rigidbench.eval.npz import load_depth, load_masks, load_tracks

from .depth import affine_align_disparity
from .temporal import (
    align_visibility,
    interpolate_depth,
    interpolate_frames,
    interpolate_masks,
    interpolate_tracks,
)
from .trajectory import quat_wxyz_to_rotmat, reconstruct_centroids


class ScoreContext:
    """Lazy loader of GT and predicted artifacts for one sample, shared across metrics."""

    def __init__(
        self,
        sample: RenderedSample,
        paths: OutputPaths,
        gen_fps: int | None,
        lpips_model: Any = None,
        dinov2_model: Any = None,
        cotracker_model: Any = None,
        device: str = "cuda",
    ):
        self.sample = sample
        self.paths = paths
        self.gen_fps = gen_fps
        self.lpips_model = lpips_model
        self.dinov2_model = dinov2_model
        self.cotracker_model = cotracker_model
        self.device = device

    @cached_property
    def masks(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        """GT and predicted active-actor masks aligned to a common T and N."""
        gt_data = np.load(self.sample.gt_mask)
        object_names = list(gt_data["object_names"])
        actors_dict = self.sample.metadata.get("actors", {})
        active = [i for i, name in enumerate(object_names) if actors_dict.get(name, {}).get("role") == "active"]
        gt_masks = gt_data["masks"][:, active] if active else gt_data["masks"]

        pred_masks = load_masks(self.paths.mask(self.sample.id))
        if self.gen_fps and self.gen_fps != GT_FPS:
            pred_masks = interpolate_masks(pred_masks, self.gen_fps, GT_FPS, len(gt_masks))

        T = min(len(gt_masks), len(pred_masks))
        gt_masks, pred_masks = gt_masks[:T], pred_masks[:T]
        n = min(gt_masks.shape[1], pred_masks.shape[1])
        return gt_masks, pred_masks, n

    @cached_property
    def tracks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
        """GT and predicted point tracks with AND-ed visibility, per-actor offsets, image height."""
        tracks_path = self.paths.tracks(self.sample.id)
        gt_tracks_path = self.paths.gt_tracks(self.sample.id)
        if not (tracks_path.exists() and gt_tracks_path.exists()):
            return None

        pred_data = load_tracks(tracks_path)
        gt_data = load_tracks(gt_tracks_path)
        pred_tr = pred_data["tracks"]
        gt_tr = gt_data["tracks"]
        pred_vis = pred_data["visibility"]
        gt_vis = gt_data["visibility"]
        actor_offsets = pred_data["actor_offsets"]

        if self.gen_fps and self.gen_fps != GT_FPS:
            T_gt = gt_tr.shape[1]
            pred_tr = interpolate_tracks(pred_tr, self.gen_fps, GT_FPS, T_gt)
            pred_vis = align_visibility(pred_vis, self.gen_fps, GT_FPS, T_gt)

        T = min(gt_tr.shape[1], pred_tr.shape[1])
        gt_tr, pred_tr = gt_tr[:, :T], pred_tr[:, :T]
        vis = pred_vis[:, :T] & gt_vis[:, :T]

        height = self.sample.metadata["camera"]["intrinsics"]["height"]
        return gt_tr, pred_tr, vis, actor_offsets, height

    @cached_property
    def depth(self) -> tuple[np.ndarray, np.ndarray] | None:
        """GT depth and predicted disparity aligned to a common T."""
        depth_path = self.paths.depth(self.sample.id)
        if not (depth_path.exists() and self.sample.gt_depth):
            return None
        gt_depth = load_depth(self.sample.gt_depth)
        pred_depth = load_depth(depth_path)
        if self.gen_fps and self.gen_fps != GT_FPS:
            pred_depth = interpolate_depth(pred_depth, self.gen_fps, GT_FPS, len(gt_depth))
        T = min(len(gt_depth), len(pred_depth))
        return gt_depth[:T], pred_depth[:T]

    @cached_property
    def trajectory(self) -> tuple[list[np.ndarray], dict, list[str]] | None:
        """Reconstructed 3D centroids, GT trajectories dict, and active actor names."""
        if self.tracks is None or self.depth is None:
            return None
        if self.sample.trajectories is None:
            return None

        gt_tr, pred_tr, vis, actor_offsets, _height = self.tracks
        gt_depth, pred_disp = self.depth

        actors_dict = self.sample.metadata.get("actors", {})
        actors = [a for a, info in actors_dict.items() if info.get("role") == "active"]
        if not actors:
            return None

        camera = self.sample.metadata.get("camera", {})
        K = camera.get("intrinsics", {})
        ext = camera.get("extrinsics", {})
        if not K or not ext:
            return None
        cam_loc = np.array(ext["location"], dtype=np.float64)
        R_c2w = quat_wxyz_to_rotmat(np.array(ext["rotation"], dtype=np.float64))

        pred_depth_aligned, _, _ = affine_align_disparity(pred_disp, gt_depth)
        pred_centroids = reconstruct_centroids(
            pred_tr,
            vis,
            pred_depth_aligned,
            K,
            cam_loc,
            R_c2w,
            actor_offsets,
        )

        gt_traj = dict(np.load(self.sample.trajectories))
        return pred_centroids, gt_traj, actors

    @cached_property
    def frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """GT and predicted RGB frames aligned to a common T and resolution."""
        gt_files = sorted(self.sample.first_frame.parent.glob("*.png"))
        pred_files = sorted(self.paths.generated_dir(self.sample.id).glob("*.jpg"))
        if not pred_files:
            return None
        if gt_files and self.sample.reference_video is None:
            gt_frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in gt_files])
        elif self.sample.reference_video is not None:
            gt_frames = _read_video(self.sample.reference_video)
        else:
            return None
        pred_frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in pred_files])
        if self.gen_fps and self.gen_fps != GT_FPS:
            pred_frames = interpolate_frames(pred_frames, self.gen_fps, GT_FPS, len(gt_frames))
        T = min(len(gt_frames), len(pred_frames))
        gt_frames, pred_frames = gt_frames[:T], pred_frames[:T]
        if pred_frames.shape[1:3] != gt_frames.shape[1:3]:
            h, w = gt_frames.shape[1:3]
            pred_frames = np.stack([cv2.resize(f, (w, h)) for f in pred_frames])
        return gt_frames, pred_frames

    @cached_property
    def identity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """GT and pred frames + tracks + visibility + actor offsets, for track-based IdDrift."""
        if self.frames is None or self.tracks is None:
            return None
        gt_frames, pred_frames = self.frames
        gt_tracks, pred_tracks, visibility, actor_offsets, _ = self.tracks
        return gt_frames, pred_frames, gt_tracks, pred_tracks, visibility, actor_offsets

    @cached_property
    def bgdrift_input(self) -> tuple[np.ndarray, np.ndarray, int, Any, str] | None:
        """Pred frames, foreground mask, height, cotracker model, device for BG-Drift."""
        if self.frames is None or self.masks is None or self.cotracker_model is None:
            return None
        _, pred_frames = self.frames
        _, pred_masks, _ = self.masks
        fg_mask = pred_masks[0].any(axis=0)
        height = self.sample.metadata["camera"]["intrinsics"]["height"]
        return pred_frames, fg_mask, height, self.cotracker_model, self.device


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode reference video {path}")
    return np.stack(frames)
