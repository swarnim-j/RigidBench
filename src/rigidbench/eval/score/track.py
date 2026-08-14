from __future__ import annotations

import numpy as np

from rigidbench.core.metric import Metric, MetricResult, register_metric


def ate_per_frame(
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray,
    height: int,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Mean point error / H per frame. NaN at frames with no visible points."""
    T = min(gt_tracks.shape[1], pred_tracks.shape[1])
    errors = np.linalg.norm(gt_tracks[:, :T] - pred_tracks[:, :T], axis=-1)
    if visibility is None:
        return errors.mean(axis=0) / height
    vis = visibility[:, :T]
    out = np.full(T, np.nan)
    for t in range(T):
        if vis[:, t].any():
            out[t] = errors[vis[:, t], t].mean() / height
    return out


def compute_ate_scalar(
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray,
    height: int,
    visibility: np.ndarray | None = None,
) -> dict[str, float]:
    """ATE mean and std over visible (k, t) pairs, normalized by image height."""
    T = min(gt_tracks.shape[1], pred_tracks.shape[1])
    errors = np.linalg.norm(gt_tracks[:, :T] - pred_tracks[:, :T], axis=-1)
    if visibility is not None:
        mask = visibility[:, :T]
        if not mask.any():
            return {"ate": float("nan"), "ate_std": float("nan")}
        errors = errors[mask]
    return {"ate": float(errors.mean() / height), "ate_std": float(errors.std() / height)}


@register_metric
class ATE(Metric):
    name = "ate"
    requires = "tracks"

    def compute(self, ctx) -> MetricResult:
        gt, pred, vis, _offsets, height = ctx.tracks
        scalars = compute_ate_scalar(gt, pred, height, visibility=vis)
        per_frame = {"ate": ate_per_frame(gt, pred, height, visibility=vis)}
        return MetricResult(scalars, per_frame)
