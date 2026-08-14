from __future__ import annotations

import cv2
import numpy as np

from rigidbench.core.metric import Metric, MetricResult, register_metric

MAX_DEPTH = 100.0


def affine_align_disparity(
    pred_disp: np.ndarray,
    gt_depth: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Align pred disparity to GT depth via per-video LSQ: gt_disp = s*pred_disp + c."""
    T = min(len(pred_disp), len(gt_depth))
    H, W = gt_depth.shape[-2:]
    if pred_disp.shape[-2:] != (H, W):
        pred_disp = np.stack(
            [cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for d in pred_disp[:T]]
        )
    else:
        pred_disp = pred_disp[:T].astype(np.float32)

    gt_disp = 1.0 / np.clip(gt_depth[:T], 1e-3, None)
    valid = np.isfinite(gt_depth[:T]) & np.isfinite(pred_disp) & (gt_depth[:T] > 0) & (gt_depth[:T] < MAX_DEPTH)
    if valid.sum() == 0:
        return np.full_like(pred_disp, np.nan), np.nan, np.nan

    A = np.column_stack([pred_disp[valid], np.ones(valid.sum())])
    result = np.linalg.lstsq(A, gt_disp[valid], rcond=None)
    if len(result[0]) < 2:
        return np.full_like(pred_disp, np.nan), np.nan, np.nan
    s, c = result[0]
    aligned_disp = s * pred_disp + c
    aligned_depth = np.where(aligned_disp > 0, 1.0 / aligned_disp, np.nan)
    return aligned_depth, float(s), float(c)


def scale_invariant_mse(pred_depth: np.ndarray, gt_depth: np.ndarray) -> float:
    """Variance of log(pred) - log(gt) over valid pixels. Scale-invariant by construction."""
    valid = (gt_depth > 0) & (gt_depth < MAX_DEPTH) & np.isfinite(pred_depth) & (pred_depth > 0)
    if not valid.any():
        return float("nan")
    d = np.log(pred_depth[valid]) - np.log(gt_depth[valid])
    return float(np.mean(d**2) - np.mean(d) ** 2)


def compute_si_mse(gt: np.ndarray, pred_disp: np.ndarray) -> float:
    """Per-frame SI-MSE after affine-aligning disparity to GT depth."""
    pred_depth, _, _ = affine_align_disparity(pred_disp, gt)
    T = min(len(gt), len(pred_depth))
    pf = np.array([scale_invariant_mse(pred_depth[t], gt[t]) for t in range(T)])
    return float(np.nanmean(pf)) if np.any(np.isfinite(pf)) else float("nan")


@register_metric
class SiMSE(Metric):
    name = "si_mse"
    requires = "depth"

    def compute(self, ctx) -> MetricResult:
        gt, pred_disp = ctx.depth
        return MetricResult({"si_mse": compute_si_mse(gt, pred_disp)})
