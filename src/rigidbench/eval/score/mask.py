from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from rigidbench.core.metric import Metric, MetricResult, register_metric


def iou_per_frame(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """gt, pred: (T, 1, H, W). Returns (T,)."""
    out = np.zeros(gt.shape[0])
    for t in range(gt.shape[0]):
        m1, m2 = gt[t, 0], pred[t, 0]
        inter = np.count_nonzero(np.logical_and(m1, m2))
        union = np.count_nonzero(m1) + np.count_nonzero(m2) - inter
        out[t] = 1.0 if union == 0 else inter / union
    return out


def l2_per_frame(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Centroid distance / H. NaN where either mask is empty."""
    H = gt.shape[2]
    out = np.full(gt.shape[0], np.nan)
    for t in range(gt.shape[0]):
        ys1, xs1 = np.where(gt[t, 0])
        ys2, xs2 = np.where(pred[t, 0])
        if len(xs1) == 0 or len(xs2) == 0:
            continue
        c1 = np.array([xs1.mean(), ys1.mean()])
        c2 = np.array([xs2.mean(), ys2.mean()])
        out[t] = np.linalg.norm(c1 - c2) / H
    return out


def chamfer_per_frame(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Bidirectional nearest-neighbor distance / H. NaN where either mask is empty."""
    H = gt.shape[2]
    out = np.full(gt.shape[0], np.nan)
    for t in range(gt.shape[0]):
        p1 = np.column_stack(np.where(gt[t, 0])).astype(np.float32)
        p2 = np.column_stack(np.where(pred[t, 0])).astype(np.float32)
        if len(p1) == 0 or len(p2) == 0:
            continue
        d1, _ = cKDTree(p1).query(p2)
        d2, _ = cKDTree(p2).query(p1)
        out[t] = (d1.mean() + d2.mean()) / H
    return out


def _per_actor_mean(fn, gt_masks, pred_masks, n):
    """Apply `fn` to each actor's (T, 1, H, W) masks and nan-mean the (T,) results across actors."""
    stacked = np.stack([fn(gt_masks[:, i : i + 1], pred_masks[:, i : i + 1]) for i in range(n)])
    return np.nanmean(stacked, axis=0)


@register_metric
class IoU(Metric):
    name = "iou"
    requires = "masks"

    def compute(self, ctx) -> MetricResult:
        gt, pred, n = ctx.masks
        pf = _per_actor_mean(iou_per_frame, gt, pred, n)
        return MetricResult({"iou": float(np.nanmean(pf))}, {"iou": pf})


@register_metric
class L2(Metric):
    name = "l2"
    requires = "masks"

    def compute(self, ctx) -> MetricResult:
        gt, pred, n = ctx.masks
        pf = _per_actor_mean(l2_per_frame, gt, pred, n)
        return MetricResult({"l2": float(np.nanmean(pf))}, {"l2": pf})


@register_metric
class Chamfer(Metric):
    name = "chamfer"
    requires = "masks"

    def compute(self, ctx) -> MetricResult:
        gt, pred, n = ctx.masks
        pf = _per_actor_mean(chamfer_per_frame, gt, pred, n)
        return MetricResult({"chamfer": float(np.nanmean(pf))}, {"chamfer": pf})
