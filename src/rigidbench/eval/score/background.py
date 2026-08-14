from __future__ import annotations

import cv2
import numpy as np
import torch

from rigidbench.core.metric import Metric, MetricResult, register_metric

CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 10
MAX_CORNERS = 200


def detect_bg_corners(
    first_frame: np.ndarray,
    fg_mask: np.ndarray,
    erode_px: int = 15,
) -> np.ndarray | None:
    """Detect Shi-Tomasi corners in eroded background region."""
    bg_mask = ~fg_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
    bg_eroded = cv2.erode(bg_mask.astype(np.uint8), kernel)

    gray = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE,
        mask=bg_eroded * 255,
    )
    if corners is None or len(corners) < 4:
        return None
    return corners.squeeze(1)


def track_points_cotracker3(
    frames: np.ndarray,
    query_points: np.ndarray,
    model,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Track query points through frames using CoTracker3. Returns (tracks, confidence)."""
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    N = len(query_points)
    queries = np.zeros((1, N, 3), dtype=np.float32)
    queries[0, :, 1:] = query_points
    queries_t = torch.from_numpy(queries).to(device)

    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=queries_t)

    return (
        pred_tracks[0].cpu().numpy().transpose(1, 0, 2),
        pred_visibility[0].cpu().numpy().T,
    )


def compute_bgdrift(tracks: np.ndarray, confidence: np.ndarray, height: int) -> float:
    """Compute background drift as confidence-weighted residual after similarity transform."""
    if tracks.size == 0:
        return float("nan")

    N, T, _ = tracks.shape
    residuals = []
    weights = []

    for t in range(1, T):
        conf_t = confidence[:, t] * confidence[:, 0]
        valid = conf_t > 0
        if valid.sum() < 4:
            continue

        src = tracks[valid, 0].astype(np.float32)
        dst = tracks[valid, t].astype(np.float32)
        conf_valid = conf_t[valid]

        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if M is None:
            disp = np.linalg.norm(dst - src, axis=1)
        else:
            transformed = (M[:, :2] @ src.T).T + M[:, 2]
            disp = np.linalg.norm(dst - transformed, axis=1)

        residuals.append(float(np.average(disp, weights=conf_valid)))
        weights.append(float(conf_valid.sum()))

    if not residuals:
        return float("nan")
    return float(np.average(residuals, weights=weights) / height)


@register_metric
class BGDrift(Metric):
    name = "bgdrift"
    requires = "bgdrift_input"

    def compute(self, ctx) -> MetricResult:
        frames, fg_mask, height, cotracker_model, device = ctx.bgdrift_input

        corners = detect_bg_corners(frames[0], fg_mask)
        if corners is None:
            return MetricResult({"bgdrift": float("nan")})

        tracks, confidence = track_points_cotracker3(frames, corners, cotracker_model, device)
        drift = compute_bgdrift(tracks, confidence, height)
        return MetricResult({"bgdrift": drift})
