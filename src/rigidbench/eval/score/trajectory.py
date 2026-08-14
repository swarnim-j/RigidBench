from __future__ import annotations

import numpy as np

from rigidbench.core.metric import Metric, MetricResult, register_metric


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def bilinear_sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = img.shape
    u = np.clip(uv[..., 0], 0, W - 1.0001)
    v = np.clip(uv[..., 1], 0, H - 1.0001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = u - u0, v - v0
    return (
        img[v0, u0] * (1 - du) * (1 - dv)
        + img[v0, np.minimum(u0 + 1, W - 1)] * du * (1 - dv)
        + img[np.minimum(v0 + 1, H - 1), u0] * (1 - du) * dv
        + img[np.minimum(v0 + 1, H - 1), np.minimum(u0 + 1, W - 1)] * du * dv
    )


def unproject(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    K: dict,
    cam_loc: np.ndarray,
    R_c2w: np.ndarray,
) -> np.ndarray:
    X = (u - K["cx"]) * z / K["fx"]
    Y = (v - K["cy"]) * z / K["fy"]
    P_cam = np.stack([X, Y, z], axis=-1)
    return (P_cam * np.array([1, -1, -1])) @ R_c2w.T + cam_loc


def geometric_median(X: np.ndarray, eps: float = 1e-5, max_iter: int = 100) -> np.ndarray:
    """Weiszfeld's algorithm for geometric median."""
    if len(X) == 1:
        return X[0]
    y = X.mean(axis=0)
    for _ in range(max_iter):
        dists = np.linalg.norm(X - y, axis=1, keepdims=True)
        dists = np.maximum(dists, eps)
        weights = 1.0 / dists
        y_new = (X * weights).sum(axis=0) / weights.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


def reconstruct_centroids(
    tracks: np.ndarray,
    visibility: np.ndarray,
    pred_depth: np.ndarray,
    K: dict,
    cam_loc: np.ndarray,
    R_c2w: np.ndarray,
    actor_offsets: np.ndarray,
) -> list[np.ndarray]:
    """Reconstruct 3D centroid per actor per frame using geometric median."""
    n_actors = len(actor_offsets) - 1
    T = min(tracks.shape[1], pred_depth.shape[0])
    centroids = []

    for ai in range(n_actors):
        s, e = int(actor_offsets[ai]), int(actor_offsets[ai + 1])
        actor_tracks = tracks[s:e, :T]
        actor_vis = visibility[s:e, :T]
        recon = np.full((T, 3), np.nan)

        for t in range(T):
            v_t = actor_vis[:, t].astype(bool)
            if v_t.sum() < 1:
                continue
            uv = actor_tracks[v_t, t]
            z = bilinear_sample(pred_depth[t], uv)
            ok = np.isfinite(z) & (z > 0)
            if ok.sum() < 1:
                continue
            world = unproject(
                uv[ok, 0].astype(np.float64),
                uv[ok, 1].astype(np.float64),
                z[ok].astype(np.float64),
                K,
                cam_loc,
                R_c2w,
            )
            recon[t] = geometric_median(world)

        centroids.append(recon)
    return centroids


def compute_ate3d(
    pred_centroids: list[np.ndarray],
    gt_trajectories: dict[str, np.ndarray],
    actors: list[str],
) -> dict[str, float]:
    """Compute RMSE of 3D trajectory error, normalized by mean object displacement."""
    errors = []
    displacements = []
    for ai, actor in enumerate(actors):
        key = f"{actor}_positions"
        if key not in gt_trajectories:
            continue
        gt = gt_trajectories[key]
        pred = pred_centroids[ai]
        T = min(len(gt), len(pred))

        gt_disp = np.linalg.norm(gt[T - 1] - gt[0]) if T > 1 else 0.0
        if gt_disp > 0.01:
            displacements.append(gt_disp)

        for t in range(T):
            if np.isfinite(pred[t]).all():
                errors.append(np.linalg.norm(pred[t] - gt[t]))

    if not errors:
        return {"ate3d": float("nan"), "ate3d_n": 0}
    arr = np.array(errors)
    rmse = float(np.sqrt(np.mean(arr**2)))
    scale = float(np.mean(displacements)) if displacements else 1.0
    return {
        "ate3d": rmse / scale,
        "ate3d_rmse_raw": rmse,
        "ate3d_scale": scale,
        "ate3d_n": int(len(arr)),
    }


@register_metric
class ATE3D(Metric):
    name = "ate3d"
    requires = "trajectory"

    def compute(self, ctx) -> MetricResult:
        pred_centroids, gt_traj, actors = ctx.trajectory
        scalars = compute_ate3d(pred_centroids, gt_traj, actors)
        return MetricResult(scalars)
