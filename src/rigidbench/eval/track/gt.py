from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

from rigidbench.core.constants import ERODE_ITERATIONS, N_QUERY_POINTS


def compute_gt_trajectories(
    sample_dir: Path,
    actor_name: str,
    n_points: int | None = None,
    seed: int = 42,
    z_tol_rel: float = 0.01,
) -> dict:
    """Project an actor's first-frame mask points along its simulated rigid trajectory."""
    n_points = n_points or N_QUERY_POINTS
    sample_dir = Path(sample_dir)

    depth = np.load(sample_dir / "depth.npz")["depth"]
    masks_data = np.load(sample_dir / "masks.npz")
    trajectories = np.load(sample_dir / "trajectories.npz")
    with open(sample_dir / "metadata.json") as f:
        metadata = json.load(f)

    intr = metadata["camera"]["intrinsics"]
    extr = metadata["camera"]["extrinsics"]
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    cam_loc = np.array(extr["location"])
    cam_rot = np.array(extr["rotation"])

    positions = trajectories[f"{actor_name}_positions"]
    rotations = trajectories[f"{actor_name}_rotations"]
    T = len(positions)

    object_names = list(masks_data["object_names"])
    if actor_name not in object_names:
        raise ValueError(f"Actor '{actor_name}' not found. Available: {object_names}")
    actor_idx = object_names.index(actor_name)
    actor_masks = masks_data["masks"][:, actor_idx]

    query_points = sample_points_on_mask(actor_masks[0], n_points, seed=seed)
    N = len(query_points)
    H_img, W_img = actor_masks.shape[1], actor_masks.shape[2]

    R0 = _quat_wxyz_to_rotation(rotations[0])
    R_cam_inv = _quat_wxyz_to_rotation(cam_rot).inv()

    local_points = np.zeros((N, 3))
    for i, (u, v) in enumerate(query_points):
        u_int = int(np.clip(round(u), 0, W_img - 1))
        v_int = int(np.clip(round(v), 0, H_img - 1))
        Z = float(depth[0, v_int, u_int])
        world = _unproject_to_world((u, v), Z, cam_loc, cam_rot, fx, fy, cx, cy)
        local_points[i] = R0.inv().apply(world - positions[0])

    result_trajectories = np.zeros((N, T, 2))
    visibility = np.zeros((N, T), dtype=bool)
    for t in range(T):
        Rt = _quat_wxyz_to_rotation(rotations[t])
        world_t = Rt.apply(local_points) + positions[t]
        uv, in_front = _project_world_to_image(world_t, cam_loc, cam_rot, fx, fy, cx, cy)
        result_trajectories[:, t] = uv

        cam_local = R_cam_inv.apply(world_t - cam_loc)
        Z_cv = -cam_local[:, 2]
        for i in range(N):
            if not in_front[i]:
                continue
            u_int = int(round(uv[i, 0]))
            v_int = int(round(uv[i, 1]))
            if not (0 <= u_int < W_img and 0 <= v_int < H_img):
                continue
            d_pixel = depth[t, v_int, u_int]
            if not bool(actor_masks[t, v_int, u_int]):
                continue
            if Z_cv[i] > d_pixel * (1 + z_tol_rel):
                continue
            visibility[i, t] = True

    return {
        "query_points": query_points,
        "trajectories": result_trajectories,
        "visibility": visibility,
    }


def sample_points_on_mask(mask: np.ndarray, n_points: int, seed: int = 42) -> np.ndarray:
    """Pick `n_points` pixel coordinates uniformly inside an eroded copy of `mask`."""
    eroded = ndimage.binary_erosion(mask, iterations=ERODE_ITERATIONS) if ERODE_ITERATIONS > 0 else mask
    if eroded.sum() < n_points:
        eroded = mask
    ys, xs = np.where(eroded)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), min(n_points, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1)


def _quat_wxyz_to_rotation(quat_wxyz) -> Rotation:
    """Build a scipy Rotation from a Blender-order (w, x, y, z) quaternion."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    return Rotation.from_quat([x, y, z, w])


def _unproject_to_world(
    uv,
    depth_value: float,
    cam_loc,
    cam_rot_wxyz,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject one image-space pixel at depth Z into world coordinates."""
    u, v = float(uv[0]), float(uv[1])
    Z = float(depth_value)
    point_cv = np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])
    point_blender = point_cv * np.array([1.0, -1.0, -1.0])
    return _quat_wxyz_to_rotation(cam_rot_wxyz).apply(point_blender) + np.asarray(cam_loc)


def _project_world_to_image(
    points_world,
    cam_loc,
    cam_rot_wxyz,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points to image-space pixels, returning (uv, in_front_mask)."""
    R = _quat_wxyz_to_rotation(cam_rot_wxyz)
    cam_local = R.inv().apply(np.asarray(points_world) - np.asarray(cam_loc))
    X = cam_local[:, 0]
    Y = -cam_local[:, 1]
    Z = -cam_local[:, 2]
    in_front = Z > 0
    Z_safe = np.where(in_front, Z, 1.0)
    return np.stack([fx * X / Z_safe + cx, fy * Y / Z_safe + cy], axis=1), in_front
