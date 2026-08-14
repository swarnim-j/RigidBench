from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def interpolate_frames(frames: np.ndarray, src_fps: int, dst_fps: int, dst_frames: int) -> np.ndarray:
    """Linear-interpolate (T, H, W, 3) frames from src_fps to dst_fps."""
    if src_fps == dst_fps:
        return frames[:dst_frames]
    T_src = len(frames)
    src_times = np.arange(T_src) / src_fps
    dst_times = np.clip(np.arange(dst_frames) / dst_fps, 0, (T_src - 1) / src_fps)
    f = interp1d(src_times, frames.astype(np.float32), axis=0, kind="linear")
    return np.clip(f(dst_times), 0, 255).astype(np.uint8)


def interpolate_tracks(tracks: np.ndarray, src_fps: int, dst_fps: int, dst_frames: int) -> np.ndarray:
    """Linear-interpolate (N, T, 2) tracks from src_fps to dst_fps."""
    if src_fps == dst_fps:
        return tracks[:, :dst_frames]
    N, T_src, _ = tracks.shape
    src_times = np.arange(T_src) / src_fps
    dst_times = np.clip(np.arange(dst_frames) / dst_fps, 0, (T_src - 1) / src_fps)
    out = np.zeros((N, dst_frames, 2), dtype=tracks.dtype)
    for n in range(N):
        for d in range(2):
            f = interp1d(src_times, tracks[n, :, d], kind="linear", fill_value="extrapolate")
            out[n, :, d] = f(dst_times)
    return out


def interpolate_masks(masks: np.ndarray, src_fps: int, dst_fps: int, dst_frames: int) -> np.ndarray:
    """Nearest-neighbor resample boolean masks from src_fps to dst_fps."""
    # Nearest-neighbor preserves the boolean values, linear interp would yield 0..1.
    if src_fps == dst_fps:
        return masks[:dst_frames]
    T_src = masks.shape[0]
    src_times = np.arange(T_src) / src_fps
    dst_times = np.arange(dst_frames) / dst_fps
    indices = np.array([np.argmin(np.abs(src_times - t)) for t in dst_times])
    return masks[np.clip(indices, 0, T_src - 1)]


def interpolate_depth(depth: np.ndarray, src_fps: int, dst_fps: int, dst_frames: int) -> np.ndarray:
    """Linear-interpolate (T, H, W) depth from src_fps to dst_fps."""
    if src_fps == dst_fps:
        return depth[:dst_frames]
    T_src = len(depth)
    src_times = np.arange(T_src) / src_fps
    dst_times = np.clip(np.arange(dst_frames) / dst_fps, 0, (T_src - 1) / src_fps)
    return interp1d(src_times, depth, axis=0, kind="linear")(dst_times)


def align_visibility(vis: np.ndarray, src_fps: int, dst_fps: int, dst_frames: int) -> np.ndarray:
    """Nearest-neighbor resample a boolean visibility array along axis 1 (T)."""
    if src_fps == dst_fps:
        return vis[:, :dst_frames]
    T_src = vis.shape[1]
    src_times = np.arange(T_src) / src_fps
    dst_times = np.clip(np.arange(dst_frames) / dst_fps, 0, src_times[-1])
    idx = np.array([np.argmin(np.abs(src_times - t)) for t in dst_times])
    return vis[:, np.clip(idx, 0, T_src - 1)]
