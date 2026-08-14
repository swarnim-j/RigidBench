from __future__ import annotations

import cv2
import numpy as np
import torch

from rigidbench.core.metric import Metric, MetricResult, register_metric

PATCH_SIZE = 64
EMBED_BATCH_SIZE = 256


def extract_patch(frame: np.ndarray, center: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray | None:
    """Extract square patch centered at (x, y), resize to 224x224 for DINO."""
    H, W = frame.shape[:2]
    x, y = int(round(center[0])), int(round(center[1]))
    half = patch_size // 2

    x0, x1 = max(0, x - half), min(W, x + half)
    y0, y1 = max(0, y - half), min(H, y + half)

    if x1 - x0 < patch_size // 2 or y1 - y0 < patch_size // 2:
        return None

    patch = frame[y0:y1, x0:x1]
    return cv2.resize(patch, (224, 224))


def embed_patches(patches: list[np.ndarray], model, device: str) -> np.ndarray:
    """Embed patches with DINOv2 in chunks, return L2-normalized features (N, D)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    all_feats = []
    for i in range(0, len(patches), EMBED_BATCH_SIZE):
        chunk = np.stack(patches[i : i + EMBED_BATCH_SIZE])
        batch = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(device) / 255.0
        batch = ((batch - mean) / std).to(torch.bfloat16)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = model(batch)
        all_feats.append(feats.float().cpu().numpy())
    feats = np.concatenate(all_feats, axis=0)
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)


def compute_iddrift(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray,
    visibility: np.ndarray,
    actor_offsets: np.ndarray,
    dinov2_model,
    device: str,
) -> dict[str, float]:
    """Mean DINOv2 cosine similarity between GT and pred patches at tracked points, per actor."""
    T = min(len(gt_frames), len(pred_frames), gt_tracks.shape[1], pred_tracks.shape[1])

    gt_patches: list[np.ndarray] = []
    pred_patches: list[np.ndarray] = []
    actor_ids: list[int] = []
    for ai in range(len(actor_offsets) - 1):
        for pi in range(int(actor_offsets[ai]), int(actor_offsets[ai + 1])):
            for t in range(1, T):
                if not visibility[pi, t]:
                    continue
                gt_p = extract_patch(gt_frames[t], gt_tracks[pi, t])
                pred_p = extract_patch(pred_frames[t], pred_tracks[pi, t])
                if gt_p is not None and pred_p is not None:
                    gt_patches.append(gt_p)
                    pred_patches.append(pred_p)
                    actor_ids.append(ai)

    if not gt_patches:
        return {}

    gt_feats = embed_patches(gt_patches, dinov2_model, device)
    pred_feats = embed_patches(pred_patches, dinov2_model, device)
    sims = np.sum(gt_feats * pred_feats, axis=1)

    ids = np.asarray(actor_ids)
    per_actor = [float(sims[ids == ai].mean()) for ai in np.unique(ids)]
    similarity = float(np.mean(per_actor))
    return {"iddrift": 1.0 - similarity, "id_similarity": similarity}


@register_metric
class IdDrift(Metric):
    name = "iddrift"
    requires = "identity"

    def compute(self, ctx) -> MetricResult:
        gt_frames, pred_frames, gt_tracks, pred_tracks, vis, offsets = ctx.identity
        scalars = compute_iddrift(
            gt_frames,
            pred_frames,
            gt_tracks,
            pred_tracks,
            vis,
            offsets,
            ctx.dinov2_model,
            ctx.device,
        )
        return MetricResult(scalars)
