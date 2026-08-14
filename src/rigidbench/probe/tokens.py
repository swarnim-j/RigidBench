from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn


def pixel_frames_for_slot(t_lat: int) -> slice:
    """Wan VAE's 4n+1 latent-to-pixel mapping: slot 0 covers frame 0 alone, then 4 frames per slot."""
    if t_lat == 0:
        return slice(0, 1)
    return slice((t_lat - 1) * 4 + 1, t_lat * 4 + 1)


def compute_actor_token_mask(masks: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    T_lat, H_lat, W_lat = grid
    N = masks.shape[1]
    out = torch.zeros(N, T_lat, H_lat, W_lat, dtype=torch.bool)
    for tau in range(T_lat):
        merged = masks[pixel_frames_for_slot(tau)].any(dim=0).float()
        pooled = F.adaptive_max_pool2d(merged.unsqueeze(0), (H_lat, W_lat))[0]
        out[:, tau] = pooled.bool()
    return out


def pool_by_actor(activation: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Slots with no actor coverage come back as NaN."""
    N, T_lat, H_lat, W_lat = token_mask.shape
    HW = H_lat * W_lat
    dim = activation.shape[-1]
    act = activation.view(T_lat, HW, dim).float()
    mask = token_mask.view(N, T_lat, HW).to(act.device, act.dtype)
    masked_sum = torch.einsum("nth,thd->ntd", mask, act)
    count = mask.sum(dim=-1, keepdim=True)
    return torch.where(count > 0, masked_sum / count, torch.full_like(masked_sum, float("nan")))


@contextmanager
def capture_blocks(blocks: Iterable[nn.Module], layers: list[int] | None = None):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i, block in enumerate(blocks):
        if layers is not None and i not in layers:
            continue
        handles.append(block.register_forward_hook(_make_capture_hook(i, captured)))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _make_capture_hook(layer_idx: int, store: dict[int, torch.Tensor]):
    def hook(_module, _inputs, output):
        store[layer_idx] = output[0] if isinstance(output, tuple) else output

    return hook


def make_subspace_hook(basis: torch.Tensor, alpha: float, capture: dict | None = None):
    """Subtracts alpha * (B B^T) x from the residual stream. Capture dict gets the post-modification tensor."""

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        orig_dtype = x.dtype
        B = basis.to(x.device).float()
        x_new = (x.float() - alpha * (x.float() @ B) @ B.T).to(orig_dtype)
        if capture is not None:
            capture["x"] = x_new
        return (x_new,) + output[1:] if is_tuple else x_new

    return hook


def make_capture_hook(store: dict):
    def hook(_module, _inputs, output):
        store["x"] = output[0] if isinstance(output, tuple) else output

    return hook
