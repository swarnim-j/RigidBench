from __future__ import annotations

import numpy as np
import torch
from pytorch_msssim import ssim as torch_ssim

from rigidbench.core.metric import Metric, MetricResult, register_metric


def ssim_per_frame(gt: np.ndarray, pred: np.ndarray, device: str = "cuda") -> np.ndarray:
    """gt, pred: (T, H, W, 3) uint8. Returns (T,) float."""
    T = min(len(gt), len(pred))
    g = torch.from_numpy(gt[:T]).permute(0, 3, 1, 2).float().to(device)
    p = torch.from_numpy(pred[:T]).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        pf = torch_ssim(g, p, data_range=255.0, size_average=False)
    return pf.cpu().numpy()


def lpips_per_frame(gt: np.ndarray, pred: np.ndarray, model, device: str = "cuda") -> np.ndarray:
    """gt, pred: (T, H, W, 3) uint8. Returns (T,) float."""
    T = min(len(gt), len(pred))
    g = torch.from_numpy(gt[:T]).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
    p = torch.from_numpy(pred[:T]).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
    with torch.no_grad():
        out = model(g, p).squeeze().cpu().numpy()
    return out


@register_metric
class SSIM(Metric):
    name = "ssim"
    requires = "frames"

    def compute(self, ctx) -> MetricResult:
        gt, pred = ctx.frames
        pf = ssim_per_frame(gt, pred, ctx.device)
        return MetricResult({"ssim": float(pf.mean())}, {"ssim": pf})


@register_metric
class LPIPS(Metric):
    name = "lpips"
    requires = "frames"

    def compute(self, ctx) -> MetricResult:
        gt, pred = ctx.frames
        pf = lpips_per_frame(gt, pred, ctx.lpips_model, ctx.device)
        return MetricResult({"lpips": float(pf.mean())}, {"lpips": pf})
