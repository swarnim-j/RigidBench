from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from rigidbench.core.constants import DEPTH_INPUT_SIZE, DEPTH_KEY, DEPTH_MODEL
from rigidbench.core.io import save_npz
from rigidbench.core.paths import OutputPaths
from rigidbench.core.sample import RenderedSample
from rigidbench.core.tracker import Tracker, register_tracker


@register_tracker
class VDATracker(Tracker):
    """Run VideoDepthAnything to predict per-frame disparity for the generated video."""

    name = "depth"

    @classmethod
    def output_path(cls, paths: OutputPaths, sample_id: str) -> Path:
        return paths.depth(sample_id)

    @classmethod
    def can_track(cls, sample: RenderedSample, paths: OutputPaths) -> bool:
        return super().can_track(sample, paths) and sample.gt_depth is not None

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def __enter__(self):
        from video_depth_anything.video_depth import VideoDepthAnything

        ckpt = hf_hub_download(repo_id=DEPTH_MODEL, filename="video_depth_anything_vitl.pth")
        self._model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        self._model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        self._model = self._model.to(self.device).eval()
        return self

    def __exit__(self, *_):
        del self._model
        self._model = None
        gc.collect()
        torch.cuda.empty_cache()

    def track(self, sample: RenderedSample, paths: OutputPaths) -> None:
        """Run VDA on the generated frames and save the disparity volume."""
        output_path = paths.depth(sample.id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_files = sorted(paths.generated_dir(sample.id).glob("*.jpg"))
        frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files])
        with torch.no_grad():
            disparity, _ = self._model.infer_video_depth(
                frames,
                target_fps=24,
                input_size=DEPTH_INPUT_SIZE,
                device=self.device,
            )
        save_npz(output_path, **{DEPTH_KEY: disparity})
