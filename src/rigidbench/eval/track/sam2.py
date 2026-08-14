from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from rigidbench.core.constants import SEGMENTATION_MODEL
from rigidbench.core.io import save_npz
from rigidbench.core.paths import OutputPaths
from rigidbench.core.sample import RenderedSample
from rigidbench.core.tracker import Tracker, register_tracker


@register_tracker
class SAM2Tracker(Tracker):
    """Propagate first-frame GT masks across the generated video with SAM2."""

    name = "mask"

    @classmethod
    def output_path(cls, paths: OutputPaths, sample_id: str) -> Path:
        return paths.mask(sample_id)

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._predictor = None

    def __enter__(self):
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self._predictor = SAM2VideoPredictor.from_pretrained(SEGMENTATION_MODEL)
        self._predictor.to(self.device)
        return self

    def __exit__(self, *_):
        del self._predictor
        self._predictor = None
        gc.collect()
        torch.cuda.empty_cache()

    def track(self, sample: RenderedSample, paths: OutputPaths) -> None:
        """Seed SAM2 with first-frame GT masks and write the propagated masks to disk."""
        gt_data = np.load(sample.gt_mask)
        object_names = list(gt_data["object_names"])
        actors_dict = sample.metadata.get("actors", {})
        active = [i for i, name in enumerate(object_names) if actors_dict.get(name, {}).get("role") == "active"]
        if not active:
            raise ValueError(f"{sample.id}: no active actors to track")

        first_frame_masks = gt_data["masks"][0, active]
        output_path = paths.mask(sample.id)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        masks = self._propagate(paths.generated_dir(sample.id), first_frame_masks)
        save_npz(output_path, masks=masks)

    def _propagate(self, video_dir: Path, first_frame_masks: np.ndarray) -> np.ndarray:
        """Run SAM2 propagation across video_dir and return a (T, N, H, W) bool mask volume."""
        state = self._predictor.init_state(video_path=str(video_dir))
        self._predictor.reset_state(state)
        for obj_id in range(first_frame_masks.shape[0]):
            self._predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=obj_id,
                mask=first_frame_masks[obj_id],
            )

        frame_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, obj_ids, mask_logits in self._predictor.propagate_in_video(state):
            frame_masks[frame_idx] = {
                obj_id: (mask_logits[i].squeeze() > 0.0).cpu().numpy() for i, obj_id in enumerate(obj_ids)
            }

        num_objects = first_frame_masks.shape[0]
        h, w = first_frame_masks.shape[-2:]
        masks = np.zeros((len(frame_masks), num_objects, h, w), dtype=bool)
        for frame_idx, obj_masks in frame_masks.items():
            for obj_id, mask in obj_masks.items():
                masks[frame_idx, obj_id] = mask
        return masks
