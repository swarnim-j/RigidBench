from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image

from rigidbench.core.constants import NEGATIVE_PROMPT, PROMPT_SUFFIX

from .base import BaseGenerator
from .models import LOCAL_MODELS


class LocalGenerator(BaseGenerator):
    def __init__(self, model: str, checkpoint: str | None = None):
        super().__init__(model, checkpoint)
        self._pipe = None

    def __enter__(self):
        cfg = LOCAL_MODELS[self.model]
        if cfg["type"] == "wan":
            self._pipe = _load_wan(cfg)
            if self.checkpoint:
                self._load_wan_checkpoint(self.checkpoint)
        elif cfg["type"] == "cosmos2.5":
            self._pipe = _load_cosmos(cfg)
        else:
            raise ValueError(f"Unsupported local type: {cfg['type']}")
        return self

    def __exit__(self, *_):
        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()

    def _load_wan_checkpoint(self, path: str) -> None:
        """Layer a finetuned LoRA (or full state dict) over the loaded Wan DiT."""
        from peft import LoraConfig, inject_adapter_in_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = {k.removeprefix("dit."): v for k, v in ckpt["state_dict"].items() if k.startswith("dit.")}
        state_dict = {k: v.to("cuda", torch.bfloat16) for k, v in state_dict.items()}
        lora_cfg = ckpt.get("hyper_parameters", {}).get("lora")
        if lora_cfg:
            inject_adapter_in_model(
                LoraConfig(
                    r=lora_cfg["rank"],
                    lora_alpha=lora_cfg.get("alpha", lora_cfg["rank"]),
                    target_modules=lora_cfg["target_modules"],
                ),
                self._pipe.dit,
            )
            self._pipe.dit.load_state_dict(state_dict, strict=False)
        else:
            self._pipe.dit.load_state_dict(state_dict)

    def generate(
        self,
        prompt: str,
        image: str | Path,
        output_dir: str | Path,
        force: bool = False,
    ) -> Path:
        """Generate one video from (prompt, image) and write its frames to output_dir."""
        output_dir = Path(output_dir)
        if not force and (output_dir / "00000.jpg").exists():
            return output_dir
        cfg = LOCAL_MODELS[self.model]
        img = Image.open(image).convert("RGB").resize((cfg["width"], cfg["height"]))
        full_prompt = prompt + PROMPT_SUFFIX
        with torch.no_grad():
            if cfg["type"] == "wan":
                frames = self._pipe(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    input_image=img,
                    num_inference_steps=50,
                    height=cfg["height"],
                    width=cfg["width"],
                    num_frames=cfg["num_frames"],
                    tiled=True,
                )
            else:
                frames = self._pipe(
                    prompt=full_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image=img,
                    num_frames=cfg["num_frames"],
                    num_inference_steps=50,
                ).frames[0]
        return _save_frames(frames, output_dir)


def _load_wan(cfg: dict):
    """Load the WanVideoPipeline (DiT + T5 + VAE) from HuggingFace at bfloat16."""
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    device = f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
    patterns = ["diffusion_pytorch_model*.safetensors", "models_t5*.pth", "*VAE*.pth"]
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=cfg["model_id"], origin_file_pattern=p, download_source="huggingface")
            for p in patterns
        ],
    )


class _DummySafetyChecker:
    def to(self, device):
        return self

    def check_text_safety(self, _):
        return True

    def check_video_safety(self, video):
        return video


def _load_cosmos(cfg: dict):
    """Load NVIDIA Cosmos 2.5 with the safety checker stubbed out."""
    import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as cosmos_module
    from diffusers import Cosmos2_5_PredictBasePipeline

    cosmos_module.CosmosSafetyChecker = _DummySafetyChecker
    return Cosmos2_5_PredictBasePipeline.from_pretrained(
        cfg["model_id"],
        revision=cfg["revision"],
        torch_dtype=torch.bfloat16,
    ).to("cuda")


def _save_frames(frames, output_dir: Path) -> Path:
    """Write each frame as a JPEG named 00000.jpg, 00001.jpg, ... in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(output_dir / f"{i:05d}.jpg", quality=95, subsampling=0)
    return output_dir
