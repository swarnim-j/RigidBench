from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .base import BackboneAdapter


class WanAdapter(BackboneAdapter):
    """BackboneAdapter for Wan 2.2 TI2V-5B via DiffSynth's WanVideoPipeline."""

    name = "wan"
    sample_artifacts = ["latent"]

    def __init__(self, model_id: str = "Wan-AI/Wan2.2-TI2V-5B", num_train_timesteps: int | None = None):
        self.model_id = model_id
        self.num_train_timesteps = num_train_timesteps
        self._pipe = None

    @classmethod
    def from_config(cls, cfg: dict) -> WanAdapter:
        return cls(
            model_id=cfg["model_id"],
            num_train_timesteps=cfg.get("num_train_timesteps"),
        )

    def load(self, device: str | torch.device, mode: str = "train", rank: int = 0, world_size: int = 1) -> None:
        from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

        dit = "diffusion_pytorch_model*.safetensors"
        t5 = "models_t5*.pth"
        vae = "*VAE*.pth"
        patterns_by_mode = {
            "train": [dit],
            "preprocess": [t5, vae],
            "probe": [dit, vae],
            "generate": [dit, t5, vae],
        }
        if mode not in patterns_by_mode:
            raise ValueError(f"Unknown load mode: {mode!r}")
        configs = [
            ModelConfig(model_id=self.model_id, origin_file_pattern=p, download_source="huggingface")
            for p in patterns_by_mode[mode]
        ]

        if rank == 0:
            for c in configs:
                c.download_if_necessary()
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        self._pipe = WanVideoPipeline.from_pretrained(torch.bfloat16, device, configs)

        if mode in ("train", "probe", "generate") and self.num_train_timesteps is not None:
            self._pipe.scheduler.set_timesteps(self.num_train_timesteps, training=True)

    def get_trainable_module(self) -> nn.Module:
        return self._pipe.dit

    def fsdp_wrap_classes(self) -> set[type[nn.Module]]:
        from diffsynth.models.wan_video_dit import DiTBlock

        return {DiTBlock}

    def timestep_by_fraction(self, frac: float) -> torch.Tensor:
        timesteps = self._pipe.scheduler.timesteps
        idx = round(frac * (len(timesteps) - 1))
        return timesteps[idx : idx + 1]

    def noise_latent(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(latents)
        noisy = self._pipe.scheduler.add_noise(latents, noise, timestep)
        noisy[:, :, 0:1] = latents[:, :, 0:1]
        return noisy, noise

    def dit_blocks(self) -> Iterable[nn.Module]:
        return self._pipe.dit.blocks

    def dit_forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        return self._pipe.model_fn(
            dit=self._pipe.dit,
            latents=noisy_latents,
            context=context,
            timestep=timestep,
            height=noisy_latents.shape[3] * 8,
            width=noisy_latents.shape[4] * 8,
            num_frames=noisy_latents.shape[2] * 4,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    def token_grid_shape(self, latent_shape: tuple[int, ...]) -> tuple[int, int, int]:
        return (latent_shape[2], latent_shape[3] // 2, latent_shape[4] // 2)

    def sigma_by_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        sched = self._pipe.scheduler
        t = timestep.detach().cpu()
        idx = torch.argmin((sched.timesteps - t).abs())
        return sched.sigmas[idx]

    def dit_head_apply(
        self,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

        dit = self._pipe.dit
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(hidden.dtype))
        v_tokens = dit.head(hidden, t)
        return dit.unpatchify(v_tokens, grid)

    def load_checkpoint(self, path: str | Path, device: str | torch.device) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = {
            k.removeprefix("dit."): v.to(device, torch.bfloat16)
            for k, v in ckpt["state_dict"].items()
            if k.startswith("dit.")
        }
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

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        use_gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latents = batch["latent"].to(dtype=torch.bfloat16, device=device)
        context = batch["text_emb"].to(dtype=torch.bfloat16, device=device)

        scheduler = self._pipe.scheduler
        timestep_id = torch.randint(0, len(scheduler.timesteps), (1,))
        timestep = scheduler.timesteps[timestep_id].to(dtype=torch.bfloat16, device=device)

        noisy_latents, noise = self.noise_latent(latents, timestep)
        target = scheduler.training_target(latents, noise, timestep)
        pred = self.dit_forward(noisy_latents, timestep, context, use_gradient_checkpointing)

        mse_loss = F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float())
        loss = mse_loss * scheduler.training_weight(timestep)

        return loss, {
            "mse_loss": mse_loss.detach(),
            "timestep": timestep.float().detach(),
        }

    def encode_prompts(self, prompts: list[str], device: str | torch.device) -> dict[str, torch.Tensor]:
        ids, mask = self._pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        embs = self._pipe.text_encoder(ids.to(device), mask.to(device))
        return {p: embs[j].cpu().to(torch.float16) for j, p in enumerate(prompts)}

    def preprocess_frames(self, frames: list) -> torch.Tensor:
        return self._pipe.preprocess_video(frames)

    def encode_artifacts(
        self,
        stacked: torch.Tensor,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        latents = self._pipe.vae.single_encode(stacked, device=device).cpu().to(torch.float16)
        return {"latent": latents}

    def is_artifact_valid(self, name: str, path: Path, source_num_frames: int) -> bool:
        if name != "latent" or not path.exists():
            return False
        try:
            t = torch.load(path, map_location="cpu", weights_only=True)
            expected_t = 1 + (source_num_frames - 1) // 4
            return t.dim() == 5 and t.shape[2] == expected_t
        except Exception:
            return False

    def default_lora_target_modules(self) -> list[str]:
        return ["q", "k", "v", "o", "ffn.0", "ffn.2"]

    def conditioning_param_prefixes(self) -> tuple[str, ...]:
        return ("text_embedding.", "time_embedding.", "time_projection.")
