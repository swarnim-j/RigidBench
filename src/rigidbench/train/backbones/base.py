from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn


class BackboneAdapter(ABC):
    """Backbone-specific operations for I2V latent-diffusion fine-tuning."""

    name: str
    sample_artifacts: list[str]

    @classmethod
    def from_config(cls, cfg: dict) -> BackboneAdapter:
        """Construct from a config dict. Default: model_id only. Override for extras."""
        return cls(model_id=cfg["model_id"])

    @abstractmethod
    def load(self, device: str | torch.device, mode: str = "train", rank: int = 0, world_size: int = 1) -> None:
        """Load components for `mode` (train or preprocess), downloading on rank 0."""

    @abstractmethod
    def get_trainable_module(self) -> nn.Module:
        """The module whose parameters are optimized."""

    @abstractmethod
    def fsdp_wrap_classes(self) -> set[type[nn.Module]]:
        """Transformer block classes for FSDP auto-wrap."""

    @abstractmethod
    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        use_gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One training step from a batch dict, returning (loss, log_dict)."""

    @abstractmethod
    def encode_prompts(self, prompts: list[str], device: str | torch.device) -> dict[str, torch.Tensor]:
        """Encode prompts, returning a {prompt: embedding} dict."""

    @abstractmethod
    def preprocess_frames(self, frames: list) -> torch.Tensor:
        """Per-video preprocessing (PIL frames to tensor) before VAE encoding."""

    @abstractmethod
    def encode_artifacts(
        self,
        stacked: torch.Tensor,
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """Encode all per-sample artifacts from a batch. Keys must match self.sample_artifacts."""

    @abstractmethod
    def is_artifact_valid(self, name: str, path: Path, source_num_frames: int) -> bool:
        """Whether a cached artifact file is valid for the given source length."""

    @abstractmethod
    def timestep_by_fraction(self, frac: float) -> torch.Tensor:
        """Scheduler timestep at the given fraction of the schedule (0 = first, 1 = last)."""

    @abstractmethod
    def noise_latent(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise the clean latent with the first-frame slot pinned, returning (noisy, noise)."""

    @abstractmethod
    def dit_blocks(self) -> Iterable[nn.Module]:
        """Transformer blocks for forward-hook attachment."""

    @abstractmethod
    def dit_forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        use_gradient_checkpointing: bool = False,
    ) -> torch.Tensor:
        """Single DiT forward pass."""

    @abstractmethod
    def token_grid_shape(self, latent_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """(T_lat, H_lat, W_lat) of the patchified token grid for a given clean-latent shape."""

    def sigma_by_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        """Scheduler sigma matching add_noise's lookup, so Tweedie inverts add_noise consistently."""
        raise NotImplementedError(f"{type(self).__name__} does not expose sigma_by_timestep")

    def dit_head_apply(
        self,
        hidden: torch.Tensor,
        timestep: torch.Tensor,
        grid: tuple[int, int, int],
    ) -> torch.Tensor:
        """Replay the final head and unpatchify on an intermediate hidden state, returning velocity in latent shape."""
        raise NotImplementedError(f"{type(self).__name__} does not expose dit_head_apply")

    def reset_dit_parameters(self, seed: int = 0) -> None:
        """Reinitialize every DiT submodule that exposes reset_parameters() (random-init baseline)."""
        torch.manual_seed(seed)
        for m in self.get_trainable_module().modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def load_checkpoint(self, path: str | Path, device: str | torch.device) -> None:
        """Load a fine-tuned checkpoint over the loaded base model. Backbone-specific format."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoint loading")

    @abstractmethod
    def default_lora_target_modules(self) -> list[str]:
        """Default LoRA target module patterns."""

    def conditioning_param_prefixes(self) -> tuple[str, ...]:
        """Param-name prefixes of the conditioning pathway (text/time embedders), frozen under `freeze_conditioning`."""
        return ()

    def filter_lora_state_dict(self, state_dict: dict) -> dict:
        """Strip non-LoRA tensors from a state dict. Default: peft `lora_` convention."""
        return {k: v for k, v in state_dict.items() if "lora_" in k}
