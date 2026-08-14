from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rigidbench.train.backbones import get_adapter

from . import runtime
from .tokens import capture_blocks, compute_actor_token_mask, pool_by_actor


class ActivationExtractor:
    def __init__(
        self,
        backbone: str = "wan",
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        num_train_timesteps: int = 1000,
        timestep_fracs: list[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        layers: list[int] | None = None,
        seed: int = 0,
        random_init: bool = False,
        checkpoint: str | None = None,
    ):
        self.backbone = backbone
        self.model_id = model_id
        self.num_train_timesteps = num_train_timesteps
        self.timestep_fracs = list(timestep_fracs)
        self.layers = layers
        self.seed = seed
        self.random_init = random_init
        self.checkpoint = checkpoint
        self.adapter = None

    def load(self, device: str, rank: int = 0, world_size: int = 1) -> None:
        self.adapter = get_adapter(
            self.backbone,
            {"model_id": self.model_id, "num_train_timesteps": self.num_train_timesteps},
        )
        self.adapter.load(device=device, mode="probe", rank=rank, world_size=world_size)
        if self.random_init:
            self.adapter.reset_dit_parameters(seed=self.seed)
            if rank == 0:
                print(f"DiT reinitialized with seed={self.seed}")
        if self.checkpoint:
            self.adapter.load_checkpoint(self.checkpoint, device)
            if rank == 0:
                print(f"checkpoint loaded: {self.checkpoint}")

    @torch.no_grad()
    def extract(self, sid: str, embeddings_dir: Path, data_dir: Path, device: str) -> dict:
        latent = torch.load(embeddings_dir / f"{sid}_latent.pt", weights_only=True).to(device, torch.bfloat16)
        context = torch.load(embeddings_dir / f"{sid}_text.pt", weights_only=True).to(device, torch.bfloat16)
        if context.dim() == 2:
            context = context.unsqueeze(0)
        npz = np.load(data_dir / sid / "masks.npz")
        masks = torch.from_numpy(npz["masks"])
        actor_names = [str(n) for n in npz["object_names"]]
        metadata = json.loads((data_dir / sid / "metadata.json").read_text())

        grid = self.adapter.token_grid_shape(tuple(latent.shape))
        token_mask = compute_actor_token_mask(masks, grid).to(device)

        activations: dict[int, dict[float, torch.Tensor]] = {}
        timesteps: dict[float, torch.Tensor] = {}
        for frac in self.timestep_fracs:
            timestep = self.adapter.timestep_by_fraction(frac).to(device, torch.bfloat16)
            timesteps[frac] = timestep.float().cpu()
            torch.manual_seed(self.seed)
            noisy, _ = self.adapter.noise_latent(latent, timestep)
            with capture_blocks(self.adapter.dit_blocks(), self.layers) as hidden:
                self.adapter.dit_forward(noisy, timestep, context)
            for layer_idx, h in hidden.items():
                activations.setdefault(layer_idx, {})[frac] = pool_by_actor(h, token_mask).to(torch.float16).cpu()
        return {
            "sample_id": sid,
            "task_type": metadata["task_type"],
            "token_grid": grid,
            "actor_names": actor_names,
            "timesteps": timesteps,
            "activations": activations,
        }

    def run(
        self,
        sample_ids: list[str],
        embeddings_dir: Path,
        data_dir: Path,
        output_dir: Path,
        rank: int,
        world_size: int,
        device: str,
        wandb_run=None,
    ) -> tuple[int, int, list]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            print(f"{len(sample_ids)} samples total")
        completed, skipped, failed = 0, 0, []
        for sid in runtime.shard(sample_ids, rank, world_size):
            target = output_dir / f"{sid}.pt"
            if target.exists():
                skipped += 1
                continue
            t0 = time.perf_counter()
            try:
                result = self.extract(sid, embeddings_dir, data_dir, device)
                torch.save(result, target)
                _log_sample(wandb_run, sid, result, self.timestep_fracs, time.perf_counter() - t0, rank)
                completed += 1
            except Exception as e:
                failed.append((sid, str(e)))
                if rank == 0:
                    tqdm.write(f"FAIL {sid}: {e}")
        return completed, skipped, failed


def load_manifest(path: Path) -> list[str]:
    return [json.loads(line)["latent"].removesuffix("_latent.pt") for line in path.read_text().splitlines()]


def _log_sample(run, sid: str, result: dict, fracs: list[float], wall_s: float, rank: int) -> None:
    if run is None:
        return
    finite_ratios = []
    for layer_acts in result["activations"].values():
        for frac in fracs:
            finite_ratios.append(torch.isfinite(layer_acts[frac]).all(dim=-1).float().mean().item())
    run.log(
        {
            "sample_id": sid,
            "task_type": result["task_type"],
            "num_actors": len(result["actor_names"]),
            "finite_ratio": sum(finite_ratios) / len(finite_ratios),
            "wall_s": wall_s,
            "rank": rank,
        }
    )
