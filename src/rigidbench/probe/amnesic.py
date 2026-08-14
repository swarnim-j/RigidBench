from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rigidbench.train.backbones import get_adapter

from . import runtime
from .tokens import make_capture_hook, make_subspace_hook


def _flow_match_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    """First temporal slot is excluded to match Wan's I2V first-frame pinning."""
    return F.mse_loss(pred[:, :, 1:].float(), target[:, :, 1:].float()).item()


def _forward_with_intervention(
    adapter,
    blocks,
    layer,
    hook_fn,
    target,
    timestep,
    noisy,
    context,
    grid,
):
    """Returns (full-model loss, lens-local loss), the latter applying Wan's head locally to the ablated layer."""
    store: dict = {}
    h_abl = blocks[layer].register_forward_hook(hook_fn)
    h_cap = blocks[layer].register_forward_hook(make_capture_hook(store))
    try:
        loss_full = _flow_match_loss(adapter.dit_forward(noisy, timestep, context), target)
        loss_local = _flow_match_loss(adapter.dit_head_apply(store["x"], timestep, grid), target)
    finally:
        h_abl.remove()
        h_cap.remove()
    return loss_full, loss_local


@torch.no_grad()
def run_one_sample(
    adapter,
    sid: str,
    emb_dir: Path,
    data_dir: Path,
    inlp_results: dict,
    layers: list[int],
    timestep_fracs: list[float],
    device: str,
    seed: int,
) -> list[dict]:
    latent = torch.load(emb_dir / f"{sid}_latent.pt", weights_only=True).to(device, torch.bfloat16)
    context = torch.load(emb_dir / f"{sid}_text.pt", weights_only=True).to(device, torch.bfloat16)
    if context.dim() == 2:
        context = context.unsqueeze(0)
    metadata = json.loads((data_dir / sid / "metadata.json").read_text())
    blocks = list(adapter.dit_blocks())
    grid = adapter.token_grid_shape(tuple(latent.shape))

    rows: list[dict] = []
    for frac in timestep_fracs:
        timestep = adapter.timestep_by_fraction(frac).to(device, torch.bfloat16)
        torch.manual_seed(seed)
        noisy, noise = adapter.noise_latent(latent, timestep)
        target = adapter._pipe.scheduler.training_target(latent, noise, timestep)

        baseline_caps = {L: {} for L in layers}
        baseline_handles = [blocks[L].register_forward_hook(make_capture_hook(baseline_caps[L])) for L in layers]
        try:
            loss_base = _flow_match_loss(adapter.dit_forward(noisy, timestep, context), target)
            local_baselines = {
                L: _flow_match_loss(adapter.dit_head_apply(baseline_caps[L]["x"], timestep, grid), target)
                for L in layers
            }
        finally:
            for h in baseline_handles:
                h.remove()

        for L in layers:
            inlp = inlp_results.get(f"L{L:02d}_t{frac}")
            if inlp is None:
                continue
            for variant_name, ctrl in inlp["controls"].items():
                if ctrl["alpha"] <= 0:
                    continue
                hook_fn = make_subspace_hook(ctrl["basis"], ctrl["alpha"])
                loss_full, loss_local = _forward_with_intervention(
                    adapter,
                    blocks,
                    L,
                    hook_fn,
                    target,
                    timestep,
                    noisy,
                    context,
                    grid,
                )
                rows.append(
                    {
                        "sample_id": sid,
                        "task_type": metadata["task_type"],
                        "layer": L,
                        "timestep_frac": float(frac),
                        "variant": variant_name,
                        "removed_rank": int(inlp["removed_rank"]),
                        "loss_base": loss_base,
                        "loss_ablated": loss_full,
                        "delta_loss": loss_full - loss_base,
                        "loss_local_base": local_baselines[L],
                        "loss_local_ablated": loss_local,
                        "delta_loss_local": loss_local - local_baselines[L],
                    }
                )
    return rows


def sweep(
    inlp_results_path: Path,
    embeddings_dir: Path,
    data_dir: Path,
    output_path: Path,
    layers: list[int],
    timestep_fracs: list[float],
    n_samples: int,
    seed: int,
    rank: int,
    world_size: int,
    device: str,
    wandb_run=None,
) -> None:
    adapter = get_adapter("wan", {"model_id": "Wan-AI/Wan2.2-TI2V-5B", "num_train_timesteps": 1000})
    adapter.load(device=device, mode="probe", rank=rank, world_size=world_size)
    inlp = torch.load(inlp_results_path, weights_only=False, map_location="cpu")
    sample_ids = sorted(f.stem.removesuffix("_latent") for f in embeddings_dir.glob("*_latent.pt"))[:n_samples]

    all_rows: list[dict] = []
    for sid in runtime.shard(sample_ids, rank, world_size):
        t0 = time.perf_counter()
        try:
            rows = run_one_sample(
                adapter,
                sid,
                embeddings_dir,
                data_dir,
                inlp,
                layers,
                timestep_fracs,
                device,
                seed,
            )
            all_rows.extend(rows)
            if wandb_run:
                wandb_run.log({"sample_id": sid, "n_rows": len(rows), "wall_s": time.perf_counter() - t0, "rank": rank})
        except Exception as e:
            print(f"[rank{rank}] FAIL {sid}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(f".rank{rank}.json").write_text(json.dumps(all_rows))

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        n = runtime.merge_json_shards(output_path, world_size)
        print(f"wrote {n} rows -> {output_path}")
