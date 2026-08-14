from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import torch
import torch.distributed as dist
from tqdm import tqdm

T = TypeVar("T")


def setup(backend: str = "nccl") -> tuple[int, int, str]:
    """Init torchrun distributed group and return (rank, world_size, device)."""
    if "RANK" not in os.environ:
        return 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
    dist.init_process_group(backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"


def teardown() -> None:
    """Clean up torchrun distributed group if active."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def shard(items: Iterable[T], rank: int, world_size: int, desc: str = "rank") -> Iterator[T]:
    """Yield this rank's slice of `items`, wrapped in a rank-0 tqdm progress bar."""
    mine = list(items)[rank::world_size]
    yield from (tqdm(mine, desc=f"{desc}{rank}") if rank == 0 else mine)


def wandb_init(project: str | None, run_id: str | None, config: dict):
    """Init a wandb run if `project` is set, otherwise return None. Caller is responsible for being on rank 0."""
    if not project:
        return None
    import wandb

    return wandb.init(
        project=project,
        id=run_id,
        name=run_id,
        resume="allow" if run_id else None,
        config=config,
    )


def merge_json_shards(out_path: Path, world_size: int) -> int:
    """On rank 0, concatenate per-rank JSON shards into out_path and delete the shards. Return total rows."""
    merged: list = []
    for r in range(world_size):
        shard_path = out_path.with_suffix(f".rank{r}.json")
        merged.extend(json.loads(shard_path.read_text()))
        shard_path.unlink()
    out_path.write_text(json.dumps(merged, indent=2))
    return len(merged)
