from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from . import runtime
from .targets import Target


class CellBuilder:
    def __init__(self, targets: list[Target], layers: list[int], timestep_fracs: list[float]):
        self.targets = targets
        self.layers = layers
        self.timestep_fracs = timestep_fracs

    def run(
        self,
        sample_files: list[Path],
        data_dir: Path,
        output_dir: Path,
        rank: int,
        world_size: int,
    ) -> tuple[int, int]:
        if rank == 0:
            print(f"{len(sample_files)} samples, {len(self.layers)} layers, {len(self.timestep_fracs)} timesteps")
        cells: dict[tuple[int, float], list[torch.Tensor]] = {
            (L, t): [] for L in self.layers for t in self.timestep_fracs
        }
        target_buf: dict[str, list[torch.Tensor]] = {tgt.name: [] for tgt in self.targets}
        sample_ids: list[str] = []
        task_types: list[str] = []
        actor_indices: list[torch.Tensor] = []
        slot_indices: list[torch.Tensor] = []

        t0 = time.perf_counter()
        skipped = 0
        for f in runtime.shard(sample_files, rank, world_size):
            sid = f.stem
            try:
                sample = torch.load(f, weights_only=False)
            except Exception as e:
                skipped += 1
                if rank == 0:
                    tqdm.write(f"SKIP {sid}: {e}")
                continue
            N = len(sample["actor_names"])
            T_lat = sample["token_grid"][0]
            n_rows = N * T_lat
            for (L, t), bucket in cells.items():
                bucket.append(sample["activations"][L][t].reshape(n_rows, -1))
            for tgt in self.targets:
                y = tgt.load(data_dir / sid, sample["actor_names"], T_lat)
                target_buf[tgt.name].append(y.reshape(n_rows, -1) if y.dim() == 3 else y.reshape(n_rows))
            actor_indices.append(torch.arange(N).view(N, 1).expand(N, T_lat).reshape(-1))
            slot_indices.append(torch.arange(T_lat).view(1, T_lat).expand(N, T_lat).reshape(-1))
            sample_ids.extend([sid] * n_rows)
            task_types.extend([sample["task_type"]] * n_rows)

        output_dir.mkdir(parents=True, exist_ok=True)
        targets_out = {tgt.name: torch.cat(target_buf[tgt.name], dim=0) for tgt in self.targets}
        torch.save(
            {
                **targets_out,
                "sample_id": sample_ids,
                "task_type": task_types,
                "actor_idx": torch.cat(actor_indices, dim=0),
                "slot_idx": torch.cat(slot_indices, dim=0),
            },
            output_dir / f"targets_rank{rank}.pt",
        )
        for (L, t), bucket in cells.items():
            torch.save(torch.cat(bucket, dim=0), output_dir / f"L{L:02d}_t{t}_rank{rank}.pt")
        processed = len(sample_files[rank::world_size]) - skipped
        wall_s = time.perf_counter() - t0
        print(f"rank{rank}: {processed} samples ({skipped} skipped), {len(sample_ids)} rows in {wall_s:.1f}s")
        return processed, skipped
