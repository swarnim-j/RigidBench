from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from rigidbench.core.paths import ProbePaths, probe_variant_dir

from . import runtime
from .amnesic import sweep as amnesic_sweep
from .cells import CellBuilder
from .extract import ActivationExtractor, load_manifest
from .fit import sweep_probes
from .inlp import fit_cell, load_cell_shards, load_target_shards
from .targets import TARGETS


def _paths(args) -> ProbePaths:
    return ProbePaths(probe_variant_dir(args.output_root, args.variant))


def _cmd_extract(args, rank, world_size, device):
    paths = _paths(args)
    extractor = ActivationExtractor(
        timestep_fracs=args.timestep_fracs,
        layers=args.layers,
        seed=args.seed,
        random_init=args.random_init,
        checkpoint=args.checkpoint,
    )
    extractor.load(device=device, rank=rank, world_size=world_size)
    samples = load_manifest(Path(args.embeddings_dir) / "manifest.jsonl")
    run = runtime.wandb_init(args.wandb_project, args.wandb_run, vars(args)) if rank == 0 else None
    completed, _, failed = extractor.run(
        samples,
        Path(args.embeddings_dir),
        Path(args.data_dir),
        paths.activations_dir,
        rank,
        world_size,
        device,
        wandb_run=run,
    )
    if rank == 0:
        print(f"completed={completed} failed={len(failed)} -> {paths.activations_dir}")
        if run:
            run.summary["completed"] = completed
            run.finish()


def _cmd_cells(args, rank, world_size, device):
    paths = _paths(args)
    targets = [TARGETS[n] for n in (args.targets or list(TARGETS.keys()))]
    sample_files = sorted(paths.activations_dir.glob("*.pt"))
    CellBuilder(targets, args.layers, args.timestep_fracs).run(
        sample_files,
        Path(args.data_dir),
        paths.cells_dir,
        rank,
        world_size,
    )


def _cmd_fit(args, rank, world_size, device):
    paths = _paths(args)
    run = runtime.wandb_init(args.wandb_project, args.wandb_run, vars(args)) if rank == 0 else None
    targets = [TARGETS[n] for n in (args.targets or list(TARGETS.keys()))]
    results = sweep_probes(
        paths.cells_dir,
        targets,
        args.layers,
        args.timestep_fracs,
        args.kinds,
        args.label_modes,
        args.epochs,
        args.batch_size,
        args.lr,
        args.seed,
        rank,
        world_size,
        device,
        wandb_run=run,
    )
    out = paths.probe_results
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(f".rank{rank}.json").write_text(json.dumps(results, indent=2))
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        n = runtime.merge_json_shards(out, world_size)
        print(f"wrote {n} probe results -> {out}")
        if run:
            run.finish()


def _cmd_inlp(args, rank, world_size, device):
    paths = _paths(args)
    targets = load_target_shards(paths.cells_dir)
    y = targets[args.target].float()
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    y_valid = torch.isfinite(y).all(dim=-1)
    cells = [(L, f) for L in args.layers for f in args.timestep_fracs]
    results: dict[str, dict] = {}
    for L, f in runtime.shard(cells, rank, world_size, desc="cells_rank"):
        X = load_cell_shards(paths.cells_dir, L, f)
        valid = torch.isfinite(X).all(dim=-1) & y_valid
        r = fit_cell(X, y, valid, args.max_iters, args.eps, args.reg, device)
        results[f"L{L:02d}_t{f}"] = r
        if rank == 0:
            post = r["r2_test_post"]
            print(
                f"L{L:02d}_t{f}  pre={r['r2_test_pre']:.3f}  "
                f"post[pos/rand/aPC]={post['position']:.3f}/{post['random']:.3f}/{post['alpha_pc']:.3f}  "
                f"rank={r['removed_rank']} (aPC {r['alpha_pc_rank']})  t={r['fit_time_s']:.1f}s"
            )
        del X

    out = paths.inlp(args.target)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out.with_suffix(f".rank{rank}.pt"))
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        merged: dict[str, dict] = {}
        for r_ in range(world_size):
            part = out.with_suffix(f".rank{r_}.pt")
            merged.update(torch.load(part, weights_only=False))
            part.unlink()
        torch.save(merged, out)
        print(f"wrote {len(merged)} cells -> {out}")


def _cmd_amnesic(args, rank, world_size, device):
    paths = _paths(args)
    run = runtime.wandb_init(args.wandb_project, args.wandb_run, vars(args)) if rank == 0 else None
    amnesic_sweep(
        paths.inlp(args.target),
        Path(args.embeddings_dir),
        Path(args.data_dir),
        paths.amnesic(args.target),
        args.layers,
        args.timestep_fracs,
        args.n_samples,
        args.seed,
        rank,
        world_size,
        device,
        wandb_run=run,
    )
    if rank == 0 and run:
        run.finish()


_DISPATCH = {
    "extract": (_cmd_extract, "nccl"),
    "cells": (_cmd_cells, "gloo"),
    "fit": (_cmd_fit, "nccl"),
    "inlp": (_cmd_inlp, "nccl"),
    "amnesic": (_cmd_amnesic, "nccl"),
}


def main():
    p = argparse.ArgumentParser(prog="probe")
    p.add_argument("stage", choices=list(_DISPATCH.keys()))
    p.add_argument("--output_root", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--embeddings_dir")
    p.add_argument("--data_dir")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--timestep_fracs", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--target", default="position")
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--kinds", nargs="+", default=["linear", "mlp"])
    p.add_argument("--label_modes", nargs="+", default=["trained", "shuffled"])
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--reg", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random_init", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT"))
    p.add_argument("--wandb_run", default=os.environ.get("WANDB_RUN_ID"))
    args = p.parse_args()

    fn, backend = _DISPATCH[args.stage]
    rank, world_size, device = runtime.setup(backend=backend)
    try:
        fn(args, rank, world_size, device)
    finally:
        runtime.teardown()


if __name__ == "__main__":
    main()
