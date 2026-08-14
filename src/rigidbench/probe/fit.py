from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from . import runtime
from .inlp import load_cell_shards, load_target_shards
from .targets import Target


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.net(x)


def build_probe(kind: str, in_dim: int, out_dim: int) -> nn.Module:
    if kind == "linear":
        return LinearProbe(in_dim, out_dim)
    if kind == "mlp":
        return MLPProbe(in_dim, out_dim)
    raise ValueError(f"unknown probe kind: {kind!r}")


def _row_splits(sample_ids: list[str], seed: int) -> dict[str, torch.Tensor]:
    unique = sorted(set(sample_ids))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique), generator=rng).tolist()
    n_train, n_val = int(0.8 * len(unique)), int(0.1 * len(unique))
    split_of = {}
    for i, idx in enumerate(perm):
        sid = unique[idx]
        split_of[sid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    row_split = [split_of[sid] for sid in sample_ids]
    return {
        s: torch.tensor([i for i, sp in enumerate(row_split) if sp == s], dtype=torch.long)
        for s in ("train", "val", "test")
    }


@torch.no_grad()
def _score(probe, X, y, metric: str) -> float:
    if len(X) == 0:
        return float("nan")
    pred = probe(X)
    if metric == "auc":
        y_np = y.view(-1).cpu().numpy()
        p_np = pred.view(-1).cpu().numpy()
        if y_np.min() == y_np.max():
            return float("nan")
        return float(roc_auc_score(y_np, p_np))
    ss_res = ((y - pred) ** 2).sum().item()
    ss_tot = ((y - y.mean(dim=0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_probe(
    X: torch.Tensor,
    target: Target,
    y_all: torch.Tensor,
    mask: torch.Tensor,
    splits: dict[str, torch.Tensor],
    variant: str,
    kind: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict:
    """variant='shuffled' is the Hewitt-Liang selectivity control."""
    out_dim = target.output_dim
    y = y_all.view(-1, out_dim).to(device)
    tr = splits["train"][mask[splits["train"]]].to(device)
    va = splits["val"][mask[splits["val"]]].to(device)
    te = splits["test"][mask[splits["test"]]].to(device)
    Xt, yt = X[tr], y[tr]
    if variant == "shuffled":
        gen = torch.Generator(device="cpu").manual_seed(seed)
        yt = yt[torch.randperm(len(yt), generator=gen)]
    probe = build_probe(kind, X.shape[-1], out_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = F.binary_cross_entropy_with_logits if target.loss_type == "bce" else F.mse_loss
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(perm), batch_size):
            idx = perm[s : s + batch_size]
            loss = loss_fn(probe(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return {
        "target": target.name,
        "kind": kind,
        "variant": variant,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "val_metric": _score(probe, X[va], y[va], target.metric),
        "test_metric": _score(probe, X[te], y[te], target.metric),
    }


def sweep_probes(
    cells_dir,
    targets: list[Target],
    layers: list[int],
    timestep_fracs: list[float],
    kinds: list[str],
    variants: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    rank: int,
    world_size: int,
    device: str,
    wandb_run=None,
) -> list[dict]:
    target_data = load_target_shards(cells_dir)
    splits = _row_splits(target_data["sample_id"], seed)
    cells = [(L, t) for L in layers for t in timestep_fracs]
    if rank == 0:
        print(f"{len(cells)} cells total")

    results: list[dict] = []
    for L, t in runtime.shard(cells, rank, world_size, desc="cells_rank"):
        X = load_cell_shards(cells_dir, L, t).to(device, torch.float32)
        valid = torch.isfinite(X).all(dim=-1).cpu()
        for tgt in targets:
            y_all = target_data[tgt.name].float()
            y_valid = torch.isfinite(y_all).all(dim=-1) if y_all.dim() > 1 else torch.isfinite(y_all)
            mask = valid & y_valid
            for kind in kinds:
                for variant in variants:
                    r = fit_probe(X, tgt, y_all, mask, splits, variant, kind, device, epochs, batch_size, lr, seed)
                    r.update({"layer": L, "timestep_frac": t})
                    results.append(r)
                    if wandb_run:
                        wandb_run.log(r)
        del X
    return results
