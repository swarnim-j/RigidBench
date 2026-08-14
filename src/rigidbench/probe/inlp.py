from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Control:
    """Linear residual-stream intervention `x -> x - alpha (B B^T) x`."""

    name: str
    basis: torch.Tensor
    alpha: float


def _ridge_regress(X: torch.Tensor, y: torch.Tensor, reg: float = 1e-2) -> torch.Tensor:
    """fp64 closed-form for numerical stability at our N and d."""
    Xd, yd = X.double(), y.double()
    N, d = Xd.shape
    XtX = Xd.T @ Xd + reg * N * torch.eye(d, device=Xd.device, dtype=Xd.dtype)
    Xty = Xd.T @ yd
    return torch.linalg.solve(XtX, Xty).T.to(X.dtype)


def _r2(W: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> float:
    pred = X @ W.T
    ss_res = ((y - pred) ** 2).sum().item()
    ss_tot = ((y - y.mean(0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _null_basis(W: torch.Tensor, tol_ratio: float = 1e-6) -> torch.Tensor:
    _, S, Vh = torch.linalg.svd(W, full_matrices=True)
    s_max = float(S.max()) if S.numel() > 0 else 0.0
    rank = int((S > tol_ratio * s_max).sum()) if s_max > 0 else 0
    return Vh[rank:].T


def fit_inlp(
    X: torch.Tensor,
    y: torch.Tensor,
    max_iters: int = 30,
    eps: float = 0.05,
    reg: float = 0.05,
    verbose: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """Ravfogel et al. 2020 INLP for regression. Iterates linear ridge + nullspace projection until R^2 < eps."""
    device, dtype = X.device, X.dtype
    d = X.shape[1]
    X_proj = X.clone()
    collected: list[torch.Tensor] = []
    history: list[float] = []
    for i in range(max_iters):
        W_i = _ridge_regress(X_proj, y, reg=reg)
        r2 = _r2(W_i, X_proj, y)
        history.append(r2)
        if verbose:
            print(f"iter {i:2d}: R^2 = {r2:.4f}")
        if r2 < eps:
            break
        collected.append(W_i)
        B_i = _null_basis(W_i)
        X_proj = (X_proj @ B_i) @ B_i.T
    if not collected:
        return torch.eye(d, device=device, dtype=dtype), history
    B_joint = _null_basis(torch.cat(collected, dim=0))
    return B_joint @ B_joint.T, history


def random_orthonormal(d: int, rank: int, device: str, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Haar-uniform rank-`rank` orthonormal basis in d dims."""
    g = torch.Generator(device=device).manual_seed(seed)
    R = torch.randn(d, rank, device=device, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(R, mode="reduced")
    return Q


def fit_alpha_pc(
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    target_var_loss_tr: float,
) -> tuple[torch.Tensor, float]:
    """Variance-matched control via closed-form alpha. Picks fewest top-K PCs covering the target, then alpha < 1."""
    cum = torch.cumsum(eigvals, dim=0)
    K = int((cum < target_var_loss_tr).sum().item()) + 1
    K = min(K, eigvecs.shape[1])
    basis = eigvecs[:, :K].contiguous()
    cum_lambda = cum[K - 1].item()
    if cum_lambda <= 0:
        return basis, 0.0
    ratio = min(1.0, target_var_loss_tr / cum_lambda)
    return basis, 1.0 - ((1.0 - ratio) ** 0.5)


def variance_along(X_centered: torch.Tensor, basis: torch.Tensor) -> float:
    n = X_centered.shape[0]
    return ((X_centered @ basis).pow(2).sum() / (n - 1)).item()


@torch.no_grad()
def apply_subtract(X: torch.Tensor, basis: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return X - alpha * (X @ basis) @ basis.T


@torch.no_grad()
def fit_cell(
    X: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
    max_iters: int,
    eps: float,
    reg: float,
    device: str,
    seed: int = 0,
) -> dict:
    idx = torch.where(valid)[0]
    perm = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))]
    n_tr = int(0.8 * len(perm))
    tr, te = perm[:n_tr], perm[n_tr:]
    X_tr, y_tr = X[tr].to(device), y[tr].to(device)
    X_te, y_te = X[te].to(device), y[te].to(device)
    n, d = X_tr.shape

    W_base = _ridge_regress(X_tr, y_tr, reg=reg)
    r2_pre = _r2(W_base, X_te, y_te)
    var_pre = X_te.pow(2).mean().item()

    t0 = time.perf_counter()
    P_null, history = fit_inlp(X_tr, y_tr, max_iters=max_iters, eps=eps, reg=reg)
    fit_time_s = time.perf_counter() - t0
    pos = Control("position", _null_basis(P_null), 1.0)

    rand = Control(
        "random",
        random_orthonormal(d, pos.basis.shape[1], device=device, dtype=X_tr.dtype, seed=seed + 1),
        1.0,
    )

    X_centered = X_tr - X_tr.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    eigvals = S.pow(2) / max(n - 1, 1)
    target_var = variance_along(X_centered, pos.basis)
    apc_basis, apc_alpha = fit_alpha_pc(eigvals, Vh.T, target_var)
    apc = Control("alpha_pc", apc_basis, apc_alpha)

    def diagnostics(c: Control):
        if c.alpha <= 0:
            return {"r2_test_post": r2_pre, "var_pres": 1.0}
        X_te_p = apply_subtract(X_te, c.basis, c.alpha)
        X_tr_p = apply_subtract(X_tr, c.basis, c.alpha)
        W_p = _ridge_regress(X_tr_p, y_tr, reg=reg)
        return {
            "r2_test_post": _r2(W_p, X_te_p, y_te),
            "var_pres": X_te_p.pow(2).mean().item() / var_pre,
        }

    return {
        "controls": {c.name: {"basis": c.basis.cpu(), "alpha": float(c.alpha)} for c in (pos, rand, apc)},
        "history": history,
        "fit_time_s": fit_time_s,
        "n_train": len(tr),
        "n_test": len(te),
        "removed_rank": pos.basis.shape[1],
        "alpha_pc_rank": apc.basis.shape[1],
        "r2_test_pre": r2_pre,
        "r2_test_post": {c.name: diagnostics(c)["r2_test_post"] for c in (pos, rand, apc)},
        "fraction_variance_preserved": {c.name: diagnostics(c)["var_pres"] for c in (pos, rand, apc)},
    }


def load_cell_shards(cells_dir: Path, layer: int, frac: float) -> torch.Tensor:
    shards = sorted(cells_dir.glob(f"L{layer:02d}_t{frac}_rank*.pt"))
    return torch.cat([torch.load(s, weights_only=True) for s in shards], dim=0).float()


def load_target_shards(cells_dir: Path) -> dict:
    shards = sorted(cells_dir.glob("targets_rank*.pt"))
    parts = [torch.load(s, weights_only=False) for s in shards]
    out: dict = {}
    for k in parts[0].keys():
        if isinstance(parts[0][k], torch.Tensor):
            out[k] = torch.cat([p[k] for p in parts], dim=0)
        else:
            out[k] = [item for p in parts for item in p[k]]
    return out
