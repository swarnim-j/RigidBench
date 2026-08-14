from __future__ import annotations

import math
import random
from collections.abc import Sequence

Interval = tuple[float, float]
IntervalSet = list[Interval]

_TOL = 1e-9


def normalize(intervals: Sequence[tuple[float, float]]) -> IntervalSet:
    pieces: IntervalSet = []
    for lo, hi in intervals:
        if hi - lo >= 360 - _TOL:
            return [(0.0, 360.0)]
        lo_m = lo % 360
        hi_m = (hi % 360 + 360) if hi < lo else (hi - lo + lo_m)
        if hi_m <= 360 + _TOL:
            pieces.append((lo_m, min(hi_m, 360.0)))
        else:
            pieces.append((lo_m, 360.0))
            pieces.append((0.0, hi_m - 360.0))
    pieces = [(lo, hi) for lo, hi in pieces if hi - lo > _TOL]
    pieces.sort()
    merged: IntervalSet = []
    for lo, hi in pieces:
        if merged and lo <= merged[-1][1] + _TOL:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    if (
        len(merged) >= 2
        and merged[0][0] <= _TOL
        and merged[-1][1] >= 360 - _TOL
        and (merged[-1][1] - merged[-1][0]) + (merged[0][1] - merged[0][0]) >= 360 - _TOL
    ):
        return [(0.0, 360.0)]
    return merged


def shift(intervals: Sequence[tuple[float, float]], theta: float) -> IntervalSet:
    return normalize([(lo + theta, hi + theta) for lo, hi in intervals])


def fit_rotations(L_T: float, W_T: float, L_S: float, W_S: float) -> IntervalSet:
    """Rotations ψ (deg, surface-local) at which (L_T × W_T) fits inside (L_S × W_S)."""
    if min(L_T, W_T, L_S, W_S) <= 0:
        raise ValueError(f"non-positive footprint: task=({L_T},{W_T}) surface=({L_S},{W_S})")
    x_band = _solve_extent_band(L_T / 2, W_T / 2, L_S / 2)
    y_band = _solve_extent_band(W_T / 2, L_T / 2, W_S / 2)
    on_quadrant = _intersect_in_quadrant(x_band, y_band)
    tiled: IntervalSet = []
    for lo, hi in on_quadrant:
        tiled.append((lo, hi))
        tiled.append((180 - hi, 180 - lo))
        tiled.append((180 + lo, 180 + hi))
        tiled.append((360 - hi, 360 - lo))
    return normalize(tiled)


def sample_from_intervals(intervals: Sequence[tuple[float, float]], rng: random.Random) -> float:
    intervals = normalize(intervals)
    if not intervals:
        raise ValueError("cannot sample from empty interval set")
    weights = [hi - lo for lo, hi in intervals]
    ((lo, hi),) = rng.choices(intervals, weights=weights, k=1)
    return rng.uniform(lo, hi)


def _solve_extent_band(alpha: float, beta: float, gamma: float) -> IntervalSet:
    """Solve α·cos ψ + β·sin ψ ≤ γ for ψ ∈ [0°, 90°], yielding up to two bands."""
    R = math.hypot(alpha, beta)
    if gamma >= R - _TOL:
        return [(0.0, 90.0)]
    fits_at_0 = alpha <= gamma + _TOL
    fits_at_90 = beta <= gamma + _TOL
    if not fits_at_0 and not fits_at_90:
        return []
    psi_peak = math.degrees(math.atan2(beta, alpha))
    delta = math.degrees(math.acos(max(-1.0, min(1.0, gamma / R))))
    bands: IntervalSet = []
    if fits_at_0:
        bands.append((0.0, max(0.0, min(90.0, psi_peak - delta))))
    if fits_at_90:
        bands.append((max(0.0, min(90.0, psi_peak + delta)), 90.0))
    return [(lo, hi) for lo, hi in bands if hi - lo > _TOL]


def _intersect_in_quadrant(A: IntervalSet, B: IntervalSet) -> IntervalSet:
    out: IntervalSet = []
    i = j = 0
    while i < len(A) and j < len(B):
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if hi - lo > _TOL:
            out.append((lo, hi))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out
