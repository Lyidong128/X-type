from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from scripts.run_scan import build_ribbon_hamiltonian
from scripts.spatial_projectors import (
    RegionProjectors,
    build_cell_probability_grid as _build_cell_probability_grid,
    compute_region_weights as _compute_region_weights,
)


@dataclass
class MethodSelection:
    """Container of one selected mode (single-state selection output)."""

    method_name: str
    index: int
    energy: float
    vector: np.ndarray
    weight: float
    window_low: float
    window_high: float


@dataclass
class RibbonWindow:
    """Ribbon-inferred energy window for flake-side state search."""

    low: float
    high: float
    center: float
    ky_values: np.ndarray
    eigvals: np.ndarray


@dataclass
class StateMetrics:
    """Per-eigenstate metrics used for selection."""

    index: int
    energy: float
    w_left: float
    w_right: float
    w_top: float
    w_bottom: float
    w_edge_total: float
    w_corner: float
    w_bulk: float


def compute_state_metrics(
    energies: np.ndarray,
    evecs: np.ndarray,
    nx: int,
    ny: int,
    projectors: RegionProjectors,
    candidate_indices: Iterable[int],
) -> list[StateMetrics]:
    """Compute spatial weights for each candidate eigenstate."""
    def _get_w(weight_dict: dict[str, float], key: str) -> float:
        if key in weight_dict:
            return float(weight_dict[key])
        alt = key.replace("w_", "W_")
        if alt in weight_dict:
            return float(weight_dict[alt])
        return 0.0

    out: list[StateMetrics] = []
    for idx in candidate_indices:
        vec = evecs[:, idx]
        grid = _build_cell_probability_grid(vec, nx=nx, ny=ny)
        weights = _compute_region_weights(grid, projectors)
        out.append(
            StateMetrics(
                index=int(idx),
                energy=float(energies[idx]),
                w_left=_get_w(weights, "w_left"),
                w_right=_get_w(weights, "w_right"),
                w_top=_get_w(weights, "w_top"),
                w_bottom=_get_w(weights, "w_bottom"),
                w_edge_total=_get_w(weights, "w_edge_total"),
                w_corner=_get_w(weights, "w_corner"),
                w_bulk=_get_w(weights, "w_bulk"),
            )
        )
    return out


def _pick_candidate_indices(energies: np.ndarray, center: float, half_width: float, fallback_half_width: float) -> np.ndarray:
    mask = np.abs(energies - center) <= half_width
    idx = np.where(mask)[0]
    if idx.size == 0:
        mask = np.abs(energies - center) <= fallback_half_width
        idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmin(np.abs(energies - center)))], dtype=int)
    return idx


def pick_old_mode_index(energies: np.ndarray) -> int:
    """Baseline old rule: pick mode closest to zero energy."""
    return int(np.argmin(np.abs(energies)))


def choose_old_mode(eigvals: np.ndarray, eigvecs: np.ndarray) -> MethodSelection:
    """Compatibility helper: old_mode near-zero single state."""
    idx = pick_old_mode_index(eigvals)
    return MethodSelection(
        method_name="old_mode",
        index=idx,
        energy=float(eigvals[idx]),
        vector=eigvecs[:, idx],
        weight=float(np.nan),
        window_low=float(eigvals[idx]),
        window_high=float(eigvals[idx]),
    )


def _argmax_metric(metrics: list[StateMetrics], key: str) -> StateMetrics:
    if not metrics:
        raise ValueError("metrics list is empty")
    return max(metrics, key=lambda m: getattr(m, key))


def pick_best_edge_state(metrics: list[StateMetrics]) -> StateMetrics:
    """Pick single state with maximal total edge weight."""
    return _argmax_metric(metrics, "w_edge_total")


def pick_best_corner_state(metrics: list[StateMetrics]) -> StateMetrics:
    """Pick single state with maximal corner weight."""
    return _argmax_metric(metrics, "w_corner")


def pick_best_side_states(metrics: list[StateMetrics]) -> dict[str, StateMetrics]:
    """Pick most localized state for each side."""
    return {
        "left": _argmax_metric(metrics, "w_left"),
        "right": _argmax_metric(metrics, "w_right"),
        "top": _argmax_metric(metrics, "w_top"),
        "bottom": _argmax_metric(metrics, "w_bottom"),
    }


def choose_best_edge_state(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    projectors: RegionProjectors,
    center: float,
    half_width: float,
    fallback_half_width: float,
) -> MethodSelection:
    """Choose single state with maximal edge weight in energy window."""
    idx = _pick_candidate_indices(eigvals, center, half_width, fallback_half_width)
    metrics = compute_state_metrics(
        energies=eigvals,
        evecs=eigvecs,
        nx=projectors.nx,
        ny=projectors.ny,
        projectors=projectors,
        candidate_indices=idx,
    )
    best = pick_best_edge_state(metrics)
    return MethodSelection(
        method_name="best_edge_state",
        index=best.index,
        energy=best.energy,
        vector=eigvecs[:, best.index],
        weight=best.w_edge_total,
        window_low=float(np.min(eigvals[idx])),
        window_high=float(np.max(eigvals[idx])),
    )


def choose_best_corner_state(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    projectors: RegionProjectors,
    center: float,
    half_width: float,
    fallback_half_width: float,
) -> MethodSelection:
    """Choose single state with maximal corner weight in energy window."""
    idx = _pick_candidate_indices(eigvals, center, half_width, fallback_half_width)
    metrics = compute_state_metrics(
        energies=eigvals,
        evecs=eigvecs,
        nx=projectors.nx,
        ny=projectors.ny,
        projectors=projectors,
        candidate_indices=idx,
    )
    best = pick_best_corner_state(metrics)
    return MethodSelection(
        method_name="best_corner_state",
        index=best.index,
        energy=best.energy,
        vector=eigvecs[:, best.index],
        weight=best.w_corner,
        window_low=float(np.min(eigvals[idx])),
        window_high=float(np.max(eigvals[idx])),
    )


def choose_best_side_states(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    side_projectors: dict[str, np.ndarray],
    center: float,
    half_width: float,
    fallback_half_width: float,
) -> dict[str, MethodSelection]:
    """Choose best left/right/top/bottom states in the same energy window."""
    idx = _pick_candidate_indices(eigvals, center, half_width, fallback_half_width)
    if not side_projectors:
        return {}
    out: dict[str, MethodSelection] = {}
    for side, proj in side_projectors.items():
        if proj.ndim != 2:
            raise ValueError("side_projectors must contain 2D masks.")
        ny, nx = proj.shape
        weights = []
        for i in idx:
            grid = _build_cell_probability_grid(eigvecs[:, i], nx=nx, ny=ny)
            weights.append(float(np.sum(grid[proj])))
        weights = np.array(weights, dtype=float)
        best_local = int(np.argmax(weights))
        best_idx = int(idx[best_local])
        out[side] = MethodSelection(
            method_name=f"optional_best_{side}_state",
            index=best_idx,
            energy=float(eigvals[best_idx]),
            vector=eigvecs[:, best_idx],
            weight=float(weights[best_local]),
            window_low=float(np.min(eigvals[idx])),
            window_high=float(np.max(eigvals[idx])),
        )
    return out


def detect_ribbon_window(
    v: float,
    t: float,
    lm: float,
    nx: int,
    nk: int,
    edge_cells: int,
    edge_threshold: float,
) -> RibbonWindow:
    """Infer ribbon edge-branch energy window and keep raw ribbon spectrum."""
    ky_values = np.linspace(-np.pi, np.pi, nk)
    eigvals = []
    edge_energies = []
    for ky in ky_values:
        ham = build_ribbon_hamiltonian(v=v, t=t, lm=lm, ky=ky, w=1.0, j=0.0, nx=nx)
        e, vecs = np.linalg.eigh(ham)
        e = np.real(e)
        eigvals.append(e)
        probs = np.abs(vecs) ** 2
        dim = probs.shape[0]
        cell_prob = probs.reshape(nx, 8, dim).sum(axis=1)
        edge_weight = cell_prob[:edge_cells, :].sum(axis=0) + cell_prob[-edge_cells:, :].sum(axis=0)
        edge_mask = edge_weight >= edge_threshold
        edge_energies.extend(e[edge_mask].tolist())

    eigvals_arr = np.array(eigvals)
    if edge_energies:
        low = float(np.percentile(edge_energies, 20))
        high = float(np.percentile(edge_energies, 80))
        if high <= low:
            low = float(np.min(edge_energies))
            high = float(np.max(edge_energies))
    else:
        flat = eigvals_arr.reshape(-1)
        low = float(np.percentile(flat, 48))
        high = float(np.percentile(flat, 52))
    center = 0.5 * (low + high)
    return RibbonWindow(low=low, high=high, center=center, ky_values=ky_values, eigvals=eigvals_arr)


def map_for_state(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Return normalized spatial map for a single state vector."""
    return _build_cell_probability_grid(vec, nx=nx, ny=ny)


def ldos_map(
    energies: np.ndarray,
    evecs: np.ndarray,
    nx: int,
    ny: int,
    center: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate unweighted LDOS map in an energy window."""
    mask = np.abs(energies - center) <= half_width
    picked = np.where(mask)[0]
    if picked.size == 0:
        picked = np.array([int(np.argmin(np.abs(energies - center)))], dtype=int)

    acc = np.zeros((ny, nx), dtype=float)
    for idx in picked:
        acc += _build_cell_probability_grid(evecs[:, idx], nx=nx, ny=ny)
    total = float(np.sum(acc))
    if total > 0:
        acc = acc / total
    return acc, picked


def build_cell_probability_grid(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Compatibility wrapper for main script import."""
    return _build_cell_probability_grid(vec, nx=nx, ny=ny)


def compute_region_weights(grid: np.ndarray, projectors: RegionProjectors) -> dict[str, float]:
    """Compatibility wrapper to compute region weights from a grid."""
    return _compute_region_weights(grid, projectors)

