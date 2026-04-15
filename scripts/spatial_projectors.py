"""Spatial projector utilities for wavefunction localization analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionProjectors:
    """Region masks and projector diagonals for a flake grid."""

    nx: int
    ny: int
    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    edge: np.ndarray
    corner: np.ndarray
    bulk: np.ndarray
    left_diag: np.ndarray
    right_diag: np.ndarray
    top_diag: np.ndarray
    bottom_diag: np.ndarray
    edge_diag: np.ndarray
    corner_diag: np.ndarray
    bulk_diag: np.ndarray

    def side_projectors(self) -> dict[str, np.ndarray]:
        """Return side masks for optional side-state selection."""
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }


def _projector_diagonal(mask: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Build diagonal projector in full basis (8 dof per cell)."""
    return np.repeat(mask.reshape(nx * ny).astype(float), 8)


def build_region_projectors(nx: int, ny: int, edge_width: int, corner_size: int) -> RegionProjectors:
    """Construct geometric projectors for left/right/top/bottom/edge/corner/bulk."""
    ew = max(1, min(edge_width, nx // 2, ny // 2))
    cs = max(1, min(corner_size, nx // 2, ny // 2))

    left = np.zeros((ny, nx), dtype=bool)
    right = np.zeros((ny, nx), dtype=bool)
    top = np.zeros((ny, nx), dtype=bool)
    bottom = np.zeros((ny, nx), dtype=bool)
    left[:, :ew] = True
    right[:, nx - ew :] = True
    top[ny - ew :, :] = True
    bottom[:ew, :] = True

    edge = left | right | top | bottom
    corner = np.zeros((ny, nx), dtype=bool)
    corner[:cs, :cs] = True
    corner[:cs, nx - cs :] = True
    corner[ny - cs :, :cs] = True
    corner[ny - cs :, nx - cs :] = True
    bulk = ~(edge | corner)

    return RegionProjectors(
        nx=nx,
        ny=ny,
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        edge=edge,
        corner=corner,
        bulk=bulk,
        left_diag=_projector_diagonal(left, nx, ny),
        right_diag=_projector_diagonal(right, nx, ny),
        top_diag=_projector_diagonal(top, nx, ny),
        bottom_diag=_projector_diagonal(bottom, nx, ny),
        edge_diag=_projector_diagonal(edge, nx, ny),
        corner_diag=_projector_diagonal(corner, nx, ny),
        bulk_diag=_projector_diagonal(bulk, nx, ny),
    )


def build_cell_probability_grid(eigvec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Convert a full eigenvector to normalized cell probability grid."""
    n_cells = nx * ny
    prob = np.zeros(n_cells, dtype=float)
    for cid in range(n_cells):
        start = cid * 8
        prob[cid] = float(np.sum(np.abs(eigvec[start : start + 8]) ** 2))
    grid = prob.reshape((ny, nx))
    total = float(np.sum(grid))
    return grid / total if total > 0 else grid


def compute_region_weights(grid: np.ndarray, projectors: RegionProjectors) -> dict[str, float]:
    """Compute W_left/W_right/W_top/W_bottom/W_edge_total/W_corner/W_bulk."""
    return {
        "W_left": float(np.sum(grid[projectors.left])),
        "W_right": float(np.sum(grid[projectors.right])),
        "W_top": float(np.sum(grid[projectors.top])),
        "W_bottom": float(np.sum(grid[projectors.bottom])),
        "W_edge_total": float(np.sum(grid[projectors.edge])),
        "W_corner": float(np.sum(grid[projectors.corner])),
        "W_bulk": float(np.sum(grid[projectors.bulk])),
    }


def compute_side_weights(grid: np.ndarray, side_masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute side-specific localization weights for a probability grid."""
    return {name: float(np.sum(grid[mask])) for name, mask in side_masks.items()}


def spatial_ldos_from_indices(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    nx: int,
    ny: int,
    center_energy: float,
    half_width: float,
) -> np.ndarray:
    """Accumulate normalized LDOS map over an energy window."""
    picked = np.where(np.abs(eigvals - center_energy) <= half_width)[0]
    if picked.size == 0:
        picked = np.array([int(np.argmin(np.abs(eigvals - center_energy)))], dtype=int)
    acc = np.zeros((ny, nx), dtype=float)
    for idx in picked:
        acc += build_cell_probability_grid(eigvecs[:, idx], nx=nx, ny=ny)
    total = float(np.sum(acc))
    return acc / total if total > 0 else acc

