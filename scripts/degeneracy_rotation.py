from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RotationResult:
    """Container for near-degenerate subspace rotation results."""

    found_subspace: bool
    subspace_dim: int
    energy_center: float
    edge_before_max: float
    edge_after_max: float
    corner_before_max: float
    corner_after_max: float
    edge_state: np.ndarray | None
    corner_state: np.ndarray | None
    state_indices: np.ndarray


def find_degenerate_groups(eigvals: np.ndarray, abs_tol: float, rel_tol: float) -> list[np.ndarray]:
    """Compatibility wrapper used by the main analysis entrypoint."""
    return group_near_degenerate_indices(
        energies=np.asarray(eigvals, dtype=float),
        abs_delta_e=float(abs_tol),
        relative_delta_e=float(rel_tol),
    )


def group_near_degenerate_indices(
    energies: np.ndarray,
    abs_delta_e: float,
    relative_delta_e: float,
) -> list[np.ndarray]:
    """Group sorted energies into near-degenerate clusters."""
    if energies.size == 0:
        return []
    groups: list[list[int]] = []
    current = [0]
    for idx in range(1, energies.size):
        e_prev = float(energies[idx - 1])
        e_curr = float(energies[idx])
        delta = abs(e_curr - e_prev)
        local_scale = max(abs(e_prev), abs(e_curr), 1e-8)
        tol = max(abs_delta_e, relative_delta_e * local_scale)
        if delta <= tol:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)
    return [np.array(g, dtype=int) for g in groups]


def rotate_subspace_max_projector(
    vectors: np.ndarray,
    projector_diag: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Rotate degenerate subspace and return max-projector state."""
    # vectors: (dim, m), projector_diag: (dim,)
    weighted = vectors * projector_diag[:, None]
    proj_sub = vectors.conj().T @ weighted
    evals, evecs = np.linalg.eigh(proj_sub)
    best = int(np.argmax(np.real(evals)))
    coeff = evecs[:, best]
    state = vectors @ coeff
    norm = float(np.linalg.norm(state))
    if norm > 0:
        state = state / norm
    return state, float(np.real(evals[best])), np.real(evals)


def rotate_max_localized_state(
    eigvecs: np.ndarray,
    indices: np.ndarray,
    projector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper: rotate a given subspace by a diagonal projector."""
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        raise ValueError("indices must be non-empty")
    sub = eigvecs[:, idx]
    proj = np.asarray(projector)
    if proj.ndim == 2:
        # projector provided as cell mask (ny, nx); lift to full basis diagonal.
        proj_diag = np.repeat(proj.reshape(-1).astype(float), 8)
    else:
        proj_diag = proj.reshape(-1).astype(float)
    state, _, eigvals = rotate_subspace_max_projector(sub, proj_diag)
    return state, eigvals


def find_and_rotate_near_zero_subspace(
    energies: np.ndarray,
    eigvecs: np.ndarray,
    edge_projector_diag: np.ndarray,
    corner_projector_diag: np.ndarray,
    abs_delta_e: float,
    relative_delta_e: float,
) -> RotationResult:
    """Find near-zero degenerate cluster and optimize edge/corner localized states."""
    if energies.size == 0:
        return RotationResult(
            found_subspace=False,
            subspace_dim=0,
            energy_center=0.0,
            edge_before_max=0.0,
            edge_after_max=0.0,
            corner_before_max=0.0,
            corner_after_max=0.0,
            edge_state=None,
            corner_state=None,
            state_indices=np.array([], dtype=int),
        )

    groups = group_near_degenerate_indices(
        energies=energies,
        abs_delta_e=abs_delta_e,
        relative_delta_e=relative_delta_e,
    )
    if not groups:
        return RotationResult(
            found_subspace=False,
            subspace_dim=0,
            energy_center=0.0,
            edge_before_max=0.0,
            edge_after_max=0.0,
            corner_before_max=0.0,
            corner_after_max=0.0,
            edge_state=None,
            corner_state=None,
            state_indices=np.array([], dtype=int),
        )

    group_score = [float(np.mean(np.abs(energies[g]))) for g in groups]
    best_group = groups[int(np.argmin(group_score))]
    if best_group.size <= 1:
        idx = int(best_group[0])
        vec = eigvecs[:, idx]
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        edge_w = float(np.sum(np.abs(vec) ** 2 * edge_projector_diag))
        corner_w = float(np.sum(np.abs(vec) ** 2 * corner_projector_diag))
        return RotationResult(
            found_subspace=False,
            subspace_dim=1,
            energy_center=float(energies[idx]),
            edge_before_max=edge_w,
            edge_after_max=edge_w,
            corner_before_max=corner_w,
            corner_after_max=corner_w,
            edge_state=vec,
            corner_state=vec,
            state_indices=np.array([idx], dtype=int),
        )

    sub_vecs = eigvecs[:, best_group]
    sub_prob = np.abs(sub_vecs) ** 2
    edge_before = float(np.max(np.sum(sub_prob * edge_projector_diag[:, None], axis=0)))
    corner_before = float(np.max(np.sum(sub_prob * corner_projector_diag[:, None], axis=0)))

    edge_state, edge_after, _ = rotate_subspace_max_projector(
        vectors=sub_vecs,
        projector_diag=edge_projector_diag,
    )
    corner_state, corner_after, _ = rotate_subspace_max_projector(
        vectors=sub_vecs,
        projector_diag=corner_projector_diag,
    )

    return RotationResult(
        found_subspace=True,
        subspace_dim=int(best_group.size),
        energy_center=float(np.mean(energies[best_group])),
        edge_before_max=edge_before,
        edge_after_max=edge_after,
        corner_before_max=corner_before,
        corner_after_max=corner_after,
        edge_state=edge_state,
        corner_state=corner_state,
        state_indices=best_group.astype(int),
    )
