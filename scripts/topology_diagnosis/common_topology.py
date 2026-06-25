#!/usr/bin/env python3
"""Shared helpers for topology diagnosis scripts."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hoti_v_lt_0p6.run_hoti_v_lt_0p6 import h0_k, h_soc_orbital, h8_k, unitary_part


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def orb_spin_idx(orb: int, spin: int) -> int:
    return orb * 2 + spin


def open_open_index(ix: int, iy: int, orb: int, spin: int, lx: int) -> int:
    return ((iy * lx + ix) * 8) + orb_spin_idx(orb, spin)


def compute_bulk_gap(v: float, t: float, w: float, lm: float, nk: int = 61, n_occ: int = 4) -> float:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    valence_max = -np.inf
    conduction_min = np.inf
    for kx, ky in product(kgrid, kgrid):
        evals = np.real(np.linalg.eigvalsh(h8_k(float(kx), float(ky), v=v, t=t, w=w, lm=lm)))
        valence_max = max(valence_max, float(evals[n_occ - 1]))
        conduction_min = min(conduction_min, float(evals[n_occ]))
    return float(conduction_min - valence_max)


def build_open_open_hamiltonian(
    lx: int,
    ly: int,
    v: float,
    t: float,
    w: float,
    lm: float,
    termination: str = "A",
) -> np.ndarray:
    n = lx * ly * 8
    h = np.zeros((n, n), dtype=complex)
    hs = h_soc_orbital(lm)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    tx = np.zeros((8, 8), dtype=complex)
    ty = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[orb_spin_idx(0, spin), orb_spin_idx(3, spin)] = w
        ty[orb_spin_idx(1, spin), orb_spin_idx(2, spin)] = w

    for iy in range(ly):
        for ix in range(lx):
            v03 = v
            if termination == "B" and (ix == 0 or ix == lx - 1):
                # Different boundary cut: remove boundary intracell 0-3 dimer.
                v03 = 0.0

            h0_cell = np.zeros((4, 4), dtype=complex)
            h0_cell[0, 1] = t
            h0_cell[0, 2] = t
            h0_cell[0, 3] = v03
            h0_cell[1, 0] = t
            h0_cell[1, 2] = v
            h0_cell[1, 3] = t
            h0_cell[2, 0] = t
            h0_cell[2, 1] = v
            h0_cell[2, 3] = t
            h0_cell[3, 0] = v03
            h0_cell[3, 1] = t
            h0_cell[3, 2] = t
            onsite8 = np.kron(h0_cell, s0) + np.kron(hs, sz)

            s = (iy * lx + ix) * 8
            h[s : s + 8, s : s + 8] += onsite8

            if ix > 0:
                p = (iy * lx + (ix - 1)) * 8
                h[s : s + 8, p : p + 8] += tx
                h[p : p + 8, s : s + 8] += tx.conj().T
            if iy > 0:
                p = ((iy - 1) * lx + ix) * 8
                h[s : s + 8, p : p + 8] += ty
                h[p : p + 8, s : s + 8] += ty.conj().T

    h = 0.5 * (h + h.conj().T)
    return h


def diagonalize_open_open(h: np.ndarray, num_eigs: int = 40) -> tuple[np.ndarray, np.ndarray]:
    n = h.shape[0]
    if n <= 1500:
        evals, evecs = np.linalg.eigh(h)
        return np.real(evals), evecs

    # Fallback for unexpectedly large matrices.
    k = min(max(8, num_eigs), n - 2)
    evals, evecs = eigsh(h, k=k, sigma=0.0, which="LM")
    order = np.argsort(np.real(evals))
    evals = np.real(evals[order])
    evecs = evecs[:, order]
    return evals, evecs


def cell_density(vec: np.ndarray, lx: int, ly: int) -> np.ndarray:
    rho = np.zeros((ly, lx), dtype=float)
    for iy in range(ly):
        for ix in range(lx):
            s = (iy * lx + ix) * 8
            rho[iy, ix] = float(np.sum(np.abs(vec[s : s + 8]) ** 2))
    return rho


def region_masks(lx: int, ly: int, corner_size: int = 2, edge_width: int = 2):
    corner = np.zeros((ly, lx), dtype=bool)
    corner[:corner_size, :corner_size] = True
    corner[:corner_size, lx - corner_size :] = True
    corner[ly - corner_size :, :corner_size] = True
    corner[ly - corner_size :, lx - corner_size :] = True

    edge = np.zeros((ly, lx), dtype=bool)
    edge[:edge_width, :] = True
    edge[ly - edge_width :, :] = True
    edge[:, :edge_width] = True
    edge[:, lx - edge_width :] = True
    edge &= ~corner

    bulk = ~(corner | edge)
    return corner, edge, bulk


def state_weights(rho: np.ndarray, corner: np.ndarray, edge: np.ndarray, bulk: np.ndarray) -> tuple[float, float, float]:
    w_corner = float(np.sum(rho[corner]))
    w_edge = float(np.sum(rho[edge]))
    w_bulk = float(np.sum(rho[bulk]))
    return w_corner, w_edge, w_bulk


def classify_state(w_corner: float, w_edge: float) -> str:
    if w_corner > 0.6:
        return "corner_state"
    if w_edge > 0.6:
        return "edge_state"
    return "bulk_or_mixed_state"


def choose_in_gap_indices(evals: np.ndarray, bulk_gap: float, gap_closing_thr: float = 1e-4) -> tuple[np.ndarray, float, str]:
    if bulk_gap <= gap_closing_thr:
        return np.array([], dtype=int), max(0.05, 0.4 * max(bulk_gap, 0.0)), "gap_closing"
    window = max(0.05, 0.4 * bulk_gap)
    idx = np.where(np.abs(evals) < window)[0]
    if idx.size == 0:
        return np.array([], dtype=int), window, "no_clear_ingap"
    return idx.astype(int), window, "ok"


def commutator_norm_h_sz(v: float, t: float, w: float, lm: float, nk: int = 11) -> float:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    sz = np.kron(np.eye(4, dtype=complex), np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))
    max_norm = 0.0
    for kx, ky in product(kgrid, kgrid):
        h = h8_k(float(kx), float(ky), v=v, t=t, w=w, lm=lm)
        comm = h @ sz - sz @ h
        denom = max(float(np.linalg.norm(h)), 1e-15)
        max_norm = max(max_norm, float(np.linalg.norm(comm) / denom))
    return float(max_norm)


def chern_fukui(h_func, n_occ: int, nk: int) -> float:
    occ = [[None for _ in range(nk)] for __ in range(nk)]
    for i in range(nk):
        kx = -np.pi + 2.0 * np.pi * (i / nk)
        for j in range(nk):
            ky = -np.pi + 2.0 * np.pi * (j / nk)
            _, vecs = np.linalg.eigh(h_func(float(kx), float(ky)))
            occ[i][j] = vecs[:, :n_occ]

    def link(a: np.ndarray, b: np.ndarray) -> complex:
        q = unitary_part(a.conj().T @ b)
        d = np.linalg.det(q)
        if abs(d) < 1e-14:
            return 1.0 + 0.0j
        return d / abs(d)

    ux = np.zeros((nk, nk), dtype=complex)
    uy = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            ux[i, j] = link(occ[i][j], occ[ip][j])
            uy[i, j] = link(occ[i][j], occ[i][jp])

    f12 = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            z = ux[i, j] * uy[ip, j] / (ux[i, jp] * uy[i, j])
            f12[i, j] = np.log(z)
    return float(np.real(np.sum(f12) / (2j * np.pi)))


def h_spin_block(kx: float, ky: float, v: float, t: float, w: float, lm: float, spin_sign: int) -> np.ndarray:
    return h0_k(kx, ky, v=v, t=t, w=w) + float(spin_sign) * h_soc_orbital(lm)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
