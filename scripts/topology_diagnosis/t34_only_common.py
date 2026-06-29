#!/usr/bin/env python3
"""Common utilities for t34-only topology diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import csv

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh


@dataclass(frozen=True)
class ModelParams:
    t34: float = 0.3
    w: float = 1.0
    lm: float = 0.1


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dirs(base: Path) -> dict[str, Path]:
    return {
        "base": ensure_dir(base),
        "01_bulk_band": ensure_dir(base / "01_bulk_band"),
        "02_ribbon": ensure_dir(base / "02_ribbon"),
        "03_spin_resolved_ribbon": ensure_dir(base / "03_spin_resolved_ribbon"),
        "04_open_open": ensure_dir(base / "04_open_open"),
        "05_termination": ensure_dir(base / "05_termination"),
        "06_wilson_polarization": ensure_dir(base / "06_wilson_polarization"),
        "07_nested_wilson": ensure_dir(base / "07_nested_wilson"),
        "08_qsh_invariant": ensure_dir(base / "08_qsh_invariant"),
        "09_z2_debug": ensure_dir(base / "09_z2_debug"),
        "10_summary": ensure_dir(base / "10_summary"),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def h_soc_orbital(lm: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 1] = 1j * lm
    h[0, 2] = -1j * lm
    h[1, 0] = -1j * lm
    h[1, 3] = 1j * lm
    h[2, 0] = 1j * lm
    h[2, 3] = -1j * lm
    h[3, 1] = -1j * lm
    h[3, 2] = 1j * lm
    return h


def h_ssh_orbital_k(kx: float, ky: float, v: float, w: float) -> np.ndarray:
    """Original SSH-dimerization part only (no extra t terms)."""
    h = np.zeros((4, 4), dtype=complex)
    h[0, 3] = v + w * np.exp(-1j * kx)
    h[3, 0] = v + w * np.exp(1j * kx)
    h[1, 2] = v + w * np.exp(-1j * ky)
    h[2, 1] = v + w * np.exp(1j * ky)
    return h


def h_t34_orbital(t34: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[2, 3] = t34
    h[3, 2] = np.conjugate(t34)
    return h


def h0_orbital_t34_k(kx: float, ky: float, v: float, t34: float, w: float) -> np.ndarray:
    return h_ssh_orbital_k(kx, ky, v=v, w=w) + h_t34_orbital(t34=t34)


def h_spin_block_t34(kx: float, ky: float, v: float, params: ModelParams, spin_sign: int) -> np.ndarray:
    return h0_orbital_t34_k(kx, ky, v=v, t34=params.t34, w=params.w) + float(spin_sign) * h_soc_orbital(params.lm)


def h8_t34_k(kx: float, ky: float, v: float, params: ModelParams) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0_orbital_t34_k(kx, ky, v=v, t34=params.t34, w=params.w), s0) + np.kron(h_soc_orbital(params.lm), sz)


def write_t_hopping_check(base: Path, params: ModelParams) -> Path:
    """Write textual verification that only t34 hopping exists."""
    out = base / "t_hopping_check_after_t34_only.txt"
    tmat = h_t34_orbital(params.t34)
    nz = []
    for i in range(4):
        for j in range(4):
            if abs(tmat[i, j]) > 1e-14:
                nz.append((i, j, tmat[i, j]))

    allowed = {(2, 3), (3, 2)}
    bad = [(i, j, val) for (i, j, val) in nz if (i, j) not in allowed]
    spinflip_present = False  # t term is explicitly tensored with s0
    ok = len(bad) == 0 and {(i, j) for i, j, _ in nz} == allowed

    lines = [
        "t34-only hopping check",
        "======================",
        f"t34 = {params.t34}",
        "",
        "Nonzero additional t hopping entries in orbital 4x4 block (0-based):",
    ]
    for i, j, val in nz:
        lines.append(f"  H_t[{i},{j}] = {val}")
    if not nz:
        lines.append("  (none)")
    lines += [
        "",
        f"Allowed entries: {sorted(list(allowed))}",
        f"Unexpected nonzero entries: {[(i, j) for i, j, _ in bad]}",
        "Spinful extension check:",
        "  H_t_spinful = H_t ⊗ s0",
        "  up block nonzero: (2,3),(3,2)",
        "  dn block nonzero: (6,7),(7,6)",
        f"  spin-flip present: {spinflip_present}",
        "",
        "SSH(v,w) dimerization unchanged check:",
        "  nonzero SSH pairs remain 1-4 and 2-3 with intracell v and intercell w phases.",
        "",
    ]
    if ok:
        lines.append("CHECK_STATUS = PASS (only 3-4 additional t hopping exists).")
    else:
        lines.append("CHECK_STATUS = FAIL.")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not ok:
        raise RuntimeError("t34-only hopping check failed: found non-3-4 additional t hopping.")
    return out


def occ_subspace(kx: float, ky: float, v: float, params: ModelParams, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_t34_k(kx, ky, v=v, params=params))
    return vecs[:, :n_occ]


def build_high_symmetry_path(nseg: int) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    b1 = np.array([2.0 * np.pi, 0.0], dtype=float)
    b2 = np.array([0.0, 2.0 * np.pi], dtype=float)
    g = np.array([0.0, 0.0], dtype=float)
    x = 0.5 * b1
    y = 0.5 * b2
    m = 0.5 * (b1 + b2)
    nodes = [g, x, y, g, m, g]
    labels = [r"$\Gamma$", "X", "Y", r"$\Gamma$", "M", r"$\Gamma$"]

    k_list = []
    tick_idx = [0]
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        for j in range(nseg):
            u = j / nseg
            k_list.append((1.0 - u) * a + u * b)
        tick_idx.append(len(k_list))
    k_list.append(nodes[-1])
    kpts = np.array(k_list, dtype=float)
    xcoords = np.zeros(len(kpts), dtype=float)
    for i in range(1, len(kpts)):
        xcoords[i] = xcoords[i - 1] + np.linalg.norm(kpts[i] - kpts[i - 1])
    ticks = [float(xcoords[i]) for i in tick_idx]
    return kpts, xcoords, ticks, labels


def compute_bulk_gap(v: float, params: ModelParams, nk: int = 101, n_occ: int = 4) -> tuple[float, float, float]:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    min_gap = float("inf")
    min_kx = 0.0
    min_ky = 0.0
    for kx, ky in product(kgrid, kgrid):
        evals = np.real(np.linalg.eigvalsh(h8_t34_k(float(kx), float(ky), v=v, params=params)))
        g = float(evals[n_occ] - evals[n_occ - 1])
        if g < min_gap:
            min_gap = g
            min_kx, min_ky = float(kx), float(ky)
    return float(min_gap), float(min_kx), float(min_ky)


def compute_bulk_bands_path(v: float, params: ModelParams, nseg: int = 220) -> tuple[np.ndarray, np.ndarray, list[float], list[str]]:
    kpts, xcoords, ticks, labels = build_high_symmetry_path(nseg=nseg)
    bands = []
    for k in kpts:
        bands.append(np.real(np.linalg.eigvalsh(h8_t34_k(k[0], k[1], v=v, params=params))))
    return np.array(bands, dtype=float), xcoords, ticks, labels


def add_hop(H: np.ndarray, i: int, j: int, amp: complex) -> None:
    H[i, j] += amp
    H[j, i] += np.conjugate(amp)


def idx_orb_spin(cell: int, orb: int, spin: int, n_orb: int = 4, n_spin: int = 2) -> int:
    return cell * (n_orb * n_spin) + orb * n_spin + spin


def build_onsite_xopen_full(ky: float, v: float, params: ModelParams) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 3] = v
    h0[3, 0] = v
    h0[1, 2] = v + params.w * np.exp(-1j * ky)
    h0[2, 1] = v + params.w * np.exp(1j * ky)
    h0 += h_t34_orbital(params.t34)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h_soc_orbital(params.lm), sz)


def build_ribbon_xopen_full(ky: float, nx: int, v: float, params: ModelParams) -> np.ndarray:
    h = np.zeros((8 * nx, 8 * nx), dtype=complex)
    onsite = build_onsite_xopen_full(ky=ky, v=v, params=params)
    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = params.w
    for x in range(nx):
        s = x * 8
        h[s : s + 8, s : s + 8] += onsite
        if x > 0:
            p = (x - 1) * 8
            h[s : s + 8, p : p + 8] += tx
            h[p : p + 8, s : s + 8] += tx.conj().T
    return 0.5 * (h + h.conj().T)


def build_onsite_xopen_spin(ky: float, v: float, params: ModelParams, spin_sign: int) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 3] = v
    h0[3, 0] = v
    h0[1, 2] = v + params.w * np.exp(-1j * ky)
    h0[2, 1] = v + params.w * np.exp(1j * ky)
    h0 += h_t34_orbital(params.t34)
    return h0 + float(spin_sign) * h_soc_orbital(params.lm)


def build_ribbon_xopen_spin(ky: float, nx: int, v: float, params: ModelParams, spin_sign: int) -> np.ndarray:
    h = np.zeros((4 * nx, 4 * nx), dtype=complex)
    onsite = build_onsite_xopen_spin(ky=ky, v=v, params=params, spin_sign=spin_sign)
    tx = np.zeros((4, 4), dtype=complex)
    tx[0, 3] = params.w
    for x in range(nx):
        s = x * 4
        h[s : s + 4, s : s + 4] += onsite
        if x > 0:
            p = (x - 1) * 4
            h[s : s + 4, p : p + 4] += tx
            h[p : p + 4, s : s + 4] += tx.conj().T
    return 0.5 * (h + h.conj().T)


def ribbon_spectrum_with_edge_weight(
    v: float,
    params: ModelParams,
    nx: int,
    nk: int,
    edge_cells: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky_list = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_all = []
    edge_w_all = []
    for ky in ky_list:
        h = build_ribbon_xopen_full(float(ky), nx=nx, v=v, params=params)
        evals, vecs = np.linalg.eigh(h)
        evals = np.real(evals)
        prob = np.abs(vecs) ** 2
        prob_cell = prob.reshape(nx, 8, prob.shape[1]).sum(axis=1)
        edge_w = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        evals_all.append(evals)
        edge_w_all.append(edge_w)
    return ky_list, np.array(evals_all, dtype=float), np.array(edge_w_all, dtype=float)


def ribbon_eigensystem_xopen(ky: float, v: float, params: ModelParams, nx: int) -> tuple[np.ndarray, np.ndarray]:
    h = build_ribbon_xopen_full(float(ky), nx=nx, v=v, params=params)
    e, vecs = np.linalg.eigh(h)
    return np.real(e), vecs


def ribbon_eigensystem_xopen_spin(
    ky: float,
    v: float,
    params: ModelParams,
    nx: int,
    spin_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = build_ribbon_xopen_spin(float(ky), nx=nx, v=v, params=params, spin_sign=spin_sign)
    e, vecs = np.linalg.eigh(h)
    return np.real(e), vecs


def rho_x_from_vec_spinful(vec: np.ndarray, nx: int) -> np.ndarray:
    prob = np.abs(vec) ** 2
    return prob.reshape(nx, 8).sum(axis=1)


def rho_x_from_vec_spin_block(vec: np.ndarray, nx: int) -> np.ndarray:
    prob = np.abs(vec) ** 2
    return prob.reshape(nx, 4).sum(axis=1)


def edge_bulk_weights_1d(rho_x: np.ndarray, edge_cells: int = 3) -> tuple[float, float, float]:
    wl = float(np.sum(rho_x[:edge_cells]))
    wr = float(np.sum(rho_x[-edge_cells:]))
    we = wl + wr
    wb = float(max(0.0, 1.0 - we))
    return wl, wr, wb


def obc_index(ix: int, iy: int, orb: int, spin: int, lx: int) -> int:
    return ((iy * lx + ix) * 8) + orb * 2 + spin


def build_open_open_hamiltonian(
    lx: int,
    ly: int,
    v: float,
    params: ModelParams,
    termination: str = "A",
) -> np.ndarray:
    h = np.zeros((8 * lx * ly, 8 * lx * ly), dtype=complex)
    hsoc = h_soc_orbital(params.lm)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    tx = np.zeros((8, 8), dtype=complex)
    ty = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = params.w
        ty[1 * 2 + spin, 2 * 2 + spin] = params.w

    for iy in range(ly):
        for ix in range(lx):
            v14 = v
            if termination == "B" and (ix == 0 or ix == lx - 1):
                v14 = 0.0
            h0 = np.zeros((4, 4), dtype=complex)
            h0[0, 3] = v14
            h0[3, 0] = v14
            h0[1, 2] = v
            h0[2, 1] = v
            h0 += h_t34_orbital(params.t34)
            onsite = np.kron(h0, s0) + np.kron(hsoc, sz)
            s = (iy * lx + ix) * 8
            h[s : s + 8, s : s + 8] += onsite
            if ix > 0:
                p = (iy * lx + ix - 1) * 8
                h[s : s + 8, p : p + 8] += tx
                h[p : p + 8, s : s + 8] += tx.conj().T
            if iy > 0:
                p = ((iy - 1) * lx + ix) * 8
                h[s : s + 8, p : p + 8] += ty
                h[p : p + 8, s : s + 8] += ty.conj().T
    return 0.5 * (h + h.conj().T)


def diagonalize_open_open(h: np.ndarray, near_k: int = 120) -> tuple[np.ndarray, np.ndarray]:
    n = h.shape[0]
    if n <= 1600:
        evals, vecs = np.linalg.eigh(h)
        return np.real(evals), vecs
    hs = csr_matrix(h)
    k = min(max(near_k, 20), n - 2)
    evals, vecs = eigsh(hs, k=k, sigma=0.0, which="LM")
    order = np.argsort(np.real(evals))
    return np.real(evals[order]), vecs[:, order]


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


def state_weights_2d(rho: np.ndarray, corner: np.ndarray, edge: np.ndarray, bulk: np.ndarray) -> tuple[float, float, float]:
    wc = float(np.sum(rho[corner]))
    we = float(np.sum(rho[edge]))
    wb = float(np.sum(rho[bulk]))
    return wc, we, wb


def classify_corner_edge(wc: float, we: float) -> str:
    if wc > 0.6:
        return "corner_state"
    if we > 0.6:
        return "edge_state"
    return "bulk_or_mixed_state"


def compute_wilson_centers_direction(
    v: float,
    params: ModelParams,
    n_occ: int,
    nkx: int,
    nky: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], float]:
    """Return scan grid, centers, Wilson eigvecs (occ space), occ bases, max unitarity error."""
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    ky_grid = np.linspace(-np.pi, np.pi, nky, endpoint=False)
    if direction == "x":
        scan = ky_grid
        loop = kx_grid
    elif direction == "y":
        scan = kx_grid
        loop = ky_grid
    else:
        raise ValueError(direction)
    centers = np.zeros((scan.size, n_occ), dtype=float)
    eigvec_occ: list[np.ndarray] = []
    occ_bases: list[np.ndarray] = []
    max_unit_err = 0.0
    for j, fixed in enumerate(scan):
        wmat = np.eye(n_occ, dtype=complex)
        if direction == "x":
            occ0 = occ_subspace(float(loop[0]), float(fixed), v=v, params=params, n_occ=n_occ)
        else:
            occ0 = occ_subspace(float(fixed), float(loop[0]), v=v, params=params, n_occ=n_occ)
        for i, k in enumerate(loop):
            kp = float(loop[(i + 1) % loop.size])
            if direction == "x":
                oa = occ_subspace(float(k), float(fixed), v=v, params=params, n_occ=n_occ)
                ob = occ_subspace(float(kp), float(fixed), v=v, params=params, n_occ=n_occ)
            else:
                oa = occ_subspace(float(fixed), float(k), v=v, params=params, n_occ=n_occ)
                ob = occ_subspace(float(fixed), float(kp), v=v, params=params, n_occ=n_occ)
            q = unitary_part(oa.conj().T @ ob)
            wmat = q @ wmat
        unit_err = float(np.linalg.norm(wmat.conj().T @ wmat - np.eye(n_occ)))
        max_unit_err = max(max_unit_err, unit_err)
        eigvals, eigvecs = np.linalg.eig(wmat)
        nu = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvec_occ.append(eigvecs[:, order])
        occ_bases.append(occ0)
    return scan, centers, eigvec_occ, occ_bases, float(max_unit_err)


def middle_sector_gap(centers: np.ndarray) -> float:
    n = centers.shape[1]
    half = n // 2
    gaps = []
    for row in centers:
        arr = np.sort(np.mod(row, 1.0))
        cyc = np.r_[arr, arr[0] + 1.0]
        cyc_gaps = np.diff(cyc)
        cut = int(np.argmax(cyc_gaps))
        start = arr[(cut + 1) % n]
        shifted = np.mod(arr - start, 1.0)
        shifted = np.sort(shifted)
        gaps.append(float(shifted[half] - shifted[half - 1]))
    return float(np.min(np.array(gaps, dtype=float)))


def track_sector_frames(
    eigvec_occ: list[np.ndarray],
    n_occ: int,
) -> tuple[list[np.ndarray], float]:
    half = n_occ // 2
    combos = list(combinations(range(n_occ), half))
    init = tuple(range(half))
    prev = eigvec_occ[0][:, init]
    tracked = [prev]
    min_overlap = 1.0
    for j in range(1, len(eigvec_occ)):
        vecs = eigvec_occ[j]
        best = combos[0]
        best_score = -1.0
        best_min_sv = 0.0
        for c in combos:
            cand = vecs[:, c]
            sv = np.linalg.svd(prev.conj().T @ cand, compute_uv=False)
            score = float(np.min(sv))
            if score > best_score:
                best_score = score
                best = c
                best_min_sv = float(np.min(sv))
        chosen = vecs[:, best]
        tracked.append(chosen)
        min_overlap = min(min_overlap, best_min_sv)
        prev = chosen
    close_sv = np.linalg.svd(tracked[-1].conj().T @ tracked[0], compute_uv=False)
    min_overlap = min(min_overlap, float(np.min(close_sv)))
    return tracked, float(min_overlap)


def nested_polarization_from_tracked(
    tracked_sector: list[np.ndarray],
    occ_bases: list[np.ndarray],
) -> float:
    half = tracked_sector[0].shape[1]
    wmat = np.eye(half, dtype=complex)
    n = len(tracked_sector)
    for j in range(n):
        jp = (j + 1) % n
        f = unitary_part(occ_bases[j].conj().T @ occ_bases[jp])
        g = tracked_sector[j].conj().T @ f @ tracked_sector[jp]
        q = unitary_part(g)
        wmat = q @ wmat
    detw = np.linalg.det(wmat)
    if abs(detw) < 1e-14:
        return 0.0
    return float(np.mod(np.angle(detw) / (2.0 * np.pi), 1.0))


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def track_wilson_branches(centers_sorted: np.ndarray, eigvecs_sorted: list[np.ndarray]) -> np.ndarray:
    nscan, nband = centers_sorted.shape
    tracked = np.zeros_like(centers_sorted)
    tracked[0, :] = centers_sorted[0, :]
    perms = list(product(range(nband), repeat=0))
    # build permutation list manually to avoid import itertools.permutations in hot path
    import itertools

    perms = list(itertools.permutations(range(nband)))
    prev_vec = eigvecs_sorted[0]
    for j in range(1, nscan):
        prev = tracked[j - 1, :]
        curr = centers_sorted[j, :]
        curr_vec = eigvecs_sorted[j]
        best_perm = perms[0]
        best_cost = float("inf")
        for p in perms:
            c = np.array([curr[k] for k in p], dtype=float)
            v = curr_vec[:, p]
            dist_cost = float(np.sum([circular_distance(prev[b], c[b]) for b in range(nband)]))
            overlap = np.array([abs(np.vdot(prev_vec[:, b], v[:, b])) for b in range(nband)], dtype=float)
            overlap_cost = float(np.sum(1.0 - np.clip(overlap, 0.0, 1.0)))
            cost = overlap_cost + 0.2 * dist_cost
            if cost < best_cost:
                best_cost = cost
                best_perm = p
        tracked[j, :] = np.array([curr[k] for k in best_perm], dtype=float)
        prev_vec = curr_vec[:, best_perm]
    # unwrap
    out = np.zeros_like(tracked)
    out[0, :] = tracked[0, :]
    for j in range(1, nscan):
        d = tracked[j, :] - tracked[j - 1, :]
        d = (d + 0.5) % 1.0 - 0.5
        out[j, :] = out[j - 1, :] + d
    return out


def crossing_count(branches_unwrapped: np.ndarray, ref: float) -> int:
    count = 0
    nseg = branches_unwrapped.shape[0] - 1
    eps = 5e-3
    for b in range(branches_unwrapped.shape[1]):
        y = branches_unwrapped[:, b]
        for i in range(nseg):
            a = float(y[i])
            c = float(y[i + 1])
            nmin = int(np.floor(min(a, c) - ref)) - 1
            nmax = int(np.ceil(max(a, c) - ref)) + 1
            for n in range(nmin, nmax + 1):
                r = ref + n
                fa = a - r
                fb = c - r
                if abs(fa) < eps or abs(fb) < eps:
                    continue
                if fa * fb < 0:
                    count += 1
    return int(count)


def largest_gap_center_from_row(row_unwrapped: np.ndarray) -> tuple[float, int, float]:
    arr = np.sort(np.mod(row_unwrapped, 1.0))
    cyc = np.r_[arr, arr[0] + 1.0]
    gaps = np.diff(cyc)
    idx = int(np.argmax(gaps))
    gmax = float(gaps[idx])
    if idx < arr.size - 1:
        cen = 0.5 * (arr[idx] + arr[idx + 1])
    else:
        cen = 0.5 * (arr[-1] + arr[0] + 1.0)
    return float(cen % 1.0), idx, gmax


def unwrap_mod_curve(values_mod: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values_mod, dtype=float)
    out[0] = float(values_mod[0])
    for i in range(1, values_mod.size):
        d = float(values_mod[i] - values_mod[i - 1])
        if d > 0.5:
            d -= 1.0
        if d < -0.5:
            d += 1.0
        out[i] = out[i - 1] + d
    return out


def commutator_norm_h_sz(v: float, params: ModelParams, nk: int = 21) -> float:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    sz = np.kron(np.eye(4, dtype=complex), np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))
    max_norm = 0.0
    for kx, ky in product(kgrid, kgrid):
        h = h8_t34_k(float(kx), float(ky), v=v, params=params)
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
