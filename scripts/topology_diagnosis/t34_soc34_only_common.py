#!/usr/bin/env python3
"""Common utilities for strict t34 + soc34-only physical model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import csv
import itertools

import numpy as np
from scipy.sparse import csr_matrix
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


def token2(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_dirs(base: Path) -> dict[str, Path]:
    root = ensure_dir(base)
    fine = ensure_dir(root)
    return {
        "base": fine,
        "01_bulk_gap": ensure_dir(fine / "01_bulk_gap"),
        "02_spin_chern_z2": ensure_dir(fine / "02_spin_chern_z2"),
        "03_ribbon": ensure_dir(fine / "03_ribbon"),
        "04_wilson_flow": ensure_dir(fine / "04_wilson_flow"),
        "05_open_open": ensure_dir(fine / "05_open_open"),
        "06_summary": ensure_dir(fine / "06_summary"),
    }


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat)
    return u @ vh


def h_ssh_orbital_k(kx: float, ky: float, v: float, w: float) -> np.ndarray:
    """Original SSH(v,w) part; unchanged."""
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


def h_soc34_orbital(lm: float) -> np.ndarray:
    """SOC only on 3-4 channel:
    [[0,0,0,0],[0,0,0,0],[0,0,0,-i lm],[0,0,i lm,0]]
    """
    h = np.zeros((4, 4), dtype=complex)
    h[2, 3] = -1j * lm
    h[3, 2] = 1j * lm
    return h


def h0_orbital_t34_soc34_k(kx: float, ky: float, v: float, t34: float, w: float) -> np.ndarray:
    return h_ssh_orbital_k(kx, ky, v=v, w=w) + h_t34_orbital(t34)


def h_spin_block_t34_soc34(
    kx: float, ky: float, v: float, params: ModelParams, spin_sign: int
) -> np.ndarray:
    return h0_orbital_t34_soc34_k(kx, ky, v=v, t34=params.t34, w=params.w) + float(
        spin_sign
    ) * h_soc34_orbital(params.lm)


def h8_t34_soc34_k(kx: float, ky: float, v: float, params: ModelParams) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0_orbital_t34_soc34_k(kx, ky, v=v, t34=params.t34, w=params.w), s0) + np.kron(
        h_soc34_orbital(params.lm),
        sz,
    )


def write_model_check(base: Path, params: ModelParams) -> Path:
    out = base / "t_soc_hopping_check_t34_soc34_only.txt"
    tmat = h_t34_orbital(params.t34)
    smat = h_soc34_orbital(params.lm)
    nz_t = [(i, j, tmat[i, j]) for i in range(4) for j in range(4) if abs(tmat[i, j]) > 1e-14]
    nz_s = [(i, j, smat[i, j]) for i in range(4) for j in range(4) if abs(smat[i, j]) > 1e-14]
    ok_t = {(i, j) for i, j, _ in nz_t} == {(2, 3), (3, 2)}
    ok_s = {(i, j) for i, j, _ in nz_s} == {(2, 3), (3, 2)}
    lines = [
        "strict t34 + soc34-only model check",
        "===================================",
        f"t34={params.t34}",
        f"lambda={params.lm}",
        "",
        "Nonzero t entries:",
        *[f"  H_t[{i},{j}]={v}" for i, j, v in nz_t],
        "",
        "Nonzero SOC entries:",
        *[f"  H_soc[{i},{j}]={v}" for i, j, v in nz_s],
        "",
        f"t34_only_ok={ok_t}",
        f"soc34_only_ok={ok_s}",
        "spin extension: H_t⊗s0 and H_soc34⊗sz (no spin-flip hopping/SOC)",
        "SSH(v,w) remains only on (1,4) and (2,3) channels.",
    ]
    status = ok_t and ok_s
    lines.append(f"CHECK_STATUS={'PASS' if status else 'FAIL'}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not status:
        raise RuntimeError("strict t34+soc34 check failed")
    return out


def compute_bulk_gap(v: float, params: ModelParams, nk: int = 101, n_occ: int = 4) -> tuple[float, float, float]:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    min_gap = float("inf")
    min_kx, min_ky = 0.0, 0.0
    for kx, ky in product(kgrid, kgrid):
        evals = np.real(np.linalg.eigvalsh(h8_t34_soc34_k(float(kx), float(ky), v=v, params=params)))
        gap = float(evals[n_occ] - evals[n_occ - 1])
        if gap < min_gap:
            min_gap = gap
            min_kx, min_ky = float(kx), float(ky)
    return float(min_gap), float(min_kx), float(min_ky)


def occ_subspace(kx: float, ky: float, v: float, params: ModelParams, n_occ: int) -> np.ndarray:
    _, vecs = np.linalg.eigh(h8_t34_soc34_k(kx, ky, v=v, params=params))
    return vecs[:, :n_occ]


def compute_wilson_centers_direction(
    v: float,
    params: ModelParams,
    n_occ: int,
    nkx: int,
    nky: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], float]:
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


def track_sector_frames(eigvec_occ: list[np.ndarray], n_occ: int) -> tuple[list[np.ndarray], float]:
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


def nested_polarization_from_tracked(tracked_sector: list[np.ndarray], occ_bases: list[np.ndarray]) -> float:
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


def build_onsite_xopen_full(ky: float, v: float, params: ModelParams) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 3] = v
    h0[3, 0] = v
    h0[1, 2] = v + params.w * np.exp(-1j * ky)
    h0[2, 1] = v + params.w * np.exp(1j * ky)
    h0 += h_t34_orbital(params.t34)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h_soc34_orbital(params.lm), sz)


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


def ribbon_spectrum_with_edge_weight(
    v: float, params: ModelParams, nx: int, nk: int, edge_cells: int = 3
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


def build_open_open_hamiltonian(
    lx: int, ly: int, v: float, params: ModelParams, termination: str = "A"
) -> np.ndarray:
    h = np.zeros((8 * lx * ly, 8 * lx * ly), dtype=complex)
    hsoc = h_soc34_orbital(params.lm)
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


def commutator_norm_h_sz(v: float, params: ModelParams, nk: int = 17) -> float:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    sz = np.kron(np.eye(4, dtype=complex), np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex))
    max_norm = 0.0
    for kx, ky in product(kgrid, kgrid):
        h = h8_t34_soc34_k(float(kx), float(ky), v=v, params=params)
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


def circular_distance(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 1.0 - d)


def track_wilson_branches(centers_sorted: np.ndarray, eigvecs_sorted: list[np.ndarray]) -> np.ndarray:
    nscan, nband = centers_sorted.shape
    tracked = np.zeros_like(centers_sorted)
    tracked[0, :] = centers_sorted[0, :]
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


def z2_crossing_raw(v: float, params: ModelParams, nkx: int = 101, nky: int = 101) -> tuple[int, tuple[int, int, int]]:
    ky_list = np.linspace(0.0, np.pi, nky)
    kx_grid = np.linspace(-np.pi, np.pi, nkx, endpoint=False)
    centers = np.zeros((nky, 4), dtype=float)
    eigvecs = []
    for j, ky in enumerate(ky_list):
        wmat = np.eye(4, dtype=complex)
        for i, kx in enumerate(kx_grid):
            kx2 = float(kx_grid[(i + 1) % nkx])
            _, va = np.linalg.eigh(h8_t34_soc34_k(float(kx), float(ky), v=v, params=params))
            _, vb = np.linalg.eigh(h8_t34_soc34_k(float(kx2), float(ky), v=v, params=params))
            oa = va[:, :4]
            ob = vb[:, :4]
            u, _, vh = np.linalg.svd(oa.conj().T @ ob)
            q = u @ vh
            wmat = q @ wmat
        ew, ev = np.linalg.eig(wmat)
        nu = np.mod(np.angle(ew) / (2.0 * np.pi), 1.0)
        order = np.argsort(nu)
        centers[j, :] = nu[order]
        eigvecs.append(ev[:, order])
    tracked = track_wilson_branches(centers, eigvecs)
    c45 = crossing_count(tracked, 0.45)
    c50 = crossing_count(tracked, 0.50)
    c55 = crossing_count(tracked, 0.55)
    return int(c50 % 2), (int(c45), int(c50), int(c55))
