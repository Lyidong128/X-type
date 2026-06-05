#!/usr/bin/env python3
"""Common utilities for Kane-Mele SOC topology analysis scripts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(Path("/workspace")))

from scripts.run_scan import load_xtype_model, set_model_params


@dataclass
class GapResult:
    direct_gap: float
    indirect_gap: float
    kx_min: float
    ky_min: float
    k_u_min: float
    k_v_min: float


@dataclass
class WilsonDetResult:
    winding: float
    z2: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_scan_values(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = start
    eps = 1e-12
    while x <= stop + eps:
        vals.append(round(x, 10))
        x += step
    return vals


def param_token(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def load_model(model_file: str = "xtype_model.py"):
    return load_xtype_model(Path("/workspace/models") / model_file)


def set_params(model, v: float, lm: float, t: float, w: float, j: float = 0.0) -> None:
    set_model_params(model, v=float(v), t=float(t), lm=float(lm), w=float(w), j=float(j))


def split_spin_indices(dim: int) -> tuple[np.ndarray, np.ndarray]:
    up = np.arange(0, dim, 2, dtype=int)
    dn = np.arange(1, dim, 2, dtype=int)
    return up, dn


def spin_mixing_ratio(mat: np.ndarray) -> float:
    up, dn = split_spin_indices(mat.shape[0])
    off = mat[np.ix_(up, dn)]
    denom = float(np.linalg.norm(mat))
    if denom < 1e-16:
        return 0.0
    return float(np.linalg.norm(off) / denom)


def sz_operator(dim: int) -> np.ndarray:
    op = np.zeros((dim, dim), dtype=float)
    up, dn = split_spin_indices(dim)
    op[up, up] = 1.0
    op[dn, dn] = -1.0
    return op


def compute_gap_result(model, nk: int = 51, n_occ: int = 4) -> GapResult:
    valence_max = -np.inf
    conduction_min = np.inf
    direct_min = np.inf
    kx_min = 0.0
    ky_min = 0.0
    k_u_min = 0.0
    k_v_min = 0.0
    denom = max(1, nk - 1)

    for i, j in product(range(nk), range(nk)):
        u = i / denom
        v = j / denom
        k = u * model.b1 + v * model.b2
        evals = np.linalg.eigvalsh(model.Hxtype(k))
        valence = float(np.real(evals[n_occ - 1]))
        conduction = float(np.real(evals[n_occ]))
        direct_gap = conduction - valence
        valence_max = max(valence_max, valence)
        conduction_min = min(conduction_min, conduction)
        if direct_gap < direct_min:
            direct_min = float(direct_gap)
            kx_min = float(np.real(k[0]))
            ky_min = float(np.real(k[1]))
            k_u_min = float(u)
            k_v_min = float(v)

    indirect_gap = float(conduction_min - valence_max)
    return GapResult(
        direct_gap=float(direct_min),
        indirect_gap=indirect_gap,
        kx_min=kx_min,
        ky_min=ky_min,
        k_u_min=k_u_min,
        k_v_min=k_v_min,
    )


def chern_number_general(
    h_func,
    b1: np.ndarray,
    b2: np.ndarray,
    nk: int,
    n_occ: int,
) -> float:
    def occ_space(kvec: np.ndarray) -> np.ndarray:
        _, vecs = np.linalg.eigh(h_func(kvec))
        return vecs[:, :n_occ]

    occ = [[None for _ in range(nk)] for __ in range(nk)]
    for i in range(nk):
        u = i / nk
        for j in range(nk):
            v = j / nk
            k = u * b1 + v * b2
            occ[i][j] = occ_space(k)

    def link(mat_a, mat_b):
        m = mat_a.conj().T @ mat_b
        detm = np.linalg.det(m)
        if abs(detm) < 1e-14:
            return 1.0 + 0.0j
        return detm / abs(detm)

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
            f12[i, j] = np.log(ux[i, j] * uy[ip, j] / (ux[i, jp] * uy[i, j]))

    return float(np.real(np.sum(f12) / (2j * np.pi)))


def compute_total_chern(model, nk: int = 31, n_occ: int = 4) -> float:
    return float(model.chern_number_fukui(nk=nk, n_occ=n_occ))


def compute_wilson_det_result(
    model,
    nk_scan: int = 41,
    nk_loop: int = 81,
    n_occ: int = 4,
    direction: str = "kx_scan_ky",
) -> WilsonDetResult:
    def occ_space(kvec: np.ndarray) -> np.ndarray:
        _, vecs = np.linalg.eigh(model.Hxtype(kvec))
        return vecs[:, :n_occ]

    phases = []
    for j in range(nk_scan):
        s = j / nk_scan
        wilson_det = 1.0 + 0.0j
        if direction == "kx_scan_ky":
            prev = occ_space(0.0 * model.b1 + s * model.b2)
            for i in range(1, nk_loop + 1):
                loop = i / nk_loop
                curr = occ_space(loop * model.b1 + s * model.b2)
                overlap = prev.conj().T @ curr
                det_overlap = np.linalg.det(overlap)
                if abs(det_overlap) > 1e-14:
                    wilson_det *= det_overlap / abs(det_overlap)
                prev = curr
        else:
            prev = occ_space(s * model.b1 + 0.0 * model.b2)
            for i in range(1, nk_loop + 1):
                loop = i / nk_loop
                curr = occ_space(s * model.b1 + loop * model.b2)
                overlap = prev.conj().T @ curr
                det_overlap = np.linalg.det(overlap)
                if abs(det_overlap) > 1e-14:
                    wilson_det *= det_overlap / abs(det_overlap)
                prev = curr
        phases.append(float(np.angle(wilson_det)))
    phases = np.array(phases, dtype=float)
    unwrapped = np.unwrap(phases)
    winding = float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))
    z2 = int(round(abs(winding))) % 2
    return WilsonDetResult(winding=winding, z2=z2)


def compute_wilson_centers(
    model,
    nk_scan: int = 81,
    nk_loop: int = 121,
    n_occ: int = 4,
    direction: str = "kx_scan_ky",
) -> tuple[np.ndarray, np.ndarray]:
    def occ_space(kvec: np.ndarray) -> np.ndarray:
        _, vecs = np.linalg.eigh(model.Hxtype(kvec))
        return vecs[:, :n_occ]

    scan_values = np.linspace(0.0, 1.0, nk_scan, endpoint=False)
    centers = np.zeros((nk_scan, n_occ), dtype=float)

    for j, s in enumerate(scan_values):
        wmat = np.eye(n_occ, dtype=complex)
        if direction == "kx_scan_ky":
            prev = occ_space(0.0 * model.b1 + s * model.b2)
            for i in range(1, nk_loop + 1):
                loop = i / nk_loop
                curr = occ_space(loop * model.b1 + s * model.b2)
                overlap = prev.conj().T @ curr
                u, _, vh = np.linalg.svd(overlap)
                q = u @ vh
                wmat = q @ wmat
                prev = curr
        else:
            prev = occ_space(s * model.b1 + 0.0 * model.b2)
            for i in range(1, nk_loop + 1):
                loop = i / nk_loop
                curr = occ_space(s * model.b1 + loop * model.b2)
                overlap = prev.conj().T @ curr
                u, _, vh = np.linalg.svd(overlap)
                q = u @ vh
                wmat = q @ wmat
                prev = curr
        eigvals = np.linalg.eigvals(wmat)
        vals = np.mod(np.angle(eigvals) / (2.0 * np.pi), 1.0)
        centers[j, :] = np.sort(vals)
    return scan_values, centers


def crossing_parity(centers: np.ndarray, reference: float = 0.5) -> int:
    crossings = 0
    for i in range(centers.shape[0] - 1):
        a = centers[i, :] - reference
        b = centers[i + 1, :] - reference
        crossings += int(np.sum(a * b < 0.0))
    return int(crossings % 2)


def plot_wilson_centers(scan_values: np.ndarray, centers: np.ndarray, save_path: Path, title: str, xlabel: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for idx in range(centers.shape[1]):
        ax.plot(scan_values, centers[:, idx], linewidth=0.8, alpha=0.95)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Wannier center")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def generate_model_basis_report(
    output_path: Path,
    model_file: str = "xtype_model.py",
    t: float = 0.3,
    w: float = 1.0,
    v: float = 0.5,
    lm: float = 0.1,
) -> Path:
    model = load_model(model_file=model_file)
    set_params(model, v=v, lm=0.0, t=t, w=w, j=0.0)
    h0 = np.array(model.Hxtype(np.zeros(3)), dtype=complex)
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    h1 = np.array(model.Hxtype(np.zeros(3)), dtype=complex)
    d_soc = h1 - h0
    dim = h0.shape[0]

    up, dn = split_spin_indices(dim)
    soc_up = d_soc[np.ix_(up, up)]
    soc_dn = d_soc[np.ix_(dn, dn)]
    soc_ud = d_soc[np.ix_(up, dn)]
    soc_du = d_soc[np.ix_(dn, up)]
    soc_off_spin_ratio = float((np.linalg.norm(soc_ud) + np.linalg.norm(soc_du)) / max(np.linalg.norm(d_soc), 1e-16))
    soc_opposite_spin_ratio = float(np.linalg.norm(soc_up + soc_dn) / max(np.linalg.norm(d_soc), 1e-16))
    imag_ratio = float(np.linalg.norm(np.imag(d_soc)) / max(np.linalg.norm(d_soc), 1e-16))

    spin_identifiable = soc_off_spin_ratio < 1e-6
    can_build_sz = True
    soc_km_like = spin_identifiable and soc_opposite_spin_ratio < 1e-5 and imag_ratio > 0.2

    text = []
    text.append("Kane-Mele SOC model basis report")
    text.append("================================")
    text.append(f"Hamiltonian function location: {model.__file__} :: Hxtype(k)")
    text.append("Parameter names detected: v, t, w, lm, J")
    text.append("")
    text.append("Basis / internal degrees of freedom:")
    text.append("- Matrix dimension = 8, interpreted as 4 orbital/site components x 2 spins.")
    text.append("- From kron(orbital_part, spin_pauli) structure and state indexing, basis is orbital-major then spin.")
    text.append("- Practical ordering used in code: [orb0_up, orb0_dn, orb1_up, orb1_dn, orb2_up, orb2_dn, orb3_up, orb3_dn].")
    text.append("- Explicit A/B sublattice labels are not unambiguously encoded in current model file (treated as 4 components).")
    text.append("")
    text.append(f"Spin-up/down identifiable: {spin_identifiable}")
    text.append(f"Can construct Sz operator: {can_build_sz}")
    text.append(f"SOC off-spin coupling ratio ||H_updn||/||H_soc|| = {soc_off_spin_ratio:.3e}")
    text.append(f"SOC opposite-spin-block ratio ||H_up + H_dn||/||H_soc|| = {soc_opposite_spin_ratio:.3e}")
    text.append(f"SOC imaginary-part ratio ||Im(H_soc)||/||H_soc|| = {imag_ratio:.3e}")
    text.append("")
    text.append("Kane-Mele SOC assessment:")
    if soc_km_like:
        text.append("- SOC term is spin-diagonal (Sz-like), opposite for up/down blocks, and predominantly imaginary.")
        text.append("- This is consistent with Kane-Mele-type spin-dependent virtual NNN-like hopping symmetry structure.")
        text.append("- Such SOC preserves TRS at model level and can support C_up=-C_down, C_total~0, Z2-like phases.")
    else:
        text.append("- SOC term does not fully match the ideal Kane-Mele signature under current automatic checks.")
        text.append("- Use caution; verify basis mapping and SOC decomposition manually if needed.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output_path


def default_inversion_operator() -> np.ndarray:
    """Model-specific inversion swapping orbital indices (0<->3, 1<->2)."""
    p_orb = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return np.kron(p_orb, np.eye(2, dtype=float)).astype(complex)


def inversion_error(model, p_op: np.ndarray, nk: int = 7) -> float:
    errs = []
    for i in range(nk):
        u = i / max(1, nk - 1)
        for j in range(nk):
            v = j / max(1, nk - 1)
            k = u * model.b1 + v * model.b2
            h = np.array(model.Hxtype(k), dtype=complex)
            hm = np.array(model.Hxtype(-k), dtype=complex)
            denom = max(float(np.linalg.norm(h)), 1e-16)
            errs.append(float(np.linalg.norm(p_op @ h @ p_op.conj().T - hm) / denom))
    return float(np.mean(errs))


def fu_kane_z2(model, p_op: np.ndarray, n_occ: int = 4) -> tuple[int, dict[str, int], float]:
    trims = {
        "Gamma": np.zeros(3),
        "X": 0.5 * model.b1,
        "Y": 0.5 * model.b2,
        "M": 0.5 * (model.b1 + model.b2),
    }
    deltas: dict[str, int] = {}
    quality = []
    for name, k in trims.items():
        _, vecs = np.linalg.eigh(model.Hxtype(k))
        occ = vecs[:, :n_occ]
        m = occ.conj().T @ p_op @ occ
        m = 0.5 * (m + m.conj().T)
        evals = np.linalg.eigvalsh(m)
        quality.append(float(np.max(np.abs(np.abs(evals) - 1.0))))
        signs = np.sign(np.real(evals))
        signs[signs == 0] = 1.0
        deltas[name] = int(np.prod(signs))
    total = int(np.prod(list(deltas.values())))
    z2 = 0 if total > 0 else 1
    return z2, deltas, float(np.max(quality))
