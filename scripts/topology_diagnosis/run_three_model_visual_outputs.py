#!/usr/bin/env python3
"""Generate comparable visual outputs for models A/B/C."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams as T34Params,
    build_high_symmetry_path,
    build_open_open_hamiltonian as build_open_open_b,
    build_ribbon_xopen_full as build_ribbon_b,
    compute_bulk_gap as compute_bulk_gap_b,
    ensure_dir,
    h8_t34_k,
    region_masks,
    state_weights_2d,
    token,
    write_csv,
)
from scripts.topology_diagnosis.t34_soc34_only_common import (  # noqa: E402
    ModelParams as T34Soc34Params,
    build_open_open_hamiltonian as build_open_open_c,
    build_ribbon_xopen_full as build_ribbon_c,
    h8_t34_soc34_k,
)


def h_soc_full_orbital(lm: float) -> np.ndarray:
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


def h0_full_t_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v + w * np.exp(-1j * kx)
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = v + w * np.exp(1j * kx)
    h0[3, 1] = t
    h0[3, 2] = t
    return h0


def h8_full_t_full_soc(kx: float, ky: float, v: float, t: float, w: float, lm: float) -> np.ndarray:
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return np.kron(h0_full_t_k(kx, ky, v=v, t=t, w=w), s0) + np.kron(h_soc_full_orbital(lm), sz)


def compute_bulk_gap_from_h8(v: float, h8_func, nk: int, n_occ: int = 4) -> tuple[float, float, float]:
    kgrid = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    min_gap = float("inf")
    min_kx, min_ky = 0.0, 0.0
    for kx in kgrid:
        for ky in kgrid:
            evals = np.real(np.linalg.eigvalsh(h8_func(float(kx), float(ky), v)))
            gap = float(evals[n_occ] - evals[n_occ - 1])
            if gap < min_gap:
                min_gap = gap
                min_kx, min_ky = float(kx), float(ky)
    return float(min_gap), float(min_kx), float(min_ky)


def build_ribbon_a(ky: float, nx: int, v: float, t: float, w: float, lm: float) -> np.ndarray:
    h = np.zeros((8 * nx, 8 * nx), dtype=complex)
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = v
    h0[3, 1] = t
    h0[3, 2] = t
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    onsite = np.kron(h0, s0) + np.kron(h_soc_full_orbital(lm), sz)
    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = w
    for x in range(nx):
        s = x * 8
        h[s : s + 8, s : s + 8] += onsite
        if x > 0:
            p = (x - 1) * 8
            h[s : s + 8, p : p + 8] += tx
            h[p : p + 8, s : s + 8] += tx.conj().T
    return 0.5 * (h + h.conj().T)


def build_open_open_a(lx: int, ly: int, v: float, t: float, w: float, lm: float) -> np.ndarray:
    h = np.zeros((8 * lx * ly, 8 * lx * ly), dtype=complex)
    hsoc = h_soc_full_orbital(lm)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    tx = np.zeros((8, 8), dtype=complex)
    ty = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = w
        ty[1 * 2 + spin, 2 * 2 + spin] = w

    for iy in range(ly):
        for ix in range(lx):
            h0 = np.zeros((4, 4), dtype=complex)
            h0[0, 1] = t
            h0[0, 2] = t
            h0[0, 3] = v
            h0[1, 0] = t
            h0[1, 2] = v
            h0[1, 3] = t
            h0[2, 0] = t
            h0[2, 1] = v
            h0[2, 3] = t
            h0[3, 0] = v
            h0[3, 1] = t
            h0[3, 2] = t
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


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str


def add_uniform_fm_term(h: np.ndarray, fm_out: float, fm_in: float) -> np.ndarray:
    if abs(fm_out) < 1e-15 and abs(fm_in) < 1e-15:
        return h
    n = h.shape[0]
    if n % 8 != 0:
        raise ValueError(f"Hamiltonian size {n} is not divisible by 8.")
    n_cells = n // 8
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    fm_spin = float(fm_out) * sz + float(fm_in) * sx
    fm8 = np.kron(np.eye(4, dtype=complex), fm_spin)
    fm = np.kron(np.eye(n_cells, dtype=complex), fm8)
    return 0.5 * ((h + fm) + (h + fm).conj().T)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate A/B/C visuals (bulk/ribbon/OBC-mark/WF-sum).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_three_models_visual")
    p.add_argument("--v-list", default="0.5,0.8,1.0")
    p.add_argument("--t", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    p.add_argument("--fm-out", type=float, default=0.0, help="Out-of-plane FM exchange (Mz * sz).")
    p.add_argument("--fm-in", type=float, default=0.0, help="In-plane FM exchange (Mx * sx).")
    p.add_argument("--bulk-nseg", type=int, default=180)
    p.add_argument("--bulk-gap-nk", type=int, default=81)
    p.add_argument("--ribbon-nx", type=int, default=40)
    p.add_argument("--ribbon-nk", type=int, default=121)
    p.add_argument("--edge-cells", type=int, default=3)
    p.add_argument("--lx", type=int, default=10)
    p.add_argument("--ly", type=int, default=10)
    p.add_argument("--near-k", type=int, default=120)
    p.add_argument("--corner-size", type=int, default=2)
    p.add_argument("--edge-width", type=int, default=2)
    p.add_argument("--energy-window-floor", type=float, default=0.02)
    p.add_argument("--window-gap-factor", type=float, default=0.32)
    p.add_argument("--energy-window-max", type=float, default=0.20)
    p.add_argument("--window-jump-ratio", type=float, default=10.0)
    p.add_argument("--max-states-per-v", type=int, default=48)
    return p.parse_args()


def parse_v_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def select_near_zero_states(
    evals: np.ndarray,
    bulk_gap: float,
    floor: float,
    gap_factor: float,
    window_max: float,
    max_states: int,
    jump_ratio: float,
) -> tuple[np.ndarray, float, float, float, str]:
    abs_e = np.abs(np.asarray(evals, dtype=float))
    order = np.argsort(abs_e)
    abs_sorted = abs_e[order]
    if abs_sorted.size == 0:
        return np.array([], dtype=int), float(floor), float(floor), float(floor), "empty"

    raw_window = max(float(floor), float(gap_factor) * max(float(bulk_gap), 1e-6))
    cap_window = max(float(floor), min(raw_window, float(window_max)))
    nin = int(np.searchsorted(abs_sorted, cap_window, side="right"))
    fallback_used = nin == 0
    mode = "window"
    if fallback_used:
        # No genuine near-zero state under current window: keep two closest for a stable visualization.
        nin = min(2, abs_sorted.size)
        mode = "fallback_two_closest"
    nin = min(max(2, nin), int(max_states), abs_sorted.size)

    scan_n = min(abs_sorted.size, max(12, int(max_states) * 2))
    if (not fallback_used) and scan_n >= 6:
        diffs = np.diff(abs_sorted[:scan_n])
        if diffs.size > 0:
            ref = float(np.median(diffs[: max(3, min(12, diffs.size))]))
            if ref > 1e-12:
                jump_idx = np.where(diffs > float(jump_ratio) * ref)[0]
                if jump_idx.size > 0:
                    jump_cut = int(jump_idx[0] + 1)
                    if jump_cut >= 2:
                        nin = min(nin, jump_cut)
                        mode = "window_with_jump_cut"

    idxs = np.sort(order[:nin].astype(int))
    eff_window = cap_window if mode == "fallback_two_closest" else float(abs_sorted[nin - 1])
    eff_window = max(float(floor), float(eff_window))
    return idxs, float(raw_window), float(cap_window), float(eff_window), mode


def spin_up_weight(vec: np.ndarray) -> float:
    prob = np.abs(vec) ** 2
    total = float(np.sum(prob))
    if total < 1e-15:
        return 0.5
    return float(np.sum(prob[0::2]) / total)


def blend_spin_rgb(w_up: float) -> tuple[float, float, float]:
    w_up = float(min(max(w_up, 0.0), 1.0))
    w_dn = 1.0 - w_up
    up_rgb = (0.839, 0.153, 0.157)  # red
    dn_rgb = (0.122, 0.467, 0.706)  # blue
    return tuple(w_up * up_rgb[i] + w_dn * dn_rgb[i] for i in range(3))


def track_bulk_bands_spin(
    evals_list: list[np.ndarray],
    evecs_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    n_k = len(evals_list)
    n_b = int(evals_list[0].size)
    tracked_e = np.zeros((n_k, n_b), dtype=float)
    tracked_wup = np.zeros((n_k, n_b), dtype=float)

    order = list(range(n_b))
    tracked_e[0] = evals_list[0][order]
    for b, j in enumerate(order):
        tracked_wup[0, b] = spin_up_weight(evecs_list[0][:, j])
    prev_vecs = evecs_list[0][:, order]

    for ik in range(1, n_k):
        overlaps = np.abs(prev_vecs.conj().T @ evecs_list[ik])
        used: set[int] = set()
        order = []
        for b in range(n_b):
            j_best = max((j for j in range(n_b) if j not in used), key=lambda j: overlaps[b, j])
            order.append(j_best)
            used.add(j_best)
        for b, j in enumerate(order):
            tracked_e[ik, b] = float(evals_list[ik][j])
            tracked_wup[ik, b] = spin_up_weight(evecs_list[ik][:, j])
        prev_vecs = evecs_list[ik][:, order]
    return tracked_e, tracked_wup


def plot_bulk_band(out_png: Path, v: float, title: str, h8_func, nseg: int) -> None:
    kpts, xcoords, ticks, labels = build_high_symmetry_path(nseg=nseg)
    evals_list: list[np.ndarray] = []
    evecs_list: list[np.ndarray] = []
    for k in kpts:
        evals, evecs = np.linalg.eigh(h8_func(float(k[0]), float(k[1]), v))
        evals_list.append(np.real(evals))
        evecs_list.append(evecs)
    bands, spin_up = track_bulk_bands_spin(evals_list=evals_list, evecs_list=evecs_list)

    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    n_k, n_b = bands.shape
    for b in range(n_b):
        for ik in range(n_k - 1):
            w_up = 0.5 * (spin_up[ik, b] + spin_up[ik + 1, b])
            ax.plot(
                xcoords[ik : ik + 2],
                bands[ik : ik + 2, b],
                color=blend_spin_rgb(w_up),
                linewidth=1.0,
            )
    for x in ticks:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"{title} | bulk band (spin-colored) | v={v:.1f}")
    ax.grid(alpha=0.2)
    legend_handles = [
        Line2D([0], [0], color=blend_spin_rgb(1.0), lw=2.0, label="spin-up dominant"),
        Line2D([0], [0], color=blend_spin_rgb(0.0), lw=2.0, label="spin-down dominant"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ribbon(out_png: Path, v: float, title: str, ribbon_builder, nx: int, nk: int, edge_cells: int) -> None:
    ky_vals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    evals_all = []
    edge_w_all = []
    for ky in ky_vals:
        h = ribbon_builder(float(ky), nx, v)
        evals, vecs = np.linalg.eigh(h)
        evals = np.real(evals)
        prob = np.abs(vecs) ** 2
        prob_cell = prob.reshape(nx, 8, prob.shape[1]).sum(axis=1)
        ew = prob_cell[:edge_cells, :].sum(axis=0) + prob_cell[-edge_cells:, :].sum(axis=0)
        evals_all.append(evals)
        edge_w_all.append(ew)
    evals_all = np.array(evals_all, dtype=float)
    edge_w_all = np.array(edge_w_all, dtype=float)
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    x = np.repeat(ky_vals / np.pi, evals_all.shape[1])
    y = evals_all.reshape(-1)
    c = edge_w_all.reshape(-1)
    sc = ax.scatter(x, y, c=c, s=4.5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(f"{title} | ribbon x-open | v={v:.1f}")
    ax.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax, label="edge weight")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_one_model(
    base: Path,
    spec: ModelSpec,
    v_list: list[float],
    args: argparse.Namespace,
    h8_func,
    bulk_gap_func,
    ribbon_builder,
    open_open_builder,
) -> None:
    out = ensure_dir(base / spec.key)
    out_bulk = ensure_dir(out / "01_bulk_band")
    out_ribbon = ensure_dir(out / "02_ribbon")
    out_obc = ensure_dir(out / "03_obc_marked")
    out_wf = ensure_dir(out / "04_wf_sum")

    corner, edge, bulk = region_masks(lx=args.lx, ly=args.ly, corner_size=args.corner_size, edge_width=args.edge_width)
    summary_rows = []
    marked_rows = []

    for v in v_list:
        plot_bulk_band(
            out_png=out_bulk / f"bulk_band_v{token(v)}_{spec.key}.png",
            v=v,
            title=f"{spec.title} | FM(out={args.fm_out:.3f}, in={args.fm_in:.3f})",
            h8_func=h8_func,
            nseg=args.bulk_nseg,
        )
        plot_ribbon(
            out_png=out_ribbon / f"ribbon_v{token(v)}_{spec.key}.png",
            v=v,
            title=f"{spec.title} | FM(out={args.fm_out:.3f}, in={args.fm_in:.3f})",
            ribbon_builder=ribbon_builder,
            nx=args.ribbon_nx,
            nk=args.ribbon_nk,
            edge_cells=args.edge_cells,
        )

        bulk_gap, _, _ = bulk_gap_func(v)
        h_obc = open_open_builder(v)
        evals, vecs = diagonalize_open_open(h_obc, near_k=args.near_k)
        idxs, ewin_raw, ewin_cap, ewin, select_mode = select_near_zero_states(
            evals=evals,
            bulk_gap=bulk_gap,
            floor=args.energy_window_floor,
            gap_factor=args.window_gap_factor,
            window_max=args.energy_window_max,
            max_states=args.max_states_per_v,
            jump_ratio=args.window_jump_ratio,
        )

        rho_sum = np.zeros((args.ly, args.lx), dtype=float)
        for idx in idxs:
            rho_sum += cell_density(vecs[:, idx], lx=args.lx, ly=args.ly)
            marked_rows.append(
                {
                    "model": spec.key,
                    "v": float(v),
                    "state_index": int(idx),
                    "energy": float(evals[idx]),
                    "abs_energy": float(abs(evals[idx])),
                    "energy_window": float(ewin),
                    "selection_mode": select_mode,
                }
            )
        rho_sum = rho_sum / max(np.sum(rho_sum), 1e-15)
        wc, we, wb = state_weights_2d(rho_sum, corner=corner, edge=edge, bulk=bulk)
        summary_rows.append(
            {
                "model": spec.key,
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "energy_window_raw": float(ewin_raw),
                "energy_window_cap": float(ewin_cap),
                "energy_window": float(ewin),
                "num_states_summed": int(idxs.size),
                "min_abs_energy_in_window": float(np.min(np.abs(evals[idxs]))),
                "max_abs_energy_in_window": float(np.max(np.abs(evals[idxs]))),
                "W_corner_sum": float(wc),
                "W_edge_sum": float(we),
                "W_bulk_sum": float(wb),
                "selection_mode": select_mode,
            }
        )

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(evals.size)
        ax.plot(x, evals, marker="o", linestyle="none", markersize=2.4, color="#2a66b8", alpha=0.75, label="all states")
        ax.scatter(idxs, evals[idxs], s=26, color="red", zorder=5, label="states used in WF sum")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(ewin, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(-ewin, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("state index")
        ax.set_ylabel("energy")
        ax.set_title(f"{spec.title} | OBC E-index (red = WF-sum states) | v={v:.1f}")
        ax.grid(alpha=0.22)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_obc / f"open_open_spectrum_mark_sum_v{token(v)}_{spec.key}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.8, 4.1))
        im = ax.imshow(rho_sum, origin="lower", cmap="magma")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{spec.title} | summed near-zero WF | v={v:.1f}\n"
            f"N={idxs.size}, Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}"
        )
        fig.colorbar(im, ax=ax, label=r"$\sum |\psi_n|^2$")
        fig.tight_layout()
        fig.savefig(out_wf / f"wf_near_zero_sum_v{v:.1f}_{spec.key}.png", dpi=180)
        plt.close(fig)
        print(
            f"[{spec.key}] v={v:.1f} gap={bulk_gap:.5f} "
            f"window_raw={ewin_raw:.5f} window_eff={ewin:.5f} states={idxs.size} mode={select_mode}"
        )

    write_csv(
        out / f"summary_{spec.key}.csv",
        summary_rows,
        [
            "model",
            "v",
            "bulk_gap",
            "energy_window_raw",
            "energy_window_cap",
            "energy_window",
            "num_states_summed",
            "min_abs_energy_in_window",
            "max_abs_energy_in_window",
            "W_corner_sum",
            "W_edge_sum",
            "W_bulk_sum",
            "selection_mode",
        ],
    )
    write_csv(
        out / f"obc_marked_states_{spec.key}.csv",
        marked_rows,
        ["model", "v", "state_index", "energy", "abs_energy", "energy_window", "selection_mode"],
    )


def main() -> None:
    args = parse_args()
    v_list = parse_v_list(args.v_list)
    out_root = ensure_dir(Path(args.output_root))

    p_b = T34Params(t34=args.t, w=args.w, lm=args.lm)
    p_c = T34Soc34Params(t34=args.t, w=args.w, lm=args.lm)

    h8_a = lambda kx, ky, v: add_uniform_fm_term(  # noqa: E731
        h8_full_t_full_soc(kx, ky, v=v, t=args.t, w=args.w, lm=args.lm),
        fm_out=args.fm_out,
        fm_in=args.fm_in,
    )
    h8_b = lambda kx, ky, v: add_uniform_fm_term(  # noqa: E731
        h8_t34_k(kx, ky, v=v, params=p_b),
        fm_out=args.fm_out,
        fm_in=args.fm_in,
    )
    h8_c = lambda kx, ky, v: add_uniform_fm_term(  # noqa: E731
        h8_t34_soc34_k(kx, ky, v=v, params=p_c),
        fm_out=args.fm_out,
        fm_in=args.fm_in,
    )

    run_one_model(
        base=out_root,
        spec=ModelSpec("model_a_full_t_full_soc", "Model A (full t + full SOC)"),
        v_list=v_list,
        args=args,
        h8_func=h8_a,
        bulk_gap_func=lambda v: compute_bulk_gap_from_h8(v=v, h8_func=h8_a, nk=args.bulk_gap_nk, n_occ=4),
        ribbon_builder=lambda ky, nx, v: add_uniform_fm_term(
            build_ribbon_a(ky, nx=nx, v=v, t=args.t, w=args.w, lm=args.lm),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
        open_open_builder=lambda v: add_uniform_fm_term(
            build_open_open_a(lx=args.lx, ly=args.ly, v=v, t=args.t, w=args.w, lm=args.lm),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
    )

    run_one_model(
        base=out_root,
        spec=ModelSpec("model_b_t34_full_soc", "Model B (t34-only + full SOC)"),
        v_list=v_list,
        args=args,
        h8_func=h8_b,
        bulk_gap_func=lambda v: compute_bulk_gap_from_h8(v=v, h8_func=h8_b, nk=args.bulk_gap_nk, n_occ=4),
        ribbon_builder=lambda ky, nx, v: add_uniform_fm_term(
            build_ribbon_b(ky=ky, nx=nx, v=v, params=p_b),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
        open_open_builder=lambda v: add_uniform_fm_term(
            build_open_open_b(lx=args.lx, ly=args.ly, v=v, params=p_b, termination="A"),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
    )

    run_one_model(
        base=out_root,
        spec=ModelSpec("model_c_t34_soc34_only", "Model C (t34-only + SOC34-only)"),
        v_list=v_list,
        args=args,
        h8_func=h8_c,
        bulk_gap_func=lambda v: compute_bulk_gap_from_h8(v=v, h8_func=h8_c, nk=args.bulk_gap_nk, n_occ=4),
        ribbon_builder=lambda ky, nx, v: add_uniform_fm_term(
            build_ribbon_c(ky=ky, nx=nx, v=v, params=p_c),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
        open_open_builder=lambda v: add_uniform_fm_term(
            build_open_open_c(lx=args.lx, ly=args.ly, v=v, params=p_c, termination="A"),
            fm_out=args.fm_out,
            fm_in=args.fm_in,
        ),
    )

    (out_root / "README.txt").write_text(
        "\n".join(
            [
                "three-model visual outputs",
                f"v_list={','.join(f'{v:.3f}' for v in v_list)}",
                f"params: t={args.t}, w={args.w}, lm={args.lm}",
                f"ferromagnetism: fm_out={args.fm_out}, fm_in={args.fm_in}",
                "Each model directory contains:",
                "  01_bulk_band/ (red=spin-up dominant, blue=spin-down dominant)",
                "  02_ribbon/",
                "  03_obc_marked/ (red points = states used in WF summation)",
                "  04_wf_sum/",
                "  summary_*.csv and obc_marked_states_*.csv",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] done: {out_root}")


if __name__ == "__main__":
    main()
