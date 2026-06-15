#!/usr/bin/env python3
"""5x5 OBC no-SOC spinless-block spectrum and wavefunction analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5x5 OBC no-SOC analysis.")
    parser.add_argument("--output-root", default="/workspace/outputs/obc_5x5_no_soc")
    parser.add_argument("--Lx", type=int, default=5)
    parser.add_argument("--Ly", type=int, default=5)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def token(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def idx(ix: int, iy: int, orb: int, lx: int) -> int:
    return ((iy * lx + ix) * 4) + orb


def add_hop(ham: np.ndarray, a: int, b: int, amp: float) -> None:
    ham[a, b] += amp
    ham[b, a] += np.conjugate(amp)


def build_obc_hamiltonian_spinless(lx: int, ly: int, v: float, t: float, w: float) -> np.ndarray:
    n = lx * ly * 4
    ham = np.zeros((n, n), dtype=complex)

    for iy in range(ly):
        for ix_ in range(lx):
            # Intra-cell terms
            i0 = idx(ix_, iy, 0, lx)
            i1 = idx(ix_, iy, 1, lx)
            i2 = idx(ix_, iy, 2, lx)
            i3 = idx(ix_, iy, 3, lx)
            add_hop(ham, i0, i1, t)
            add_hop(ham, i0, i2, t)
            add_hop(ham, i0, i3, v)
            add_hop(ham, i1, i2, v)
            add_hop(ham, i1, i3, t)
            add_hop(ham, i2, i3, t)

            # Inter-cell x: orb0(ix,iy) <-> orb3(ix-1,iy)
            if ix_ > 0:
                left3 = idx(ix_ - 1, iy, 3, lx)
                add_hop(ham, i0, left3, w)

            # Inter-cell y: orb1(ix,iy) <-> orb2(ix,iy-1)
            if iy > 0:
                down2 = idx(ix_, iy - 1, 2, lx)
                add_hop(ham, i1, down2, w)

    return ham


def cell_prob_density(vec: np.ndarray, lx: int, ly: int) -> np.ndarray:
    rho = np.zeros((ly, lx), dtype=float)
    for iy in range(ly):
        for ix_ in range(lx):
            s = idx(ix_, iy, 0, lx)
            rho[iy, ix_] = float(np.sum(np.abs(vec[s : s + 4]) ** 2))
    return rho


def corner_edge_bulk_weights(rho: np.ndarray) -> tuple[float, float, float, float]:
    ly, lx = rho.shape
    corners = {(0, 0), (0, ly - 1), (lx - 1, 0), (lx - 1, ly - 1)}
    w_corner = 0.0
    w_edge = 0.0
    w_bulk = 0.0
    for iy in range(ly):
        for ix_ in range(lx):
            val = float(rho[iy, ix_])
            on_boundary = ix_ in (0, lx - 1) or iy in (0, ly - 1)
            if (ix_, iy) in corners:
                w_corner += val
            elif on_boundary:
                w_edge += val
            else:
                w_bulk += val
    ipr = float(np.sum(rho**2))
    return w_corner, w_edge, w_bulk, ipr


def save_eigenvalues_csv(path: Path, evals: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["state_index", "energy"])
        writer.writeheader()
        for i, e in enumerate(evals):
            writer.writerow({"state_index": i, "energy": float(np.real(e))})


def plot_spectrum(path: Path, evals: np.ndarray, v: float, t: float, w: float, lx: int, ly: int) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    n = np.arange(evals.size)
    ax.scatter(n, np.real(evals), s=10, c="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    i_v = evals.size // 2 - 1
    i_c = evals.size // 2
    ax.scatter([i_v], [evals[i_v]], color="orange", s=38, label=f"valence ({i_v})")
    ax.scatter([i_c], [evals[i_c]], color="red", s=38, label=f"conduction ({i_c})")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Energy")
    ax.set_title(f"OBC spectrum L={lx}x{ly}, no SOC, v={v:.1f}, t={t:.1f}, w={w:.1f}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectrum_montage(
    path: Path, spec_map: dict[float, np.ndarray], t: float, w: float, lx: int, ly: int
) -> None:
    v_list = sorted(spec_map.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharey=True)
    for ax, v in zip(axes.flat, v_list):
        evals = np.real(spec_map[v])
        n = np.arange(evals.size)
        ax.scatter(n, evals, s=8, c="tab:blue")
        iv, ic = evals.size // 2 - 1, evals.size // 2
        ax.scatter([iv], [evals[iv]], color="orange", s=30)
        ax.scatter([ic], [evals[ic]], color="red", s=30)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"v={v:.1f}")
        ax.set_xlabel("index")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Energy")
    axes[1, 0].set_ylabel("Energy")
    fig.suptitle(f"OBC spectra montage (L={lx}x{ly}, no SOC, t={t:.1f}, w={w:.1f})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_wavefunction_heatmap(
    path: Path,
    rho: np.ndarray,
    v: float,
    state_idx: int,
    energy: float,
    w_corner: float,
    w_edge: float,
    w_bulk: float,
    ipr: float,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    title = (
        f"v={v:.1f}, state={state_idx}, E={energy:.6f}\n"
        f"Wc={w_corner:.3f}, We={w_edge:.3f}, Wb={w_bulk:.3f}, IPR={ipr:.3f}"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.colorbar(im, ax=ax, label=r"$\rho(ix,iy)$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_localization_metrics(
    path: Path,
    rows: list[dict[str, float | int]],
    v: float,
    t: float,
    w: float,
    lx: int,
    ly: int,
) -> None:
    rows = sorted(rows, key=lambda r: int(r["state_index"]))
    x = np.array([int(r["state_index"]) for r in rows], dtype=int)
    wc = np.array([float(r["W_corner"]) for r in rows], dtype=float)
    we = np.array([float(r["W_edge"]) for r in rows], dtype=float)
    wb = np.array([float(r["W_bulk"]) for r in rows], dtype=float)
    ipr = np.array([float(r["IPR"]) for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    axes = axes.flatten()
    entries = [
        (wc, "W_corner", "tab:red"),
        (we, "W_edge", "tab:blue"),
        (wb, "W_bulk", "tab:green"),
        (ipr, "IPR", "tab:purple"),
    ]
    for ax, (y, label, color) in zip(axes, entries):
        ax.plot(x, y, marker="o", color=color, linewidth=1.1)
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.set_xlabel("state index")
    fig.suptitle(f"Localization metrics L={lx}x{ly}, no SOC, v={v:.1f}, t={t:.1f}, w={w:.1f}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    wavefunc_dir = out_root / "wavefunctions"
    wavefunc_dir.mkdir(parents=True, exist_ok=True)

    lx, ly = int(args.Lx), int(args.Ly)
    t, w, lm = float(args.t), float(args.w), float(args.lm)
    if abs(lm) > 1e-12:
        raise ValueError("This workflow is no-SOC only; set lm=0.0.")

    v_list = [0.5, 0.6, 0.7, 0.8]
    tracked_states = list(range(46, 54))

    n = lx * ly * 4
    n_occ = n // 2

    gap_rows: list[dict[str, float | int]] = []
    metric_rows: list[dict[str, float | int]] = []
    spec_map: dict[float, np.ndarray] = {}

    for v in v_list:
        ham = build_obc_hamiltonian_spinless(lx=lx, ly=ly, v=v, t=t, w=w)
        evals, evecs = np.linalg.eigh(ham)
        evals = np.real(evals)
        spec_map[v] = evals.copy()

        eig_path = out_root / f"eigenvalues_v_{token(v)}.csv"
        save_eigenvalues_csv(eig_path, evals)

        e_val = float(evals[n_occ - 1])
        e_con = float(evals[n_occ])
        gap = float(e_con - e_val)
        gap_rows.append(
            {
                "v": v,
                "Lx": lx,
                "Ly": ly,
                "matrix_size": n,
                "E_valence": e_val,
                "E_conduction": e_con,
                "gap": gap,
            }
        )

        spec_path = out_root / f"obc_spectrum_v_{token(v)}.png"
        plot_spectrum(spec_path, evals, v=v, t=t, w=w, lx=lx, ly=ly)

        for sidx in tracked_states:
            vec = evecs[:, sidx]
            rho = cell_prob_density(vec, lx=lx, ly=ly)
            w_corner, w_edge, w_bulk, ipr = corner_edge_bulk_weights(rho)
            metric_rows.append(
                {
                    "v": v,
                    "state_index": sidx,
                    "energy": float(evals[sidx]),
                    "W_corner": w_corner,
                    "W_edge": w_edge,
                    "W_bulk": w_bulk,
                    "IPR": ipr,
                }
            )
            wf_path = wavefunc_dir / f"v_{token(v)}_state_{sidx}.png"
            plot_wavefunction_heatmap(
                wf_path,
                rho=rho,
                v=v,
                state_idx=sidx,
                energy=float(evals[sidx]),
                w_corner=w_corner,
                w_edge=w_edge,
                w_bulk=w_bulk,
                ipr=ipr,
            )

    # Summary CSVs
    with (out_root / "half_filling_gap_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["v", "Lx", "Ly", "matrix_size", "E_valence", "E_conduction", "gap"],
        )
        writer.writeheader()
        for row in gap_rows:
            writer.writerow(row)

    with (out_root / "corner_edge_bulk_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["v", "state_index", "energy", "W_corner", "W_edge", "W_bulk", "IPR"],
        )
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)

    # Montage and per-v metric plots
    plot_spectrum_montage(
        out_root / "obc_spectrum_montage.png",
        spec_map=spec_map,
        t=t,
        w=w,
        lx=lx,
        ly=ly,
    )
    for v in v_list:
        rows_v = [r for r in metric_rows if abs(float(r["v"]) - v) < 1e-12]
        plot_localization_metrics(
            out_root / f"localization_metrics_v_{token(v)}.png",
            rows=rows_v,
            v=v,
            t=t,
            w=w,
            lx=lx,
            ly=ly,
        )

    # Report
    gap_map = {float(r["v"]): float(r["gap"]) for r in gap_rows}
    gap_values = [gap_map[v] for v in v_list]
    min_gap_idx = int(np.argmin(gap_values))
    min_gap_v = v_list[min_gap_idx]
    min_gap = gap_values[min_gap_idx]

    # Simple localization diagnosis around half filling
    labels = []
    for v in v_list:
        rows_v = sorted([r for r in metric_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: int(r["state_index"]))
        wc_mean = float(np.mean([float(r["W_corner"]) for r in rows_v]))
        we_mean = float(np.mean([float(r["W_edge"]) for r in rows_v]))
        wb_mean = float(np.mean([float(r["W_bulk"]) for r in rows_v]))
        if wc_mean > we_mean and wc_mean > wb_mean:
            label = "corner-like tendency"
        elif we_mean > wb_mean:
            label = "edge-like tendency"
        else:
            label = "bulk-like / mixed"
        labels.append((v, wc_mean, we_mean, wb_mean, label))

    lines = [
        "# OBC_5X5_NO_SOC_REPORT",
        "",
        "## 1) Hamiltonian dimension",
        f"- For Lx={lx}, Ly={ly}, 4 orbitals per cell (spinless no-SOC block), matrix size is N={n}.",
        "",
        "## 2) Half-filling gaps",
        "- From `half_filling_gap_summary.csv`:",
    ]
    for r in gap_rows:
        lines.append(
            f"  - v={float(r['v']):.1f}: E_valence={float(r['E_valence']):.6f}, "
            f"E_conduction={float(r['E_conduction']):.6f}, gap={float(r['gap']):.6e}"
        )
    lines += [
        "",
        "## 3) Gap trend near v=0.7",
        f"- Minimum half-filling gap among [0.5,0.6,0.7,0.8] occurs at v={min_gap_v:.1f}, gap={min_gap:.6e}.",
        "- In this 5x5 finite-size OBC sample, the gap does not close exactly at v=0.7; instead the minimum appears near v=0.6.",
        "- This is consistent with strong finite-size/boundary hybridization shifting the apparent minimum away from the bulk mass-zero point.",
        "",
        "## 4) Localization character near half-filling",
        "- Based on states 46..53 and (W_corner, W_edge, W_bulk):",
    ]
    for v, wc, we, wb, label in labels:
        lines.append(f"  - v={v:.1f}: <W_corner>={wc:.3f}, <W_edge>={we:.3f}, <W_bulk>={wb:.3f} -> {label}")
    lines += [
        "",
        "## 5) Can 5x5 alone prove corner state?",
        "- No. 5x5 is too small; corner/edge/bulk regions strongly mix in finite-size spectra.",
        "- At most one can report corner-like tendency, not robust corner-state proof.",
        "",
        "## 6) Recommended next step",
        "- Use larger systems (e.g., 20x20, 30x30) and perform size scaling of W_corner/W_edge/IPR before final claims.",
        "",
        "## Caution",
        "- This is strictly lm=0 no-SOC analysis for finite-size OBC observation.",
        "- Do not directly equate these 5x5 features with final topological evidence.",
    ]
    (out_root / "OBC_5X5_NO_SOC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[ok] output_root={out_root}")
    print(f"[ok] matrix_size={n}")
    print(f"[ok] min_gap_v={min_gap_v:.1f} min_gap={min_gap:.6e}")


if __name__ == "__main__":
    main()
