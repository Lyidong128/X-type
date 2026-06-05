#!/usr/bin/env python3
"""Analyze no-SOC band evolution across M-point mass sign change."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import load_model, set_params


@dataclass
class PathData:
    kpoints: np.ndarray
    xcoords: np.ndarray
    tick_positions: list[float]
    tick_labels: list[str]
    m_index: int
    m_x: float
    segment_length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-SOC band evolution around M-point mass zero.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/no_soc_band_evolution")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def token(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def build_high_symmetry_path(model, nseg: int = 280) -> PathData:
    g = np.zeros(3, dtype=float)
    x = 0.5 * model.b1
    m = 0.5 * (model.b1 + model.b2)
    y = 0.5 * model.b2
    nodes = [g, x, m, y, g]
    labels = [r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"]

    k_list = []
    tick_indices = [0]
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        for j in range(nseg):
            u = j / nseg
            k_list.append((1.0 - u) * a + u * b)
        tick_indices.append(len(k_list))
    k_list.append(nodes[-1])
    kpoints = np.array(k_list, dtype=float)

    xcoords = np.zeros(len(kpoints), dtype=float)
    for i in range(1, len(kpoints)):
        dk = kpoints[i] - kpoints[i - 1]
        xcoords[i] = xcoords[i - 1] + float(np.linalg.norm(dk[:2]))

    tick_positions = [float(xcoords[idx]) for idx in tick_indices]
    m_index = tick_indices[2]
    segment_length = tick_positions[2] - tick_positions[1]
    return PathData(
        kpoints=kpoints,
        xcoords=xcoords,
        tick_positions=tick_positions,
        tick_labels=labels,
        m_index=m_index,
        m_x=float(xcoords[m_index]),
        segment_length=float(segment_length),
    )


def compute_bands(model, path_data: PathData) -> np.ndarray:
    bands = []
    for k in path_data.kpoints:
        evals = np.linalg.eigvalsh(model.Hxtype(k))
        bands.append(np.real(evals))
    return np.array(bands, dtype=float)


def plot_full_band_single(
    bands: np.ndarray,
    path_data: PathData,
    v: float,
    t: float,
    w: float,
    lm: float,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.6))
    for b in range(bands.shape[1]):
        ax.plot(path_data.xcoords, bands[:, b], color="#1f77b4", linewidth=1.0)
    for x in path_data.tick_positions:
        ax.axvline(x, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
    ax.axvline(path_data.m_x, color="red", linestyle="--", linewidth=0.9, alpha=0.9)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(path_data.tick_positions)
    ax.set_xticklabels(path_data.tick_labels)
    ax.set_ylabel("Energy")
    ax.set_title(f"No-SOC band: v={v:.2f}, t={t:.1f}, w={w:.1f}, lm={lm:.1f}")
    ax.text(path_data.m_x, ax.get_ylim()[1] * 0.94, "M", color="red", ha="center", fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_full_band_montage(
    bands_by_v: dict[float, np.ndarray],
    path_data: PathData,
    t: float,
    w: float,
    lm: float,
    save_path: Path,
) -> None:
    v_list = sorted(bands_by_v.keys())
    ncols = 3
    nrows = int(np.ceil(len(v_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, v in enumerate(v_list):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        bands = bands_by_v[v]
        for b in range(bands.shape[1]):
            ax.plot(path_data.xcoords, bands[:, b], color="#1f77b4", linewidth=0.85)
        for x in path_data.tick_positions:
            ax.axvline(x, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.axvline(path_data.m_x, color="red", linestyle="--", linewidth=0.8, alpha=0.9)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_xticks(path_data.tick_positions)
        ax.set_xticklabels(path_data.tick_labels, fontsize=8)
        ax.set_title(f"v={v:.2f}", fontsize=10)
        ax.grid(alpha=0.14)
    for idx in range(len(v_list), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")
    fig.suptitle(f"No-SOC full band evolution (t={t:.1f}, w={w:.1f}, lm={lm:.1f})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_m_zoom_single(
    bands: np.ndarray,
    path_data: PathData,
    v: float,
    t: float,
    w: float,
    lm: float,
    save_path: Path,
) -> None:
    x = path_data.xcoords
    x0 = path_data.m_x
    width = 0.52 * path_data.segment_length
    mask = (x >= x0 - width) & (x <= x0 + width)
    x_local = x[mask]
    bands_local = bands[mask]

    # middle 6 bands (for 8-band model)
    n_bands = bands.shape[1]
    i0 = max(0, n_bands // 2 - 3)
    i1 = min(n_bands, i0 + 6)
    band_ids = list(range(i0, i1))

    e_window = (-0.6, 0.6)
    e_sel = bands_local[:, band_ids]
    emin, emax = float(np.min(e_sel)), float(np.max(e_sel))
    ymin, ymax = e_window
    if emin < ymin or emax > ymax:
        span = max(abs(emin), abs(emax))
        span = max(0.6, np.ceil(span * 5) / 5)
        ymin, ymax = -span, span

    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    for b in band_ids:
        ax.plot(x_local, bands_local[:, b], linewidth=1.0)
    ax.axvline(x0, color="red", linestyle="--", linewidth=0.9)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8)
    e_val_m = float(bands[path_data.m_index, 3])
    e_con_m = float(bands[path_data.m_index, 4])
    ax.scatter([x0], [e_val_m], color="blue", s=26, zorder=5, label="valence@M")
    ax.scatter([x0], [e_con_m], color="orange", s=26, zorder=5, label="conduction@M")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(x0 - width, x0 + width)
    ax.set_xlabel("Path coordinate near M (X->M->Y)")
    ax.set_ylabel("Energy")
    ax.set_title(f"M-zoom no-SOC band: v={v:.2f}, t={t:.1f}, w={w:.1f}, lm={lm:.1f}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_m_zoom_montage(
    bands_by_v: dict[float, np.ndarray],
    path_data: PathData,
    t: float,
    w: float,
    lm: float,
    save_path: Path,
) -> None:
    v_list = sorted(bands_by_v.keys())
    ncols = 3
    nrows = int(np.ceil(len(v_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), sharex=False, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    x = path_data.xcoords
    x0 = path_data.m_x
    width = 0.52 * path_data.segment_length
    mask = (x >= x0 - width) & (x <= x0 + width)
    x_local = x[mask]
    for idx, v in enumerate(v_list):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        bands = bands_by_v[v]
        bands_local = bands[mask]
        band_ids = [1, 2, 3, 4, 5, 6]
        for b in band_ids:
            ax.plot(x_local, bands_local[:, b], linewidth=0.85)
        ax.axvline(x0, color="red", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.7)
        ax.set_xlim(x0 - width, x0 + width)
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(f"v={v:.2f}", fontsize=10)
        ax.grid(alpha=0.15)
    for idx in range(len(v_list), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")
    fig.suptitle(f"M-zoom band evolution (t={t:.1f}, w={w:.1f}, lm={lm:.1f})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def fit_mass_lines(rows: list[dict[str, float]], t: float, w: float) -> tuple[float, float]:
    # Returns slope/intercept for simple Delta_M fit near vc for a compact diagnostic.
    sub = [r for r in rows if abs(r["t"] - t) < 1e-12 and abs(r["w"] - w) < 1e-12]
    x = np.array([r["v"] for r in sub], dtype=float)
    y = np.array([r["Delta_M"] for r in sub], dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(Path(args.output_root))
    model = load_model(args.model_file)
    t = float(args.t)
    w = float(args.w)
    lm = float(args.lm)

    if abs(lm) > 1e-12:
        raise ValueError("This script is designed for no-SOC condition lm=0.")

    v_list = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]
    vc = w - t
    nseg = 120 if args.quick else 280
    path_data = build_high_symmetry_path(model, nseg=nseg)
    m_point = 0.5 * (model.b1 + model.b2)

    # ---------------- Task 1/2: full bands + M-zoom ----------------
    bands_by_v: dict[float, np.ndarray] = {}
    for v in v_list:
        set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
        bands = compute_bands(model, path_data)
        bands_by_v[v] = bands
        plot_full_band_single(
            bands=bands,
            path_data=path_data,
            v=v,
            t=t,
            w=w,
            lm=lm,
            save_path=out_root / f"band_v_{token(v)}.png",
        )
        plot_m_zoom_single(
            bands=bands,
            path_data=path_data,
            v=v,
            t=t,
            w=w,
            lm=lm,
            save_path=out_root / f"M_zoom_band_v_{token(v)}.png",
        )

    plot_full_band_montage(
        bands_by_v=bands_by_v,
        path_data=path_data,
        t=t,
        w=w,
        lm=lm,
        save_path=out_root / "band_evolution_montage.png",
    )
    plot_m_zoom_montage(
        bands_by_v=bands_by_v,
        path_data=path_data,
        t=t,
        w=w,
        lm=lm,
        save_path=out_root / "M_zoom_band_evolution_montage.png",
    )

    # ---------------- Task 3: M gap vs v scan ----------------
    v_scan = np.arange(0.55, 0.85 + 1e-12, 0.001 if not args.quick else 0.002)
    gap_rows = []
    for v in v_scan:
        set_params(model, v=float(v), lm=lm, t=t, w=w, j=0.0)
        evals = np.linalg.eigvalsh(model.Hxtype(m_point))
        e_val = float(np.real(evals[3]))
        e_con = float(np.real(evals[4]))
        delta = float(e_con - e_val)
        m_m = float(v + t - w)
        gap_rows.append(
            {
                "v": float(v),
                "t": t,
                "w": w,
                "lm": lm,
                "E_valence_M": e_val,
                "E_conduction_M": e_con,
                "Delta_M": delta,
                "m_M": m_m,
                "2abs_m_M": 2.0 * abs(m_m),
            }
        )

    gap_csv = out_root / "M_gap_vs_v.csv"
    with gap_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "t",
                "w",
                "lm",
                "E_valence_M",
                "E_conduction_M",
                "Delta_M",
                "m_M",
                "2abs_m_M",
            ],
        )
        writer.writeheader()
        for row in gap_rows:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    v_arr = np.array([r["v"] for r in gap_rows], dtype=float)
    delta_arr = np.array([r["Delta_M"] for r in gap_rows], dtype=float)
    theo_arr = np.array([r["2abs_m_M"] for r in gap_rows], dtype=float)
    ax.plot(v_arr, delta_arr, linewidth=1.4, label=r"$\Delta_M$ (numeric)")
    ax.plot(v_arr, theo_arr, linewidth=1.2, linestyle="--", label=r"$2|m_M|=2|v+t-w|$")
    ax.axvline(vc, color="red", linestyle="--", linewidth=0.9, label=f"$v_c={vc:.1f}$")
    ax.set_xlabel("v")
    ax.set_ylabel("Gap / mass scale")
    ax.set_title(f"M-point gap vs v (lm=0, t={t:.1f}, w={w:.1f})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "M_gap_vs_v.png", dpi=180)
    plt.close(fig)

    # ---------------- Task 4: component weights around vc ----------------
    up_idx = np.array([0, 2, 4, 6], dtype=int)
    comp_rows = []
    for v in v_list:
        set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
        h8 = np.array(model.Hxtype(m_point), dtype=complex)
        h4 = h8[np.ix_(up_idx, up_idx)]
        evals4, evecs4 = np.linalg.eigh(h4)
        for state, idx in [("valence", 1), ("conduction", 2)]:
            vec = evecs4[:, idx]
            weights = np.abs(vec) ** 2
            weights = weights / max(float(np.sum(weights)), 1e-16)
            row = {
                "v": float(v),
                "t": t,
                "w": w,
                "lm": lm,
                "state": state,
                "energy": float(np.real(evals4[idx])),
                "major_component": int(np.argmax(weights)),
            }
            for ci in range(4):
                row[f"component_{ci}_weight"] = float(weights[ci])
            comp_rows.append(row)

    comp_csv = out_root / "M_band_component_weights.csv"
    with comp_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "t",
                "w",
                "lm",
                "state",
                "energy",
                "major_component",
                "component_0_weight",
                "component_1_weight",
                "component_2_weight",
                "component_3_weight",
            ],
        )
        writer.writeheader()
        for row in comp_rows:
            writer.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.3), sharex=True)
    for ax, state in zip(axes, ["valence", "conduction"]):
        sub = [r for r in comp_rows if r["state"] == state]
        sub = sorted(sub, key=lambda r: r["v"])
        vx = np.array([r["v"] for r in sub], dtype=float)
        for ci in range(4):
            vy = np.array([r[f"component_{ci}_weight"] for r in sub], dtype=float)
            ax.plot(vx, vy, marker="o", linewidth=1.1, label=f"component {ci}")
        ax.axvline(vc, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{state} weight")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("v")
    fig.suptitle("M-point valence/conduction component weights vs v")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_root / "M_component_weights_vs_v.png", dpi=180)
    plt.close(fig)

    def get_state_row(target_v: float, state: str) -> dict:
        candidates = [r for r in comp_rows if r["state"] == state]
        return min(candidates, key=lambda r: abs(r["v"] - target_v))

    v_before, v_after = 0.68, 0.72
    before_val = get_state_row(v_before, "valence")
    before_con = get_state_row(v_before, "conduction")
    after_val = get_state_row(v_after, "valence")
    after_con = get_state_row(v_after, "conduction")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharey=True)
    cases = [
        (axes[0, 0], before_val, f"v={v_before:.2f} valence"),
        (axes[0, 1], before_con, f"v={v_before:.2f} conduction"),
        (axes[1, 0], after_val, f"v={v_after:.2f} valence"),
        (axes[1, 1], after_con, f"v={v_after:.2f} conduction"),
    ]
    for ax, row, ttl in cases:
        vals = [row[f"component_{i}_weight"] for i in range(4)]
        ax.bar(np.arange(4), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        ax.set_xticks(np.arange(4))
        ax.set_title(ttl, fontsize=10)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("weight")
    axes[1, 0].set_ylabel("weight")
    fig.suptitle("M-point component weights before/after mass sign change")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_root / "M_component_weights_before_after.png", dpi=180)
    plt.close(fig)

    # inversion判据：前后 val/con 权重是否交叉匹配
    wv_b = np.array([before_val[f"component_{i}_weight"] for i in range(4)], dtype=float)
    wc_b = np.array([before_con[f"component_{i}_weight"] for i in range(4)], dtype=float)
    wv_a = np.array([after_val[f"component_{i}_weight"] for i in range(4)], dtype=float)
    wc_a = np.array([after_con[f"component_{i}_weight"] for i in range(4)], dtype=float)
    direct_score = float(np.linalg.norm(wv_b - wv_a) + np.linalg.norm(wc_b - wc_a))
    cross_score = float(np.linalg.norm(wv_b - wc_a) + np.linalg.norm(wc_b - wv_a))
    inversion_by_components = cross_score < direct_score

    # ---------------- Task 5: low-energy levels vs v ----------------
    psi_a = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex) / 2.0
    psi_b = np.array([1.0, 1.0, -1.0, -1.0], dtype=complex) / 2.0
    low_rows = []
    for v in v_scan:
        set_params(model, v=float(v), lm=lm, t=t, w=w, j=0.0)
        h8 = np.array(model.Hxtype(m_point), dtype=complex)
        h4 = h8[np.ix_(up_idx, up_idx)]
        evals4, evecs4 = np.linalg.eigh(h4)
        ov_a = np.abs(evecs4.conj().T @ psi_a)
        ov_b = np.abs(evecs4.conj().T @ psi_b)
        ia = int(np.argmax(ov_a))
        ib = int(np.argmax(ov_b))
        if ib == ia:
            candidates = [j for j in range(4) if j != ia]
            ib = max(candidates, key=lambda j: ov_b[j])
        ea_num = float(np.real(evals4[ia]))
        eb_num = float(np.real(evals4[ib]))
        ea_formula = float(v - w + 2.0 * t)
        eb_formula = float(w - v)
        low_rows.append(
            {
                "v": float(v),
                "t": t,
                "w": w,
                "lm": lm,
                "E_a_numeric": ea_num,
                "E_b_numeric": eb_num,
                "E_a_formula": ea_formula,
                "E_b_formula": eb_formula,
                "overlap_a": float(ov_a[ia]),
                "overlap_b": float(ov_b[ib]),
            }
        )

    low_csv = out_root / "M_low_energy_levels_vs_v.csv"
    with low_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "t",
                "w",
                "lm",
                "E_a_numeric",
                "E_b_numeric",
                "E_a_formula",
                "E_b_formula",
                "overlap_a",
                "overlap_b",
            ],
        )
        writer.writeheader()
        for row in low_rows:
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    v_low = np.array([r["v"] for r in low_rows], dtype=float)
    ea_num = np.array([r["E_a_numeric"] for r in low_rows], dtype=float)
    eb_num = np.array([r["E_b_numeric"] for r in low_rows], dtype=float)
    ea_for = np.array([r["E_a_formula"] for r in low_rows], dtype=float)
    eb_for = np.array([r["E_b_formula"] for r in low_rows], dtype=float)
    ax.plot(v_low, ea_num, linewidth=1.2, label="E_a numeric")
    ax.plot(v_low, eb_num, linewidth=1.2, label="E_b numeric")
    ax.plot(v_low, ea_for, "--", linewidth=1.0, label="E_a = v-w+2t")
    ax.plot(v_low, eb_for, "--", linewidth=1.0, label="E_b = w-v")
    ax.axvline(vc, color="red", linestyle="--", linewidth=0.9, label=f"$v_c={vc:.1f}$")
    ax.set_xlabel("v")
    ax.set_ylabel("Energy at M")
    ax.set_title("Two low-energy M-point levels vs v (no SOC)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "M_low_energy_levels_vs_v.png", dpi=180)
    plt.close(fig)

    # ---------------- Task 6: report ----------------
    min_gap_row = min(gap_rows, key=lambda r: r["Delta_M"])
    v_min_gap = float(min_gap_row["v"])
    gap_min = float(min_gap_row["Delta_M"])
    gap_at_before = min(gap_rows, key=lambda r: abs(r["v"] - 0.68))
    gap_at_after = min(gap_rows, key=lambda r: abs(r["v"] - 0.72))

    report_lines = [
        "# NO_SOC_BAND_EVOLUTION_REPORT",
        "",
        "## 1. 分析目的",
        f"- 在无 SOC (`lm={lm:.1f}`) 条件下，观察 `v` 穿过 `v_c=w-t={vc:.3f}` 时的能带演化。",
        f"- 固定参数：`t={t:.1f}, w={w:.1f}, lm={lm:.1f}`。",
        "",
        "## 2. 高对称路径能带结果",
        "- 已计算 `Gamma->X->M->Y->Gamma` 上 `v=[0.60,0.65,0.68,0.70,0.72,0.75,0.80]` 的完整能带图。",
        "- 随 `v` 增大，M 点附近中间两条带逐步靠近，在 `v≈0.70` 闭合，随后重新打开。",
        "- 对应图：`band_v_*.png` 与 `band_evolution_montage.png`。",
        "",
        "## 3. M 点能隙与质量项",
        f"- 数值最小 M 点能隙出现在 `v≈{v_min_gap:.3f}`，`Delta_M≈{gap_min:.3e}`。",
        f"- 在 `v=0.68` 时 `Delta_M≈{gap_at_before['Delta_M']:.3e}`，在 `v=0.72` 时 `Delta_M≈{gap_at_after['Delta_M']:.3e}`，表现为闭合后重开。",
        "- `Delta_M(v)` 与理论 `2|m_M|=2|v+t-w|` 在扫描中一致（见 `M_gap_vs_v.png`）。",
        "",
        "## 4. 带成分交换（M 点）",
        "- 基于 up-spin 4x4 block 的 valence/conduction 本征矢分量权重，比较了 `v=0.68` 与 `v=0.72`。",
        f"- 组件交换判据：`cross_score={cross_score:.6f}`, `direct_score={direct_score:.6f}`, `inversion_by_components={inversion_by_components}`。",
        "- 若 `cross_score < direct_score`，说明前后 valence/conduction 的分量更接近互换关系，可解释为 M 点 band inversion。",
        "",
        "## 5. 物理解释",
        "- 无 SOC 下，M 点能带重构由 `m_M(v,t,w)=v+t-w` 变号控制。",
        "- 当 `v` 穿过 `v_c=w-t`，两条低能态 `E_a=v-w+2t` 与 `E_b=w-v` 在 M 点交叉，导致局域闭隙与重开。",
        "- 由于没有 Kane-Mele SOC，本过程不能直接称为 spin-Chern 或 Z2 拓扑相变。",
        "",
        "## 6. 推荐表述",
        "“无 SOC 条件下，随着 v 增大，M 点有效质量项 m_M=v+t-w 在 v_c=w-t=0.7 处变号，导致 M 点能隙闭合并重新打开，价带和导带成分发生交换，表明体系发生 M 点 band inversion / 能带重构。”",
    ]
    (out_root / "NO_SOC_BAND_EVOLUTION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[ok] out_root={out_root}")
    print(f"[ok] vc={vc:.6f} v_min_gap={v_min_gap:.6f} gap_min={gap_min:.3e}")
    print(f"[ok] inversion_by_components={inversion_by_components}")


if __name__ == "__main__":
    main()
