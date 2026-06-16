#!/usr/bin/env python3
"""Construct and analyze a chiral-symmetrized version of the 4x4 spinless model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GAMMA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)
SUBLATTICE = {0: "A", 1: "A", 2: "B", 3: "B"}


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chiral-symmetric model analysis for zero-energy corner states.")
    parser.add_argument("--output-root", default="/workspace/outputs/chiral_symmetric_model")
    parser.add_argument("--Lx", type=int, default=5)
    parser.add_argument("--Ly", type=int, default=5)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--v-list", default="0.5,0.6,0.7,0.8")
    parser.add_argument("--alpha-list", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--nk", type=int, default=101, help="BZ grid size for chiral-error scan.")
    parser.add_argument("--near-count", type=int, default=8, help="How many near-zero states to analyze per case.")
    parser.add_argument("--zero-tol", type=float, default=1e-8)
    parser.add_argument("--near-zero-window", type=float, default=1e-3)
    return parser.parse_args()


def token(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace("-", "m").replace(".", "p")


def idx(ix: int, iy: int, orb: int, lx: int) -> int:
    return ((iy * lx + ix) * 4) + orb


def add_hop(ham: np.ndarray, i: int, j: int, amp: complex) -> None:
    ham[i, j] += amp
    ham[j, i] += np.conjugate(amp)


def h0_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 1] = t
    h[0, 2] = t
    h[0, 3] = v + w * np.exp(-1j * kx)
    h[1, 2] = v + w * np.exp(-1j * ky)
    h[1, 3] = t
    h[2, 3] = t
    h = h + h.conj().T
    np.fill_diagonal(h, 0.0)
    return h


def h_chiral_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    h = np.zeros((4, 4), dtype=complex)
    h[0, 2] = t
    h[0, 3] = v + w * np.exp(-1j * kx)
    h[1, 2] = v + w * np.exp(-1j * ky)
    h[1, 3] = t
    h = h + h.conj().T
    np.fill_diagonal(h, 0.0)
    return h


def h_break_k(kx: float, ky: float, v: float, t: float, w: float) -> np.ndarray:
    return h0_k(kx, ky, v, t, w) - h_chiral_k(kx, ky, v, t, w)


def h_alpha_k(kx: float, ky: float, v: float, t: float, w: float, alpha: float) -> np.ndarray:
    return h_chiral_k(kx, ky, v, t, w) + alpha * h_break_k(kx, ky, v, t, w)


def chiral_error(h: np.ndarray) -> float:
    denom = max(np.linalg.norm(h), 1e-15)
    numer = np.linalg.norm(GAMMA @ h @ GAMMA + h)
    return float(numer / denom)


def build_hopping_classification(path: Path) -> list[dict[str, object]]:
    terms = [
        (0, 1, "t"),
        (0, 2, "t"),
        (0, 3, "v + w*exp(-i kx)"),
        (1, 2, "v + w*exp(-i ky)"),
        (1, 3, "t"),
        (2, 3, "t"),
    ]
    rows: list[dict[str, object]] = []
    for i, j, expr in terms:
        si, sj = SUBLATTICE[i], SUBLATTICE[j]
        chiral_class = "allowed_AB" if si != sj else "breaks_chiral_same_sublattice"
        rows.append(
            {
                "from_orbital": i,
                "to_orbital": j,
                "from_sublattice": si,
                "to_sublattice": sj,
                "hopping_expression": expr,
                "chiral_classification": chiral_class,
            }
        )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "from_orbital",
                "to_orbital",
                "from_sublattice",
                "to_sublattice",
                "hopping_expression",
                "chiral_classification",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def scan_chiral_errors(nk: int, v_ref: float, t: float, w: float, out_csv: Path) -> dict[str, dict[str, float]]:
    klist = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    rows: list[dict[str, object]] = []
    err_h0: list[float] = []
    err_hc: list[float] = []
    err_block: list[float] = []

    for kx in klist:
        for ky in klist:
            h0 = h0_k(kx, ky, v_ref, t, w)
            hc = h_chiral_k(kx, ky, v_ref, t, w)
            q = hc[:2, 2:4]
            h_block = np.block(
                [
                    [np.zeros((2, 2), dtype=complex), q],
                    [q.conj().T, np.zeros((2, 2), dtype=complex)],
                ]
            )
            e0 = chiral_error(h0)
            ec = chiral_error(hc)
            eb = float(np.linalg.norm(hc - h_block) / max(np.linalg.norm(hc), 1e-15))
            err_h0.append(e0)
            err_hc.append(ec)
            err_block.append(eb)
            rows.append(
                {
                    "kx": float(kx),
                    "ky": float(ky),
                    "err_H0": e0,
                    "err_H_chiral": ec,
                    "block_reconstruction_err": eb,
                }
            )

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["kx", "ky", "err_H0", "err_H_chiral", "block_reconstruction_err"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "H0": {"max": float(np.max(err_h0)), "mean": float(np.mean(err_h0))},
        "H_chiral": {"max": float(np.max(err_hc)), "mean": float(np.mean(err_hc))},
        "block": {"max": float(np.max(err_block)), "mean": float(np.mean(err_block))},
    }


def write_q_block_expression(path: Path) -> None:
    lines = [
        "# Q(k) block for H_chiral",
        "",
        "With basis (A0, A1 | B2, B3),",
        "",
        "H_chiral(k) = [[0, Q(k)], [Q(k)^dagger, 0]]",
        "",
        "Q(k) = [[ t,                 v + w*exp(-i*kx) ],",
        "        [ v + w*exp(-i*ky),  t               ]]",
        "",
        "Explicit 4x4 H_chiral(k):",
        "[[0, 0, t, v+w*exp(-i*kx)],",
        " [0, 0, v+w*exp(-i*ky), t],",
        " [t, v+w*exp(+i*ky), 0, 0],",
        " [v+w*exp(+i*kx), t, 0, 0]]",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_obc_hamiltonian(lx: int, ly: int, v: float, t: float, w: float, alpha: float) -> np.ndarray:
    n = lx * ly * 4
    ham = np.zeros((n, n), dtype=complex)
    t_same = alpha * t

    for iy in range(ly):
        for ix_ in range(lx):
            i0 = idx(ix_, iy, 0, lx)
            i1 = idx(ix_, iy, 1, lx)
            i2 = idx(ix_, iy, 2, lx)
            i3 = idx(ix_, iy, 3, lx)

            # A-B intracell hoppings
            add_hop(ham, i0, i2, t)
            add_hop(ham, i0, i3, v)
            add_hop(ham, i1, i2, v)
            add_hop(ham, i1, i3, t)

            # Same-sublattice breaking hoppings
            add_hop(ham, i0, i1, t_same)
            add_hop(ham, i2, i3, t_same)

            # A-B intercell hoppings from phase terms
            if ix_ > 0:
                left3 = idx(ix_ - 1, iy, 3, lx)
                add_hop(ham, i0, left3, w)
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


def plot_spectrum(path: Path, evals: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    n = np.arange(evals.size)
    ax.scatter(n, evals, s=10, c="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectrum_montage(path: Path, spectra_by_alpha: dict[float, np.ndarray], v: float) -> None:
    alphas = sorted(spectra_by_alpha.keys())
    ncols = 3
    nrows = int(np.ceil(len(alphas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.1 * nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(nrows, ncols)
    for i, alpha in enumerate(alphas):
        ax = axes.flat[i]
        e = spectra_by_alpha[alpha]
        ax.scatter(np.arange(e.size), e, s=8, c="tab:blue")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.75)
        ax.set_title(f"alpha={alpha:.2f}", fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlabel("n")
    for i in range(len(alphas), nrows * ncols):
        axes.flat[i].axis("off")
    axes.flat[0].set_ylabel("Energy")
    fig.suptitle(f"OBC spectrum vs alpha (v={v:.2f})", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_wavefunction(path: Path, rho: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ix")
    ax.set_ylabel("iy")
    fig.colorbar(im, ax=ax, label="rho(x,y)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_min_abs_energy_vs_alpha(path: Path, summary_rows: list[dict[str, float | int]]) -> None:
    v_values = sorted({float(r["v"]) for r in summary_rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for v in v_values:
        sub = sorted([r for r in summary_rows if abs(float(r["v"]) - v) < 1e-12], key=lambda r: float(r["alpha"]))
        x = np.array([float(r["alpha"]) for r in sub], dtype=float)
        y = np.array([float(r["min_abs_energy"]) for r in sub], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.2, label=f"v={v:.2f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("min |E|")
    ax.set_title("Near-zero pinning vs chiral-breaking interpolation")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_corner_weight_vs_alpha(path: Path, metric_rows: list[dict[str, float | int]]) -> None:
    v_values = sorted({float(r["v"]) for r in metric_rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for v in v_values:
        by_alpha: list[tuple[float, float]] = []
        alphas = sorted({float(r["alpha"]) for r in metric_rows if abs(float(r["v"]) - v) < 1e-12})
        for alpha in alphas:
            sub = [
                r
                for r in metric_rows
                if abs(float(r["v"]) - v) < 1e-12
                and abs(float(r["alpha"]) - alpha) < 1e-12
                and int(r["rank_by_absE"]) < 4
            ]
            mean_wc = float(np.mean([float(r["W_corner"]) for r in sub])) if sub else np.nan
            by_alpha.append((alpha, mean_wc))
        x = np.array([a for a, _ in by_alpha], dtype=float)
        y = np.array([wc for _, wc in by_alpha], dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.2, label=f"v={v:.2f}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean W_corner (4 closest-to-zero states)")
    ax.set_title("Corner localization trend vs alpha")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    err_scan_rows: list[dict[str, float | int | str]],
    err_stats_agg: dict[str, dict[str, float]],
    q_path: Path,
    class_rows: list[dict[str, object]],
    summary_rows: list[dict[str, float | int]],
    metric_rows: list[dict[str, float | int]],
    lx: int,
    ly: int,
    t: float,
    w: float,
    nk: int,
    zero_tol: float,
    near_zero_window: float,
) -> None:
    v_values = sorted({float(r["v"]) for r in summary_rows})
    alpha_values = sorted({float(r["alpha"]) for r in summary_rows})

    def get_summary(v: float, alpha: float) -> dict[str, float]:
        return next(
            r
            for r in summary_rows
            if abs(float(r["v"]) - v) < 1e-12 and abs(float(r["alpha"]) - alpha) < 1e-12
        )

    lines: list[str] = [
        "# CHIRAL_SYMMETRY_ZERO_MODE_REPORT",
        "",
        "## 1) 原始 H0 的手征对称性检查",
        f"- 使用 Gamma=diag(+1,+1,-1,-1)，每个 v 在 BZ ({nk}x{nk}) 网格上计算。",
        f"- H0 最大误差（跨全部 v 取最大）: {err_stats_agg['H0']['max']:.6e}",
        f"- H0 平均误差（对每个 v 的 mean 再取平均）: {err_stats_agg['H0']['mean']:.6e}",
        "",
        "## 2) 跃迁项手征分类",
        "- 允许 A-B: (0,2), (0,3), (1,2), (1,3)",
        "- 破缺同子格: (0,1), (2,3)",
        "- 详见 `hopping_chiral_classification.csv`。",
        "",
        "## 3) 手征对称化 H_chiral 验证",
        f"- H_chiral 最大误差（跨全部 v 取最大）: {err_stats_agg['H_chiral']['max']:.6e}",
        f"- H_chiral 平均误差（对每个 v 的 mean 再取平均）: {err_stats_agg['H_chiral']['mean']:.6e}",
        f"- block-off-diagonal 重构误差 (max/mean): {err_stats_agg['block']['max']:.6e} / {err_stats_agg['block']['mean']:.6e}",
        "",
        "## 4) Q(k) block 输出",
        f"- 已输出: `{q_path.name}`",
        "",
        "## 5) 5x5 OBC 谱比较设置",
        f"- Lx={lx}, Ly={ly}, t={t:.3f}, w={w:.3f}",
        f"- v_list={v_values}",
        f"- alpha_list={alpha_values}",
        f"- 零能阈值: |E| <= {zero_tol:.1e}",
        f"- 近零窗口: |E| <= {near_zero_window:.1e}",
        "",
        "## 6) 关键数值结论（每个 v）",
    ]
    lines += ["", "### 6.1 Chiral error per v"]
    for v in v_values:
        h0_row = next(
            r
            for r in err_scan_rows
            if r["model"] == "H0" and abs(float(r["v_ref"]) - v) < 1e-12
        )
        hc_row = next(
            r
            for r in err_scan_rows
            if r["model"] == "H_chiral" and abs(float(r["v_ref"]) - v) < 1e-12
        )
        lines.append(
            f"- v={v:.2f}: H0(max/mean)=({float(h0_row['max_error']):.3e}, {float(h0_row['mean_error']):.3e}), "
            f"H_chiral(max/mean)=({float(hc_row['max_error']):.3e}, {float(hc_row['mean_error']):.3e})"
        )
    lines += ["", "### 6.2 OBC near-zero summary per v"]

    for v in v_values:
        row_a0 = get_summary(v, 0.0)
        row_a1 = get_summary(v, 1.0)
        min_a0 = float(row_a0["min_abs_energy"])
        min_a1 = float(row_a1["min_abs_energy"])
        n0_a0 = int(row_a0["n_zero_tol"])
        n0_a1 = int(row_a1["n_zero_tol"])
        nn_a0 = int(row_a0["n_near_window"])
        nn_a1 = int(row_a1["n_near_window"])
        lines.append(
            f"- v={v:.2f}: alpha=0 -> min|E|={min_a0:.6e}, n_zero={n0_a0}, n_near={nn_a0}; "
            f"alpha=1 -> min|E|={min_a1:.6e}, n_zero={n0_a1}, n_near={nn_a1}"
        )

    lines += [
        "",
        "## 7) 近零态局域性（alpha=0）",
    ]
    for v in v_values:
        sub = [
            r
            for r in metric_rows
            if abs(float(r["v"]) - v) < 1e-12 and abs(float(r["alpha"]) - 0.0) < 1e-12 and int(r["rank_by_absE"]) < 4
        ]
        wc = float(np.mean([float(r["W_corner"]) for r in sub])) if sub else float("nan")
        we = float(np.mean([float(r["W_edge"]) for r in sub])) if sub else float("nan")
        wb = float(np.mean([float(r["W_bulk"]) for r in sub])) if sub else float("nan")
        label = "corner-dominant" if wc > we and wc > wb else ("edge-dominant" if we > wb else "bulk/mixed")
        lines.append(f"- v={v:.2f}: <W_corner, W_edge, W_bulk> = ({wc:.3f}, {we:.3f}, {wb:.3f}) -> {label}")

    lines += [
        "",
        "## 8) 对你提出的四个判断",
        "1. alpha=0 是否有零能态：见 `zero_mode_summary.csv` 中 n_zero_tol 与 min|E|。",
        "2. 这些近零态是否角局域：见 `near_zero_state_metrics.csv` 与波函数热图。",
        "3. alpha 从 0 到 1 是否偏离零能：见 `min_abs_energy_vs_alpha.png`。",
        "4. 若偏离，说明 H_break 破坏零能钉扎：由 alpha 依赖趋势直接判断。",
        "",
        "## 9) 输出文件索引",
        "- `hopping_chiral_classification.csv`",
        "- `chiral_error_map.csv`, `chiral_symmetry_error_scan.csv`",
        "- `Q_block_expression.md`",
        "- `obc_eigenvalues_all.csv`, `zero_mode_summary.csv`, `near_zero_state_metrics.csv`",
        "- `spectra/*.png`, `wavefunctions/*.png`, `min_abs_energy_vs_alpha.png`, `near_zero_corner_weight_vs_alpha.png`",
    ]

    # Keep report self-contained with explicit classification dump.
    lines += ["", "## Appendix: hopping classification rows"]
    for row in class_rows:
        lines.append(
            f"- ({int(row['from_orbital'])},{int(row['to_orbital'])}) "
            f"{row['hopping_expression']} -> {row['chiral_classification']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    spectra_dir = out_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    wavefunc_dir = out_root / "wavefunctions"
    wavefunc_dir.mkdir(parents=True, exist_ok=True)

    lx, ly = int(args.Lx), int(args.Ly)
    t, w = float(args.t), float(args.w)
    v_list = parse_float_list(args.v_list)
    alpha_list = parse_float_list(args.alpha_list)
    nk = int(args.nk)
    near_count = int(args.near_count)
    zero_tol = float(args.zero_tol)
    near_zero_window = float(args.near_zero_window)

    if 0.0 not in alpha_list:
        alpha_list = [0.0] + alpha_list
    if 1.0 not in alpha_list:
        alpha_list = alpha_list + [1.0]
    alpha_list = sorted(set(alpha_list))

    # 1) BZ chiral-error scan for each v.
    err_scan_rows: list[dict[str, float | int | str]] = []
    h0_max_list: list[float] = []
    h0_mean_list: list[float] = []
    hc_max_list: list[float] = []
    hc_mean_list: list[float] = []
    block_max_list: list[float] = []
    block_mean_list: list[float] = []

    for v_ref in v_list:
        err_stats = scan_chiral_errors(
            nk=nk,
            v_ref=float(v_ref),
            t=t,
            w=w,
            out_csv=out_root / f"chiral_error_map_v_{token(v_ref)}.csv",
        )
        h0_max_list.append(err_stats["H0"]["max"])
        h0_mean_list.append(err_stats["H0"]["mean"])
        hc_max_list.append(err_stats["H_chiral"]["max"])
        hc_mean_list.append(err_stats["H_chiral"]["mean"])
        block_max_list.append(err_stats["block"]["max"])
        block_mean_list.append(err_stats["block"]["mean"])
        err_scan_rows.extend(
            [
                {
                    "model": "H0",
                    "v_ref": float(v_ref),
                    "nk": nk,
                    "max_error": err_stats["H0"]["max"],
                    "mean_error": err_stats["H0"]["mean"],
                },
                {
                    "model": "H_chiral",
                    "v_ref": float(v_ref),
                    "nk": nk,
                    "max_error": err_stats["H_chiral"]["max"],
                    "mean_error": err_stats["H_chiral"]["mean"],
                },
            ]
        )

    err_stats_agg = {
        "H0": {"max": float(np.max(h0_max_list)), "mean": float(np.mean(h0_mean_list))},
        "H_chiral": {"max": float(np.max(hc_max_list)), "mean": float(np.mean(hc_mean_list))},
        "block": {"max": float(np.max(block_max_list)), "mean": float(np.mean(block_mean_list))},
    }

    # Keep one representative chiral map filename for convenience.
    repr_map = out_root / "chiral_error_map.csv"
    if (out_root / "chiral_error_map_v_0p70.csv").exists():
        src = out_root / "chiral_error_map_v_0p70.csv"
    else:
        src = out_root / f"chiral_error_map_v_{token(v_list[0])}.csv"
    repr_map.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    with (out_root / "chiral_symmetry_error_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "v_ref", "nk", "max_error", "mean_error"],
        )
        writer.writeheader()
        writer.writerows(err_scan_rows)

    # 2) hopping classification.
    class_rows = build_hopping_classification(out_root / "hopping_chiral_classification.csv")

    # 3/4) write Q block expression and verify exact anticommutation at sample points.
    q_path = out_root / "Q_block_expression.md"
    write_q_block_expression(q_path)

    q_sample_rows = []
    sample_points = {
        "Gamma": (0.0, 0.0),
        "X": (np.pi, 0.0),
        "Y": (0.0, np.pi),
        "M": (np.pi, np.pi),
    }
    for label, (kx, ky) in sample_points.items():
        h = h_chiral_k(kx, ky, v_list[0], t, w)
        q = h[:2, 2:4]
        err = chiral_error(h)
        q_sample_rows.append(
            {
                "k_label": label,
                "kx": kx,
                "ky": ky,
                "chiral_error": err,
                "Q00_real": float(np.real(q[0, 0])),
                "Q00_imag": float(np.imag(q[0, 0])),
                "Q01_real": float(np.real(q[0, 1])),
                "Q01_imag": float(np.imag(q[0, 1])),
                "Q10_real": float(np.real(q[1, 0])),
                "Q10_imag": float(np.imag(q[1, 0])),
                "Q11_real": float(np.real(q[1, 1])),
                "Q11_imag": float(np.imag(q[1, 1])),
            }
        )
    with (out_root / "Q_block_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "k_label",
                "kx",
                "ky",
                "chiral_error",
                "Q00_real",
                "Q00_imag",
                "Q01_real",
                "Q01_imag",
                "Q10_real",
                "Q10_imag",
                "Q11_real",
                "Q11_imag",
            ],
        )
        writer.writeheader()
        writer.writerows(q_sample_rows)

    # 5/6/7) OBC spectra + near-zero wavefunctions/metrics for all (v, alpha).
    eig_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float | int]] = []
    metric_rows: list[dict[str, float | int]] = []
    spectra_by_v: dict[float, dict[float, np.ndarray]] = {v: {} for v in v_list}

    for v in v_list:
        for alpha in alpha_list:
            ham = build_obc_hamiltonian(lx=lx, ly=ly, v=v, t=t, w=w, alpha=alpha)
            evals, evecs = np.linalg.eigh(ham)
            evals = np.real(evals)
            spectra_by_v[v][alpha] = evals.copy()

            for n, e in enumerate(evals):
                eig_rows.append(
                    {
                        "v": v,
                        "alpha": alpha,
                        "state_index": n,
                        "energy": float(e),
                    }
                )

            title = f"OBC spectrum L={lx}x{ly}, v={v:.2f}, alpha={alpha:.2f}, t={t:.2f}, w={w:.2f}"
            spec_path = spectra_dir / f"spectrum_v_{token(v)}_alpha_{token(alpha)}.png"
            plot_spectrum(spec_path, evals, title)

            abs_order = np.argsort(np.abs(evals))
            near_indices = abs_order[:near_count]
            near_indices = near_indices[np.argsort(np.abs(evals[near_indices]))]

            min_abs_energy = float(np.min(np.abs(evals)))
            n_zero = int(np.sum(np.abs(evals) <= zero_tol))
            n_near = int(np.sum(np.abs(evals) <= near_zero_window))
            summary_rows.append(
                {
                    "v": v,
                    "alpha": alpha,
                    "min_abs_energy": min_abs_energy,
                    "n_zero_tol": n_zero,
                    "n_near_window": n_near,
                    "closest_state_index": int(abs_order[0]),
                    "closest_state_energy": float(evals[abs_order[0]]),
                }
            )

            for rank, sidx in enumerate(near_indices):
                vec = evecs[:, int(sidx)]
                rho = cell_prob_density(vec, lx=lx, ly=ly)
                w_corner, w_edge, w_bulk, ipr = corner_edge_bulk_weights(rho)
                e = float(evals[int(sidx)])
                metric_rows.append(
                    {
                        "v": v,
                        "alpha": alpha,
                        "state_index": int(sidx),
                        "rank_by_absE": rank,
                        "energy": e,
                        "abs_energy": abs(e),
                        "W_corner": w_corner,
                        "W_edge": w_edge,
                        "W_bulk": w_bulk,
                        "IPR": ipr,
                    }
                )
                wf_path = wavefunc_dir / f"wf_v_{token(v)}_alpha_{token(alpha)}_state_{int(sidx)}.png"
                wf_title = (
                    f"v={v:.2f}, alpha={alpha:.2f}, state={int(sidx)}, E={e:.3e}\n"
                    f"Wc={w_corner:.3f}, We={w_edge:.3f}, Wb={w_bulk:.3f}, IPR={ipr:.3f}"
                )
                plot_wavefunction(wf_path, rho, wf_title)

    # Write tables.
    with (out_root / "obc_eigenvalues_all.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["v", "alpha", "state_index", "energy"])
        writer.writeheader()
        writer.writerows(eig_rows)

    with (out_root / "zero_mode_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "alpha",
                "min_abs_energy",
                "n_zero_tol",
                "n_near_window",
                "closest_state_index",
                "closest_state_energy",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (out_root / "near_zero_state_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "alpha",
                "state_index",
                "rank_by_absE",
                "energy",
                "abs_energy",
                "W_corner",
                "W_edge",
                "W_bulk",
                "IPR",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    # Spectrum montages per v and trend plots.
    for v in v_list:
        plot_spectrum_montage(
            out_root / f"spectrum_montage_v_{token(v)}.png",
            spectra_by_alpha=spectra_by_v[v],
            v=v,
        )

    plot_min_abs_energy_vs_alpha(out_root / "min_abs_energy_vs_alpha.png", summary_rows)
    plot_corner_weight_vs_alpha(out_root / "near_zero_corner_weight_vs_alpha.png", metric_rows)

    # 8) Report with direct answers.
    write_report(
        path=out_root / "CHIRAL_SYMMETRY_ZERO_MODE_REPORT.md",
        err_scan_rows=err_scan_rows,
        err_stats_agg=err_stats_agg,
        q_path=q_path,
        class_rows=class_rows,
        summary_rows=summary_rows,
        metric_rows=metric_rows,
        lx=lx,
        ly=ly,
        t=t,
        w=w,
        nk=nk,
        zero_tol=zero_tol,
        near_zero_window=near_zero_window,
    )

    print(f"[ok] output_root={out_root}")
    print(f"[ok] chiral_error_H0_max={err_stats_agg['H0']['max']:.6e} mean={err_stats_agg['H0']['mean']:.6e}")
    print(
        f"[ok] chiral_error_Hchiral_max={err_stats_agg['H_chiral']['max']:.6e} "
        f"mean={err_stats_agg['H_chiral']['mean']:.6e}"
    )


if __name__ == "__main__":
    main()
