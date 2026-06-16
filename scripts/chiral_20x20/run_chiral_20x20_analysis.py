#!/usr/bin/env python3
"""20x20 OBC spectrum and near-zero-state analysis for chiral/original models."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigsh


GAMMA_CELL = np.diag([1.0, 1.0, -1.0, -1.0]).astype(float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 20x20 chiral/original OBC analysis.")
    parser.add_argument("--output-root", default="/workspace/outputs/chiral_20x20")
    parser.add_argument("--Lx", type=int, default=20)
    parser.add_argument("--Ly", type=int, default=20)
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--v-list", default="0.5,0.6")
    parser.add_argument("--alpha-list", default="0.0,1.0")
    parser.add_argument("--near-k", type=int, default=40)
    parser.add_argument("--near-state-count", type=int, default=8)
    parser.add_argument("--corner-size", type=int, default=3)
    parser.add_argument("--edge-width", type=int, default=3)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def token_v(v: float) -> str:
    return f"{v:.1f}".replace("-", "m").replace(".", "p")


def token_alpha(alpha: float) -> str:
    if abs(alpha - round(alpha)) < 1e-12:
        return str(int(round(alpha)))
    return f"{alpha:.2f}".replace("-", "m").replace(".", "p")


def idx(ix: int, iy: int, orb: int, lx: int) -> int:
    return ((iy * lx + ix) * 4) + orb


def add_hop(h: np.ndarray, i: int, j: int, amp: float) -> None:
    h[i, j] += amp
    h[j, i] += amp


def build_h0_obc(lx: int, ly: int, v: float, t: float, w: float) -> np.ndarray:
    """Build original-model OBC Hamiltonian from requested hopping rules."""
    n = lx * ly * 4
    h = np.zeros((n, n), dtype=float)

    for iy in range(ly):
        for ix_ in range(lx):
            i0 = idx(ix_, iy, 0, lx)
            i1 = idx(ix_, iy, 1, lx)
            i2 = idx(ix_, iy, 2, lx)
            i3 = idx(ix_, iy, 3, lx)

            # intra-cell hoppings
            add_hop(h, i0, i1, t)  # 0-1: t
            add_hop(h, i0, i2, t)  # 0-2: t
            add_hop(h, i1, i3, t)  # 1-3: t
            add_hop(h, i2, i3, t)  # 2-3: t
            add_hop(h, i0, i3, v)  # 0-3: v
            add_hop(h, i1, i2, v)  # 1-2: v

            # inter-cell hoppings (OBC)
            if ix_ > 0:
                add_hop(h, i0, idx(ix_ - 1, iy, 3, lx), w)  # x-direction: 0 <-> left 3
            if iy > 0:
                add_hop(h, i1, idx(ix_, iy - 1, 2, lx), w)  # y-direction: 1 <-> down 2

    return h


def gamma_obc(lx: int, ly: int) -> np.ndarray:
    return np.kron(np.eye(lx * ly, dtype=float), GAMMA_CELL).astype(float)


def cell_density(vec: np.ndarray, lx: int, ly: int) -> np.ndarray:
    rho = np.zeros((ly, lx), dtype=float)
    for iy in range(ly):
        for ix_ in range(lx):
            s = idx(ix_, iy, 0, lx)
            rho[iy, ix_] = float(np.sum(np.abs(vec[s : s + 4]) ** 2))
    return rho


def region_weights(
    rho: np.ndarray,
    corner_size: int,
    edge_width: int,
) -> tuple[float, float, float, float]:
    ly, lx = rho.shape
    corner_mask = np.zeros((ly, lx), dtype=bool)
    corner_mask[:corner_size, :corner_size] = True
    corner_mask[:corner_size, lx - corner_size :] = True
    corner_mask[ly - corner_size :, :corner_size] = True
    corner_mask[ly - corner_size :, lx - corner_size :] = True

    edge_mask = np.zeros((ly, lx), dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[ly - edge_width :, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, lx - edge_width :] = True
    edge_mask &= ~corner_mask

    bulk_mask = ~(corner_mask | edge_mask)

    w_corner = float(np.sum(rho[corner_mask]))
    w_edge = float(np.sum(rho[edge_mask]))
    w_bulk = float(np.sum(rho[bulk_mask]))
    ipr = float(np.sum(rho**2))
    return w_corner, w_edge, w_bulk, ipr


def spectrum_symmetry_error(evals: np.ndarray) -> float:
    e = np.array(evals, dtype=float)
    return float(np.max(np.abs(e + e[::-1])))


def nearzero_counts(evals: np.ndarray) -> dict[str, int]:
    abs_e = np.abs(evals)
    return {
        "count_absE_lt_1e-2": int(np.sum(abs_e < 1e-2)),
        "count_absE_lt_1e-4": int(np.sum(abs_e < 1e-4)),
        "count_absE_lt_1e-6": int(np.sum(abs_e < 1e-6)),
        "count_absE_lt_1e-8": int(np.sum(abs_e < 1e-8)),
    }


def plot_spectrum_full(
    out_path: Path,
    evals: np.ndarray,
    title_lines: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(evals.size, dtype=int)
    ax.scatter(x, evals, s=7, color="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("state index")
    ax.set_ylabel("energy")
    ax.set_title("\n".join(title_lines), fontsize=9)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_spectrum_zoom(
    out_path: Path,
    evals: np.ndarray,
    title_lines: list[str],
    window: float = 0.05,
) -> None:
    n = evals.size
    mid = n // 2
    i0 = max(0, mid - 20)
    i1 = min(n, mid + 20)

    mask = np.abs(evals) < window
    if np.sum(mask) >= 8:
        x = np.where(mask)[0]
        y = evals[mask]
    else:
        x = np.arange(i0, i1)
        y = evals[i0:i1]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(x, y, s=14, color="tab:orange")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("state index")
    ax.set_ylabel("energy")
    ax.set_title("\n".join(title_lines), fontsize=9)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_wavefunction(
    out_path: Path,
    rho: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(rho, origin="lower", cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="rho(x,y)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    lx: int,
    ly: int,
    v: float,
    alpha: float,
    t: float,
    w: float,
    near_k: int,
    near_state_count: int,
    corner_size: int,
    edge_width: int,
    out_root: Path,
    gamma_full: np.ndarray,
) -> tuple[dict, list[dict], list[dict]]:
    h0 = build_h0_obc(lx=lx, ly=ly, v=v, t=t, w=w)
    h_chiral = 0.5 * (h0 - gamma_full @ h0 @ gamma_full)
    h_break = 0.5 * (h0 + gamma_full @ h0 @ gamma_full)
    h = h_chiral + alpha * h_break

    n = h.shape[0]
    if n != lx * ly * 4:
        raise RuntimeError(f"matrix size mismatch: got {n}, expected {lx*ly*4}")
    if not np.allclose(h, h.T, atol=1e-12):
        raise RuntimeError("Hamiltonian is not Hermitian; check hopping/H.c. construction.")

    ch_err = float(np.linalg.norm(gamma_full @ h @ gamma_full + h) / max(np.linalg.norm(h), 1e-15))

    # Prefer sparse near-zero solve first as requested.
    eigsh_method = "eigsh_shift_invert"
    try:
        _vals_sparse, _vecs_sparse = eigsh(csr_matrix(h), k=min(near_k, n - 2), sigma=0.0, which="LM")
    except (ArpackNoConvergence, RuntimeError, ValueError):
        eigsh_method = "dense_fallback"

    # Full dense spectrum/eigenvectors for complete outputs.
    evals, evecs = np.linalg.eigh(h)
    evals = np.real(evals)
    sym_err = spectrum_symmetry_error(evals)
    min_abs_e = float(np.min(np.abs(evals)))
    counts = nearzero_counts(evals)

    # Mandatory strict checks for alpha=0.
    if abs(alpha) < 1e-14:
        if ch_err > 1e-10 or sym_err > 1e-8:
            raise RuntimeError(
                "alpha=0 symmetry check failed. Verify: x-direction w hopping, y-direction w hopping, "
                "Hermitian completion, and Gamma_OBC orbital ordering."
            )

    # Near-zero states from full spectrum.
    near_idx = np.argsort(np.abs(evals))[:near_state_count]
    near_idx = near_idx[np.argsort(np.abs(evals[near_idx]))]

    rows_eigs: list[dict] = []
    for i, e in enumerate(evals):
        rows_eigs.append({"v": v, "alpha": alpha, "state_index": i, "energy": float(e)})

    rows_near: list[dict] = []
    for rank, sidx in enumerate(near_idx, start=1):
        vec = evecs[:, int(sidx)]
        rho = cell_density(vec, lx=lx, ly=ly)
        wc, we, wb, ipr = region_weights(rho, corner_size=corner_size, edge_width=edge_width)
        e = float(evals[int(sidx)])
        rows_near.append(
            {
                "v": v,
                "alpha": alpha,
                "state_index": int(sidx),
                "energy": e,
                "abs_energy": abs(e),
                "W_corner": wc,
                "W_edge": we,
                "W_bulk": wb,
                "IPR": ipr,
            }
        )
        wf_path = out_root / f"wf_v{token_v(v)}_alpha{token_alpha(alpha)}_state_nearzero_{rank:02d}.png"
        title = (
            f"v={v:.1f}, alpha={alpha:.1f}, state={int(sidx)}, E={e:.3e}\n"
            f"Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f}, IPR={ipr:.3f}"
        )
        plot_wavefunction(wf_path, rho, title)

    title_lines = [
        f"L={lx}x{ly}, v={v:.1f}, alpha={alpha:.1f}",
        f"chiral_error={ch_err:.3e}, spectrum_symmetry_error={sym_err:.3e}",
        f"min_abs_E={min_abs_e:.3e}, counts(<1e-4)={counts['count_absE_lt_1e-4']}, eigsh={eigsh_method}",
    ]
    full_png = out_root / f"spectrum_full_v{token_v(v)}_alpha{token_alpha(alpha)}.png"
    zoom_png = out_root / f"spectrum_zoom_v{token_v(v)}_alpha{token_alpha(alpha)}.png"
    plot_spectrum_full(full_png, evals, title_lines)
    plot_spectrum_zoom(zoom_png, evals, title_lines)

    summary = {
        "v": v,
        "alpha": alpha,
        "chiral_error": ch_err,
        "spectrum_symmetry_error": sym_err,
        "min_abs_E": min_abs_e,
        "count_absE_lt_1e-2": counts["count_absE_lt_1e-2"],
        "count_absE_lt_1e-4": counts["count_absE_lt_1e-4"],
        "count_absE_lt_1e-6": counts["count_absE_lt_1e-6"],
        "count_absE_lt_1e-8": counts["count_absE_lt_1e-8"],
        "eigsh_method": eigsh_method,
    }
    return summary, rows_eigs, rows_near


def make_report(
    out_path: Path,
    summary_rows: list[dict],
    near_rows: list[dict],
    baseline_5x5_path: Path,
) -> None:
    s_map = {(float(r["v"]), float(r["alpha"])): r for r in summary_rows}

    # Compare with previous 5x5 min|E| when available.
    baseline_map: dict[tuple[float, float], float] = {}
    if baseline_5x5_path.exists():
        with baseline_5x5_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                baseline_map[(float(row["v"]), float(row["alpha"]))] = float(row["min_abs_energy"])

    def near_group(v: float, alpha: float) -> list[dict]:
        sub = [r for r in near_rows if abs(float(r["v"]) - v) < 1e-12 and abs(float(r["alpha"]) - alpha) < 1e-12]
        return sorted(sub, key=lambda r: float(r["abs_energy"]))

    lines: list[str] = [
        "# CHIRAL_20X20_REPORT",
        "",
        "## Summary table",
    ]
    for r in summary_rows:
        lines.append(
            f"- v={float(r['v']):.1f}, alpha={float(r['alpha']):.1f}: "
            f"chiral_error={float(r['chiral_error']):.3e}, "
            f"sym_err={float(r['spectrum_symmetry_error']):.3e}, "
            f"min|E|={float(r['min_abs_E']):.3e}, "
            f"counts(<1e-2/1e-4/1e-6/1e-8)="
            f"{int(r['count_absE_lt_1e-2'])}/{int(r['count_absE_lt_1e-4'])}/"
            f"{int(r['count_absE_lt_1e-6'])}/{int(r['count_absE_lt_1e-8'])}"
        )

    # Question-by-question answers
    lines += [
        "",
        "## Questions",
    ]

    a0_cases = [r for r in summary_rows if abs(float(r["alpha"])) < 1e-12]
    alpha0_ok = all(float(r["chiral_error"]) < 1e-12 for r in a0_cases)
    lines.append(f"1) alpha=0 是否严格满足手征对称性：{'是' if alpha0_ok else '否'}。")

    alpha0_spec_ok = all(float(r["spectrum_symmetry_error"]) <= 1e-10 for r in a0_cases)
    lines.append(f"2) alpha=0 全谱是否关于 E=0 对称：{'是' if alpha0_spec_ok else '否'}。")

    # Better-than-5x5 min|E| comparison for alpha=0.
    better_flags = []
    for r in a0_cases:
        key = (float(r["v"]), float(r["alpha"]))
        if key in baseline_map:
            better_flags.append(float(r["min_abs_E"]) < baseline_map[key])
    if better_flags:
        q3 = any(better_flags)
        lines.append(f"3) 20x20 是否比 5x5 更接近零能：{'是' if q3 else '否'}（以 min|E| 比较）。")
    else:
        lines.append("3) 20x20 是否比 5x5 更接近零能：缺少 5x5 基线文件，无法直接比较。")

    # Corner localization for alpha=0.
    def is_corner_like(rr: dict) -> bool:
        return float(rr["W_corner"]) > 0.6 and float(rr["W_corner"]) > float(rr["W_edge"])

    corner_like_alpha0 = []
    for v in [0.5, 0.6]:
        grp = near_group(v, 0.0)
        frac = float(np.mean([1.0 if is_corner_like(r) else 0.0 for r in grp])) if grp else 0.0
        corner_like_alpha0.append((v, frac))
    lines.append(
        "4) 近零态是否主要局域在四角："
        + ", ".join([f"v={v:.1f} corner-like比例={frac:.2f}" for v, frac in corner_like_alpha0])
        + "。"
    )

    # Which v clearer corner states (alpha=0).
    score_v = {}
    for v in [0.5, 0.6]:
        grp = near_group(v, 0.0)
        if grp:
            score_v[v] = float(np.mean([float(r["W_corner"]) - float(r["W_edge"]) for r in grp]))
    if len(score_v) == 2:
        clearer = max(score_v, key=score_v.get)
        lines.append(f"5) v=0.5 与 v=0.6 哪个角态更清晰：v={clearer:.1f}（按 <W_corner-W_edge>）。")
    else:
        lines.append("5) v=0.5 与 v=0.6 哪个角态更清晰：数据不足。")

    # alpha=1 zero pinning and morphology.
    a1_cases = [r for r in summary_rows if abs(float(r["alpha"]) - 1.0) < 1e-12]
    pinning_lost = all(float(r["min_abs_E"]) > 1e-6 for r in a1_cases)
    lines.append(f"6) alpha=1 原始模型中零能钉扎是否消失：{'是' if pinning_lost else '否'}。")

    mix_desc = []
    for v in [0.5, 0.6]:
        grp = near_group(v, 1.0)
        if not grp:
            continue
        wc = float(np.mean([float(r["W_corner"]) for r in grp]))
        we = float(np.mean([float(r["W_edge"]) for r in grp]))
        wb = float(np.mean([float(r["W_bulk"]) for r in grp]))
        if we > wc:
            tag = "edge-like/mixed"
        elif wc > 0.6 and wc > we:
            tag = "corner-like"
        else:
            tag = "mixed"
        mix_desc.append(f"v={v:.1f}: {tag} (Wc={wc:.3f}, We={we:.3f}, Wb={wb:.3f})")
    lines.append("7) alpha=1 近零态形态：" + "; ".join(mix_desc) + "。")

    if alpha0_ok and alpha0_spec_ok and pinning_lost:
        conclusion = (
            "支持“手征对称极限存在准零能角态，而原始模型中零能角态不受保护”的总体结论；"
            "但若 W_edge 同时较大，应表述为 corner-like / edge-mixed。"
        )
    else:
        conclusion = "暂不支持该结论，需先修复对称性或零能钉扎检查。"
    lines.append(f"8) 是否支持目标结论：{conclusion}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    lx = int(args.Lx)
    ly = int(args.Ly)
    t = float(args.t)
    w = float(args.w)
    v_list = parse_float_list(args.v_list)
    alpha_list = parse_float_list(args.alpha_list)
    near_k = int(args.near_k)
    near_state_count = int(args.near_state_count)
    corner_size = int(args.corner_size)
    edge_width = int(args.edge_width)

    gamma_full = gamma_obc(lx=lx, ly=ly)

    summary_rows: list[dict] = []
    eig_rows: list[dict] = []
    near_rows: list[dict] = []

    for v in v_list:
        for alpha in alpha_list:
            summary, rows_e, rows_n = run_case(
                lx=lx,
                ly=ly,
                v=v,
                alpha=alpha,
                t=t,
                w=w,
                near_k=near_k,
                near_state_count=near_state_count,
                corner_size=corner_size,
                edge_width=edge_width,
                out_root=out_root,
                gamma_full=gamma_full,
            )
            summary_rows.append(summary)
            eig_rows.extend(rows_e)
            near_rows.extend(rows_n)

    # Output CSV files
    write_csv(
        out_root / "eigenvalues_20x20.csv",
        eig_rows,
        fieldnames=["v", "alpha", "state_index", "energy"],
    )
    write_csv(
        out_root / "nearzero_metrics_20x20.csv",
        near_rows,
        fieldnames=["v", "alpha", "state_index", "energy", "abs_energy", "W_corner", "W_edge", "W_bulk", "IPR"],
    )
    write_csv(
        out_root / "summary_20x20.csv",
        summary_rows,
        fieldnames=[
            "v",
            "alpha",
            "chiral_error",
            "spectrum_symmetry_error",
            "min_abs_E",
            "count_absE_lt_1e-2",
            "count_absE_lt_1e-4",
            "count_absE_lt_1e-6",
            "count_absE_lt_1e-8",
            "eigsh_method",
        ],
    )

    make_report(
        out_path=out_root / "CHIRAL_20X20_REPORT.md",
        summary_rows=summary_rows,
        near_rows=near_rows,
        baseline_5x5_path=Path("/workspace/outputs/chiral_symmetric_model/zero_mode_summary.csv"),
    )

    print(f"[ok] output_root={out_root}")
    for row in summary_rows:
        print(
            "[ok] "
            f"v={float(row['v']):.1f} alpha={float(row['alpha']):.1f} "
            f"chiral_error={float(row['chiral_error']):.3e} "
            f"sym_err={float(row['spectrum_symmetry_error']):.3e} "
            f"min_abs_E={float(row['min_abs_E']):.3e}"
        )


if __name__ == "__main__":
    main()
