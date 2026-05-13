from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian_sparse

matplotlib.use("Agg")


def load_points(summary_csv: Path) -> list[dict[str, float | str]]:
    """Load first-stage points from band/ribbon summary table."""
    rows: list[dict[str, float | str]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "point_id": row["point_id"],
                    "v": float(row["v"]),
                    "t": float(row["t"]),
                    "lm": float(row["lm"]),
                    "w": float(row["w"]),
                }
            )
    return rows


def full_dense_eigvals(v: float, t: float, lm: float, w: float, nx: int, ny: int) -> np.ndarray:
    """Build OBC Hamiltonian and compute full Hermitian spectrum."""
    ham_sparse = build_obc_hamiltonian_sparse(v=v, t=t, lm=lm, w=w, j=0.0, nx=nx, ny=ny)
    eigvals = np.linalg.eigvalsh(ham_sparse.toarray())
    return np.real(eigvals)


def compute_obc_metrics(eigvals: np.ndarray) -> dict[str, float | int]:
    """Extract near-zero and gap descriptors from full E-vs-index spectrum."""
    eigvals = np.sort(np.real(eigvals))
    abs_e = np.abs(eigvals)
    min_abs = float(np.min(abs_e))
    cnt_0p01 = int(np.sum(abs_e <= 0.01))
    cnt_0p02 = int(np.sum(abs_e <= 0.02))
    cnt_0p05 = int(np.sum(abs_e <= 0.05))

    neg = eigvals[eigvals < 0]
    pos = eigvals[eigvals > 0]
    if neg.size > 0 and pos.size > 0:
        obc_gap = float(np.min(pos) - np.max(neg))
    else:
        obc_gap = 0.0

    # Larger score means more "special" in E-vs-index:
    # dense near-zero states + tiny min|E| + tiny OBC gap.
    score = (
        2.5 * cnt_0p02
        + 1.0 * cnt_0p05
        + 0.08 / (min_abs + 1e-7)
        + 0.06 / (abs(obc_gap) + 1e-7)
    )
    return {
        "min_abs_energy": min_abs,
        "obc_gap": obc_gap,
        "count_absE_le_0p01": cnt_0p01,
        "count_absE_le_0p02": cnt_0p02,
        "count_absE_le_0p05": cnt_0p05,
        "special_score": float(score),
    }


def mark_red_spectrum(
    eigvals: np.ndarray,
    point_id: str,
    save_path: Path,
    min_abs_energy: float,
    abs_floor: float,
    scale: float,
) -> float:
    """Plot full OBC E-vs-index and highlight near-zero special states in red."""
    eigvals = np.sort(np.real(eigvals))
    highlight_window = max(abs_floor, scale * min_abs_energy)
    idx = np.arange(eigvals.size)
    red_mask = np.abs(eigvals) <= highlight_window

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(idx[~red_mask], eigvals[~red_mask], s=3, c="#7f7f7f", alpha=0.75, label="other states")
    ax.scatter(idx[red_mask], eigvals[red_mask], s=7, c="red", alpha=0.95, label="special states (red)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Energy")
    ax.set_title(f"OBC E vs index (special marked)\n{point_id}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    return float(highlight_window)


def plot_special_map(rows: list[dict[str, float | int | str]], save_path: Path) -> None:
    """Create a parameter map with special points highlighted in red."""
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    lm = np.array([float(r["lm"]) for r in rows], dtype=float)
    special = np.array([int(r["is_special"]) for r in rows], dtype=int)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(v[special == 0], lm[special == 0], c="#bdbdbd", s=80, marker="s", edgecolors="none", label="normal")
    ax.scatter(
        v[special == 1],
        lm[special == 1],
        c="red",
        s=95,
        marker="s",
        edgecolors="black",
        linewidths=0.4,
        label="special (red)",
    )
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title("Special points from OBC E-index spectra (red)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    summary_csv = root / args.summary_csv
    points_root = root / args.points_root
    output_csv = root / args.output_csv
    output_txt = root / args.output_txt
    output_map = root / args.output_map_png

    points = load_points(summary_csv)
    if not points:
        raise RuntimeError(f"No valid points found in {summary_csv}")

    analysis_rows: list[dict[str, float | int | str]] = []
    eigvals_by_point: dict[str, np.ndarray] = {}

    total = len(points)
    for i, p in enumerate(points, start=1):
        point_id = str(p["point_id"])
        eigvals = full_dense_eigvals(
            v=float(p["v"]),
            t=float(p["t"]),
            lm=float(p["lm"]),
            w=float(p["w"]),
            nx=args.obc_nx,
            ny=args.obc_ny,
        )
        metrics = compute_obc_metrics(eigvals)
        row: dict[str, float | int | str] = {
            "point_id": point_id,
            "v": float(p["v"]),
            "t": float(p["t"]),
            "lm": float(p["lm"]),
            "w": float(p["w"]),
            "obc_nx": int(args.obc_nx),
            "obc_ny": int(args.obc_ny),
            "eigvals_count": int(eigvals.size),
            **metrics,
            "special_rank": "",
            "is_special": 0,
            "highlight_window_used": "",
        }
        analysis_rows.append(row)
        eigvals_by_point[point_id] = eigvals
        if i % 10 == 0 or i == total:
            print(f"processed {i}/{total}")

    # pick top-N points by special_score as "special"
    sorted_idx = sorted(range(len(analysis_rows)), key=lambda i: float(analysis_rows[i]["special_score"]), reverse=True)
    top_n = max(1, min(args.top_n, len(sorted_idx)))
    special_ids = set()
    for rank, idx in enumerate(sorted_idx[:top_n], start=1):
        analysis_rows[idx]["special_rank"] = rank
        analysis_rows[idx]["is_special"] = 1
        special_ids.add(str(analysis_rows[idx]["point_id"]))

    # mark red directly on each special point's E-index plot
    for row in analysis_rows:
        if int(row["is_special"]) != 1:
            continue
        point_id = str(row["point_id"])
        point_dir = points_root / point_id
        marked_path = point_dir / "obc_spectrum_e_vs_index_marked_red.png"
        window = mark_red_spectrum(
            eigvals=eigvals_by_point[point_id],
            point_id=point_id,
            save_path=marked_path,
            min_abs_energy=float(row["min_abs_energy"]),
            abs_floor=args.highlight_abs_floor,
            scale=args.highlight_scale,
        )
        row["highlight_window_used"] = window

    # write outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "w",
        "obc_nx",
        "obc_ny",
        "eigvals_count",
        "min_abs_energy",
        "obc_gap",
        "count_absE_le_0p01",
        "count_absE_le_0p02",
        "count_absE_le_0p05",
        "special_score",
        "special_rank",
        "is_special",
        "highlight_window_used",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(analysis_rows)

    plot_special_map(analysis_rows, output_map)

    lines = [
        "Special-point analysis from OBC E-vs-index spectra",
        f"total_points={len(analysis_rows)}",
        f"special_top_n={top_n}",
        f"obc_size={args.obc_nx}x{args.obc_ny}",
        f"highlight_window=max({args.highlight_abs_floor}, {args.highlight_scale}*min_abs_energy)",
        "",
        "Selection rule:",
        "- Compute special_score = 2.5*N(|E|<=0.02) + 1.0*N(|E|<=0.05) + 0.08/(min|E|+1e-7) + 0.06/(|gap|+1e-7)",
        "- Select top-N by score as special points.",
        "",
        "Special points (rank, point_id, score):",
    ]
    for row in sorted(analysis_rows, key=lambda r: (0 if int(r["is_special"]) == 1 else 1, int(r["special_rank"]) if str(r["special_rank"]) else 9999)):
        if int(row["is_special"]) != 1:
            continue
        lines.append(
            f"rank={int(row['special_rank']):02d} "
            f"point_id={row['point_id']} "
            f"score={float(row['special_score']):.3f} "
            f"min|E|={float(row['min_abs_energy']):.4e} "
            f"N0.02={int(row['count_absE_le_0p02'])}"
        )
    output_txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"done total={len(analysis_rows)} special={top_n}")
    print(f"csv={output_csv}")
    print(f"summary={output_txt}")
    print(f"map={output_map}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze OBC E-vs-index spectra and mark special points in red."
    )
    parser.add_argument("--summary-csv", default="outputs/first_stage_band_ribbon/band_ribbon_summary.csv")
    parser.add_argument("--points-root", default="outputs/first_stage_band_ribbon/points")
    parser.add_argument("--output-csv", default="outputs/first_stage_band_ribbon/obc_special_points_analysis.csv")
    parser.add_argument("--output-txt", default="outputs/first_stage_band_ribbon/obc_special_points_summary.txt")
    parser.add_argument("--output-map-png", default="outputs/first_stage_band_ribbon/figures/obc_special_points_map_red.png")
    parser.add_argument("--obc-nx", type=int, default=20)
    parser.add_argument("--obc-ny", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--highlight-abs-floor", type=float, default=0.02)
    parser.add_argument("--highlight-scale", type=float, default=6.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
