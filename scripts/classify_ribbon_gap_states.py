from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_ribbon_hamiltonian  # noqa: E402

matplotlib.use("Agg")


CLASS_CONNECTING = "connecting_edge_branch"
CLASS_ISOLATED = "isolated_midgap_state"
CLASS_EMPTY = "empty_gap"

CLASS_CN = {
    CLASS_CONNECTING: "连接上下能带边缘态",
    CLASS_ISOLATED: "能隙中间孤立态",
    CLASS_EMPTY: "能隙中间无态",
}


def load_points(summary_csv: Path) -> list[dict[str, float | str]]:
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


def edge_weight_from_vecs(evecs: np.ndarray, nx: int, edge_cells: int) -> np.ndarray:
    """
    Compute edge localization weight for ribbon eigenvectors.
    Ribbon basis is 8*nx, one x-cell has 8 internal states.
    """
    n_modes = evecs.shape[1]
    weights = np.zeros(n_modes, dtype=float)
    edge_cells = max(1, min(edge_cells, nx // 2))
    left_end = edge_cells * 8
    right_start = (nx - edge_cells) * 8
    for i in range(n_modes):
        psi2 = np.abs(evecs[:, i]) ** 2
        weights[i] = float(np.sum(psi2[:left_end]) + np.sum(psi2[right_start:]))
    return weights


def classify_point(
    v: float,
    t: float,
    lm: float,
    w: float,
    nx: int,
    nk: int,
    edge_cells: int,
    edge_threshold: float,
) -> dict[str, float | int | str]:
    ky_values = np.linspace(-np.pi, np.pi, nk)
    all_energies = []
    all_edge_weights = []
    all_k_idx = []

    for k_idx, ky in enumerate(ky_values):
        ham = build_ribbon_hamiltonian(v=v, t=t, lm=lm, ky=ky, w=w, j=0.0, nx=nx)
        evals, evecs = np.linalg.eigh(ham)
        ew = edge_weight_from_vecs(evecs=evecs, nx=nx, edge_cells=edge_cells)
        all_energies.append(evals.real)
        all_edge_weights.append(ew)
        all_k_idx.append(np.full(evals.shape, k_idx, dtype=int))

    energies = np.concatenate(all_energies)
    edge_w = np.concatenate(all_edge_weights)
    k_idx = np.concatenate(all_k_idx)

    edge_like = edge_w >= edge_threshold
    non_edge_like = ~edge_like

    lower_candidates = energies[(energies < 0) & non_edge_like]
    upper_candidates = energies[(energies > 0) & non_edge_like]
    if lower_candidates.size == 0:
        lower_candidates = energies[energies < 0]
    if upper_candidates.size == 0:
        upper_candidates = energies[energies > 0]
    if lower_candidates.size == 0 or upper_candidates.size == 0:
        # Degenerate metallic fallback: classify as empty_gap for this rubric.
        return {
            "classification": CLASS_EMPTY,
            "classification_cn": CLASS_CN[CLASS_EMPTY],
            "lower_bulk_edge": float("nan"),
            "upper_bulk_edge": float("nan"),
            "gap_width": 0.0,
            "in_gap_edge_state_count": 0,
            "ky_coverage": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "span_ratio": 0.0,
            "reason": "no_positive_or_negative_bulk_edge",
        }

    lower_bulk = float(np.max(lower_candidates))
    upper_bulk = float(np.min(upper_candidates))
    gap_width = float(max(upper_bulk - lower_bulk, 0.0))
    if gap_width <= 1e-10:
        return {
            "classification": CLASS_EMPTY,
            "classification_cn": CLASS_CN[CLASS_EMPTY],
            "lower_bulk_edge": lower_bulk,
            "upper_bulk_edge": upper_bulk,
            "gap_width": gap_width,
            "in_gap_edge_state_count": 0,
            "ky_coverage": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "span_ratio": 0.0,
            "reason": "gap_collapsed_or_zero",
        }

    in_gap_edge = edge_like & (energies > lower_bulk) & (energies < upper_bulk)
    in_gap_energy = energies[in_gap_edge]
    in_gap_k = k_idx[in_gap_edge]
    n_in_gap = int(in_gap_energy.size)

    if n_in_gap == 0:
        return {
            "classification": CLASS_EMPTY,
            "classification_cn": CLASS_CN[CLASS_EMPTY],
            "lower_bulk_edge": lower_bulk,
            "upper_bulk_edge": upper_bulk,
            "gap_width": gap_width,
            "in_gap_edge_state_count": 0,
            "ky_coverage": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "span_ratio": 0.0,
            "reason": "no_edge_like_state_inside_gap",
        }

    touch_tol = max(0.02, 0.06 * gap_width)
    touches_lower = int(np.any(np.abs(in_gap_energy - lower_bulk) <= touch_tol))
    touches_upper = int(np.any(np.abs(in_gap_energy - upper_bulk) <= touch_tol))
    span_ratio = float((np.max(in_gap_energy) - np.min(in_gap_energy)) / max(gap_width, 1e-12))
    ky_coverage = float(np.unique(in_gap_k).size / nk)

    if (
        touches_lower == 1
        and touches_upper == 1
        and span_ratio >= 0.65
        and ky_coverage >= 0.25
    ):
        cls = CLASS_CONNECTING
        reason = "touches_both_bulk_edges_with_large_span"
    else:
        cls = CLASS_ISOLATED
        reason = "in_gap_edge_states_exist_but_not_connecting_both_edges"

    return {
        "classification": cls,
        "classification_cn": CLASS_CN[cls],
        "lower_bulk_edge": lower_bulk,
        "upper_bulk_edge": upper_bulk,
        "gap_width": gap_width,
        "in_gap_edge_state_count": n_in_gap,
        "ky_coverage": ky_coverage,
        "touches_lower": touches_lower,
        "touches_upper": touches_upper,
        "span_ratio": span_ratio,
        "reason": reason,
    }


def plot_class_map(rows: list[dict[str, float | int | str]], save_path: Path) -> None:
    class_to_int = {
        CLASS_EMPTY: 0,
        CLASS_ISOLATED: 1,
        CLASS_CONNECTING: 2,
    }
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    lm = np.array([float(r["lm"]) for r in rows], dtype=float)
    c = np.array([class_to_int[str(r["classification"])] for r in rows], dtype=float)

    cmap = mcolors.ListedColormap(["#8c8c8c", "#ff7f0e", "#1f77b4"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(v, lm, c=c, cmap=cmap, norm=norm, s=120, marker="s", edgecolors="black", linewidths=0.3)
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title("Ribbon gap-state classification map (t=0.5, w=2-v)")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(
        [
            CLASS_CN[CLASS_EMPTY],
            CLASS_CN[CLASS_ISOLATED],
            CLASS_CN[CLASS_CONNECTING],
        ]
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    summary_csv = root / args.summary_csv
    output_csv = root / args.output_csv
    output_txt = root / args.output_txt
    map_png = root / args.output_map_png

    points = load_points(summary_csv)
    if not points:
        raise RuntimeError(f"No valid points found in {summary_csv}")

    rows: list[dict[str, float | int | str]] = []
    total = len(points)
    for i, p in enumerate(points, start=1):
        rec = {
            "point_id": p["point_id"],
            "v": p["v"],
            "t": p["t"],
            "lm": p["lm"],
            "w": p["w"],
        }
        rec.update(
            classify_point(
                v=float(p["v"]),
                t=float(p["t"]),
                lm=float(p["lm"]),
                w=float(p["w"]),
                nx=args.ribbon_nx,
                nk=args.ribbon_nk,
                edge_cells=args.edge_cells,
                edge_threshold=args.edge_threshold,
            )
        )
        rows.append(rec)
        if i % 10 == 0 or i == total:
            print(f"processed {i}/{total}")

    fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "w",
        "classification",
        "classification_cn",
        "lower_bulk_edge",
        "upper_bulk_edge",
        "gap_width",
        "in_gap_edge_state_count",
        "ky_coverage",
        "touches_lower",
        "touches_upper",
        "span_ratio",
        "reason",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    cnt = Counter(str(r["classification"]) for r in rows)
    lines = [
        "Ribbon gap-state classification summary",
        f"total_points={len(rows)}",
        f"connecting_edge_branch={cnt.get(CLASS_CONNECTING, 0)}",
        f"isolated_midgap_state={cnt.get(CLASS_ISOLATED, 0)}",
        f"empty_gap={cnt.get(CLASS_EMPTY, 0)}",
        "",
        f"edge_threshold={args.edge_threshold}",
        f"ribbon_nx={args.ribbon_nx}",
        f"ribbon_nk={args.ribbon_nk}",
        f"edge_cells={args.edge_cells}",
    ]
    output_txt.write_text("\n".join(lines), encoding="utf-8")
    plot_class_map(rows=rows, save_path=map_png)

    print(f"total={len(rows)}")
    print(f"csv={output_csv}")
    print(f"summary={output_txt}")
    print(f"map={map_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify all points by ribbon in-gap state morphology."
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/first_stage_band_ribbon/band_ribbon_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/first_stage_band_ribbon/ribbon_gap_state_classification.csv",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/first_stage_band_ribbon/ribbon_gap_state_classification_summary.txt",
    )
    parser.add_argument(
        "--output-map-png",
        default="outputs/first_stage_band_ribbon/figures/ribbon_gap_state_classification_map.png",
    )
    parser.add_argument("--ribbon-nx", type=int, default=20)
    parser.add_argument("--ribbon-nk", type=int, default=121)
    parser.add_argument("--edge-cells", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
