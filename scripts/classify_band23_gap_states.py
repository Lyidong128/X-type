from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (  # noqa: E402
    build_ribbon_hamiltonian,
    compute_band_data,
    compute_dynamic_w,
    load_xtype_model,
    set_model_params,
)

matplotlib.use("Agg")


CLASS_CONNECTING = "connecting_edge_branch"
CLASS_ISOLATED = "isolated_midgap_state"
CLASS_EMPTY = "empty_gap"


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
    Compute edge localization weights for ribbon eigenvectors.
    Ribbon basis is 8*nx, where each x-cell has 8 internal states.
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


def infer_band23_window(
    model_module,
    v: float,
    t: float,
    lm: float,
    lower_band_1based: int,
    upper_band_1based: int,
) -> tuple[float, float, float]:
    """
    Infer global energy window between selected bulk bands from normal band structure.

    Example: lower_band_1based=2 and upper_band_1based=3 means:
      lower_edge = max(E_band2 over k-path)
      upper_edge = min(E_band3 over k-path)
    """
    w = compute_dynamic_w(v)
    set_model_params(model_module, v=v, t=t, lm=lm, w=w, j=0.0)
    band_data = compute_band_data(model_module)
    n_bands = band_data.shape[1]

    lower_idx = lower_band_1based - 1
    upper_idx = upper_band_1based - 1
    if lower_idx < 0 or upper_idx < 0 or lower_idx >= n_bands or upper_idx >= n_bands:
        raise ValueError(
            f"Band index out of range: lower={lower_band_1based}, "
            f"upper={upper_band_1based}, n_bands={n_bands}"
        )
    if lower_idx >= upper_idx:
        raise ValueError(
            f"Require lower_band < upper_band, got {lower_band_1based} and {upper_band_1based}"
        )

    lower_edge = float(np.max(band_data[:, lower_idx]))
    upper_edge = float(np.min(band_data[:, upper_idx]))
    width = float(upper_edge - lower_edge)
    return lower_edge, upper_edge, width


def classify_point_in_band_window(
    v: float,
    t: float,
    lm: float,
    w: float,
    lower_edge: float,
    upper_edge: float,
    window_width: float,
    nx: int,
    nk: int,
    edge_cells: int,
    edge_threshold: float,
    min_window_width: float,
) -> dict[str, float | int | str]:
    if window_width <= min_window_width:
        return {
            "classification": CLASS_EMPTY,
            "has_edge_state": 0,
            "has_isolated_state": 0,
            "has_connecting_edge_branch": 0,
            "in_window_edge_state_count": 0,
            "ky_coverage": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "span_ratio": 0.0,
            "reason": "band23_window_collapsed_or_negative",
        }

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

    in_window = (energies > lower_edge) & (energies < upper_edge)
    edge_like = edge_w >= edge_threshold
    in_window_edge = in_window & edge_like
    in_window_energy = energies[in_window_edge]
    in_window_k = k_idx[in_window_edge]
    n_in_window_edge = int(in_window_energy.size)

    if n_in_window_edge == 0:
        return {
            "classification": CLASS_EMPTY,
            "has_edge_state": 0,
            "has_isolated_state": 0,
            "has_connecting_edge_branch": 0,
            "in_window_edge_state_count": 0,
            "ky_coverage": 0.0,
            "touches_lower": 0,
            "touches_upper": 0,
            "span_ratio": 0.0,
            "reason": "no_edge_like_state_inside_band23_window",
        }

    touch_tol = max(0.02, 0.06 * window_width)
    touches_lower = int(np.any(np.abs(in_window_energy - lower_edge) <= touch_tol))
    touches_upper = int(np.any(np.abs(in_window_energy - upper_edge) <= touch_tol))
    span_ratio = float((np.max(in_window_energy) - np.min(in_window_energy)) / max(window_width, 1e-12))
    ky_coverage = float(np.unique(in_window_k).size / nk)

    if (
        touches_lower == 1
        and touches_upper == 1
        and span_ratio >= 0.65
        and ky_coverage >= 0.25
    ):
        cls = CLASS_CONNECTING
        has_isolated = 0
        has_connecting = 1
        reason = "touches_both_band_edges_with_large_span"
    else:
        cls = CLASS_ISOLATED
        has_isolated = 1
        has_connecting = 0
        reason = "edge_states_exist_but_not_connecting_band2_and_band3"

    return {
        "classification": cls,
        "has_edge_state": 1,
        "has_isolated_state": has_isolated,
        "has_connecting_edge_branch": has_connecting,
        "in_window_edge_state_count": n_in_window_edge,
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
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.scatter(v, lm, c=c, cmap=cmap, norm=norm, s=120, marker="s", edgecolors="black", linewidths=0.3)
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title("Band-2/3 window ribbon state classification (t=0.5, w=2-v)")
    legend_handles = [
        Patch(facecolor="#8c8c8c", edgecolor="black", label="0: Empty gap"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="1: Isolated mid-gap state"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="2: Connecting edge branch"),
    ]
    ax.legend(
        handles=legend_handles,
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    summary_csv = root / args.summary_csv
    output_csv = root / args.output_csv
    output_txt = root / args.output_txt
    output_map_png = root / args.output_map_png

    model_path = root / "models" / args.model_file
    model = load_xtype_model(model_path)
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
        lower_edge, upper_edge, window_width = infer_band23_window(
            model_module=model,
            v=float(p["v"]),
            t=float(p["t"]),
            lm=float(p["lm"]),
            lower_band_1based=args.lower_band,
            upper_band_1based=args.upper_band,
        )
        rec["band_lower_edge"] = lower_edge
        rec["band_upper_edge"] = upper_edge
        rec["band_window_width"] = window_width
        rec.update(
            classify_point_in_band_window(
                v=float(p["v"]),
                t=float(p["t"]),
                lm=float(p["lm"]),
                w=float(p["w"]),
                lower_edge=lower_edge,
                upper_edge=upper_edge,
                window_width=window_width,
                nx=args.ribbon_nx,
                nk=args.ribbon_nk,
                edge_cells=args.edge_cells,
                edge_threshold=args.edge_threshold,
                min_window_width=args.min_window_width,
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
        "band_lower_edge",
        "band_upper_edge",
        "band_window_width",
        "classification",
        "has_edge_state",
        "has_isolated_state",
        "has_connecting_edge_branch",
        "in_window_edge_state_count",
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
    n_has_edge = int(sum(int(r["has_edge_state"]) for r in rows))
    n_has_isolated = int(sum(int(r["has_isolated_state"]) for r in rows))
    n_has_connecting = int(sum(int(r["has_connecting_edge_branch"]) for r in rows))
    lines = [
        "Band-2/3 window ribbon-state classification summary",
        f"total_points={len(rows)}",
        f"connecting_edge_branch={cnt.get(CLASS_CONNECTING, 0)}",
        f"isolated_midgap_state={cnt.get(CLASS_ISOLATED, 0)}",
        f"empty_gap={cnt.get(CLASS_EMPTY, 0)}",
        "",
        f"points_with_edge_state={n_has_edge}",
        f"points_with_isolated_state={n_has_isolated}",
        f"points_with_connecting_edge_branch={n_has_connecting}",
        "",
        f"lower_band={args.lower_band}",
        f"upper_band={args.upper_band}",
        f"edge_threshold={args.edge_threshold}",
        f"ribbon_nx={args.ribbon_nx}",
        f"ribbon_nk={args.ribbon_nk}",
        f"edge_cells={args.edge_cells}",
        f"min_window_width={args.min_window_width}",
        f"model_file={args.model_file}",
    ]
    output_txt.write_text("\n".join(lines), encoding="utf-8")
    plot_class_map(rows=rows, save_path=output_map_png)

    print(f"total={len(rows)}")
    print(f"csv={output_csv}")
    print(f"summary={output_txt}")
    print(f"map={output_map_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify ribbon states inside the normal-band 2/3 gap window."
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/first_stage_band_ribbon/band_ribbon_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/first_stage_band_ribbon/ribbon_gap_state_classification_band23.csv",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/first_stage_band_ribbon/ribbon_gap_state_classification_band23_summary.txt",
    )
    parser.add_argument(
        "--output-map-png",
        default="outputs/first_stage_band_ribbon/figures/ribbon_gap_state_classification_band23_map.png",
    )
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--lower-band", type=int, default=2, help="1-based lower band index")
    parser.add_argument("--upper-band", type=int, default=3, help="1-based upper band index")
    parser.add_argument("--ribbon-nx", type=int, default=20)
    parser.add_argument("--ribbon-nk", type=int, default=121)
    parser.add_argument("--edge-cells", type=int, default=2)
    parser.add_argument("--edge-threshold", type=float, default=0.55)
    parser.add_argument("--min-window-width", type=float, default=1e-10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
