from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from scipy.interpolate import griddata

matplotlib.use("Agg")


def load_topology_rows(summary_csv: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("topology_status") != "ok":
                continue
            rows.append(
                {
                    "v": float(row["v"]),
                    "lm": float(row["lm"]),
                    "chern": float(row["chern"]),
                    "z2": int(float(row["z2"])),
                }
            )
    return rows


def phase_index(chern: float, z2: int, chern_threshold: float) -> int:
    if abs(chern) >= chern_threshold:
        return 2  # Chern-like topological phase
    if z2 == 1:
        return 1  # Z2-like topological phase
    return 0  # trivial


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    summary_csv = root / args.summary_csv
    output_png = root / args.output_png
    output_png.parent.mkdir(parents=True, exist_ok=True)

    rows = load_topology_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"No valid topology rows found in {summary_csv}")

    pts = np.array([(float(r["v"]), float(r["lm"])) for r in rows], dtype=float)
    phase = np.array(
        [
            phase_index(
                chern=float(r["chern"]),
                z2=int(r["z2"]),
                chern_threshold=args.chern_threshold,
            )
            for r in rows
        ],
        dtype=float,
    )

    v_grid = np.arange(args.v_min, args.v_max + 1e-12, args.step)
    lm_grid = np.arange(args.lm_min, args.lm_max + 1e-12, args.step)
    vv, ll = np.meshgrid(v_grid, lm_grid, indexing="xy")

    # Dense phase map for plotting only (no dense result CSV written)
    phase_dense = griddata(pts, phase, (vv, ll), method="nearest")

    cmap = mcolors.ListedColormap(["#8c8c8c", "#1f77b4", "#d62728"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        phase_dense,
        origin="lower",
        extent=[v_grid.min(), v_grid.max(), lm_grid.min(), lm_grid.max()],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    ax.set_xlabel("v")
    ax.set_ylabel("lm")
    ax.set_title(f"Dense phase diagram (step={args.step:.2f}, t=0.5, w=2-v)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["trivial", "Z2-like", "Chern-like"])
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    print(f"rows={len(rows)}")
    print(f"grid={len(v_grid)}x{len(lm_grid)}")
    print(f"figure={output_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw dense phase diagram only (no dense result file)."
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/first_stage_band_ribbon/topology_summary.csv",
    )
    parser.add_argument(
        "--output-png",
        default="outputs/first_stage_band_ribbon/figures/phase_diagram_v_lm_dense_step_0p01.png",
    )
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--lm-min", type=float, default=0.0)
    parser.add_argument("--lm-max", type=float, default=0.5)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--chern-threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
