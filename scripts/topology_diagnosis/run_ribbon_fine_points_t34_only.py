#!/usr/bin/env python3
"""Ribbon spectra at representative fine-scan points for t34-only model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (  # noqa: E402
    ModelParams,
    compute_bulk_gap,
    ribbon_spectrum_with_edge_weight,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Representative ribbon points around fine transition (t34-only).")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-points", default="0.80,0.90,0.95,1.00,1.05,1.10,1.20")
    p.add_argument("--nx", type=int, default=40)
    p.add_argument("--nk", type=int, default=181)
    p.add_argument("--edge-cells", type=int, default=3)
    p.add_argument("--edge-threshold", type=float, default=0.6)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def vtok2(v: float) -> str:
    return f"{v:.2f}".replace("-", "m").replace(".", "p")


def load_bulk_gap_map(path: Path) -> dict[float, float]:
    if not path.exists():
        return {}
    out: dict[float, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[round(float(row["v"]), 10)] = float(row["bulk_gap"])
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = base / "fine_scan_0p8_1p2" / "03_ribbon"
    out.mkdir(parents=True, exist_ok=True)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))

    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    v_points = parse_float_list(args.v_points)
    bulk_map = load_bulk_gap_map(base / "fine_scan_0p8_1p2" / "bulk_gap_fine_scan_t34_only.csv")

    rows = []
    panel_paths = []
    for v in v_points:
        bulk_gap = bulk_map.get(round(v, 10))
        if bulk_gap is None:
            bulk_gap, _, _ = compute_bulk_gap(v=v, params=params, nk=101, n_occ=4)
        ky, evals, edge_w = ribbon_spectrum_with_edge_weight(
            v=v,
            params=params,
            nx=args.nx,
            nk=args.nk,
            edge_cells=args.edge_cells,
        )
        edge_mask = edge_w >= args.edge_threshold
        in_gap = np.abs(evals) <= max(1e-4, 0.5 * bulk_gap)
        num_ingap = int(np.sum(edge_mask & in_gap))
        max_edge_w = float(np.max(edge_w)) if edge_w.size else 0.0
        min_abs_edge = float(np.min(np.abs(evals[edge_mask]))) if np.any(edge_mask) else float("nan")
        edge_state_present = int(num_ingap > 0)
        comment = "finite_size_sensitive_small_bulk_gap" if bulk_gap < 0.05 else "ok"
        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(bulk_gap),
                "num_in_gap_edge_states": int(num_ingap),
                "max_edge_weight": float(max_edge_w),
                "min_abs_edge_energy": float(min_abs_edge),
                "edge_state_present": int(edge_state_present),
                "comment": comment,
            }
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.9))
        x = np.repeat(ky / np.pi, evals.shape[1])
        y = evals.reshape(-1)
        c = edge_w.reshape(-1)
        sc = ax.scatter(x, y, c=c, s=5, cmap="viridis", edgecolors="none", vmin=0.0, vmax=1.0)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$k_y/\pi$")
        ax.set_ylabel("Energy")
        ax.set_title(
            f"ribbon t34-only v={v:.2f}, bulk_gap={bulk_gap:.4f}, "
            f"in-gap edge={num_ingap}, maxW={max_edge_w:.3f}"
        )
        ax.grid(alpha=0.2)
        fig.colorbar(sc, ax=ax, label="edge weight")
        fig.tight_layout()
        png = out / f"ribbon_v{vtok2(v)}_t34_only.png"
        fig.savefig(png, dpi=180)
        fig.savefig(base / "fine_scan_0p8_1p2" / png.name, dpi=180)
        plt.close(fig)
        panel_paths.append(png)
        print(
            f"[ribbon-fine-t34] v={v:.2f} gap={bulk_gap:.5f} "
            f"in-gap edge={num_ingap} maxW={max_edge_w:.3f}"
        )

    csv_path = out / "ribbon_fine_points_summary_t34_only.csv"
    write_csv(
        csv_path,
        rows,
        ["v", "bulk_gap", "num_in_gap_edge_states", "max_edge_weight", "min_abs_edge_energy", "edge_state_present", "comment"],
    )
    (base / "fine_scan_0p8_1p2" / "ribbon_fine_points_summary_t34_only.csv").write_bytes(csv_path.read_bytes())

    cols = 3
    n = len(panel_paths)
    r = int(np.ceil(n / cols))
    fig, axes = plt.subplots(r, cols, figsize=(5.2 * cols, 3.9 * r))
    axes = np.array(axes).reshape(r, cols)
    for i in range(r * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(plt.imread(panel_paths[i]))
            ax.axis("off")
            ax.set_title(panel_paths[i].stem)
        else:
            ax.axis("off")
    fig.suptitle("Representative ribbon points (t34-only fine scan)", fontsize=12)
    fig.tight_layout()
    montage = out / "ribbon_fine_points_montage_t34_only.png"
    fig.savefig(montage, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / montage.name, dpi=180)
    plt.close(fig)
    print(f"[ok] ribbon fine summary: {csv_path}")


if __name__ == "__main__":
    main()
