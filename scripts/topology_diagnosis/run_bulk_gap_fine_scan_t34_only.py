#!/usr/bin/env python3
"""Fine bulk-gap scan for t34-only model in v in [0.8,1.2]."""

from __future__ import annotations

import argparse
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
    ensure_dir,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine bulk-gap scan in 0.8-1.2 for t34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    p.add_argument("--v-min", type=float, default=0.80)
    p.add_argument("--v-max", type=float, default=1.20)
    p.add_argument("--v-step", type=float, default=0.02)
    p.add_argument("--dense-step", type=float, default=0.01)
    p.add_argument("--use-dense", action="store_true")
    p.add_argument("--nkx", type=int, default=151)
    p.add_argument("--nky", type=int, default=151)
    p.add_argument("--n-occ", type=int, default=4)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def make_v_list(vmin: float, vmax: float, step: float) -> list[float]:
    n = int(round((vmax - vmin) / step))
    return [round(vmin + i * step, 10) for i in range(n + 1)]


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    out = ensure_dir(base / "fine_scan_0p8_1p2" / "01_bulk_gap")
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))

    step = args.dense_step if args.use_dense else args.v_step
    v_list = make_v_list(args.v_min, args.v_max, step)
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    rows = []
    for v in v_list:
        gap, kx, ky = compute_bulk_gap(v=v, params=params, nk=args.nkx, n_occ=args.n_occ)
        if gap < 1e-4:
            status = "gap_closed"
        elif gap < 0.05:
            status = "near_gap_closing"
        else:
            status = "gapped"
        rows.append(
            {
                "v": float(v),
                "bulk_gap": float(gap),
                "gap_kx": float(kx),
                "gap_ky": float(ky),
                "n_occ": int(args.n_occ),
                "gap_status": status,
            }
        )
        print(f"[bulk-fine-t34] v={v:.2f} gap={gap:.6f} status={status} k=({kx:.4f},{ky:.4f})")

    csv_path = out / "bulk_gap_fine_scan_t34_only.csv"
    write_csv(csv_path, rows, ["v", "bulk_gap", "gap_kx", "gap_ky", "n_occ", "gap_status"])
    (base / "fine_scan_0p8_1p2" / "bulk_gap_fine_scan_t34_only.csv").write_bytes(csv_path.read_bytes())

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    g = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    i_min = int(np.argmin(g))
    v_gap_min = float(v[i_min])
    g_min = float(g[i_min])
    near = g < 0.05
    est_transition = v_gap_min

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(v, g, marker="o", linewidth=1.3, label="bulk gap")
    if np.any(near):
        ax.fill_between(v, 0.0, g, where=near, alpha=0.25, color="orange", label="bulk gap < 0.05")
    ax.scatter([v_gap_min], [g_min], s=90, color="red", zorder=5, label=f"min gap @ v={v_gap_min:.2f}")
    ax.axvline(est_transition, color="purple", linestyle="--", linewidth=1.1, label=f"estimated vc={est_transition:.2f}")
    ax.set_xlabel("v")
    ax.set_ylabel("bulk gap")
    ax.set_title(f"Fine bulk gap scan (t34-only), nkx=nky={args.nkx}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = out / "bulk_gap_fine_scan_t34_only.png"
    fig.savefig(png, dpi=180)
    fig.savefig(base / "fine_scan_0p8_1p2" / "bulk_gap_fine_scan_t34_only.png", dpi=180)
    plt.close(fig)

    summary_txt = out / "bulk_gap_fine_scan_summary_t34_only.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"grid_nkx={args.nkx}",
                f"grid_nky={args.nky}",
                f"v_gap_min={v_gap_min:.6f}",
                f"bulk_gap_min={g_min:.10f}",
                f"estimated_transition_point={est_transition:.6f}",
                f"used_dense_scan={int(args.use_dense)}",
                f"scan_step={step:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"[ok] bulk fine summary: v_gap_min={v_gap_min:.3f}, gap_min={g_min:.6f}, "
        f"grid={args.nkx}x{args.nky}, csv={csv_path}"
    )


if __name__ == "__main__":
    main()
