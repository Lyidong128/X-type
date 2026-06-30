#!/usr/bin/env python3
"""Bulk-gap scan for strict t34 + soc34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_soc34_only_common import (  # noqa: E402
    ModelParams,
    compute_bulk_gap,
    output_dirs,
    parse_float_list,
    token,
    write_csv,
    write_model_check,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bulk-gap scan for strict t34+soc34-only model.")
    p.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_soc34_only")
    p.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    p.add_argument("--nk", type=int, default=101)
    p.add_argument("--n-occ", type=int, default=4)
    p.add_argument("--t34", type=float, default=0.3)
    p.add_argument("--w", type=float, default=1.0)
    p.add_argument("--lm", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_model_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out = dirs["01_bulk_gap"]

    v_list = parse_float_list(args.v_list)
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    rows = []
    for v in v_list:
        gap, kx, ky = compute_bulk_gap(v=v, params=params, nk=args.nk, n_occ=args.n_occ)
        status = "gap_closed" if gap < 1e-4 else ("near_gap_closing" if gap < 0.05 else "gapped")
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
        print(f"[bulk-soc34] v={v:.1f} gap={gap:.6f} status={status}")

    csv_path = out / "bulk_gap_t34_soc34_only.csv"
    write_csv(csv_path, rows, ["v", "bulk_gap", "gap_kx", "gap_ky", "n_occ", "gap_status"])
    write_csv(base / "bulk_gap_t34_soc34_only.csv", rows, ["v", "bulk_gap", "gap_kx", "gap_ky", "n_occ", "gap_status"])

    v = np.array([float(r["v"]) for r in rows], dtype=float)
    g = np.array([float(r["bulk_gap"]) for r in rows], dtype=float)
    i_min = int(np.argmin(g))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(v, g, marker="o", linewidth=1.2)
    ax.scatter([v[i_min]], [g[i_min]], color="red", s=80, label=f"min @ v={v[i_min]:.1f}")
    near = g < 0.05
    if np.any(near):
        ax.fill_between(v, 0.0, g, where=near, alpha=0.2, color="orange", label="bulk_gap < 0.05")
    ax.axvline(v[i_min], color="purple", linestyle="--", linewidth=1.0, label="estimated transition")
    ax.set_xlabel("v")
    ax.set_ylabel("bulk gap")
    ax.set_title(f"bulk gap vs v (t34+soc34-only), nk={args.nk}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bulk_gap_t34_soc34_only.png", dpi=180)
    fig.savefig(base / "bulk_gap_t34_soc34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] bulk summary: v_gap_min={v[i_min]:.3f}, gap_min={g[i_min]:.6f}")


if __name__ == "__main__":
    main()
