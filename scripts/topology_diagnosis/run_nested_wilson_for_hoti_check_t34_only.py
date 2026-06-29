#!/usr/bin/env python3
"""Strict nested-Wilson HOTI check for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    compute_wilson_centers_direction,
    nested_polarization_from_tracked,
    output_dirs,
    parse_float_list,
    token,
    track_sector_frames,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nested Wilson HOTI check for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nkx", type=int, default=101)
    parser.add_argument("--nky", type=int, default=101)
    parser.add_argument("--n-occ", type=int, default=4)
    return parser.parse_args()


def load_wilson_summary(path: Path) -> dict[float, dict]:
    import csv

    out: dict[float, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out[float(row["v"])] = row
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["07_nested_wilson"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)

    wilson_summary_path = base / "wilson_polarization_summary_t34_only.csv"
    if not wilson_summary_path.exists():
        raise FileNotFoundError(
            f"Missing {wilson_summary_path}. Run run_wilson_polarization_check_t34_only.py first."
        )
    wsum = load_wilson_summary(wilson_summary_path)

    rows = []
    for v in parse_float_list(args.v_list):
        wrow = wsum.get(float(v))
        if wrow is None:
            rows.append(
                {
                    "v": float(v),
                    "nested_allowed": 0,
                    "nested_reliable": 0,
                    "p_y_nu_x": float("nan"),
                    "p_x_nu_y": float("nan"),
                    "qxy": float("nan"),
                    "wannier_gap_x": float("nan"),
                    "wannier_gap_y": float("nan"),
                    "min_sector_overlap_x": float("nan"),
                    "min_sector_overlap_y": float("nan"),
                    "nested_supports_hoti": 0,
                    "comment": "missing_wilson_summary",
                }
            )
            continue

        nested_allowed = int(float(wrow["nested_allowed"]) > 0.5)
        wx_gap = float(wrow["wannier_gap_x"])
        wy_gap = float(wrow["wannier_gap_y"])
        if not nested_allowed:
            rows.append(
                {
                    "v": float(v),
                    "nested_allowed": 0,
                    "nested_reliable": 0,
                    "p_y_nu_x": float("nan"),
                    "p_x_nu_y": float("nan"),
                    "qxy": float("nan"),
                    "wannier_gap_x": float(wx_gap),
                    "wannier_gap_y": float(wy_gap),
                    "min_sector_overlap_x": 0.0,
                    "min_sector_overlap_y": 0.0,
                    "nested_supports_hoti": 0,
                    "comment": "nested_not_allowed_by_wannier_gap",
                }
            )
            print(f"[nested-t34] v={v:.1f} skipped (nested_allowed=False)")
            continue

        # direction x: sector polarization along y
        _, _, eigvec_x, occ_x, _ = compute_wilson_centers_direction(
            v=v,
            params=params,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="x",
        )
        tr_x, min_ov_x = track_sector_frames(eigvec_x, n_occ=args.n_occ)
        p_y_nu_x = nested_polarization_from_tracked(tr_x, occ_x)

        # direction y: sector polarization along x
        _, _, eigvec_y, occ_y, _ = compute_wilson_centers_direction(
            v=v,
            params=params,
            n_occ=args.n_occ,
            nkx=args.nkx,
            nky=args.nky,
            direction="y",
        )
        tr_y, min_ov_y = track_sector_frames(eigvec_y, n_occ=args.n_occ)
        p_x_nu_y = nested_polarization_from_tracked(tr_y, occ_y)
        qxy = float(np.mod(0.5 * (p_y_nu_x + p_x_nu_y), 1.0))

        nested_reliable = int(
            (wx_gap > 0.05)
            and (wy_gap > 0.05)
            and (min_ov_x > 0.5)
            and (min_ov_y > 0.5)
        )
        supports_hoti = int(nested_reliable and (abs(qxy - 0.5) < 0.05))
        comment = (
            "reliable_and_hoti_like"
            if supports_hoti
            else ("reliable_but_non_hoti_qxy" if nested_reliable else "nested_unreliable")
        )
        rows.append(
            {
                "v": float(v),
                "nested_allowed": int(nested_allowed),
                "nested_reliable": int(nested_reliable),
                "p_y_nu_x": float(p_y_nu_x),
                "p_x_nu_y": float(p_x_nu_y),
                "qxy": float(qxy),
                "wannier_gap_x": float(wx_gap),
                "wannier_gap_y": float(wy_gap),
                "min_sector_overlap_x": float(min_ov_x),
                "min_sector_overlap_y": float(min_ov_y),
                "nested_supports_hoti": int(supports_hoti),
                "comment": comment,
            }
        )
        print(
            f"[nested-t34] v={v:.1f} qxy={qxy:.4f} "
            f"overlaps=({min_ov_x:.3f},{min_ov_y:.3f}) reliable={bool(nested_reliable)}"
        )

    fields = [
        "v",
        "nested_allowed",
        "nested_reliable",
        "p_y_nu_x",
        "p_x_nu_y",
        "qxy",
        "wannier_gap_x",
        "wannier_gap_y",
        "min_sector_overlap_x",
        "min_sector_overlap_y",
        "nested_supports_hoti",
        "comment",
    ]
    write_csv(out_dir / "nested_hoti_check_summary_t34_only.csv", rows, fields)
    write_csv(base / "nested_hoti_check_summary_t34_only.csv", rows, fields)

    # qxy summary figure
    v = np.array([float(r["v"]) for r in rows], dtype=float)
    qxy = np.array(
        [float(r["qxy"]) if not math.isnan(float(r["qxy"])) else np.nan for r in rows],
        dtype=float,
    )
    rel = np.array([int(r["nested_reliable"]) for r in rows], dtype=int)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(v, qxy, marker="o", label="qxy")
    ax.scatter(v[rel > 0], qxy[rel > 0], color="green", s=70, label="reliable")
    ax.scatter(v[rel == 0], np.nan_to_num(qxy[rel == 0], nan=0.0), color="red", s=40, label="unreliable")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("v")
    ax.set_ylabel("qxy")
    ax.set_title("nested Wilson HOTI check (t34-only)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "qxy_hoti_check_t34_only.png", dpi=180)
    fig.savefig(base / "qxy_hoti_check_t34_only.png", dpi=180)
    plt.close(fig)
    print(f"[ok] nested summary: {out_dir / 'nested_hoti_check_summary_t34_only.csv'}")


if __name__ == "__main__":
    main()
