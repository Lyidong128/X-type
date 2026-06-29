#!/usr/bin/env python3
"""Spin-resolved ribbon spectra for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    output_dirs,
    ribbon_eigensystem_xopen_spin,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spin-resolved ribbon check for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.5,0.8")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--nk", type=int, default=161)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def spectrum_spin(v: float, params: ModelParams, nx: int, nk: int, spin_sign: int) -> tuple[np.ndarray, np.ndarray]:
    ky = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    all_e = []
    for k in ky:
        e, _ = ribbon_eigensystem_xopen_spin(float(k), v=v, params=params, nx=nx, spin_sign=spin_sign)
        all_e.append(e)
    return ky, np.array(all_e, dtype=float)


def plot_single(ky: np.ndarray, e: np.ndarray, out_png: Path, color: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x = np.repeat(ky / np.pi, e.shape[1])
    y = e.reshape(-1)
    ax.scatter(x, y, s=4, color=color, alpha=0.75)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_overlay(ky: np.ndarray, e_up: np.ndarray, e_dn: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    x_up = np.repeat(ky / np.pi, e_up.shape[1])
    y_up = e_up.reshape(-1)
    x_dn = np.repeat(ky / np.pi, e_dn.shape[1])
    y_dn = e_dn.reshape(-1)
    ax.scatter(x_up, y_up, s=4, color="red", alpha=0.65, label="spin-up")
    ax.scatter(x_dn, y_dn, s=4, color="blue", alpha=0.55, label="spin-down")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["03_spin_resolved_ribbon"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    rows = []

    for v in parse_float_list(args.v_list):
        ky, e_up = spectrum_spin(v=v, params=params, nx=args.nx, nk=args.nk, spin_sign=+1)
        _, e_dn = spectrum_spin(v=v, params=params, nx=args.nx, nk=args.nk, spin_sign=-1)

        # metrics
        neg_up = e_up[e_up < 0.0]
        pos_up = e_up[e_up > 0.0]
        neg_dn = e_dn[e_dn < 0.0]
        pos_dn = e_dn[e_dn > 0.0]
        spin_gap_up = float(np.min(pos_up) - np.max(neg_up)) if neg_up.size and pos_up.size else float("nan")
        spin_gap_dn = float(np.min(pos_dn) - np.max(neg_dn)) if neg_dn.size and pos_dn.size else float("nan")

        e_up_sort = np.sort(e_up, axis=1)
        e_dn_sort = np.sort(e_dn, axis=1)
        spin_deg_err = float(np.max(np.abs(e_up_sort - e_dn_sort)))

        mid_up = np.min(np.abs(e_up), axis=1)
        mid_dn = np.min(np.abs(e_dn), axis=1)
        # simple indicator: both channels approach zero in similar k regions
        edge_spin_locking_indicator = float(np.mean((mid_up < 0.05) & (mid_dn < 0.05)))
        if abs(v - 0.5) < 1e-9:
            comment = "v=0.5 remains non-QSH candidate (check edge-localized SSH-like states via wavefunctions)"
        else:
            comment = "v=0.8 expected QSH-like region if C_spin and Z2 remain nontrivial"

        rows.append(
            {
                "v": float(v),
                "spin_gap_up": float(spin_gap_up),
                "spin_gap_down": float(spin_gap_dn),
                "spin_degeneracy_error": float(spin_deg_err),
                "edge_spin_locking_indicator": float(edge_spin_locking_indicator),
                "comment": comment,
            }
        )

        up_png = out_dir / f"ribbon_spin_up_v{token(v)}_t34_only.png"
        dn_png = out_dir / f"ribbon_spin_down_v{token(v)}_t34_only.png"
        ov_png = out_dir / f"ribbon_spin_overlay_v{token(v)}_t34_only.png"
        plot_single(ky, e_up, up_png, "red", f"spin-up ribbon t34-only v={v:.1f}")
        plot_single(ky, e_dn, dn_png, "blue", f"spin-down ribbon t34-only v={v:.1f}")
        plot_overlay(ky, e_up, e_dn, ov_png, f"spin overlay t34-only v={v:.1f}")

        # root-level copies for required names
        (base / up_png.name).write_bytes(up_png.read_bytes())
        (base / dn_png.name).write_bytes(dn_png.read_bytes())
        (base / ov_png.name).write_bytes(ov_png.read_bytes())
        print(
            f"[spin-ribbon-t34] v={v:.1f} gap_up={spin_gap_up:.5f} "
            f"gap_dn={spin_gap_dn:.5f} deg_err={spin_deg_err:.3e}"
        )

    write_csv(
        out_dir / "spin_resolved_ribbon_t34_only_summary.csv",
        rows,
        ["v", "spin_gap_up", "spin_gap_down", "spin_degeneracy_error", "edge_spin_locking_indicator", "comment"],
    )
    write_csv(
        base / "spin_resolved_ribbon_t34_only_summary.csv",
        rows,
        ["v", "spin_gap_up", "spin_gap_down", "spin_degeneracy_error", "edge_spin_locking_indicator", "comment"],
    )
    print(f"[ok] spin-resolved summary: {out_dir / 'spin_resolved_ribbon_t34_only_summary.csv'}")


if __name__ == "__main__":
    main()
