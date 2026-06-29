#!/usr/bin/env python3
"""Representative ribbon wavefunctions for t34-only model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.topology_diagnosis.t34_only_common import (
    ModelParams,
    edge_bulk_weights_1d,
    output_dirs,
    ribbon_eigensystem_xopen_spin,
    rho_x_from_vec_spin_block,
    token,
    write_csv,
    write_t_hopping_check,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ribbon wavefunction diagnostics for t34-only model.")
    parser.add_argument("--output-root", default="/workspace/topology_diagnosis_outputs_t34_only")
    parser.add_argument("--v-list", default="0.5,0.8")
    parser.add_argument("--t34", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--edge-cells", type=int, default=3)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def ky_label(ky: float) -> str:
    if abs(ky) < 1e-12:
        return "0"
    if abs(ky - 0.25 * np.pi) < 1e-12:
        return "0p25pi"
    if abs(ky - 0.5 * np.pi) < 1e-12:
        return "0p5pi"
    return f"{ky/np.pi:.2f}pi".replace("-", "m").replace(".", "p")


def main() -> None:
    args = parse_args()
    base = Path(args.output_root)
    dirs = output_dirs(base)
    write_t_hopping_check(base, ModelParams(t34=args.t34, w=args.w, lm=args.lm))
    out_dir = dirs["03_spin_resolved_ribbon"]
    params = ModelParams(t34=args.t34, w=args.w, lm=args.lm)
    rows = []
    ky_list = [0.0, 0.25 * np.pi, 0.5 * np.pi]

    for v in parse_float_list(args.v_list):
        for spin_sign, spin_name in [(+1, "spin_up"), (-1, "spin_down")]:
            for ky in ky_list:
                evals, vecs = ribbon_eigensystem_xopen_spin(
                    ky=float(ky),
                    v=v,
                    params=params,
                    nx=args.nx,
                    spin_sign=spin_sign,
                )
                idx = int(np.argmin(np.abs(evals)))
                e = float(evals[idx])
                vec = vecs[:, idx]
                rho_x = rho_x_from_vec_spin_block(vec, nx=args.nx)
                rho_x = rho_x / max(np.sum(rho_x), 1e-15)
                wl, wr, wb = edge_bulk_weights_1d(rho_x, edge_cells=args.edge_cells)
                if wl + wr > 0.6:
                    cls = "edge_state"
                elif wb > 0.6:
                    cls = "bulk_state"
                else:
                    cls = "mixed_state"
                rows.append(
                    {
                        "v": float(v),
                        "spin": spin_name,
                        "ky": float(ky),
                        "energy": float(e),
                        "left_edge_weight": float(wl),
                        "right_edge_weight": float(wr),
                        "bulk_weight": float(wb),
                        "classification": cls,
                    }
                )

                fig, ax = plt.subplots(figsize=(6.8, 4.1))
                ax.plot(np.arange(args.nx), rho_x, marker="o", linewidth=1.1)
                ax.set_xlabel("x cell index")
                ax.set_ylabel(r"$\rho(x)$")
                ax.set_title(
                    f"{spin_name} v={v:.1f}, ky={ky/np.pi:.2f}pi, E={e:.4f}, "
                    f"Wl={wl:.3f}, Wr={wr:.3f}, Wb={wb:.3f}"
                )
                ax.grid(alpha=0.25)
                fig.tight_layout()
                out_png = out_dir / f"wf_{spin_name}_v{token(v)}_ky{ky_label(ky)}_t34_only.png"
                fig.savefig(out_png, dpi=180)
                plt.close(fig)
                # compatibility with requested subset names at root
                if abs(v - 0.5) < 1e-9 or abs(v - 0.8) < 1e-9:
                    (base / out_png.name).write_bytes(out_png.read_bytes())

                print(
                    f"[wf-t34] v={v:.1f} {spin_name} ky={ky/np.pi:.2f}pi "
                    f"E={e:.5f} Wl={wl:.3f} Wr={wr:.3f} Wb={wb:.3f} {cls}"
                )

    write_csv(
        out_dir / "ribbon_wavefunction_t34_only_summary.csv",
        rows,
        ["v", "spin", "ky", "energy", "left_edge_weight", "right_edge_weight", "bulk_weight", "classification"],
    )
    write_csv(
        base / "ribbon_wavefunction_t34_only_summary.csv",
        rows,
        ["v", "spin", "ky", "energy", "left_edge_weight", "right_edge_weight", "bulk_weight", "classification"],
    )
    print(f"[ok] wavefunction summary: {out_dir / 'ribbon_wavefunction_t34_only_summary.csv'}")


if __name__ == "__main__":
    main()
