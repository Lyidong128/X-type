#!/usr/bin/env python3
"""Generate representative-point package for SOC on/off comparison."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (
    compute_band_data,
    compute_bulk_gap,
    compute_chern_number,
    compute_obc_spectrum_and_probability,
    compute_ribbon_spectrum,
    compute_wilson_loop_and_z2,
    load_xtype_model,
    plot_band_structure,
    plot_obc_spectrum,
    plot_obc_wavefunction,
    plot_ribbon_spectrum,
    set_model_params,
    write_results_csv,
)


def parse_v_values(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("No valid v values were provided.")
    return vals


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    project_root = Path("/workspace")
    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    points_root = output_root / "points"
    points_root.mkdir(parents=True, exist_ok=True)

    model = load_xtype_model(project_root / "models" / args.model_file)
    v_values = parse_v_values(args.v_values)

    summary_rows: list[dict[str, float | str | int]] = []
    summary_md_lines = [
        "# SOC on/off representative points summary",
        "",
        f"- model_file: `{args.model_file}`",
        f"- fixed_t: `{args.t}`",
        f"- fixed_w: `{args.w}`",
        f"- v_values: `{v_values}`",
        f"- soc_on_lm: `{args.lm_on}`",
        f"- soc_off_lm: `{args.lm_off}`",
        "",
        "|point_id|v|t|w|lm|soc_state|gap|chern|wilson|z2|target_obc_energy|",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for v in v_values:
        for soc_state, lm in (("soc_on", args.lm_on), ("soc_off", args.lm_off)):
            w = float(args.w)
            set_model_params(model, v=v, t=args.t, lm=lm, w=w, j=0.0)

            point_id = f"v_{v:.2f}_{soc_state}"
            point_dir = points_root / point_id
            point_dir.mkdir(parents=True, exist_ok=True)

            gap = compute_bulk_gap(model, nk=args.nk_gap, n_occ=4)
            chern = compute_chern_number(model, nk=args.nk_chern, n_occ=4)
            wilson, z2 = compute_wilson_loop_and_z2(model, nk=args.nk_wilson, n_occ=4)

            band_data = compute_band_data(model)
            plot_band_structure(
                eigvals=band_data,
                save_path=point_dir / "band.png",
                title=f"Band: v={v:.2f}, lm={lm:.2f}, t={args.t:.2f}, w={w:.2f}",
            )

            ky_values, ribbon_eigs = compute_ribbon_spectrum(
                v=v,
                t=args.t,
                lm=lm,
                w=w,
                j=0.0,
                nx=args.ribbon_nx,
                nk=args.ribbon_nk,
            )
            plot_ribbon_spectrum(
                ky_values=ky_values,
                eigvals=ribbon_eigs,
                save_path=point_dir / "ribbon.png",
                title=f"Ribbon: v={v:.2f}, lm={lm:.2f}, t={args.t:.2f}, w={w:.2f}",
            )

            eigvals, target_idx, target_energy, prob_grid = compute_obc_spectrum_and_probability(
                v=v,
                t=args.t,
                lm=lm,
                w=w,
                j=0.0,
                nx=args.obc_nx,
                ny=args.obc_ny,
                mode_count=args.obc_mode_count,
            )
            plot_obc_spectrum(
                eigvals=eigvals,
                save_path=point_dir / "obc_spectrum_e_vs_index.png",
                title=f"OBC spectrum: v={v:.2f}, lm={lm:.2f}, L={args.obc_nx}x{args.obc_ny}",
            )
            plot_obc_wavefunction(
                prob_grid=prob_grid,
                target_energy=target_energy,
                save_path=point_dir / "obc_nearzero_wavefunction.png",
                title=f"OBC near-zero mode: v={v:.2f}, lm={lm:.2f}",
            )

            point_meta = {
                "point_id": point_id,
                "v": v,
                "t": args.t,
                "w": w,
                "lm": lm,
                "soc_state": soc_state,
                "gap": gap,
                "chern": chern,
                "wilson": wilson,
                "z2": int(z2),
                "obc_target_index": int(target_idx),
                "obc_target_energy": float(target_energy),
                "obc_nx": args.obc_nx,
                "obc_ny": args.obc_ny,
                "ribbon_nx": args.ribbon_nx,
                "ribbon_nk": args.ribbon_nk,
            }
            (point_dir / "point_summary.json").write_text(
                json.dumps(point_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            summary_rows.append(point_meta)
            summary_md_lines.append(
                f"|{point_id}|{v:.2f}|{args.t:.2f}|{w:.2f}|{lm:.2f}|{soc_state}|"
                f"{gap:.6e}|{chern:.6e}|{wilson:.6f}|{int(z2)}|{target_energy:.6e}|"
            )

    write_results_csv(
        output_root / "summary.csv",
        [
            {
                "v": float(row["v"]),
                "t": float(row["t"]),
                "lm": float(row["lm"]),
                "gap": float(row["gap"]),
                "chern": float(row["chern"]),
                "edge_state": 0,
                "corner_state": 0,
                "Wilson_loop": float(row["wilson"]),
                "Z2_topology": int(row["z2"]),
            }
            for row in summary_rows
        ],
    )

    (output_root / "summary_full.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text("\n".join(summary_md_lines) + "\n", encoding="utf-8")

    archive_base = project_root / args.archive_name
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_root))
    return output_root, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate representative v-point package for SOC on/off comparisons."
    )
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--v-values", default="0.5,0.8,1.0")
    parser.add_argument("--t", type=float, default=0.5)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm-on", type=float, default=0.2)
    parser.add_argument("--lm-off", type=float, default=0.0)
    parser.add_argument("--nk-gap", type=int, default=11)
    parser.add_argument("--nk-chern", type=int, default=21)
    parser.add_argument("--nk-wilson", type=int, default=15)
    parser.add_argument("--ribbon-nx", type=int, default=40)
    parser.add_argument("--ribbon-nk", type=int, default=121)
    parser.add_argument("--obc-nx", type=int, default=20)
    parser.add_argument("--obc-ny", type=int, default=20)
    parser.add_argument("--obc-mode-count", type=int, default=64)
    parser.add_argument("--output-dir", default="outputs/soc_comparison_v_points")
    parser.add_argument("--archive-name", default="outputs/soc_comparison_v_points_package")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir, zip_file = run(args)
    print(f"[ok] output_dir={out_dir}")
    print(f"[ok] zip_file={zip_file}")
