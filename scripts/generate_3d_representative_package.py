#!/usr/bin/env python3
"""Generate representative 3D result figures and package them."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (
    compute_band_data,
    compute_bulk_gap,
    load_xtype_model,
    plot_band_structure,
    set_model_params,
)


def parse_values(raw: str) -> list[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("No valid numeric values parsed.")
    return vals


def compute_wilson_winding_z2_on_slice(
    model_module,
    kz_frac: float,
    nk: int = 15,
    n_occ: int = 4,
) -> tuple[float, int]:
    """Wilson winding/Z2 on a fixed-kz 2D slice."""

    def occ_space(kvec: np.ndarray) -> np.ndarray:
        _, evecs = np.linalg.eigh(model_module.Hxtype(kvec))
        return evecs[:, :n_occ]

    k0 = kz_frac * model_module.b3
    phases = []
    for j in range(nk):
        ky = j / nk
        wilson = 1.0 + 0.0j
        prev = occ_space(k0 + 0.0 * model_module.b1 + ky * model_module.b2)
        for i in range(1, nk + 1):
            kx = i / nk
            curr = occ_space(k0 + kx * model_module.b1 + ky * model_module.b2)
            overlap = prev.conj().T @ curr
            det_overlap = np.linalg.det(overlap)
            if abs(det_overlap) > 1e-14:
                wilson *= det_overlap / abs(det_overlap)
            prev = curr
        phases.append(float(np.angle(wilson)))

    phases = np.array(phases, dtype=float)
    unwrapped = np.unwrap(phases)
    winding = float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))
    z2 = int(round(abs(winding))) % 2
    return winding, z2


def compute_slice_gap(
    model_module,
    kz_frac: float,
    nk: int = 11,
    n_occ: int = 4,
) -> float:
    """Bulk gap on fixed-kz 2D BZ slice."""
    valence_max = -np.inf
    conduction_min = np.inf
    k0 = kz_frac * model_module.b3
    for i in range(nk):
        for j in range(nk):
            ux = i / (nk - 1)
            uy = j / (nk - 1)
            k = k0 + ux * model_module.b1 + uy * model_module.b2
            evals = np.linalg.eigvalsh(model_module.Hxtype(k))
            valence_max = max(valence_max, float(np.real(evals[n_occ - 1])))
            conduction_min = min(conduction_min, float(np.real(evals[n_occ])))
    return float(conduction_min - valence_max)


def save_kz_line_plot(
    kz_values: np.ndarray,
    y_values: np.ndarray,
    save_path: Path,
    title: str,
    ylabel: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(kz_values, y_values, marker="o", linewidth=1.2, markersize=3.5)
    ax.set_xlabel("kz fraction (kz / b3)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def save_kz_wilson_plot(
    kz_values: np.ndarray,
    windings: np.ndarray,
    z2_values: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.6, 4.3))
    ax1.plot(kz_values, windings, color="tab:blue", marker="o", linewidth=1.2, markersize=3.3)
    ax1.set_xlabel("kz fraction (kz / b3)")
    ax1.set_ylabel("Wilson winding", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.step(kz_values, z2_values, where="mid", color="tab:red", linewidth=1.2)
    ax2.set_ylabel("Z2", color="tab:red")
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    project_root = Path("/workspace")
    out_root = project_root / args.output_dir
    if out_root.exists() and args.clean_output:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    points_root = out_root / "points"
    points_root.mkdir(parents=True, exist_ok=True)

    model = load_xtype_model(project_root / "models" / args.model_file)
    v_values = parse_values(args.v_values)
    kz_values = np.linspace(0.0, 1.0, args.kz_slices, endpoint=False)
    soc_cfg = [("soc_on", args.lm_on), ("soc_off", args.lm_off)] if not args.soc_only else [("soc_on", args.lm_on)]

    summary_rows: list[dict[str, float | int | str]] = []

    for v in v_values:
        for soc_state, lm in soc_cfg:
            point_id = f"v_{v:.2f}_{soc_state}"
            point_dir = points_root / point_id
            point_dir.mkdir(parents=True, exist_ok=True)

            set_model_params(model, v=v, t=args.t, lm=lm, w=args.w, j=args.j)

            # 3D band path (uses PATH_SEGMENTS in xtype_model_3d.py).
            band_data = compute_band_data(model)
            plot_band_structure(
                eigvals=band_data,
                save_path=point_dir / "band_3d_path.png",
                title=f"3D Band Path: v={v:.2f}, lm={lm:.2f}, t={args.t:.2f}, w={args.w:.2f}",
            )

            # kz-slice diagnostics.
            chern_vals = []
            gap_vals = []
            wilson_vals = []
            z2_vals = []
            for kz in kz_values:
                k_origin = kz * model.b3
                ch = float(model.chern_number_fukui(nk=args.nk_chern, n_occ=4, k_origin=k_origin))
                gp = compute_slice_gap(model, kz_frac=kz, nk=args.nk_gap, n_occ=4)
                ww, zz = compute_wilson_winding_z2_on_slice(model, kz_frac=kz, nk=args.nk_wilson, n_occ=4)
                chern_vals.append(ch)
                gap_vals.append(gp)
                wilson_vals.append(ww)
                z2_vals.append(zz)

            chern_arr = np.array(chern_vals, dtype=float)
            gap_arr = np.array(gap_vals, dtype=float)
            wilson_arr = np.array(wilson_vals, dtype=float)
            z2_arr = np.array(z2_vals, dtype=int)

            save_kz_line_plot(
                kz_values=kz_values,
                y_values=chern_arr,
                save_path=point_dir / "kz_slice_chern.png",
                title=f"kz-slice Chern: {point_id}",
                ylabel="Chern(kz-slice)",
            )
            save_kz_line_plot(
                kz_values=kz_values,
                y_values=gap_arr,
                save_path=point_dir / "kz_slice_gap.png",
                title=f"kz-slice gap: {point_id}",
                ylabel="Gap(kz-slice)",
            )
            save_kz_wilson_plot(
                kz_values=kz_values,
                windings=wilson_arr,
                z2_values=z2_arr,
                save_path=point_dir / "kz_slice_wilson_z2.png",
                title=f"kz-slice Wilson/Z2: {point_id}",
            )

            # Global 3D-style summary metric at kz=0 slice for compatibility.
            gap_kz0 = compute_bulk_gap(model, nk=args.nk_gap, n_occ=4)
            chern_kz0 = float(model.chern_number_fukui(nk=args.nk_chern, n_occ=4, k_origin=np.zeros(3)))
            wilson_kz0, z2_kz0 = compute_wilson_winding_z2_on_slice(model, kz_frac=0.0, nk=args.nk_wilson, n_occ=4)

            point_meta = {
                "point_id": point_id,
                "v": float(v),
                "t": float(args.t),
                "w": float(args.w),
                "lm": float(lm),
                "J": float(args.j),
                "soc_state": soc_state,
                "kz_slices": int(args.kz_slices),
                "kz_grid_start_end": [float(kz_values[0]), float(kz_values[-1])],
                "gap_kz0": float(gap_kz0),
                "chern_kz0": float(chern_kz0),
                "wilson_kz0": float(wilson_kz0),
                "z2_kz0": int(z2_kz0),
                "chern_kz_mean": float(np.mean(chern_arr)),
                "chern_kz_std": float(np.std(chern_arr)),
                "gap_kz_min": float(np.min(gap_arr)),
                "gap_kz_max": float(np.max(gap_arr)),
                "z2_kz_ones_count": int(np.sum(z2_arr)),
            }
            (point_dir / "point_summary_3d.json").write_text(json.dumps(point_meta, indent=2), encoding="utf-8")
            np.savetxt(
                point_dir / "kz_slice_data.csv",
                np.column_stack([kz_values, chern_arr, gap_arr, wilson_arr, z2_arr]),
                delimiter=",",
                header="kz_frac,chern_slice,gap_slice,wilson_winding,z2",
                comments="",
            )
            summary_rows.append(point_meta)

    # Global heatmaps: (v, kz) for chern/gap and soc_on/soc_off separately.
    for soc_state in {str(r["soc_state"]) for r in summary_rows}:
        rows = [r for r in summary_rows if str(r["soc_state"]) == soc_state]
        fig, ax = plt.subplots(figsize=(6.0, 4.3))
        xs = np.array([float(r["v"]) for r in rows], dtype=float)
        ys = np.array([float(r["chern_kz_mean"]) for r in rows], dtype=float)
        ax.scatter(xs, ys, s=62, c=ys, cmap="RdBu_r", edgecolors="black", linewidths=0.3)
        ax.set_xlabel("v")
        ax.set_ylabel("mean Chern(kz)")
        ax.set_title(f"3D summary mean Chern vs v ({soc_state})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_root / f"summary_mean_chern_vs_v_{soc_state}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 4.3))
        ys_gap = np.array([float(r["gap_kz_min"]) for r in rows], dtype=float)
        ax.scatter(xs, ys_gap, s=62, c=ys_gap, cmap="viridis", edgecolors="black", linewidths=0.3)
        ax.set_xlabel("v")
        ax.set_ylabel("min Gap(kz)")
        ax.set_title(f"3D summary min gap vs v ({soc_state})")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_root / f"summary_min_gap_vs_v_{soc_state}.png", dpi=180)
        plt.close(fig)

    # Summary files.
    summary_csv = out_root / "summary_3d.csv"
    header = [
        "point_id",
        "v",
        "t",
        "w",
        "lm",
        "J",
        "soc_state",
        "kz_slices",
        "gap_kz0",
        "chern_kz0",
        "wilson_kz0",
        "z2_kz0",
        "chern_kz_mean",
        "chern_kz_std",
        "gap_kz_min",
        "gap_kz_max",
        "z2_kz_ones_count",
    ]
    with summary_csv.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for r in summary_rows:
            fh.write(",".join(str(r[k]) for k in header) + "\n")

    readme = [
        "# 3D representative results package",
        "",
        f"- model_file: `{args.model_file}`",
        f"- fixed_t: `{args.t}`",
        f"- fixed_w: `{args.w}`",
        f"- fixed_J: `{args.j}`",
        f"- v_values: `{v_values}`",
        f"- soc_on_lm: `{args.lm_on}`",
        f"- soc_off_lm: `{args.lm_off}`",
        f"- soc_only: `{args.soc_only}`",
        f"- kz_slices: `{args.kz_slices}`",
        "",
        "Each point folder includes:",
        "- `band_3d_path.png`",
        "- `kz_slice_chern.png`",
        "- `kz_slice_gap.png`",
        "- `kz_slice_wilson_z2.png`",
        "- `kz_slice_data.csv`",
        "- `point_summary_3d.json`",
        "",
        f"Summary CSV: `{summary_csv.name}`",
    ]
    (out_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (out_root / "summary_3d_full.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    archive_base = project_root / args.archive_name
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=out_root))
    return out_root, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D representative result package.")
    parser.add_argument("--model-file", default="xtype_model_3d.py")
    parser.add_argument("--v-values", default="0.1,0.3,0.5,0.8,1.0")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--j", type=float, default=0.0)
    parser.add_argument("--lm-on", type=float, default=0.1)
    parser.add_argument("--lm-off", type=float, default=0.0)
    parser.add_argument("--soc-only", action="store_true")
    parser.add_argument("--nk-gap", type=int, default=11)
    parser.add_argument("--nk-chern", type=int, default=15)
    parser.add_argument("--nk-wilson", type=int, default=15)
    parser.add_argument("--kz-slices", type=int, default=21)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="outputs/soc_comparison_3d")
    parser.add_argument("--archive-name", default="outputs/soc_comparison_3d_package")
    return parser.parse_args()


if __name__ == "__main__":
    out_dir, zip_file = run(parse_args())
    print(f"[ok] output_dir={out_dir}")
    print(f"[ok] zip_file={zip_file}")
