#!/usr/bin/env python3
"""Batch-generate OBC wavefunction maps and marked E-vs-index spectra."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian


def build_energy_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be > 0")
    values = []
    x = start
    eps = 1e-12
    if end >= start:
        while x <= end + eps:
            values.append(round(x, 10))
            x += step
    else:
        while x >= end - eps:
            values.append(round(x, 10))
            x -= step
    return values


def energy_token(energy: float) -> str:
    return f"{energy:.3f}".replace("-", "m").replace(".", "p")


def load_points(summary_path: Path, soc_state: str, lm: float) -> list[dict[str, float | str]]:
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        if str(row.get("soc_state", "")) != soc_state:
            continue
        if abs(float(row.get("lm", 0.0)) - lm) > 1e-10:
            continue
        out.append(row)
    out.sort(key=lambda r: float(r["v"]))
    return out


def compute_prob_grid(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    probs = np.zeros(nx * ny, dtype=float)
    for cid in range(nx * ny):
        start = cid * 8
        probs[cid] = float(np.sum(np.abs(vec[start : start + 8]) ** 2))
    return probs.reshape((ny, nx))


def plot_wavefunction(prob_grid: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(prob_grid, origin="lower", cmap="magma")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=r"$|\psi|^2$")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_spectrum_marked(
    eigvals: np.ndarray,
    target_energy: float,
    target_idx: int,
    target_eval: float,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = np.arange(eigvals.size)
    ax.scatter(x, eigvals, s=8, c="tab:blue", alpha=0.9, label="OBC eigenvalues")
    ax.scatter([target_idx], [target_eval], s=45, c="red", marker="*", zorder=4, label="Nearest state")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, label="E=0")
    ax.axhline(target_energy, color="red", linestyle="--", linewidth=0.9, label=f"E target={target_energy:.3f}")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-point OBC wavefunctions for energy sweeps.")
    parser.add_argument("--package-root", default="/workspace/outputs/soc_comparison_v_points")
    parser.add_argument("--soc-state", default="soc_on")
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--special-v", type=float, default=1.0)
    parser.add_argument("--special-start", type=float, default=-0.5)
    parser.add_argument("--special-end", type=float, default=0.0)
    parser.add_argument("--special-step", type=float, default=0.1)
    parser.add_argument("--default-start", type=float, default=0.0)
    parser.add_argument("--default-end", type=float, default=0.5)
    parser.add_argument("--default-step", type=float, default=0.1)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--output-subdir", default="energy_sweep_v1_neg_rest_pos_soc_on_lm0p1")
    parser.add_argument("--archive-name", default="soc_comparison_v_points_energy_sweep_v1_neg_rest_pos_soc_on_lm0p1")
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    package_root = Path(args.package_root)
    summary_path = package_root / "summary_full.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    points = load_points(summary_path, soc_state=args.soc_state, lm=args.lm)
    if not points:
        raise RuntimeError("No matching points found in summary_full.json")

    out_root = package_root / args.output_subdir
    if out_root.exists() and args.clean_output:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    special_energies = build_energy_values(args.special_start, args.special_end, args.special_step)
    default_energies = build_energy_values(args.default_start, args.default_end, args.default_step)
    global_rows: list[dict[str, float | int | str]] = []

    for row in points:
        point_id = str(row["point_id"])
        v = float(row["v"])
        t = float(row["t"])
        w = float(row["w"])
        lm = float(row["lm"])
        point_dir = out_root / point_id
        point_dir.mkdir(parents=True, exist_ok=True)
        energies = special_energies if abs(v - args.special_v) < 1e-10 else default_energies

        ham = build_obc_hamiltonian(v=v, t=t, lm=lm, w=w, j=0.0, nx=args.nx, ny=args.ny)
        eigvals, eigvecs = np.linalg.eigh(ham)
        eigvals = np.real(eigvals)

        local_rows: list[dict[str, float | int | str]] = []
        for target_energy in energies:
            idx = int(np.argmin(np.abs(eigvals - target_energy)))
            selected_energy = float(eigvals[idx])
            abs_delta = float(abs(selected_energy - target_energy))
            token = energy_token(target_energy)
            vec = eigvecs[:, idx]
            prob_grid = compute_prob_grid(vec, nx=args.nx, ny=args.ny)

            wave_path = point_dir / f"obc_wavefunction_target_E{token}.png"
            spec_path = point_dir / f"obc_spectrum_e_vs_index_marked_E{token}.png"
            title_suffix = f"v={v:.2f}, lm={lm:.2f}, target E={target_energy:.2f}"
            plot_wavefunction(
                prob_grid=prob_grid,
                save_path=wave_path,
                title=f"OBC wavefunction near {title_suffix}\nselected E={selected_energy:.5f} (idx={idx})",
            )
            plot_spectrum_marked(
                eigvals=eigvals,
                target_energy=target_energy,
                target_idx=idx,
                target_eval=selected_energy,
                save_path=spec_path,
                title=f"OBC E vs index ({title_suffix})",
            )

            result = {
                "point_id": point_id,
                "v": v,
                "t": t,
                "w": w,
                "lm": lm,
                "target_energy": float(target_energy),
                "selected_index": idx,
                "selected_energy": selected_energy,
                "abs_delta": abs_delta,
                "nx": int(args.nx),
                "ny": int(args.ny),
                "wavefunction_path": str(wave_path.relative_to(package_root)),
                "spectrum_path": str(spec_path.relative_to(package_root)),
            }
            local_rows.append(result)
            global_rows.append(result)

        local_csv = point_dir / "energy_sweep_selection_summary.csv"
        with local_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "point_id",
                    "v",
                    "t",
                    "w",
                    "lm",
                    "target_energy",
                    "selected_index",
                    "selected_energy",
                    "abs_delta",
                    "nx",
                    "ny",
                    "wavefunction_path",
                    "spectrum_path",
                ],
            )
            writer.writeheader()
            for item in local_rows:
                writer.writerow(item)

    global_csv = out_root / "energy_sweep_selection_summary_all_points.csv"
    with global_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "point_id",
                "v",
                "t",
                "w",
                "lm",
                "target_energy",
                "selected_index",
                "selected_energy",
                "abs_delta",
                "nx",
                "ny",
                "wavefunction_path",
                "spectrum_path",
            ],
        )
        writer.writeheader()
        for item in global_rows:
            writer.writerow(item)

    readme = out_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# OBC energy-sweep wavefunction package",
                "",
                f"- package_root: `{package_root}`",
                f"- soc_state: `{args.soc_state}`",
                f"- lm: `{args.lm}`",
                f"- special_v: `{args.special_v}`",
                f"- special_energy_range: `[{args.special_start}, {args.special_end}] step {args.special_step}`",
                f"- default_energy_range: `[{args.default_start}, {args.default_end}] step {args.default_step}`",
                f"- lattice: `{args.nx} x {args.ny}`",
                "",
                "Per point outputs:",
                "- `obc_wavefunction_target_E*.png`",
                "- `obc_spectrum_e_vs_index_marked_E*.png`",
                "- `energy_sweep_selection_summary.csv`",
                "",
                "Global summary:",
                "- `energy_sweep_selection_summary_all_points.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    archive_path = Path(
        shutil.make_archive(str(package_root / args.archive_name), "zip", root_dir=out_root)
    )
    return out_root, archive_path


if __name__ == "__main__":
    out_dir, archive = run(parse_args())
    print(f"[ok] output_dir={out_dir}")
    print(f"[ok] zip_file={archive}")
