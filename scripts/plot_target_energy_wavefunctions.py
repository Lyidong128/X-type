#!/usr/bin/env python3
"""Plot OBC target-energy wavefunctions and marked E-vs-index spectra."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import build_obc_hamiltonian


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate wavefunction maps near a target OBC energy and mark target on E-vs-index plots."
    )
    parser.add_argument("--package-root", default="/workspace/outputs/soc_comparison_v_points")
    parser.add_argument("--soc-state", default="soc_on")
    parser.add_argument("--lm", type=float, default=0.1)
    parser.add_argument("--target-energy", type=float, default=0.2)
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=20)
    parser.add_argument("--output-subdir", default="target_energy_E0p2_soc_on")
    return parser.parse_args()


def load_points(summary_path: Path, soc_state: str, lm: float) -> list[dict[str, float | str]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = []
    for row in data:
        if str(row.get("soc_state", "")) != soc_state:
            continue
        if abs(float(row.get("lm", 0.0)) - float(lm)) > 1e-10:
            continue
        selected.append(row)
    selected.sort(key=lambda r: float(r["v"]))
    return selected


def compute_prob_grid(vec: np.ndarray, nx: int, ny: int) -> np.ndarray:
    probs = np.zeros(nx * ny, dtype=float)
    for cell_id in range(nx * ny):
        start = cell_id * 8
        probs[cell_id] = float(np.sum(np.abs(vec[start : start + 8]) ** 2))
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


def run(args: argparse.Namespace) -> Path:
    package_root = Path(args.package_root)
    summary_path = package_root / "summary_full.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    points = load_points(summary_path, soc_state=args.soc_state, lm=args.lm)
    if not points:
        raise RuntimeError(
            f"No points found for soc_state={args.soc_state}, lm={args.lm} in {summary_path}"
        )

    out_root = package_root / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []

    for row in points:
        v = float(row["v"])
        t = float(row["t"])
        w = float(row["w"])
        lm = float(row["lm"])
        point_id = str(row["point_id"])
        point_dir = out_root / point_id
        point_dir.mkdir(parents=True, exist_ok=True)

        ham = build_obc_hamiltonian(v=v, t=t, lm=lm, w=w, j=0.0, nx=args.nx, ny=args.ny)
        eigvals, eigvecs = np.linalg.eigh(ham)
        eigvals = np.real(eigvals)

        idx = int(np.argmin(np.abs(eigvals - args.target_energy)))
        selected_energy = float(eigvals[idx])
        delta = float(abs(selected_energy - args.target_energy))
        vec = eigvecs[:, idx]
        prob_grid = compute_prob_grid(vec, nx=args.nx, ny=args.ny)

        wave_path = point_dir / "obc_wavefunction_target_E0p2.png"
        spec_path = point_dir / "obc_spectrum_e_vs_index_marked_E0p2.png"
        title_suffix = f"v={v:.2f}, lm={lm:.2f}, target E={args.target_energy:.2f}"
        plot_wavefunction(
            prob_grid=prob_grid,
            save_path=wave_path,
            title=f"OBC wavefunction near {title_suffix}\nselected E={selected_energy:.5f} (idx={idx})",
        )
        plot_spectrum_marked(
            eigvals=eigvals,
            target_energy=args.target_energy,
            target_idx=idx,
            target_eval=selected_energy,
            save_path=spec_path,
            title=f"OBC E vs index ({title_suffix})",
        )

        point_info = {
            "point_id": point_id,
            "v": v,
            "t": t,
            "w": w,
            "lm": lm,
            "target_energy": float(args.target_energy),
            "selected_index": idx,
            "selected_energy": selected_energy,
            "abs_delta": delta,
            "nx": int(args.nx),
            "ny": int(args.ny),
            "wavefunction_path": str(wave_path.relative_to(package_root)),
            "spectrum_path": str(spec_path.relative_to(package_root)),
        }
        summary_rows.append(point_info)
        (point_dir / "target_E0p2_selection.json").write_text(
            json.dumps(point_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    csv_path = out_root / "target_energy_selection_summary.csv"
    fields = [
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
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for item in summary_rows:
            writer.writerow(item)

    readme = out_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Target-energy OBC wavefunction maps",
                "",
                f"- package_root: `{package_root}`",
                f"- soc_state: `{args.soc_state}`",
                f"- lm: `{args.lm}`",
                f"- target_energy: `{args.target_energy}`",
                f"- lattice: `{args.nx} x {args.ny}`",
                "",
                "Per point outputs:",
                "- `obc_wavefunction_target_E0p2.png`",
                "- `obc_spectrum_e_vs_index_marked_E0p2.png`",
                "- `target_E0p2_selection.json`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return out_root


if __name__ == "__main__":
    output_dir = run(parse_args())
    print(f"[ok] output_dir={output_dir}")
