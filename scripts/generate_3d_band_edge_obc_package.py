#!/usr/bin/env python3
"""Generate 3D package with band, edge-state ribbon, and OBC E-vs-index."""

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

from scripts.run_scan import (  # noqa: E402
    compute_band_data,
    get_band_path_metadata,
    load_xtype_model,
    plot_band_structure,
    plot_obc_spectrum,
    set_model_params,
)


def _state_index(x: int, y: int, orb: int, spin: int, nx: int) -> int:
    return ((y * nx + x) * 8) + (orb * 2 + spin)


def _z_phase(kz_frac: float, model_module) -> tuple[complex, complex]:
    k_vec = float(kz_frac) * model_module.b3
    phase = float(np.dot(k_vec, model_module.a3))
    return np.exp(-1j * phase), np.exp(1j * phase)


def build_onsite_cell_obc_3d_kz(
    t: float,
    v: float,
    lm: float,
    j: float,
    z_minus: complex,
    z_plus: complex,
) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t * (1.0 + z_minus)
    h0[0, 2] = t * (1.0 + z_minus)
    h0[0, 3] = v
    h0[1, 0] = t * (1.0 + z_plus)
    h0[1, 2] = v
    h0[1, 3] = t * (1.0 + z_plus)
    h0[2, 0] = t * (1.0 + z_plus)
    h0[2, 1] = v
    h0[2, 3] = t * (1.0 + z_plus)
    h0[3, 0] = v
    h0[3, 1] = t * (1.0 + z_minus)
    h0[3, 2] = t * (1.0 + z_minus)

    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm * (1.0 + z_minus)
    h1[0, 2] = -1j * lm * (1.0 + z_minus)
    h1[2, 0] = 1j * lm * (1.0 + z_plus)
    h1[2, 3] = -1j * lm * (1.0 + z_plus)
    h1[3, 2] = 1j * lm * (1.0 + z_minus)
    h1[3, 1] = -1j * lm * (1.0 + z_minus)
    h1[1, 0] = -1j * lm * (1.0 + z_plus)
    h1[1, 3] = 1j * lm * (1.0 + z_plus)

    h2 = np.diag([j, j, j, j]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h1, sz) + np.kron(h2, sz)


def build_onsite_cell_ribbon_3d_kz_ky(
    t: float,
    v: float,
    lm: float,
    j: float,
    w: float,
    ky: float,
    z_minus: complex,
    z_plus: complex,
) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t * (1.0 + z_minus)
    h0[0, 2] = t * (1.0 + z_minus)
    h0[0, 3] = v
    h0[1, 0] = t * (1.0 + z_plus)
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t * (1.0 + z_plus)
    h0[2, 0] = t * (1.0 + z_plus)
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t * (1.0 + z_plus)
    h0[3, 0] = v
    h0[3, 1] = t * (1.0 + z_minus)
    h0[3, 2] = t * (1.0 + z_minus)

    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm * (1.0 + z_minus)
    h1[0, 2] = -1j * lm * (1.0 + z_minus)
    h1[2, 0] = 1j * lm * (1.0 + z_plus)
    h1[2, 3] = -1j * lm * (1.0 + z_plus)
    h1[3, 2] = 1j * lm * (1.0 + z_minus)
    h1[3, 1] = -1j * lm * (1.0 + z_minus)
    h1[1, 0] = -1j * lm * (1.0 + z_plus)
    h1[1, 3] = 1j * lm * (1.0 + z_plus)

    h2 = np.diag([j, j, j, j]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h1, sz) + np.kron(h2, sz)


def build_obc_hamiltonian_3d_kz(
    v: float,
    t: float,
    lm: float,
    w: float,
    j: float,
    nx: int,
    ny: int,
    z_minus: complex,
    z_plus: complex,
) -> np.ndarray:
    dim = nx * ny * 8
    ham = np.zeros((dim, dim), dtype=complex)
    onsite = build_onsite_cell_obc_3d_kz(t=t, v=v, lm=lm, j=j, z_minus=z_minus, z_plus=z_plus)

    for y in range(ny):
        for x in range(nx):
            start = (y * nx + x) * 8
            ham[start : start + 8, start : start + 8] += onsite

            if x > 0:
                for spin in range(2):
                    i_idx = _state_index(x, y, 0, spin, nx)
                    jx = _state_index(x - 1, y, 3, spin, nx)
                    ham[i_idx, jx] += w
                    ham[jx, i_idx] += w

            if y > 0:
                for spin in range(2):
                    i_idx = _state_index(x, y, 1, spin, nx)
                    jy = _state_index(x, y - 1, 2, spin, nx)
                    ham[i_idx, jy] += w
                    ham[jy, i_idx] += w
    return ham


def build_ribbon_hamiltonian_3d_kz(
    v: float,
    t: float,
    lm: float,
    w: float,
    j: float,
    ky: float,
    nx: int,
    z_minus: complex,
    z_plus: complex,
) -> np.ndarray:
    dim = 8 * nx
    ham = np.zeros((dim, dim), dtype=complex)
    onsite = build_onsite_cell_ribbon_3d_kz_ky(
        t=t, v=v, lm=lm, j=j, w=w, ky=ky, z_minus=z_minus, z_plus=z_plus
    )
    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = w

    for x in range(nx):
        start = x * 8
        ham[start : start + 8, start : start + 8] += onsite
        if x > 0:
            prev = (x - 1) * 8
            ham[start : start + 8, prev : prev + 8] += tx
            ham[prev : prev + 8, start : start + 8] += tx.conj().T
    return ham


def compute_ribbon_spectrum_3d_kz(
    v: float,
    t: float,
    lm: float,
    w: float,
    j: float,
    kz_frac: float,
    model_module,
    nx: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_minus, z_plus = _z_phase(kz_frac, model_module)
    ky_values = np.linspace(-np.pi, np.pi, nk)
    eigvals_all = []

    for ky in ky_values:
        ham = build_ribbon_hamiltonian_3d_kz(
            v=v, t=t, lm=lm, w=w, j=j, ky=ky, nx=nx, z_minus=z_minus, z_plus=z_plus
        )
        evals = np.linalg.eigvalsh(ham)
        evals = np.real(evals)
        eigvals_all.append(evals)

    return ky_values, np.array(eigvals_all)


def plot_ribbon_spectrum_2d_style(
    ky_values: np.ndarray,
    eigvals: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    xvals = ky_values / np.pi
    for band_idx in range(eigvals.shape[1]):
        ax.plot(xvals, eigvals[:, band_idx], linewidth=0.6, color="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y / \pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path("/workspace")
    model = load_xtype_model(root / "models" / args.model_file)
    path_ticks, path_labels = get_band_path_metadata(model)

    input_root = root / args.input_dir
    input_points = input_root / "points"
    if not input_points.exists():
        raise FileNotFoundError(f"Missing input points directory: {input_points}")

    output_root = root / args.output_dir
    if output_root.exists() and args.clean_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_points = output_root / "points"
    output_points.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float | int]] = []
    for point_dir in sorted(input_points.iterdir()):
        if not point_dir.is_dir():
            continue
        summary_path = point_dir / "point_summary_3d.json"
        if not summary_path.exists():
            continue
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
        point_id = str(meta["point_id"])
        v = float(meta["v"])
        t = float(meta["t"])
        w = float(meta["w"])
        lm = float(meta["lm"])
        j = float(meta.get("J", 0.0))

        out_dir = output_points / point_id
        out_dir.mkdir(parents=True, exist_ok=True)

        set_model_params(model, v=v, t=t, lm=lm, w=w, j=j)
        band = compute_band_data(model)
        plot_band_structure(
            eigvals=band,
            save_path=out_dir / "band.png",
            title=f"3D Band: {point_id}",
            path_ticks=path_ticks,
            path_labels=path_labels,
        )

        ky, ribbon_eigs = compute_ribbon_spectrum_3d_kz(
            v=v,
            t=t,
            lm=lm,
            w=w,
            j=j,
            kz_frac=args.kz_frac,
            model_module=model,
            nx=args.ribbon_nx,
            nk=args.ribbon_nk,
        )
        plot_ribbon_spectrum_2d_style(
            ky_values=ky,
            eigvals=ribbon_eigs,
            save_path=out_dir / "ribbon.png",
            title=f"Ribbon @ kz={args.kz_frac:.2f}: {point_id}",
        )

        z_minus, z_plus = _z_phase(args.kz_frac, model)
        obc_h = build_obc_hamiltonian_3d_kz(
            v=v,
            t=t,
            lm=lm,
            w=w,
            j=j,
            nx=args.obc_nx,
            ny=args.obc_ny,
            z_minus=z_minus,
            z_plus=z_plus,
        )
        obc_eigs = np.real(np.linalg.eigvalsh(obc_h))
        plot_obc_spectrum(
            eigvals=obc_eigs,
            save_path=out_dir / "obc_spectrum_e_vs_index.png",
            title=f"OBC E vs index @ kz={args.kz_frac:.2f}: {point_id}",
        )

        min_abs_e = float(np.min(np.abs(obc_eigs)))
        row = {
            "point_id": point_id,
            "v": v,
            "t": t,
            "w": w,
            "lm": lm,
            "soc_state": str(meta.get("soc_state", "")),
            "kz_frac": float(args.kz_frac),
            "ribbon_nx": int(args.ribbon_nx),
            "ribbon_nk": int(args.ribbon_nk),
            "obc_nx": int(args.obc_nx),
            "obc_ny": int(args.obc_ny),
            "obc_total_modes": int(obc_eigs.size),
            "obc_min_abs_energy": min_abs_e,
        }
        rows.append(row)
        (out_dir / "point_summary_edge_obc.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary_csv = output_root / "summary_3d_band_edge_obc.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "point_id",
            "v",
            "t",
            "w",
            "lm",
            "soc_state",
            "kz_frac",
            "ribbon_nx",
            "ribbon_nk",
            "obc_nx",
            "obc_ny",
            "obc_total_modes",
            "obc_min_abs_energy",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    readme = output_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# 3D band + edge + OBC E-index package",
                "",
                f"- source_input: `{input_root}`",
                f"- model_file: `{args.model_file}`",
                f"- kz_frac_for_edge_and_obc: `{args.kz_frac}`",
                f"- ribbon_nx: `{args.ribbon_nx}`",
                f"- ribbon_nk: `{args.ribbon_nk}`",
                f"- obc_size: `{args.obc_nx}x{args.obc_ny}`",
                "",
                "Each point folder includes:",
                "- `band.png`",
                "- `ribbon.png`",
                "- `obc_spectrum_e_vs_index.png`",
                "- `point_summary_edge_obc.json`",
                "",
                f"Summary CSV: `{summary_csv.name}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    archive_base = root / args.archive_name
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=output_root))
    return output_root, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D band + edge + OBC package.")
    parser.add_argument("--model-file", default="xtype_model_3d.py")
    parser.add_argument("--input-dir", default="outputs/soc_comparison_3d")
    parser.add_argument("--output-dir", default="outputs/soc_comparison_3d_band_edge_obc")
    parser.add_argument("--archive-name", default="outputs/soc_comparison_3d_band_edge_obc_package")
    parser.add_argument("--kz-frac", type=float, default=0.0)
    parser.add_argument("--ribbon-nx", type=int, default=40)
    parser.add_argument("--ribbon-nk", type=int, default=121)
    parser.add_argument("--obc-nx", type=int, default=12)
    parser.add_argument("--obc-ny", type=int, default=12)
    parser.add_argument("--clean-output", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    output_dir, zip_file = run(parse_args())
    print(f"[ok] output_dir={output_dir}")
    print(f"[ok] zip_file={zip_file}")
