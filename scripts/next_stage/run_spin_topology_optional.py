#!/usr/bin/env python3
"""Optional spin-topology analysis (spin Chern / spin Bott)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.next_stage.common import (
    auto_select_points,
    infer_spin_operator,
    load_model,
    set_params,
    write_csv,
    write_text,
)
from scripts.run_scan import build_obc_hamiltonian_sparse


def subspace_chern(frames: list[list[np.ndarray]]) -> tuple[float, np.ndarray]:
    """Fukui Chern number from subspace frame grid."""
    nk = len(frames)
    Ux = np.zeros((nk, nk), dtype=complex)
    Uy = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            a = frames[i][j]
            bx = frames[ip][j]
            by = frames[i][jp]
            mx = a.conj().T @ bx
            my = a.conj().T @ by
            dx = np.linalg.det(mx)
            dy = np.linalg.det(my)
            Ux[i, j] = dx / (abs(dx) + 1e-14)
            Uy[i, j] = dy / (abs(dy) + 1e-14)
    F = np.zeros((nk, nk), dtype=complex)
    for i in range(nk):
        ip = (i + 1) % nk
        for j in range(nk):
            jp = (j + 1) % nk
            F[i, j] = np.log(Ux[i, j] * Uy[ip, j] / (Ux[i, jp] * Uy[i, j] + 1e-14))
    chern = float(np.sum(F) / (2j * np.pi))
    return chern, np.real(F / (2j * np.pi))


def unitary_part(mat: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(mat, full_matrices=False)
    return u @ vh


def spin_bott_from_occ(occ: np.ndarray, spin_occ: np.ndarray, nx: int, ny: int) -> float:
    """Compute Bott index for one occupied subspace."""
    if occ.shape[1] == 0:
        return 0.0
    dim = nx * ny * 8
    px = np.zeros(dim, dtype=complex)
    py = np.zeros(dim, dtype=complex)
    idx = 0
    for y in range(ny):
        for x in range(nx):
            phx = np.exp(1j * 2.0 * np.pi * x / max(nx, 1))
            phy = np.exp(1j * 2.0 * np.pi * y / max(ny, 1))
            for _ in range(8):
                px[idx] = phx
                py[idx] = phy
                idx += 1
    mx = spin_occ.conj().T @ (px[:, None] * spin_occ)
    my = spin_occ.conj().T @ (py[:, None] * spin_occ)
    ux = unitary_part(mx)
    uy = unitary_part(my)
    w = uy @ ux @ uy.conj().T @ ux.conj().T
    angles = np.angle(np.linalg.eigvals(w))
    return float(np.sum(angles) / (2.0 * np.pi))


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional spin Chern / spin Bott analysis.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/outputs/next_stage_analysis/spin_topology_optional"),
    )
    parser.add_argument("--nk", type=int, default=21)
    parser.add_argument("--obc-size", type=int, default=12)
    parser.add_argument("--bott-max-dim", type=int, default=2200)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    nk = 13 if args.quick else args.nk

    model = load_model(args.model_file)
    spin_op, reason = infer_spin_operator(model)
    if spin_op is None:
        write_text(out_root / "skip_reason.txt", reason + "\n")
        print(f"[skip] spin topology: {reason}")
        return

    points, _ = auto_select_points(model_path=args.model_file)
    pmap = {p.label: p for p in points}
    labels = [lb for lb in ("A", "B", "C", "D") if lb in pmap]
    rows = []
    run_notes = [f"spin_operator={reason}"]

    for label in labels:
        p = pmap[label]
        set_params(model, v=p.v, lm=p.lm, t=p.t)

        grid = np.linspace(0.0, 1.0, nk, endpoint=False)
        frames_plus = [[None for _ in range(nk)] for __ in range(nk)]
        frames_minus = [[None for _ in range(nk)] for __ in range(nk)]
        for i, ux in enumerate(grid):
            for j, uy in enumerate(grid):
                k = ux * model.b1 + uy * model.b2
                evals, evecs = np.linalg.eigh(model.Hxtype(k))
                occ = evecs[:, :4]
                s_occ = occ.conj().T @ spin_op @ occ
                se, sv = np.linalg.eigh(s_occ)
                # split by sign of projected spin eigenvalues
                plus_mask = se >= 0
                minus_mask = se < 0
                if np.sum(plus_mask) == 0 or np.sum(minus_mask) == 0:
                    # fallback split half-half by sorted eigenvalues
                    order = np.argsort(se)
                    minus_idx = order[:2]
                    plus_idx = order[2:]
                    frames_plus[i][j] = occ @ sv[:, plus_idx]
                    frames_minus[i][j] = occ @ sv[:, minus_idx]
                else:
                    frames_plus[i][j] = occ @ sv[:, plus_mask]
                    frames_minus[i][j] = occ @ sv[:, minus_mask]

        c_plus, f_plus = subspace_chern(frames_plus)
        c_minus, f_minus = subspace_chern(frames_minus)
        spin_chern = 0.5 * (c_plus - c_minus)

        # optional spin Bott on small OBC.
        spin_bott = np.nan
        dim = args.obc_size * args.obc_size * 8
        if dim <= args.bott_max_dim:
            ham = build_obc_hamiltonian_sparse(
                v=p.v,
                t=p.t,
                lm=p.lm,
                w=p.w,
                j=0.0,
                nx=args.obc_size,
                ny=args.obc_size,
            ).toarray()
            evals, evecs = np.linalg.eigh(ham)
            occ_mask = evals < 0.0
            if np.sum(occ_mask) == 0:
                occ_mask = np.arange(evals.size) < (evals.size // 2)
            occ = evecs[:, occ_mask]
            s_occ = occ.conj().T @ spin_op @ occ
            se, sv = np.linalg.eigh(s_occ)
            plus_mask = se >= 0
            minus_mask = se < 0
            occ_plus = occ @ sv[:, plus_mask] if np.sum(plus_mask) > 0 else np.zeros((occ.shape[0], 0), dtype=complex)
            occ_minus = occ @ sv[:, minus_mask] if np.sum(minus_mask) > 0 else np.zeros((occ.shape[0], 0), dtype=complex)
            b_plus = spin_bott_from_occ(occ, occ_plus, nx=args.obc_size, ny=args.obc_size)
            b_minus = spin_bott_from_occ(occ, occ_minus, nx=args.obc_size, ny=args.obc_size)
            spin_bott = 0.5 * (b_plus - b_minus)

        berry = f_plus - f_minus
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        im = ax.imshow(berry, origin="lower", cmap="RdBu_r")
        ax.set_title(f"{label}: spin Berry curvature diff")
        ax.set_xlabel("k-index x")
        ax.set_ylabel("k-index y")
        fig.colorbar(im, ax=ax, label="F_plus - F_minus")
        fig.tight_layout()
        fig.savefig(out_root / f"{label}_spin_berry_curvature.png", dpi=180)
        plt.close(fig)

        rows.append(
            {
                "point_label": label,
                "v": p.v,
                "lm": p.lm,
                "t": p.t,
                "w": p.w,
                "chern_plus": c_plus,
                "chern_minus": c_minus,
                "spin_chern": spin_chern,
                "spin_bott": spin_bott,
                "comment": reason,
            }
        )
        run_notes.append(
            f"{label}: C+={c_plus:.3f}, C-={c_minus:.3f}, spin_chern={spin_chern:.3f}, spin_bott={spin_bott:.3f}"
        )

    write_csv(
        out_root / "spin_topology_summary.csv",
        rows,
        ["point_label", "v", "lm", "t", "w", "chern_plus", "chern_minus", "spin_chern", "spin_bott", "comment"],
    )
    write_text(out_root / "run_log.txt", "\n".join(run_notes) + "\n")
    print(f"[ok] spin topology outputs at {out_root}")


if __name__ == "__main__":
    main()
