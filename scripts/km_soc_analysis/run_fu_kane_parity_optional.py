#!/usr/bin/env python3
"""Optional Fu-Kane parity analysis if inversion operator is identifiable."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
import sys

import numpy as np

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import ensure_dir, load_model, set_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional Fu-Kane parity calculation.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/km_soc_topology_analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--t", type=float, default=0.3)
    parser.add_argument("--w", type=float, default=1.0)
    parser.add_argument("--lm", type=float, default=0.1)
    return parser.parse_args()


def _candidate_inversion_operators() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(4)):
        p_orb = np.zeros((4, 4), dtype=float)
        for i, j in enumerate(perm):
            p_orb[i, j] = 1.0
        for signs in itertools.product([-1.0, 1.0], repeat=4):
            s = np.diag(signs)
            po = s @ p_orb
            if np.allclose(po.T @ po, np.eye(4), atol=1e-12):
                out.append(np.kron(po, np.eye(2)).astype(complex))
    return out


def _inversion_error(model, p_op: np.ndarray, v: float, lm: float, t: float, w: float, nk: int) -> float:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    errs = []
    for i in range(nk):
        u = i / max(1, nk - 1)
        for j in range(nk):
            vv = j / max(1, nk - 1)
            k = u * model.b1 + vv * model.b2
            h = np.array(model.Hxtype(k), dtype=complex)
            hm = np.array(model.Hxtype(-k), dtype=complex)
            denom = max(float(np.linalg.norm(h)), 1e-16)
            err = float(np.linalg.norm(p_op @ h @ p_op.conj().T - hm) / denom)
            errs.append(err)
    return float(np.mean(errs))


def _find_best_inversion(model, t: float, w: float, lm: float, quick: bool) -> tuple[np.ndarray | None, float]:
    nk = 5 if quick else 7
    best_p = None
    best_err = float("inf")
    for p in _candidate_inversion_operators():
        err = _inversion_error(model, p, v=0.5, lm=lm, t=t, w=w, nk=nk)
        if err < best_err:
            best_err = err
            best_p = p
    return best_p, best_err


def _fu_kane_z2(model, p_op: np.ndarray, v: float, lm: float, t: float, w: float, n_occ: int = 4) -> tuple[int, dict[str, int], float]:
    set_params(model, v=v, lm=lm, t=t, w=w, j=0.0)
    trims = {
        "Gamma": np.zeros(3),
        "X": 0.5 * model.b1,
        "Y": 0.5 * model.b2,
        "M": 0.5 * (model.b1 + model.b2),
    }
    delta = {}
    quality = []
    for name, k in trims.items():
        _, vecs = np.linalg.eigh(model.Hxtype(k))
        occ = vecs[:, :n_occ]
        m = occ.conj().T @ p_op @ occ
        m = 0.5 * (m + m.conj().T)
        evals = np.linalg.eigvalsh(m)
        q = float(np.min(np.abs(np.abs(evals) - 1.0)))
        quality.append(q)
        signs = np.sign(np.real(evals))
        signs[signs == 0] = 1
        delta[name] = int(np.prod(signs))
    prod = int(np.prod(list(delta.values())))
    z2 = 0 if prod > 0 else 1
    return z2, delta, float(np.max(quality))


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.output_root) / "fu_kane")
    model = load_model(args.model_file)
    scan_csv = Path(args.output_root) / "gap_z2_scan" / "gap_z2_scan.csv"
    if not scan_csv.exists():
        skip = out_dir / "skip_reason.txt"
        skip.write_text("gap_z2_scan.csv missing; run run_gap_z2_scan.py first.\n", encoding="utf-8")
        print(f"[skip] {skip}")
        return

    p_op, inv_err = _find_best_inversion(model, t=args.t, w=args.w, lm=args.lm, quick=args.quick)
    if p_op is None or inv_err > 5e-3:
        skip = out_dir / "skip_reason.txt"
        skip.write_text(
            f"Unable to identify reliable inversion operator. best_error={inv_err:.6e}\n",
            encoding="utf-8",
        )
        print(f"[skip] {skip}")
        return

    scan_rows = list(csv.DictReader(scan_csv.open("r", encoding="utf-8")))
    lm_rows = [r for r in scan_rows if abs(float(r["lm"]) - args.lm) < 1e-12]
    v_values = sorted({round(float(r["v"]), 10) for r in lm_rows})
    if not args.quick:
        # Keep a representative subset for parity diagnostics.
        keep = []
        for x in v_values:
            if abs((x * 100) % 10) < 1e-9 or x in {0.0, 1.1}:
                keep.append(x)
        if keep:
            v_values = sorted(set(keep))

    rows = []
    for v in v_values:
        z2_fk, deltas, quality = _fu_kane_z2(model, p_op=p_op, v=v, lm=args.lm, t=args.t, w=args.w, n_occ=4)
        closest = min(lm_rows, key=lambda r: abs(float(r["v"]) - v))
        z2_w = int(closest["z2"]) if int(closest["z2_reliable"]) == 1 else -1
        comment = "ok"
        if quality > 5e-2:
            comment = "parity eigenvalues deviate from ±1; caution."
        rows.append(
            {
                "v": float(v),
                "lm": float(args.lm),
                "t": float(args.t),
                "w": float(args.w),
                "delta_Gamma": int(deltas["Gamma"]),
                "delta_X": int(deltas["X"]),
                "delta_Y": int(deltas["Y"]),
                "delta_M": int(deltas["M"]),
                "z2_fu_kane": int(z2_fk),
                "z2_wilson_scan": int(z2_w),
                "inversion_error": float(inv_err),
                "parity_quality": float(quality),
                "comment": comment,
            }
        )

    csv_path = out_dir / "fu_kane_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "lm",
                "t",
                "w",
                "delta_Gamma",
                "delta_X",
                "delta_Y",
                "delta_M",
                "z2_fu_kane",
                "z2_wilson_scan",
                "inversion_error",
                "parity_quality",
                "comment",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[ok] csv={csv_path}")
    print(f"[ok] out_dir={out_dir}")


if __name__ == "__main__":
    main()
