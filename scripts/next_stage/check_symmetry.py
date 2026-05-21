#!/usr/bin/env python3
"""Check candidate symmetry constraints for selected analysis points."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.next_stage.common import (
    PROJECT_ROOT,
    auto_select_points,
    load_model,
    set_params,
    write_csv,
    write_json,
    write_text,
)


@dataclass
class SymCandidate:
    name: str
    operator: np.ndarray
    antiunitary: bool
    target: str  # "H(-k)", "-H(-k)", "-H(k)"
    source: str


def _lift_op(op: np.ndarray) -> np.ndarray | None:
    arr = np.array(op, dtype=complex)
    if arr.shape == (8, 8):
        return arr
    if arr.shape == (2, 2):
        return np.kron(np.eye(4, dtype=complex), arr)
    if arr.shape == (4, 4):
        return np.kron(arr, np.eye(2, dtype=complex))
    return None


def discover_candidates(model) -> dict[str, list[SymCandidate]]:
    """Discover explicit and fallback candidate symmetry operators."""
    out: dict[str, list[SymCandidate]] = {"TRS": [], "INV": [], "PHS": [], "CHIRAL": []}

    # Explicit operators from model symbols, if provided.
    explicit_map = [("T", "TRS", True, "H(-k)"), ("P", "INV", False, "H(-k)"), ("C", "PHS", True, "-H(-k)"), ("S", "CHIRAL", False, "-H(k)")]
    for symbol, key, anti, target in explicit_map:
        if hasattr(model, symbol):
            op = _lift_op(getattr(model, symbol))
            if op is not None:
                out[key].append(
                    SymCandidate(
                        name=f"model_{symbol}",
                        operator=op,
                        antiunitary=anti,
                        target=target,
                        source=f"from model.{symbol}",
                    )
                )

    # Common candidate set (marked as candidates, not guaranteed).
    s0 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    o0 = np.eye(4, dtype=complex)
    o_inv = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=complex,
    )
    o_chiral = np.diag([1, -1, 1, -1]).astype(complex)

    out["TRS"].append(
        SymCandidate(
            name="candidate_iSy",
            operator=np.kron(o0, 1j * sy),
            antiunitary=True,
            target="H(-k)",
            source="candidate: spinful TR U=i*sigma_y",
        )
    )
    out["INV"].extend(
        [
            SymCandidate(
                name="candidate_orbital_swap",
                operator=np.kron(o_inv, s0),
                antiunitary=False,
                target="H(-k)",
                source="candidate: orbital inversion swap (1<->4,2<->3)",
            ),
            SymCandidate(
                name="candidate_identity",
                operator=np.eye(8, dtype=complex),
                antiunitary=False,
                target="H(-k)",
                source="candidate: identity inversion baseline",
            ),
        ]
    )
    out["PHS"].append(
        SymCandidate(
            name="candidate_sigma_x",
            operator=np.kron(o0, sx),
            antiunitary=True,
            target="-H(-k)",
            source="candidate: particle-hole trial U=sigma_x",
        )
    )
    out["CHIRAL"].append(
        SymCandidate(
            name="candidate_orbital_chiral",
            operator=np.kron(o_chiral, s0),
            antiunitary=False,
            target="-H(k)",
            source="candidate: orbital chiral diag(+,-,+,-)",
        )
    )
    return out


def symmetry_error(model, cand: SymCandidate, kvec: np.ndarray, km: np.ndarray) -> float:
    hk = model.Hxtype(kvec)
    if cand.target == "H(-k)":
        target = model.Hxtype(km)
    elif cand.target == "-H(-k)":
        target = -model.Hxtype(km)
    elif cand.target == "-H(k)":
        target = -hk
    else:
        target = hk

    u = cand.operator
    inv_u = np.linalg.inv(u)
    transformed = u @ (hk.conj() if cand.antiunitary else hk) @ inv_u
    denom = max(1e-12, np.linalg.norm(hk))
    return float(np.linalg.norm(transformed - target) / denom)


def evaluate_point(model, candidates: dict[str, list[SymCandidate]], nk: int) -> dict:
    grid = np.linspace(0.0, 1.0, nk, endpoint=False)
    results: dict[str, list[dict[str, float | str | bool]]] = {}
    for sym_name, cands in candidates.items():
        sym_rows = []
        for cand in cands:
            errs = []
            for ux in grid:
                for uy in grid:
                    k = ux * model.b1 + uy * model.b2
                    km = (-ux) * model.b1 + (-uy) * model.b2
                    errs.append(symmetry_error(model, cand, k, km))
            errs_arr = np.array(errs, dtype=float)
            sym_rows.append(
                {
                    "candidate": cand.name,
                    "source": cand.source,
                    "antiunitary": cand.antiunitary,
                    "target": cand.target,
                    "mean_error": float(np.mean(errs_arr)),
                    "max_error": float(np.max(errs_arr)),
                    "median_error": float(np.median(errs_arr)),
                }
            )
        results[sym_name] = sorted(sym_rows, key=lambda r: float(r["max_error"]))
    return results


def status_from_error(max_error: float) -> str:
    if max_error < 1e-6:
        return "reliably_satisfied"
    if max_error < 1e-3:
        return "approximately_satisfied"
    return "not_satisfied"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check candidate symmetries for topology model.")
    parser.add_argument("--model-file", type=Path, default=Path("/workspace/models/xtype_model.py"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/outputs/next_stage_analysis/symmetry"))
    parser.add_argument("--nk", type=int, default=21)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    nk = 11 if args.quick else args.nk
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_file)
    points, notes = auto_select_points(model_path=args.model_file)
    points_map = {p.label: p for p in points}
    focus_labels = [x for x in ("A", "B") if x in points_map]
    if not focus_labels:
        raise RuntimeError("A/B points not available for symmetry check.")

    candidates = discover_candidates(model)
    summary_rows = []
    report_lines = [
        "Symmetry check report",
        f"model_file={args.model_file}",
        f"nk={nk}",
        "",
    ]
    if notes:
        report_lines.append("point selection notes:")
        report_lines.extend([f"- {n}" for n in notes])
        report_lines.append("")

    for label in focus_labels:
        p = points_map[label]
        set_params(model, v=p.v, lm=p.lm, t=p.t)
        result = evaluate_point(model=model, candidates=candidates, nk=nk)
        out_json = output_root / f"symmetry_check_{label}.json"
        write_json(
            out_json,
            {
                "point": p.as_dict(),
                "nk": nk,
                "results": result,
                "note": "Operators from model.* are treated as explicit; others are candidate symmetries.",
            },
        )

        report_lines.append(f"[{label}] v={p.v:.3f}, lm={p.lm:.3f}, t={p.t:.3f}, w={p.w:.3f}")
        for sym_name, rows in result.items():
            best = rows[0] if rows else None
            if best is None:
                report_lines.append(f"  - {sym_name}: no candidate")
                summary_rows.append(
                    {
                        "point_label": label,
                        "v": p.v,
                        "lm": p.lm,
                        "t": p.t,
                        "w": p.w,
                        "symmetry": sym_name,
                        "best_candidate": "",
                        "best_source": "",
                        "max_error": "",
                        "mean_error": "",
                        "status": "undetermined",
                        "comment": "No candidate matrix available.",
                    }
                )
                continue
            best_max = float(best["max_error"])
            status = status_from_error(best_max)
            report_lines.append(
                f"  - {sym_name}: best={best['candidate']} max_err={best_max:.3e} "
                f"mean_err={float(best['mean_error']):.3e} status={status}"
            )
            summary_rows.append(
                {
                    "point_label": label,
                    "v": p.v,
                    "lm": p.lm,
                    "t": p.t,
                    "w": p.w,
                    "symmetry": sym_name,
                    "best_candidate": best["candidate"],
                    "best_source": best["source"],
                    "max_error": best_max,
                    "mean_error": float(best["mean_error"]),
                    "status": status,
                    "comment": "explicit_operator" if str(best["candidate"]).startswith("model_") else "candidate_symmetry",
                }
            )
        report_lines.append("")

    report_lines.append("Conclusion:")
    report_lines.append("- reliable status only means low numerical residual under selected operator.")
    report_lines.append("- candidate_symmetry entries are heuristic and require physical confirmation.")
    report_lines.append("- If no explicit operator is found, protection symmetry remains undetermined from current code.")

    write_csv(
        output_root / "symmetry_summary.csv",
        summary_rows,
        [
            "point_label",
            "v",
            "lm",
            "t",
            "w",
            "symmetry",
            "best_candidate",
            "best_source",
            "max_error",
            "mean_error",
            "status",
            "comment",
        ],
    )
    write_text(output_root / "symmetry_report.txt", "\n".join(report_lines))
    print(f"[ok] symmetry summary: {output_root / 'symmetry_summary.csv'}")


if __name__ == "__main__":
    main()
