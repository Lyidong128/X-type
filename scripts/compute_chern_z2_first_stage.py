from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (  # noqa: E402
    compute_chern_number,
    compute_wilson_loop_and_z2,
    load_xtype_model,
    set_model_params,
)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    model = load_xtype_model(root / "models" / args.model_file)
    input_summary = root / args.input_summary
    output_summary = root / args.output_summary
    points_root = root / args.points_dir
    logs_path = root / args.logs_path

    if logs_path.exists():
        logs_path.unlink()

    with input_summary.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict[str, str | float | int]] = []
    total = len(rows)
    done = 0

    for row in rows:
        done += 1
        point_id = str(row["point_id"])
        status = str(row.get("status", "ok"))
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        w = float(row["w"])
        point_dir = points_root / point_id
        topo_json = point_dir / "topology_invariants.json"
        topo_txt = point_dir / "topology_invariants.txt"

        rec: dict[str, str | float | int] = {
            "point_id": point_id,
            "v": v,
            "t": t,
            "lm": lm,
            "w": w,
            "status": status,
            "chern": "",
            "z2": "",
            "wilson": "",
            "topology_status": "skipped",
        }

        if status != "ok":
            rec["topology_status"] = "skip_input_failed"
            out_rows.append(rec)
            continue

        if not point_dir.exists():
            rec["topology_status"] = "missing_point_dir"
            out_rows.append(rec)
            with logs_path.open("a", encoding="utf-8") as lf:
                lf.write(f"missing_point_dir point_id={point_id}\n")
            continue

        try:
            set_model_params(model, v=v, t=t, lm=lm, w=w, j=0.0)
            chern = float(compute_chern_number(model, nk=int(args.chern_nk), n_occ=int(args.n_occ)))
            wilson, z2 = compute_wilson_loop_and_z2(model, nk=int(args.wilson_nk), n_occ=int(args.n_occ))
            z2 = int(z2)
            wilson = float(wilson)

            topo_payload = {
                "point_id": point_id,
                "v": v,
                "t": t,
                "lm": lm,
                "w": w,
                "chern": chern,
                "z2": z2,
                "wilson": wilson,
                "n_occ": int(args.n_occ),
                "chern_nk": int(args.chern_nk),
                "wilson_nk": int(args.wilson_nk),
            }
            topo_json.write_text(json.dumps(topo_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            topo_txt.write_text(
                (
                    f"point_id={point_id}\n"
                    f"v={v:.6f}\n"
                    f"t={t:.6f}\n"
                    f"lm={lm:.6f}\n"
                    f"w={w:.6f}\n"
                    f"chern={chern:.12e}\n"
                    f"z2={z2:d}\n"
                    f"wilson={wilson:.12e}\n"
                    f"n_occ={int(args.n_occ)}\n"
                    f"chern_nk={int(args.chern_nk)}\n"
                    f"wilson_nk={int(args.wilson_nk)}\n"
                ),
                encoding="utf-8",
            )

            rec["chern"] = chern
            rec["z2"] = z2
            rec["wilson"] = wilson
            rec["topology_status"] = "ok"
        except Exception as exc:  # pragma: no cover
            rec["topology_status"] = "failed"
            with logs_path.open("a", encoding="utf-8") as lf:
                lf.write(
                    f"topology_failed point_id={point_id} v={v:.6f} t={t:.6f} lm={lm:.6f} w={w:.6f} error={exc!r}\n"
                )
        out_rows.append(rec)

        if done % 10 == 0 or done == total:
            print(f"processed {done}/{total}")

    fields = [
        "point_id",
        "v",
        "t",
        "lm",
        "w",
        "status",
        "chern",
        "z2",
        "wilson",
        "topology_status",
    ]
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_summary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"total={total}")
    print(f"summary={output_summary}")
    if logs_path.exists():
        print(f"logs={logs_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Chern and Z2 for first-stage points.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--input-summary", default="outputs/first_stage_band_ribbon/band_ribbon_summary.csv")
    parser.add_argument("--points-dir", default="outputs/first_stage_band_ribbon/points")
    parser.add_argument("--output-summary", default="outputs/first_stage_band_ribbon/topology_summary.csv")
    parser.add_argument("--logs-path", default="outputs/first_stage_band_ribbon/topology_logs.txt")
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--chern-nk", type=int, default=21)
    parser.add_argument("--wilson-nk", type=int, default=15)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
