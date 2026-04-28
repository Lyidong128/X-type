from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.run_scan import (  # noqa: E402
    build_scan_values,
    compute_band_data,
    compute_ribbon_spectrum,
    load_xtype_model,
    plot_band_structure,
    plot_ribbon_spectrum,
    set_model_params,
)


def dynamic_w(v: float) -> float:
    return float(2.0 - v)


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    model = load_xtype_model(root / "models" / args.model_file)

    out_dir = root / args.output_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "band_ribbon_summary.csv"
    logs_txt = out_dir / "logs.txt"
    if logs_txt.exists():
        logs_txt.unlink()

    v_values = build_scan_values(args.v_min, args.v_max, args.v_step)
    lm_values = build_scan_values(args.lm_min, args.lm_max, args.lm_step)
    t = float(args.fixed_t)

    rows: list[dict[str, str | float]] = []
    total = len(v_values) * len(lm_values)
    done = 0
    for v in v_values:
        for lm in lm_values:
            done += 1
            w = dynamic_w(v)
            band_path = fig_dir / f"band_v{v:.2f}_t{t:.2f}_lm{lm:.2f}.png"
            ribbon_path = fig_dir / f"ribbon_v{v:.2f}_t{t:.2f}_lm{lm:.2f}.png"
            status = "ok"
            try:
                set_model_params(model, v=v, t=t, lm=lm, w=w, j=0.0)
                band = compute_band_data(model)
                plot_band_structure(
                    eigvals=band,
                    save_path=band_path,
                    title=f"Band (v={v:.2f}, t={t:.2f}, lm={lm:.2f}, w={w:.2f})",
                )
                ky, eig = compute_ribbon_spectrum(
                    v=v,
                    t=t,
                    lm=lm,
                    w=w,
                    j=0.0,
                    nx=args.ribbon_nx,
                    nk=args.ribbon_nk,
                )
                plot_ribbon_spectrum(
                    ky_values=ky,
                    eigvals=eig,
                    save_path=ribbon_path,
                    title=f"Ribbon (v={v:.2f}, t={t:.2f}, lm={lm:.2f}, w={w:.2f})",
                )
            except Exception as exc:  # pragma: no cover
                status = "failed"
                with logs_txt.open("a", encoding="utf-8") as lf:
                    lf.write(f"v={v:.3f} t={t:.3f} lm={lm:.3f} w={w:.3f} error={exc!r}\n")
            rows.append(
                {
                    "v": v,
                    "t": t,
                    "lm": lm,
                    "w": w,
                    "band_path": str(band_path.relative_to(root)),
                    "ribbon_path": str(ribbon_path.relative_to(root)),
                    "status": status,
                }
            )
            if done % 10 == 0 or done == total:
                print(f"processed {done}/{total}")

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["v", "t", "lm", "w", "band_path", "ribbon_path", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"total={total}")
    print(f"summary={summary_csv}")
    print(f"figures={fig_dir}")
    if logs_txt.exists():
        print(f"logs={logs_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-stage scan: compute only band and ribbon figures.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-dir", default="outputs/first_stage_band_ribbon")
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=2.0)
    parser.add_argument("--v-step", type=float, default=0.1)
    parser.add_argument("--lm-min", type=float, default=0.0)
    parser.add_argument("--lm-max", type=float, default=0.5)
    parser.add_argument("--lm-step", type=float, default=0.1)
    parser.add_argument("--fixed-t", type=float, default=0.5)
    parser.add_argument("--ribbon-nx", type=int, default=20)
    parser.add_argument("--ribbon-nk", type=int, default=61)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
