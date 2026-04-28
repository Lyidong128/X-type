from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def v_group_name(v: float) -> str:
    return f"v_{v:.2f}".replace("-", "m").replace(".", "p")


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    base_dir = root / args.base_dir
    src_points = base_dir / "points"
    summary_csv = base_dir / "band_ribbon_summary.csv"
    grouped_dir = base_dir / "grouped_by_v"
    grouped_summary = base_dir / "grouped_by_v_summary.csv"

    if not src_points.exists():
        raise FileNotFoundError(f"Source points folder not found: {src_points}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    if grouped_dir.exists():
        shutil.rmtree(grouped_dir)
    grouped_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8", newline="")))
    out_rows: list[dict[str, str]] = []

    for row in rows:
        point_id = str(row["point_id"])
        v = float(row["v"])
        src_dir = src_points / point_id
        if not src_dir.exists():
            continue

        v_dir = grouped_dir / v_group_name(v)
        dst_dir = v_dir / point_id
        v_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir)

        out_rows.append(
            {
                "v_group": v_group_name(v),
                "point_id": point_id,
                "v": row["v"],
                "t": row["t"],
                "lm": row["lm"],
                "w": row["w"],
                "status": row["status"],
                "band_path": str((dst_dir / "band.png").relative_to(root)),
                "ribbon_path": str((dst_dir / "ribbon.png").relative_to(root)),
            }
        )

    with grouped_summary.open("w", encoding="utf-8", newline="") as f:
        fields = ["v_group", "point_id", "v", "t", "lm", "w", "status", "band_path", "ribbon_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"grouped_points={len(out_rows)}")
    print(f"grouped_dir={grouped_dir}")
    print(f"grouped_summary={grouped_summary}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group first-stage points by v value.")
    parser.add_argument("--base-dir", default="outputs/first_stage_band_ribbon")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
