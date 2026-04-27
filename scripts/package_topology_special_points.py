from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable
import zipfile


def format_param_token(value: float) -> str:
    """Format numeric parameter to point-folder token."""
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def point_name(v: float, t: float, lm: float) -> str:
    """Build folder name used by per-point outputs."""
    return f"v_{format_param_token(v)}_t_{format_param_token(t)}_lm_{format_param_token(lm)}"


def should_select(
    gap: float,
    chern: float,
    z2: int,
    edge_state: int,
    corner_state: int,
    robust_gap_threshold: float,
    transition_gap_threshold: float,
    chern_threshold: float,
) -> str:
    """Return category label for selected points, or empty string if not selected."""
    nontrivial = (z2 == 1) or (abs(chern) >= chern_threshold) or (corner_state == 1)
    if nontrivial and gap > robust_gap_threshold:
        return "robust_nontrivial"
    if abs(gap) <= transition_gap_threshold and (nontrivial or edge_state == 1):
        return "near_transition"
    return ""


def write_split_parts(zip_path: Path, part_prefix: Path, part_size_bytes: int) -> list[Path]:
    """Split zip into fixed-size parts and return part paths."""
    for old_part in sorted(part_prefix.parent.glob(f"{part_prefix.name}*")):
        if old_part.is_file():
            old_part.unlink()

    created: list[Path] = []
    with zip_path.open("rb") as source:
        index = 0
        while True:
            chunk = source.read(part_size_bytes)
            if not chunk:
                break
            part_path = part_prefix.parent / f"{part_prefix.name}{index:03d}"
            part_path.write_bytes(chunk)
            created.append(part_path)
            index += 1
    return created


def verify_rebuilt_zip(parts: Iterable[Path], rebuilt_path: Path) -> None:
    """Rebuild zip from parts and validate archive integrity."""
    with rebuilt_path.open("wb") as rebuilt:
        for part in parts:
            rebuilt.write(part.read_bytes())
    with zipfile.ZipFile(rebuilt_path) as zf:
        broken_entry = zf.testzip()
    if broken_entry is not None:
        raise RuntimeError(f"Rebuilt zip integrity check failed at entry: {broken_entry}")


def run(args: argparse.Namespace) -> None:
    root = Path("/workspace")
    results_csv = root / args.results_csv
    source_points_dir = root / args.source_points_dir
    output_dir = root / args.output_dir
    zip_base = root / args.zip_base

    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    if not source_points_dir.exists():
        raise FileNotFoundError(f"Point source folder not found: {source_points_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    points_out = output_dir / "points"
    points_out.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "special_points_summary.csv"
    criteria_path = output_dir / "selection_criteria.txt"
    missing_path = output_dir / "missing_points.txt"

    selected_rows: list[dict[str, float | int | str]] = []
    missing_points: list[str] = []

    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["v"])
            t = float(row["t"])
            lm = float(row["lm"])
            gap = float(row["gap"])
            chern = float(row["chern"])
            z2 = int(float(row["Z2_topology"]))
            edge_state = int(float(row["edge_state"]))
            corner_state = int(float(row["corner_state"]))
            wilson = float(row["Wilson_loop"])

            category = should_select(
                gap=gap,
                chern=chern,
                z2=z2,
                edge_state=edge_state,
                corner_state=corner_state,
                robust_gap_threshold=args.robust_gap_threshold,
                transition_gap_threshold=args.transition_gap_threshold,
                chern_threshold=args.chern_threshold,
            )
            if not category:
                continue

            point_id = point_name(v, t, lm)
            src = source_points_dir / point_id
            dst = points_out / point_id
            if not src.exists():
                missing_points.append(point_id)
                continue
            shutil.copytree(src, dst)

            selected_rows.append(
                {
                    "point_id": point_id,
                    "v": v,
                    "t": t,
                    "lm": lm,
                    "gap": gap,
                    "chern": chern,
                    "z2": z2,
                    "edge_state": edge_state,
                    "corner_state": corner_state,
                    "Wilson_loop": wilson,
                    "category": category,
                }
            )

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "point_id",
            "v",
            "t",
            "lm",
            "gap",
            "chern",
            "z2",
            "edge_state",
            "corner_state",
            "Wilson_loop",
            "category",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    criteria_path.write_text(
        "Topological special-point selection criteria\n"
        f"1) robust_nontrivial: gap>{args.robust_gap_threshold} and "
        f"(Z2=1 or |Chern|>={args.chern_threshold} or corner_state=1)\n"
        f"2) near_transition: |gap|<={args.transition_gap_threshold} and "
        f"((Z2=1 or |Chern|>={args.chern_threshold} or corner_state=1) or edge_state=1)\n"
        f"Artifacts are copied from {args.source_points_dir} for each selected point.\n",
        encoding="utf-8",
    )
    missing_path.write_text("\n".join(missing_points), encoding="utf-8")

    zip_base.parent.mkdir(parents=True, exist_ok=True)
    zip_path = Path(
        shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=str(output_dir.parent),
            base_dir=output_dir.name,
        )
    )

    created_parts: list[Path] = []
    if args.make_split_parts:
        part_size_bytes = max(1, int(args.split_size_mb * 1024 * 1024))
        part_prefix = zip_path.parent / f"{zip_path.name}.part."
        created_parts = write_split_parts(zip_path, part_prefix, part_size_bytes)
        if args.verify_split_parts and created_parts:
            verify_rebuilt_zip(created_parts, Path("/tmp") / f"{zip_path.stem}_rebuilt.zip")

    robust_count = sum(1 for row in selected_rows if row["category"] == "robust_nontrivial")
    transition_count = sum(1 for row in selected_rows if row["category"] == "near_transition")

    print(f"selected_total={len(selected_rows)}")
    print(f"robust_nontrivial={robust_count}")
    print(f"near_transition={transition_count}")
    print(f"missing_points={len(missing_points)}")
    print(f"output_dir={output_dir}")
    print(f"zip={zip_path}")
    if created_parts:
        print(f"split_parts={len(created_parts)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract topological special points and create package artifacts."
    )
    parser.add_argument("--results-csv", default="outputs/results.csv")
    parser.add_argument("--source-points-dir", default="outputs/all_points_full_package/points")
    parser.add_argument("--output-dir", default="outputs/topology_special_points_package")
    parser.add_argument("--zip-base", default="outputs/topology_special_points_package")
    parser.add_argument("--robust-gap-threshold", type=float, default=0.1)
    parser.add_argument("--transition-gap-threshold", type=float, default=0.1)
    parser.add_argument("--chern-threshold", type=float, default=0.5)
    parser.add_argument("--split-size-mb", type=float, default=45.0)
    parser.add_argument("--make-split-parts", action="store_true", default=True)
    parser.add_argument("--no-split-parts", action="store_false", dest="make_split_parts")
    parser.add_argument("--verify-split-parts", action="store_true", default=True)
    parser.add_argument("--no-verify-split-parts", action="store_false", dest="verify_split_parts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
