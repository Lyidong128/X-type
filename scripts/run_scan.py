import csv
import importlib.util
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np


def build_scan_values(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    epsilon = 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def load_xtype_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    spec = importlib.util.spec_from_file_location("xtype_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_message(logs_path: Path, message: str) -> None:
    with logs_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def load_scan_points_from_results(results_path: Path) -> list[tuple[float, float, float]]:
    if not results_path.exists():
        return []
    try:
        with results_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not {"v", "t", "lm"}.issubset(set(fieldnames)):
                return []
            points = []
            for row in reader:
                points.append((float(row["v"]), float(row["t"]), float(row["lm"])))
            return points
    except Exception:
        return []


def compute_bulk_gap(model_module, nk: int = 11, n_occ: int = 4) -> float:
    valence_max = -np.inf
    conduction_min = np.inf
    for i, j in product(range(nk), range(nk)):
        u = i / (nk - 1)
        vfrac = j / (nk - 1)
        k = u * model_module.b1 + vfrac * model_module.b2
        evals = np.linalg.eigvalsh(model_module.Hxtype(k))
        valence_max = max(valence_max, float(np.real(evals[n_occ - 1])))
        conduction_min = min(conduction_min, float(np.real(evals[n_occ])))
    return float(conduction_min - valence_max)


def compute_chern_number(model_module, nk: int = 21, n_occ: int = 4) -> float:
    return float(model_module.chern_number_fukui(nk=nk, n_occ=n_occ))


def set_model_params(model, v: float, t: float, lm: float, w: float = 1.0, j: float = 0.0) -> None:
    model.v = float(v)
    model.t = float(t)
    model.lm = float(lm)
    model.w = float(w)
    model.J = float(j)


def compute_band_data(model_module) -> np.ndarray:
    segments = [model_module.gx, model_module.xy, model_module.yg, model_module.gm, model_module.mg]
    eigvals = [np.array([np.linalg.eigvalsh(model_module.Hxtype(k)) for k in seg]) for seg in segments]
    return np.vstack(eigvals)


def plot_band_structure(eigvals: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(eigvals.shape[0])
    for band in range(eigvals.shape[1]):
        ax.plot(x, eigvals[:, band], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("k-path index")
    ax.set_ylabel("Energy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_chern_relationship(rows: list[dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.array([row["v"] for row in rows], dtype=float)
    t = np.array([row["t"] for row in rows], dtype=float)
    lm = np.array([row["lm"] for row in rows], dtype=float)
    chern = np.array([row["chern"] for row in rows], dtype=float)
    gap = np.array([row["gap"] for row in rows], dtype=float)
    gap_min = float(np.min(gap))
    gap_max = float(np.max(gap))
    if gap_min < 0.0 < gap_max:
        # Use symmetric diverging normalization for signed gap values.
        bound = max(abs(gap_min), abs(gap_max))
        norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        cmap = "coolwarm"
    else:
        norm = mcolors.Normalize(vmin=gap_min, vmax=gap_max)
        cmap = "viridis"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    scatter_config = [("v", v), ("t", t), ("lm", lm)]
    for ax, (label, values) in zip(axes, scatter_config):
        ax.scatter(
            values,
            chern,
            c=gap,
            cmap=cmap,
            norm=norm,
            s=55,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Chern number")
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=axes.ravel().tolist(), label="Gap")
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.14, top=0.90, wspace=0.28)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def validate_rows(rows: list[dict[str, float]]) -> list[str]:
    errors = []
    required_fields = ["v", "t", "lm", "gap", "chern"]
    for idx, row in enumerate(rows, start=1):
        for field in required_fields:
            if field not in row:
                errors.append(f"row={idx} missing_field={field}")
                continue
            value = row[field]
            if value is None:
                errors.append(f"row={idx} field={field} is_none")
                continue
            if not np.isfinite(float(value)):
                errors.append(f"row={idx} field={field} is_non_finite value={value}")
    return errors


def recalculate_invalid_rows(
    model,
    rows: list[dict[str, float]],
    invalid_indices: list[int],
    logs_path: Path,
) -> None:
    for idx in invalid_indices:
        row = rows[idx]
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        try:
            set_model_params(model, v=v, t=t, lm=lm, w=1.0, j=0.0)
            row["gap"] = compute_bulk_gap(model, nk=11, n_occ=4)
            row["chern"] = compute_chern_number(model, nk=21, n_occ=4)
            log_message(logs_path, f"repair_success row={idx + 1} v={v} t={t} lm={lm}")
        except Exception as error:  # pragma: no cover
            log_message(
                logs_path,
                f"repair_failed row={idx + 1} v={v} t={t} lm={lm} error={error!r}",
            )


def write_results_csv(results_path: Path, rows: list[dict[str, float]]) -> None:
    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=["v", "t", "lm", "gap", "chern"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "v": row["v"],
                    "t": row["t"],
                    "lm": row["lm"],
                    "gap": row["gap"],
                    "chern": row["chern"],
                }
            )


def run_parameter_scan() -> dict[str, int | str]:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    logs_path = output_dir / "logs.txt"
    model_path = project_root / "models" / "xtype_model.py"

    os.environ.setdefault("MPLBACKEND", "Agg")
    model = load_xtype_model(model_path)

    if logs_path.exists():
        logs_path.unlink()

    scan_points = load_scan_points_from_results(results_path)
    if not scan_points:
        values = build_scan_values(0.0, 1.0, 0.5)
        scan_points = list(product(values, values, values))

    rows: list[dict[str, float]] = []
    total_count = len(scan_points)
    success_count = 0
    failure_count = 0

    for idx, (v, t, lm) in enumerate(scan_points, start=1):
        try:
            set_model_params(model, v=float(v), t=float(t), lm=float(lm), w=1.0, j=0.0)
            gap = compute_bulk_gap(model, nk=11, n_occ=4)
            chern = compute_chern_number(model, nk=21, n_occ=4)
            rows.append({"v": float(v), "t": float(t), "lm": float(lm), "gap": gap, "chern": chern})

            band_data = compute_band_data(model)
            band_path = figures_dir / (
                f"band_v{float(v):.1f}_t{float(t):.1f}_lm{float(lm):.1f}.png"
            )
            plot_band_structure(
                band_data,
                save_path=band_path,
                title=f"Band structure (v={v:.1f}, t={t:.1f}, lm={lm:.1f})",
            )
            success_count += 1
        except Exception as error:  # pragma: no cover
            failure_count += 1
            log_message(
                logs_path,
                f"compute_failed idx={idx} v={v} t={t} lm={lm} error={error!r}",
            )

    write_results_csv(results_path, rows)

    validation_errors = validate_rows(rows)
    if validation_errors:
        for error in validation_errors:
            log_message(logs_path, f"validation_error {error}")
        invalid_indices = sorted(
            {int(error.split()[0].split("=")[1]) - 1 for error in validation_errors if "row=" in error}
        )
        recalculate_invalid_rows(model, rows, invalid_indices, logs_path)
        write_results_csv(results_path, rows)
        validation_errors = validate_rows(rows)
        if validation_errors:
            for error in validation_errors:
                log_message(logs_path, f"post_repair_validation_error {error}")

    if rows:
        plot_chern_relationship(rows, figures_dir / "chern_vs_parameters.png")

    if failure_count == 0 and not validation_errors and logs_path.exists():
        logs_path.unlink()

    return {
        "total": total_count,
        "success": success_count,
        "failure": failure_count + len(validation_errors),
        "results_path": str(results_path),
        "logs_path": str(logs_path),
        "figures_path": str(figures_dir),
    }


if __name__ == "__main__":
    summary = run_parameter_scan()
    print("summary")
    print(f"total={summary['total']}")
    print(f"success={summary['success']}")
    print(f"failure={summary['failure']}")
    print(f"results={summary['results_path']}")
    print(f"figures={summary['figures_path']}")
    if summary["failure"]:
        print(f"logs={summary['logs_path']}")
