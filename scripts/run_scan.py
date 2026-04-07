import csv
import importlib.util
import os
from itertools import product
from pathlib import Path

import numpy as np


def build_scan_values(start: float, stop: float, step: float) -> list[float]:
    """Build a stable float range, inclusive of stop."""
    values = []
    current = start
    epsilon = 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def load_xtype_model(model_path: Path):
    """Load model module from file path."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("xtype_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_bulk_gap(model_module, nk: int = 11, n_occ: int = 4) -> float:
    """
    Compute indirect bulk gap at half filling:
    gap = min_k E_c(k) - max_k E_v(k), where E_v=band[n_occ-1], E_c=band[n_occ].
    """
    valence_max = -np.inf
    conduction_min = np.inf

    for i, j in product(range(nk), range(nk)):
        u = i / (nk - 1)
        vfrac = j / (nk - 1)
        k = u * model_module.b1 + vfrac * model_module.b2
        evals = np.linalg.eigvalsh(model_module.Hxtype(k))
        valence_max = max(valence_max, float(np.real(evals[n_occ - 1])))
        conduction_min = min(conduction_min, float(np.real(evals[n_occ])))

    return round(conduction_min - valence_max, 10)


def run_parameter_scan() -> dict[str, int | str]:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    logs_path = output_dir / "logs.txt"
    model_path = project_root / "models" / "xtype_model.py"

    # Avoid GUI backend issues when loading model code with plotting imports.
    os.environ.setdefault("MPLBACKEND", "Agg")
    model = load_xtype_model(model_path)

    values = build_scan_values(0.0, 1.0, 0.5)
    w_fixed = 1.0
    j_fixed = 0.0
    total_count = 0
    success_count = 0
    failure_count = 0

    if logs_path.exists():
        logs_path.unlink()

    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=["v", "t", "w", "lm", "J", "gap"])
        writer.writeheader()

        for v, t, lm in product(values, values, values):
            total_count += 1
            try:
                model.v = float(v)
                model.t = float(t)
                model.w = float(w_fixed)
                model.lm = float(lm)
                model.J = float(j_fixed)
                gap = compute_bulk_gap(model)
                writer.writerow(
                    {
                        "v": v,
                        "t": t,
                        "w": w_fixed,
                        "lm": lm,
                        "J": j_fixed,
                        "gap": gap,
                    }
                )
                success_count += 1
            except Exception as error:  # pragma: no cover - defensive logging
                failure_count += 1
                with logs_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"v={v}, t={t}, w={w_fixed}, lm={lm}, J={j_fixed}, error={error!r}\n"
                    )

    if failure_count == 0 and logs_path.exists():
        logs_path.unlink()

    return {
        "total": total_count,
        "success": success_count,
        "failure": failure_count,
        "results_path": str(results_path),
        "logs_path": str(logs_path),
    }


if __name__ == "__main__":
    summary = run_parameter_scan()
    print("summary")
    print(f"total={summary['total']}")
    print(f"success={summary['success']}")
    print(f"failure={summary['failure']}")
    print(f"results={summary['results_path']}")
    if summary["failure"]:
        print(f"logs={summary['logs_path']}")
