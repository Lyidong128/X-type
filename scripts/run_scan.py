import csv
import random
from itertools import product
from pathlib import Path


def build_scan_values(start: float, stop: float, step: float) -> list[float]:
    """Build a stable float range, inclusive of stop."""
    values = []
    current = start
    epsilon = 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def run_parameter_scan() -> dict[str, int | str]:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    logs_path = output_dir / "logs.txt"

    values = build_scan_values(0.0, 1.0, 0.5)
    total_count = 0
    success_count = 0
    failure_count = 0

    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=["v", "t", "lm", "result"])
        writer.writeheader()

        for v, t, lm in product(values, values, values):
            total_count += 1
            try:
                simulated_result = round(random.random(), 6)
                writer.writerow({"v": v, "t": t, "lm": lm, "result": simulated_result})
                success_count += 1
            except Exception as error:  # pragma: no cover - defensive logging
                failure_count += 1
                with logs_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"v={v}, t={t}, lm={lm}, error={error!r}\n"
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
