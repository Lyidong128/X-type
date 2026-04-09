import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


RESULT_FIELDS = [
    "v",
    "t",
    "lm",
    "gap",
    "chern",
    "edge_state",
    "corner_state",
    "Wilson_loop",
    "Z2_topology",
]


def classify_phase(row: dict[str, float]) -> str:
    chern = float(row["chern"])
    z2 = int(row["Z2_topology"])
    wilson = abs(float(row["Wilson_loop"]))
    edge_state = int(row["edge_state"])
    corner_state = int(row["corner_state"])
    if corner_state == 1 or (z2 == 1 and edge_state == 1 and wilson >= 0.25):
        return "HOTI"
    if abs(chern) >= 0.5 or z2 == 1:
        return "TI"
    return "trivial"


def load_rows(results_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with results_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [field for field in RESULT_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing fields in results.csv: {missing}")
        for row in reader:
            rows.append(
                {
                    "v": float(row["v"]),
                    "t": float(row["t"]),
                    "lm": float(row["lm"]),
                    "gap": float(row["gap"]),
                    "chern": float(row["chern"]),
                    "edge_state": int(float(row["edge_state"])),
                    "corner_state": int(float(row["corner_state"])),
                    "Wilson_loop": float(row["Wilson_loop"]),
                    "Z2_topology": int(float(row["Z2_topology"])),
                }
            )
    return rows


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    content = [title, ""] + lines
    ax.text(0.03, 0.98, "\n".join(content), va="top", family="monospace", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, title: str | None = None) -> None:
    if not image_path.exists():
        return
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title or image_path.name)
    pdf.savefig(fig)
    plt.close(fig)


def summarize_parameter_dependence(rows: list[dict[str, float]], key: str) -> list[str]:
    grouped: dict[float, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[float(row[key])][classify_phase(row)] += 1
    lines = []
    for value in sorted(grouped):
        counter = grouped[value]
        lines.append(
            f"{key}={value:.1f}: trivial={counter['trivial']}, TI={counter['TI']}, HOTI={counter['HOTI']}"
        )
    return lines


def make_statistics_page(pdf: PdfPages, rows: list[dict[str, float]]) -> None:
    phase_counts = Counter(classify_phase(row) for row in rows)
    gaps = np.array([float(row["gap"]) for row in rows], dtype=float)
    cherns = np.array([float(row["chern"]) for row in rows], dtype=float)
    wilsons = np.array([float(row["Wilson_loop"]) for row in rows], dtype=float)
    z2 = np.array([int(row["Z2_topology"]) for row in rows], dtype=int)
    edge = np.array([int(row["edge_state"]) for row in rows], dtype=int)
    corner = np.array([int(row["corner_state"]) for row in rows], dtype=int)

    lines = [
        f"Total scan points: {len(rows)}",
        "",
        f"Phase counts: trivial={phase_counts['trivial']}, TI={phase_counts['TI']}, HOTI={phase_counts['HOTI']}",
        f"gap range: [{gaps.min():.3e}, {gaps.max():.3e}]",
        f"|chern| max: {np.max(np.abs(cherns)):.3e}",
        f"|Wilson_loop| range: [{np.min(np.abs(wilsons)):.3e}, {np.max(np.abs(wilsons)):.3e}]",
        f"Z2=1 count: {int(np.sum(z2 == 1))}",
        f"edge_state=1 count: {int(np.sum(edge == 1))}",
        f"corner_state=1 count: {int(np.sum(corner == 1))}",
        "",
        "Parameter dependence (phase-count by each axis):",
        *summarize_parameter_dependence(rows, "v"),
        *summarize_parameter_dependence(rows, "t"),
        *summarize_parameter_dependence(rows, "lm"),
    ]
    add_text_page(pdf, "Detailed Topology Scan Report - Statistics", lines)


def make_top_candidate_page(pdf: PdfPages, rows: list[dict[str, float]]) -> None:
    by_small_gap = sorted(rows, key=lambda row: abs(float(row["gap"])))[:10]
    lines = [
        "Top 10 smallest-|gap| points (potential transition candidates):",
        "",
    ]
    for i, row in enumerate(by_small_gap, start=1):
        lines.append(
            f"{i:02d}. v={row['v']:.1f}, t={row['t']:.1f}, lm={row['lm']:.1f}, "
            f"gap={row['gap']:.3e}, chern={row['chern']:.3e}, W={row['Wilson_loop']:.3f}, "
            f"Z2={row['Z2_topology']}, edge={row['edge_state']}, corner={row['corner_state']}, "
            f"phase={classify_phase(row)}"
        )
    add_text_page(pdf, "Detailed Topology Scan Report - Candidate Points", lines)


def generate_detailed_report(project_root: Path) -> Path:
    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures"
    results_path = outputs_dir / "results.csv"
    report_path = outputs_dir / "detailed_report.pdf"

    rows = load_rows(results_path)
    with PdfPages(report_path) as pdf:
        make_statistics_page(pdf, rows)
        make_top_candidate_page(pdf, rows)

        for image_name in [
            "topology_phase_map.png",
            "phase_map_v_t_by_lm.png",
            "phase_map_v_lm_by_t.png",
            "phase_map_t_lm_by_v.png",
            "chern_vs_parameters.png",
            "edge_state_vs_parameters.png",
            "corner_state_vs_parameters.png",
        ]:
            add_image_page(pdf, figures_dir / image_name)

        for image_path in sorted(figures_dir.glob("obc_spectrum_*.png")):
            add_image_page(pdf, image_path)
        for image_path in sorted(figures_dir.glob("obc_wavefunc_*.png")):
            add_image_page(pdf, image_path)

    return report_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = generate_detailed_report(root)
    print(f"generated={out}")
