import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors

plt.rcParams["font.family"] = "WenQuanYi Micro Hei"
plt.rcParams["axes.unicode_minus"] = False


def classify_phase(row: dict[str, float]) -> str:
    chern = float(row["chern"])
    z2 = int(float(row["Z2_topology"]))
    wilson = abs(float(row["Wilson_loop"]))
    edge_state = int(float(row["edge_state"]))
    corner_state = int(float(row["corner_state"]))
    if corner_state == 1 or (z2 == 1 and edge_state == 1 and wilson >= 0.25):
        return "HOTI"
    if abs(chern) >= 0.5 or z2 == 1:
        return "TI"
    return "trivial"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def add_text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.03, 0.98, title + "\n\n" + "\n".join(lines), va="top", family="monospace", fontsize=fontsize)
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


def make_transition_density_plot(all_rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v_vals = sorted({to_float(r, "v") for r in all_rows})
    t_vals = sorted({to_float(r, "t") for r in all_rows})
    lm_vals = sorted({to_float(r, "lm") for r in all_rows})
    phase_by_key = {
        (to_float(r, "v"), to_float(r, "t"), to_float(r, "lm")): classify_phase(r)
        for r in all_rows
    }

    transition_count = defaultdict(int)
    for v in v_vals:
        for t in t_vals:
            for lm in lm_vals:
                key = (v, t, lm)
                base = phase_by_key[key]
                for axis, vals in (("v", v_vals), ("t", t_vals), ("lm", lm_vals)):
                    idx = vals.index(key[0] if axis == "v" else key[1] if axis == "t" else key[2])
                    for step in (-1, 1):
                        nidx = idx + step
                        if not (0 <= nidx < len(vals)):
                            continue
                        if axis == "v":
                            nkey = (vals[nidx], t, lm)
                        elif axis == "t":
                            nkey = (v, vals[nidx], lm)
                        else:
                            nkey = (v, t, vals[nidx])
                        if phase_by_key[nkey] != base:
                            transition_count[key] += 1

    xs = np.array([k[0] for k in transition_count], dtype=float)
    ys = np.array([k[1] for k in transition_count], dtype=float)
    zs = np.array([k[2] for k in transition_count], dtype=float)
    cs = np.array([transition_count[k] for k in transition_count], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, x, y, label in (
        (axes[0], xs, cs, "v"),
        (axes[1], ys, cs, "t"),
        (axes[2], zs, cs, "lm"),
    ):
        sc = ax.scatter(x, y, c=y, cmap="viridis", s=40, edgecolors="black", linewidths=0.2)
        ax.set_xlabel(label)
        ax.set_ylabel("邻域相变计数")
        ax.grid(alpha=0.2)
    fig.suptitle("相变敏感度随参数变化（邻域相标签变化计数）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def generate_special_points_report(project_root: Path) -> Path:
    outputs = project_root / "outputs"
    special_dir = outputs / "special_points"
    figures_dir = outputs / "figures"
    points_dir = special_dir / "points"
    out_pdf = outputs / "special_points_report.pdf"
    transition_fig = figures_dir / "phase_transition_sensitivity.png"

    all_rows = read_csv_rows(outputs / "results.csv")
    special_rows = read_csv_rows(special_dir / "special_points.csv")
    criteria_lines = (special_dir / "selection_criteria.txt").read_text(encoding="utf-8").splitlines()

    phase_counts = Counter(classify_phase(r) for r in all_rows)
    special_phase_counts = Counter(classify_phase(r) for r in special_rows)
    small_gap_count = sum(int(float(r["is_small_gap"])) for r in special_rows)
    chern_jump_count = sum(int(float(r["is_chern_jump"])) for r in special_rows)
    corner_count = sum(int(float(r["corner_state"])) for r in special_rows)

    make_transition_density_plot(all_rows, transition_fig)

    with PdfPages(out_pdf) as pdf:
        add_text_page(
            pdf,
            "特殊点拓扑分析报告（1331点全扫描）",
            [
                f"总参数点数: {len(all_rows)}",
                f"特殊点数: {len(special_rows)}",
                "",
                "总体拓扑相计数:",
                f"  trivial={phase_counts['trivial']}, TI={phase_counts['TI']}, HOTI={phase_counts['HOTI']}",
                "特殊点拓扑相计数:",
                f"  trivial={special_phase_counts['trivial']}, TI={special_phase_counts['TI']}, HOTI={special_phase_counts['HOTI']}",
                "",
                "特殊点触发统计:",
                f"  small_gap={small_gap_count}",
                f"  chern_jump={chern_jump_count}",
                f"  corner_candidate={corner_count}",
            ],
        )

        add_text_page(pdf, "特殊点判据说明（中文）", criteria_lines, fontsize=11)

        # Top candidates by smallest gap
        top_gap = sorted(special_rows, key=lambda r: abs(float(r["gap"])))[:30]
        lines = ["最小|gap|的前30个特殊点（相变边界候选）:", ""]
        for i, r in enumerate(top_gap, start=1):
            phase = classify_phase(r)
            lines.append(
                f"{i:02d}. v={float(r['v']):.1f}, t={float(r['t']):.1f}, lm={float(r['lm']):.1f}, "
                f"gap={float(r['gap']):.3e}, chern={float(r['chern']):.3e}, "
                f"W={float(r['Wilson_loop']):.3f}, Z2={int(float(r['Z2_topology']))}, "
                f"edge={int(float(r['edge_state']))}, corner={int(float(r['corner_state']))}, "
                f"phase={phase}, reason={r['selection_reason']}"
            )
        add_text_page(pdf, "关键特殊点列表", lines, fontsize=8)

        for name in [
            "topology_phase_map.png",
            "phase_map_v_t_by_lm.png",
            "phase_map_v_lm_by_t.png",
            "phase_map_t_lm_by_v.png",
            "phase_transition_sensitivity.png",
            "chern_vs_parameters.png",
            "wilson_vs_parameters.png",
            "edge_state_vs_parameters.png",
            "corner_state_vs_parameters.png",
        ]:
            add_image_page(pdf, figures_dir / name, title=name)

        # Sample per-point deep diagnostics
        point_dirs = sorted(points_dir.glob("v_*"))[:18]
        for pdir in point_dirs:
            note_path = pdir / "说明.txt"
            if note_path.exists():
                text = note_path.read_text(encoding="utf-8").splitlines()
                add_text_page(pdf, f"点级说明: {pdir.name}", text, fontsize=10)
            for img_name in ["band.png", "ribbon.png", "obc_spectrum.png", "obc_wavefunction.png"]:
                add_image_page(pdf, pdir / img_name, title=f"{pdir.name}/{img_name}")

    return out_pdf


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = generate_special_points_report(root)
    print(f"generated={out}")
