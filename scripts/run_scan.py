import csv
import importlib.util
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


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

SPECIAL_POINT_FIELDS = [
    "v",
    "t",
    "lm",
    "gap",
    "chern",
    "Wilson_loop",
    "Z2_topology",
    "edge_state",
    "corner_state",
    "is_small_gap",
    "is_chern_jump",
    "chern_jump_axes",
    "max_chern_delta",
    "selection_reason",
]

SCAN_MIN = 0.0
SCAN_MAX = 1.0
DEFAULT_SCAN_STEP = 0.25
DEFAULT_OBC_NX = 20
DEFAULT_OBC_NY = 20
DEFAULT_OBC_MODE_COUNT = 48
DEFAULT_RIBBON_NX = 20
DEFAULT_RIBBON_NK = 61


def build_scan_values(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    epsilon = 1e-9
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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


def normalize_key(v: float, t: float, lm: float) -> tuple[float, float, float]:
    return (round(float(v), 10), round(float(t), 10), round(float(lm), 10))


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


def load_cached_rows(results_path: Path, logs_path: Path) -> dict[tuple[float, float, float], dict[str, float]]:
    cache: dict[tuple[float, float, float], dict[str, float]] = {}
    if not results_path.exists():
        return cache
    try:
        with results_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            if not set(RESULT_FIELDS).issubset(fieldnames):
                return cache
            for row in reader:
                try:
                    item = {
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
                    if not all(np.isfinite(float(item[k])) for k in ("v", "t", "lm", "gap", "chern", "Wilson_loop")):
                        continue
                    if item["edge_state"] not in (0, 1) or item["corner_state"] not in (0, 1) or item["Z2_topology"] not in (0, 1):
                        continue
                    cache[normalize_key(item["v"], item["t"], item["lm"])] = item
                except Exception:
                    continue
    except Exception as error:
        log_message(logs_path, f"cache_read_failed error={error!r}")
    return cache


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


def compute_wilson_loop_and_z2(model_module, nk: int = 15, n_occ: int = 4) -> tuple[float, int]:
    def occ_space(kvec: np.ndarray) -> np.ndarray:
        _, evecs = np.linalg.eigh(model_module.Hxtype(kvec))
        return evecs[:, :n_occ]

    phases = []
    for j in range(nk):
        ky = j / nk
        wilson = 1.0 + 0.0j
        prev = occ_space(0.0 * model_module.b1 + ky * model_module.b2)
        for i in range(1, nk + 1):
            kx = i / nk
            curr = occ_space(kx * model_module.b1 + ky * model_module.b2)
            overlap = prev.conj().T @ curr
            det_overlap = np.linalg.det(overlap)
            if abs(det_overlap) > 1e-14:
                wilson *= det_overlap / abs(det_overlap)
            prev = curr
        phases.append(float(np.angle(wilson)))

    phases = np.array(phases, dtype=float)
    unwrapped = np.unwrap(phases)
    winding = float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))
    z2 = int(round(abs(winding))) % 2
    return winding, z2


def set_model_params(model, v: float, t: float, lm: float, w: float = 1.0, j: float = 0.0) -> None:
    model.v = float(v)
    model.t = float(t)
    model.lm = float(lm)
    model.w = float(w)
    model.J = float(j)


def format_param_token(value: float) -> str:
    return f"{value:.2f}".replace("-", "m").replace(".", "p")


def _state_index(x: int, y: int, orb: int, spin: int, nx: int) -> int:
    return ((y * nx + x) * 8) + (orb * 2 + spin)


def build_onsite_cell_matrix(t: float, v: float, lm: float, j: float) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v
    h0[1, 0] = t
    h0[1, 2] = v
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v
    h0[2, 3] = t
    h0[3, 0] = v
    h0[3, 1] = t
    h0[3, 2] = t

    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm
    h1[0, 2] = -1j * lm
    h1[2, 3] = -1j * lm
    h1[3, 2] = 1j * lm
    h1[3, 1] = -1j * lm
    h1[1, 0] = -1j * lm
    h1[2, 0] = 1j * lm
    h1[1, 3] = 1j * lm

    h3 = np.diag([j, -j, j, -j]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h1, sz) + np.kron(h3, sz)


def build_obc_hamiltonian(
    v: float,
    t: float,
    lm: float,
    w: float = 1.0,
    j: float = 0.0,
    nx: int = 6,
    ny: int = 6,
) -> np.ndarray:
    dim = nx * ny * 8
    ham = np.zeros((dim, dim), dtype=complex)
    onsite = build_onsite_cell_matrix(t=t, v=v, lm=lm, j=j)

    for y in range(ny):
        for x in range(nx):
            start = (y * nx + x) * 8
            ham[start : start + 8, start : start + 8] += onsite

            if x > 0:
                for spin in range(2):
                    i = _state_index(x, y, 0, spin, nx)
                    jx = _state_index(x - 1, y, 3, spin, nx)
                    ham[i, jx] += w
                    ham[jx, i] += w

            if y > 0:
                for spin in range(2):
                    i = _state_index(x, y, 1, spin, nx)
                    jy = _state_index(x, y - 1, 2, spin, nx)
                    ham[i, jy] += w
                    ham[jy, i] += w
    return ham


def build_obc_hamiltonian_sparse(
    v: float,
    t: float,
    lm: float,
    w: float = 1.0,
    j: float = 0.0,
    nx: int = 20,
    ny: int = 20,
) -> sparse.csr_matrix:
    dim = nx * ny * 8
    ham = sparse.lil_matrix((dim, dim), dtype=complex)
    onsite = build_onsite_cell_matrix(t=t, v=v, lm=lm, j=j)

    for y in range(ny):
        for x in range(nx):
            start = (y * nx + x) * 8
            for i in range(8):
                for j_idx in range(8):
                    value = onsite[i, j_idx]
                    if value != 0:
                        ham[start + i, start + j_idx] += value

            if x > 0:
                for spin in range(2):
                    i_idx = _state_index(x, y, 0, spin, nx)
                    jx = _state_index(x - 1, y, 3, spin, nx)
                    ham[i_idx, jx] += w
                    ham[jx, i_idx] += w

            if y > 0:
                for spin in range(2):
                    i_idx = _state_index(x, y, 1, spin, nx)
                    jy = _state_index(x, y - 1, 2, spin, nx)
                    ham[i_idx, jy] += w
                    ham[jy, i_idx] += w
    return ham.tocsr()


def compute_obc_spectrum_and_probability(
    v: float,
    t: float,
    lm: float,
    w: float = 1.0,
    j: float = 0.0,
    nx: int = 6,
    ny: int = 6,
    mode_count: int = DEFAULT_OBC_MODE_COUNT,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    if nx * ny <= 64:
        ham = build_obc_hamiltonian(v=v, t=t, lm=lm, w=w, j=j, nx=nx, ny=ny)
        eigvals, eigvecs = np.linalg.eigh(ham)
        target_idx = int(np.argmin(np.abs(eigvals)))
        target_energy = float(np.real(eigvals[target_idx]))
        vec = eigvecs[:, target_idx]
    else:
        ham_sparse = build_obc_hamiltonian_sparse(v=v, t=t, lm=lm, w=w, j=j, nx=nx, ny=ny)
        target_k = min(max(8, mode_count), ham_sparse.shape[0] - 2)
        eigvals, eigvecs = eigsh(ham_sparse, k=target_k, sigma=0.0, which="LM")
        sort_idx = np.argsort(np.real(eigvals))
        eigvals = np.real(eigvals[sort_idx])
        eigvecs = eigvecs[:, sort_idx]
        target_idx = int(np.argmin(np.abs(eigvals)))
        target_energy = float(eigvals[target_idx])
        vec = eigvecs[:, target_idx]

    cell_prob = np.zeros(nx * ny, dtype=float)
    for cell_id in range(nx * ny):
        start = cell_id * 8
        cell_prob[cell_id] = float(np.sum(np.abs(vec[start : start + 8]) ** 2))
    prob_grid = cell_prob.reshape((ny, nx))
    return eigvals, target_idx, target_energy, prob_grid


def plot_obc_spectrum(eigvals: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    indices = np.arange(eigvals.size)
    ax.scatter(indices, np.real(eigvals), s=8, c="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def build_onsite_cell_matrix_ribbon(t: float, v: float, lm: float, j: float, w: float, ky: float) -> np.ndarray:
    h0 = np.zeros((4, 4), dtype=complex)
    h0[0, 1] = t
    h0[0, 2] = t
    h0[0, 3] = v
    h0[1, 0] = t
    h0[1, 2] = v + w * np.exp(-1j * ky)
    h0[1, 3] = t
    h0[2, 0] = t
    h0[2, 1] = v + w * np.exp(1j * ky)
    h0[2, 3] = t
    h0[3, 0] = v
    h0[3, 1] = t
    h0[3, 2] = t

    h1 = np.zeros((4, 4), dtype=complex)
    h1[0, 1] = 1j * lm
    h1[0, 2] = -1j * lm
    h1[2, 3] = -1j * lm
    h1[3, 2] = 1j * lm
    h1[3, 1] = -1j * lm
    h1[1, 0] = -1j * lm
    h1[2, 0] = 1j * lm
    h1[1, 3] = 1j * lm

    h3 = np.diag([j, -j, j, -j]).astype(complex)
    s0 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.kron(h0, s0) + np.kron(h1, sz) + np.kron(h3, sz)


def build_ribbon_hamiltonian(v: float, t: float, lm: float, ky: float, w: float = 1.0, j: float = 0.0, nx: int = 20) -> np.ndarray:
    dim = 8 * nx
    ham = np.zeros((dim, dim), dtype=complex)
    onsite = build_onsite_cell_matrix_ribbon(t=t, v=v, lm=lm, j=j, w=w, ky=ky)
    tx = np.zeros((8, 8), dtype=complex)
    for spin in range(2):
        tx[0 * 2 + spin, 3 * 2 + spin] = w

    for x in range(nx):
        start = x * 8
        ham[start : start + 8, start : start + 8] += onsite
        if x > 0:
            prev = (x - 1) * 8
            ham[start : start + 8, prev : prev + 8] += tx
            ham[prev : prev + 8, start : start + 8] += tx.conj().T
    return ham


def compute_ribbon_spectrum(
    v: float,
    t: float,
    lm: float,
    w: float = 1.0,
    j: float = 0.0,
    nx: int = DEFAULT_RIBBON_NX,
    nk: int = DEFAULT_RIBBON_NK,
) -> tuple[np.ndarray, np.ndarray]:
    ky_values = np.linspace(-np.pi, np.pi, nk)
    eigvals = []
    for ky in ky_values:
        ham = build_ribbon_hamiltonian(v=v, t=t, lm=lm, ky=ky, w=w, j=j, nx=nx)
        eigvals.append(np.linalg.eigvalsh(ham).real)
    return ky_values, np.array(eigvals)


def plot_ribbon_spectrum(ky_values: np.ndarray, eigvals: np.ndarray, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for band_idx in range(eigvals.shape[1]):
        ax.plot(ky_values / np.pi, eigvals[:, band_idx], linewidth=0.6, color="tab:blue")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$k_y/\pi$")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_obc_wavefunction(prob_grid: np.ndarray, target_energy: float, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(prob_grid, origin="lower", cmap="magma")
    ax.set_title(f"{title}\n|psi|^2 of mode near E={target_energy:.3e}")
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def assign_edge_corner_states(
    rows: list[dict[str, float]],
    logs_path: Path,
    gap_threshold: float = 0.1,
    chern_jump_threshold: float = 0.5,
) -> tuple[int, list[dict[str, float | int | str]]]:
    if not rows:
        return 0, []
    index = {normalize_key(row["v"], row["t"], row["lm"]): row for row in rows}
    unique_v = sorted({float(row["v"]) for row in rows})
    unique_t = sorted({float(row["t"]) for row in rows})
    unique_lm = sorted({float(row["lm"]) for row in rows})

    special_count = 0
    special_records: list[dict[str, float | int | str]] = []
    for row in rows:
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        gap = float(row["gap"])
        chern = float(row["chern"])

        chern_jump_axes = 0
        max_chern_delta = 0.0
        for axis, value, axis_values in (("v", v, unique_v), ("t", t, unique_t), ("lm", lm, unique_lm)):
            value_idx = axis_values.index(value)
            axis_has_jump = False
            for direction in (-1, 1):
                n_idx = value_idx + direction
                if not (0 <= n_idx < len(axis_values)):
                    continue
                neighbor_val = axis_values[n_idx]
                if axis == "v":
                    n_key = normalize_key(neighbor_val, t, lm)
                elif axis == "t":
                    n_key = normalize_key(v, neighbor_val, lm)
                else:
                    n_key = normalize_key(v, t, neighbor_val)
                if n_key not in index:
                    continue
                delta = abs(chern - float(index[n_key]["chern"]))
                max_chern_delta = max(max_chern_delta, delta)
                if delta >= chern_jump_threshold:
                    axis_has_jump = True
            if axis_has_jump:
                chern_jump_axes += 1

        is_gap_small = abs(gap) < gap_threshold
        has_chern_jump = max_chern_delta >= chern_jump_threshold
        edge_state = int(is_gap_small or has_chern_jump)
        corner_state = int(edge_state and chern_jump_axes >= 2)
        row["edge_state"] = edge_state
        row["corner_state"] = corner_state

        if is_gap_small:
            log_message(logs_path, f"special_point type=small_gap v={v} t={t} lm={lm} gap={gap}")
        if has_chern_jump:
            log_message(logs_path, f"special_point type=chern_jump v={v} t={t} lm={lm} max_delta={max_chern_delta} axes={chern_jump_axes}")
        if corner_state:
            log_message(logs_path, f"special_point type=corner_candidate v={v} t={t} lm={lm}")
        if edge_state or corner_state:
            special_count += 1
            reasons = []
            if is_gap_small:
                reasons.append("small_gap")
            if has_chern_jump:
                reasons.append("chern_jump")
            if corner_state:
                reasons.append("corner_candidate")
            special_records.append(
                {
                    "v": v,
                    "t": t,
                    "lm": lm,
                    "gap": gap,
                    "chern": chern,
                    "Wilson_loop": float(row["Wilson_loop"]),
                    "Z2_topology": int(row["Z2_topology"]),
                    "edge_state": edge_state,
                    "corner_state": corner_state,
                    "is_small_gap": int(is_gap_small),
                    "is_chern_jump": int(has_chern_jump),
                    "chern_jump_axes": int(chern_jump_axes),
                    "max_chern_delta": float(max_chern_delta),
                    "selection_reason": "+".join(reasons),
                }
            )
    return special_count, special_records


def write_special_points_outputs(
    special_points_dir: Path,
    special_records: list[dict[str, float | int | str]],
    gap_threshold: float,
    chern_jump_threshold: float,
) -> None:
    special_points_dir.mkdir(parents=True, exist_ok=True)

    csv_path = special_points_dir / "special_points.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SPECIAL_POINT_FIELDS)
        writer.writeheader()
        for row in special_records:
            writer.writerow({key: row.get(key, "") for key in SPECIAL_POINT_FIELDS})

    criteria_path = special_points_dir / "selection_criteria.txt"
    with criteria_path.open("w", encoding="utf-8") as f:
        f.write("Special-point selection criteria\n")
        f.write("--------------------------------\n")
        f.write(f"1) small_gap: abs(gap) < {gap_threshold}\n")
        f.write(
            "2) chern_jump: max Chern delta against nearest neighbors "
            f"on v/t/lm axes >= {chern_jump_threshold}\n"
        )
        f.write("3) edge_state = 1 when (small_gap OR chern_jump)\n")
        f.write("4) corner_state = 1 when edge_state=1 and chern_jump_axes >= 2\n")
        f.write(
            "5) selection_reason stores trigger tags joined by '+' "
            "(small_gap, chern_jump, corner_candidate)\n"
        )

    summary_path = special_points_dir / "summary.txt"
    small_gap_count = sum(int(row["is_small_gap"]) for row in special_records)
    chern_jump_count = sum(int(row["is_chern_jump"]) for row in special_records)
    corner_count = sum(int(row["corner_state"]) for row in special_records)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"total_special_points={len(special_records)}\n")
        f.write(f"small_gap_count={small_gap_count}\n")
        f.write(f"chern_jump_count={chern_jump_count}\n")
        f.write(f"corner_candidate_count={corner_count}\n")


def write_special_point_note_chinese(
    note_path: Path,
    record: dict[str, float | int | str],
    gap_threshold: float,
    chern_jump_threshold: float,
) -> None:
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with note_path.open("w", encoding="utf-8") as f:
        f.write("特殊点说明（自动生成）\n")
        f.write("====================\n")
        f.write(
            f"参数：v={float(record['v']):.2f}, t={float(record['t']):.2f}, "
            f"lm={float(record['lm']):.2f}\n"
        )
        f.write(
            f"物理量：gap={float(record['gap']):.6e}, "
            f"chern={float(record['chern']):.6e}, "
            f"Wilson_loop={float(record['Wilson_loop']):.6f}, "
            f"Z2={int(record['Z2_topology'])}\n"
        )
        f.write(
            f"状态：edge_state={int(record['edge_state'])}, "
            f"corner_state={int(record['corner_state'])}\n\n"
        )
        f.write("该点被选为“特殊点”的原因：\n")
        f.write(
            f"1) small_gap 判据：abs(gap) < {gap_threshold}，"
            f"当前 is_small_gap={int(record['is_small_gap'])}\n"
        )
        f.write(
            "2) chern_jump 判据：参数最近邻上的最大 Chern 跳变 >= "
            f"{chern_jump_threshold}，当前 is_chern_jump={int(record['is_chern_jump'])}，"
            f"max_chern_delta={float(record['max_chern_delta']):.6e}，"
            f"chern_jump_axes={int(record['chern_jump_axes'])}\n"
        )
        f.write(
            "3) selection_reason："
            f"{record['selection_reason']}\n"
        )
        f.write(
            "\n本目录文件含义：\n"
            "- band.png：周期边界下体能带\n"
            "- ribbon.png：ribbon 几何下 E-k_y 色散\n"
            "- obc_spectrum.png：OBC 近零能谱（E vs 模态索引）\n"
            "- obc_wavefunction.png：近零模在实空间的概率分布\n"
        )


def generate_special_point_artifacts(
    model,
    special_records: list[dict[str, float | int | str]],
    special_points_dir: Path,
    logs_path: Path,
    gap_threshold: float,
    chern_jump_threshold: float,
    obc_nx: int,
    obc_ny: int,
    obc_mode_count: int,
    ribbon_nx: int,
    ribbon_nk: int,
) -> int:
    points_dir = special_points_dir / "points"
    points_dir.mkdir(parents=True, exist_ok=True)

    generated_point_count = 0
    for record in special_records:
        v = float(record["v"])
        t = float(record["t"])
        lm = float(record["lm"])
        point_name = (
            f"v_{format_param_token(v)}_t_{format_param_token(t)}_lm_{format_param_token(lm)}"
        )
        point_dir = points_dir / point_name
        point_dir.mkdir(parents=True, exist_ok=True)

        band_path = point_dir / "band.png"
        ribbon_path = point_dir / "ribbon.png"
        obc_spectrum_path = point_dir / "obc_spectrum.png"
        obc_wavefunc_path = point_dir / "obc_wavefunction.png"
        note_path = point_dir / "说明.txt"

        try:
            set_model_params(model, v=v, t=t, lm=lm, w=1.0, j=0.0)
            if not band_path.exists():
                band_data = compute_band_data(model)
                plot_band_structure(
                    eigvals=band_data,
                    save_path=band_path,
                    title=f"Band (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
                )

            if not ribbon_path.exists():
                ky_values, ribbon_eigs = compute_ribbon_spectrum(
                    v=v,
                    t=t,
                    lm=lm,
                    w=1.0,
                    j=0.0,
                    nx=ribbon_nx,
                    nk=ribbon_nk,
                )
                plot_ribbon_spectrum(
                    ky_values=ky_values,
                    eigvals=ribbon_eigs,
                    save_path=ribbon_path,
                    title=f"Ribbon (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
                )

            if not obc_spectrum_path.exists() or not obc_wavefunc_path.exists():
                eigvals, _, target_energy, prob_grid = compute_obc_spectrum_and_probability(
                    v=v,
                    t=t,
                    lm=lm,
                    w=1.0,
                    j=0.0,
                    nx=obc_nx,
                    ny=obc_ny,
                    mode_count=obc_mode_count,
                )
                if not obc_spectrum_path.exists():
                    plot_obc_spectrum(
                        eigvals=eigvals,
                        save_path=obc_spectrum_path,
                        title=f"OBC spectrum (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
                    )
                if not obc_wavefunc_path.exists():
                    plot_obc_wavefunction(
                        prob_grid=prob_grid,
                        target_energy=target_energy,
                        save_path=obc_wavefunc_path,
                        title=f"OBC wavefunction (v={v:.2f}, t={t:.2f}, lm={lm:.2f})",
                    )

            write_special_point_note_chinese(
                note_path=note_path,
                record=record,
                gap_threshold=gap_threshold,
                chern_jump_threshold=chern_jump_threshold,
            )
            generated_point_count += 1
        except Exception as error:  # pragma: no cover
            log_message(
                logs_path,
                f"special_point_artifact_failed v={v} t={t} lm={lm} error={error!r}",
            )
    return generated_point_count


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


def _get_gap_colormap(gap: np.ndarray) -> tuple[str, mcolors.Normalize]:
    gap_min = float(np.min(gap))
    gap_max = float(np.max(gap))
    if gap_min < 0.0 < gap_max:
        bound = max(abs(gap_min), abs(gap_max))
        return "coolwarm", mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    return "viridis", mcolors.Normalize(vmin=gap_min, vmax=gap_max)


def plot_chern_relationship(rows: list[dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.array([row["v"] for row in rows], dtype=float)
    t = np.array([row["t"] for row in rows], dtype=float)
    lm = np.array([row["lm"] for row in rows], dtype=float)
    chern = np.array([row["chern"] for row in rows], dtype=float)
    gap = np.array([row["gap"] for row in rows], dtype=float)
    cmap, norm = _get_gap_colormap(gap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (label, values) in zip(axes, [("v", v), ("t", t), ("lm", lm)]):
        ax.scatter(values, chern, c=gap, cmap=cmap, norm=norm, s=55, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Chern number")
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.90, wspace=0.28)
    cax = fig.add_axes([0.90, 0.16, 0.02, 0.70])
    fig.colorbar(scalar_mappable, cax=cax, label="Gap")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_state_relationship(rows: list[dict[str, float]], state_key: str, save_path: Path, title: str) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.array([row["v"] for row in rows], dtype=float)
    t = np.array([row["t"] for row in rows], dtype=float)
    lm = np.array([row["lm"] for row in rows], dtype=float)
    state_values = np.array([row[state_key] for row in rows], dtype=float)
    gap = np.array([row["gap"] for row in rows], dtype=float)
    cmap, norm = _get_gap_colormap(gap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (label, values) in zip(axes, [("v", v), ("t", t), ("lm", lm)]):
        ax.scatter(values, state_values, c=gap, cmap=cmap, norm=norm, s=55, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(label)
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(state_key)
    fig.suptitle(title)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.86, wspace=0.28)
    cax = fig.add_axes([0.90, 0.16, 0.02, 0.66])
    fig.colorbar(scalar_mappable, cax=cax, label="Gap")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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


def plot_topology_phase_map(rows: list[dict[str, float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v = np.array([row["v"] for row in rows], dtype=float)
    t = np.array([row["t"] for row in rows], dtype=float)
    lm = np.array([row["lm"] for row in rows], dtype=float)
    phase_labels = [classify_phase(row) for row in rows]
    phase_map = {"trivial": 0, "TI": 1, "HOTI": 2}
    phase_values = np.array([phase_map[label] for label in phase_labels], dtype=float)
    cmap = mcolors.ListedColormap(["#8c8c8c", "#1f77b4", "#d62728"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (label, values) in zip(axes, [("v", v), ("t", t), ("lm", lm)]):
        ax.scatter(values, phase_values, c=phase_values, cmap=cmap, norm=norm, s=60, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Phase index (0:trivial, 1:TI, 2:HOTI)")
    fig.suptitle("Topological phase map")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.14, top=0.86, wspace=0.28)
    cax = fig.add_axes([0.92, 0.18, 0.02, 0.62])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["trivial", "TI", "HOTI"])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pairwise_phase_map(
    rows: list[dict[str, float]],
    x_key: str,
    y_key: str,
    fixed_key: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    phase_map = {"trivial": 0, "TI": 1, "HOTI": 2}
    cmap = mcolors.ListedColormap(["#8c8c8c", "#1f77b4", "#d62728"])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fixed_values = sorted({float(row[fixed_key]) for row in rows})
    x_values = sorted({float(row[x_key]) for row in rows})
    y_values = sorted({float(row[y_key]) for row in rows})

    fig, axes = plt.subplots(1, len(fixed_values), figsize=(4.2 * len(fixed_values), 4), sharex=True, sharey=True)
    if len(fixed_values) == 1:
        axes = [axes]

    for ax, fixed_value in zip(axes, fixed_values):
        subset = [row for row in rows if abs(float(row[fixed_key]) - fixed_value) < 1e-9]
        x = np.array([float(row[x_key]) for row in subset], dtype=float)
        y = np.array([float(row[y_key]) for row in subset], dtype=float)
        phases = np.array([phase_map[classify_phase(row)] for row in subset], dtype=float)
        ax.scatter(x, y, c=phases, cmap=cmap, norm=norm, s=240, marker="s", edgecolors="black", linewidths=0.4)
        ax.set_title(f"{fixed_key}={fixed_value:.1f}")
        ax.set_xticks(x_values)
        ax.set_yticks(y_values)
        ax.set_xlabel(x_key)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(y_key)
    fig.suptitle(f"Pairwise phase map: ({x_key}, {y_key}) at fixed {fixed_key}")
    fig.subplots_adjust(left=0.06, right=0.89, bottom=0.14, top=0.84, wspace=0.22)
    cax = fig.add_axes([0.91, 0.18, 0.02, 0.60])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["trivial", "TI", "HOTI"])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def select_obc_diagnostic_points(rows: list[dict[str, float]], max_points: int = 6) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    seen: set[tuple[float, float, float]] = set()

    def add_row(row: dict[str, float]) -> None:
        key = normalize_key(row["v"], row["t"], row["lm"])
        if key in seen:
            return
        selected.append(row)
        seen.add(key)

    for phase in ("trivial", "TI", "HOTI"):
        for row in rows:
            if classify_phase(row) == phase:
                add_row(row)
                break

    for row in sorted(rows, key=lambda r: abs(float(r["gap"]))):
        if len(selected) >= max_points:
            break
        add_row(row)

    return selected


def generate_obc_diagnostics(
    rows: list[dict[str, float]],
    figures_dir: Path,
    logs_path: Path,
    nx: int = 20,
    ny: int = 20,
    mode_count: int = DEFAULT_OBC_MODE_COUNT,
) -> list[str]:
    generated: list[str] = []
    for row in select_obc_diagnostic_points(rows, max_points=4):
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        spectrum_path = figures_dir / f"obc_spectrum_v{v:.1f}_t{t:.1f}_lm{lm:.1f}_nx{nx}_ny{ny}.png"
        wavefunc_path = figures_dir / f"obc_wavefunc_v{v:.1f}_t{t:.1f}_lm{lm:.1f}_nx{nx}_ny{ny}.png"

        if spectrum_path.exists() and wavefunc_path.exists():
            continue

        try:
            eigvals, _, target_energy, prob_grid = compute_obc_spectrum_and_probability(
                v=v,
                t=t,
                lm=lm,
                w=1.0,
                j=0.0,
                nx=nx,
                ny=ny,
                mode_count=mode_count,
            )
            if not spectrum_path.exists():
                plot_obc_spectrum(
                    eigvals=eigvals,
                    save_path=spectrum_path,
                    title=f"OBC spectrum (v={v:.1f}, t={t:.1f}, lm={lm:.1f})",
                )
                generated.append(str(spectrum_path))
            if not wavefunc_path.exists():
                plot_obc_wavefunction(
                    prob_grid=prob_grid,
                    target_energy=target_energy,
                    save_path=wavefunc_path,
                    title=f"OBC wavefunction (v={v:.1f}, t={t:.1f}, lm={lm:.1f})",
                )
                generated.append(str(wavefunc_path))
        except Exception as error:  # pragma: no cover
            log_message(
                logs_path,
                f"obc_diagnostic_failed v={v} t={t} lm={lm} error={error!r}",
            )
    return generated


def generate_report_pdf(rows: list[dict[str, float]], figures_dir: Path, report_path: Path, summary: dict[str, int | str]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(report_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [
            "Topology Scan Report",
            "",
            f"Total points: {summary['total']}",
            f"Computed points: {summary['success']}",
            f"Skipped cached points: {summary['cached']}",
            f"Failures: {summary['failure']}",
            f"Special points: {summary['special_points']}",
            "",
            "Columns: v,t,lm,gap,chern,edge_state,corner_state,Wilson_loop,Z2_topology",
            "",
            "Sample rows:",
        ]
        for row in rows[:10]:
            lines.append(
                f"v={row['v']:.1f}, t={row['t']:.1f}, lm={row['lm']:.1f}, "
                f"gap={row['gap']:.3e}, chern={row['chern']:.3e}, "
                f"W={row['Wilson_loop']:.3f}, Z2={row['Z2_topology']}, "
                f"edge={row['edge_state']}, corner={row['corner_state']}"
            )
        ax.text(0.03, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        figure_paths = [
            "topology_phase_map.png",
            "phase_map_v_t_by_lm.png",
            "phase_map_v_lm_by_t.png",
            "phase_map_t_lm_by_v.png",
            "chern_vs_parameters.png",
            "edge_state_vs_parameters.png",
            "corner_state_vs_parameters.png",
        ]
        selected_paths = [figures_dir / name for name in figure_paths if (figures_dir / name).exists()]
        selected_paths.extend(sorted(figures_dir.glob("obc_spectrum_*.png")))
        selected_paths.extend(sorted(figures_dir.glob("obc_wavefunc_*.png")))

        for path in selected_paths:
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.name)
            pdf.savefig(fig)
            plt.close(fig)


def validate_rows(rows: list[dict[str, float]]) -> tuple[list[str], list[int]]:
    errors = []
    invalid_indices: set[int] = set()
    state_fields = {"edge_state", "corner_state", "Z2_topology"}
    for idx, row in enumerate(rows, start=1):
        for field in RESULT_FIELDS:
            if field not in row:
                errors.append(f"row={idx} missing_field={field}")
                invalid_indices.add(idx - 1)
                continue
            value = row[field]
            if value is None:
                errors.append(f"row={idx} field={field} is_none")
                invalid_indices.add(idx - 1)
                continue
            float_value = float(value)
            if not np.isfinite(float_value):
                errors.append(f"row={idx} field={field} is_non_finite value={value}")
                invalid_indices.add(idx - 1)
                continue
            if field in state_fields and float_value not in (0.0, 1.0):
                errors.append(f"row={idx} field={field} is_non_binary value={value}")
                invalid_indices.add(idx - 1)
    return errors, sorted(invalid_indices)


def recalculate_invalid_rows(model, rows: list[dict[str, float]], invalid_indices: list[int], logs_path: Path) -> None:
    for idx in invalid_indices:
        row = rows[idx]
        v = float(row["v"])
        t = float(row["t"])
        lm = float(row["lm"])
        try:
            set_model_params(model, v=v, t=t, lm=lm, w=1.0, j=0.0)
            row["gap"] = compute_bulk_gap(model, nk=11, n_occ=4)
            row["chern"] = compute_chern_number(model, nk=21, n_occ=4)
            wilson, z2 = compute_wilson_loop_and_z2(model, nk=15, n_occ=4)
            row["Wilson_loop"] = wilson
            row["Z2_topology"] = z2
            row["edge_state"] = 0
            row["corner_state"] = 0
            log_message(logs_path, f"repair_success row={idx + 1} v={v} t={t} lm={lm}")
        except Exception as error:  # pragma: no cover
            log_message(logs_path, f"repair_failed row={idx + 1} v={v} t={t} lm={lm} error={error!r}")


def write_results_csv(results_path: Path, rows: list[dict[str, float]]) -> None:
    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDS})


def run_parameter_scan() -> dict[str, int | str]:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    figures_dir = output_dir / "figures"
    special_points_dir = output_dir / "special_points"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    special_points_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    logs_path = output_dir / "logs.txt"
    report_path = output_dir / "report.pdf"
    model_path = project_root / "models" / "xtype_model.py"

    os.environ.setdefault("MPLBACKEND", "Agg")
    model = load_xtype_model(model_path)

    if logs_path.exists():
        logs_path.unlink()

    scan_step = env_float("SCAN_STEP", DEFAULT_SCAN_STEP)
    scan_step = scan_step if scan_step > 0 else DEFAULT_SCAN_STEP
    gap_threshold = env_float("SPECIAL_GAP_THRESHOLD", 0.1)
    if gap_threshold <= 0:
        gap_threshold = 0.1
    chern_jump_threshold = env_float("CHERN_JUMP_THRESHOLD", 0.5)
    if chern_jump_threshold <= 0:
        chern_jump_threshold = 0.5
    values = build_scan_values(SCAN_MIN, SCAN_MAX, scan_step)
    scan_points = list(product(values, values, values))

    cache = load_cached_rows(results_path, logs_path)
    rows: list[dict[str, float]] = []
    total_count = len(scan_points)
    success_count = 0
    failure_count = 0
    special_count = 0
    cached_count = 0

    for idx, (v, t, lm) in enumerate(scan_points, start=1):
        key = normalize_key(v, t, lm)
        band_path = figures_dir / f"band_v{float(v):.1f}_t{float(t):.1f}_lm{float(lm):.1f}.png"

        if key in cache:
            rows.append(dict(cache[key]))
            success_count += 1
            cached_count += 1
            log_message(logs_path, f"skip_cached idx={idx} v={v} t={t} lm={lm}")
            continue

        try:
            set_model_params(model, v=float(v), t=float(t), lm=float(lm), w=1.0, j=0.0)
            gap = compute_bulk_gap(model, nk=11, n_occ=4)
            chern = compute_chern_number(model, nk=21, n_occ=4)
            wilson, z2 = compute_wilson_loop_and_z2(model, nk=15, n_occ=4)
            rows.append(
                {
                    "v": float(v),
                    "t": float(t),
                    "lm": float(lm),
                    "gap": gap,
                    "chern": chern,
                    "edge_state": 0,
                    "corner_state": 0,
                    "Wilson_loop": wilson,
                    "Z2_topology": int(z2),
                }
            )
            band_data = compute_band_data(model)
            plot_band_structure(band_data, save_path=band_path, title=f"Band structure (v={v:.1f}, t={t:.1f}, lm={lm:.1f})")
            success_count += 1
        except Exception as error:  # pragma: no cover
            failure_count += 1
            log_message(logs_path, f"compute_failed idx={idx} v={v} t={t} lm={lm} error={error!r}")

    special_count, special_records = assign_edge_corner_states(
        rows=rows,
        logs_path=logs_path,
        gap_threshold=gap_threshold,
        chern_jump_threshold=chern_jump_threshold,
    )
    write_special_points_outputs(
        special_points_dir=special_points_dir,
        special_records=special_records,
        gap_threshold=gap_threshold,
        chern_jump_threshold=chern_jump_threshold,
    )
    write_results_csv(results_path, rows)

    validation_errors, invalid_indices = validate_rows(rows)
    if validation_errors:
        for error in validation_errors:
            log_message(logs_path, f"validation_error {error}")
        recalculate_invalid_rows(model, rows, invalid_indices, logs_path)
        special_count, special_records = assign_edge_corner_states(
            rows=rows,
            logs_path=logs_path,
            gap_threshold=gap_threshold,
            chern_jump_threshold=chern_jump_threshold,
        )
        write_special_points_outputs(
            special_points_dir=special_points_dir,
            special_records=special_records,
            gap_threshold=gap_threshold,
            chern_jump_threshold=chern_jump_threshold,
        )
        write_results_csv(results_path, rows)
        validation_errors, _ = validate_rows(rows)
        for error in validation_errors:
            log_message(logs_path, f"post_repair_validation_error {error}")

    if rows:
        plot_chern_relationship(rows, figures_dir / "chern_vs_parameters.png")
        plot_state_relationship(rows, "edge_state", figures_dir / "edge_state_vs_parameters.png", "Edge state vs parameters")
        plot_state_relationship(rows, "corner_state", figures_dir / "corner_state_vs_parameters.png", "Corner state vs parameters")
        plot_topology_phase_map(rows, figures_dir / "topology_phase_map.png")
        plot_pairwise_phase_map(rows, "v", "t", "lm", figures_dir / "phase_map_v_t_by_lm.png")
        plot_pairwise_phase_map(rows, "v", "lm", "t", figures_dir / "phase_map_v_lm_by_t.png")
        plot_pairwise_phase_map(rows, "t", "lm", "v", figures_dir / "phase_map_t_lm_by_v.png")
        obc_nx = env_int("OBC_NX", DEFAULT_OBC_NX)
        obc_ny = env_int("OBC_NY", DEFAULT_OBC_NY)
        obc_mode_count = env_int("OBC_MODE_COUNT", DEFAULT_OBC_MODE_COUNT)
        ribbon_nx = env_int("RIBBON_NX", DEFAULT_RIBBON_NX)
        ribbon_nk = env_int("RIBBON_NK", DEFAULT_RIBBON_NK)
        if obc_nx < 2:
            obc_nx = DEFAULT_OBC_NX
        if obc_ny < 2:
            obc_ny = DEFAULT_OBC_NY
        if obc_mode_count < 8:
            obc_mode_count = DEFAULT_OBC_MODE_COUNT
        if ribbon_nx < 2:
            ribbon_nx = DEFAULT_RIBBON_NX
        if ribbon_nk < 10:
            ribbon_nk = DEFAULT_RIBBON_NK

        obc_generated_paths = generate_obc_diagnostics(
            rows,
            figures_dir,
            logs_path,
            nx=obc_nx,
            ny=obc_ny,
            mode_count=obc_mode_count,
        )
        special_point_artifacts = generate_special_point_artifacts(
            model=model,
            special_records=special_records,
            special_points_dir=special_points_dir,
            logs_path=logs_path,
            gap_threshold=gap_threshold,
            chern_jump_threshold=chern_jump_threshold,
            obc_nx=obc_nx,
            obc_ny=obc_ny,
            obc_mode_count=obc_mode_count,
            ribbon_nx=ribbon_nx,
            ribbon_nk=ribbon_nk,
        )
    else:
        obc_generated_paths = []
        special_point_artifacts = 0

    summary = {
        "total": total_count,
        "success": success_count,
        "failure": failure_count + len(validation_errors),
        "special_points": special_count,
        "special_points_dir": str(special_points_dir),
        "cached": cached_count,
        "obc_generated": len(obc_generated_paths),
        "special_point_artifacts": special_point_artifacts,
        "results_path": str(results_path),
        "logs_path": str(logs_path),
        "figures_path": str(figures_dir),
        "report_path": str(report_path),
    }
    generate_report_pdf(rows, figures_dir, report_path, summary)

    if (
        failure_count == 0
        and not validation_errors
        and special_count == 0
        and cached_count == 0
        and logs_path.exists()
    ):
        logs_path.unlink()

    return summary


if __name__ == "__main__":
    summary = run_parameter_scan()
    print("summary")
    print(f"total={summary['total']}")
    print(f"success={summary['success']}")
    print(f"cached={summary['cached']}")
    print(f"failure={summary['failure']}")
    print(f"special_points={summary['special_points']}")
    print(f"special_points_dir={summary['special_points_dir']}")
    print(f"obc_generated={summary['obc_generated']}")
    print(f"special_point_artifacts={summary['special_point_artifacts']}")
    print(f"results={summary['results_path']}")
    print(f"figures={summary['figures_path']}")
    print(f"report={summary['report_path']}")
    if Path(summary["logs_path"]).exists():
        print(f"logs={summary['logs_path']}")
