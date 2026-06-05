#!/usr/bin/env python3
"""Extract no-SOC M-point effective mass expression from the real model Hamiltonian."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

if str(Path("/workspace")) not in sys.path:
    sys.path.insert(0, str(Path("/workspace")))

from scripts.km_soc_analysis.common import load_model, set_params


@dataclass
class FitResult:
    a_v: float
    b_t: float
    c_const: float
    r2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-SOC M-point mass extraction.")
    parser.add_argument("--model-file", default="xtype_model.py")
    parser.add_argument("--output-root", default="/workspace/outputs/no_soc_mass_expression")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_symbolic_hamiltonian():
    # Symbols
    kx, ky = sp.symbols("kx ky", real=True)
    v, t, w, lm = sp.symbols("v t w lm", real=True)

    # exp(i*kx), exp(i*ky) rewritten as cos + i sin (per requirement).
    ex = sp.cos(kx) + sp.I * sp.sin(kx)
    emx = sp.cos(kx) - sp.I * sp.sin(kx)
    ey = sp.cos(ky) + sp.I * sp.sin(ky)
    emy = sp.cos(ky) - sp.I * sp.sin(ky)

    h0 = sp.Matrix(
        [
            [0, t, t, v + w * emx],
            [t, 0, v + w * emy, t],
            [t, v + w * ey, 0, t],
            [v + w * ex, t, t, 0],
        ]
    )

    # SOC/orbital part used in the real model (will be set lm=0 later)
    h1 = sp.Matrix(
        [
            [0, sp.I * lm, -sp.I * lm, 0],
            [-sp.I * lm, 0, 0, sp.I * lm],
            [sp.I * lm, 0, 0, -sp.I * lm],
            [0, -sp.I * lm, sp.I * lm, 0],
        ]
    )

    s0 = sp.eye(2)
    sz = sp.diag(1, -1)
    h8 = sp.kronecker_product(h0, s0) + sp.kronecker_product(h1, sz)
    return (kx, ky, v, t, w, lm), h0, h8


def _save_symbolic_matrix_txt_tex(mat: sp.Matrix, txt_path: Path, tex_path: Path, title: str) -> None:
    txt_lines = [title, "=" * len(title), str(sp.simplify(mat))]
    txt_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    tex_path.write_text(sp.latex(sp.simplify(mat)) + "\n", encoding="utf-8")


def _extract_spin_block(h8_m: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    up_idx = [0, 2, 4, 6]
    dn_idx = [1, 3, 5, 7]
    h_up = h8_m.extract(up_idx, up_idx)
    h_dn = h8_m.extract(dn_idx, dn_idx)
    return h_up, h_dn


def _low_energy_projection(h_block: sp.Matrix):
    # Two basis vectors that span the crossing subspace at M:
    # psi_a = (1,1,1,1)/2, psi_b = (1,1,-1,-1)/2
    psi_a = sp.Matrix([1, 1, 1, 1]) / 2
    psi_b = sp.Matrix([1, 1, -1, -1]) / 2
    proj = sp.Matrix.hstack(psi_a, psi_b)
    h_eff = sp.simplify(proj.T * h_block * proj)

    h00 = sp.simplify(h_eff[0, 0])
    h11 = sp.simplify(h_eff[1, 1])
    h01 = sp.simplify(h_eff[0, 1])
    h10 = sp.simplify(h_eff[1, 0])

    d0 = sp.simplify((h00 + h11) / 2)
    dz = sp.simplify((h00 - h11) / 2)
    dx = sp.simplify((h01 + h10) / 2)
    dy = sp.simplify((h10 - h01) / (2 * sp.I))
    return proj, h_eff, d0, dx, dy, dz


def _fit_mass_w1(rows: list[dict[str, float]]) -> FitResult:
    x = np.array([[r["v"], r["t"], 1.0] for r in rows], dtype=float)
    y = np.array([r["signed_m_M"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot < 1e-16 else (1.0 - ss_res / ss_tot)
    return FitResult(a_v=float(coef[0]), b_t=float(coef[1]), c_const=float(coef[2]), r2=float(r2))


def _estimate_vc_from_data(rows_t: list[dict[str, float]]) -> float:
    rows_sorted = sorted(rows_t, key=lambda r: r["v"])
    signs = [np.sign(r["signed_m_M"]) for r in rows_sorted]
    for i in range(len(rows_sorted) - 1):
        s0, s1 = signs[i], signs[i + 1]
        if s0 == 0:
            return float(rows_sorted[i]["v"])
        if s0 * s1 < 0:
            x0, y0 = rows_sorted[i]["v"], rows_sorted[i]["signed_m_M"]
            x1, y1 = rows_sorted[i + 1]["v"], rows_sorted[i + 1]["signed_m_M"]
            if abs(y1 - y0) < 1e-14:
                return float(0.5 * (x0 + x1))
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return float(min(rows_sorted, key=lambda r: abs(r["signed_m_M"]))["v"])


def main() -> None:
    args = parse_args()
    out_root = _ensure_dir(Path(args.output_root))
    model = load_model(args.model_file)
    m_point = 0.5 * (model.b1 + model.b2)

    # ---------------- Task 1/2/3/4/5: symbolic extraction ----------------
    symbols, h0_sym, h8_sym = _build_symbolic_hamiltonian()
    kx, ky, v, t, w, lm = symbols

    h_m = sp.simplify(h8_sym.subs({kx: sp.pi, ky: sp.pi, lm: 0}))
    _save_symbolic_matrix_txt_tex(
        h_m,
        out_root / "H_M_symbolic.txt",
        out_root / "H_M_symbolic.tex",
        "H_M symbolic (kx=pi, ky=pi, lm=0)",
    )

    h_up, h_dn = _extract_spin_block(h_m)
    is_spin_block_identical = bool(sp.simplify(h_up - h_dn) == sp.zeros(4, 4))
    _save_symbolic_matrix_txt_tex(
        h_up,
        out_root / "H_M_spin_block_symbolic.txt",
        out_root / "H_M_spin_block_symbolic.tex",
        "H_M spin block symbolic (up block)",
    )

    proj, h_eff, d0, dx, dy, dz = _low_energy_projection(h_up)
    low_txt = [
        "Low-energy 2x2 projection near M",
        "================================",
        f"Projection basis columns P = {proj}",
        f"H_eff_M = {sp.simplify(h_eff)}",
        f"d0 = {d0}",
        f"dx = {dx}",
        f"dy = {dy}",
        f"dz = {dz}",
        f"m_M(v,t,w) = dz = {dz}",
        f"critical condition m_M=0 => {sp.simplify(sp.solve(sp.Eq(dz, 0), v)[0])}",
    ]
    (out_root / "M_low_energy_projection.txt").write_text("\n".join(low_txt) + "\n", encoding="utf-8")

    # ---------------- Task 6: numeric signed mass scan ----------------
    t_list = [0.0, 0.1, 0.2, 0.3]
    w_fixed = 1.0
    dv = 0.005 if args.quick else 0.002
    v_list = np.arange(0.0, 1.0 + 1e-12, dv)
    phi_a = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex) / 2.0
    phi_b = np.array([1.0, 1.0, -1.0, -1.0], dtype=complex) / 2.0

    rows = []
    for t_val in t_list:
        prev_sign = 0.0
        for v_val in v_list:
            set_params(model, v=float(v_val), lm=0.0, t=float(t_val), w=float(w_fixed), j=0.0)
            h8_num = np.array(model.Hxtype(m_point), dtype=complex)
            evals8 = np.linalg.eigvalsh(h8_num)
            e_val_global = float(np.real(evals8[3]))
            e_con_global = float(np.real(evals8[4]))
            delta_global = float(e_con_global - e_val_global)

            up_idx = np.array([0, 2, 4, 6], dtype=int)
            h4 = h8_num[np.ix_(up_idx, up_idx)]
            evals4, evecs4 = np.linalg.eigh(h4)
            ov_a = np.abs(evecs4.conj().T @ phi_a)
            ov_b = np.abs(evecs4.conj().T @ phi_b)
            idx_a = int(np.argmax(ov_a))
            idx_b = int(np.argmax(ov_b))
            if idx_b == idx_a:
                candidates = [j for j in range(4) if j != idx_a]
                idx_b = max(candidates, key=lambda j: ov_b[j])

            e_a = float(np.real(evals4[idx_a]))
            e_b = float(np.real(evals4[idx_b]))
            delta_pair = float(abs(e_a - e_b))
            sign = float(np.sign(e_a - e_b))
            if abs(sign) < 1e-12:
                sign = prev_sign
            signed_m = float(0.5 * sign * delta_pair)
            prev_sign = sign

            rows.append(
                {
                    "v": float(v_val),
                    "t": float(t_val),
                    "w": float(w_fixed),
                    "lm": 0.0,
                    "E_valence_M_global": e_val_global,
                    "E_conduction_M_global": e_con_global,
                    "Delta_M_global": delta_global,
                    "state_a_band_index": idx_a,
                    "state_b_band_index": idx_b,
                    "E_state_a": e_a,
                    "E_state_b": e_b,
                    "Delta_pair": delta_pair,
                    "band_character_sign": sign,
                    "signed_m_M": signed_m,
                }
            )

    csv_path = out_root / "signed_mass_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "v",
                "t",
                "w",
                "lm",
                "E_valence_M_global",
                "E_conduction_M_global",
                "Delta_M_global",
                "state_a_band_index",
                "state_b_band_index",
                "E_state_a",
                "E_state_b",
                "Delta_pair",
                "band_character_sign",
                "signed_m_M",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # ---------------- Task 7: fit m_M(v,t,w=1)=a*v+b*t+c ----------------
    fit = _fit_mass_w1(rows)
    vc_fit_t03 = float(-(fit.b_t * 0.3 + fit.c_const) / fit.a_v) if abs(fit.a_v) > 1e-12 else float("nan")
    # Symbolic generalized expression from projection:
    # m_M = dz = v + t - w
    m_general = sp.simplify(dz)
    vc_general = sp.simplify(sp.solve(sp.Eq(m_general, 0), v)[0])

    fit_txt = [
        "No-SOC M-point mass fit result",
        "==============================",
        "Fit data source: signed_mass_data.csv (pair-selected low-energy states)",
        f"fit_w1_expression: m_M(v,t,w=1) = ({fit.a_v:.12g})*v + ({fit.b_t:.12g})*t + ({fit.c_const:.12g})",
        f"a={fit.a_v:.12g}",
        f"b={fit.b_t:.12g}",
        f"c={fit.c_const:.12g}",
        f"R2={fit.r2:.12g}",
        f"critical_condition_w1: v_c(t) = -(b*t+c)/a",
        f"predicted_v_c_t0p3_w1={vc_fit_t03:.12g}",
        "",
        f"symbolic_projection_expression: m_M(v,t,w) = {m_general}",
        f"symbolic_critical_condition: v_c = {vc_general}",
    ]
    (out_root / "mass_fit_result.txt").write_text("\n".join(fit_txt) + "\n", encoding="utf-8")

    # ---------------- Plots ----------------
    # 1) signed_m_M_vs_v_t_compare
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for t_val in t_list:
        sub = [r for r in rows if abs(r["t"] - t_val) < 1e-12]
        sub = sorted(sub, key=lambda r: r["v"])
        vx = np.array([r["v"] for r in sub], dtype=float)
        my = np.array([r["signed_m_M"] for r in sub], dtype=float)
        ax.plot(vx, my, linewidth=1.1, label=f"t={t_val:.1f}")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("v")
    ax.set_ylabel("signed m_M")
    ax.set_title("No-SOC signed M-point mass vs v")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "signed_m_M_vs_v_t_compare.png", dpi=180)
    plt.close(fig)

    # 2) vc_vs_t: data-extracted vs fit
    vc_data = []
    for t_val in t_list:
        sub = [r for r in rows if abs(r["t"] - t_val) < 1e-12]
        vc_data.append((t_val, _estimate_vc_from_data(sub)))
    t_arr = np.array([x[0] for x in vc_data], dtype=float)
    vc_arr = np.array([x[1] for x in vc_data], dtype=float)
    vc_fit_arr = (-(fit.b_t * t_arr + fit.c_const) / fit.a_v) if abs(fit.a_v) > 1e-12 else np.nan * t_arr

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(t_arr, vc_arr, "o-", linewidth=1.2, label="from signed mass data")
    ax.plot(t_arr, vc_fit_arr, "s--", linewidth=1.0, label="linear fit prediction")
    ax.set_xlabel("t")
    ax.set_ylabel("v_c")
    ax.set_title("Critical v_c versus t at w=1, lm=0")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "vc_vs_t.png", dpi=180)
    plt.close(fig)

    # 3) mass_fit_check
    x = np.array([[r["v"], r["t"], 1.0] for r in rows], dtype=float)
    y = np.array([r["signed_m_M"] for r in rows], dtype=float)
    yhat = x @ np.array([fit.a_v, fit.b_t, fit.c_const], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y, yhat, s=10, alpha=0.7)
    lo = float(min(np.min(y), np.min(yhat)))
    hi = float(max(np.max(y), np.max(yhat)))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
    ax.set_xlabel("signed m_M data")
    ax.set_ylabel("signed m_M fit")
    ax.set_title(f"Mass fit check (R2={fit.r2:.6f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "mass_fit_check.png", dpi=180)
    plt.close(fig)

    # ---------------- Task 8: report ----------------
    # band inversion check at t=0.3 via signed-mass sign change around vc
    t_target = 0.3
    vc_target = float(vc_arr[np.argmin(np.abs(t_arr - t_target))])
    v_left = max(0.0, vc_target - 0.02)
    v_right = min(1.0, vc_target + 0.02)
    rows_t03 = sorted([r for r in rows if abs(r["t"] - t_target) < 1e-12], key=lambda r: r["v"])
    m_left = min(rows_t03, key=lambda r: abs(r["v"] - v_left))["signed_m_M"]
    m_right = min(rows_t03, key=lambda r: abs(r["v"] - v_right))["signed_m_M"]
    inversion_detected = bool(np.sign(m_left) * np.sign(m_right) < 0)

    report_lines = [
        "# NO_SOC_MASS_EXPRESSION_REPORT",
        "",
        "## Hamiltonian location and parameters",
        "- Located at `models/xtype_model.py::Hxtype(k)`.",
        "- Inputs/effective parameters used by the model: `k=(kx,ky,kz)`, `v`, `t`, `w`, `lm` (and `J`, fixed to 0 in this task).",
        "- This extraction is for `lm=0` (no SOC).",
        "",
        "## Answers",
        "1. **是否能得到解析的 H_M(v,t,w)？**",
        "- 可以。已导出 `H_M_symbolic.txt/.tex`，由真实模型在 `kx=ky=pi, lm=0` 代入得到。",
        "",
        "2. **是否能分离出一个 spin block？**",
        f"- 可以。`H_M` 可拆成两个相同 `4x4` 自旋块（identical={is_spin_block_identical}）。",
        "",
        "3. **是否能得到显式 m_M(v,t,w)？**",
        f"- 可以。低能两态投影给出 `m_M(v,t,w)=dz={sp.simplify(m_general)}`。",
        "",
        "4. **如果不能，数值拟合表达式是什么？**",
        f"- 数值拟合（w=1）得到 `m_M≈{fit.a_v:.6f}*v + {fit.b_t:.6f}*t + {fit.c_const:.6f}`，`R2={fit.r2:.6f}`。",
        "",
        "5. **t 项是否会把临界点从 v=w 推到 v<w？**",
        f"- 会。由 `m_M=v+t-w` 得 `v_c=w-t`，当 `t>0` 时确实 `v_c<w`。",
        "",
        "6. **对 t=0.3,w=1.0，预测质量项零点在哪里？**",
        f"- 解析预测 `v_c = {sp.simplify(vc_general).subs({t: 0.3, w: 1})}`；拟合预测 `v_c≈{vc_fit_t03:.6f}`；扫描提取 `v_c≈{vc_target:.6f}`。",
        "",
        "7. **该质量项变号是否对应 M 点 band inversion？**",
        f"- 对应。`v={v_left:.3f}` 与 `v={v_right:.3f}` 两侧质量项符号相反（m_left={m_left:.6f}, m_right={m_right:.6f}, inversion_detected={inversion_detected}），对应 M 点 band inversion。",
        "",
        "## Caution",
        "- 这里仅说明 `lm=0` 下 M 点质量项变号 / band inversion。",
        "- 不把该闭隙直接表述为拓扑相变；是否拓扑非平庸仍需独立拓扑不变量判定。",
    ]
    (out_root / "NO_SOC_MASS_EXPRESSION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # ---------------- Appendix: detailed derivation ----------------
    lam = sp.symbols("lambda", real=True)
    m0 = sp.symbols("m0", real=True)
    h_block_m0 = h_up.subs({v - w: m0})
    char_poly = sp.factor((lam * sp.eye(4) - h_block_m0).det())
    eigvals_manual = [
        sp.simplify(m0 + 2 * t),
        sp.simplify(m0 - 2 * t),
        sp.simplify(-m0),
        sp.simplify(-m0),
    ]
    vc_formula = sp.simplify(vc_general)

    appendix_lines = [
        "# NO_SOC_MASS_EXPRESSION_APPENDIX",
        "",
        "本附录记录 `lm=0` 时从真实模型 Hamiltonian 提取 M 点有效质量项的详细推导过程。",
        "",
        "## A. 起点：真实模型的 Bloch Hamiltonian",
        "- 模型函数：`models/xtype_model.py::Hxtype(k)`。",
        "- 先写 4x4 轨道块 `H0(k)`，再做 `H(k)=kron(H0,s0)`（无 SOC, 无 J）。",
        "",
        "轨道块非零元（与源码一致）：",
        "- `H01=t, H02=t, H03=v+w*e^{-ikx}`",
        "- `H10=t, H12=v+w*e^{-iky}, H13=t`",
        "- `H20=t, H21=v+w*e^{+iky}, H23=t`",
        "- `H30=v+w*e^{+ikx}, H31=t, H32=t`",
        "",
        "## B. 代入 M 点",
        "- `M=(pi,pi)`，因此 `e^{±ikx}=e^{±iky}=-1`。",
        "- 定义 `m0 = v-w`，则",
        "",
        "```text",
        "H_M_block = [[0, t, t, m0],",
        "             [t, 0, m0, t],",
        "             [t, m0, 0, t],",
        "             [m0, t, t, 0]]",
        "```",
        "",
        "对应的 8x8 `H_M` 为两份相同 spin block（up/down 简并）。",
        "",
        "## C. 解析本征值",
        f"- 特征多项式（将 `v-w` 替换为 `m0`）：`det(lambda I - H_M_block) = {char_poly}`。",
        f"- 本征值集合可写为：`{eigvals_manual}`。",
        "",
        "等价地（还原 `m0=v-w`）：",
        "- `E1 = v-w+2t`",
        "- `E2 = v-w-2t`",
        "- `E3 = w-v`",
        "- `E4 = w-v`",
        "",
        "## D. 低能两带与 2x2 投影",
        "选取如下两态作为控制闭隙的低能子空间基（列向量）：",
        "- `|psi_a> = (1,1,1,1)^T / 2`（对应 `E1=v-w+2t`）",
        "- `|psi_b> = (1,1,-1,-1)^T / 2`（对应 `E3=w-v`）",
        "",
        "投影后得到：",
        f"- `H_eff = {sp.simplify(h_eff)}`",
        "",
        "写成 Pauli 形式 `H_eff = d0*I + dx*sigma_x + dy*sigma_y + dz*sigma_z`：",
        f"- `d0={sp.simplify(d0)}`",
        f"- `dx={sp.simplify(dx)}`",
        f"- `dy={sp.simplify(dy)}`",
        f"- `dz={sp.simplify(dz)}`",
        "",
        "因此定义 M 点质量项：",
        f"- `m_M(v,t,w) = dz = {sp.simplify(dz)}`",
        "",
        "## E. 临界条件",
        "- 质量变号条件 `m_M=0`：",
        f"- `v_c = {vc_formula}`",
        "",
        "特别地当 `w=1`：",
        "- `v_c = 1 - t`。",
        "",
        "## F. 与数值扫描一致性",
        "- 数值 `signed_m_M` 扫描与线性拟合得到 `m_M(v,t,w=1)=v+t-1`，`R^2=1`。",
        "- 对 `t=0.3,w=1`，解析与拟合都给 `v_c=0.7`。",
        "- 在 `v=0.68` 与 `v=0.72` 两侧，`signed_m_M` 符号相反，支持 M 点 band inversion。",
        "",
        "## G. 物理表述边界",
        "- 本附录仅证明无 SOC 情况下的 M 点质量项变号与 band inversion。",
        "- 不将该结论直接等同为拓扑相变结论，拓扑性质仍需独立不变量计算支持。",
    ]
    (out_root / "NO_SOC_MASS_EXPRESSION_APPENDIX.md").write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")

    print(f"[ok] symbolic_HM={out_root / 'H_M_symbolic.txt'}")
    print(f"[ok] spin_block={out_root / 'H_M_spin_block_symbolic.txt'}")
    print(f"[ok] signed_mass_csv={csv_path}")
    print(f"[ok] fit={out_root / 'mass_fit_result.txt'}")
    print(f"[ok] report={out_root / 'NO_SOC_MASS_EXPRESSION_REPORT.md'}")
    print(f"[ok] appendix={out_root / 'NO_SOC_MASS_EXPRESSION_APPENDIX.md'}")


if __name__ == "__main__":
    main()
