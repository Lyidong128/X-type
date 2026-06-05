# NO_SOC_MASS_EXPRESSION_REPORT

## Hamiltonian location and parameters
- Located at `models/xtype_model.py::Hxtype(k)`.
- Inputs/effective parameters used by the model: `k=(kx,ky,kz)`, `v`, `t`, `w`, `lm` (and `J`, fixed to 0 in this task).
- This extraction is for `lm=0` (no SOC).

## Answers
1. **是否能得到解析的 H_M(v,t,w)？**
- 可以。已导出 `H_M_symbolic.txt/.tex`，由真实模型在 `kx=ky=pi, lm=0` 代入得到。

2. **是否能分离出一个 spin block？**
- 可以。`H_M` 可拆成两个相同 `4x4` 自旋块（identical=True）。

3. **是否能得到显式 m_M(v,t,w)？**
- 可以。低能两态投影给出 `m_M(v,t,w)=dz=t + v - w`。

4. **如果不能，数值拟合表达式是什么？**
- 数值拟合（w=1）得到 `m_M≈1.000000*v + 1.000000*t + -1.000000`，`R2=1.000000`。

5. **t 项是否会把临界点从 v=w 推到 v<w？**
- 会。由 `m_M=v+t-w` 得 `v_c=w-t`，当 `t>0` 时确实 `v_c<w`。

6. **对 t=0.3,w=1.0，预测质量项零点在哪里？**
- 解析预测 `v_c = 0.700000000000000`；拟合预测 `v_c≈0.700000`；扫描提取 `v_c≈0.700000`。

7. **该质量项变号是否对应 M 点 band inversion？**
- 对应。`v=0.680` 与 `v=0.720` 两侧质量项符号相反（m_left=-0.020000, m_right=0.020000, inversion_detected=True），对应 M 点 band inversion。

## Caution
- 这里仅说明 `lm=0` 下 M 点质量项变号 / band inversion。
- 不把该闭隙直接表述为拓扑相变；是否拓扑非平庸仍需独立拓扑不变量判定。
