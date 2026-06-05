# NO_SOC_MASS_EXPRESSION_APPENDIX

本附录记录 `lm=0` 时从真实模型 Hamiltonian 提取 M 点有效质量项的详细推导过程。

## A. 起点：真实模型的 Bloch Hamiltonian
- 模型函数：`models/xtype_model.py::Hxtype(k)`。
- 先写 4x4 轨道块 `H0(k)`，再做 `H(k)=kron(H0,s0)`（无 SOC, 无 J）。

轨道块非零元（与源码一致）：
- `H01=t, H02=t, H03=v+w*e^{-ikx}`
- `H10=t, H12=v+w*e^{-iky}, H13=t`
- `H20=t, H21=v+w*e^{+iky}, H23=t`
- `H30=v+w*e^{+ikx}, H31=t, H32=t`

## B. 代入 M 点
- `M=(pi,pi)`，因此 `e^{±ikx}=e^{±iky}=-1`。
- 定义 `m0 = v-w`，则

```text
H_M_block = [[0, t, t, m0],
             [t, 0, m0, t],
             [t, m0, 0, t],
             [m0, t, t, 0]]
```

对应的 8x8 `H_M` 为两份相同 spin block（up/down 简并）。

## C. 解析本征值
- 特征多项式（将 `v-w` 替换为 `m0`）：`det(lambda I - H_M_block) = -(lambda + m0)**2*(-lambda + m0 + 2*t)*(lambda - m0 + 2*t)`。
- 本征值集合可写为：`[m0 + 2*t, m0 - 2*t, -m0, -m0]`。

等价地（还原 `m0=v-w`）：
- `E1 = v-w+2t`
- `E2 = v-w-2t`
- `E3 = w-v`
- `E4 = w-v`

## D. 低能两带与 2x2 投影
选取如下两态作为控制闭隙的低能子空间基（列向量）：
- `|psi_a> = (1,1,1,1)^T / 2`（对应 `E1=v-w+2t`）
- `|psi_b> = (1,1,-1,-1)^T / 2`（对应 `E3=w-v`）

投影后得到：
- `H_eff = Matrix([[2*t + v - w, 0], [0, -v + w]])`

写成 Pauli 形式 `H_eff = d0*I + dx*sigma_x + dy*sigma_y + dz*sigma_z`：
- `d0=t`
- `dx=0`
- `dy=0`
- `dz=t + v - w`

因此定义 M 点质量项：
- `m_M(v,t,w) = dz = t + v - w`

## E. 临界条件
- 质量变号条件 `m_M=0`：
- `v_c = -t + w`

特别地当 `w=1`：
- `v_c = 1 - t`。

## F. 与数值扫描一致性
- 数值 `signed_m_M` 扫描与线性拟合得到 `m_M(v,t,w=1)=v+t-1`，`R^2=1`。
- 对 `t=0.3,w=1`，解析与拟合都给 `v_c=0.7`。
- 在 `v=0.68` 与 `v=0.72` 两侧，`signed_m_M` 符号相反，支持 M 点 band inversion。

## G. 物理表述边界
- 本附录仅证明无 SOC 情况下的 M 点质量项变号与 band inversion。
- 不将该结论直接等同为拓扑相变结论，拓扑性质仍需独立不变量计算支持。
