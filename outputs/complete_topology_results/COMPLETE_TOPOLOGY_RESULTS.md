# Complete Topology Results (Curated Final Version)

## 0) Final one-paragraph conclusion

在当前模型中，最稳健的拓扑不变量证据来自 **SOC 开启 (`lm=0.1`) 后 `v≈0.6` 附近的 spin-Chern 跳变**：`C_up: 0 -> 1`、`C_down: 0 -> -1`、`C_total≈0`，并且与 M 点闭隙/重开隙、M 点 band inversion、有效质量代理变号共同出现。  
因此，当前最稳妥表述是：**Kane-Mele SOC 触发了以 M 点闭隙为核心机制的 spin-Chern 拓扑相变候选**。  
Fu-Kane / Wilson 在部分点与 spin-Chern 不一致，建议作为“实现与规范追查项”，不用于覆盖主结论。

---

## 1) Evidence ladder (by reliability)

### A. Strong evidence (推荐作为主文核心)
1. **M 点控制的临界闭隙（`v_c≈0.600000`）**  
   - `Delta_M_min ~ 1.39e-16`，`Delta_global_min ~ 1.39e-16`，且 `distance_to_M=0`。  
2. **Spin-Chern 跳变**  
   - `v≈0.58` 前后：`C_spin=0`；`v≈0.62` 后：`C_spin=1`。  
   - 同时满足 `C_up=-C_down` 与 `C_total≈0`。  
3. **机制证据配套完整**  
   - M 点 band inversion 检测为 detected（重叠交换判据）。  
   - M 点有效质量代理在 `v_c` 附近变号。

### B. Medium evidence (可作为次级结果)
1. `v≈1.09` 附近存在次级近闭隙与符号变化迹象。  
2. 但该窗口对网格参数更敏感，需加密 Nk 与更细 dv 再确认稳健性。

### C. Weak / disputed evidence (只作对照，不作终判)
1. Fu-Kane Z2 与 spin-Chern 在多个点不一致。  
2. Wilson parity 与 spin-Chern 在部分点不一致。  
3. 这些应归入“实现细节/规范跟踪/临界点数值稳定性”排查项。

---

## 2) Invariant-centric summary table

| Quantity | Main observation | Current status |
|---|---|---|
| `C_total` | SOC on/off 扫描下均接近 0（机器误差量级） | Stable |
| `C_up, C_down` | `v≈0.6` 前后由 `(0,0)` 变为 `(1,-1)` | Stable (SOC on) |
| `C_spin=(C_up-C_down)/2` | `0 -> 1`（主转变） | Stable (SOC on, main window) |
| M-point gap | 在 `v≈0.6` 闭合并重开 | Stable |
| Fu-Kane Z2 | 部分点与 spin-Chern 不一致 | Disputed |
| Wilson Z2/parity | 部分点与 spin-Chern 不一致 | Disputed |

---

## 3) What to claim / what not to claim

### 推荐 claim
- “Kane-Mele SOC 开启后，体系在 `v≈0.6` 附近发生由 M 点闭隙-重开隙驱动的 spin-Chern 重排（`C_up=-C_down`, `C_total≈0`）。”
- “该结果支持 Kane-Mele 型自旋分辨拓扑机制（QSH / Z2-like 候选）。”

### 不建议 claim
- 不把 Fu-Kane 单点结果当作最终结论。
- 不把无 SOC 的 band inversion 直接称为 Z2 相变。
- 不把近零态或边角混合态直接宣称为高阶拓扑角态。

---

## 4) No-SOC baseline (for mechanism control)

无 SOC (`lm=0`) 的 M 点质量项已解析得到：  
`m_M(v,t,w) = t + v - w`，临界条件 `v_c = w - t`。  
对 `t=0.3, w=1.0`，`v_c=0.7`，与数值能带闭隙与重开隙一致。  
这部分用于“机制对照”，不单独证明 Z2 / spin-Chern 拓扑相。

---

## 5) Curated figure order (presentation-ready)

1. `soc_M_gap_vs_v.png`  
2. `soc_M_gap_vs_global_gap.png`  
3. `soc_kmin_distance_to_M_vs_v.png`  
4. `soc_Cspin_vs_v.png`  
5. `soc_Cup_Cdown_vs_v.png`  
6. `soc_spin_chern_convergence_vs_Nk.png`  
7. `soc_M_component_weights_before_critical_after.png`  
8. `soc_M_effective_mass_vs_v.png`  
9. `no_soc_M_gap_vs_v.png`  
10. `no_soc_band_evolution_montage.png`

---

## 6) Deliverables in this folder

- This report: `COMPLETE_TOPOLOGY_RESULTS.md`  
- Key SOC figures: `soc_*.png`  
- Key no-SOC figures: `no_soc_*.png`  
- Key diagnostic table: `soc_z2_spin_chern_comparison.csv`  
- No-SOC mass fit text: `no_soc_mass_fit_result.txt`

---

## 7) Recommended next technical action

优先做“Z2 consistency 修复包”：
1. Wilson loop gauge continuity + crossing counting 的鲁棒实现；  
2. Fu-Kane parity 的 occupied 子空间与反演算符实现交叉校验；  
3. 在 `v≈0.6` 与 `v≈1.09` 两窗口做统一高分辨率复算，输出同一网格下的 `spin-Chern / Fu-Kane / Wilson` 对照表。
