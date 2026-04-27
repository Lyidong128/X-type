# 拓扑研究执行流程（可直接落地）

本流程面向当前仓库，目标是把“参数扫描 -> 特殊点筛选 -> 单点深度分析 -> 报告/交付”标准化为可复现流水线。

## 0. 总体原则

- 每个阶段都要有：`输入`、`命令`、`输出`、`验收`。
- 所有结果默认写入 `outputs/`，避免散落。
- 每次运行前后记录 `git commit`，保证可追溯。
- 若中断，优先使用脚本已有的“跳过已完成点”能力续跑，不重复计算。

## 1. 阶段A：基线拓扑扫描（全参数空间）

### 1.1 目标

在 `v,t,lm ∈ [0,1]` 上生成基础拓扑量与相图：

- `gap`, `chern`, `Wilson_loop`, `Z2_topology`
- `edge_state`, `corner_state`

### 1.2 命令

- 默认 2D 模型：
  - `python3 scripts/run_scan.py`
- 指定模型（如 3D）：
  - `MODEL_FILE=xtype_model_3d.py python3 scripts/run_scan.py`
- 常用参数（可选）：
  - `SCAN_STEP=0.1`
  - `OBC_NX=20 OBC_NY=20`

### 1.3 关键输出

- `outputs/results.csv`
- `outputs/figures/*.png`
- `outputs/report.pdf`
- `outputs/logs.txt`（若有错误）

### 1.4 验收

- `results.csv` 行数与扫描点数一致（0~1 步长 0.1 时应为 1331）。
- 关键字段无缺失：`v,t,lm,gap,chern,Wilson_loop,Z2_topology`。

---

## 2. 阶段B：每个点完整产物生成（单点文件夹）

### 2.1 目标

对每个点生成完整可视化与筛选态图：

- `band.png`
- `ribbon.png`
- `obc_spectrum_e_vs_index.png`
- `obc_wavefunction_filtered.png`
- `point_summary.json`

### 2.2 命令

- 全量（推荐）：
  - `python3 scripts/run_all_points_full_package.py --obc-k 64 --obc-min-k 12`
- 小规模验证：
  - `python3 scripts/run_all_points_full_package.py --point-limit 5`

### 2.3 关键输出

- `outputs/all_points_full_package/points/<point_id>/...`
- `outputs/all_points_full_package/all_points_summary.csv`

### 2.4 验收

- 点目录数 = 1331：
  - `ls outputs/all_points_full_package/points | wc -l`
- 汇总行数 = 1332（含表头）：
  - `wc -l outputs/all_points_full_package/all_points_summary.csv`

---

## 3. 阶段C：筛选特殊拓扑点并独立打包

### 3.1 筛选口径（当前项目采用）

- `robust_nontrivial`：
  - `gap > 0.1` 且 `(Z2=1 或 |Chern|>=0.5 或 corner_state=1)`
- `near_transition`：
  - `|gap| <= 0.1` 且 `((Z2=1 或 |Chern|>=0.5 或 corner_state=1) 或 edge_state=1)`

### 3.2 输出

- `outputs/topology_special_points_package/`
  - `points/<point_id>/...`
  - `special_points_summary.csv`
  - `selection_criteria.txt`
  - `missing_points.txt`

### 3.3 压缩交付

- 单包（本地）：
  - `outputs/topology_special_points_package.zip`
- GitHub 网页下载友好分卷（45MB）：
  - `outputs/topology_special_points_package.zip.part.000` ~ `.004`

### 3.4 合并命令（用户侧）

- `cat topology_special_points_package.zip.part.* > topology_special_points_package.zip`
- `unzip topology_special_points_package.zip`

---

## 4. 阶段D：重选态深度分析（科研论证增强）

### 4.1 目标

验证“旧选态是否低估边界性”，输出 old/new 对照与报告。

### 4.2 命令

- `python3 scripts/run_reselected_analysis.py --package-dir outputs/physical_special_points_package --output-dir results_reselected --config scripts/reselected_analysis_config.json`

### 4.3 关键输出

- `results_reselected/summary_tables/*.csv`
- `results_reselected/per_point/*/*.png`
- `results_reselected/report/special_points_reselected_report.pdf`

---

## 5. 阶段E：发布与归档规范

### 5.1 Git 规则

- 代码、清单、分卷包分开提交，避免超大单次变更。
- 超过 100MB 的单文件不要直接 push 到 GitHub（需分卷或 LFS）。

### 5.2 最终交付最小集合

- 主结果说明（筛选依据 + 点数统计）
- `special_points_summary.csv`
- 分卷压缩包及合并命令
- 若有报告：PDF + 关键图路径

---

## 6. 每次执行的标准检查清单（Checklist）

- [ ] `results.csv` 存在且字段齐全  
- [ ] `all_points_full_package` 点目录数正确  
- [ ] `special_points_summary.csv` 分类列存在  
- [ ] 分卷可重组并通过 `zip` 完整性校验  
- [ ] 交付说明包含“筛选依据 + 下载方式 + 合并命令”

