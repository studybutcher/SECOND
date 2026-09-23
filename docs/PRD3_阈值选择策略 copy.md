# PRD 3：阈值选择策略 —— 用 POT 替换 bf_search

> 2026-09-13 ｜ 适用范围：**SECOND** 与 **MUTANT** 两个代码库 ｜ 状态：待评审
> 关联：`PRD1_消融实验.md`、`PRD2_推理成本统计.md`、`推理成本统计结果.md`

---

## 1. 背景与目标

### 1.1 现状：两边用的都是 oracle 阈值

两个仓库的评测入口完全一样（`eval_method.py` 的数学逐行一致）：

```python
# SECOND/eval_method.py:89        MUTANT/eval_method.py:85（函数体相同）
def bf_search(score, label, step_num):
    threshold = 93                       # 从第 93 百分位开始
    m = (-1., -1., -1.)
    for i in range(step_num):            # 700 步，步长 0.01
        target = calc_seq(score, label, threshold, calc_latency=True)   # ← 用了 label
        if target[0] > m[0]:             # 取 F1 最大的那个阈值
            m_t, m = threshold, target
        threshold += 0.01
    return m, m_t
```

`calc_seq` → `adjust_predicts` 里做 `thresh = np.percentile(score, float(threshold))`，然后在 **测试集标签** 上算 F1，取最大者。**这是拿测试集标签搜出来的最优阈值（oracle）**，带来两个问题：

| 问题 | 后果 |
|---|---|
| **不可部署** | 真实系统拿不到测试集标签，无法复现这个阈值 |
| **不可比** | 每个方法各自在自己的测试集上搜最优，等于各调各的超参；两个库/两篇论文的 F1 不在同一规则下产生 |

### 1.2 目标

把 **SECOND 和 MUTANT 都换成 POT**（Peaks Over Threshold，基于极值理论的**标签无关**阈值选择），在 **4 个数据集**（SMAP / MSL / SWaT / WADI）上跑。

| 目标 | 可验证产出 |
|---|---|
| G1 标签无关的阈值 | 阈值只由分数分布决定，不接触 `label` |
| G2 两库同协议 | 同一份 `pot_eval` 实现、同一组参数、同一 `init_score` 口径，产出一致 schema 的 JSON |
| G3 量化 oracle 的虚高 | 同时报 `bf_search`(oracle) 与 `POT` 两个 F1，给出差距 |
| G4 不改现有结论 | `bf_search` 路径保留、结果不变（回归） |

**非目标**：不改模型、不改训练、不改 `adjust_predicts`（point-adjust 保持，与领域惯例一致）。

---

## 2. POT 是什么，以及协议怎么定

### 2.1 方法

POT = 对分数分布的**尾部**拟合广义帕累托分布（GPD），再反解出"超过该阈值的概率为 `q`"的分位点。

1. 用 `init_score`（一段**无标签**的参照分数）估一个初始阈值 `t`（`level` 分位）；
2. 取 `init_score` 中超过 `t` 的**超阈值部分** `Y = X - t`，拟合 GPD(γ, σ)；
3. 由 `P(X > z | X > t) = q` 反解 `z = t + σ/γ · ((q·n/n_t)^(-γ) − 1)`，即最终阈值。

**关键：全程只用分数，不用标签。** 参考实现见 `DTAAD-main/src/pot.py` 的 `pot_eval` 与 `DTAAD-main/src/spot.py`（SPOT 库，2031 行，依赖 scipy/matplotlib/pandas/tqdm，本机 `anomaly` 环境已具备：scipy 1.15.3）。

### 2.2 `init_score` 从哪来（**必须先定死**）

两个仓库**目前都只在 val / test 上算过分**（`main.py` 里 `is_anomaly` 只被 val_loader 与 test_loader 调用），**没有训练集分数**。三个候选：

| 方案 | 来源 | 成本 | 评价 |
|---|---|---|---|
| **A. 训练集分数（推荐）** | 对 `train_loader` 跑一次推理 | 多一次推理；SWaT 训练集 49.5 万窗口，SECOND ~15 min、MUTANT ~30 min | **OmniAnomaly / DTAAD 的标准做法**，最可引用；训练集天然无标签 |
| B. 验证集分数 | 训练循环里已有的 `val_score` | **零额外成本** | 但这两个仓库的 val 是**带标签**的；虽然 POT 不用标签，但审稿人可能质疑"val 来自测试集切分" |
| C. 测试集前 N 个窗口 | 测试分数前缀 | 零成本 | 假定开头正常，SMD/PSM 类工作常用；在这 4 个数据集上不成立（异常可能出现在早期） |

**决定：主口径用 A（训练集分数）**，且为控制成本，取训练集的**前 20000 个窗口**作为 `init_score`（对 GPD 尾部拟合足够，且与测试集完全独立）。方案 C 作为敏感性对照。

> 注意：两个仓库的训练集都是**无标签**的正常数据（`get_data` 返回 `(train_data, None)`），所以方案 A 是干净的。

### 2.3 参数取值（**核心决策**）

| 参数 | 含义 | 建议主口径 | 敏感性扫描 |
|---|---|---|---|
| `q` | 风险水平（超阈概率） | **1e-3** | {1e-2, 1e-3, 1e-4, 1e-5} |
| `level` | 初始阈值分位数 | **0.98** | {0.95, 0.98, 0.99, 0.995} |
| `dynamic` | 静态 POT vs 在线自适应 DSPOT | **False**（静态） | True 作为对照 |

> ⚠️ **不做 per-dataset 调参。**
> DTAAD 的 `src/constants.py` 里有一张 `lm_d` 表（`SMAP: (0.98, 1)`、`MSL: (0.999, 1.04)`、`SWaT: (0.993, 1)`、`WADI: (0.999, 1)`），即**每个数据集用不同的 (level, scale)**，并且 `pot_eval` 里还有 `pot_th = np.mean(ret['thresholds']) * lm[1]` 这个乘性 scale。
> **这等于把 oracle 从"阈值"转移到了"(level, scale)"**——扫 4 个数据集的参数表本身就是在测试集上选优，审稿人会抓。本 PRD 采用**全数据集固定参数**，只用敏感性曲线说明结论不依赖单点取值。

### 2.4 阈值语义的衔接（必须改一处代码）

现在 `adjust_predicts` 的 `threshold` 参数**是百分位而不是绝对分数**：

```python
thresh = np.percentile(score, float(threshold))   # eval_method.py:42
predict = score > thresh
```

`bf_search` 传的是 93~100 的百分位。而 **POT 返回的是绝对阈值**。所以需要新增一个接受绝对阈值的路径，**不能复用现有 `adjust_predicts` 的 `threshold` 参数**（否则 POT 算出的 0.83 会被当成第 0.83 百分位）。

最小改法：给 `adjust_predicts` 加一个 `abs_threshold=None` 参数（默认 `None` 时行为完全不变），或新增 `adjust_predicts_abs`。**推荐前者**，保持单一实现。

---

## 3. 改动清单

### 3.1 两个仓库共同的改动

| # | 文件 | 改动 | 量级 |
|---|---|---|---|
| 1 | `eval_method.py` | `adjust_predicts` 增加 `abs_threshold` 参数（默认 None，行为不变）；新增 `pot_eval(init_score, score, label, q, level, dynamic=False)`，返回与 `bf_search` 同构的 `(metrics_tuple, threshold, predict)` | ~40 行 |
| 2 | `eval_method.py`（或新文件 `pot.py`） | 内置精简版 SPOT（只保留 POT 所需：GPD 拟合 + 分位反解），**不直接依赖 DTAAD 的 2031 行 spot.py**；若复用则整文件拷入并注明出处 | ~60 行 |
| 3 | `main.py` | 新增 `--threshold-mode {bf,pot,both}`（默认 `bf`，向后兼容）；新增 `--pot-q`、`--pot-level`、`--pot-init-n`；新增 **训练集分数计算**（仅 `pot`/`both` 模式下执行） | ~30 行 |
| 4 | 结果 JSON | 增加 `threshold_mode`、`pot` 参数字段与对应 metrics；与已有 schema 兼容（新增字段，不改旧字段） | — |

### 3.2 逐库落点

| 库 | 入口 | 训练分数怎么拿 |
|---|---|---|
| SECOND | `main.py:test` 分支（213 行附近） | `model.is_anomaly(train_loader_sub, num, con)`，返回 tensor，`.detach().numpy()` |
| MUTANT | `main.py:138` 之后 | 同上，但 MUTANT 的 `is_anomaly` 返回 **numpy 数组**（`Model.py:248`），且 `h_n/s_n` 是跨窗口状态，**必须用与训练相同的窗口顺序**（`shuffle=False`） |

> MUTANT 的 `h_n/s_n` 是有状态 Parameter，取 `init_score` 时会推进状态。**取完后必须重新加载 ckpt 再跑 test**，否则测试分数被污染。这是本 PRD 最容易踩的坑。

### 3.3 脚本

| 脚本 | 作用 |
|---|---|
| `script/addition/threshold_eval.sh` | 对已有 ckpt 跑 `--threshold-mode both`，不重训 |
| `bench/summarize_threshold.py` | 汇总 `(数据集 × 库 × 阈值模式)` → `summary.md` / `.csv` |

**不需要重训**：直接加载已有 ckpt 跑测试即可（SECOND 的 `checkpoint/ablation/*/best.pt`、MUTANT 的 `checkpoints/*_model.pt`）。

---

## 4. 实验矩阵

| 维度 | 取值 |
|---|---|
| 数据集 | SMAP / MSL / SWaT / WADI |
| 代码库 | SECOND、MUTANT |
| 阈值模式 | `bf`（oracle 基线）、`pot`（主口径）、`pot-dynamic`（对照） |
| POT 参数 | 主口径 `q=1e-3, level=0.98`；敏感性扫描见 §2.3 |
| `init_score` | 训练集前 20000 窗口（主口径）；测试前缀（对照） |

**产出**（每格一个 JSON）：`{codebase} × {dataset} × {mode}` 的 `f1 / precision / recall / threshold / TP / FP / FN`，外加记录 POT 参数、`init_score` 来源与长度、GPD 拟合是否成功。

**核心输出表**（示例形态）：

| 数据集 | 库 | bf_search (oracle) | POT (q=1e-3) | Δ |
|---|---|---|---|---|
| SMAP | SECOND | ? | ? | ? |
| SMAP | MUTANT | 0.9794（历史值） | ? | ? |
| … | | | | |

> Δ 就是"oracle 阈值虚高了多少"，这是本实验最想量化的量。

---

## 5. 验收标准

1. **回归**：`--threshold-mode bf` 的结果与改动前**逐位一致**（同 ckpt、同 seed）。
2. **标签无关性**：`pot_eval` 的函数签名与实现中**不出现 `label` 参与阈值计算**（`label` 只在算完阈值后用于评测），代码审查可验证。
3. **两库同协议**：`pot_eval` 在两个仓库中**逐字节相同**（记 SHA256），产出的 JSON 能被同一脚本 UNION。
4. **可复现**：GPD 拟合确定性，同输入同参数字节相同。
5. **拟合失败可观测**：GPD 拟合失败时**显式记录并回退**（回退到 `level` 分位点），不得静默返回 0 或 NaN。
6. **敏感性**：给出 `q`、`level` 的扫描曲线，证明结论不依赖单点取值。
7. 8 个 (库 × 数据集) 组合全部产出，`summary.md` 可直接贴论文。

---

## 6. 风险

| # | 风险 | 应对 |
|---|---|---|
| **R1** | **POT 会显著拉低 F1**（相对 oracle）。这是**预期结果**，也是本实验的价值所在 | 双列并列报告，不掩盖；把 Δ 作为主要发现 |
| **R2** | **MUTANT 的 `h_n/s_n` 状态污染**：取 `init_score` 会推进状态，导致随后的测试分数不是干净推理 | 取完 `init_score` 后**重新加载 ckpt**；在 JSON 里记录这一步 |
| **R3** | GPD 拟合失败（尾部样本不足 / 分数退化，如全 0） | 捕获异常并回退到 `level` 分位点，记录 `fit_ok: false` |
| **R4** | 两个模型的**分数尺度与分布形态差异大**，同一组 (q, level) 不一定都合适 | 敏感性扫描；分数分布直方图作为附录 |
| **R5** | 现有 `adjust_predicts` 的 `threshold` 语义是**百分位**，直接喂绝对阈值会静默出错 | §2.4 的 `abs_threshold` 参数；加一条断言（阈值必须与分数同量纲） |
| **R6** | point-adjust 本身也会抬高 F1，与 POT 的讨论混淆 | 本 PRD **不改** point-adjust（保持与领域惯例一致）；如需另议，单开一卷 |
| **R7** | DTAAD 的 `lm_d` per-dataset 表很有诱惑力，用了就"好看" | §2.3 明确禁止；若要用，必须作为"上界参考"单列一栏并注明 |
| **R8** | `init_score` 取训练集要跑一次推理，SWaT 上 SECOND ~15 min、MUTANT ~30 min | 限定前 20000 窗口；一次性成本，可接受 |

---

## 7. 待确认

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | `init_score` 用**训练集**（方案 A，标准做法）还是**验证集**（零成本）？ | 决定是否要加一次训练集推理，以及审稿风险 |
| **Q2** | 主口径 `q` / `level` 定多少？（建议 `q=1e-3, level=0.98`） | 直接决定 F1 数字 |
| **Q3** | 是否要跑 `dynamic=True`（DSPOT 在线自适应）作为对照？ | 增加实验量与叙事复杂度 |
| **Q4** | **是否连 `bf_search` 一起报**？还是只报 POT？ | 我建议双报——Δ 本身就是最有价值的发现；只报 POT 会失去"oracle 虚高多少"这个量 |
| **Q5** | 历史已有的 mamba 结果（用 `bf_search` 得到的 F1）是否需要在 POT 下重算？ | 若不重算，新旧数字不可混用；若重算，要跑全部 8 个 (库 × 数据集) |
| **Q6** | 消融实验（PRD1）的三个结构是否也要在 POT 下重评？ | 否则消融表与主表阈值口径不一致 |
