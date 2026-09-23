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

### 2.1.1 ⚠️ 实测发现：DTAAD 的静态 POT **退化成了分位数规则**

在 SPOT 库上实测（合成数据，`init` 2 万点、`test` 1 万点含 5% 异常，`level=0.98, q=1e-5`）：

```
init_threshold (训练分数 0.98 分位) = 2.0408
POT 分位点 extreme_quantile        = 3.9907      ← GPD 拟合的真实产出
mean(thresholds)                   = 2.0445
ratio mean/init_threshold          = 1.0018
第一个超阈点位置                    = 第 19 个 / 10000
th 中 != 首值的占比                 = 99.81%
```

原因在 `spot.py:417-420`（`run(dynamic=False)` 分支）：

```python
for i in range(self.data.size):
    if not dynamic:
        if self.data[i] > self.init_threshold and with_alarm:
            self.extreme_quantile = self.init_threshold   # ← 永久改写，之后不再恢复
            alarm.append(i)
    th.append(self.extreme_quantile)
```

即：遇到**第一个**超过 `init_threshold` 的点之后，`extreme_quantile` 就被永久改写成 `init_threshold`，此后 99.8% 的 `thresholds` 都是它。**GPD 拟合出的真 EVT 分位点被完全丢弃。**

因此 DTAAD 的有效阈值实为：

```
DTAAD 有效阈值  ≈  quantile(训练分数, level) × scale
```

这仍然是**合法且标签无关**的规则（部署时可从训练分数直接算出），也解释了它为什么掉点不多——它本质是分位数规则，且比真 EVT 阈值低得多（上例 2.04 vs 3.99）。

**据此定义三个口径**：

| 口径 | 阈值定义 | 标签无关 | 说明 |
|---|---|---|---|
| **`pot_dtaad`（主列）** | `mean(run(dynamic=False)['thresholds']) × scale` ≈ `quantile(训练分数, level) × scale` | ✅ | 忠实复现 DTAAD；掉点最少 |
| `pot_evt`（附列） | `initialize()` 之后的 `extreme_quantile × scale`，**绕过 `run()` 的改写** | ✅ | **真正的 EVT/POT** |
| `pot_dynamic`（附列） | `run(dynamic=True)` 的 `mean(thresholds) × scale` | ✅ | 在线自适应；实测会用到真 POT |

三者共用一个 `init_score`、一套 `(q, level, scale)`，实现上只差「取哪个量当阈值」，因此**同时产出不增加推理成本**——同一次 `initialize()` 之后即可拿到 `pot_evt` 与 `pot_dtaad`。

> **诚实标注要求（新增）**：`pot_dtaad` 这一列在论文里必须写成「沿用 DTAAD 的阈值规则」；若同时给出 `pot_evt`，可明确说明「真 EVT 阈值下 F1 为 X，DTAAD 的退化规则下为 Y」——这本身就是一个有价值的对比发现。

### 2.2 `init_score` 从哪来（**必须先定死**）

两个仓库**目前都只在 val / test 上算过分**（`main.py` 里 `is_anomaly` 只被 val_loader 与 test_loader 调用），**没有训练集分数**。三个候选：

| 方案 | 来源 | 成本 | 评价 |
|---|---|---|---|
| **A. 训练集分数（推荐）** | 对 `train_loader` 跑一次推理 | 多一次推理；SWaT 训练集 49.5 万窗口，SECOND ~15 min、MUTANT ~30 min | **OmniAnomaly / DTAAD 的标准做法**，最可引用；训练集天然无标签 |
| B. 验证集分数 | 训练循环里已有的 `val_score` | **零额外成本** | 但这两个仓库的 val 是**带标签**的；虽然 POT 不用标签，但审稿人可能质疑"val 来自测试集切分" |
| C. 测试集前 N 个窗口 | 测试分数前缀 | 零成本 | 假定开头正常，SMD/PSM 类工作常用；在这 4 个数据集上不成立（异常可能出现在早期） |

**决定：主口径用 A（训练集分数）**，且为控制成本，取训练集的**前 20000 个窗口**作为 `init_score`（对 GPD 尾部拟合足够，且与测试集完全独立）。方案 C 作为敏感性对照。

> 注意：两个仓库的训练集都是**无标签**的正常数据（`get_data` 返回 `(train_data, None)`），所以方案 A 是干净的。

### 2.3 参数取值（**已决策：沿用 DTAAD 设置 + 固定参数对照**）

**主口径：沿用 DTAAD 的 `lm_d` 设置**（每数据集一套 `(level, scale)`）。

理由：① DTAAD 用的就是 SMAP/MSL/SWaT/WADI 这四个数据集，"沿用既有工作的设置"是可公开声明的理由，便于与该文对照；② 相比 `bf_search`（在测试集上直接搜 700 个阈值），`lm_d` 只在二维里取一个点，oracle 成分弱得多。

> ⚠️ **仓库里那行取值代码有 bug，照搬时必须按实际生效值取。**
> `DTAAD-main/src/constants.py`：
> ```python
> lm = lm_d[args.dataset][1 if 'TranAD' or 'DTTAD' in args.model else 0]
> ```
> `'TranAD'` 是非空字符串、恒为真，**条件永远成立**，所以实际生效的是 **index 1**：

| 数据集 | 实际生效 `(level, scale)` | `lm_d` 里的另一项（未生效） |
|---|---|---|
| SMAP | **(0.98, 1)** | (0.98, 1) |
| MSL | **(0.999, 1.04)** | (0.97, 1) |
| SWaT | **(0.993, 1)** | (0.993, 1) |
| WADI | **(0.999, 1)** | (0.99, 1) |

`q` 沿用 DTAAD 的默认值 **`q=1e-5`**。`dynamic=False`（静态 POT，与 DTAAD 一致）。

**对照列：全数据集固定参数** —— `q=1e-3, level=0.98, scale=1`，用于回答"如果不做 per-dataset 选择，POT 是什么水平"。两列并列报告，被问到时拿得出统一口径的数。

**敏感性扫描**（只用来说明结论不依赖单点取值，不作为主表）：`q ∈ {1e-2, 1e-3, 1e-4, 1e-5}`、`level ∈ {0.95, 0.98, 0.99, 0.995}`。

> 诚实标注要求：主表须注明"`(level, scale)` 沿用 DTAAD 设置"，**不得写成"我们的 POT 配置"**。若审稿人追问参数来源，答案是"引用既有工作"，而不是"我们调出来的"。

### 2.3.1 参考实现怎么用

**直接复用 DTAAD 的结构**，不重新实现：

| 文件 | 处置 |
|---|---|
| `DTAAD-main/src/spot.py`（2031 行，SPOT 库） | **整文件拷入**两个仓库（放 `third_party/spot.py` 或 `eval_spot.py`），文件头注明出处与许可证 |
| `DTAAD-main/src/pot.py` 的 `pot_eval` | 参考其结构（`fit → initialize → run(dynamic=False)` → `mean(thresholds) * scale` → `adjust_predicts`），去掉对 `src.constants` 的依赖，把 `(level, scale)` 改为显式传参 |

依赖已具备（`anomaly` 环境：scipy 1.15.3、matplotlib、pandas、tqdm）。

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

### 3.1 两个仓库共同的改动（**全部条件式，默认行为零变化**）

总原则：**只新增，不改变既有路径**。所有新行为都挂在开关后面，默认关闭时输出与现在**逐位一致**。

| # | 文件 | 改动 | 条件式设计 | 量级 |
|---|---|---|---|---|
| 1 | `eval_method.py` | `adjust_predicts` 增加 `abs_threshold=None` 参数 | **默认 `None` 时走原有 `np.percentile` 分支**，行为完全不变 | ~6 行 |
| 2 | `eval_method.py` | 新增 `pot_eval(init_score, score, label, q, level, scale, dynamic=False)`，返回与 `bf_search` 同构的 `(metrics_tuple, threshold, predict)` | **纯新增函数**，`bf_search` 一行不动 | ~40 行 |
| 3 | 新文件 `pot_spots.py` | 拷入 SPOT 库（DTAAD 的 `src/spot.py`）+ `pot_eval` 包装 | **纯新增文件**，不改任何既有文件 | ~2100 行（整文件拷入） |
| 4 | `main.py` | 新增 `--threshold-mode {bf,pot,both}`、`--pot-q`、`--pot-level`、`--pot-scale`、`--pot-init-n`、`--pot-dynamic` | **`--threshold-mode` 默认 `bf`**，即默认完全走现有逻辑 | ~15 行 |
| 5 | `main.py` | 新增训练集分数计算（取前 `--pot-init-n` 窗口） | **仅在 `pot`/`both` 模式下执行**；`bf` 模式下这段代码不运行，`train_loader` 不额外推理 | ~20 行 |
| 6 | 结果 JSON | 增加 `threshold_mode` / `pot_params` / `init_score` 字段与对应 metrics | **新增字段，不改旧字段**；旧脚本读旧字段仍可用 | — |

**不做的事**：不删 `bf_search`、不改 `calc_point2point` / `calc_seq` 的现有签名、不改 point-adjust 逻辑、不动模型与训练代码。

> 验收项 1（§5）就是这条原则的可验证形式：`--threshold-mode bf` 的输出与改动前逐位一致。

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

### 4.0 范围（已收窄）

**只评原始代码的 checkpoint**，**不评消融变体**（transformer / lstm 不纳入）。范围 = **2 个代码库 × 4 个数据集 = 8 格**。

被评估的 ckpt（与当前 `Model.py` 兼容的 4 个）：

| 代码库 | 数据集 | ckpt | W |
|---|---|---|---|
| SECOND | SMAP | `checkpoint/casestudy补充/SMAP/best.pt` | 70 |
| SECOND | MSL | `checkpoint/casestudy/MSL/best.pt` | 70 |
| SECOND | SWaT | `checkpoint/casestudy补充/SWaT/best.pt` | 35（`main.py` 强制） |
| SECOND | WADI | `checkpoint/1-casestudy/WADI/best.pt` | ⚠️ **140**，与默认 70 不符 |
| MUTANT | 4 个 | `checkpoints/{dataset}_model.pt` | 各自推荐值 |

> ⚠️ **WADI 的 W 不一致**：仓库里唯一可用的 SECOND-WADI ckpt 是 W=140 训的，而默认配置是 W=70。评估时必须用 **W=140** 才能加载。结果表里要标注这一格是 W=140，不能与其余三格默认并列而不说明。

### 4.1 实验矩阵

| 维度 | 取值 |
|---|---|
| 数据集 | SMAP / MSL / SWaT / WADI |
| 代码库 | SECOND、MUTANT |
| 阈值模式 | `bf`（oracle 基线）、**`pot_dtaad`（主口径）**、`pot_evt`、`pot_dynamic`（后两者附列） |
| POT 参数 | 主口径 = DTAAD 的 `lm_d`（见 §2.3 表），`q=1e-5` |
| `init_score` | 训练集前 20000 窗口 |

**四个模式都跑**：`bf` 是 oracle 基线；`pot_dtaad` / `pot_evt` / `pot_dynamic` 三个共用同一次 `initialize()`，**只差取哪个量当阈值，不增加推理成本**。

**主张以 `pot_dtaad` 为准**（忠实复用 DTAAD 的规则）。若 `pot_evt` 或 `pot_dynamic` 明显更好，可以在文中提一句，但**不得据此改换主口径**——那等于在测试集上挑口径，又回到 oracle 的老路。

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

## 7. 决策记录

### 已定

| # | 问题 | 决定 |
|---|---|---|
| **Q1** | `init_score` 来源 | ✅ **训练集**（方案 A）。取前 **20000** 窗口（`level=0.98` 时约 400 个超阈值样本，足够拟合 GPD）。训练集本就无标签（`get_data` 返回 `(train_data, None)`） |
| **Q2** | POT 参数 | ✅ **沿用 DTAAD 的 `lm_d`**：`(level, scale)` 按 §2.3 表逐数据集取值，`q=1e-5`，`dynamic=False`。另设**全数据集固定参数**（`q=1e-3, level=0.98, scale=1`）为对照列 |
| **Q3** | 实现来源 | ✅ **直接复用 DTAAD 的结构**：整文件拷入 `spot.py`，`pot_eval` 参考 `DTAAD-main/src/pot.py` 改写为显式传参版 |
| **Q4** | 代码改动方式 | ✅ **全部条件式增量**：`--threshold-mode` 默认 `bf`，默认路径输出与现在逐位一致；只新增不修改（详见 §3.1） |
| **Q5** | 是否双报 `bf_search` | ✅ **双报**。`bf`(oracle) 与 `pot` 两列并列，Δ 就是"oracle 虚高了多少"——这是本实验最有价值的产出 |
| **Q6** | `dynamic` 用哪个？ | ✅ **两个都跑**。主口径 = `dynamic=False`（DTAAD 设置）；`dynamic=True` 作附列。成本几乎为零（同一次推理），无需先验猜测 |
| **Q7** | 哪些 ckpt 要评？ | ✅ **只评原始代码的 ckpt**，不评消融变体。范围 = 2 库 × 4 数据集 = **8 格**（清单见 §4.0） |
| **Q8** | 消融变体是否重评？ | ✅ **不重评**。消融表沿用 `bf_search`，与主表口径不同，两张表分别标注、**不得并排混用** |
| **Q9** | 固定参数对照列进不进主表？ | ✅ **不进主表，放附录**。它的作用是回应"你的参数是不是也在测试集上选的"，不是提分手段 |

### 关于 Q9 的说明（避免误解）

**固定参数列大概率给出更低的 F1**——它没有为每个数据集调过参数。它的价值是**在被质疑时拿得出统一口径的数**：

> 审稿人："你的 `(level, scale)` 是按数据集选的，是不是也在测试集上调的？"
> 你："主表沿用 DTAAD 设置以便对照；附录也给了全数据集统一参数（`q=1e-3, level=0.98, scale=1`）的结果。"

所以它默认生成、进附录，**不参与主表比较**。如果只关心"不掉分"，主表看 DTAAD 那套即可。

### 仍开放

| # | 问题 | 影响 |
|---|---|---|
| **Q10** | WADI 那一格 SECOND 用的是 W=140 的 ckpt（仓库里唯一可加载的），与其余三格的 W=70 不同。是接受并标注，还是另训一个 W=70 的 WADI？ | 影响表格可读性与一致性；另训约 20 分钟（WADI 训练集仅 2 万点） |
