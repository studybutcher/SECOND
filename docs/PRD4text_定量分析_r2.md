# PRD4 · 异常污染动机定量实验（r2）

> r1 见 `docs/PRD4text_定量分析.md`。r2 的改动集中在三处：**(a) 把术语钉到代码行**（r1 的 `MPA`/`CRR` 在代码里不存在）、**(b) 补上 r1 完全缺失的执行流水线**（先造数据、再训练）、**(c) 修掉三个会让结论无法归因的方法论漏洞**（数据量混淆、重建指标的循环论证、图表与自身原则不一致）。
>
> 依赖：**PRD1**（消融变体，已实现）、**PRD3.1**（non-oracle 阈值协议）。产出图规格见 `docs/PRD4_FigX_FigY_绘图规格.md`。

---

## 0. r2 相对 r1 的变更清单

| # | r1 的问题 | r2 的处理 |
|---|---|---|
| 1 | `MPA`/`CRR` 在代码中**无对应标识符**；`MSSA` 未与 `--backbone` 对应 | §2 给术语→代码映射表（带 `file:line`） |
| 2 | 完全**没有执行流水线**：没说先造数据再训练，没有目录/命名/顺序/产物 | §4 五阶段流水线 + §12 计算预算 + §14 交付物清单 |
| 3 | **数据量混淆**：ρ=20% 时若总长固定，则正常样本少了 20%，性能下降无法归因于"污染" | §5.2 增加 **length-matched clean 对照**（`clean-L80`） |
| 4 | `E_A(ρ)` 在 **test 集**上测，但污染只加在 **train** 集——"模型学会重建异常"在 test 上**原理上测不出来** | §8.3 增加 `E_A^inj`（在被注入的训练段上测），`E_A` 保留为 test 侧泛化指标 |
| 5 | 图 (a) 画**绝对** AUPRC，但三个变体的 ρ=0 基线不同 → 混了"起点低"与"退化快" | §9.1 改画**归一化退化** `ΔAUPRC(ρ)`；绝对曲线仅作附录 |
| 6 | `--noise` 注入发生在 MinMax 缩放**之后**，注入值 ±2/±5 落在 [0,1] 空间 → 物理无意义 | §6 明确注入空间的选择及对应代码改动 |
| 7 | 未提**窗口级污染率**：ρ 定义在时间戳级，但模型训练在窗口上，短段会放大曝光 | §5.6 强制报告 `ρ_win` |
| 8 | 异常强度 `α` 只说"保持不变"，未给值、未说怎么定 | §5.4 加 **α 标定流程**（以 clean 模型的检出率为准） |
| 9 | seed 同时扰动数据与初始化，两个方差源**混在一起** | §9.3 数据 seed × 初始化 seed 的方差分解 |
| 10 | 只有 mean±std，没有显著性检验 | §9.2 配对检验（同 seed 同数据 → 可配对） |
| 11 | 假设 H1–H4 没有**证伪条件** | §11 每个假设加"什么结果会推翻它" |
| 12 | `loss_c`（视图一致性损失）r1 从未提及，却与 MPA 结构性耦合 | §2.1 明确其归属，§3 变体设计给出处置 |
| 13 | 大量重复（ρ 网格出现 4 次、`E_N/E_A/Gap` 出现 4 次）；894 行里约 2/3 是重复与排版 | r2 全文约 300 行，同一事实只出现一次 |
| 14 | 未说明与 `script/robust/*.sh`（现役 ρ=0.5 随机替换实验）的关系 | §15 明确处置 |

---

## 1. 目的

回应两条审稿意见：

- **R1｜动机不明确**：需定量说明"训练集中的异常观测会干扰正常模式学习"这一现象的真实影响。
- **R2｜传统重构方法局限性说明不足**：需说明 PESR 的模块设计为何能缓解该干扰。

**方法**：**controlled training contamination** —— 固定其余一切，只调训练集的异常污染比例 ρ，观察检测性能与重建行为如何随 ρ 变化。

**不做**：不重跑 SOTA 对比。跨模型比较（架构、重构目标、分数定义都不同）无法归因到 PESR 的内部机制，故本实验**只与自身消融比**（理由见 §3.3）。

---

## 2. 术语 → 代码映射（r1 缺失，必读）

r1 通篇使用 `MPA`/`MSSA`/`CRR`，但**这三个词在代码中没有任何对应标识符**（全仓 grep 仅命中该 PRD 自身）。实现前必须先钉死：

| 论文术语 | 代码实体 | 位置 | 消融方式 | 现在可跑？ |
|---|---|---|---|---|
| **MSSA** | `self.attmamba`（8 × `attmambaBlock`） | `model/Model.py:120`，`model/mamba.py:340` | `--backbone {transformer,lstm}` | ✅ 已实现（PRD1） |
| **MPA** | 多 patch 分块聚合：`data_patch` + `embedding_inpatch` + `embedding_patchwise` + `projection1/2`，`patch_size=[5,7]` | `model/Model.py:201-223` | 需新变体，见 §3.2 | ⚠️ 未实现 |
| **CRR** | 重构头：`encoder`/`decoder`（tabular VAE）+ `predict`（`recon_mu` 出分） | `model/Model.py:111-112, 269-294` | **不消融**（所有变体保留） | ✅ 已实现 |
| ⚠️ **（r1 未提及）** | **视图一致性损失 `loss_c`** = `x_inpatch` 与 `x_patchwise` 的余弦相似度 | `model/Model.py:218` | 无法单独消融，见下 | ✅ 已实现 |

### 2.1 `loss_c` 的归属必须显式声明

`loss_c` 定义在**两个 patch 视图之间**（`criterion(x_inpatch, x_patchwise)`）。因此：

> **移除 MPA 必然同时移除 `loss_c`** —— 该损失在数学上没有 MPA 就不存在。

这在论文里必须写清楚，否则审稿人会问"你的 w/o MPA 到底是只去了多尺度分块，还是连一致性正则也去了"。r2 的处置（见 §3.2）是**承认二者不可分**，并把 `w/o MPA` 定义为"去掉多视图分块聚合及其一致性正则"，在正文明确披露。若论文坚持要把 `loss_c` 作为独立的 **CRR** 来消融，则 r1 的"所有模型均保留 CRR"这一行必须删掉——二者矛盾，r1 没有察觉。

---

## 3. 实验变体

### 3.1 主实验（3 个变体）

| 变体 | MPA | MSSA | CRR | `--backbone` | 备注 |
|---|:-:|:-:|:-:|---|---|
| **PESR** | ✓ | ✓ | ✓ | `mamba` | 完整模型 |
| **w/o MPA** | ✗ | ✓ | ✓ | *（需新增）* | 去掉多视图分块 + `loss_c`（§2.1） |
| **w/o MSSA** | ✓ | ✗ | ✓ | `transformer` 或 `lstm` | 二选一，默认 `transformer` |

> `CRR` 列在所有变体中恒为 ✓（`predict`/VAE 路径一字不改），故该列在论文表格中可省略，只在正文用一句话说明"未做 AE 对照"。

### 3.2 `w/o MPA` 的实现路径

代码里**已经存在**一条替代管线 `simpip`（`model/Model.py:251-267`，调用点 `133`/`174` 被注释掉）。它与 `pipline` 的差异恰好就是 MPA + `loss_c`：

| | `pipline`（PESR） | `simpip` |
|---|---|---|
| 分块嵌入 | `embedding_inpatch` + `embedding_patchwise`（2 个尺度） | 单个 `simpembdding`（整窗） |
| 一致性损失 | `loss_c ≠ 0` | `return ..., 0` |
| 其余（`attmamba`→`project4`→`attention_sim`→`attention_ALL`→`projection3`→`predict`） | — | 完全相同 |

所以 `w/o MPA` **不需要新写模型**，只需把它接进 `--backbone` 注册表：

```python
# main.py:20 — 新增一项即可（变体文件由 make_ablation_models.py 生成，禁止手改）
BACKBONE_REGISTRY = {
    'mamba':       'model.Model',
    'transformer': 'model.Model_trans',
    'lstm':        'model.Model_lstm',
    'nompa':       'model.Model_nompa',   # 新增：forward/reconstructed_probability 调 simpip
}
```

> **前置校验（必须先做）**：`simpip` 是历史遗留路径，可能已与当前 `Model.py` 脱节。实现前先跑一次 `--backbone nompa` 的 **SMAP ρ=0 冒烟测试**，确认它能训出接近论文报告的 F1；若显著偏低，说明 `simpip` 有 bug，此时再退回"新建变体文件"方案。**不要把"未验证的旧路径"当成已验证的消融基线。**

### 3.3 为什么只与自己比

`w/o MPA` 与 `w/o MSSA` 就是 R2 要回答的对象（PESR 的**具体设计**是否有用）。外部重建基线（AE/USAD 等）的重构目标与分数定义都不同，即使它们在污染下表现不同，也无法归因到 PESR 的内部机制。故不引入外部 baseline；若审稿人要求 external validity，**另做**一个附录实验，不混入本篇。

### 3.4 其它消融（不做）

`w/o Mutual Attention`、`w/o Global Attention`、`w/o Local Correlation`、`w/o Temporal State Tracker` 等**不纳入**。否则 motivation experiment 会膨胀成第二套完整 ablation study，反而稀释 R1/R2 的针对性。

---

## 4. 执行流水线

r1 完全没有这一节——这是它最不可实现的地方。五个阶段，**严格串行**，中间产物落盘、可校验、可重入。

```
Stage 0  α 标定          →  artifacts/contam/calib_<ds>.json
Stage 1  污染数据生成     →  artifacts/contam/<ds>_rho<RR>_seed<S>.npy
                            artifacts/contam/<ds>_rho<RR>_seed<S>.json   （manifest）
Stage 2  校验             →  断言 nested / ρ 精确 / 哈希 / ρ_win 记录
Stage 3  训练+评测        →  results/contam/<backbone>_<ds>_rho<RR>_seed<S>.json
Stage 4  聚合出表出图     →  results/contam/summary.{md,csv} + figs/*.pdf
```

**为什么必须先把数据物化、再训练**（用户的直觉是对的，但理由要写实）：

1. **可审计**：`.npy` 是确定的字节流，可算 SHA256、可在论文附录里给；进程内 `np.random` 生成的数据无法事后核验。
2. **跨变体同一份数据**：虽然 `set_seed` + `add_noise` 在模型创建之前（`main.py:155` vs `173`）理论上能让三个变体抽到同一串随机数，但这依赖"`main.py` 第 173 行之前不再引入任何随机调用"——**任何上游改动都会静默改变污染数据**，且无法察觉。物化后这个耦合消失。
3. **省算力**：生成一次（秒级），训练 58 次/数据集（见 §12）。进程内生成等于每次训练都要重新生成。
4. **能校验**：nested 性质（§5.3）只能在生成后断言，不能在训练时顺带检查。

### 4.1 目录与命名

```
artifacts/contam/<dataset>_rho<RR>_seed<S>.npy     # 污染后的原始尺度训练数组，[N, C] float32
artifacts/contam/<dataset>_rho<RR>_seed<S>.json    # manifest：见 §5.5
artifacts/contam/calib_<dataset>.json              # α 标定结果：见 §5.4
results/contam/<backbone>_<dataset>_rho<RR>_seed<S>.json   # 复用 main.py 现有落盘 schema
checkpoint/contam/<backbone>/<dataset>/rho<RR>_seed<S>/best.pt
nohp-out/contam/<backbone>_<dataset>_rho<RR>.out
```

`<RR>` 用两位整数（`00`,`01`,`05`,`10`,`20`），保证字典序 = 数值序，`ls` 出来天然有序。

---

## 5. Stage 0–2：污染数据生成

### 5.1 ρ 的定义

$$\rho=\frac{\text{被污染的原始时间戳数}}{\text{训练集原始时间戳总数}}$$

定义在 **raw timestamp level**（不是窗口级）——因为一个异常点会进入多个重叠窗口，按窗口统计会重复计数、无法解释。取值范围：

| ρ | 角色 |
|---|---|
| 0% | clean 参照（**必须复现论文主表 F1/AUPRC**，见 §7.3） |
| 1% | 轻度 |
| 5% | 较低 |
| 10% | 中等 |
| 20% | 压力测试 |

训练集规模（实测，`/data/sdc/yrlbkp/anomaly/`）：

| 数据集 | train 时间戳 | ρ=20% 的污染点数 | test 时间戳 |
|---|---|---|---|
| SWaT | 495,000 | 99,000 | 449,919 |
| WADI | 20,000 | 4,000 | 17,281 |

### 5.2 长度口径 —— r1 的混淆必须在这里切断

「污染 20%」有两种实现，**结论完全不同**：

| 方案 | 训练集长度 | ρ=20% 时正常样本量 | 混淆 |
|---|---|---|---|
| **(a) 替换**（固定总长 N） | N | **0.8N** | 性能下降可能是"正常数据少了"，而非"被污染了" |
| **(b) 增补**（clean ∪ anomaly） | N + 0.2N | N | 训练样本变多，与 (a) 不可比 |

**r2 采用 (a) 替换**（保持训练规模恒定，符合"污染"的字面含义），**并强制增加一个对照**：

```
clean-L80   =  把 clean 训练集确定性截断到 0.8N（同一时刻区间，无污染），ρ 记为 0%-L80
```

于是退化可以被三方分解：

$$\underbrace{P(0) - P(\rho)}_{\text{总退化}} = \underbrace{\big[P(0)-P(\text{0\%-L80})\big]}_{\text{数据量效应}} + \underbrace{\big[P(\text{0\%-L80})-P(\rho)\big]}_{\text{污染效应}}$$

**论文只需要在 ρ=20% 这一档报告这个分解**（其余档位不必）。代价是每 dataset 多 3 次训练，但没有它，"污染影响性能"这句话就立不住——这是 r1 最大的方法论缺口。

> 截断方式：取前 0.8N 个时间戳（确定性），避免再引入一个随机源。

### 5.3 异常注入形态

**连续异常段**，不是随机独立替换单点（r1 的判断正确，且旧的 `add_noise` 正是后者）：

- 每段 $[t_s,\ t_s+L]$，$L$ 从固定的段长分布抽取；
- 段内随机选一部分**连续传感器通道**施加扰动；
- 段与段不重叠（生成时去重，写进 manifest）。

**Nested 集合**（保证 ρ 增加时"只有数量变了"，而不是"每档重抽一套更难的数据"）：

$$\mathcal{A}_{1\%}\subset\mathcal{A}_{5\%}\subset\mathcal{A}_{10\%}\subset\mathcal{A}_{20\%}$$

实现：先一次性**生成 20% 这么多段**并排序；1%/5%/10% 分别取前若干段。这样嵌套是构造性成立的，且生成后可用断言验证（§5.5）。同时固定：异常类型分布、强度分布、段长分布、受影响通道分布。

**段的几何参数**（r1 未给值的自由参数，直接决定 §5.6 的 ρ_win，必须写死）：

| 参数 | 默认 | 说明 |
|---|---|---|
| 段数 $n_{seg}(\rho)$ | $\max\left(5,\ \mathrm{round}\left(\rho N/\bar L_{typ}\right)\right)$，$\bar L_{typ}=300$ s | **下限 5 是为了让"段位置随机"这件事在 3 个 seed 上有意义**（见下） |
| 实际段长 | $({\rho N})/{n_{seg}(\rho)}$ 为均值，逐段从 {0.33, 1, 3}×均值 按 5:3:2 抽 | 先定段数再摊长度，保证 ρ 精确 |
| 每段受影响通道数 | 从 {1, 2, 3} 按 5:3:2 抽，再从连续通道集合中均匀抽该数量的**连续**下标 | 避免全通道污染把样本变得不可辨识 |
| 异常类型 | 三类均匀（1/3 各） | 与下表对应 |

> ⚠️ **WADI 的 $\rho=1\%$ 档需要特殊处理**：WADI 训练集只有 20,000 个时间戳，$1\%=200$ s，若沿用 $\bar L=300$ s 则**不足一段**，该档位在物理上不可表示。用上面的 $\max(5,\cdot)$ 规则，WADI 各档的 $\bar L$ 变成 40/200/200/308 s（ρ=1/5/10/20%），段数恒为 5/5/5/13。
> 代价是**短段会放大窗口级曝光**：ρ=1%、$\bar L=40$ s、W=70 时 $\rho_{win}\approx0.01\times(1+69/40)=2.7\%$，是标称值的 2.7 倍。这正是 §5.6 要求**实测** ρ_win 而非套公式的原因；WADI 的 ρ=1% 档在论文中必须与 ρ_win 同时给出，否则会被读成"1% 污染"。

参数一经确定即冻结并写进 manifest；附录做一组"段长减半"的敏感性（§13-R8）。

**三类异常**（只作用在连续值传感器通道上；`α` 为强度，`σ_j`/`μ_j` 为该通道在 clean 训练集上的 std/mean）：

| 类型 | 公式 | 模拟 |
|---|---|---|
| Level shift | $x'_{t,j}=x_{t,j}+\alpha\sigma_j$ | 传感器偏置、系统性漂移 |
| Amplitude scaling | $x'_{t,j}=\mu_j+\alpha(x_{t,j}-\mu_j)$ | 动态范围被压缩/放大 |
| Short-duration perturbation | $x'_{t,j}=x_{t,j}+\delta_{t,j}$ | 瞬时冲击、局部异常 |

**binary/discrete 执行器变量不注入**（`x + ασ` 会产生物理上不存在的状态）。若需扩展，另设 binary flip / 状态驻留实验，**不进主实验**。

### 5.4 强度 α 的标定（Stage 0）

r1 只说"强度保持不变"，**但没给值，也没说怎么定**——这是审稿人一定会问的自由参数。r2 用**可检出来标定**：

1. 取 **clean-trained 的 PESR**（可直接复用 `results/ablation/mamba_<ds>_seed42.json` 对应的 ckpt）；
2. 用它在 clean 数据上的 bf 阈值 $t_0$（`main.py:250` 的 `bf_search` 产出）；
3. 调整 α，使**注入段的 ≥90% 时间戳被 clean 模型判为异常**；
4. 记下 $\alpha^\*$（每 dataset 一个），写入 `calib_<ds>.json`。

**理由**：污染必须与真实异常"同等可检出"，否则要么 α 太小（等于什么都没加，曲线全平），要么 α 太大（污染变成了数据集破坏，与"真实异常混入正常训练"的动机不符）。

**敏感性**：主实验用 $\alpha^\*$；另在 $\alpha\in\{0.5\alpha^\*,2\alpha^\*\}$ 上各跑一遍 **ρ=20% 单点**作为附录，证明结论不依赖 α 的取值。

### 5.5 manifest 字段与生成后断言

```jsonc
{
  "dataset": "SWaT", "rho": 0.10, "seed": 7,
  "n_timestamps_total": 495000, "n_contaminated": 49500,
  "rho_win": 0.118,                      // 窗口级曝光率，见 §5.6
  "segments": [{"t_start": 1234, "L": 500, "type": "level_shift",
                "channels": [3, 7, 11], "alpha": 1.2}],
  "nested_key": "s7_swat_v1",            // 同一 key 下的各 ρ 共享段序
  "alpha_star": 1.2, "source": "calib_SWaT.json",
  "clean_sha256": "...", "out_sha256": "...",
  "gen_script": "bench/gen_contam.py", "gen_sha256": "..."
}
```

**必须通过的断言**（生成脚本内，失败即退出）：

1. 精确性：`n_contaminated == round(rho * n_total)`（四舍五入误差记为 0）；
2. 嵌套：$\mathcal{A}_{1\%}\subseteq\mathcal{A}_{5\%}\subseteq\mathcal{A}_{10\%}\subseteq\mathcal{A}_{20\%}$（按 `t_start` 逐段比对）；
3. 非污染区**逐元素相等**于 clean 数组（只有被注入的位置允许变化）；
4. 执行器通道（依据 §5.3 的 continuous/discrete 划分表）在污染前后完全一致。

### 5.6 窗口级污染率 $\rho_{win}$ 必须报告

ρ 定义在时间戳级（§5.1），但**模型训练在窗口上**。一段长 L 的污染段会覆盖约 $L+W-1$ 个窗口，所以

$$\rho_{win}\approx\rho\cdot\left(1+\frac{W-1}{\bar{L}}\right)$$

短段会显著放大曝光：$\bar{L}=W$ 时 $\rho_{win}\approx 2\rho$；$\bar{L}\gg W$ 时 $\rho_{win}\to\rho$。

三个后果，都要在文中交代：

1. **不同数据集的 ρ_win 不可直接横比**（W 不同：SWaT 35、其余 70），叙述时以 ρ 为准、ρ_win 附列；
2. 段长分布是**隐藏的自由参数**——它通过 ρ_win 影响模型实际看到多少污染。既然 §5.3 固定了段长分布，ρ_win 就是 ρ 的单调函数，结论不受影响，但**数值必须披露**；
3. `ρ_win` 由生成脚本直接统计（数被污染窗口数 / 总窗口数），不靠公式估算。

---

## 6. 预处理与缩放控制

**现状**（已核对代码）：`get_data()` 内部调用 `preprocess()`，而 `preprocess = MinMaxScaler().fit_transform(df)`（`utils.py:75-96`）；`add_noise()` 在 `get_data()` **之后**（`main.py:168` → `173`）。即：**scaler 永远拟合在 clean 原始训练集上，污染发生在缩放之后。**

这带来一个必须显式选择的岔路：

| 方案 | 注入空间 | 改动 | §7「scaler 只拟合一次」 |
|---|---|---|---|
| **(A) 缩放后注入** | MinMax 输出空间 $[0,1]$ | 无（沿用现有顺序） | **自动满足**（scaler 在注入前拟合完） |
| **(B) 原始空间注入** | 真实物理单位 | 需给 `preprocess` 加 `fit_data` 参数，用 clean 数组拟合、再 transform 污染数组 | 需实现 |

**r2 选 (B)**，理由：`ασ_j` 只有在原始物理单位下才可解释（§5.3 的 σ_j、μ_j 都是物理量）；方案 (A) 下 MinMax 是逐通道仿射，等价于给每个通道一个不同的有效 α，"同一强度"失去意义。当前 `--noise` 之所以物理无意义（±5 的尖峰落在 [0,1] 空间），根源就是 (A)。

**对应代码改动**（唯一需要动 `utils.py` 的地方）：

```python
def preprocess(df, fit_data=None):
    """fit_data 非 None 时，用 fit_data 拟合 scaler 再 transform df（两者须同维度）。"""
    scaler = MinMaxScaler().fit(fit_data if fit_data is not None else df)
    return scaler.transform(df).astype(np.float32)
```

**测试集/验证集的处理不变**：仍然各自拟合（与论文主表口径一致，保证 ρ=0 可复现主表）。

---

## 7. Stage 3：训练与评测

### 7.1 命令模板

```bash
# 复用 script/addition/ablation.sh 的调度骨架（3 卡并行、已有结果跳过、可重入）
for data in SWaT WADI; do
  for bb in mamba nompa transformer; do
    for rho in 00 01 05 10 20; do
      for s in 7 17 27; do
        CUDA_VISIBLE_DEVICES=$gpu python main.py \
          --dataset $data --backbone $bb \
          --contam-file artifacts/contam/${data}_rho${rho}_seed${s}.npy \
          --scaler-from  artifacts/contam/${data}_rho00_seed${s}.npy \
          --seed $s --name contam/${bb} --trainmode train-test
      done
    done
  done
done
# length-matched 对照：多跑 3×(data,bb,seed) 的 clean-L80
```

### 7.2 main.py 需要的接口改动

| 改动 | 内容 | 必要性 |
|---|---|---|
| `--contam-file` | 若给出，则跳过 `add_noise`，直接 `np.load` 污染数组替换 `train_data` | 核心 |
| `--scaler-from` | 传给 `preprocess(fit_data=...)`，用 clean 数组拟合 scaler（§6 方案 B） | 核心 |
| `--backbone nompa` | 注册 §3.2 的变体 | 核心 |
| `--noise` | **保留不动**（旧实验仍要能跑通），但污染实验不使用它 | 兼容 |

`config.noise`、`seed`、`dataset`、`backbone`、`rho` 全部写进落盘 JSON（现有 schema 已含前三项，加 `contam` 字段）。

### 7.3 既成事实：阈值选择天然不受污染 ✅

`main.py:175-182`：验证集是**从 test 切出来的最后 35%**，不是从训练集切。

$$\text{train} \to \text{加污染},\qquad \text{val}\subset\text{test} \to \text{永不接触污染}$$

所以 `bf_search` 选出的阈值**不受 ρ 影响**，F1 的变化里不含"阈值被污染带偏"这一项。这是本设计的一个优点，**必须在论文里写出来**（否则审稿人会怀疑阈值选择是混淆源）。

**同时必须做的一个校验**：ρ=0% 且 clean-L80 两条腿的 `val` 必须与主表完全一致（同一段数据），否则阈值不可比。

**模型选择口径也一致**：`main.py:227-250` 按 val F1 选 best epoch。因为 val 不接触污染，**三个变体、五个 ρ 档的 best-epoch 判据是同一个**，污染的效应只通过训练数据进入模型，不通过模型选择。这一点同样要在论文里写明（否则会被怀疑"污染改变了早停点，从而改变了模型复杂度"）。

### 7.4 ⚠️ 已知隐患：val 与 test 重叠

同一段代码还有一处**既存缺陷**：`test_data = test_data[:-n]` 之后紧接 `val_data = test_data[-n:]`，取的是**已被截断数组**的末尾，因此

$$\text{val} = [0.30L,\ 0.65L),\qquad \text{test} = [0,\ 0.65L)\ \Rightarrow\ \text{val}\subsetneq\text{test}$$

即验证集是测试集的子集，用于选阈值的样本也被计入测试指标 → F1 **系统性偏乐观**，且偏移量是否随 ρ 变化未知。

处置（三选一，**推荐第 3 个**）：
1. 不动，仅在附录披露（最省事，但 F1 结果会被质疑）；
2. 修成 `val_data = test_data[n:]` 之类的互斥切分（干净，但会让本项目**所有历史 F1 都不可比**）；
3. **主指标用 AUPRC（阈值无关，不受此缺陷影响），F1 降级为辅助并在此处明确披露**——本 PRD 的 ρ=0% 行仍报 F1，但只用于"与主表一致性核对"，不用于跨变体比较。

---

## 8. 指标

### 8.1 检测层

| 指标 | 角色 | 说明 |
|---|---|---|
| **AUPRC** | **主** | 阈值无关、对不平衡稳健、绕开 §7.4 的重叠缺陷与 `bf_search` 的 oracle 质疑 |
| F1 | 次 | 走 PRD3.1 的 non-oracle 阈值协议；仅用于与主表口径对齐，不做跨变体主结论 |

### 8.2 重建层（test 侧）

$$E_N(\rho)=\tfrac{1}{|\mathcal D_N|}\sum_{x\in\mathcal D_N}e(x),\quad E_A(\rho)=\tfrac{1}{|\mathcal D_A|}\sum_{x\in\mathcal D_A}e(x),\quad Gap=E_A-E_N$$

$Gap\downarrow$ 是最直接的信号（重构式检测依赖正常/异常之间足够大的重构差）。

### 8.3 重建层（train-injected 侧）—— r1 的循环论证修补

r1 把 $E_A$ 定义在 **test 集**上，然后用"$E_A\downarrow$"来论证"模型学会了重建异常"。**但 test 集里的真实异常从未进入训练**，污染不可能直接教模型重建它们——这个假设在 r1 的指标下**原理上无法验证**。

r2 增加一个量：

$$E_A^{inj}(\rho)=\tfrac{1}{|\mathcal A_\rho|}\sum_{t\in\mathcal A_\rho}e(x_t)\quad(\text{被注入的训练段本身，不做留出})$$

并明确两者的分工：

| 量 | 测什么 | 支撑的结论 |
|---|---|---|
| $E_A^{inj}$ | 模型对**它见过**的异常的拟合程度 | "污染让模型把异常当正常来学"（直接证据） |
| $E_A$ | 对**未见**真实异常的泛化 | "污染损害了对真实异常的判别"（间接、更接近实际危害） |
| $E_N$ | clean 正常模式是否被破坏 | "正常模式学习受损" |
| $Gap$ | 端到端判别余量 | 与 AUPRC 最相关的量 |

### 8.4 跨变体归一化

不同变体的表示空间不同，**绝对**重建误差不可跨变体比。故对每个变体按自身 clean 状态归一化：

$$R_N(\rho)=\frac{E_N(\rho)}{E_N(0)},\qquad R_A(\rho)=\frac{E_A(\rho)}{E_A(0)},\qquad R_N(0)=R_A(0)=1$$

**统计顺序必须写死**：先在**每个 seed 内部**归一化，再对 seed 求 mean±std。若先求均值再相除，误差棒会丢失（r1 未规定顺序）。

### 8.5 可选的分离度指标

$$Sep(\rho)=\frac{\mu_A-\mu_N}{\sqrt{(\sigma_A^2+\sigma_N^2)/2}}$$

与 `Gap` 冗余。**主文只报 `Gap`**；`Sep` 作为附录的稳健性校验（若两者结论一致，说明结论不依赖具体度量选择）。

---

## 9. 分析与判据

### 9.1 图 (a) 必须画归一化退化，不能画绝对 AUPRC

三个变体在 ρ=0 时 AUPRC **本来就不同**（那是 PRD1 消融的结论）。直接画绝对 AUPRC 会把"起点低"误读成"退化快"。故：

$$\Delta\mathrm{AUPRC}(\rho)=\mathrm{AUPRC}(0)-\mathrm{AUPRC}(\rho)\quad\text{或}\quad \frac{\Delta\mathrm{AUPRC}}{\mathrm{AUPRC}(0)}$$

**主图用归一化退化；绝对曲线放附录。**（r1 的 §11.4 已经为重建误差写对了这条原则，却在 §13(a) 对 AUPRC 违反了它——r2 统一。）

### 9.2 显著性检验

同一 seed 下三个变体用**同一份**污染数据，所以**可配对**：

- 每 seed 得到 $d_s=\Delta\mathrm{AUPRC}^{\text{PESR}}_s-\Delta\mathrm{AUPRC}^{\text{w/o MPA}}_s$；
- 3 seeds 上做单侧配对 t 检验（或 Wilcoxon signed-rank）；
- 报告 $d$ 的 mean±std 与 p 值。

仅在 mean±std 上比较、不做检验，会招致"2% 的差距在噪声内"的质疑。

### 9.3 方差来源分解（r1 的 seed 设计有混淆）

r1 的 seed 同时扰动**污染位置**与**模型初始化**，所以 $\sigma$ 里混了两个来源，无法判断退化曲线的抖动是"数据噪声"还是"训练噪声"。r2 用**嵌套 seed**：

$$\text{数据 seed}\ s_d\in\{7,17,27\}\ \times\ \text{初始化 seed}\ s_i\in\{1,2,3\}$$

- **主曲线**：每个 $s_d$ 固定对应的 $s_i$（3 条，共 3 次训练的代价与 r1 相同）；
- **方差分解**（只做 PESR × ρ=0% 和 ρ=20% 两档）：在 $s_d{=}7$ 下扫 $s_i{=}1,2,3$，报告组内 σ（初始化引起）与组间 σ（污染位置引起）。

这样"曲线抖动主要来自初始化而非污染强度"这类风险就能被直接排除，代价只有 4 次额外训练。

### 9.4 判据（不要求严格单调）

允许 $P(5\%)<P(10\%)$ 这类小波动。判据按优先级：

1. 总体趋势（ρ 高 → 性能低）；
2. mean±std 与 §9.2 的检验；
3. ρ=20% 相对 clean 的**归一化**退化幅度；
4. **不同变体的退化斜率**（这是 Q3 的核心，比绝对水平更重要）。

### 9.5 核心量化量

$$\Delta P_{20}=\frac{P(0)-P(20)}{P(0)}$$

对每个变体报一个数，作为"20% 污染下的相对性能损失"。Q3 的判据即 $\Delta P_{20}^{\text{PESR}}<\Delta P_{20}^{\text{w/o MPA/MSSA}}$。

---

## 10. 图表规格

正文放一个 3-panel figure（排版细节见 `docs/PRD4_FigX_FigY_绘图规格.md`）。三条曲线 = `w/o MPA` / `w/o MSSA` / `PESR`。

| Panel | 横轴 | 纵轴 | 目的 |
|---|---|---|---|
| **(a) 检测退化** | ρ | **归一化** $\Delta\mathrm{AUPRC}$ | 谁的退化斜率更小 |
| **(b) 重建行为** | ρ | $R_N,\ R_A$ | 正常侧恶化？异常侧变易被重建？ |
| **(c) 判别余量** | ρ | $Gap$（归一化后） | 正常/异常的分数分离是否被压缩 |

**每个 panel 都必须带 error bar**（3 个数据 seed）。只画均值线不画误差棒，会被判"看不出显著性"。

**(a) 的四个关键数字**（正文用表格给出，与图互补）：每变体在 ρ=0% 与 ρ=20% 的绝对 AUPRC、$\Delta P_{20}$、以及 §5.2 的 `clean-L80` 对照值。

---

## 11. 假设与证伪条件

| # | 假设 | 若成立 → 支撑 | **什么结果会推翻它** |
|---|---|---|---|
| **H1** | $\rho\uparrow\Rightarrow \mathrm{AUPRC}\downarrow$ | R1：污染确实损害检测 | ρ=20% 时 AUPRC 与 ρ=0% 的差 < 2×std；或 `clean-L80` 的退化 ≥ 污染退化（说明是数据量效应） |
| **H2** | $\rho\uparrow\Rightarrow Gap\downarrow$（或 $Sep\downarrow$） | R1：判别余量被压缩 | $Gap$ 无单调趋势；或 $E_A$ 反而上升（污染让模型对异常更敏感，与预期机制相反） |
| **H3** | $\Delta P_{20}^{\text{PESR}}<\Delta P_{20}^{\text{w/o MPA/MSSA}}$ | R2：MPA/MSSA 提升污染鲁棒性 | 不等式反向；或配对检验 p>0.05 |
| **H4** | PESR 的 $Sep$ 退化更小 | R2：分离更稳定 | 同 H3 |

**H1 是 H3 的前提**：若 H1 被推翻（污染根本不伤性能），则整个 motivation 叙事需要重写——**这必须在实验设计阶段就作为"可能结局"接受，而不是事后补救**。r1 通篇没有给出任何证伪条件，等于预设了结论。

**可以支持的 claim**（H3+H4 同时成立时）：

> As the training contamination ratio increases, PESR exhibits a smaller degradation in detection performance and maintains a more stable normal–anomaly separation than its ablated variants, indicating that multi-view and multi-scale representations improve the robustness of reconstruction learning against anomalous contamination.

**不能支持的 claim**：

> ~~PESR removes/filters anomalies.~~ 实验只能说明表示更稳定，不能说明模块在"剔除"异常。
> ~~PESR eliminates the influence of anomalous observations.~~ 只能说明退化更小。

---

## 12. 计算预算

**估算方法**：用 `docs/推理成本统计结果_v2.md` 的实测单窗口推理延时 × 训练窗口数 × 3（前向+反向≈3×前向）× 10 epoch。**属数量级估算，开工前必须实测一个 epoch 校准**（改成 `range(1)` 跑一次计时）。

| 数据集 | 训练窗口 | 单窗口 ms | 估算 /run | runs | 小计 |
|---|---|---|---|---|---|
| SWaT | ~495k | ~1.6（b=64 实测 1.640） | ~6.8 h | 45 + 9(L80) + 4(方差分解) = 58 | **~395 GPU·h** |
| WADI | ~20k | ~3.8（b=32 实测 3.769；b=24 未测） | ~0.63 h | 58 | **~37 GPU·h** |

**合计约 430 GPU·h ≈ 8 卡 54 小时**。SWaT 占 91%，是唯一需要优化的地方。可选削减（**须在 PRD 里选一个，不能默认全量**）：

| 方案 | 节省 | 代价 |
|---|---|---|
| SWaT 训练集下采样到 100k 窗口 | −77% | 需证明下采样不改变 ρ=0 的性能（跑一次校验） |
| ρ 网格减为 {0, 10, 20}% | −40% | 曲线只有 3 点，趋势的说服力下降 |
| seed 减为 2 | −33% | 配对检验的统计力不足（n=2 无法做检验） |
| **只做 WADI 主实验 + SWaT 单点校验** | −85% | 与论文 IIoT 叙事冲突（SWaT 是核心数据集） |

> ⚠️ `--max_train_size` **当前不可用**：`utils.py:37` 把它乘上一个硬编码常数 `58317`（MSL 的训练长度），对其他数据集语义错误。若选下采样方案，需先修这行。

---

## 13. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| R1 | **H1 不成立**：污染不伤性能（可能因为合成异常太"假"，模型轻易忽略） | §5.4 的 α 标定保证异常可检出；若仍无退化，改为报告"PESR 对污染不敏感"这一正面结论，并检查 α 敏感性（0.5α*/2α*） |
| R2 | **数据量混淆**（§5.2） | `clean-L80` 对照；ρ=20% 档必报分解 |
| R3 | **val∩test 重叠**（§7.4） | AUPRC 为主指标；F1 明确披露 |
| R4 | **SWaT 训练集并非严格 clean** | 先测：clean-trained 模型在 SWaT train 上的平均分数分布。若 train 侧已有高分点，说明训练集含真实异常，需在文中改为"additional synthetic contamination on top of a not-provably-clean training set" |
| R5 | **`simpip` 已与当前模型脱节**（§3.2） | 实现前跑 SMAP ρ=0 冒烟测试；不达标则新建变体文件 |
| R6 | **`loss_c` 的归属**（§2.1） | 正文显式声明 w/o MPA 同时去掉一致性正则；不试图单独消融它 |
| R7 | **ρ_win ≠ ρ** 导致跨数据集误比（§5.6） | 叙述以 ρ 为准、ρ_win 附列；段长分布固定并在 manifest 中记录 |
| R8 | **段长/通道数分布是隐藏自由参数** | 全部固定并写进 manifest；附录报告在 2 组不同段长分布下 ρ=20% 的结果 |
| R9 | 算力超预算 | §12 的削减方案**必须在开工前选定** |
| R10 | **WADI 的 ρ=1% 档短段放大曝光**（$\rho_{win}=2.7\%$，§5.3） | 该档必须与 ρ_win 同时报；或直接将 WADI 的 ρ 网格改为 {0, 5, 10, 20}%（推荐，避免"1%"名不副实） |
| R11 | **WADI 训练集仅 20k**，段数下限 5 时 ρ=1/5/10% 都是 5 段，实际只有段长在变 | 三个 seed 的段位置只有 5 个自由选择 → 方差大。若 §9.2 的配对检验不显著，需增加 seed 或改用更长的训练片段 |

---

## 14. 交付物清单

| 类型 | 路径 | 谁产出 |
|---|---|---|
| α 标定 | `artifacts/contam/calib_<ds>.json` | `bench/gen_contam.py --calib` |
| 污染数据 + manifest | `artifacts/contam/<ds>_rho<RR>_seed<S>.{npy,json}` | `bench/gen_contam.py` |
| 生成器校验报告 | `artifacts/contam/verify_<ds>.md`（含 §5.5 四项断言结果） | `bench/verify_contam.py` |
| 单次训练结果 | `results/contam/<bb>_<ds>_rho<RR>_seed<S>.json` | `main.py`（现有 schema + `contam` 字段） |
| 聚合表 | `results/contam/summary.{md,csv}` | `bench/summarize_contam.py` |
| 主图 | `figs/contam_<ds>.pdf`（3 panel + error bar） | `bench/plot_contam.py` |
| 调度脚本 | `script/contam/<ds>.sh`（仿 `script/addition/ablation.sh`） | 手写 |

**代码改动汇总**：`utils.py` 的 `preprocess` 加 `fit_data` 参数；`main.py` 加 `--contam-file` / `--scaler-from` / `BACKBONE_REGISTRY['nompa']`；新增 `model/Model_nompa.py`（由 `script/addition/make_ablation_models.py` 生成，禁止手改）；新增 4 个 `bench/*.py` + 1 个 `script/contam/*.sh`。

---

## 15. 与现有 robustness 实验的关系

`script/robust/{SWaT,WADI,msl,SMAP}.sh` 是现役实验，用 `--noise 0.5`（**单点 ρ=50%**），`add_noise`（`main.py:88-112`）做的是**随机独立时间戳替换**，噪声为 uniform(±2) / spike(±5) / random-walk 三选一。

**它不能承担 R1/R2**，原因有二（r1 只说了第一条）：

1. **形态不符**：随机独立替换不具时间结构，测的是"噪声鲁棒性"，不是"异常污染下的正常模式学习"；
2. **物理无意义**：注入发生在 MinMax 缩放**之后**（§6），±5 的尖峰落在 $[0,1]$ 空间，不对应任何真实传感器读数。

**处置**：保留 `script/robust/*.sh` 作为**附录的噪声鲁棒性实验**（它是独立的、可报的结果），`--noise` 代码路径不动。本 PRD 的污染实验是**新增**，不替换它。论文中两者分开表述：

| 实验 | 测什么 | 对应审稿点 |
|---|---|---|
| `script/robust/*`（ρ=0.5 随机替换） | 输入噪声鲁棒性 | 附录 |
| 本 PRD（ρ 曲线，结构化污染段） | 训练期异常污染的动机验证 | R1 / R2 主文 |

---

## 16. 最终配置一览

| 项 | 设定 |
|---|---|
| 数据集 | SWaT、WADI（主）；α 敏感性实验另算 |
| ρ | SWaT: 0/1/5/10/20%；**WADI: 0/5/10/20%**（1% 档不可表示，§5.3-R10）|
| runs | 3 backbone × 5 ρ × 3 seed + 9 (L80) + 4 (方差分解) = **58 / dataset**（WADI 因 4 档为 49）|
| 段几何 | $n_{seg}=\max(5,\mathrm{round}(\rho N/300))$，长按均值 {0.33,1,3}× 抽；通道数 {1,2,3} |
| 注入层级 | raw timestamp level（原始物理空间注入，scaler 用 clean 拟合） |
| 注入形态 | 连续异常段，段内随机连续通道；nested 集合 |
| 异常类型 | level shift / amplitude scaling / short-duration perturbation |
| 强度 α | $\alpha^\*$ 由 clean 模型 ≥90% 检出来标定；敏感性别测 0.5α*/2α* |
| 变体 | PESR / w/o MPA（`nompa`）/ w/o MSSA（`transformer`） |
| 一致性正则 | `loss_c` 随 MPA 一并移除，正文显式披露 |
| seed | 数据 seed {7,17,27} × 初始化 seed {1,2,3}（嵌套，§9.3） |
| 主指标 | AUPRC（阈值无关） |
| 次指标 | F1（走 PRD3.1 非 oracle 协议；披露 val∩test 重叠） |
| 重建指标 | $E_N$、$E_A$、$E_A^{inj}$、$Gap$、归一化 $R_N/R_A$ |
| 统计 | 每 seed 内归一化 → mean±std → 配对检验 |
| 预算 | ~370 GPU·h（**削减方案须先选定**，§12） |
