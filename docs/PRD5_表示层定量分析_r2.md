# PRD5 · 表示层定量分析（r2）—— 跨重构类算法的表示可分性

> **版本** r2 · 2026-10-10
> **相对 v2 的核心变更**：实验主体从「PESR vs 自己的消融」改为
> 「**PESR vs 传统重构类算法**」，并把「**修表示塌缩**」提为 Phase 0 硬门。
> **上游依赖**：`docs/诊断_表示塌缩_20261010.md`（**必读，否则本文的实验全部无意义**）。
> **保留自 v2 且不变**：术语/tap 定义（§3）、实体口径与泄漏控制（§5）、指标口径（§6）、Gate（§10）。
> **状态**：待实施。不预设任何结果。

---

## 0. r2 相对 v2 的变更清单

| # | 变更 | 理由 |
|---|---|---|
| R1 | **删掉「跨变体消融」作为实验主块** | 改动后主块是「跨算法」；消融降为可选的附录 E5 |
| R2 | **新增 Phase 0：修表示塌缩（硬门）** | 已实测：**所有 ckpt 的 `V_t`/`μ_t` 都是常量**（`诊断_表示塌缩_20261010.md`）。无法度量常量的可分性 |
| R3 | **新增"对照三件套"为必需项** | 今天在 MSL 上学到的教训：维度会伪造可分性（raw 3850 维 AUROC 0.596 → PCA-256 掉回 0.505）。跨架构比较必须带维度对照 |
| R4 | **聚类降为可视化 + 次要列** | 聚类指标（silhouette/k-means）对尺度与几何敏感，并暗含"正常是单个凸簇"，工业多工况不成立 |
| R5 | **H3 提为核心假设** | 原生分数无阈值 AUPRC ≈ base rate（多数据集实测），"表示 > 分数"是最有论文价值的一条 |
| R6 | 明确「本实验替代消融的哪一半、不替代哪一半」 | 见 §1.2，防止把外部比较当成模块归因 |

---

## 1. 定位

### 1.1 回应哪条审稿意见

**R2（本文的目标）**

> The authors need to effectively address the limitations of conventional reconstruction-based methods
> and demonstrate a meaningful contribution to the anomaly detection literature.

**R1（本文不回答，必须写明）**

> …quantify how the identified issue (i.e., anomalies contaminating the learning of normal patterns)
> actually affects real systems…

本实验**不改变训练污染比例**，因此**不能**回答 R1 的因果问题（同 `PRD5` v2 §1.3、`PRD4text_r2` §3.4）。

### 1.2 它替代消融的哪一半、不替代哪一半

| 论文想说的话 | 跨算法对比（本文） | 消融（v2 主块） |
|---|---|---|
| 「相对传统重构方法，我们的表示更可分 / 范式确实有局限」 | ✅ **就是它** | ✗ |
| 「MPA / MSSA 各自有贡献」 | ✗ **跨模型差异无法归因到具体模块** | ✅ 唯一手段 |
| 「CRR 编码后表示更好/更差」 | 部分（阶段对比，§3） | — |

> ⚠️ **所以：若论文仍主张"MPA/MSSA 有用"，消融不能省，只是从主块降为附录 E5。**
> 若放弃该主张（或改由其他证据支撑），才可以不做。这个取舍要在动笔前定。

### 1.3 主张（收紧后）

> 重构式 MTS-AD 的共性问题是：**重建目标本身不保证学到一个可分的正常模式表示**。
> 本文用**标签无关的 normal-reference kNN** 度量"进入重建之前的表示"的可分性，
> 在**同一数据、同一划分、同一口径**下比较 PESR 与若干传统重构类方法，
> 并同时报告**阈值无关**的原生检测指标（AUPRC/AUROC）。

**不主张**：PESR 在 F1 上更高（既有 F1 已证明无分辨力，见 §6.1）、训练污染被缓解。

---

## 2. Phase 0（硬门）：先让表示不再是常量

**未通过 Phase 0，本文所有实验不得开始。** 依据：`docs/诊断_表示塌缩_20261010.md`。

### 2.1 已确证的事实（复述，细节见诊断文档）

- `eval()` 下 `V_t`/`μ_t` 逐维 sd ~1e-7，取值 O(1~6)，相对变化 ~4e-8 = **float32 最后一位**
- 真实窗 / 全零窗 / 随机窗 → `V_t` 只差最后一位，score 为 `22.75971 / 22.759705 / 22.759708`
- **塌缩点 = `attention_sim`**（`model/Model.py:112` 第一个 `attentionstacking(selfattentionblock1d,4,512)`）
- **未训练不塌缩**（`as_sdmax` 1.09 / 1.04），**已训练塌缩**（8e-7）→ 训练造成的
- **7/7 个 ckpt 全部塌缩**（MSL×2、nomssa、SMAP、SWaT×2、WADI W=140）
- `main.py` 推理不调 `model.eval()`；`Dropout(0.1)` 硬编码 → **train 口径把常量"搅活"**，
  所以论文里的 F1 是 point-adjust 作用在 dropout 噪声上

### 2.2 Phase 0 的产物（三选一）

| 结果 | 后续 |
|---|---|
| **A. 定位到机制并修好** | 得到一个非塌缩的 PESR → 进入 Phase 1 |
| **B. 定位到机制但改动大** | 需先与导师/合作者确认改动是否进入论文方法 |
| **C. 定位不到** | **停止**。在塌缩未解释前，任何表示层数字都不能进论文 |

### 2.3 Phase 0 的最小定位实验（成本 ~15 min）

MSL、10 epoch、seed 42，**每个 step 记录**：

1. `as_sdmax`（`attention_sim` 输出的跨窗 sd）→ 看从第几步开始掉
2. loss 三项分解：`l1`(recon MSE) / `l2`(KL) / `loss_c`(视图一致性余弦)（`Model.py:138-142`）
3. `V_t` / `μ_t` 的 sd

**判读**：

- `as_sdmax` 前几十步就崩 → 指向候选①（`[B,512]` 被 `nn.MultiheadAttention` 当成无 batch 维的
  `(L=B,E=512)`，跨窗口注意力饱和成常量）
- `l1` 先降到 ~0、再带动 `as_sdmax` 崩 → 指向候选②（损失目标是模型自产的 `x_a`，
  且 CRR 压缩比 275/256≈1.07 无瓶颈 → "把 `x_a` 压成常量"是平凡零损失解）

**注意**：候选②与观测有一点不自洽（`Z_p` 在 `x_a` 的更上游却没有塌缩），故必须实测判读，不得先验定因。

### 2.4 Phase 0 的验收判据（Gate P0）

对**每一个**进入 Phase 1 的模型（PESR 与所有 baseline）：

```
喂 8 真实窗 + 8 全零窗 + 8 随机窗（同一 batch）→ 目标 stage 的跨窗 sd_max
    必须 > 该 stage 取值量级的 1e-3
```

低于此值 → 该模型的表示不可用于本实验，标记并**从比较中剔除**（不是"结果不好"，是"没测到东西"）。

> **v2 的 Gate 0 只查了重复前向一致性 —— 常量也满足它。这是 2026-10-10 那轮假阳性的直接原因。**
> 本 r2 把"非恒定"写成**独立且必需**的一条。

---

## 3. 术语与 tap（沿用 v2，不变）

前向链与三个 tap：

| 记号 | 张量 | MSL 维 | 代码位置 |
|---|---|---|---|
| $Z_p$ | MPA 输出 = MSSA 输入 | `[B, C, 512]` → pooling | `attmamba[0]` pre-hook（`nomssa` 退回 `project4`） |
| $V_t$ | **进入 CRR 的表示**（= `pipline` 的 `x_a`） | `[B, 275]` | `projection3` post-hook |
| $\mu_t$ | CRR 编码器均值 | `[B, 256]` | `encoder` post-hook，取 `chunk(2,dim=1)` 前半 |
| （对照） | `attention_ALL` 输出 | `[B, 512]` | 免费附带 |

**跨模型比较的合法位置 = 「进入重建之前的表示」**：
PESR 是 $V_t$；AE/VAE 类是它的瓶颈向量；LSTM-AE 是编码器末状态。三者语义对等、维数可对齐（§4.2）。

三条实现层约束（v2 §3.4，**仍然强制**）：

| # | 约束 | 原因 |
|---|---|---|
| 1 | 提取一律 `model.eval()` | `Dropout(0.1)` 硬编码在 `selfattentionblock1d` |
| 2 | 固定 `batch_size` + `shuffle=False` | `attention_sim` 等把 `[B,512]` 当前无 batch 维的 `(L=B,E=512)`，表示依赖同 batch 的其它窗口。**对所有 baseline 也要固定同一 batch 组成** |
| 3 | 标签口径显式重定（`any-in-window` 主 / `last-point` 次） | `main.py:335` 的 `test_label[-len(score):]` 有 +1 位移 |

---

## 4. 对比对象

### 4.1 PESR

用 Phase 0 修好后的 ckpt；`--backbone mamba`。**必须报告 Phase 0 的验收数**（`as_sdmax`、`vt_sdmax`）。

### 4.2 传统重构类 baseline（按实现成本递增，至少做前 3 个）

| ID | 方法 | 表示位置 | 相对 PESR 的对照含义 | 实现成本 |
|---|---|---|---|---|
| B1 | **AE-Linear**（flatten 窗 → 隐层 → 重构） | 瓶颈向量 | "没有任何时序结构"的下界 | 低 |
| B2 | **VAE-Linear**（与 PESR 的 CRR 同构） | `latent_mu` | **直接对照"没有 MPA/MSSA 的 CRR"** | 低 |
| B3 | **LSTM-AE** | 编码器末隐状态 | "经典时序重构" | 中 |
| B4 | USAD（双阶段对抗 AE） | 瓶颈 | 训练目标不同的重构类 | 中 |
| B5 | TranAD（可选） | 瓶颈 | 注意力式重构 | 中 |

**公平性（必须逐条落实）**：

- 同一 train/test 划分、同一 `preprocess`（scaler 只在 train 上拟合）
- 同一 W、stride、batch 组成、`eval()` 口径
- 同一 bank 规则（§6.2）、同一 k
- **同维对齐**：B1/B2 的瓶颈维设成 **256**（与 $\mu_t$ 同维）；
  $V_t$ 是 275 维，接近但不等 —— 因此**必须**加 §8 的维度对照
- 参数量不作对齐（不补 FLOPs 表，沿用 `ablation-design-principles`）；
  但**参数与容量差异要在 limitation 里写明**，且**不做"哪个架构更好"的强结论**

### 4.3 不做的事

不做外部 SOTA 排行榜（MUTANT/DTAAD/AnomalyTransformer 等各有不同协议）；不把本实验当
leaderboard。若审稿人要求，另做附录。

---

## 5. 数据、实体与标签（沿用 v2，不变）

### 5.1 数据集

| 数据集 | 角色 | 依据 |
|---|---|---|
| **SWaT**（首选） | 单一实体、51 通道、35 个真实攻击段、异常率 12.14% | **实测原始输入窗 AUROC = 0.881** → 数据里确实有可学的异常结构，不是"没得学" |
| **MSL** | 27 实体（stacked 布局） | 既有主数据集；但注意 5 个 σ=0 实体 |
| SMAP | 可选 | 53 实体，同 stacked 布局 |
| WADI | 探索性 | test 只有 1 个异常段，std 大（状态记录 §3.3） |

> **优先 SWaT**：唯一一个已验证"原始空间可分"的数据集，能区分"模型没学到"与"数据没东西"。

### 5.2 多实体口径（MSL/SMAP 必需）

按实体建 bank、按实体评价、macro mean 为主；`σ=0 实体`（MSL: C-2/D-14/M-6/S-2/T-5，信号恒值）
单列，macro 报"含/不含"两版。实测：27 个实体全部含异常，逐实体异常率 **0.60% ~ 50.11%（84×）**，
逐实体测试窗 1026~6030 —— 不分实体的话 pooled 数字不属于任何实体。

### 5.3 数据角色与泄漏控制

train 只建 bank（逐实体）；test 只作 query；**所有标准化参数只从 train bank 拟合**；
两套标签口径都报。**禁止**用 test 标签选 bank / 选 $k$ / 选层 / 选图。

### 5.4 一条假设

**train 全为正常**是假设不是事实 → 对 MSL/SMAP 称 **nominal/unlabeled training reference**。

---

## 6. 指标（预注册）

### 6.1 原生检测指标（无阈值）

$$\text{score-AUPRC},\quad \text{score-AUROC}$$

用各模型**自己的原生异常分数**、同一批窗口、**不做 point-adjust**。
**必须与 prevalence 一起报告**。

> **v2 的 §0 结论仍然成立**：point-adjust F1（0.94~0.98）无分辨力，不能用于任何比较。
> Phase 0 之后需**重新测**各模型的原生 score-AUPRC（旧值是在塌缩模型上得到的）。

### 6.2 表示可分性（主指标）

bank：**逐实体**训练窗等间隔抽 `min(N, 400)` 条（确定性、无 RNG）；
标准化：用该 bank 的逐维 mean/std，丢掉方差≈0 的维度；
分数：$s_k(z)=\frac1k\sum_{z_j\in NN_k}\|z-z_j\|_2$，**$k=10$**（敏感性 5/20），越大越异常。

$$\boxed{\text{repr-AUPRC}(V_t)},\qquad \boxed{\text{repr-AUROC}(V_t)}$$

> ⚠️ **这是表示诊断，不是模型的检测性能**，不得偷换；与 §6.1 并列报告。

### 6.3 辅助量

$D_N, D_A, \mathrm{RSR}=D_A/D_N, \mathrm{Sep}_{score}=(D_A-D_N)/\sqrt{(\sigma_A^2+\sigma_N^2)/2}$
（$\sigma$ 是 **kNN 距离分数**的标准差，不是坐标方差）。**只作辅助**，主判据是 AUPRC/AUROC。

### 6.4 敏感性（一次报完，不许事后挑）

| 维度 | 主口径 | 敏感性 |
|---|---|---|
| $k$ | 10 | 5, 20 |
| 标签 | any-in-window | last-point |
| 提取口径 | `eval()` + 固定 bs | `train()`（**仅作对照，主口径必须是 eval**） |
| 维度 | 各自瓶颈维 | **PCA 到同一维（256）** ← 见 §8 |
| bank | 等间隔 min(N,400) | 全量训练窗 |

---

## 7. 聚类：只作可视化 + 次要列

**主指标不是聚类。** 理由：silhouette / k-means 对尺度与几何敏感，且暗含"正常是单个凸簇"；
工业系统有多个正常工况（v2 §6.4 已述）。

| 用途 | 做法 | 地位 |
|---|---|---|
| 可视化 | PCA-2 / t-SNE-2，三类着色（nominal train / normal test / anomalous test），各方法一列 | **解释用**，不是证据 |
| 次要列 | **白化空间**里的 silhouette（仅在 train bank 上拟合白化） | 次要，须与 AUPRC 同向才可提 |
| 可选 | k-means 的簇纯度 / 与真标签的 ARI | **仅当簇数由预先规则确定**（如按实体或已知工况数），不得扫簇数挑最好 |

> 图上"分得开"不构成证据；不同方法的二维投影**绝对距离不可比**（特征空间不同）。

---

## 8. 对照三件套（**缺一不可**）

今天的教训：MSL 的原始输入在 **3850 维**下 AUROC=0.596，压到 **PCA-256** 掉回 **0.505** ——
**维度会伪造可分性**。跨架构比较尤其容易踩这个坑。

| # | 对照 | 做法 | 排除什么 |
|---|---|---|---|
| C1 | **原始输入窗** | 把展平的 raw 窗当"表示"跑同一套 kNN（全维 + PCA-同维） | "信号本来就在数据里，与模型无关" |
| C2 | **PCA 同维** | 各模型表示 PCA 到同一维（256）后算 AUPRC/AUROC | 维度/距离集中效应 |
| C3 | **植入信号正对照** | 给异常窗的随机一半维度加 `0.5σ/1σ` 偏移 | **指标通路本身有没有分辨力**（已实测：0.5σ → AUROC 1.0000） |

**报告纪律**：C2 与主口径不一致时，以**两者都报**为准，并在正文说明差异来源；
**不得**只报更好看的那个。

---

## 9. 实验分块

### E1（主）跨算法的表示可分性

同一位置（进入重建之前）比较 PESR 与 B1–B5：

| 数据集 | 方法 | repr-AUPRC ↑ | repr-AUROC ↑ | D_N | D_A | Sep ↑ | score-AUPRC | score-AUROC | Phase0 验收 |
|---|---|---|---|---|---|---|---|---|---|
| SWaT | PESR | | | | | | | | `as_sdmax=…` |
| SWaT | AE-Linear | | | | | | | | |
| SWaT | VAE-Linear | | | | | | | | |
| SWaT | LSTM-AE | | | | | | | | |
| MSL | …（同上） | | | | | | | | |

**判读**（预注册）：

- PESR 的 repr-AUPRC 显著高于 baseline → 支持"我们的表示更可分"（要注意 §4.2 的容量限定）
- PESR **低于**某 baseline → **如实报告**。此时应讨论"是不是塌缩没修干净"（回到 Phase 0），
  **不得**用换指标/换实体掩盖
- 全部 ≈ base rate → 见 §8 C1：说明数据/口径层面就没有可分结构，与模型无关

### E2 逐阶段演化（仅 PESR，附带所有 baseline 的"等价位"）

$Z_p \to V_t \to \mu_t$ 的 repr-AUPRC。**不预设单调递增**（CRR 的首要目的是给出重建潜分布，
并非显式优化 kNN 可分性）。baseline 侧给"输入 → 瓶颈 → 重构"的等价位。

### E3 聚类/可视化

§7 的规定：PCA/t-SNE 面板（每方法一列）+ 白化空间 silhouette 次要列。

### E4 原生分数的无阈值检测性能

各方法的 score-AUPRC/AUROC（未 point-adjust）+ 与 repr-AUPRC 的对比（**H3**）。

### E5（可选/附录）消融

`PESR − MPA` / `PESR − MSSA` 在同一口径下重跑，仅当论文仍主张模块贡献时做。见 §1.2。

---

## 10. Gate（全过才铺开）

| Gate | 检查 | 判据 |
|---|---|---|
| **P0** | **非恒定表示**（§2.4） | 目标 stage 跨窗 `sd_max > 1e-3 × 取值量级`；**每个模型都要过** |
| 0 | 重复前向一致性 | `max|Δ| == 0`（注意：常量也满足，故**不能单独用**） |
| 1 | bank/kNN 合理 | bank ≥ k；最近邻时间间隔中位 ≫ 1（防"bank 就是查询窗的拷贝"） |
| 2 | 标签与方向 | 有两类标签；AUPRC 方向正确（越异常分越高）；**不事后反转分数方向**；与 prevalence 同报 |

---

## 11. 执行与成本

### 11.1 阶段

| 阶段 | 内容 | 成本 |
|---|---|---|
| **P0** | 塌缩定位实验（逐 step 记录） | ~15 min |
| **P0'** | 按定位结果修复 + 重训 PESR | 视改动而定（MSL mamba 单 run ≈1 h；SWaT ≈8.5 h，**优先在 MSL 上验证修复**） |
| **P1** | baseline B1–B3 实现 + 训练（小模型，MSL 各 <20 min） | ~1 h |
| **P2** | 全模型提取（`eval()`、固定 bs）+ E1–E4 | ~1 h（MSL）/ ~2 h（SWaT） |
| **P3** | E5 消融（可选） | ~2 h |

**SWaT 的成本警告**：train 494,965 窗（MSL 的 8.5×），`mamba` 一个 run ≈8.5 h。
**因此 Phase 0 的修复验证只在 MSL 上做**，修好后再决定是否铺 SWaT。

### 11.2 复用

- `bench/extract_repr.py` / `bench/analyze_repr.py`（已写，含实体切窗、hook、Gate 记录）
  —— 需补：**非恒定表示检查**（Gate P0）、baseline 的适配器、PCA 同维对照
- `bench/channel_split.py::stacked_blocks`（实体边界，已测）
- `bench/cache_scores.py`（切分/窗口化口径模板，逐行照抄）

---

## 12. 判据（预注册）

| # | 判据 | 结论 |
|---|---|---|
| **D0** | 任一模型未过 Gate P0 | 该模型从比较中**剔除**（不是"结果差"，是"没测到"） |
| **D1** | 某模型 repr-AUPRC 显著 > base rate（$m-2\mathrm{SEM}>$ base） | 该模型的表示确实携带异常信息 |
| **D2** | PESR 的 $m-2\mathrm{SEM}$ 高于全部 baseline | 支持"我们的表示更可分"（附容量限定） |
| **D2'** | PESR **低于**某 baseline | 如实报告 + 回查 Phase 0；**不得**换指标掩盖 |
| **D3** | 全部模型 ≈ base rate | 结合 C1 判"数据/口径问题"还是"模型问题" |
| **D4 (H3)** | 某模型的 `repr-AUPRC` 明显 > 其 `score-AUPRC` | **瓶颈在分数定义而非表示** → 论文强论点 |
| **D5** | 逐阶段非单调 | 写成 trend，**不写成"逐层提升"** |
| **D6** | 跨条目的 Spearman(repr-AUPRC, score-AUPRC) | 仅报告，不作主张 |

---

## 13. 能写 / 不能写

| 观察 | 可以写 | 不能写 |
|---|---|---|
| PESR repr-AUPRC 高于 baseline | 在真实异常上，PESR 的表示比传统重构方法更可分 | 证明了训练污染被消除 |
| 某 baseline 的表示更可分 | 该范式在表示层面有优势；分析 PESR 的哪一环丢了信息 | 全盘否定 PESR（未过 Gate 的模型不可比） |
| 表示可分但原生分数差 | 分数定义是瓶颈（H3） | 模型检测性能好 |
| 全部 ≈ base rate | 该数据/口径下无可分结构 | 为符合叙事挑数据集/挑指标 |
| 修复后 F1 变化很大 | 说明原 F1 是平台性问题 | 不能把新旧 F1 并列成"改进" |

**禁止**：只报"最能证明模块有用"的指标/数据集而不报其余已运行设置；做跨变体排名；
为做结论而扫超参挑口径。

---

## 14. 交付物

| 类型 | 路径 |
|---|---|
| **Phase 0 诊断** | `results/diag/collapse_stepwise.{json,md}`（逐 step 的 `as_sdmax` + loss 分解） |
| 特征 | `results/repr/<model>_<ds>_seed<S>_{test,train}.npz` + `.json`（含 `as_sdmax`、Gate P0 判据） |
| 指标 | `results/repr/summary_r2.{md,csv}`（E1–E4 + 对照三件套 + 敏感性） |
| 图 | `figs/repr/{method_pca,method_tsne,score_dist,stage_auprc}.pdf` |
| 代码 | `bench/extract_repr.py`（补 Gate P0 + baseline 适配器）、`bench/analyze_repr.py`（补 PCA 同维对照）、`bench/baselines.py`（B1–B3）、`script/repr/retrain.sh` |
| 文档 | 本文、`docs/诊断_表示塌缩_20261010.md` |

---

## 15. 与其它文档的关系

| 文档 | 关系 |
|---|---|
| `docs/诊断_表示塌缩_20261010.md` | **Phase 0 的依据**，必读 |
| `docs/PRD5_T0结果_20261010.md` | 第一轮 T0。**§2–§7 的 kNN 数字作废**（在 float32 舍入噪声上算的）；其工程与统计口径仍有效 |
| `docs/PRD5_表示层定量分析.md`（v2） | 被本文取代；§3/§5/§6/§10 的内容原样继承 |
| `docs/archive/PRD5_PESR_Representation_Discriminability_Experiment_PRD.md` | v2 的设计主干（RQ/H、Gate、实体口径、回复信落地）仍有效 |
| `PRD4text_*` / `PRD4_噪声定量实验.md` | 暂停；且后者以 point-adjust F1 为主指标，**若要做须改主指标** |
| `PRD1_消融实验.md` | 替换型消融；降为 E5 |
