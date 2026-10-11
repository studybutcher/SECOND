# PRD5 · PESR 表示空间定量分析（normal–anomaly separability）

> **版本** v2 · 2026-10-10
> **来源**：合并 `PRD5_PESR_Representation_Discriminability_Experiment_PRD.md`（设计与口径，为主干）
> 与 `PRD5_表示层定量分析.md`（v1：detector 诊断、实现约束、成本实测）。两者已归档到 `docs/archive/`。
> **用途**：Major Revision · Motivation / Methodology / Ablation
> **状态**：待实施。本文只规定实验方案，**不预设任何结果**。
> **核心约束**：不构造合成异常、不改变 PESR 训练目标、不把 CRR 换成 AE；
> 真实测试标签**只**用于评价与可视化，不参与任何模型/kNN/超参/选图决策。

---

## 0. 前置实测：现有检测指标已经失去分辨力（必读）

在选定指标口径之前，先用**已训练好的检查点**测了一次"现有指标到底有没有分辨力"。结果：

| 数据集 | ckpt | 口径 | 窗口级 AUROC | 窗口级 AUPRC | F1(未调整) | F1(point-adjust) |
|---|---|---|---|---|---|---|
| MSL | `checkpoint/casestudy/MSL/best.pt` | train | **0.4975** | 0.1380 | **0.0297** | 0.9408 |
| MSL | `checkpoint/contam/mamba/MSL/best.pt` | eval | **0.5105** | 0.1422 | **0.0149** | 0.9254 |
| MSL | 同上 | train | **0.4995** | 0.1381 | **0.0359** | 0.9367 |
| MSL | `checkpoint/contam/nomssa/MSL/best.pt` | eval / train | 0.4992 / 0.4955 | 0.1380 / 0.1385 | 0.0411 / 0.0271 | 0.9167 / 0.9447 |
| SMAP | `checkpoint/casestudy补充/SMAP/best.pt` | train | **0.5003** | 0.1522 | **0.0141** | 0.9684 |
| SWaT | `checkpoint/casestudy补充/SWaT/best.pt` | train | **0.4998** | 0.1673 | **0.0101** | 0.9775 |

base rate（= 随机排序时 AUPRC 的参考水平）：MSL 13.88%、SMAP 15.26%、SWaT 16.74%。
随机分数实测 AUPRC = 0.1387，与观测一致。

**读法**：异常分数 `score = |Σ(x_a − recon_mu)|`（`model/Model.py:178`）在**窗口级排序上等价于随机**。
0.94 / 0.97 / 0.98 全部来自 `adjust_predicts`（`eval_method.py:59-73`）的 point-adjust：
只要一个真异常段里有**任意一个**窗口越阈，整段就被判为命中。

### 0.1 已排除的三种替代解释（全部实测）

| 质疑 | 检验 | 结果 |
|---|---|---|
| 「标签错位了」 | 窗口标签相对窗口末端做 −3…+1 共 5 个位移扫描 | AUROC ∈ [0.4954, 0.4978]，**没有位移能救回来** |
| 「异常段确实有响应，只是稀疏」 | 逐段取段内最大分数的全局分位，与"随机分数下 max of L"的零假设 `L/(L+1)` 比 | 观测 0.99317 vs 零假设 0.99174，**只有 59% 的段跑赢零假设**（= 抛硬币） |
| 「是 dropout 噪声压掉了信号」 | 同 ckpt 在 `eval()` 与 train 两口径各测一遍 | 两口径 AUROC 均 ≈ 0.50（eval 口径分数 std 仅 9.2e-6，train 口径 4.29），**结论不变** |

根因与状态记录 §3.2 是同一件事：CRR 瓶颈压缩比 `w_size/latent = 275/256 ≈ 1.07` ≈ 无瓶颈，
重构残差退化成与异常无关的量。

### 0.2 由此产生的三条硬约束

1. **F1（point-adjust）不能用于变体比较。** 它测的是"每个异常段里有没有一个窗口越阈"，
   而阈值又是 oracle 扫出来的 —— AUROC=0.5 的分数也能给出 0.94。
   论文可以继续报 F1（社区口径、便于横向比），但**它不是本实验的证据**。
2. **一切比较走无阈值指标**（AUPRC / AUROC，排序类，与分数绝对量纲无关）。
3. **既有消融表的 F1 差异是噪声**：MSL 单次 std≈0.0075，且今天实测**同一个 ckpt** 换 RNG 状态
   就漂到 0.9254↔0.9568（JSON 里记的是 0.9568）。**本实验不以 F1 为主证据。**

> 这条要写进论文的 limitation：不是"我们的 F1 有问题"，而是
> **point-adjusted F1 在重构类 MTS-AD 上已失去分辨力**（时序 AD 评测里已知的批评）。
> 补上无阈值指标反而让这一节更可信。§14.4 的"不得用 repr-kNN 掩盖原生 detector 的问题"
> 与本条同向，必须同时写。

---

## 1. 背景与实验定位

### 1.1 审稿意见

**R1 — Unclear motivation**

> The authors should better quantify how the identified issue (i.e., anomalies contaminating the learning
> of normal patterns) actually affects real systems, for example, by analyzing how performance degrades as
> anomaly ratios increase, or how much reconstruction quality on normal data is compromised in the presence
> of anomalies, so that the practical significance of the motivation is more clearly justified.

**R2 — Limitations of reconstruction-based methods**

> The authors need to effectively address the limitations of conventional reconstruction-based methods and
> demonstrate a meaningful contribution to the anomaly detection literature.

### 1.2 已有实验为何不作为主方案

此前分别尝试了训练侧随机噪声替换（`PRD4_噪声定量实验.md`）与连续异常段注入（`PRD4text`）。
根据 `PRD4_两轴实验结果分析.md` 与 `PRD4text_状态记录_20261007.md`：

- SMAP、SWaT 的噪声比例曲线没有清晰的性能退化趋势；
- MSL 的消融模型在污染率提高时也未出现稳定劣化；
- 注入异常的强度 α **无法按可检出性标定**（放大到 1000σ，检出率仍饱和在 47.5%）；
- WADI 重复运行差异大（std 是 MSL 的 13 倍）、test 只有 1 个异常段；
- **§0**：最终 reconstruction/anomaly score 的窗口级排序已无分辨力。

**结论：不能为了得到预期下降曲线，继续无依据地提高噪声强度或异常比例。**

因此本实验不再把"训练污染率引起性能下降"设为必要前提，改用原始真实数据，研究模型形成的
**正常模式表示质量**与**正常/异常可分性** —— 也就是 §0 证明"分数层已测不出东西"之后，
唯一还有信息量的地方。

### 1.3 本实验能证明什么、不证明什么

**能证明：**

1. 完整 PESR 的表示是否比 `PESR − MPA` / `PESR − MSSA` 的表示更有利于区分真实正常/异常状态；
2. MPA、MSSA 与 CRR 编码阶段如何改变表示结构；
3. 中间表示是否比最终重构分数携带更多异常信息（H3，§2）。

**不能单独证明：**

- 训练集中异常比例增加会导致表示退化（本文不改变污染比例）；
- 模型是否实际"记住/吸收"了训练异常；
- MPA/MSSA 在因果意义上直接过滤了异常；
- CRR 编码后的可分性一定比其输入更强。

如果修订稿把 motivation 收紧为"**稳定、具有判别力的正常时序表示难以学习**"，本实验可提供直接量化支持；
如果坚持原来的强表述"**训练异常污染正常模式学习**"，则本实验**不能替代污染因果实验**（§14.3）。

---

## 2. 研究问题与假设

### RQ1 · 模块级表示质量

在**同一功能位置**（进入 CRR 的特征）比较 `PESR − MPA`、`PESR − MSSA`、`PESR` 三者的表示。

**H1（待验证）**：完整 PESR 的 representation-based separability 优于至少部分消融变体。

### RQ2 · 完整 PESR 的逐阶段演化

$$X \longrightarrow Z_p\ (\text{MPA 输出}) \longrightarrow V_t\ (\text{CRR 输入}) \longrightarrow \mu_t\ (\text{CRR 编码器均值})$$

**H2（待验证）**：三个阶段产生不同的表示几何结构，可由定量分离指标体现；
**不预设指标必须逐层单调增加**（理由见 §8.3）。

### RQ3 · 表示可分性与检测结果的一致性

**H3（待验证，§0 之后提级为核心假设）**：
**中间表示携带的异常判别信息多于最终重构分数** —— 即 `repr-AUPRC(V_t)` 显著高于
**同一模型自己的** score-AUPRC。

依据 §0：三个数据集的 score-AUROC 都 ≈ 0.50。若 H3 成立，可论证
**瓶颈在分数定义（重构目标坍缩了判别信息），而不在表示** —— 这是本实验最有论文价值的一条。
若 H3 不成立，则按 §13 如实收束，**不得**用 repr-kNN 的 AUPRC 去替代或美化原生 detector 的评价。

---

## 3. 模型、提取点与实现约束

### 3.1 主对比模型（命名纪律）

| ID | 模型 | MPA | MSSA | CRR | 注册名 | MSL 总参数 | 比较目的 |
|---|---|:-:|:-:|:-:|---|---|---|
| M1 | **PESR − MPA** | ✗ | ✓ | ✓ | `nompa` | 199.23 M | 多视角分块对齐的作用 |
| M2 | **PESR − MSSA** | ✓ | ✗ | ✓ | `nomssa` | 53.62 M | 多尺度时序建模的作用 |
| M3 | **PESR** | ✓ | ✓ | ✓ | `mamba` | 201.62 M | 完整模型 |

> ⚠️ **命名**：一律写 `PESR − MPA` / `PESR − MSSA`，**不写 `w/o MPA` / `w/o MSSA`**。
> 后者在 2026-10-07 已定名弃用（见 `docs/PRD4text_定量分析_r2.md` §3.2 与 §13-R12、`ablation-variant-naming`）。
> 与 PRD1 的**替换型**（`Mamba→Transformer` / `Mamba→LSTM`）是不同模型，不可混用同一名称。
>
> **不新增 CRR→AE 的替换实验。** CRR 是本文重建流程的一部分，所有变体均保留。
>
> **容量不对称（一句限定即可）**：M2 砍掉 −73.4%，M1 只砍 −1.2%。
> 本实验的目的是说明"各模块是否有用"，**不是给模块排名**；不补 FLOPs 表、不做容量对齐。
> 若 M2 的表示更差，其中有容量的贡献 —— 结论只陈述相对关系，不写"MPA 比 MSSA 重要"。

### 3.2 逐阶段分析：仅完整 PESR

| Stage | 记号 | 实际张量 | 维度（MSL） | 含义 |
|---|---|---|---|---|
| S1 | $Z_p$ | `attmamba` 的**输入**（= MPA 输出） | `[B, 55, 512]` → pooling | 多视角对齐后的表示 |
| S2 | $V_t$ | `projection3` 的**输出**（= CRR 输入） | `[B, 275]` | 多尺度融合 + 投影后的表示 |
| S3 | $\mu_t$ | `encoder` 输出 `chunk(2,dim=1)` 的 **mu** | `[B, 256]` | 重建潜空间的确定性表示 |

**S3 用编码器均值 $\mu_t$，不用每次随机采样的 $z$**（否则 VAE 采样随机性会人为放大类内离散度）。

### 3.3 tap 的精确定义（不钉死则三变体不可比）

前向链（`model/Model.py:195-249`）：

```
x [B,70,55]
 └─ MPA: patch_size=[5,7] 双视图分块 + 余弦一致性 loss_c
     └─► Z_p  x_combined [B, 55, 512]
          └─ MSSA: 8 × attmambaBlock (n_layer=8, d_model=512)
              └─► [B, 55*512] → project4 → attention_sim → attention_ALL
                   └─► x_all [B, 512]                              ← 附赠做对照
                        └─ projection3 → V_t  x_a [B, 275]  (= C·out_dim = 55×5)
                             └─ CRR: tabular_encoder → μ_t [B,256] → decoder → score
```

> ⚠️ **"MSSA 输出"与"CRR 输入"不是同一个张量** —— 中间隔着
> `project4 → attention_sim → attention_ALL → projection3` 四层。
> 二者必须分别定义、分别记录。**主分析位置 = `projection3` 输出（275 维）**，
> 因为它就是 CRR 真正吃到的东西，且三变体**同维同义** → 是跨变体比较的**唯一合法位置**。
> 原始的 MSSA 输出（`[B,55,512]`）需要先 pooling，语义不同，只作 `V_t` 的对照。

**实现方式：零改动模型文件，用 forward hook。** 四个变体文件由
`script/addition/make_ablation_models.py` 生成且**禁止手工编辑**，故一律走 `register_forward_hook`：
`attmamba[0]`（pre-hook → $Z_p$）、`projection3`（post-hook → $V_t$）、`encoder`（post-hook → $\mu_t$）。
⚠️ `nomssa` 的 `attmamba` 是空 `ModuleList`，挂不上 → 该变体改挂 `project4` 的 pre-hook
（此时"MSSA 输出" ≡ "MPA 输出"，因为中间块不存在）。
提取前**断言这四个模块都存在**，否则退出。

### 3.4 ⚠️ 三条实现层约束（会直接影响数值）

| # | 事实 | 代码位置 | 处置 |
|---|---|---|---|
| 1 | **batch 内跨样本注意力**：`attention_sim` / `attention_ALL` / `encoder` / `decoder` 把 `[B,512]` 交给 `nn.MultiheadAttention`，PyTorch 按**无 batch 维**解释成 `(L=B, E=512)` → **每个窗口的表示取决于同 batch 的其它窗口** | `model/attn/attentionblock.py:99,115` | 提取时**必须固定 `batch_size=60` + `shuffle=False`**，与评测口径一致；再做一次 `bs=1` 敏感性对照（抽 test 5k）量化该耦合 |
| 2 | **Dropout 是开的**：`selfattentionblock1d` 里 `nn.Dropout(0.1)` 硬编码；`main.py` 推理全程不调 `model.eval()` | `attentionblock.py:105,110-111`；`bench/cache_scores.py:91` 已记录 | 提取**一律 `model.eval()`**（否则每次前向的表示都不同，kNN 指标不可复现）。**repeated-forward 一致性是 Gate 0 的必检项** |
| 3 | **标签位移 +1**：评测用 `test_label[-len(score):]`，而窗口 `i` 覆盖点 `[i, i+W-1]` → 标签取的是窗口**之后**那个点 | `main.py:335,363` | 显式重定标签口径（§4.3），两种口径都报 |

### 3.5 ⚠️ ckpt 完整性：现有 ckpt 一律不得复用

状态记录 §4.2 警告过"同变体多 run 并发写同一个 `best.pt`"。**实际已经发生了**：

| JSON | 时间戳 | 记录的 F1 | 写的 ckpt 路径 |
|---|---|---|---|
| `nomssa_MSL_rho00_seed42.json` | 12:23:32 | 0.9624 | `checkpoint/contam/nomssa/MSL/best.pt` |
| `nomssa_MSL_rho20_seed7.json` | 12:36:11 | 0.9490 | 同上 |
| `nomssa_MSL_rho00_seed7.json` | 13:04:28 | 0.9334 | 同上 |

该文件 mtime = **12:55:56**（晚于第二行 run 结束）→ 最后一次写入发生在 seed7 的某次 run 期间。
**已实测**该 ckpt：F1(point-adjust) = **0.9167 / 0.9447**（eval / train），AUROC = 0.4992 / 0.4955。
dropout 随机性让同 ckpt 的 F1 有 ±0.03 漂移，故**无法唯一指认**它属于哪次 run；
但可确定：**它复现不出自己 JSON 里记的 0.9624**。**(ckpt, 数字) 配对已断。**

**决定：T1 全部重训**，`--ckpt-dir` 按 run 唯一化。代价见 §12，收益是 ckpt↔数字一一对应，
且一次拿到 3 个 seed（现有材料只有 1 个可用 seed）。

### 3.6 为什么逐阶段 tap 不能替代消融（本实验为何仍需训练消融模型）

自然会产生一个疑问：既然在完整 PESR 上就能抽出 $Z_p \to V_t \to \mu_t$，是不是不用训消融模型了？
**不能互相替代，因为两者回答的是不同问题：**

| | 逐阶段 tap（实验 B） | 跨变体消融（实验 A） |
|---|---|---|
| 做法 | 在**同一个**已训练模型上取三个中间张量 | 训练**三个不同的**模型，在同一位置取张量 |
| 回答 | 这个模型的表示**经过这些层后**变成什么样 | **去掉某模块重训**后，同一位置的表示是否变差 |
| 性质 | 描述性（descriptive） | 归因性（attributive） |
| 能否证明模块必要 | **不能**。$V_t$ 比 $Z_p$ 更可分，也可能完全是它俩之间那四层
（`project4`/两个 attention）的功劳，而 `nomssa` 里这四层**照样存在** | 能。缺了模块的模型学出的表示更差，才说明该模块有贡献 |

另外两点：

- **消融是"重训"，不是"推理时跳过"。** 在 `mamba` 权重上推理时把 8 层中间块跳过，
  得到的是"一个带着该模块训练过的模型，临时不用它"，与"从头就不带它训练"是两个量，
  也**对不上论文既有的消融表**。所以即使它更便宜，也**不采用**（顶多作为附注）。
- §0 之后，**实验 A 反而更必要**：F1 消融表已证明是噪声，实验 A 于是成了唯一能支撑
  "模块有用"的地方。**逐阶段曲线是解释，跨变体比较才是证据。**

---

## 4. 数据集、实体与标签

### 4.1 首选数据集：**MSL + SWaT**

- **MSL**：27 个实体、真实异常区间，是既有主数据集（W=70, bs=60 的配置族都在 MSL 上）。
- **SWaT**：51 个通道、单一工厂，35 个真实攻击段、异常率 12.14%，与工业场景直接相关。
  （注意 SWaT 的默认 `W=35`，与 MSL 的 70 不同，成本与窗口数都不同。）
- **可选扩展 SMAP**（53 实体，同 stacked 布局，抽取管线与 MSL 共用）。
- **WADI**：探索性补充。此前提示 F1 波动大（std 0.0972）、test 只有 1 个异常段、
  ρ=1% 档物理上不可表示。若使用，必须报告有效测试事件、多 seed 与不确定性，**不能以单次结果作强结论**。

### 4.2 多实体数据：按实体建 bank、按实体评价（**本 PRD 的核心口径之一**）

MSL/SMAP 的 `.npy` 不是"时间戳×通道"，而是**逐通道文件纵向拼接**：
每块第 0 列是该通道信号，第 1..K−1 列是二值指示列。**实体边界可精确恢复** ——
`bench/channel_split.py::stacked_blocks(ds, raw_dir, split=...)` 按 `labeled_anomalies.csv`
的通道顺序切块（逐 split 的块长不同，往 test 分析必须用 `split='test'`）。
**已实测**：切出的逐块 `n_anomalous_points` 求和 = 7766 = `MSL_test_label` 的异常点总数 → 边界与标签严格对齐。

**MSL 实测（test 切分）**：

| 项 | 数值 |
|---|---|
| 实体数 | **27**（全部含至少 1 个异常段） |
| 逐实体异常点率 | **0.60%（S-2） ~ 50.11%（M-1 / M-2）**，跨度 **84×** |
| 逐实体测试窗数 | 1026（T-9） ~ 6030（P-10 / P-14） |
| 逐实体训练窗数 | 56427 合计（每块之间损失 26×70=1820 个跨接缝窗口） |
| 逐实体 bank（stride 10） | 36（T-9） ~ 423（F-7），中位 208 |
| **信号恒值（σ=0）的实体** | **5 个**：C-2、D-14、M-6、S-2、T-5 |

**为什么必须按实体做**：

1. 逐实体异常率跨度 84×。pooled AUPRC 会被高异常率的少数实体主导，得到的数字**不是任何实体的数字**。
2. 各实体 col0 的 μ/σ 差异极大（C-1 μ=−0.67 vs P-10 μ=0.993），加上指示列，
   实体身份本身就是表示空间里的一个强因子 → 跨实体的 kNN 距离主要在量实体身份，而非正常/异常。
3. 按实体切窗（窗口不跨接缝）同时消掉 26 个跨接缝的**无意义窗口**。

**报告方式**：

1. 逐实体 AUPRC / AUROC（**仅当该实体正常/异常标签均存在时才计算**；只有单一标签的标记"不可定义"并说明）；
2. **实体级 macro mean ± dispersion 为主结果**；
3. pooled 结果作附加列，并注明其受实体样本数与异常率影响；
4. **5 个 σ=0 实体**：信号恒值，重构类方法在原理上无法检出 → 单独标记，
   **macro mean 报"含/不含这 5 个"两版**，并解释。
5. 逐实体 bank 大小一并报告（bank < 20 时该实体结果不可靠，标记而非静默使用）。

对 **SWaT**（单一实体，51 个平铺通道）：参考正常训练段即 normal reference，无实体切分。

### 4.3 窗口标签口径（预注册）

| 口径 | 定义 | 用途 |
|---|---|---|
| **any-in-window**（主） | 窗口 `[i, i+W−1]` 内**含有**任何异常点 ⇒ 1 | 语义最干净：窗口级重构分数本身就是对整窗算的 |
| last-point（次） | 取 `test_label[i+W−1]` | 与 `main.py` 现有口径最近（差一个位移，见 §3.4-3），作敏感性对照 |

MSL 评测区（全 test 的全局口径）：any-in-window 17.05%（8161 窗）、last-point 13.88%（6643 窗）。
**两种口径在所有变体上必须完全一致。两套数都报，主表用 any-in-window。不得混用两种粒度。**

### 4.4 数据角色与泄漏控制

| Split | 用途 | 能否用标签 |
|---|---|---|
| 原训练集 | 建立 reference bank（**逐实体**） | 不用 test 标签 |
| 固定测试集 | 提取待评价特征 | 标签**只**用于最后算指标与图上着色 |
| 可选 validation | 只用于**预先**固定技术超参数 | 不得用 test 标签调参 |

**禁止**：用测试标签筛选 bank、选择最优 $k$、挑选最优层、
或"看哪张图分得开再把它当成预先指定的主结果"。

### 4.5 一条必须写明的假设

**train 全为正常**是**假设**，不是事实。MSL/SMAP 的训练数组无标签，
`labeled_anomalies.csv` 的异常区间是针对 test 的。因此对 MSL/SMAP 应称
**nominal / unlabeled training reference**，**不得**未经核验就断言其中绝对没有异常。
若参考库本身被污染，这恰好是 §6.4 中 $D_N$ 偏大、RSR 偏小的一个可能解释，应作为讨论而非前提。

---

## 5. 特征抽取协议

### 5.1 对齐清单（所有模型必须一致）

- 原始 train/test 划分与 `preprocess`（MinMaxScaler 只在 train 上拟合）；
- **逐实体窗口化**：窗口索引范围为 `[t_start, t_end − W]`，不跨实体接缝（MSL/SMAP）；
- window length、stride、entity order、时间顺序；
- window-to-label 对齐规则（§4.3）；
- **`model.eval()`（关 dropout）**；
- **batch 组成固定**：`batch_size=60` + `shuffle=False`（§3.4-1，这一条不写清则不可复现）；
- 固定的特征抽取接口（§3.3 的 hook）与样本索引。

### 5.2 特征形状与 pooling

1. 记录每个 stage 的原张量形状；
2. 明确每条 embedding 对应的时间点/窗口/实体；
3. **预先指定 pooling**：$Z_p$ 是 `[B,55,512]`，主口径取 **按实体通道维 mean pooling → `[B,512]`**；
   不因为某种 pooling 效果更好而临时改变（可作敏感性附列）；
4. 固定输出 `[N, d]`，与标签 `[N]` 一一对应；
5. 原始 shape 与 pooling 规则写入实验元数据。

### 5.3 泄漏控制

- reference bank 只来自训练集特征（逐实体）；
- 测试特征只作 kNN query；**不使用 test-test 邻居计算主结果**；
- 所有特征预处理参数（标准化）**只**从训练 reference bank 拟合；
- 所有模型固定相同的样本索引、batch 划分与测试标签。

---

## 6. 核心指标

### 6.1 Normal-reference kNN score（主指标基础）

对每个（模型/阶段，实体）：

$$\mathcal{Z}_{\mathrm{ref}} = \{z_j^{\mathrm{train}}\}_{j=1}^{M},\qquad
s_k(z_i) = \frac{1}{k}\sum_{z_j \in NN_k(z_i, \mathcal{Z}_{\mathrm{ref}})} \|z_i - z_j\|_2$$

**bank 口径（预注册）**：**逐实体**从训练窗中**等间隔抽 $\min(N, 400)$ 条**（确定性、无 RNG）。
理由：① 相邻训练窗几乎重复，全量库会让"到正常流形的距离"退化成"到最近一份拷贝的距离"；
② 等间隔抽使各实体 bank 规模可比（实测 stride-10 的 bank 仅 36~423，T-9 只有 36，与 k=10 相冲突）。
逐实体 bank 大小一并报告。

**标准化**：先用该实体 $\mathcal{Z}_{\mathrm{ref}}$ 的逐维均值/标准差标准化，该参数固定应用于该实体的测试表示；
**去掉训练集上方差≈0 的维度**避免除零。
（`nomssa` 的常数维不同 → 各变体各自拟合，维度数可能要报。）

**默认 $k=10$**；敏感性 $k \in \{5, 10, 20\}$。
**主结果固定用 $k=10$，不允许用测试标签搜索最优 $k$。**
高维可能有距离集中问题 → 同时检查 score 分布、有效维数、不同 $k$ 下的排序稳定性。

> **计算量**（实测口径）：逐实体 kNN 的最大距离矩阵是 `6030 × 400 × 275` —— 毫秒级。
> 逐实体口径比 pooled（47854 × 5825 × 275）小约 3 个数量级，**不需要 faiss / 分块优化**。

### 6.2 repr-AUPRC（主指标）

$$\boxed{\mathrm{AUPRC}_{repr}}$$

- threshold-independent；
- 直接衡量 learned representation 是否使异常更远离参考正常模式；
- 可跨模型/阶段比较（排序类，不要求原始距离同尺度）。

**必须同时报告该分组的 anomaly prevalence**（随机排序的参考水平），否则跨数据集/跨实体不可读。

> ⚠️ 这是**附加的 representation diagnostic**，**不是** PESR reconstruction-based detector 的 AUPRC。
> 不得偷换成模型原有的异常检测性能。原生 detector 的数字见 §0，与之并列报告。

### 6.3 repr-AUROC（辅助）

与 AUPRC 互补。**若该实体/分组没有正负两类，则不计算**（标记为不可定义）。

### 6.4 kNN 距离分布与分离度

$$D_N = \mathbb{E}_{i: y_i=0}[s_k(z_i)],\quad D_A = \mathbb{E}_{i: y_i=1}[s_k(z_i)],\quad
\mathrm{RSR} = \frac{D_A}{D_N + \varepsilon},\quad
\mathrm{Sep}_{score} = \frac{D_A - D_N}{\sqrt{(\sigma_A^2 + \sigma_N^2)/2} + \varepsilon}$$

$\sigma_N, \sigma_A$ 是 **kNN 距离分数的标准差**，**不是**原始 embedding 的坐标方差。

> **为什么用这个而不是质心距离**：工业系统存在多个正常工况，把正常表示强行视为单个高斯簇
> 并以"离全局 centroid 越近越好"下结论，会把数据的性质报告成模型的缺陷。
> 距离分数口径不做这种假设，且与主指标同源。

**解读**：$D_A > D_N$、$\mathrm{Sep}_{score}$ 更大 ⇒ 异常通常更远离训练参考。
但 **RSR / Sep 只作辅助**：分布重叠、长尾与异常率都可能让均值差误导。**AUPRC/AUROC 是主要依据。**

### 6.5 敏感性（全部一次性报完，不得事后挑选）

| 维度 | 主口径 | 敏感性 |
|---|---|---|
| $k$ | 10 | 5, 20 |
| 标签 | any-in-window | last-point |
| 采样 | `eval()` + `bs=60` | train 模式 / `bs=1`（各抽 5k） |
| 阶段 3 | $\mu_t$ | 采样 $z$（固定 seed） |
| bank | 等间隔 min(N,400) | 全量训练窗 |

---

## 7. 实验 A：跨变体表示可分性（主实验）

对每个变体：加载对应 ckpt → 抽同一功能位置（**$V_t$**）的特征 → 逐实体建 bank →
对固定真实 test 窗口算 kNN score → 用真实标签算 AUPRC/AUROC → 统计 $D_N, D_A, \mathrm{Sep}_{score}$。

| 数据集 | 模型 | repr-AUPRC ↑ | repr-AUROC ↑ | $D_N$ | $D_A$ | Sep ↑ | score-AUPRC（原生） | F1(adj, 参照) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MSL | PESR − MPA | | | | | | | |
| MSL | PESR − MSSA | | | | | | | |
| MSL | PESR | | | | | | | |
| SWaT | …（同上三行） | | | | | | | |

**逐实体表另附**（27 行 × 3 变体），主表是 macro mean ± std（跨 seed 与跨实体两种离散度分别报）。

**解读**：

- PESR 的 repr-AUPRC/AUROC 更好 → 说明其学到的结构更有利于区分真实正常/异常；
- 某个消融版更好 → **不能宣称所有模块都提高可分性**；应分析该模块是否主要经由
  下游重建评分、时序关联或训练正则化改善最终任务指标。
- **不直接比较 $D_N$ 绝对值跨变体排名**（尺度不同）；用排序类指标与比率。

---

## 8. 实验 B：完整 PESR 的逐阶段演化

### 8.1 设置

仅用完整 PESR，依次抽 $Z_p \to V_t \to \mu_t$。
**各阶段独立建训练 reference bank、独立拟合训练侧标准化参数**，使用相同测试样本与标签评价。

### 8.2 输出

| 数据集 | Stage | Dim | repr-AUPRC ↑ | repr-AUROC ↑ | Sep ↑ |
|---|---|---:|---:|---:|---:|
| MSL | MPA 输出 $Z_p$ | 512 | | | |
| MSL | CRR 输入 $V_t$ | 275 | | | |
| MSL | CRR 编码器均值 $\mu_t$ | 256 | | | |
| SWaT | …（同上三行） | | | | |

### 8.3 解读边界

**不要求** $\mathrm{AUPRC}(Z_p) < \mathrm{AUPRC}(V_t) < \mathrm{AUPRC}(\mu_t)$。
CRR 的首要目的是为重建提供潜在分布，**并非显式优化 kNN separability**。
若 $\mu_t$ 的 kNN 可分性下降，**不自动说明 CRR 无效**；
应结合 §0 的原生 anomaly score 与 H3 讨论，但**不能包装成"逐层提升"**。

> 注：$Z_p$ 的 512 维是 pooling 后的；若换成全量 `[B,55,512]`，近 28k 维，
> kNN 会有严重距离集中问题，**主口径不用**（作附注）。

---

## 9. 实验 C：可视化

### 9.1 图

**Figure A — Ablation-level**：三列（PESR − MPA / PESR − MSSA / PESR），同一数据集、同一批真实窗口的 $V_t$。

**Figure B — Stage-wise**：三列（$Z_p$ / $V_t$ / $\mu_t$，仅完整 PESR）。

每张三类着色：**nominal train / normal test / anomalous test**。

### 9.2 顺序（与 v1 不同，按信息量排序）

1. **kNN score distributions（KDE / violin / ECDF）** —— 最直接对应 repr-AUPRC 与 normal–anomaly separability；
2. **PCA（2D）** —— 确定性更强，优先于 t-SNE；
3. **t-SNE** —— 可选补充，只看局部邻域。

### 9.3 正文最省版面的图

一张多面板：(a) 三变体 repr-AUPRC 点图（mean ± std）；(b) 完整 PESR 的逐阶段 repr-AUPRC；
(c) PESR 与关键消融的 normal/anomaly kNN 距离分布。二维投影放附录。

### 9.4 约束

- 固定样本索引与抽样规则；异常/正常**预先固定**抽样比例并注明"可视化分布 ≠ 真实类别比例"；
- t-SNE/PCA **不使用标签**作为优化目标；
- **不依靠不同面板的二维绝对距离**判断谁在原始高维空间更分离；
- 标明特征空间不同，二维投影只作辅助解释；
- **不能从单张"看起来分得开"的图得出统计结论**；
- 按工作区约定，图片落盘后**不以 Read 打开**，只查尺寸/数量/元数据。

---

## 10. 重复性、CI 与运行控制

每个变体至少 $n_{\mathrm{seed}} = 3$（如 42 / 7 / 17）。每个 seed：

- 用对应 seed 训练的模型；
- train/test 划分、window 参数、样本 index 固定；
- **batch 划分固定**（§3.4-1）；
- 同一 seed 下不同模型用**相同索引**；
- bank 抽样规则与 $k$ 固定（bank 为确定性等间隔抽样，无 RNG）。

**报告**：seed mean ± std；每个数据集/**实体的有效正负样本数**；$k \in \{5,10,20\}$ 的敏感性。

⚠️ **区间估计不能用天真的 iid 窗口抽样**：相邻时间窗重叠 $W-1$ 个点，高度自相关。
如需 CI，用**实体级或攻击事件级的 bootstrap**（对事件重采样，而非对窗口重采样）。

**单 seed 只作 feasibility check，不得当作有统计显著性的最终证据。**

---

## 11. 有效性 Gate（0 → 1 → 2，全过才铺开）

### Gate 0 · forward feature 是否真实有效

- shape / dtype / NaN / Inf；
- 每维与整体的 variance；**重复 forward 是否一致**（§3.4-2：不一致 = 没调 `eval()`）；
- 真实异常与正常输入是否得到**非恒定**表示；
- **`window index ↔ label` 是否严格对应**（含实体归属）。

> 此前记录显示**最终异常分数近似常数**（§0）。这**不等于**所有内部特征也近似常数，
> 所以必须先抽内部特征核验。**如果抽出的表示也接近常数 → 暂停分析，先排查提取接口或网络数值行为。**

### Gate 1 · reference bank 与 kNN 是否正常

- bank size 足够（逐实体报告，< 20 标记）；
- 特征预处理**没有用到 test statistics**；
- $k$ 小于 bank size；
- 训练正常样本的邻居距离与测试 score 有合理变化；
- **没有重复时间窗或实体身份造成的检索捷径**（MSL 特有：检查最近邻是否总是同实体，若不是，说明实体切分没生效）。

### Gate 2 · 标签与指标是否有效

- 有正负两类标签；
- test label 粒度与 embedding 一致；
- AUPRC > 0 且按**正确方向**（越异常 score 越高）计算；
- 按**预先指定**的指标输出，**不事后反转 score 方向**去取更高结果；
- 结果与 **test prevalence** 一起报告。

---

## 12. 执行

### 12.1 阶段

| 阶段 | 内容 | 成本 | 目的 |
|---|---|---|---|
| **T0** | 重训 `nomssa` MSL ρ=0 seed42（唯一 ckpt 归属不明的那个）+ 三变体提 $V_t$/$\mu_t$ + **Gate 0/1/2** + 出实验 A/B 的单 seed 数 | **~1 h** | 先看 kNN 指标有没有信号（会不会也是 base rate）。**Gate 未过或全无信号 → 先不铺 seed** |
| **T1** | MSL：9 run 全重训（3 变体 × 3 seed）+ 全量提取 + 实验 A/B/C + 敏感性 | ~2 h | 主结论 |
| **T2** | SWaT 复制一遍（W=35，3 变体 × 3 seed） | ~2 h | external validity，**MSL 不成立则不做** |

### 12.2 命令模板

```bash
# (1) 重训：唯一化 ckpt，绝不共用
CUDA_VISIBLE_DEVICES=$gpu python main.py --dataset MSL --name repr/$bb \
    --trainmode train-test --backbone $bb --window_length 70 --batch_size 60 \
    --seed $S --noise 0 \
    --result-dir results/repr --result-name "${bb}_MSL_seed${S}" \
    --ckpt-dir checkpoint/repr/${bb}_MSL_seed${S}

# (2) 提取（新脚本，零改动模型文件）
CUDA_VISIBLE_DEVICES=$gpu python bench/extract_repr.py \
    --ckpt checkpoint/repr/${bb}_MSL_seed${S}/best.pt --backbone $bb --dataset MSL \
    --split test  --out results/repr/${bb}_MSL_seed${S}_test.npz
CUDA_VISIBLE_DEVICES=$gpu python bench/extract_repr.py \
    --ckpt checkpoint/repr/${bb}_MSL_seed${S}/best.pt --backbone $bb --dataset MSL \
    --split train --out results/repr/${bb}_MSL_seed${S}_train.npz

# (3) 分析 + 画图
python bench/analyze_repr.py --glob 'results/repr/*_MSL_*.npz' --k 10 --entity-wrap \
    --out results/repr/summary
python bench/plot_repr.py --glob 'results/repr/*_MSL_*.npz' --out figs/repr
```

`--split test` 落盘 `vt/rt/zp`、`score_eval`、`score_train`、`label_any`、`label_last`、`entity_id`；
`--split train` 落 bank（按实体等间隔 min(N,400)）。全部在 `model.eval()`、`bs=60`、`shuffle=False` 下提取。

**复用**：`bench/cache_scores.py`（切分/窗口化/loader 口径的模板，**逐行照抄**，否则窗口顺序与 label 对不上）；
`bench/channel_split.py::stacked_blocks`（**实体边界与逐实体异常点，已有且已测**）。

### 12.3 新增脚本

| 文件 | 说明 |
|---|---|
| `script/repr/retrain_msl.sh` | 9 run 队列（仿 `script/contam/B2_grid.sh`：worker-pool + `mkdir` 原子占坑，支持 `GPUS`/`TASK_FILTER`） |
| `bench/extract_repr.py` | hook 提取；断言 `attmamba`/`project4`/`projection3`/`encoder` 存在；实体切窗；写元数据 |
| `bench/analyze_repr.py` | 逐实体 kNN / 标准化 / AUPRC / AUROC / $D_N,D_A,RSR,\mathrm{Sep}$ / macro 聚合 / 事件级 bootstrap |
| `bench/plot_repr.py` | score 分布 + PCA + t-SNE |

**模型文件零改动** → 不需要重跑 `make_ablation_models.py`，也不扰动 PRD1 既有结果。

### 12.4 成本实测依据

| 项 | 数值 | 来源 |
|---|---|---|
| 每 epoch（MSL, 971 步） | `mamba` 360.7 s、`nompa` 360.9 s、`nomssa` 81.2 s | `results/contam/*.json` 的 `timing` |
| 单次训练（10 epoch） | `mamba`/`nompa` ≈ 60 min、`nomssa` ≈ 14 min | 同上 |
| 一次前向（47854 测试窗, bs=60） | **88 s**（`nomssa` 20 s） | 2026-10-10 实测 |
| MSL 9 run 训练墙钟 | 6 张空卡并行 → **~2 h** | GPU 1/6 被他人占用，0/2/3/4/5/7 空闲 |
| 逐实体 kNN | 最大 6030 × 400 × 275 → **毫秒级**，无需 GPU 优化 | §6.1 |

---

## 13. 结果解释与决策规则（预注册）

设 seed 均值 $m$、标准误 SEM；base rate = 该分组的标签异常率。

| # | 判据 | 结论 |
|---|---|---|
| **D1** | `repr-AUPRC(V_t)` 显著 > base rate（$m - 2\mathrm{SEM} > \text{base}$） | 表示层**确实**携带异常信息 → 继续 D2/D3 |
| **D2** | PESR 的 $m - 2\mathrm{SEM}$ 同时高于两个消融 | 支持"模块改善了可分性"（**要的结果**） |
| **D3** | **D1 不成立** | 诚实结论：**该模型学到的表示里也不含异常判别信息**。此时**不得**用任何检测类指标声称模块有效；如实报告，并与 §0 并列成"重构目标坍缩了判别信息"的 limitation |
| **D4 (H3)** | `repr-AUPRC(V_t) > score-AUPRC` 且差距 > 2·SEM（**同一模型内比较**） | 瓶颈在**分数定义**而非表示 → 论文的强论点 |
| **D5** | 变体内 `AUPRC(Z_p) ≤ AUPRC(V_t) ≤ AUPRC(μ_t)` 在 ≥2/3 seed 成立 | 写成 trend，不写成强单调 |
| **D6** | 跨 run 的 `Spearman(repr-AUPRC, score-AUPRC)` | 仅报告（n=9 太小），不作主张 |

### 13.1 "观察 → 可写 / 不可写"对照

| 观察结果 | 可以写 | 不能写 |
|---|---|---|
| PESR repr-AUPRC 高于消融 | MPA/MSSA 的组合改善了正常/异常表示判别能力 | 证明了训练污染被消除 |
| PESR 的 $V_t$ 好，但 $\mu_t$ 一般 | 主要收益可能集中在多尺度表征阶段 | CRR 没有价值 |
| $\mu_t$ 比 $V_t$ 更好 | CRR 编码后的表示更利于该 kNN 诊断 | CRR 在所有下游机制上更优 |
| 表示 AUPRC 接近，但原生 F1 不同 | 最终判别差异可能依赖重建评分与阈值 | 表示没有任何价值 |
| 全部表示近乎常数或无 separability | 检查特征接口、训练状态与打分机制 | 为符合叙事挑选少数漂亮图 |

**禁止根据结果只报"最能证明模块有用"的指标或数据集而不报其余已运行设置。**
**不做跨变体排名**（容量差 76.5% vs 1.2%，排名会被容量差异解释掉）。
**不补 FLOPs 表、不做容量对齐。**

---

## 14. 论文与回复信落地

### 14.1 正文位置

`Section 4 (Experiments), after Ablation Study`，新增小节 **Representation Discriminability Analysis**，
含：模块级 repr-AUPRC/AUROC 对比、完整 PESR 的逐阶段变化、分布图/PCA。

**E4（既有 F1 消融表）照旧保留**（社区口径，便于横向比），但明确标注
**F1 不用于变体间归因**（§0），三变体的 F1 明细放附录。

### 14.2 Introduction 需同步收紧

把 "learn normal patterns without the impact of anomalies" 改为：

> **learn stable and discriminative representations of normal temporal behavior,
> while reducing sensitivity to potentially anomalous observations**

**只做本 PRD 时，不应继续宣称已定量证明"训练污染如何伤害正常模式学习"。** 动机段建议：

> Learning reliable normal patterns from multivariate time series is challenging because industrial
> observations exhibit complex temporal dependencies and may contain local deviations that are difficult
> to distinguish from normal state variations. Reconstruction-based detectors rely on the representations
> learned from these observations, and insufficiently discriminative representations can reduce the
> separation between normal and anomalous states. PESR therefore combines multi-view patching alignment
> and multi-scale temporal modeling to construct more informative representations before reconstruction.

（**拟议修改**，不代表实验结果已支持全部效果表述。）

### 14.3 Response Letter 的证据边界

**对 R1**：可以说"我们收紧了过强的 contamination claim，并改用真实异常的表示可分性做定量分析"；
**不能**说已用该实验测量了"训练污染比例对正常模式学习的因果影响"。editor 也强调 contamination impact，
**仅凭本实验仍有二审追问风险**（§1.3）。

**对 R2**：能解释 MPA/MSSA 如何改善重建前表示及其与最终消融的联系；
**不能**单凭该实验证明 PESR 已普遍克服所有 conventional reconstruction-based methods 的局限。

### 14.4 与已有实验的一致性

- 主结果：仍报最终 detector 指标（**并附 §0 的无阈值口径**）；
- Ablation：仍负责验证模块对最终任务的贡献；
- 本实验：提供 feature-space mechanism evidence；
- 随机替换 robustness：保留，但明确它是 **corruption tolerance**，不把平曲线解释成训练污染机制的证据；
- ⚠️ **原生 anomaly score 近乎无分辨力这件事必须单独立项说明**（§0），
  **不得用 repr-kNN AUPRC 掩盖原生 detector 的问题**。

### 14.5 一句话结论目标

> The representation-level analysis determines whether the complete PESR learns more discriminative
> normal-pattern representations than its ablated variants, and how this discriminability evolves across
> MPA, MSSA, and the CRR encoder, using naturally occurring anomalies and label-independent
> normal-reference kNN scores.

实验完成后才依真实结果决定能否写：
*These results suggest that the integrated multi-view and multi-scale modeling contributes to more
discriminative representations before reconstruction.*
**不把上述结论偷换成"已证明训练异常污染被直接缓解"。**

---

## 15. MVP 与成功标准

**第一轮（T0+T1）**：

1. 数据集 MSL（SWaT 待 T2）；
2. 模型 PESR − MPA / PESR − MSSA / PESR，统一在 **$V_t$** 比较；
3. 指标 repr-AUPRC（主）、repr-AUROC、$D_N/D_A$/RSR/$\mathrm{Sep}_{score}$；
4. 逐阶段：完整 PESR 的 $Z_p, V_t, \mu_t$；
5. 图：两组 AUPRC 对比 + 一张代表性 score 分布图（必要时加 PCA）；
6. **先做单 seed 的 Gate 0/1/2**；
7. 正式报告：≥3 seeds + $k$ 敏感性。

**成功标准**：得到一个方法学成立、可复现、**无需测试标签参与任何模型/kNN/选图决策**的 representation analysis；
结果是否支持各项假设由真实数据决定。
**不能把"PESR 必须第一、指标必须逐层递增"当作运行成功标准。**

---

## 16. 交付物

| 类型 | 路径 |
|---|---|
| 提取特征 | `results/repr/<bb>_<ds>_seed<S>_{test,train}.npz` + 同名 `.json`（ckpt / hook 位置 / W / bs / seed / pooling / 实体表） |
| 指标 | `results/repr/summary.{md,csv}`（实验 A 主表 + 逐实体表 + 实验 B 阶段表 + 敏感性网格） |
| 训练结果 | `results/repr/<bb>_<ds>_seed<S>.json`（复用 main.py 既有 schema） |
| 图 | `figs/repr/{ablation_auprc,stage_auprc,score_dist,pca_or_tsne}.pdf` |
| 有效性检查 | `results/repr/validity_checks.md`（Gate 0/1/2 的逐项记录） |
| 脚本 | `script/repr/retrain_msl.sh`、`bench/extract_repr.py`、`bench/analyze_repr.py`、`bench/plot_repr.py` |
| 本文 | `docs/PRD5_表示层定量分析.md` |

`npz` 至少记录：feature、window index、**entity id**、labels（仅测试侧）、stage、seed、pooling 配置。

---

## 17. 与其它 PRD 的关系 / 归档

| 文档 | 关系 |
|---|---|
| `docs/archive/PRD5_PESR_Representation_Discriminability_Experiment_PRD.md` | 本 v2 的**设计主干**（RQ/H、Gate、实体口径、回复信落地）。已归档 |
| `docs/archive/PRD5_表示层定量分析_v1.md` | v1。其 §0（detector 诊断）、实现约束、成本实测已全部并入本文。已归档 |
| `PRD4text_定量分析_r2.md` / `PRD4text_状态记录_20261007.md` | **暂停**。本 PRD 继承其术语表与变体定义，**放弃**其核心变量 ρ |
| `PRD4_噪声定量实验.md` | 动机需重估：该文以 point-adjust F1 为主指标（§4），而该指标已被证明无分辨力。**若要做，主指标须改成未调整 AUPRC** |
| `PRD1_消融实验.md` | 替换型消融，本 PRD 不做；结果照旧保留（§14.1 E4） |
| `PRD3_阈值选择策略.md` | 不受影响（阈值问题与表示可分性是两件事；§0 只说明阈值之上的排序无信息） |
