# PESR 表示空间定量分析实验 PRD

> **Experiment:** Representation-level Quantitative Analysis for PESR  
> **Purpose:** Major Revision · Motivation / Methodology / Ablation  
> **Status:** 待实施（本文只规定实验方案，不预设任何结果）  
> **版本:** v1.0  
> **核心约束:** 不构造合成异常、不改变已有 PESR 训练目标、不将 CRR 替换为 AE；真实测试标签仅用于评价与可视化。

---

## 1. 背景与实验定位

### 1.1 审稿意见

**R1 — Unclear motivation**

> The authors should better quantify how the identified issue (i.e., anomalies contaminating the learning of normal patterns) actually affects real systems, for example, by analyzing how performance degrades as anomaly ratios increase, or how much reconstruction quality on normal data is compromised in the presence of anomalies, so that the practical significance of the motivation is more clearly justified.

**R2 — Limitations of reconstruction-based methods**

> The authors need to effectively address the limitations of conventional reconstruction-based methods and demonstrate a meaningful contribution to the anomaly detection literature.

### 1.2 已有实验为何不作为主方案

此前已经分别尝试了训练侧随机噪声替换和连续异常段注入。根据现有《PRD4_两轴实验结果分析.md》：

- SMAP、SWaT 的训练噪声比例曲线没有清晰的性能退化趋势；
- MSL 的消融模型在污染率提高时，也未出现稳定的劣化关系；
- WADI 在部分设置下重复运行差异较大，不适合仅依据单次 F1 说明污染机制；
- 已有诊断提示最终 reconstruction/anomaly score 的动态范围和异常区分度可能存在问题。

**结论：不能为了获得预期下降曲线而继续无依据地提高噪声强度或异常比例。**

因此，本 PRD 不再把“训练污染率引起性能下降”设为必要前提，而是改用原始真实数据，研究模型形成的**正常模式表示质量**以及**正常/异常样本的可分性**。

### 1.3 本实验能够证明什么、不证明什么

**本实验要验证：**

1. 完整 PESR 的表示是否比去除 MPA/MSSA 后的表示更有利于区分真实正常与异常状态；
2. MPA、MSSA 和 CRR 编码阶段如何改变表示结构；
3. 模块消融造成的最终异常检测性能差异，是否可以从表示空间的几何结构中得到解释。

**本实验不能单独证明：**

- 训练集中异常比例增加会导致表示退化；
- 模型是否实际“记住/吸收”了训练异常；
- MPA/MSSA 在因果意义上直接过滤了异常；
- CRR 编码后的可分性一定比其输入更强。

这些是不同的命题。如果修订稿将第一条 motivation 收紧为“稳定、具有判别力的正常时序表示难以学习”，本实验可以提供直接的量化支持；如果仍坚持原有强表述“训练异常污染正常模式学习”，则 **本实验不能完全替代污染因果实验**。

---

## 2. 研究问题与假设

### RQ1：模块级表示质量

比较：

\[
\mathrm{w/o\ MPA},\quad
\mathrm{w/o\ MSSA},\quad
\mathrm{PESR}
\]

在同一功能位置（**CRR 输入端 / 最终重建前的表示**）输出的特征。

**H1（待验证）：** 完整 PESR 的 representation-based anomaly separability 优于至少部分消融变体。

### RQ2：PESR 内部逐阶段表示演化

对完整模型提取：

\[
X \;\longrightarrow\;
Z_p\ \text{(MPA output)}
\;\longrightarrow\;
V_t\ \text{(MSSA output / CRR input)}
\;\longrightarrow\;
\mu_t\ \text{(CRR encoder mean)}.
\]

**H2（待验证）：** MPA、MSSA 与 CRR 编码阶段产生不同的表示几何结构，能够通过定量分离指标体现；**不预设指标必须逐层单调增加**。

### RQ3：几何证据与最终检测结果是否一致

结合原有 F1 消融结果，检查表示可分性较好的模型，是否也倾向于获得更好的最终检测结果。

**H3（待验证）：** 两者具有一定一致性；若不一致，需要讨论重建评分函数、后续映射以及阈值协议的影响，不能强行解释。

---

## 3. 模型与特征提取点

### 3.1 主对比模型

| ID | Model | MPA | MSSA | CRR | 比较目的 |
|---|---|---|---|---|---|
| M1 | w/o MPA | ✗ | ✓ | ✓ | 检验多视角 patching alignment 的作用 |
| M2 | w/o MSSA | ✓ | ✗ | ✓ | 检验多尺度时序建模的作用 |
| M3 | PESR | ✓ | ✓ | ✓ | 完整模型 |

**不新增 CRR→AE 的替换实验。** CRR 是本文重建流程的一部分，所有模型均保留。

对于 M1/M2/M3，在**相同功能接口（进入 CRR 的特征）**提取向量。即使某一变体删除了 MSSA，也应使用它实际传入 CRR 的特征，而不是凭名称强行视为完整模型的 \(V_t\)。

> 注意：不同变体可能形成不同特征空间。比较 AUPRC/AUROC 等任务相关的无量纲指标；不直接比较其未经标准化的特征距离绝对值。

### 3.2 逐阶段分析：仅完整 PESR

| Stage | Feature | 实际含义 | 目的 |
|---|---|---|---|
| S1 | \(Z_p\) | MPA 输出 | 多视角对齐后的表示 |
| S2 | \(V_t\) | MSSA 输出 / CRR 输入 | 多尺度时序融合后的表示 |
| S3 | \(\mu_t\) | CRR 编码器输出分布的均值 | 重建潜空间的确定性表示 |

**S3 优先使用编码器均值 \(\mu_t\)，不要用每次随机采样的 \(r_t\) 直接画图。** 否则 VAE 的采样随机性会人为放大类内离散度。

如果实际 CRR 代码中的均值接口或张量语义与上述描述不同，先核对真实 forward path，再记录正确的特征定义，不能为了图形方便自行重构一个“近似 latent”。

---

## 4. 数据集与划分

### 4.1 首选数据集

**第一优先级：MSL + SWaT。**

- **MSL**：包含多个实体及真实异常区间，可用于较丰富的正常/异常表示分析；多实体分析必须避免混合身份信息造成虚假距离。
- **SWaT**：与工业场景直接相关，正常运行与攻击区间有明确区分，适合可视化正常状态与异常状态。

**可选扩展：SMAP；WADI 作为探索性补充。**

- **SMAP**：若抽取管线与 MSL 共用，可增加覆盖范围。
- **WADI**：此前实验提示其部分切分/阈值下的 F1 波动较大且有效异常事件数有限；若继续使用，必须报告有效测试事件、多个 seed 与指标不确定性，不能以单次聚类图作强结论。

### 4.2 数据角色

| Split | 用途 | 能否使用标签 |
|---|---|---|
| 原训练集 | 建立模型自己的 reference feature bank | 不用 test labels；遵循原论文训练协议 |
| 固定测试集 | 提取待评价特征 | 测试标签只用于最后计算指标、图上着色 |
| 可选 validation | 仅用于预先固定技术超参数 | 不得用 test labels 调参 |

**禁止**使用测试标签筛选 reference bank、选择最优 \(k\)、挑选最优层或挑选“看起来分得最开”的图后再将其当作预先指定的主结果。

对 SWaT，参考正常训练段可称为 normal reference。对 NASA 的 MSL/SMAP，应依照真实训练数据性质使用 **nominal/unlabeled training reference**，不要未经核验就断言其中绝对没有异常。

### 4.3 多实体数据特殊处理

MSL/SMAP 的多实体数据应**按实体构建 reference bank 并按实体评价**，避免不同实体的身份/工作模式差异主导 kNN 距离。

报告方式：

1. 每个实体的 AUPRC、AUROC（仅当正常/异常标签均存在时才计算）；
2. 实体级 macro mean ± dispersion；
3. 如需 pooled point-level AUPRC，作为附加结果，并明确其易受实体样本数与异常率影响。

若某实体只含一种标签，AUPRC/AUROC 不应强制给出正常数值；应标记不可定义并解释。

---

## 5. 特征抽取协议

### 5.1 数据对齐

所有模型必须使用一致的：

- 原始 train/test 划分；
- normalization；
- window length、stride；
- entity order 与时间顺序；
- window-to-label 对齐规则；
- evaluation mode（`model.eval()`，关闭 dropout）；
- 固定的特征抽取接口与样本索引。

**重点核查窗口标签：** 原稿的问题定义倾向于针对窗口末端时间点 \(y_t\) 判断异常；如果现有代码采用 `any abnormal point in window`、point-adjustment 或其他规则，必须对齐到实际 protocol，并注明。**不能混用两种标签粒度。**

### 5.2 特征形状

对每一 stage 的输出：

1. 记录原张量形状：`[B, ...]`；
2. 明确每一条 embedding 对应的时间点/窗口/实体；
3. 若特征有多个 token/patch，预先指定 pooling 方式（如 mean pooling），不能因为某种 pooling 效果更好临时改变；
4. 固定输出为 `[N, d]`，与标签 `[N]` 一一对应；
5. 原始 shape 和 pooling 规则写入实验元数据。

如模型原本提供与异常打分直接对应的窗口级特征，应优先使用该特征，不额外丢失关键位置信息。

### 5.3 数据泄漏控制

- reference bank 只能来自训练集特征；
- 测试特征只作为 kNN query；
- 不使用 test-test 邻居计算主要异常距离；
- 所有特征预处理参数仅从训练 reference bank 拟合；
- 所有模型固定相同的样本索引和测试标签。

---

## 6. 核心定量指标

### 6.1 Normal-reference kNN anomaly score（主指标基础）

对于每个模型或 stage，使用其对应的训练参考特征集：

\[
\mathcal{Z}_{\mathrm{ref}}
=\{z_j^{\mathrm{train}}\}_{j=1}^{M}.
\]

先用 \(\mathcal{Z}_{\mathrm{ref}}\) 的均值和标准差对每维特征标准化，标准化参数固定应用于测试表示。去除训练集上方差几乎为零的维度，避免除零。

对于每个测试表示 \(z_i\)：

\[
s_k(z_i)
=\frac{1}{k}
\sum_{z_j\in NN_k(z_i,\mathcal{Z}_{\mathrm{ref}})}
\left\|z_i-z_j\right\|_2 .
\]

- **\(s_k\) 越大：** 表示越远离训练参考模式；
- **\(s_k\) 越小：** 表示越接近训练参考模式。

默认：

\[
k=10
\]

敏感性分析可使用：

\[
k\in\{5,10,20\}.
\]

**主结果先固定 \(k=10\)，不允许通过测试标签搜索最优 \(k\)。**

高维空间可能存在距离集中问题，因此需要同时检查 score 分布、有效特征维度以及不同 \(k\) 的排序稳定性。

### 6.2 Representation AUPRC（主指标）

用测试标签评价 kNN score：

\[
\boxed{\mathrm{AUPRC}_{repr}}
\]

其优点：

- threshold-independent；
- 直接衡量 learned representation 是否使异常更远离参考正常模式；
- 可以跨模型/阶段比较，但不要求原始距离处于相同尺度。

**注意：** 这是附加的 *representation diagnostic*，不是原始 PESR reconstruction-based detector 的 AUPRC，不能偷换成模型原有的异常检测性能。

同时报告该测试集的 anomaly prevalence（随机排序时 AUPRC 的参考水平），避免在不同数据集间误解分数。

### 6.3 Representation AUROC（辅助指标）

\[
\boxed{\mathrm{AUROC}_{repr}}
\]

与 AUPRC 互补；若该实体/分组没有正负两类，则不计算。

### 6.4 kNN 距离分布与分离度

对真实正常/异常测试样本的 \(s_k\)：

\[
D_N=\mathbb{E}_{i:y_i=0}[s_k(z_i)],\qquad
D_A=\mathbb{E}_{i:y_i=1}[s_k(z_i)].
\]

可报告：

\[
\mathrm{RSR}=\frac{D_A}{D_N+\varepsilon}.
\]

并可进一步报告标准化分离度：

\[
\mathrm{Sep}_{score}
=
\frac{D_A-D_N}
{\sqrt{(\sigma_A^2+\sigma_N^2)/2}+\varepsilon}.
\]

这里 \(\sigma_N,\sigma_A\) 是**kNN 距离分数**的标准差，而不是原始 embedding 的坐标方差。

解读原则：

- \(D_A>D_N\) 及更大的 \(\mathrm{Sep}_{score}\) 表示异常通常更远离训练参考；
- 但 **RSR、Sep 只能作为辅助指标**，因为分布重叠、长尾与异常率可能使均值差误导；
- **AUPRC/AUROC 是主要评价依据**。

### 6.5 正常样本紧凑度（可选）

仅在同一模型/特征空间内部，评估正常样本的局部结构，例如正常测试表示到训练参考 bank 的平均 kNN 距离：

\[
C_N=D_N.
\]

不建议将所有正常表示强制视为单个高斯簇，并直接以“到一个全局 centroid 越近越好”作结论，因为工业系统存在多个正常工况。

---

## 7. 实验 A：消融模型之间的表示可分性

### 7.1 设置

模型：

\[
\mathrm{w/o\ MPA},\ 
\mathrm{w/o\ MSSA},\ 
\mathrm{PESR}.
\]

对于每个模型：

1. 加载对应 checkpoint；
2. 抽取相同功能位置的 pre-CRR feature；
3. 用训练特征建立对应 reference bank；
4. 对固定真实 test windows 计算 kNN score；
5. 使用真实测试标签计算 AUPRC/AUROC；
6. 统计 \(D_N,D_A,\mathrm{Sep}_{score}\)。

### 7.2 输出表格模板

| Dataset | Model | repr-AUPRC ↑ | repr-AUROC ↑ | \(D_N\) | \(D_A\) | Sep ↑ |
|---|---|---:|---:|---:|---:|---:|
| MSL | w/o MPA | — | — | — | — | — |
| MSL | w/o MSSA | — | — | — | — | — |
| MSL | PESR | — | — | — | — | — |
| SWaT | w/o MPA | — | — | — | — | — |
| SWaT | w/o MSSA | — | — | — | — | — |
| SWaT | PESR | — | — | — | — | — |

报告训练 seed 间 mean ± std；不同变体原始距离绝对值仅用于各自分布描述，**不直接用 \(D_N\) 绝对值跨变体排名**。

### 7.3 解读

若 PESR 的 repr-AUPRC/AUROC 更好：

> 说明其整体结构学习到了更有利于区分真实正常与异常状态的特征。

若某个消融版本更好：

> 不能宣称所有模块均提高表示可分性；应分析该模块是否主要通过下游重建评分、时序关联或训练正则化改善最终任务指标。

---

## 8. 实验 B：PESR 内部逐阶段表示演化

### 8.1 设置

仅使用完整 PESR。

依次抽取：

\[
Z_p \rightarrow V_t \rightarrow \mu_t.
\]

各阶段**独立建立**训练 reference bank，独立拟合训练侧特征标准化参数，使用相同测试样本与标签评价。

### 8.2 输出表格模板

| Dataset | Stage | Dim | repr-AUPRC ↑ | repr-AUROC ↑ | Sep ↑ |
|---|---|---:|---:|---:|---:|
| MSL | MPA output \(Z_p\) | — | — | — | — |
| MSL | MSSA output \(V_t\) | — | — | — | — |
| MSL | CRR encoder mean \(\mu_t\) | — | — | — | — |
| SWaT | MPA output \(Z_p\) | — | — | — | — |
| SWaT | MSSA output \(V_t\) | — | — | — | — |
| SWaT | CRR encoder mean \(\mu_t\) | — | — | — | — |

### 8.3 解读边界

**不要求：**

\[
\mathrm{AUPRC}(Z_p)
<
\mathrm{AUPRC}(V_t)
<
\mathrm{AUPRC}(\mu_t).
\]

因为 CRR 的目的首先是为重建提供潜在分布，**并非显式优化 kNN separability**。如果 \(\mu_t\) 的 kNN 可分性下降，并不自动说明 CRR 无效；可进一步结合最终原生 anomaly score 讨论，但不能把这个结果包装成“逐层提升”。

---

## 9. 实验 C：可视化

### 9.1 主图建议

**Figure A — Ablation-level representation comparison**

三列：

1. w/o MPA；
2. w/o MSSA；
3. PESR。

每列展示同一个数据集、同一批真实窗口的 pre-CRR 特征。

**Figure B — Stage-wise representation evolution**

三列：

1. \(Z_p\)；
2. \(V_t\)；
3. \(\mu_t\)。

分类标注：

- Normal train / nominal train；
- Normal test；
- Anomalous test。

### 9.2 推荐可视化顺序

1. **优先 PCA（2D）**：直观且确定性更强；
2. **可选 t-SNE**：作为补充展示局部邻域结构；
3. **强烈推荐 kNN score distributions（KDE/violin/ECDF）**：最直接对应 repr-AUPRC 与 normal–anomaly separability。

注意：

- 同一组图固定样本索引与抽样规则；
- t-SNE/PCA 不使用标签作为优化目标；
- 不依靠不同面板的二维绝对距离判断哪个模型在原始高维空间更分离；
- 标明特征空间不同，二维投影结果只作辅助解释；
- 对异常样本和正常样本采用**预先固定的可视化抽样比例**，并注明可视化分布不等于真实测试集类别比例；
- 不能从单张“看起来分得开”的 t-SNE 图得出强的统计结论。

### 9.3 正文最推荐的图

若版面有限，优先选：

**一张多面板图：**

- (a) 三种消融模型的 repr-AUPRC 柱图或点图（mean ± std）；
- (b) 完整 PESR 的逐阶段 repr-AUPRC；
- (c) PESR 与关键 ablation 的 normal/anomaly kNN distance distributions。

二维 PCA/t-SNE 放附录或替换 (c)，根据真实结果决定。

---

## 10. 重复性与运行控制

建议每个 model variant 至少：

\[
n_{\mathrm{seed}}=3.
\]

对每个 seed：

- 使用对应 seed 的训练模型；
- 训练/test split、window 参数、样本 index 固定；
- reference bank 抽样 seed 固定；
- 同一个 seed 下不同模型用相同索引；
- kNN 的 \(k\) 和预处理规则固定。

如果现阶段只有一个 seed 的 checkpoint，先做 **single-seed feasibility check**，不能将其当成具有统计显著性的最终证据。

对结果报告：

- seed mean ± std；
- 每个数据集/实体的有效正负样本数量；
- `k=5,10,20` 的简短敏感性结果；
- 如数据量允许，给出实体级或攻击事件级 bootstrap 置信区间，避免对高度自相关的相邻时间窗进行天真的独立采样。

---

## 11. 必须通过的有效性检查（Gate）

### Gate 0：Forward feature 是否真实有效

先检查每一层：

- shape、dtype；
- 是否存在 NaN/Inf；
- 每维/整体 variance；
- 重复 forward 的一致性；
- 真实异常与正常输入是否得到非恒定表示；
- `window index ↔ label` 是否严格对应。

此前记录显示**最终异常分数近似常数**。这并不等于所有内部特征也近似常数，所以必须先抽取内部 feature 核验；**如果抽出的表示也接近常数，则暂停分析，先排查提取接口或网络数值行为。**

### Gate 1：Reference bank 与 kNN 是否正常

在任意一个数据集和 PESR checkpoint 上检查：

- bank size 足够大；
- 特征预处理没有用到 test statistics；
- `k` 小于 bank size；
- 训练正常样本的邻居距离与测试 score 有合理变化；
- 没有重复时间窗或实体身份造成的明显检索捷径。

### Gate 2：标签与指标是否有效

- 有正负标签；
- test label 粒度与 embedding 一致；
- AUPRC > 0 且是按**正确方向（越异常 score 越高）**计算；
- 按预设指标输出，不事后反转 score 方向以取更高结果；
- 结果与测试 prevalence 一起报告。

只有 Gate 0–2 全部通过才扩展所有模型/数据集。

---

## 12. 最终交付物

建议实验输出：

```text
representation_analysis/
├── config/
│   └── experiment_config.yaml
├── features/
│   ├── dataset_model_seed_train.npz
│   └── dataset_model_seed_test.npz
├── metrics/
│   ├── model_comparison.csv
│   ├── stage_comparison.csv
│   └── knn_sensitivity.csv
├── figures/
│   ├── ablation_repr_auprc.pdf
│   ├── stage_repr_auprc.pdf
│   ├── score_distributions.pdf
│   └── pca_or_tsne.pdf
└── report/
    ├── validity_checks.md
    └── findings.md
```

`npz` 至少记录 feature、window indices、entity IDs（如有）、labels（仅测试侧）、feature stage、seed、pooling 配置。不得假设上述目录/代码已经存在；这是建议产物结构。

---

## 13. 结果解释与决策规则

| 观察结果 | 可以写入论文的结论 | 不能写的结论 |
|---|---|---|
| PESR repr-AUPRC 高于消融版本 | MPA/MSSA 的组合改善了正常/异常表示判别能力 | 证明了训练污染被消除 |
| PESR pre-CRR 好，但 \(\mu_t\) 一般 | 主要收益可能集中在多尺度表征阶段 | CRR 没有价值 |
| \(\mu_t\) 比 \(V_t\) 更好 | CRR 编码后的表示更利于该 kNN 诊断 | CRR 在所有下游机制上更优 |
| 表示 AUPRC 接近，但原始 detector F1 不同 | 最终判别差异可能依赖重建评分与阈值 | 表示没有任何价值 |
| 全部表示近乎常数或无可分性 | 需检查特征接口、训练状态和异常打分机制 | 为符合叙事挑选少数漂亮图 |

**禁止根据结果选择“最能证明模块有用”的指标或数据集而不报告其余已运行设置。**

---

## 14. 对论文和回复信的落地方式

### 14.1 推荐正文位置

**Suggested location:** `Section 4 (Experiments), after Ablation Study`，新增小节：

> **Representation Discriminability Analysis**

采用：

1. 模块级 repr-AUPRC/AUROC 定量对比；
2. 完整 PESR 的逐阶段表示变化；
3. 分布图/PCA，解释真实正常与异常的空间关系。

### 14.2 Introduction 需要同步收紧

建议把原稿中类似：

> learn normal patterns without the impact of anomalies

改为：

> **learn stable and discriminative representations of normal temporal behavior, while reducing sensitivity to potentially anomalous observations**

但必须注意：**如果只做本 PRD，不应继续宣称已经定量证明“训练污染如何伤害正常模式学习”。**

更稳妥的正文动机英文草稿：

> Learning reliable normal patterns from multivariate time series is challenging because industrial observations exhibit complex temporal dependencies and may contain local deviations that are difficult to distinguish from normal state variations. Reconstruction-based detectors rely on the representations learned from these observations, and insufficiently discriminative representations can reduce the separation between normal and anomalous states. PESR therefore combines multi-view patching alignment and multi-scale temporal modeling to construct more informative representations before reconstruction.

这是一段**拟议修改**，不代表实验结果已经支持全部效果表述。

### 14.3 Response Letter 的证据边界

对 **R1**：

- 可以回应：我们收紧了过强的 contamination claim，并改用真实异常的表示可分性进行定量分析；
- 但不能声称：已用该实验直接测量了“训练污染比例对正常模式学习的因果影响”；
- 因为 editor 也强调 contamination impact，**仅凭本实验依然存在二审追问风险**。如需完全关闭 R1，仍应考虑对原主张做实质性弱化，或者另给真正的 train-contamination 证据。

对 **R2**：

- 本实验能解释 MPA/MSSA 如何改善重建前表示，以及与最终消融性能的联系；
- 不能单凭该实验证明 PESR 已普遍克服所有 conventional reconstruction-based methods 的局限。

### 14.4 与已有实验的一致性

- 原有主结果：仍报告最终 detector 指标；
- 原有 Ablation：仍负责验证模块对最终任务的贡献；
- 本实验：提供 feature-space mechanism evidence；
- 原有随机替换 robustness：可以保留，但明确其是 corruption tolerance，不把平曲线解释成训练污染机制的证据；
- 如发现最终 anomaly score 几乎常数，应对检测评估 pipeline 单独检查，不能用 repr-kNN AUPRC 掩盖原生 detector 的潜在问题。

---

## 15. 最小可执行版本（MVP）

第一轮先跑：

1. **数据集：** MSL + SWaT；
2. **模型：** w/o MPA、w/o MSSA、PESR；
3. **比较位置：** 统一 pre-CRR representation；
4. **指标：** repr-AUPRC（主）、repr-AUROC、normal/anomaly kNN score distributions；
5. **逐阶段：** 完整 PESR 的 \(Z_p,V_t,\mu_t\)；
6. **图：** 两组 AUPRC 对比 + 一张代表性 PCA（必要时增加 score KDE）；
7. **先做：** 单 seed 有效性检查；
8. **正式报告：** 至少 3 seeds + kNN \(k\) 敏感性检查。

**成功标准：** 获得一个方法学成立、可复现、无需测试标签参与模型或 kNN 调参的 representation analysis；结果是否支持各项假设由真实数据决定。不能把“必须出现 PESR 第一、逐层递增”作为运行成功标准。

---

## 16. 一句话结论目标

> **The representation-level analysis is designed to determine whether the complete PESR learns more discriminative normal-pattern representations than its ablated variants and to examine how this discriminability evolves across MPA, MSSA, and the CRR encoder, using naturally occurring anomalies and label-independent normal-reference kNN scores.**

实验完成后，才依据真实结果决定是否能够写：

> *These results suggest that the integrated multi-view and multi-scale modeling contributes to more discriminative representations before reconstruction.*

**不将上述结论偷换成已经证明训练异常污染被直接缓解。**
