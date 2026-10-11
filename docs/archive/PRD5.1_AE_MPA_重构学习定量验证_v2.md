# PRD v2｜普通 AE 的重构污染分析与 MPA 模块增益验证

**定位：** PESR Major Revision 的轻量补充实验  
**对应意见：** R1 — Unclear motivation；R2 — Limitations of reconstruction-based methods  
**核心变化：** 用 **Window-based AE** 替换上一版 LSTM-AE；**MPA 只需证明能够改善重构学习**，不要求必须改善污染鲁棒性、F1 或 kNN 可分性。  
**状态：** 实验计划，未预设任何结果。

---

## 1. 目标与论文论证边界

将原来试图通过单个实验回答的多个问题拆开：

### 实验 A｜问题是否存在（Motivation）

在同一普通 AE 上，对比正常训练与含扰动/异常训练，观察**固定正常验证集**的重建误差是否改变。

- 主要量：正常验证 MSE \(E_N\)。
- 可选解释：固定异常探针集 MSE \(E_A\)。
- 不要求引入 PESR，也不依赖 F1。

**证据边界：** 用随机噪声只能证明 *training corruption/noise* 的影响；若要直接回应 reviewer 所说的 *anomalous observations*，至少需要一个用真实异常窗口构造的污染对照。

### 实验 B｜MPA 是否有重建增益（Module validation）

比较 **AE** 与 **MPA+AE**，使用**相同的原始输入重建任务**：

\[
X \xrightarrow{\rm AE}\hat X ,
\qquad
X\xrightarrow{\rm MPA}\xrightarrow{\rm AE}\hat X.
\]

**成功条件不再设为：** MPA 必须使污染曲线下降更慢、异常 MSE 更高或 AUPRC 更高。

**真正要验证：** 相对于 AE，在相同数据条件下加入 MPA 是否能**降低固定正常验证集的重建 MSE**，从而表明多视角表示能够为下游重构提供更有用的信息。

> 这仅支持“MPA 可改善该 AE 重构框架的正常模式建模”。不能据此直接宣称“MPA 抑制了异常模式学习”或“完整 PESR 的污染鲁棒性得到证明”。

---

## 2. 最小实验矩阵

### 主实验（推荐）

| Model | Clean train（0%） | Corrupted train（10%） | Corrupted train（20%） |
|---|:---:|:---:|:---:|
| Window-based AE | ✓ | ✓ | ✓ |
| MPA + Window-based AE | ✓ | ✓ | ✓ |

- 数据集：**SWaT**。
- 主指标：固定 **clean-normal validation MSE**。
- 可选：固定异常探针 MSE；不要求计算 F1。
- Seed：先做 1 个做可行性检查；正式 **3 seeds**。
- 总量：**2 模型 × 3 污染水平 × 3 seeds = 18 次训练**。

**更省时版本：** 只保留 0% / 10%，即 **12 次训练**。如果希望画随比例变化的曲线，再扩展 20%。不为了强行得到退化曲线而增加噪声强度或试验大量比例。

### 污染来源分两级处理

**一级（低成本主实验）：** Gaussian additive noise，测 *corruption sensitivity and reconstruction quality*。

\[
x'_{t,j}=x_{t,j}+\alpha\sigma_j\epsilon_{t,j},
\qquad\epsilon_{t,j}\sim\mathcal N(0,1).
\]

- \(\rho\)：受到扰动的原始训练时间戳比例，例如 0/10/20%；
- \(\alpha\)：噪声强度，预先固定（先以 \(\alpha=1\) 为实现候选，检查其可观察性后记录；不依据结果反复调到显著）；
- \(\sigma_j\)：从 clean training data 得到的连续变量标准差；
- 只扰动连续传感器变量；离散/二值状态不添加高斯噪声；
- 噪声只加在训练集，正常验证集与测试/探针集保持不变；
- 0/10/20% 使用嵌套的扰动位置，seed 相同的 AE 和 MPA+AE 使用完全相同数据。

**二级（若要直接支撑 R1 的 anomaly contamination）：** 最小真实异常对照 **0% vs 10%**。

- 从 SWaT **独立的真实攻击事件**选取 donor windows，替换正常训练窗口；
- 固定正常验证集及与 donor 事件不重叠的评价/异常探针集；
- 这一级可以只先跑普通 AE 两组。如果 AE 已呈现明确的影响，按资源情况再运行 MPA+AE；
- 与高斯噪声协议分开命名、分开解释；不要把 Gaussian noise 结果称作“真实异常污染证明”。

**注意：** 噪声实验中的 \(\rho\) 是**受扰动时间戳比例**，真实异常替换中的 \(\rho\) 是**替换训练窗口比例**。两者不是同一个统计量，不能混入同一张污染比例曲线。

---

## 3. 数据处理与防泄漏

1. 从正常数据中划分 `train_normal` 与 **固定** `val_normal`；二者时间段互不重叠，跨边界避免滑窗重合。
2. 只在 `train_normal` 上拟合 normalization scaler；所有训练条件复用同一 scaler。
3. 先按原始时间戳添加高斯噪声，再执行与 clean setting 相同的 windowing；不可让验证数据参与噪声生成。
4. 所有配置使用一致的 window length、stride、输入变量集合及 validation indices。
5. `normal_probe` 与 `anomaly_probe` 仅用于评价或作图，不参与梯度更新。
6. 若采用真实异常 donor，**按攻击事件/连续时间段隔离** donor 与评价数据，不能随机拆分重叠窗口产生数据泄漏。
7. 检查正常验证集的输入、目标均为 **clean original data**，不能把有噪输入当作“正常质量”评价标准。

---

## 4. 模型设置

### 4.1 Baseline：Window-based AE

输入一个完整多变量窗口：

\[
X\in\mathbb R^{T\times C},\qquad \hat X\in\mathbb R^{T\times C}.
\]

可用一个**简单的 MLP autoencoder**：

\[
\operatorname{flatten}(X)\to \operatorname{Encoder}\to z
\to\operatorname{Decoder}\to \hat X .
\]

建议保持 2–3 层左右的轻量结构，使用固定的 latent bottleneck，避免近乎恒等映射导致无意义的重建结果。隐藏层、latent 维度先固定并在实验配置中记录，不能根据 test labels 反复调参。

重建损失：

\[
\mathcal L_{\mathrm{rec}}
=
\frac1{TC}\|X-\hat X\|_F^2.
\]

使用**同一个重建目标和计算口径**评价所有模型。

### 4.2 Improved：MPA+AE

\[
X \xrightarrow{\rm MPA} H
\xrightarrow{\rm Adapter} H'
\xrightarrow{\rm AE\ Encoder} z
\xrightarrow{\rm AE\ Decoder}\hat X .
\]

实施要求：

- 复用论文已有 **MPA 的 patching、多视角 embedding 和对齐思路**，不引入 MSSA、CRR 或新的专用模块；
- 只允许加入必要的维度适配层，使输入 AE encoder 的维度明确、可比；
- **保持同样的原始输入 \(X\) 重建目标**，不能让 MPA+AE 改成“重建 MPA 自己的输出”；
- 两模型使用相同的 AE encoder/decoder **结构定义**和 latent 大小，分别独立训练；
- 若保留 MPA 的一致性损失 \(L_a\)，记录
  \[
  \mathcal L_{\mathrm{MPA+AE}}=\mathcal L_{\mathrm{rec}}+\lambda L_a.
  \]
  这时只能归因于**完整 MPA（多视角提取+一致性约束）的联合效果**，不能单独归因于 patching；
- 记录额外参数量及 adapter 结构，避免把增加容量的效果误判为纯粹的多视角机制收益。

**优先采取不改变原 AE decoder、原始重建目标和 latent bottleneck 的接入方案。** 若原 MPA 输出无法无损适配，不要强行随意 reshape；先做 shape 与语义检查，再确定轻量 adapter。

### 4.3 统一训练设置

- 相同 seed 配对比较；
- 固定 epochs、optimizer、learning rate、batch size、窗口配置；
- 训练过程中只用训练集更新权重；
- 不使用 test anomalies 做模型选择；
- 主表最好以**统一的固定训练轮数**报告；正常验证 loss 可用于监测而非针对不同模型采取不一致的选择策略；
- 推理统一使用 `model.eval()` 与确定性 forward，禁止依赖 dropout 的随机输出。

---

## 5. 指标设计：以重建质量为核心

### 5.1 主指标：正常数据重建 MSE

对固定、未参与训练的正常验证集 \(\mathcal D_N^{val}\)，定义：

\[
\boxed{
E_N(M,\rho)=
\frac{1}{|\mathcal D_N^{val}|}
\sum_{X\in \mathcal D_N^{val}}
\frac{1}{TC}\left\|X-\hat X_{M,\rho}\right\|_F^2
}
\]

其中 \(M\in\{\mathrm{AE},\mathrm{MPA+AE}\}\)。

**MPA 重建改进量：**

\[
\boxed{
G_{\mathrm{MPA}}(\rho)
=
\frac{E_N(\mathrm{AE},\rho)-E_N(\mathrm{MPA+AE},\rho)}
{E_N(\mathrm{AE},\rho)+\varepsilon}
}
\]

若 \(G_{\mathrm{MPA}}(\rho)>0\)，表示相同 \(\rho\) 条件下，加入 MPA 后正常验证 MSE 更低。

> **这是实验 B 的核心指标。** 不要求 MPA 的污染退化斜率必须更小；只要在清洁数据或污染训练条件下存在稳定、可重复的正常重建质量提升，就可以报告 MPA 对该重构骨干的增益（需要明确是哪种条件）。

### 5.2 辅助指标：AE 的污染影响

\[
\boxed{
\Delta E_N(\rho)=
\frac{E_N(\mathrm{AE},\rho)-E_N(\mathrm{AE},0)}
{E_N(\mathrm{AE},0)+\varepsilon}
}
\]

- \(>0\)：普通 AE 的正常验证重建质量在当前扰动协议下变差；
- \(\approx0\)：未观察到明显影响；
- \(<0\)：观察到改善，可能存在正则化作用；不得写成污染损害了学习。

### 5.3 可选指标：固定异常 probe 重建 MSE

\[
E_A(M,\rho)=
\mathbb E_{X\in\mathcal D_A^{probe}}
\left[\frac1{TC}\|X-\hat X_{M,\rho}\|_F^2\right].
\]

用于观察是否更容易重构异常，以及解释不同训练条件下的学习行为。

**不要要求** MPA 必须让 \(E_A\) 增大；这并非“提高正常重建能力”的必要条件。

如果方便，还可报告 \(E_A-E_N\) 或重建误差 AUROC 作为诊断，但不是本 PRD 的交付必要项。

---

## 6. Loss 曲线与可视化（只做一张图）

### Figure：Normal Reconstruction Learning Curve

每个 epoch 后，在**同一固定 clean-normal validation set** 上评估：

\[
e\longrightarrow E_N^{(e)}.
\]

主图最简版本：

- 横轴：epoch；
- 纵轴：clean-normal validation MSE；
- 四条曲线：AE(0%)、MPA+AE(0%)、AE(10%)、MPA+AE(10%)；
- 如增加 20%，建议将曲线放附录或仅放结果表，避免主图拥挤。

**不必做：**

- per-sample \(m_x,v_x\) loss dynamics；
- t-SNE/PCA；
- kNN representation-AUPRC；
- F1 与 oracle threshold 搜索。

这次实验旨在验证**正常数据重建能力**，不以构造额外的 anomaly detector 为前提。

---

## 7. 主文表格模板

**Table X. Reconstruction quality of AE with and without MPA under controlled training corruption.**

| Training setting | Normal MSE — AE ↓ | Normal MSE — MPA+AE ↓ | MPA gain \(G_{\mathrm{MPA}}\) ↑ |
|---|---:|---:|---:|
| Clean (0%) | 待填 | 待填 | 待填 |
| Gaussian corruption (10%) | 待填 | 待填 | 待填 |
| Gaussian corruption (20%) | 待填 | 待填 | 待填 |

- 报告 3 seeds 的 mean ± std；
- gain 按 seed 配对计算后求均值，避免先求各模型均值再计算导致信息损失；
- 若另做真实异常窗口污染，单列一行或单列一个小表，**明确区分污染协议及比例定义**；
- 额外报告两模型参数量，可放 caption 或短句。

---

## 8. 执行流程（最省成本）

### Phase 0 — 预检（先跑 2 次）

1. SWaT clean training + fixed clean-normal validation；
2. 训练普通 AE(clean) 和 MPA+AE(clean)，各一个 seed；
3. 确认 AE 有合理瓶颈、预测与输入相关，验证 MSE 不为 NaN/Inf；
4. 确认 MPA 接口正确、重建目标始终为原始 \(X\)；
5. 比较 clean MSE，检查是否有可观察的重建质量差异。

**预检通过条件是实现正确、指标有效，而不是 MPA 必须更优。**

### Phase 1 — 污染影响（优先跑 10%）

同一个 seed：

- AE(10%)；
- MPA+AE(10%)。

如果需要三点曲线，再增加两个模型的 20%。

### Phase 2 — 正式结果

- 先完成每个正式配置 3 seeds；
- 对 0%、10%、20% 汇总 normal MSE；
- 导出主表与 validation-loss curve；
- 如正文仍需直接回应“anomalous contamination”，补最小的**真实异常 10%**协议，不以 Gaussian 替代。

---

## 9. 结果如何解释

| 观察结果 | 可以写进论文的结论 | 不应写的结论 |
|---|---|---|
| AE 噪声训练后 \(E_N\) 上升 | 该 AE 的正常重建受到训练噪声干扰 | 真实工业攻击必然具有相同影响 |
| AE 真实异常训练后 \(E_N\) 上升 | 该 AE 在该异常污染设置下正常模式建模受损 | 所有 reconstruction models 都必然如此 |
| MPA+AE 的 \(E_N\) 更低 | MPA 在该 AE 框架中改善正常模式重建质量 | MPA 已证明可消除异常污染 |
| MPA+AE clean 条件更好，但污染退化一样 | MPA 有干净数据重建收益 | MPA 有更强污染鲁棒性 |
| MPA+AE 在污染条件的 \(E_N\) 更低，但退化幅度相近 | 污染条件下仍保有重建收益 | 污染影响被专门抑制 |
| AE/MPA+AE 没有稳定差异 | 当前实验没有验证 MPA 的重建增益 | 可以只展示某次有利 seed 充当普遍效果 |

**统一原则：** 模型效果以重复实验为准。不能将“正常重建更好”自动等同于“异常检测更好”：在重构异常检测中，异常也被很好重构时反而可能降低判别能力。

---

## 10. 论文放置位置与写法

**Suggested location:** `Section 4, after Ablation Study`  
**Suggested subsection:** *Reconstruction Learning under Perturbed Training Observations*

### 英文实验设置（仅描述实施方案）

> To examine the sensitivity of reconstruction learning to corrupted training observations, we conduct a lightweight controlled study using a window-based autoencoder (AE). We evaluate the reconstruction error on a fixed, clean normal validation set under clean and perturbed training conditions. This design measures whether training perturbations affect the characterization of normal temporal patterns without relying on threshold-dependent detection metrics.
>
> We further incorporate the proposed Multi-view Patching Alignment (MPA) module into the same AE reconstruction backbone while retaining the original-input reconstruction objective and decoder architecture. Both variants are trained and evaluated using identical data partitions and preprocessing settings. By comparing their normal-data reconstruction errors, we assess whether MPA provides useful multi-view representations for downstream reconstruction.

**如果做的只有 Gaussian 噪声：** 必须用 *corrupted/perturbed observations* 描述，不能写成真实 anomaly contamination。

**如果新增真实异常 donor 窗口实验：** 可增加一段独立叙述其事件隔离与替换规则，使其对应 reviewer 的 *anomalous observations*。

### 与审稿意见的对应关系

- **R1：** 用 AE(0%) 与 AE(污染条件) 的正常验证 MSE 说明当前协议下是否出现重建质量退化；如只是 Gaussian，证据仅覆盖 noise/corruption，不足以完全替代真实异常污染分析。
- **R2：** 用 AE vs MPA+AE 的正常验证 MSE 比较，验证 MPA 模块对一个通用重构框架是否有**可迁移的重建收益**；无需额外证明 MPA 是特定的抗污染模块。

### Response Letter 写作边界

实验得到结果后再填写数字，不预先写：

- “normal learning is significantly degraded”；
- “MPA prevents anomaly fitting”；
- “MPA consistently improves detection performance”。

如果只是观察到 MPA 在 clean/perturbed setting 下降低正常 MSE，宜写：

> *Adding MPA improves the reconstruction quality of normal validation samples under the evaluated conditions.*

这比没有直接证据就宣称缓解污染更准确。

---

## 11. 最终交付清单

- [ ] 确认 SWaT normal train / normal validation 的无重叠切分；
- [ ] 固定 0%、10%（可扩 20%）的 Gaussian training corruption 协议；
- [ ] 实现普通 Window-based AE；
- [ ] 实现 AE+MPA，核实原始 \(X\) 为共同重建目标；
- [ ] 单 seed 快速检查，随后 3 seeds；
- [ ] 主表：\(E_N(\mathrm{AE}), E_N(\mathrm{MPA+AE}), G_{\mathrm{MPA}}\)；
- [ ] 一张 clean-normal validation MSE vs epoch 曲线；
- [ ] 参数量、seed、噪声强度/位置和预处理配置；
- [ ] 如果希望**直接**回应真实训练异常污染：增加隔离事件的真实异常 10% 小对照。

---

## 最终一句话

**用最简单的 AE 和 MPA+AE，在固定正常验证集上比较重建 MSE；训练噪声用于研究扰动影响，而 MPA 只需展示可重复的正常重建质量增益。真实异常污染若要作为 R1 的直接证据，应单独做最小对照，不与噪声证据混写。**
