# PESR 异常污染动机定量实验 PRD

## 1. 实验目的

本实验用于回应以下两类审稿意见：

- **R1：动机不明确。** 需要定量证明“异常观测会干扰正常模式学习”这一问题在实际系统中的影响，例如异常比例增加时检测性能如何退化，以及正常/异常样本的重建行为如何变化。
- **R2：传统重构方法局限性说明不足。** 需要更清楚地说明 PESR 如何缓解异常污染对重建学习的影响，并通过实验验证该设计的实际作用。

本实验的核心不是重新做一轮大规模 SOTA 对比，而是通过 **controlled training contamination** 构造可控的异常污染比例，并观察 PESR 及其模块消融变体在污染增强时的性能和重建行为变化。

核心问题：

\[
\rho \uparrow
\quad\Rightarrow\quad
\text{正常/异常重建行为如何变化}
\quad\Rightarrow\quad
\text{检测性能如何变化}
\]

其中，\(\rho\) 表示训练数据中的异常污染比例。

---

## 2. 核心研究问题

### Q1. 异常污染是否会显著影响模型性能？

随着训练异常污染比例增加：

\[
\rho \in \{0,1,5,10,20\}\%
\]

观察：

\[
\rho \rightarrow \mathrm{AUPRC}
\]

以及：

\[
\rho \rightarrow F1
\]

若性能整体随污染比例增加而下降，则说明训练数据中的异常观测确实会影响模型学习和最终异常检测能力。

---

### Q2. 污染具体改变了什么？

分别统计正常样本和异常样本对应的重建误差：

\[
E_N(\rho)
=
\mathbb{E}_{x\in \mathcal{D}_N}[e(x)]
\]

\[
E_A(\rho)
=
\mathbb{E}_{x\in \mathcal{D}_A}[e(x)]
\]

其中：

- \(E_N\)：正常样本平均重建误差；
- \(E_A\)：异常样本平均重建误差；
- \(e(x)\)：PESR 的重建误差或最终 reconstruction-based anomaly score。

进一步计算正常与异常之间的区分度：

\[
Gap(\rho)=E_A(\rho)-E_N(\rho)
\]

重点不是预设一定出现：

\[
E_N\uparrow
\]

而是观察污染是否造成以下任一种现象：

1. 正常模式学习被破坏：

\[
E_N\uparrow
\]

2. 模型逐渐学会重建异常：

\[
E_A\downarrow
\]

3. 两类样本之间的重建差异缩小：

\[
Gap\downarrow
\]

其中第 3 项是最直接的结果，因为 reconstruction-based anomaly detection 依赖正常和异常样本之间保持足够大的重建差异。

---

### Q3. MPA 和 MSSA 是否提高了污染条件下的稳定性？

比较完整 PESR 与其已有模块消融版本在不同污染比例下的曲线。

如果：

\[
|\Delta P_{\mathrm{PESR}}|
<
|\Delta P_{\mathrm{w/o\,MPA}}|
\]

以及：

\[
|\Delta P_{\mathrm{PESR}}|
<
|\Delta P_{\mathrm{w/o\,MSSA}}|
\]

则可以说明 MPA 和 MSSA 所构建的多视角、多尺度时序表示，使后续 CRR 在异常污染增加时仍能保持更稳定的重建与判别能力。

这里应使用：

> robustness / stability under training contamination

而不是：

> processing efficiency

后者属于 latency、memory、FLOPs 等效率实验。

---

## 3. 模型设置

### 3.1 主实验模型

建议使用以下 3 个核心版本：

| Model | MPA | MSSA | CRR | Purpose |
|---|---:|---:|---:|---|
| w/o MPA | ✗ | ✓ | ✓ | 验证多视角 patching 表示是否提高污染鲁棒性 |
| w/o MSSA | ✓ | ✗ | ✓ | 验证多尺度时序建模是否提高污染鲁棒性 |
| PESR | ✓ | ✓ | ✓ | 完整模型 |

所有模型均保留 **CRR**。

CRR 本身就是 PESR 的最终重建过程，因此本实验不额外将 CRR 替换为 AE。

本实验关注的是：

> 在相同 CRR 重建机制下，上游表示是否通过 MPA 和 MSSA 降低异常污染对正常模式学习的影响。

---

### 3.2 可选扩展

如果现有代码运行成本较低，可额外加入：

- w/o Mutual Attention
- w/o Global Attention
- w/o Local Correlation
- w/o Temporal State Tracker

但这些不是本实验的必要部分。

主文建议控制在：

- w/o MPA
- w/o MSSA
- PESR

以保持图表清晰，并避免把 motivation experiment 变成第二套完整 ablation study。

---

## 4. 数据集设置

### 4.1 推荐数据集

主实验使用：

- **SWaT**
- **WADI**

原因：

1. 两者均具有明确的工业控制系统背景；
2. 与论文的 IIoT 应用动机一致；
3. 可以从正常训练阶段出发构造可控异常污染；
4. WADI 本身是当前 PESR 提升最明显、结构最复杂的数据集之一，适合观察 contamination sensitivity。

MSL 和 SMAP 不作为该实验的主数据集，以避免其原始训练数据是否严格 clean 所带来的额外解释问题。

---

## 5. 数据污染构造

### 5.1 污染比例定义

定义：

\[
\rho=
\frac{\text{contaminated timestamps}}
{\text{total training timestamps}}
\]

推荐：

\[
\rho\in\{0,1,5,10,20\}\%
\]

其中：

- 0%：clean training reference；
- 1%：轻度污染；
- 5%：较低污染；
- 10%：中等污染；
- 20%：stress test。

污染比例在 **raw timestamp level** 上定义，而不是在 sliding-window level 上定义。

原因是一个异常时间点可能进入多个 overlapping windows，若按窗口统计会产生重复计数和解释歧义。

---

### 5.2 污染采用连续异常段

不建议随机独立替换单个时间点。

每个异常事件定义为：

\[
[t_s, t_s+L]
\]

并随机选择部分 continuous sensor channels 进行扰动。

这样更符合工业传感器异常通常具有时间持续性的特点，也与 PESR 的 temporal modeling 设计一致。

---

### 5.3 异常类型

建议控制在 3 类。

#### A. Level Shift

\[
x'_{t,j}
=
x_{t,j}+\alpha\sigma_j
\]

其中：

- \(\sigma_j\)：第 \(j\) 个变量在 clean training data 中的标准差；
- \(\alpha\)：异常强度。

用于模拟：

- sensor bias；
- systematic deviation；
- process-level shift。

---

#### B. Amplitude Scaling

\[
x'_{t,j}
=
\mu_j+\alpha(x_{t,j}-\mu_j)
\]

用于模拟正常动态范围异常放大或缩小。

---

#### C. Short-duration Perturbation

在较短连续区间内施加 spike 或 burst：

\[
x'_{t,j}
=
x_{t,j}+\delta_{t,j}
\]

用于模拟：

- 瞬时 sensor perturbation；
- 短时冲击；
- 局部异常事件。

---

### 5.4 变量类型约束

Synthetic contamination 优先只作用于 **continuous sensor channels**。

对于 binary/discrete actuator variables，不直接应用：

\[
x+\alpha\sigma
\]

避免产生物理上不合理的状态值。

如后续需要扩展，可单独设计：

- binary state flip；
- abnormal state persistence；
- delayed switching。

但不建议放入主实验。

---

## 6. Nested Contamination 设计

不同污染比例之间必须尽量采用 nested anomaly sets：

\[
\mathcal{A}_{1\%}
\subset
\mathcal{A}_{5\%}
\subset
\mathcal{A}_{10\%}
\subset
\mathcal{A}_{20\%}
\]

含义：

- 5% 中包含全部 1% 的异常；
- 10% 中包含全部 5% 的异常；
- 20% 中包含全部 10% 的异常。

这样随着 \(\rho\) 增加，主要变化是：

> 异常数量增加

而不是：

> 每个污染比例都重新生成一套难度不同的异常。

同时固定：

- anomaly type distribution；
- anomaly magnitude distribution；
- segment-length distribution；
- affected-channel distribution。

---

## 7. Preprocessing 控制

Scaler 仅在：

\[
\rho=0
\]

的 clean training data 上拟合一次。

得到：

\[
Scaler_0
\]

随后所有污染比例：

\[
\rho=0,1,5,10,20\%
\]

均使用同一个 \(Scaler_0\)。

原因：

如果每次重新计算 normalization statistics，则 contamination 同时改变：

1. training samples；
2. normalization statistics。

这样无法区分性能变化到底来自 abnormal-pattern learning 还是 normalization shift。

---

## 8. 实验控制变量

除 contamination ratio 外，所有实验保持一致：

- window length；
- batch size；
- optimizer；
- learning rate；
- training epochs；
- latent dimension；
- threshold protocol；
- preprocessing；
- anomaly-generation intensity；
- anomaly-duration distribution；
- model hyperparameters。

同一个 seed 下：

- PESR；
- w/o MPA；
- w/o MSSA；

使用 **完全相同的 contaminated training dataset**。

---

## 9. Random Seed 设置

建议：

\[
3\text{ seeds}
\]

每个 seed：

- anomaly locations 可变化；
- affected channels 可变化；
- model initialization 可变化。

但在同一 seed 内，不同 model variants 必须使用同一套污染数据。

最终报告：

\[
mean \pm std
\]

---

## 10. Detection-level Metrics

### 10.1 Primary Metric

推荐：

\[
\boxed{\mathrm{AUPRC}}
\]

原因：

1. 对 anomaly/normal imbalance 更合适；
2. threshold-independent；
3. 不会与 reviewer 对 optimal threshold search 的质疑混在一起。

---

### 10.2 Secondary Metric

使用：

\[
F1
\]

但建议采用 revision 后统一的 non-oracle threshold protocol。

本实验中 F1 主要作为与原论文结果体系保持一致的辅助指标。

---

## 11. Reconstruction-level Metrics

### 11.1 Normal Reconstruction Error

\[
E_N(\rho)
=
\frac{1}{|\mathcal D_N|}
\sum_{x\in\mathcal D_N}e(x)
\]

---

### 11.2 Anomaly Reconstruction Error

\[
E_A(\rho)
=
\frac{1}{|\mathcal D_A|}
\sum_{x\in\mathcal D_A}e(x)
\]

---

### 11.3 Reconstruction Gap

\[
Gap(\rho)
=
E_A(\rho)-E_N(\rho)
\]

核心解释：

- \(E_N\uparrow\)：正常模式学习被污染；
- \(E_A\downarrow\)：异常越来越容易被模型重建；
- \(Gap\downarrow\)：正常和异常之间的区分度下降。

---

### 11.4 Relative Reconstruction Change

因为不同 ablation 版本的 representation space 可能存在差异，不建议直接比较不同模型的绝对 reconstruction error。

建议对每个模型自身做归一化：

\[
R_N(\rho)
=
\frac{E_N(\rho)}{E_N(0)}
\]

\[
R_A(\rho)
=
\frac{E_A(\rho)}{E_A(0)}
\]

使得：

\[
R_N(0)=R_A(0)=1
\]

这样比较的是：

> contamination 相对于各模型 clean-training state 所引起的变化。

---

## 12. 可选 Separation Metric

为了比简单 \(E_A-E_N\) 更稳定，可以进一步计算：

\[
Sep(\rho)
=
\frac{\mu_A-\mu_N}
{\sqrt{(\sigma_A^2+\sigma_N^2)/2}}
\]

其中：

- \(\mu_A,\sigma_A\)：异常 score distribution；
- \(\mu_N,\sigma_N\)：正常 score distribution。

解释：

\[
Sep \downarrow
\]

表示正常与异常 reconstruction/anomaly-score distribution 越来越难区分。

若结果清晰，该指标可以作为正文中 Gap 的补充或替代。

---

## 13. 主要可视化

建议正文放一个 multi-panel figure。

### Figure (a): Detection Performance under Training Contamination

横轴：

\[
\rho
\]

纵轴：

\[
AUPRC
\]

曲线：

- w/o MPA
- w/o MSSA
- PESR

目的：

> 展示 contamination 增加后不同模型的性能退化速度。

---

### Figure (b): Reconstruction Behavior

横轴：

\[
\rho
\]

纵轴：

\[
R_N,\ R_A
\]

重点观察：

- normal reconstruction 是否恶化；
- anomalous reconstruction 是否变得更容易。

可以分别绘制 PESR 与 ablation，或只展示最有代表性的结果。

---

### Figure (c): Normal–Anomaly Separation

横轴：

\[
\rho
\]

纵轴：

\[
Gap
\]

或：

\[
Sep
\]

曲线：

- w/o MPA
- w/o MSSA
- PESR

目的：

> 直接观察随着 contamination 增加，正常和异常之间的 score separation 是否被压缩，以及完整 PESR 是否下降更慢。

---

## 14. 核心假设

### H1

随着训练污染比例增加，模型的 anomaly detection performance 会整体下降：

\[
\rho\uparrow
\Rightarrow
AUPRC\downarrow
\]

---

### H2

训练污染会导致正常/异常重建区分能力下降：

\[
\rho\uparrow
\Rightarrow
Gap\downarrow
\]

或：

\[
Sep\downarrow
\]

---

### H3

完整 PESR 的性能退化小于模块消融版本：

\[
|\Delta AUPRC_{\mathrm{PESR}}|
<
|\Delta AUPRC_{\mathrm{w/o\,MPA/MSSA}}|
\]

---

### H4

完整 PESR 的 reconstruction separation 更稳定：

\[
|\Delta Sep_{\mathrm{PESR}}|
<
|\Delta Sep_{\mathrm{w/o\,MPA/MSSA}}|
\]

若 H3 和 H4 同时成立，则可以较强地说明：

> MPA 和 MSSA 所构建的多视角、多尺度时序表示能够减轻异常污染对后续 CRR 重建学习的干扰，从而提高 PESR 在 contaminated training data 下的稳定性。

---

## 15. 结果判定原则

不要求所有曲线严格单调。

真实实验中可能出现：

\[
P(5\%) < P(10\%)
\]

但：

\[
P(20\%) > P(10\%)
\]

的小幅波动。

判断重点应放在：

1. 总体趋势；
2. mean ± std；
3. 高污染条件下相对 clean condition 的 degradation；
4. 不同 variants 的 degradation slope。

可以额外报告：

\[
\Delta P_{20}
=
\frac{P(0)-P(20)}
{P(0)}
\]

用于描述 20% contamination 下的相对性能损失。

---

## 16. 不建议的做法

### 16.1 不继续使用随机数 replacement 作为主实验

原有 robustness experiment 更适合解释：

> random corruption / noise robustness

而不是：

> anomaly contamination of reconstruction learning。

本实验应使用具有时间结构的 abnormal segments。

---

### 16.2 不需要大量外部 baseline

本实验的核心是：

> controlled mechanism analysis

而不是：

> SOTA benchmark。

外部模型之间：

- architecture 不同；
- reconstruction target 不同；
- score definition 不同；
- representation space 不同。

因此，即使它们在 contamination 下表现不同，也不容易定量说明 PESR 内部为什么更稳定。

如果后续希望增强 external validity，可以附加 1 个代表性 reconstruction baseline，但不是本 PRD 的必要项。

---

### 16.3 不把 MPA/MSSA 描述成“异常过滤器”

实验若成立，只能支持：

> MPA/MSSA improve robustness to contaminated observations through more stable contextual and multi-scale representations.

不能直接声称：

> MPA/MSSA remove anomalies.

---

## 17. 对 Reviewer Comment 的对应关系

### R1：Unclear Motivation

该实验提供：

\[
\rho
\rightarrow
AUPRC/F1
\]

以及：

\[
\rho
\rightarrow
E_N,E_A,Gap/Sep
\]

从而量化：

1. anomaly contamination 是否影响检测效果；
2. contamination 是否改变 normal/anomaly reconstruction behavior；
3. 正常和异常之间的区分度是否随着污染增强而下降。

---

### R2：Limitation of Conventional Reconstruction-based Methods

论文表述应避免简单声称：

> reconstruction itself is problematic.

更准确的表述应为：

> In unlabeled contaminated training sequences, anomalous observations also participate in representation learning and reconstruction optimization, which may distort the learned normal patterns or reduce the reconstruction-score separation between normal and anomalous observations.

PESR 的回答是：

> MPA 和 MSSA 首先构建更稳定的多视角、多尺度时序表示，随后由 CRR 完成重建。污染实验用于验证这种表示与重建流程在 contamination 增加时是否能够保持更稳定的 normal–anomaly separation 和 detection performance。

---

## 18. 最终推荐实验配置

| Item | Setting |
|---|---|
| Dataset | SWaT, WADI |
| Contamination Ratio | 0%, 1%, 5%, 10%, 20% |
| Injection Level | Raw timestamp level |
| Injection Pattern | Continuous anomalous segments |
| Anomaly Types | Level shift, amplitude scaling, short-duration perturbation |
| Channels | Continuous sensor channels |
| Contamination Design | Nested anomaly sets |
| Models | w/o MPA, w/o MSSA, PESR |
| Reconstruction | All variants use CRR |
| Seeds | 3 |
| Primary Metric | AUPRC |
| Secondary Metric | F1 |
| Reconstruction Metrics | \(E_N\), \(E_A\), \(Gap\) |
| Optional Metric | Standardized \(Sep\) |
| Normalization | Fit once on clean training data |
| Visualization | \(\rho\)-AUPRC, \(\rho\)-reconstruction behavior, \(\rho\)-separation curves |

---

## 19. 一句话实验结论目标

本实验最终需要支持的不是：

> PESR can completely eliminate the influence of anomalous observations.

而是：

> **As the training contamination ratio increases, PESR exhibits a smaller degradation in anomaly-detection performance and maintains a more stable separation between normal and anomalous reconstruction scores than its ablated variants, indicating that the multi-view and multi-scale representations improve the robustness of reconstruction learning against anomalous contamination.**
