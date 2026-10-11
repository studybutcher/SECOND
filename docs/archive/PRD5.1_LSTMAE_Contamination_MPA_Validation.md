# PRD｜训练异常对重构学习的影响与 MPA 模块验证

**版本：** v1.0（轻量版）  
**状态：** 待实验  
**对应审稿意见：** R1 — Unclear motivation；R2 — Limitations of conventional reconstruction-based methods  
**研究边界：** 本实验用于量化传统重构模型在异常训练下的行为，并验证 **MPA 模块**能否为通用重构骨干带来收益；不修改、重训或额外评价完整 PESR。

## 1. 实验目标

本 PRD 只回答两个问题，二者共用同一套数据、训练流程和指标。

- **实验 1（Motivation Validation）：** 在训练数据中混入真实异常窗口后，传统 LSTM-AE 对正常样本的重建质量和异常检测能力是否发生可测量变化？
- **实验 2（MPA Module Validation）：** 将已有 **Multi-view Patching Alignment（MPA）** 接入 LSTM-AE 前端后，是否能改善重建/检测质量，或减轻异常训练样本带来的退化？

**关键区分：** “MPA 在干净数据上提高 AUPRC”与“MPA 缓解训练污染影响”是不同结论，必须分别根据数据判断。不预设一定出现有利结果。

## 2. 最小 2×2 实验矩阵

| 模型 | Clean training（0%） | Contaminated training（10%） |
|---|:---:|:---:|
| **LSTM-AE** | ✓ | ✓ |
| **LSTM-AE + MPA** | ✓ | ✓ |

| 项目 | 固定设置 |
|---|---|
| 数据集 | SWaT |
| 污染来源 | 真实攻击事件的多变量时间窗口，不使用高斯噪声代替真实异常 |
| 污染比例 | 0%、10%（按训练窗口数定义） |
| 随机种子 | 3（先用 1 个 seed 做可行性检查） |
| 正式训练总数 | **2 模型 × 2 条件 × 3 seeds = 12 runs** |
| 主指标 | 固定正常验证集重建 MSE、独立测试集 AUPRC |
| 辅助分析 | 每 epoch 的正常验证 MSE 曲线；可选异常探针 MSE |
| 论文产出 | **一张表 + 一张图 + 简短结果讨论** |

10% 是预先规定的实验水平。如果没有明显下降，如实报告，不继续调大比例直到出现期望曲线。

## 3. 数据协议（最关键）

### 3.1 四种数据角色

1. **正常训练基础集** $D_{\mathrm{train}}^0$：SWaT 的正常训练窗口。
2. **真实异常 donor 集** $A_{\mathrm{donor}}$：从完整真实攻击事件中抽取窗口，仅用于污染训练。
3. **固定正常验证集** $D_N^{\mathrm{val}}$：与训练区间不重叠的正常窗口，不参与训练。
4. **固定 evaluation test**：包含正常与真实异常；其攻击事件不能与 donor 攻击事件重叠。

**防泄漏：** 按**攻击事件/连续时间段**拆分 donor 和测试，而不是随机划分高度重叠的滑动窗口。跨边界保留至少一个窗口长度的隔离区，避免同一事件同时用于训练与评价。如果原始 SWaT 划分不能提供独立 donor 和 test，需要先重新按事件划分，再确定整个实验方案；不可直接将最终测试窗拷回训练后仍用原测试集评价。

### 3.2 两种训练条件

Clean：

$$D_{\mathrm{train}}^{0}$$

Contaminated：从正常训练窗口中替换 **10%** 为 donor 异常窗口：

$$
D_{\mathrm{train}}^{10}
=(D_{\mathrm{train}}^{0}\setminus R)\cup A_{\mathrm{donor}},
\quad |R|=|A_{\mathrm{donor}}|=0.1|D_{\mathrm{train}}^{0}|.
$$

这样两套训练数据的**总窗口数相同**。污染率明确为 *training-window contamination ratio*；不与按原始时间戳定义的污染比例混用。

- 两模型同 seed 使用**完全相同**的污染窗口及正常替换索引。
- 从正常基础训练数据拟合 scaler 一次，全部配置复用。
- 固定窗口长度、stride、标签对齐、batch size、训练轮数、学习率等。
- donor 窗口保留完整多变量时序结构，不进行人为独立通道加噪。
- 如果 donor 异常事件不足以支持 10% 的合理窗口覆盖，不通过大量复制近似重复窗口凑比例；应在实验前统一改用更低固定比例并说明。

## 4. 模型设置

### 4.1 LSTM-AE（Baseline）

$$X\longrightarrow \mathrm{LSTM\ Encoder}
\longrightarrow \mathrm{Decoder}\longrightarrow \hat X.$$

训练目标统一为**原始输入重建**：

$$L_{\mathrm{rec}}=\mathrm{MSE}(X,\hat X).$$

选定一个可复现、能够正常收敛并产生非恒定重建分数的 LSTM-AE 实现；固定代码版本。

### 4.2 LSTM-AE + MPA（模块验证）

$$X\longrightarrow\boxed{\mathrm{MPA}}
\longrightarrow\mathrm{LSTM\ Encoder}
\longrightarrow\mathrm{Decoder}\longrightarrow\hat X.$$

只在原 LSTM-AE 前端接入论文**已有 MPA 模块**，保持 LSTM 主体、decoder 和**原始输入 $X$** 的重建目标不变。

**工程前置检查：** MPA 输出必须保留可解释的时间步顺序，才能作为 LSTM 输入；如需投影到 LSTM 预期维度，只使用最小维度适配层，并记录它的参数量。不得将多尺度维度错误地当成时间维度。

如果按原 MPA 设计保留跨视角对齐损失：

$$L_{\mathrm{MPA+AE}}=L_{\mathrm{rec}}+\lambda L_a.$$

那么结论应归于**完整 MPA 模块（表示结构 + 对齐约束）的联合影响**，不能单独归因于 patching。两种模型的主重建损失、目标空间和 decoder 保持可比。报告参数量，避免忽略 MPA 引入的容量变化。

## 5. 实验 1：污染是否损害传统重构学习？

**比较：LSTM-AE (0%) vs. LSTM-AE (10%)。**

### 核心指标 A：正常验证 MSE

对固定正常验证集 $D_N^{\mathrm{val}}$ 计算：

$$
E_N^{(\rho)}=
\frac{1}{|D_N^{\mathrm{val}}|}
\sum_{x\in D_N^{\mathrm{val}}}\|x-\hat x^{(\rho)}\|_2^2.
$$

报告相对变化：

$$\boxed{\Delta E_N=\frac{E_N^{(10)}-E_N^{(0)}}{E_N^{(0)}+\epsilon}}.$$

- $\Delta E_N>0$：当前设置下正常重建误差上升。
- $\Delta E_N\approx0$：未观察到明显影响。
- $\Delta E_N<0$：误差下降；不能声称此项支持污染有害。

### 核心指标 B：独立测试 AUPRC

用两组模型的**原始空间重建误差**作为异常分数，固定测试集、只在评价时使用标签：

$$P^{(0)}=\mathrm{AUPRC}^{(0)},\qquad P^{(10)}=\mathrm{AUPRC}^{(10)}.$$

$$\boxed{\Delta P=P^{(0)}-P^{(10)}}.$$

AUPRC 不依赖按测试标签选出的最优阈值，可减少与另一条 threshold-selection 评论的冲突。

### 辅助图：Loss dynamics（轻量版）

无需保存每个训练样本的跨 epoch loss，也无需计算 $m_x$、$v_x$。每个 epoch 结束，在**相同的固定正常验证窗口**上计算 $E_N(e)$，画出 clean 与 contaminated 两条学习曲线。可选用不参加最终评价的独立异常 probe windows 记录 $E_A(e)$，但不参与训练/模型选择。

**注意：** 异常窗口 loss 高，只说明其更难拟合，不足以证明它损害正常学习。主证据是固定正常集重建变化和测试 AUPRC 变化。

## 6. 实验 2：加入 MPA 是否带来收益？

**比较：LSTM-AE+MPA (0%) vs. LSTM-AE+MPA (10%)，再与实验 1 对应结果配对。**

无需新增其他指标。考察两种收益：

1. **干净条件下的模块价值：** $P_{\mathrm{MPA}}^{(0)}$ 与 $P_{\mathrm{base}}^{(0)}$（及正常 MSE）。
2. **污染条件下的鲁棒性变化：** $\Delta P_{\mathrm{base}}$ 与 $\Delta P_{\mathrm{MPA}}$。

定义对照量：

$$
\boxed{\Delta_{\mathrm{robust}}
=\Delta P_{\mathrm{base}}-\Delta P_{\mathrm{MPA}}}.
$$

若 $\Delta_{\mathrm{robust}}>0$ 且绝对 AUPRC 不差，才能说观察到 MPA 在该骨干上减轻污染性能损失。若只是干净条件 AUPRC 更高、退化相近，则只能说明**检测性能获益**，不能说明额外污染鲁棒性。

## 7. 正文结果展示

### Table X. Impact of Training Contamination and MPA on Reconstruction Learning

| Model | Normal MSE (0%) ↓ | Normal MSE (10%) ↓ | $\Delta E_N$ ↓ | AUPRC (0%) ↑ | AUPRC (10%) ↑ | AUPRC Drop ↓ |
|---|---:|---:|---:|---:|---:|---:|
| LSTM-AE | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
| LSTM-AE + MPA | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

汇总 **3 seeds mean ± std**。表中的绝对值和变化率由真实运行填写，不预设结论。

### Figure X. Normal Reconstruction Learning Dynamics

- 横轴：epoch；纵轴：固定正常验证集 MSE。
- 四条曲线：`LSTM-AE 0%`、`LSTM-AE 10%`、`+MPA 0%`、`+MPA 10%`。
- 优先显示 3 seeds 均值与误差带；不将不同组成的训练集 aggregate training loss 混在一张图中比较。

## 8. 执行步骤与验收

**Phase 0：最小预检（1 seed）**

1. 固定 SWaT donor/test 事件划分、窗口协议和 scaler；验证无泄漏。
2. 训练 LSTM-AE 0% 与 10%；确认重建 MSE、原生分数、AUPRC 不为常量或无效值。
3. 插入 MPA，检查时间维、shape、重建目标和 loss 计算；训练另两组。
4. 检查方法学可执行后，再扩展到 3 seeds。**不以是否出现期望下降作为预检通过条件。**

**Phase 1：正式运行**

- 4 configs × 3 seeds = 12 runs。
- 每次保存：模型标识、seed、训练条件、donor/test 事件 ID、正常验证 MSE、测试 AUPRC、逐 epoch 正常验证 MSE、参数量和配置。

**Phase 2：结果解释**

| 实际结果 | 可以写的结论 |
|---|---|
| LSTM-AE 污染后正常 MSE 上升 | 该 baseline 正常模式重建质量受到可测量影响 |
| LSTM-AE 污染后 AUPRC 下降 | 该 baseline 的检测能力受到污染影响 |
| LSTM-AE 两项都没变化 | 本实验未检测到污染危害；不能声称已定量证明 |
| +MPA 在 0% 下 AUPRC 更好 | MPA 对该重构骨干具有模块价值 |
| +MPA 在 10% 下 AUPRC 下降更少 | MPA 在该重构骨干上体现污染鲁棒性收益 |
| +MPA 绝对性能高，但退化不变 | 可以谈检测能力提升；**不能**谈污染缓解 |

## 9. Manuscript Revision（实验设置英文草稿）

**Suggested location:** Section 4, after Ablation Study.  
**Suggested title:** *Impact of Anomalous Training Observations on Reconstruction Learning*

> To quantify the influence of anomalous observations on reconstruction-based learning, we conduct a controlled experiment using LSTM-AE as a representative reconstruction baseline. We compare clean training with a contaminated setting in which a predefined proportion of normal training windows is replaced by windows drawn from separate, naturally occurring anomaly events. The normal validation set and the evaluation set remain unchanged. We measure reconstruction error on a fixed normal validation set and evaluate detection performance using threshold-independent AUPRC.
>
> To further investigate the contribution of multi-view representation learning, we incorporate the proposed Multi-view Patching Alignment (MPA) module into the LSTM-AE framework while retaining its LSTM encoder–decoder backbone and original-input reconstruction objective. Both variants are evaluated under identical training-contamination protocols, allowing us to examine the effect of MPA on reconstruction quality, anomaly-detection performance, and sensitivity to contaminated training observations.

结果性句子留待实验完成后撰写。

## 10. 两个 Reviewer Comment 的对应关系

**R1 — Motivation：** 借助实验 1 的 `0% vs. 10%` 对照，量化真实训练异常是否影响传统重构模型的正常重建质量或检测性能；loss 曲线只用于辅助解释。

**R2 — Methodological contribution：** 借助实验 2 的 `LSTM-AE vs. LSTM-AE+MPA` 对照，验证 MPA 在通用重构框架中的可迁移贡献。该结果与原稿已有 PESR 消融证据是**互补关系**，不把一个模块迁移实验写成整个 PESR 所有模块均被验证。

---

### 最终实验一句话

**使用同一个 SWaT、同一组真实异常污染数据，只比较 LSTM-AE 与 LSTM-AE+MPA 在 0%/10% 条件下的正常验证 MSE、独立测试 AUPRC 和正常重建学习曲线，以最少工作量分别说明“问题是否存在”与“MPA 是否有帮助”。**
