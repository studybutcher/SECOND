# PRD5.2 · LSTM-AE 在现有 robust 加噪协议下的训练效果

**版本** v1 · 2026-10-10 ｜ **状态**：待实施，不预设结果
**目标**：把 `script/robust/*` 那套**现有的训练侧加噪协议**原样搬到 **LSTM-AE** 上，
看一个**传统重构模型**在这套加噪下的训练行为。

---

## 0. 为什么要做（三条）

1. **现有 robust 结果是用一把没有分辨力的尺子量的。** `script/robust/*` 全部报 point-adjust F1
   （`main.py` 的 `bf_search` oracle 阈值 + `adjust_predicts`）。已实测该指标在 MSL/SMAP/SWaT 上
   **窗口级 AUROC ≈ 0.50**、未调整 F1 只有 0.01–0.04（`docs/PRD5_T0结果_20261010.md` §0）。
   所以 `nohp-out/Robust/SMAP.out` 那条"η 0.05→0.50 都不降"的平曲线**不足以作为结论**。
2. **换一个模型才能分清"协议不敏感"还是"SECOND 不敏感"。** 现在的 robust 结果全是 SECOND 及其消融变体。
   LSTM-AE 是一个**完全不同的传统重构模型**：
   - 若 LSTM-AE 的曲线**也平** → 是**协议**（±2/±5 的整行替换）不敏感；
   - 若 LSTM-AE **明显退化** → 协议有效，SECOND 的平就是**SECOND 的问题**（与塌缩一致）。
3. **这是 R2 的桥。** "传统重构方法"在本实验里第一次拿到一个**干净、无阈值**的训练行为读数，
   且与 `docs/PRD5.1_r3_AE结果_20261010.md` 的 AE 结果**直接可比**（同一划分、同一指标、噪声档重叠）。

---

## 1. 要复现的协议（`script/robust/*`，**原样**）

| 项 | 现有设置 | 来源 |
|---|---|---|
| 加噪入口 | `main.py --noise $noise` → `add_noise(train_data, ratio)` | `main.py:118-142, 225` |
| 加噪**时机** | `get_data()`（内含 `preprocess`）**之后** ⇒ 数据已在 **[0,1]** | `main.py:198, 225` |
| 噪声档 η | **{0.05, 0.10, 0.15, 0.2, 0.5}**（5 档） | `script/robust/{msl,SMAP,SWaT,WADI}.sh` |
| 噪声形态 | **整行替换**（全通道），三种类型轮流：<br>`uniform` ~ U(−2, 2)、`spike` = ±5、`random_walk` = cumsum(U(−0.5,0.5)) | `main.py:128-139` |
| η 的定义 | 被替换的**时间戳**占比（`int(N·η)` 行） | `main.py:120` |
| W / bs | main.py 默认：MSL/SMAP **W=70**、SWaT **W=35**、bs=60（WADI bs=24） | `main.py:106-109` |
| 现有调度 | seed 42、**10 epoch**、Adam lr 0.01、每 256 步 `step()` | `main.py:299,314,324` |

### 1.1 ⚠️ 必须写明的两件事

**(a) ±2 / ±5 是绝对常数，而数据在 [0,1]。** 所以这套"噪声"实际是把整行替换成**远超数据范围的垃圾**，
而且 55 个通道共用同一绝对幅度 → 跨通道不可比。这条在 `状态记录 §6` 和
`docs/PRD4_噪声定量实验.md` 里都点过名。
**本 PRD 的主口径仍然原样复现它**（要复现的就是"现有设置"），
另出一版 **σ 缩放**（`k·σ_c·ε`，逐通道）作**敏感性**，用来分离"是强度太大"还是"协议本身无效"。

**(b) 噪声 ≠ 异常。** 本协议测的是「**用部分损坏的数据训练，还能不能学好正常模式**」，
**不是**「检测含噪输入」，**也不是**「真实异常污染」。
因此结论只能支撑 "training corruption / perturbed observations"，不能写成 R1 原话的
"anomalous observations contaminate the learning"（`PRD4_噪声` §5 同一限定）。

---

## 2. 必须换掉的两样东西（都是 2026-10-10 学到的）

> 复现的是**加噪协议**，不是**评测协议**。以下两项若照抄，实验会得出不可信的读数。

| # | 现有做法 | 换成 | 依据 |
|---|---|---|---|
| **M1** | 主指标 = point-adjust F1（oracle 百分位阈值） | 主指标 = **固定正常验证集 MSE $E_N$** + **测试集未调整 AUPRC**；F1 只作**参照列** | `PRD5_T0结果` §0：F1 无分辨力 |
| **M2** | `main.py` 的 val **取自 test 尾部且含异常**，且推理**不调 `eval()`**；10 epoch | 独立驱动：固定 **clean-normal val**（与 train 不重叠）+ 统一 `model.eval()` 评价 + **收敛的轮数** | `PRD5.1_r3_AE结果` §1：50 epoch 不收敛，`lr=1e-3` 会在最小值附近抖出尖峰；cosine + 100 epoch 才稳 |

**训练调度不属于被比较的对象**（本实验比较的是同一个模型在不同 η 下的行为），
所以有自由把它调到收敛；不必也不能与既有 SECOND 的 10-epoch 数字逐位对齐（模型不同）。

---

## 3. 模型：LSTM-AE

$$\hat X = \mathrm{Dec}\big(\mathrm{Enc}(X)\big),\quad X,\hat X\in\mathbb R^{W\times C},\qquad
\mathcal L=\tfrac{1}{WC}\|X-\hat X\|_F^2$$

- **输入/输出都是原始窗口**（不是隐层别的张量）—— 这是重构类模型的共同口径，也是与 AE 实验可比的前提。
- 结构：LSTM encoder（取末隐状态 → latent）→ LSTM decoder（以 latent 为初值逐拍解码）。
- 超参**先钉默认、全配置一致、不得按 test 调**：`hidden=128`、`latent=256`、1 层、bs=128、
  Adam `lr=1e-3 → 1e-5` **cosine**、epochs 见 §5。
- **不引入 MPA**（那一支已单独记录：`docs/PRD5.1_r3_Phase0结果_20261010.md`，MPA+AE 可复现塌缩）。

### 3.1 与 `main.py` 的接口

**不复用 `main.py`**，另写 `bench/train_lstmae.py`（与 `bench/train_ae.py` 同构）。理由见 §2-M2：
`main.py` 绑定 `patchattmamba` 的接口、自带 `val ⊂ test`、推理不调 `eval()`。
复用 `bench/ae_data.py` 的**数据划分与 probe**，因此两套实验（AE / LSTM-AE）**逐窗口可比**。

---

## 4. 数据协议（复用 `bench/ae_data.py`，不重造）

| 角色 | 内容 |
|---|---|
| `train_normal` | MSL 47,389 行（逐实体 85%，26 块） |
| `val_normal` | **固定** 8,669 行（逐实体后段 15%，与 train 留 70 行隔离区） |
| scaler | **只在 train_normal 上拟合**，val/probe 复用 |
| probe | test 的 normal 61,869 / anomaly 9,970 窗（异常率 0.1388，any-in-window） |
| 加噪后 | **先加噪、后切窗**（与 `main.py` 同序，§1） |

---

## 5. 实验矩阵与成本

| η | seed | 说明 |
|---|---|---|
| **0**（clean） | 42, 7, 17 | 基准 |
| 0.05 / 0.10 / 0.15 / 0.2 / 0.5 | 42, 7, 17 | 照抄现有 5 档 |

**6 档 × 3 seed = 18 runs**（MSL）。**先做 MSL**；SMAP/SWaT 视结果再铺。
`η ∈ {0.05, 0.10, 0.2}` 与 AE 实验重叠，可**直接并排比较**。

**成本**：LSTM-AE 比 MLP-AE 慢（序列长 70）。**先跑 1 个 epoch 卡表**再定 epochs 与排期；
预期单 run 20–60 min 量级，3 张卡并行 ⇒ 数小时。**实测后再更新本节。**

**seed 数**：3 是下限。AE 实验里 **1/24 个 run 落入坏盆地**（`PRD5.1_r3_AE结果` §3），
所以每个 η 至少要 3 个 seed 才能看出离散度；若某档的离散度与效应同量级，再补 seed。

---

## 6. 指标

| 指标 | 定义 | 地位 |
|---|---|---|
| **$E_N$** | 固定 clean-normal val 上的逐元素 MSE | **主指标** |
| **$E_N$-med10** | 末 10 轮 val MSE 的中位 | 主指标的稳健版（末点抖动时以它为准） |
| $\Delta E_N(\eta)$ | $(E_N(\eta)-E_N(0))/E_N(0)$，**同 seed 配对后再平均** | **主判据** |
| **test-AUPRC** | 逐点重建误差的**未调整** AUPRC（阈值无关），报 prevalence | 与既有 robust 数字的桥 |
| test-AUROC | 同上 | 辅助 |
| $E_{A,clip}$ | anomaly probe 的 MSE（**clip 到 [0,1]**） | 辅助（未 clip 版被 MinMax 外推主导） |
| F1(point-adjust) | 现有口径 | **参照列，不参与任何比较** |

**图（一张）**：val MSE vs epoch，每 η 一条线（seed 均值 ± 带）——与 AE 实验的 Figure X 同格式。

---

## 7. Gate（全过才铺 seed）

| Gate | 检查 | 判据 |
|---|---|---|
| **G1** | 加噪与 `main.py` 一致 | 同一 seed+η 下，用 `main.py` 的 `add_noise` **逐位复现**我的加噪数组（`np.array_equal`） |
| **G2** | 顺序正确 | 加噪在 `preprocess` 之后、切窗之前 |
| **G3** | 划分无泄漏 | train/val 时间段不重叠；scaler 只从 train 拟合 |
| **G4** | 重建非平凡 | **Gate N**：真实/全零/随机三输入下重建非常量（`PRD5.1_r3` §7 同款） |
| **G5** | 已收敛 | 末 10 轮 val MSE 极点比 < 1.2；否则调 epochs/调度后重跑 |

---

## 8. 判据（预注册）

| # | 观察 | 结论 |
|---|---|---|
| **D1** | 随 η 上升 $E_N$ **单调上升**且配对显著 | 该加噪协议**确实**损害传统重构模型的正常模式学习 |
| **D2** | $E_N$ 在不同 η 间**无稳定差异** | 本协议下未检测到影响 ⇒ **协议不敏感**（与 SECOND 的平曲线同因，但这次尺子是好的） |
| **D3** | LSTM-AE 退化明显、而 SECOND 的 F1 曲线平 | 平的是 **SECOND**，不是协议 ⇒ 与塌缩诊断一致 |
| **D4** | σ 缩放版与 ±2/±5 版结论不同 | 差异来自**强度/口径**，须在正文写明主口径是哪一个 |
| **D5** | test-AUPRC 与 $E_N$ 结论不一致 | 两者都报，**不得只报支持叙事的那一个** |

**禁止**：因为没出现期望曲线就调大 η（`PRD4_噪声` §2 已写）；只报部分 η 档；
用某次有利 seed 当普遍效果。

---

## 9. 能写 / 不能写

| 观察 | 可以写 | 不能写 |
|---|---|---|
| 随 η 上升 $E_N$ 上升 | 该加噪协议损害了该 LSTM-AE 的正常模式重建 | 真实工业异常污染必然有同样影响 |
| 曲线平 | 本协议（整行替换 ±2/±5，η ≤ 0.5）对传统重构模型不造成可测损害 | 训练污染无害 |
| LSTM-AE 退化而 SECOND 平 | 差异来自模型而非协议 | 已证明 SECOND 的模块更鲁棒 |
| F1 参照列与 $E_N$ 不一致 | 说明 F1 口径不可用（再次佐证 §0） | 用 F1 数字做任何比较 |

---

## 10. 交付物

| 类型 | 路径 |
|---|---|
| 模型与驱动 | `bench/lstm_ae.py`、`bench/train_lstmae.py`（复用 `bench/ae_data.py`） |
| 加噪 | `bench/noise_robust.py`：**直接 import `main.add_noise`** 以保证逐位一致（Gate G1） |
| 结果 | `results/ae_mpa/lstmae_MSL_eta{00,05,10,15,20,50}_seed{S}.json` |
| 汇总 | `results/ae_mpa/summary_lstmae.md`（复用 `bench/summarize_ae.py` 的格式） |
| 图 | `figs/ae_mpa/lstmae_normal_mse_vs_epoch.pdf` |
| 队列 | `script/repr/lstmae_sweep.sh` |

---

## 11. 与其它文档的关系

| 文档 | 关系 |
|---|---|
| `script/robust/*` | **被复现的对象**（加噪协议）；其 F1 结果不作比较基准 |
| `docs/PRD5.1_r3_AE结果_20261010.md` | **直接可比**：同一划分/指标，噪声档重叠 ⇒ 可并排成一张表 |
| `docs/PRD5.1_r3_Phase0结果_20261010.md` | MPA 那一支（塌缩），本 PRD **不做 MPA** |
| `docs/PRD5_T0结果_20261010.md` §0 | 为何弃用 F1（M1 的依据） |
| `docs/PRD4_噪声定量实验.md` | 同族协议；其"σ 缩放必须做"与"测的是 corruption tolerance"两条被本 PRD 采纳为敏感性/限定 |
