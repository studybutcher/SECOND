# PRD5.1 r3 · Phase 0 结果：**MPA + AE 可复现地塌缩**（2026-10-10）

> Phase 0 的验收条件是"实现正确、Gate 有效"，**不以出现期望结果为通过条件**（§8）。
> **结论：普通 AE 正常；把 MPA 接上去之后模型塌缩成常量，8 种配置无一例外。**
> 这同时给 `docs/诊断_表示塌缩_20261010.md` 提供了**塌缩起源于 MPA 的独立证据**。

---

## 1. 实现了什么（工作项 1–3）

| 文件 | 内容 |
|---|---|
| `bench/ae_data.py` | 正常 train/val 划分（逐实体、后段作 val、留 `GAP=W` 隔离区）；**scaler 只在 train_normal 上拟合**；test 的 normal/anomaly probe |
| `bench/ae_contam.py` | 结构化异常段注入，复用 `gen_contam.plan_lengths/place_all`，注入**可注入块**（21/26）的 col0 |
| `bench/ae_baseline.py` | `WindowAE` / `WindowVAE` / `MPAWindowAE`；`MPAFrontend` 复刻 + **等价性检验**；Gate N/V |
| `bench/train_ae.py` | 独立训练驱动（固定轮数、`eval()` 评价、逐 epoch val MSE、Gate 记录） |

**数据**（MSL, W=70）：train 47,389 行 / val 8,669 行 / 26 块（T-9 太短被跳过）；probe 61,869 正常 + 9,970 异常窗（异常率 0.1388）。
**污染**（α=1.5σ, seed 42）：ρ=5% → 2,369 行/4 段；10% → 4,739/8；20% → 9,478/16；**嵌套 `5%⊂10%⊂20%` 成立**，四项断言全过，ρ_win/ρ = 1.11–1.12。

### 1.1 Gate 通过情况

| Gate | 结果 |
|---|---|
| **M**（MPA 复刻等价） | ✅ **`max|ΔH| = 0.000e+00`**，`strict=True` 加载权重；与 `patchattmamba.pipline` 逐值相同 |
| **A**（注入正确） | ✅ 污染行数精确、指示列未动、非污染行逐位不变、跨档嵌套 |
| **D**（ρ=0 即 clean） | ✅ clean = `MSL_train_normal.npy`（sha256 已记） |
| **V**（瓶颈有效） | ✅ 3850 → 256 = **15.0×**（对比 SECOND 的 275/256≈1.07） |
| **N**（非常量） | 普通 AE ✅；**MPA 变体全部 ❌** |

---

## 2. 核心结果：普通 AE 正常，MPA+AE 塌缩

10 epoch，MSL ρ=0，seed 42（`E_N` = 固定正常验证集 MSE，越小越好）：

| 模型 | 配置 | `E_N` | `E_A`(异常) | `E_N/E_A` | Gate N `真实vs全零` |
|---|---|---|---|---|---|
| **AE** | — | **0.001891** ✅ 仍在下降 | 0.015271 | 0.124 | **1.274** ✅ |
| MPA+AE | λ=1, mean | 0.004949 | 0.1535 | 0.032 | **0.000e+00** ❌ |
| MPA+AE | λ=0.1, mean | 0.005577 | 5.7e+05 | 0.000 | 0.000e+00 ❌ |
| MPA+AE | λ=0.01, mean | 0.004948 | 0.1535 | 0.032 | 0.000e+00 ❌ |
| MPA+AE | **λ=0, mean** | 0.004948 | 1.4e+03 | 0.000 | 0.000e+00 ❌ |
| MPA+AE | λ=0, standardize | 0.007310 | 0.1593 | 0.046 | 0.000e+00 ❌ |
| MPA+AE | λ=0, layernorm | 0.004953 | 0.1536 | 0.032 | 0.000e+00 ❌ |
| MPA+AE | λ=0, **concat** | 0.009483 | 6.2e+04 | 0.000 | 0.000e+00 ❌ |

**读法**：

1. **普通 AE 完全正常**：`E_N` 从 0.0037 单调降到 0.00189 且仍在降；`E_A/E_N = 8.1×`（异常重构得差 8 倍，符合重构式检测器的预期行为）；Gate N 过。
2. **MPA+AE 全部塌缩**：重建对"真实 / 全零 / 随机"三种输入**逐位相同**（`max|Δ| = 0.000e+00`）。
   其 `E_N ≈ 0.0049` 恰好 ≈ 数据方差 ⇒ 就是"输出常数"的 MSE。
3. **塌缩与 λ 无关**：λ=0（**完全没有一致性损失**）照样塌 → `loss_c` 不是原因。
4. **塌缩与我的适配器选择无关**：`mean` 与 `concat` 两种池化、`standardize`/`layernorm` 两种归一化，
   全部塌缩。⇒ **不是我的适配层写坏了。**
5. 最好的一档 MPA+AE（`E_N` 0.00495）仍比普通 AE（0.00189）**差 2.6 倍**。

---

## 3. 机制：MPA 的输出被固定 positional 分量主导

未训练的 MPA，同一 batch 内跨样本统计：

| 阶段 | 跨样本 sd / 幅值 |
|---|---|
| 输入 X | 2.14 |
| value_embedding | 2.14 ✅ 有信息 |
| position_embedding（广播） | 0.00（**固定 buffer，每个窗口相同**） |
| DataEmbedding 输出（value+pos） | 1.02 |
| **H（未池化，`[B,55,512]`）** | **1.03** ✅ |
| **H.mean(dim=1)** | **0.195** ⚠️ 掉 5 倍 |
| 训练 10 epoch 后喂进 encoder 的向量 | **3.5e-03** ⚠️ 再掉 60 倍 |

- `DataEmbedding.forward = value_embedding(x) + position_embedding(x)`，而 positional 是**固定 buffer**
  ⇒ H 里有一个**与输入无关的大常量**。
- **通道维平均让常量存活、各通道的变化互相抵消** ⇒ 常量/变化比被放大 ~5 倍。
- 训练再把这个残余变化压掉 60 倍 ⇒ 喂给 AE 的向量近似常量 ⇒ AE 只能输出常量。

**数值证据**（训练 10 epoch 后）：

| | `ae` | `mpa_ae` |
|---|---|---|
| 喂进 encoder 的 `\|v\|` 均值 | **0.008** | **255.7** |
| latent `\|z\|` 均值 | 0.10 | **3845** |

MPA 那条路的输入幅度比 `flatten(X)` 大 **3 万倍**，latent 炸到 3845，decoder 退化成常量映射。
（加 LayerNorm 也救不回来：LayerNorm 只保留方向，而这个方向本身几乎不变。）

---

## 4. 这条结果的两个意义

### 4.1 对 PRD5.1 本身

按 §9 的预注册判据：**D6 命中** —— "$G^{(N)}\le 0$：未验证到 MPA 的重建增益"。
更强地说：**在当前 MPA 实现下，MPA 不能给这个 AE 骨干带来增益，而是让它塌缩。**

**禁止**把它写成"MPA 无效"就完事 —— 必须同时说明这是**实现/接口层面**的问题（见 §5），
且**不得**据此否定论文里的 PESR（PESR 的 MSSA 吃的是**未池化**的 H，`sd/|v|=1.03`，与这里的情形不同）。

### 4.2 对 `docs/诊断_表示塌缩_20261010.md`（更重要）

SECOND 的塌缩点被定位在 MPA 之后的 `attention_sim`。现在：**一个 13M 参数、没有 MSSA / 没有
CRR / 没有 `attention_sim` / 没有 dropout 的模型，只要接上 MPA 就塌缩成常量，8/8 复现。**

⇒ **塌缩的源头极可能在 MPA 模块本身**，而不是下游。这比之前的证据强得多，因为它把下游全部排除了。

---

## 5. 下一步（三条，按推荐度）

| # | 方案 | 内容 | 代价 |
|---|---|---|---|
| **A** | **消掉 positional 常量再做对照** | 给 `MPAFrontend` 一个选项：只取 `value_embedding`（不加固定 positional），或对 H 减掉**训练集**上的 H 均值再池化。用来判定"是 positional 常量杀死了信息"还是"MPA 本身不可学" | ~1 h |
| **B** | **换 MPA→AE 的接口** | 不再用全局池化，改成 **per-channel 共享投影**（`Linear(2L→k)` 逐通道共享 → flatten），保留通道身份（MSL 的 55 列是 1 信号 + 54 二值指示，平均本来就不合理） | ~1 h |
| **C** | **如实收束** | 报告 D6：当前 MPA 实现下未验证到重建增益，且可复现塌缩；把它作为 limitation，不做外部骨干实验 | 0 |

**建议先做 A**：它是最小、最直接、最能分离变量的实验 —— 如果去掉固定 positional 后 MPA+AE 能正常训练，
那 MPA 的信息是被**自己加的常量**淹掉的，这是一个非常具体、可修的工程问题；
如果仍然塌缩，那就要检查 MPA 的可训练性（B）。

---

## 6. 产物

| 类型 | 路径 |
|---|---|
| 代码 | `bench/ae_data.py`、`bench/ae_contam.py`、`bench/ae_baseline.py`、`bench/train_ae.py` |
| 数据 | `artifacts/ae_mpa/MSL_{train_normal,val_normal}.npy` + `MSL_split.json` + `MSL_probe.npz` + `MSL_train_rho{05,10,20}_seed42.{npy,json}` |
| 结果 | `results/ae_mpa/*.json`（含 `curve`、`E_N`、`E_A`、Gate 详情） |
| Gate M | `python bench/ae_baseline.py` → `max|ΔH|=0.000e+00` |
