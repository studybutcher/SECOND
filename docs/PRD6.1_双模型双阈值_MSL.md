# PRD6.1 · MSL 上 2 模型 × 2 阈值口径（oracle bf vs POT）

> 日期：2026-10-19 ｜ **状态：已跑完（36/36 runs，0 失败），结果见 §5；checkpoint 结论见 §9**
> 上游：`PRD6_PESR_高斯噪声_多seed_四数据集_结果_20261019.md`（PESR 的 bf 结果）、
> `PRD5.2`（LSTM-AE）、`PRD3.1_阈值选择策略_对齐DTAAD.md`（POT 协议）。

---

## 0. 决策记录（2026-10-19 用户确认）

| # | 项 | 决定 |
|---|---|---|
| **D1** | POT 口径 | **A：`pot_dtaad` + `--pot-fixed`**（q=1e-3, level=0.98, scale=1.0） |
| **D2** | 主表 bf 列 | **存档结果**（PESR 用 PRD6；LSTM-AE 用本轮重训后的存档结果） |
| **D3** | dropout | 保持现状**不关**（与所有已有结果一致；PRD3.1 F6 已声明） |
| **D4** | 是否重训 | PESR **不重训**（18 次复评）；LSTM-AE **重训 18 个 run** |
| **D5** | LSTM-AE 加噪 | **改成 `gauss`（N(0,1)）**，与 PESR 统一，存档的 mixed 结果保留但不再用于本表 |
| **D6** | 数据集 | 只做 MSL |
| **D7** | η / seeds | {0, 0.05, 0.10, 0.15, 0.2, 0.5} / {42, 7, 17} |

---

## 1. 目标

现有 `figs/robust_ms/robust_combined_two_rulers.pdf` 只有两个模型在
**oracle 阈值（bf）**下的曲线。本轮补上 **POT（标签无关阈值）** 的对照，
形成 **2 模型 × 2 阈值口径**：

| | bf（oracle） | POT（label-free） |
|---|---|---|
| LSTM-AE | 本轮重训后存档 | 本轮训练时同时产出 |
| PESR | 已有（PRD6） | 本轮复评产出 |

要回答的问题：**当前的高 F1 有多少来自“用测试标签搜阈值”？** 去掉标签后，
两个模型在 6 档 η 上还剩下多少。

---

## 2. POT 口径 A 与 DTAAD 的异同（论文表述依据）

| 环节 | 是否与 DTAAD 相同 |
|---|---|
| `third_party/spot.py` | ✅ 逐字节相同 |
| 阈值规则 `mean(run(dynamic=False)['thresholds']) × scale` | ✅ 相同 |
| point-adjust 循环 | ✅ 逐行相同 |
| `init_score` = **全训练集**分数 | ✅ 相同 |
| `dynamic` | ✅ `False` |
| **`(level, scale)`** | ❌ DTAAD 用它自己模型的逐子序列 `lm_d`（MSL: 0.999 / 1.04）；本轮固定 0.98 / 1.0 |
| **`q`** | ❌ 本轮 1e-3；DTAAD 代码 1e-5 / 论文 1e-4。静态模式下阈值对 q 不敏感（实测 0.11%） |
| **评测粒度** | ❌ DTAAD 单子序列（MSL `C-1_`）；本轮全测试集 |
| **dropout** | ❌ 本轮不关；DTAAD 调用了 `model.eval()` |

**建议表述**：

> 复用 DTAAD 的 POT 实现与阈值规则（同一份 SPOT、同一套 point-adjust、
> `init_score` 同样取全训练集），并为本模型重新标定固定的 `(level, scale)`。

**不得写**：“与 DTAAD 完全一致”或“我们实现了 POT/EVT”——
`docs/PRD3.1` §1.1 已实测，静态 `pot_dtaad` 退化为
**“训练分数的 level 分位数 × scale”**，不真正依赖 GPD 拟合。

---

## 3. 协议

### 3.1 LSTM-AE：用 gauss 重训 18 个 run（训练时同时产出 bf 与 POT）

```bash
python main.py --dataset MSL --trainmode train-test \
  --backbone lstmae --noise {η} --noise-kind gauss --seed {s} \
  --threshold-mode both --pot-fixed \
  --result-dir results/lstmae_robust_gauss \
  --result-name lstmae_MSL_gauss_noise{η}_seed{s} \
  --pot-dir results/threshold_ms \
  --pot-result-name lstmae_MSL_noise{η}_seed{s} \
  --ckpt-dir checkpoint/lstmae_robust_gauss/lstmae_MSL_gauss_noise{η}_seed{s}
```

- **完全沿用** PESR 的训练协议：W=70、bs=60、latent=256、lr=0.01、10 epoch、
  每 256 步一次 `optimizer.step()`、val=0.35。
- 存档的 **mixed** 结果（`results/lstmae_robust/`）保留，但**不进本表**。
- 新结果目录 `results/lstmae_robust_gauss/`、ckpt 目录 `checkpoint/lstmae_robust_gauss/`。

### 3.2 PESR：不重训，复评 18 个已有 ckpt

```bash
python main.py --dataset MSL --trainmode test \
  --loadpath checkpoint/robust_ms/pesr_MSL_gauss_noise{η}_seed{s}/best.pt \
  --backbone mamba --noise {η} --noise-kind gauss --seed {s} \
  --threshold-mode both --pot-fixed \
  --result-dir results/threshold_ms \
  --result-name pesr_MSL_noise{η}_seed{s} \
  --pot-dir results/threshold_ms \
  --pot-result-name pesr_MSL_noise{η}_seed{s}
```

- 必须回传 `--noise/--noise-kind/--seed`：POT 的 `init_score` 取自训练集分数，
  `add_noise` 在 `set_seed` 之后、训练之前执行，同一命令能复现同一份含噪训练集。
- 主表的 PESR bf 列用**存档** `results/robust_ms/pesr_MSL_gauss_noise*.json`；
  本次复评的 bf 只用于一致性校验。

### 3.3 阈值定义（两个口径）

- **bf（oracle）**：`bf_search` 在测试标签上搜索百分位阈值 `93 → 99.99`
  （步长 0.01），取 point-adjust F1 最大者。**不可部署**。
- **POT（label-free）**：只用训练集分数与测试集分数，不看测试标签。
- 两者使用同一套 `adjust_predicts`（point-adjust）。

### 3.4 已知边界（论文里要写明）

1. **dropout 未关**：同一条命令两次复评的 F1 会不同（PRD3.1 F6）。
   LSTM-AE 的 bf/POT 来自同一次训练后的同一次前向；PESR 的 POT 与存档 bf
   来自不同前向，故另列一致性校验。
2. **POT 参数固定**（level=0.98，不按 η/seed 重新标定）：这正是
   “部署时只有训练集、不知道测试分布”的语义。
3. `val ⊂ test` 的既存缺陷未改；POT 不使用标签。
4. 两个模型本轮都是 `gauss N(0,1)` 整行替换，加噪分布**已统一**。

---

## 4. 实验矩阵与成本

| 模型 | 操作 | 数量 | 单 run 预估 | 4 卡墙钟 |
|---|---|---|---|---|
| LSTM-AE | 重训（train-test，含 POT） | 18 | ≈ 3–4 min（存档实测 10.9 s/epoch） | ≈ 20 min |
| PESR | 复评（test-only，含 POT） | 18 | ≈ 5–8 min | ≈ 40 min |
| **合计** | | **36** | | **≈ 1 h** |

> 实际墙钟受其他用户负载影响；先各跑一个试点核对 bf 与存档一致，再铺开。
> **实际**：13:29 → 14:14，两套脚本并发跑在 4+4 张卡上，墙钟 ≈ 45 min，
> 36/36 成功、0 个 Traceback。

---

## 5. 结果（已完成，36/36 runs，0 失败）

> 机器可读：`results/threshold_ms/summary_pot_2x2.{md,csv}`、`table_pot_2x2.tex`。
> 图：`figs/robust_ms/pot_2models_2thresholds.{pdf,png}`。
> 所有数字为 3 seeds 的 mean ± sd。

### 5.1 主表

| η | LSTM-AE bf | LSTM-AE POT | PESR bf | PESR POT |
|---|---|---|---|---|
| 0 | 0.6765 ± 0.0329 | 0.0512 ± 0.0009 | 0.9467 ± 0.0136 | 0.9272 ± 0.0017 |
| 0.05 | 0.6649 ± 0.0218 | 0.0000 ± 0.0000 | 0.9414 ± 0.0128 | 0.9211 ± 0.0082 |
| 0.10 | 0.6608 ± 0.0145 | 0.0000 ± 0.0000 | 0.9420 ± 0.0021 | 0.9265 ± 0.0116 |
| 0.15 | 0.6520 ± 0.0029 | 0.0000 ± 0.0000 | 0.9474 ± 0.0051 | 0.9318 ± 0.0061 |
| 0.20 | 0.6380 ± 0.0074 | 0.0000 ± 0.0000 | 0.9418 ± 0.0088 | 0.9368 ± 0.0036 |
| 0.50 | 0.5295 ± 0.2029 | 0.0000 ± 0.0000 | 0.9490 ± 0.0109 | 0.9309 ± 0.0157 |

bf − POT 差值（PESR）：0.0195 / 0.0203 / 0.0155 / 0.0156 / 0.0050 / 0.0181，
均值 0.0157，全部 ≤ 或接近对应 η 的 seed-sd。

### 5.2 读数

1. **PESR：标签无关的 POT 只比 oracle 低 0.5–2.0 个点。** POT 曲线在
   0.9211–0.9368 之间、与 bf 同样平坦。→ “现有高 F1 主要不是靠用测试标签
   搜阈值”这句话**对 PESR 成立**。
2. **LSTM-AE：POT 从 η=0 起就崩**（0.0512），η≥0.05 全为 0。原因是注释
   见 §5.3，**不是 checkpoint 选择问题**。
3. 因此 2 模型对比在两个口径下给出不同结论：
   - bf：PESR 0.94 vs LSTM-AE 0.53–0.68；
   - POT：PESR 0.92–0.94 vs LSTM-AE 0.00–0.05。
   **阈值口径对模型间差距的影响，远大于噪声比例本身**——这是本轮最值得写进
   论文的一句话。

### 5.3 LSTM-AE 的 POT 为什么是 0（诊断）

静态 `pot_dtaad` 实测退化为 `quantile(init_score, level) × scale`，
`init_score` 是**全训练集**的重建误差。训练集被加噪后，训练分数的 98 分位
随 η 单调抬升，而测试分数几乎不变：

| η | train 98% (init_threshold) | POT threshold | TP | FP | POT F1 |
|---|---|---|---|---|---|
| 0 | 152.2 | 153.0 | 181 | 246 | 0.0512 |
| 0.05 | 447.1 | 566.7 | 0 | 0 | 0 |
| 0.10 | 654.0 | 799.4 | 0 | 0 | 0 |
| 0.15 | 841.1 | 1006.6 | 0 | 0 | 0 |
| 0.20 | 1020.1 | 1202.5 | 0 | 0 | 0 |
| 0.50 | 1964.7 | 2151.3 | 0 | 0 | 0 |

阈值从 153 抬到 2151（×14），高到**没有任何测试点越界**（TP=FP=0）。

对照 PESR：`init_threshold` 只在 51–117 之间，与测试分数同量级，所以正常。

关键点：**每个 epoch 都在同一份加噪训练集上训练**，训练分数分布的抬升与
选中哪个 epoch 无关。按 POT 选 checkpoint 只会选到一个更欠拟合的 epoch，
并不能修好阈值。→ 这是“训练分数 ↔ 测试分数可比性”被破坏，不是 LSTM-AE
的表达能力问题。

> 论文含义：凡把阈值定在训练分数分位数上的 label-free 方法，在训练数据被
> 污染时会整体失效；报告 label-free 指标时必须同时说明训练端是否干净。

### 5.4 一致性校验：复评 bf vs 存档 bf

| 模型 | mean \|Δ\| | max \|Δ\| | 说明 |
|---|---|---|---|
| LSTM-AE | 0.0000 | 0.0000 | 无 dropout，逐位可复现 |
| PESR | 0.0098 | 0.0268 | dropout 未关，两次前向不同（PRD3.1 F6） |

PESR 的 Δ 与 seed-sd（0.002–0.014）同量级，说明复评没有引入新偏差。

### 5.5 gauss vs mixed（LSTM-AE，附录，不进主表）

旧 mixed 噪声（U(−2,2)/±5/random_walk）在 η=0.05 就把 LSTM-AE 的 bf 打到
0.4238；换成 gauss 后同一点是 0.6649。**旧结果的塌陷主要由 ±5 的脉冲噪声
造成**，不是 LSTM-AE 在该噪声比例下本身失效。两套结果不可直接比
（噪声分布不同），故只作附录。

---

## 6. 产物

| 类型 | 路径 |
|---|---|
| LSTM-AE 逐 run 结果 | `results/lstmae_robust_gauss/lstmae_MSL_gauss_noise<η>_seed<s>.json` |
| PESR 复评 bf | `results/threshold_ms/pesr_MSL_noise<η>_seed<s>.json` |
| POT（两个模型） | `results/threshold_ms/SECOND_MSL_fixed_<model>_MSL_noise<η>_seed<s>.json` |
| 汇总表 | `results/threshold_ms/summary_pot_2x2.{md,csv}` |
| 论文表 | `results/threshold_ms/table_pot_2x2.tex` |
| 合并图 | `figs/robust_ms/pot_2models_2thresholds.{pdf,png}` |
| 调度 | `script/robust/pot_multiseed.sh`、`script/robust/lstmae_gauss_pot.sh` |
| 聚合 | `script/robust/summarize_pot_2x2.py` |
| 绘图 | `bench/plot_pot_2x2.py` |
| 日志 | `nohp-out/pot_eval/*.log`、`nohp-out/lstmae_robust_gauss/*.log` |

---

## 7. 代码改动

`main.py`：

- 新增 `--pot-dir`（默认 `results/threshold`）与 `--pot-result-name`（默认 `None`）。
  批量复评时 POT JSON 若沿用固定文件名会互相覆盖；默认值下行为与改动前完全一致。

> 这一处改动**不影响任何已有结果**，但会进入 git diff，审核时可一并看。

---

## 8. 风险与不做的部分

- **不做多 POT 口径的全矩阵**（C/D 只在旧 ckpt 上验证过一次，且明显更差）。
- **不做其余三个数据集的 POT 多 seed 复评**：本轮只做 MSL。
- **不挑选 seed、不报 best-of-N**：clean 与各 η 用同一套 3 seeds，报 mean ± sd。
- **旧的 LSTM-AE mixed 结果不删除**：保留可追溯，但不进本轮的表和图。
- **不重训 PESR**：所有 PESR 结论只针对现有 ckpt。

---

## 9. 关于 checkpoint 选择（回答“两个阈值是否该各选各的 ckpt”）

> **2026-10-19 更新：本节结论已被 `PRD6.2_POT原生checkpoint与初值标定_MSL.md` 取代。**
> PRD6.2 §0.2 发现 `main.py` 的 `best.pt` 恒为 epoch 0（循环在 val 前 reload 历史
> best），因此“换一个 POT 专用 ckpt”在当前代码上根本不可行；必须先修训练循环并
> 逐 epoch 落盘。下面保留原始推理，作为“在错误前提下为什么会得出不重训”的记录。

**现状**：每个 run 只有一个 `best.pt`，由**验证集 bf F1** 选出（`main.py:355-362`），
两个阈值口径都在同一个 ckpt 上评测。理论上 POT 的最优 epoch 可能不同。

**结论：本轮不需要为 POT 全量重训。** 依据：

1. **PESR 上换 ckpt 改变不了结论。** bf→POT 只掉 0.0050–0.0203（均值 0.0157），
   且多数不超过 seed-sd。即便按 POT 单独选 ckpt 能追回这 1–2 个点，
   “PESR 在标签无关阈值下仍 ≈0.93”这一结论不变；而全量重训 PESR
   （383 s/epoch × 10 × 18 runs ≈ 19 GPU·h）成本远高于收益。
2. **LSTM-AE 的 POT=0 不是 ckpt 选错造成的。** 见 §5.3：训练阈值随 η 从
   153 抬到 2151，而训练分数分布的抬升对**所有 epoch 同时成立**。按 POT
   选 ckpt 只会偏向前几个欠拟合 epoch，不能恢复“训练/测试分数可比性”。
   重训不会改变这张表。
3. **若仍想验证第 2 点**，最便宜的做法是只做 clean LSTM-AE 的逐 epoch
   扫描（每 epoch 存一次，用验证集分别按 bf/POT 选），约 5 分钟；
   不需要重训 PESR，也不影响本表。

**建议**：按现状定稿；把“两阈值共用同一 ckpt”作为明确的实验协议写进
limitation（与 PRD3.1 的 dropout 声明并列）。
