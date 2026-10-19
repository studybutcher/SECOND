# PRD6.1 · MSL 上 2 模型 × 2 阈值口径（oracle bf vs POT）

> 日期：2026-10-19 ｜ **状态：设置已按用户选择更新，待最终确认后开跑**
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

---

## 5. 产出长什么样（审核用 mock）

### 5.1 主表（mean ± sd，n=3）

| η | LSTM-AE bf | LSTM-AE POT | PESR bf | PESR POT |
|---|---|---|---|---|
| 0 | 待重训 | 待重训 | 0.9467 ± 0.0136 | ? |
| 0.05 | 待重训 | 待重训 | 0.9414 ± 0.0128 | ? |
| 0.10 | 待重训 | 待重训 | 0.9420 ± 0.0021 | ? |
| 0.15 | 待重训 | 待重训 | 0.9474 ± 0.0051 | ? |
| 0.20 | 待重训 | 待重训 | 0.9418 ± 0.0088 | ? |
| 0.50 | 待重训 | 待重训 | 0.9490 ± 0.0109 | ? |

> PESR bf 列是当前存档结果（已核对）。其余待跑。

### 5.2 图

`figs/robust_ms/pot_2models_2thresholds.{pdf,png}`：

- 左栏：bf（oracle），两个模型；
- 右栏：POT，两个模型；
- 两栏共享 x∈[−3, 53]%、y∈[0.2, 1.0]，每 seed 散点 + mean±sd。

### 5.3 附加一致性校验

- **LSTM-AE**：新 gauss 结果与旧 mixed 结果**不可直接比**（噪声分布不同），
  只作为“换分布后模型自身变化”的附录，不入主表。
- **PESR**：同一次复评的 bf vs 存档 bf，报告 mean|Δ| / max|Δ|。
  若差异与 seed-sd 同量级，说明复评没有引入新偏差。

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
