# PRD6 · 主模型 PESR 在基础高斯加噪下的多 seed 结果（四数据集，2026-10-19）

> 依据 `docs/PRD6_PESR_高斯噪声_多seed_MSL.md`。MSL 先行轮见
> `docs/PRD6_PESR_高斯噪声_多seed_MSL_结果_20261017.md`；本文覆盖
> **MSL / SMAP / SWaT / WADI 四个数据集 × 6 档 η × 3 seeds = 72 runs**，
> 全部为 `--backbone mamba`（PESR，`patchattmamba`）、`--noise-kind gauss`。

---

## 1. 设置

| 项 | MSL | SMAP | SWaT | WADI |
|---|---|---|---|---|
| 参数量 | 201.6167 M | 193.5215 M | 199.6879 M | 221.0451 M |
| 中间块占比 | 73.40% | 76.47% | 74.11% | 66.95% |
| window length | 70 | 70 | 35 | 70 |
| batch size | 60 | 60 | 60 | 24 |
| latent | 256 | 256 | 256 | 256 |
| η | {0, 0.05, 0.10, 0.15, 0.20, 0.50} | 同 | 同 | 同 |
| seeds | 42 / 7 / 17 | 同 | 同 | 同 |
| epochs / lr | 10 / 0.01 | 同 | 同 | 同 |

- 加噪：`train[indices] = N(0,1)`，整行、逐元素独立；只加在训练集。
- 评测：`bf_search` oracle 百分位阈值 + `adjust_predicts`（point-adjust）。
- 72 个 JSON 全部 `noise_kind=gauss`，参数在各数据集内 18/18 一致
  （`results/robust_ms/pesr_<DS>_gauss_noise<η>_seed<s>.json`）。

---

## 2. 主结果：F1（mean ± sd，n=3）

| $\varepsilon$% | MSL | SMAP | SWaT | WADI |
|---|---|---|---|---|
| - (clean) | 0.9467 ± 0.0136 | 0.9729 ± 0.0026 | 0.9817 ± 0.0044 | 0.7608 ± 0.1118 |
| 5% | 0.9414 ± 0.0128 | 0.9734 ± 0.0026 | 0.9807 ± 0.0030 | 0.7486 ± 0.1549 |
| 10% | 0.9420 ± 0.0021 | 0.9711 ± 0.0011 | 0.9829 ± 0.0012 | 0.8481 ± 0.2015 |
| 15% | 0.9474 ± 0.0051 | 0.9705 ± 0.0040 | 0.9820 ± 0.0026 | 0.8446 ± 0.1721 |
| 20% | 0.9418 ± 0.0088 | 0.9707 ± 0.0027 | 0.9805 ± 0.0033 | 0.9089 ± 0.0733 |
| 50% | 0.9490 ± 0.0109 | 0.9709 ± 0.0032 | 0.9850 ± 0.0023 | 0.8556 ± 0.0840 |

完整 P/R/F1 逐档表见 `results/robust_ms/summary_<DS>.md`；
论文表见 `table_<ds>.tex` 与合并表 `table_all.tex`。

### 2.1 摆幅 vs seed 噪声

| 数据集 | F1 均值范围 | 均值极差 | 最大 seed-sd | 读法 |
|---|---|---|---|---|
| MSL | 0.9414–0.9490 | 0.0076 | 0.0136 | 极差 < sd，**无 η 效应** |
| SMAP | 0.9705–0.9734 | 0.0029 | 0.0040 | 极差 ≈ 0.7×sd，**无 η 效应** |
| SWaT | 0.9805–0.9850 | 0.0045 | 0.0044 | 极差 ≈ sd，**无 η 效应** |
| WADI | 0.7486–0.9089 | 0.1603 | 0.2015 | 极差 < sd，**无 η 效应** |

**四个数据集上，η 从 0 加到 50% 的均值差都不超过该数据集自身的 seed 波动。**

### 2.2 WADI 的特殊性

- Recall 在 6 个 η 档上全部**精确等于 1.0000（sd = 0.0000）**：
  point-adjust 把所有异常段都判中，F1 完全由 Precision 决定。
- Precision 在 0.61–0.84 之间、单档 sd 最高 0.28 ⇒ F1 的档间差异基本是
  seed 运气，而不是噪声强度。
- η=20% 反而是最高档（0.9089），clean 最低（0.7608）——方向与"噪声导致退化"
  相反，同样不能当结论。

---

## 3. 判据（PRD6 §6）

| # | 判据 | 结果 | 说明 |
|---|---|---|---|
| **D1** | 72 个 JSON 齐全；同 η 下 3 seed 的 F1 不全相同 | **✔** | 72/72，每数据集 18/18 组合，`noise_kind` 全为 `gauss` |
| **D2** | 报告每 η 的 seed-sd；sd 与 η 间差同量级则不能声称退化 | **✔ 报告，✘ 不能声称退化** | 四集极差均 ≤ 最大 seed-sd |
| **D3** | 与 PRD5.2（LSTM-AE）形状一致 | **✔ 都平** | 但来源可能不同，见 §4 |
| **D4** | clean mean 落在既有 clean 值波动内 | **✔（松校验）** | MSL 0.9467 vs 论文旧值 0.9577（W=140）；本轮 W=70 |

> **可以写**：在 `N(0,1)` 整行替换、η ≤ 0.5 这套协议下，PESR 的
> point-adjust F1 在四个数据集上都没有随 η 下降。
> **不能写**：模型鲁棒；也不能说"加噪没有影响" —— F1 在这套评测里
> 无分辨力（WADI 最明显：档间差被 seed 差完全淹没），且本轮未落分数。

---

## 4. 与 PRD5.2（LSTM-AE）并排

| 维度 | PRD5.2 | 本轮 |
|---|---|---|
| η 档 / seeds / 评测 | 同 | **同** |
| 加噪分布 | 混合噪声（U/±5/walk） | **高斯 N(0,1)** ✘ |
| 指标 | F1 + AUROC/AUPRC | **仅 F1** |
| window（MSL） | 70 | **70** ✔ |

只可比较**随 η 的形状**（两者都平）；绝对高低不可读作模型差异。
LSTM-AE 的平来自"分数有结构但噪声确实没影响"，PESR 的平可能是
"分数层接近常量 + point-adjust"的伪平（PRD5.2 §3.2：未调整 AUROC≈0.4975）。
**本轮没有 `--score-dump`，无法区分。**

---

## 5. 边界与未做

- **F1-only**：要支持鲁棒性结论必须补 AUROC/AUPRC（`main.py` 已有 `--score-dump`，
  但需重跑 72 个 run）。
- **sd 是合并方差**：`set_seed` 在 `add_noise` 之前，seed 同时改变噪声实现与
  训练随机性（PRD6 §1.3）。
- **val ⊂ test** 的既存缺陷沿用未改，所有 η 同病。
- 机器负载高时单 run 明显变慢（SWaT ≈9 h、SMAP 并发时 1000–1600 s/epoch），
  但只影响墙钟，不影响各档之间的可比性。
- WADI 的 Recall=1 与 0.5/seed7 的补跑均由调度产生，协议与其它 run 完全一致
  （同一脚本、同一参数、同一 ckpt 隔离）。

---

## 6. 产物

| 类型 | 路径 |
|---|---|
| 逐 run | `results/robust_ms/pesr_<DS>_gauss_noise<η>_seed<s>.json`（72 个） |
| 单集汇总 | `results/robust_ms/summary_<DS>.{md,csv}` |
| 四集汇总 | `results/robust_ms/summary_ALL.md` |
| 单集论文表 | `results/robust_ms/table_{msl,smap,swat,wadi}.tex` |
| 合并论文表 | `results/robust_ms/table_all.tex` |
| 四集图 | `figs/robust_ms/pesr_robust_all_datasets.{pdf,png}`（2×2，轴统一） |
| 单集图 | `figs/robust_ms/pesr_<DS>_robust_two_rulers.{pdf,png}` |
| 与 LSTM-AE 合并图 | `figs/robust_ms/robust_combined_two_rulers.{pdf,png}`（MSL） |
| 调度 | `script/robust/robust_multiseed.sh` |
| 聚合 | `script/robust/summarize_robust_ms.py`、`table_all_datasets.py` |
| 绘图 | `bench/plot_pesr_robust.py`、`plot_pesr_robust_all.py`、`plot_robust_combined.py` |
| 日志 | `nohp-out/robust_ms/*.log` |
| ckpt | `checkpoint/robust_ms/<run>/best.pt` |
