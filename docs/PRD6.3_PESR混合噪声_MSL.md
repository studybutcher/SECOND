# PRD6.3 · PESR 在历史 mixed 噪声协议下的 bf F1（MSL）

> 状态：**已完成（2026-10-20，15/15 runs）。结果与读数见 §5、§13。**
> 依据：`docs/archive/PRD5.2_LSTM-AE_噪声协议结果_20261010.md`（LSTM-AE 的
> mixed 噪声结果）与 `docs/PRD6.1_双模型双阈值_MSL.md`（PESR 的 gauss 结果）。
>
> **一句话**：现有 PESR 的 bf F1 ≈ 0.94 全部来自 `gauss N(0,1)` 训练噪声；
> 本实验只把训练噪声换回历史 `mixed` 协议（U(−2,2)/±5/random_walk），
> 其余训练、划分、评测、seed 全部照抄，回答“PESR 在 mixed 下的 bf F1 是多少”。

---

## 1. 背景与问题

PRD5.2 在 LSTM-AE 上得到两个读数：

- mixed 噪声下 bf F1 在 η=0.05 就掉到 **0.4238 ± 0.196**（clean 0.6765），
  各档均值 0.42–0.55、seed-sd 0.17–0.23；
- 但未调整 AUROC 全程 0.587–0.595，**没有趋势**——F1 的摆动是
  point-adjust + oracle 阈值造成的，不是模型排序能力变化。

PESR（本仓库主模型）目前所有 robustness 结果（PRD6.1、PRD6.2）都用
`--noise-kind gauss`（整行 N(0,1)），bf F1 在 η 0→0.5 为 **0.942–0.949**。
**还没有任何一组 PESR × mixed 的 bf F1。**

本 PRD 只回答一个问题：

> **把训练噪声换成 PRD5.2 的历史 mixed 协议，PESR 的 bf F1 是多少？
> 与 gauss 的 0.94 相比是持平、下降、还是上升？**

不在本 PRD 范围内：POT 重训、LSTM-AE 新实验、噪声强度缩放版本、
mixed 内部三种子噪声的消融、其它数据集。

---

## 2. 固定不动的部分（逐项对齐历史协议）

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | PESR（`--backbone mamba`，201.6 M 参数） | 与 PRD6.1 完全相同 |
| 加噪函数 | `main.py:add_noise(..., kind='mixed')` | **原样复用，不改一行** |
| 加噪位置 | `preprocess` 之后、窗口化之前；**只加在 train_data 上** | test/val 保持干净 |
| η 档位 | 0.05, 0.10, 0.15, 0.20, 0.50 | 再加 η=0（复用，见 §3.2） |
| seed | 42, 7, 17 | 与全部历史实验一致 |
| 数据 | MSL，`/data/sdc/yrlbkp/anomaly` | — |
| W / bs / latent | 70 / 60 / 256 | `main.py` 默认 |
| 训练 | 10 epoch、Adam lr=0.01、每 256 步 `optimizer.step()` | 与 PRD6.1 相同 |
| 划分 | `main.py` 现状（val=0.35×test，**val⊂test 缺陷不修**） | 与历史一致 |
| 评测 | `bf_search(step_num=700)` + `adjust_predicts`，完整 test 47,854 窗口 | 与 PRD5.2/PRD6.1 相同 |
| 结果指标 | **bf F1 为主**（用户指定）；P/R 一并记录 | — |

### 2.1 mixed 噪声的精确定义（`main.py:149-178`）

```python
num_samples = int(data.shape[0] * noise_ratio)
indices = np.random.choice(data.shape[0], num_samples, replace=False)
for i in range(len(indices)):
    noise_type = np.random.choice(['uniform', 'spike', 'random_walk'])  # 各 1/3
    if noise_type == 'uniform':
        noise[i] = np.random.uniform(-2, 2, data[indices[i]].shape)
    elif noise_type == 'spike':
        spike_value = np.random.choice([-5, 5])
        noise[i] = np.full_like(data[indices[i]], spike_value)
    else:
        noise[i] = np.cumsum(np.random.uniform(-0.5, 0.5, data[indices[i]].shape))
data[indices] = noise        # 整行替换
```

要点：

1. **整行替换**，55 个通道同时被覆盖；三种子噪声等概率（1/3）。
2. `±5` 是**绝对常数**（数据已归一化到 [0,1] 附近），不是 σ 的倍数。
3. η 是**被替换的行比例**，不是噪声强度；强度由三种子噪声自己决定。
4. 加噪只消耗 numpy RNG；模型初始化用 torch RNG。`set_seed` 分别播种，
   因此**同 seed 下不同 noise-kind 的模型初始化与 batch 顺序相同**，
   差异只来自训练数据本身。

---

## 3. Run 清单

### 3.1 需要重训的 15 个 run

η ∈ {0.05, 0.10, 0.15, 0.20, 0.50} × seed ∈ {42, 7, 17} = **15 runs**。
每个 run 单卡，约 **65 min**（10 epoch × 382 s + 评测）。

### 3.2 η=0 复用 gauss 的 3 个 run（不重训）

η=0 时 `add_noise` 在 `num_samples <= 0` 处直接返回，**不消耗任何 RNG，
训练数据逐位相同**；torch RNG 又与 noise-kind 无关。因此
mixed(η=0) ≡ gauss(η=0)，直接复用 `results/robust_ms/pesr_MSL_gauss_noise0_seed*`。

> 若审核时希望留一个显式对照，可加跑 **η=0 / seed42** 一次（约 65 min），
> 验证结果与存档 gauss 逐位一致。默认不做。

总计：**15 次重训 + 3 个复用点**。

---

## 4. 评测口径

### 4.1 主指标（用户指定）

`bf_search` 的 oracle 百分位阈值（93→99.99，步长 0.01）+ point-adjust，
在完整 test（47,854 窗口）上取 F1 最大者。与 PRD5.2、PRD6.1 完全同一函数、
同一数据、同一缺陷（val⊂test）。

输出：逐 seed 的 P / R / F1 + 阈值百分位；按 η 报 mean ± sd（n=3）。

### 4.2 顺带记录 POT①（用户 2026-10-20 决定）

训练命令用 `--threshold-mode both --pot-fixed`：同一进程、同一前向的
test 分数上再算一次 **POT①**（`q=1e-3, level=0.98, scale=1.0`），
写入 `results/threshold_ms_mixed/SECOND_MSL_fixed_<run>.json`。

- 代价：一次 `init_score` 前向（58,247 个训练窗口，约 1–2 min）+ 一个 JSON。
- 语义：`init_score` 来自**加噪后的训练集**，与部署时只有污染训练数据一致。
- bf 仍是主指标；POT① 只作为附加列，不替代 bf，也不做 scale 标定。

### 4.3 不做的口径

- **不做 AUROC / AUPRC，不加 `--score-dump`**（用户 2026-10-20 决定）。
  本轮只回答 bf F1（及顺带的 POT①）。
- 不计算 held-out 子集（与 PRD5.2/PRD6.1 的完整 test 对齐）。
- 不改 bf 的搜索范围、步长、point-adjust 实现。
- 不做 POT scale 标定（③），只记固定参数 ①。

---

## 5. 结果表（2026-10-20 实跑，15/15）

### 表 A：bf F1 vs η（主表，mean ± sd，n=3）

| η | PESR gauss（存档） | **PESR mixed（新）** | Δ(mixed−gauss) | LSTM-AE mixed（PRD5.2，参考） |
|---|---|---|---|---|
| 0 | 0.9467 ± 0.0136 | 0.9467 ± 0.0136（复用） | 0 | 0.6765 ± 0.033 |
| 0.05 | 0.9414 ± 0.0128 | **0.9496 ± 0.0146** | +0.0081 | 0.4238 ± 0.196 |
| 0.10 | 0.9420 ± 0.0021 | **0.9522 ± 0.0042** | +0.0102 | 0.5135 ± 0.173 |
| 0.15 | 0.9474 ± 0.0051 | **0.9567 ± 0.0069** | +0.0092 | 0.5472 ± 0.208 |
| 0.20 | 0.9418 ± 0.0088 | **0.9457 ± 0.0024** | +0.0039 | 0.5337 ± 0.232 |
| 0.50 | 0.9490 ± 0.0109 | **0.9393 ± 0.0116** | −0.0097 | 0.5345 ± 0.198 |
| **跨η平均** | **0.9447** | **0.9484** | **+0.0036** | — |

### 表 B：逐 seed（18 行，含 η=0 复用）

| η | seed | bf P | bf R | bf F1 | bf 百分位 | POT① F1 | POT① th |
|---|---|---|---|---|---|---|---|
| 0（复用） | 42 | 0.9440 | 0.9634 | 0.9536 | 99.07 | 0.9269 | 48.280 |
| 0（复用） | 7 | 0.9314 | 0.9807 | 0.9554 | 98.83 | 0.9256 | 80.946 |
| 0（复用） | 17 | 0.8898 | 0.9762 | 0.9310 | 98.11 | 0.9290 | 24.524 |
| 0.05 | 42 | 0.9112 | 1.0000 | 0.9536 | 98.41 | 0.9445 | 122.470 |
| 0.05 | 7 | 0.9435 | 0.9807 | 0.9618 | 99.05 | 0.9335 | 53.818 |
| 0.05 | 17 | 0.8967 | 0.9732 | 0.9334 | 98.22 | 0.9250 | 24.941 |
| 0.10 | 42 | 0.9241 | 0.9839 | 0.9530 | 98.70 | 0.9371 | 197.629 |
| 0.10 | 7 | 0.9525 | 0.9595 | 0.9560 | 99.22 | 0.9299 | 10.576 |
| 0.10 | 17 | 0.9420 | 0.9535 | 0.9477 | 99.05 | 0.9373 | 82.844 |
| 0.15 | 42 | 0.9460 | 0.9657 | 0.9557 | 99.11 | 0.9374 | 186.148 |
| 0.15 | 7 | 0.9380 | 0.9914 | 0.9640 | 98.94 | 0.9378 | 46.086 |
| 0.15 | 17 | 0.9123 | 0.9914 | 0.9502 | 98.50 | 0.9356 | 80.036 |
| 0.20 | 42 | 0.9340 | 0.9521 | 0.9430 | 98.92 | 0.9211 | 172.558 |
| 0.20 | 7 | 0.9329 | 0.9612 | 0.9468 | 98.90 | 0.9256 | 57.178 |
| 0.20 | 17 | 0.9646 | 0.9308 | 0.9474 | 99.45 | 0.9379 | 50.162 |
| 0.50 | 42 | 0.8996 | 0.9818 | 0.9389 | 98.22 | 0.9330 | 90.560 |
| 0.50 | 7 | 0.9233 | 0.9807 | 0.9512 | 98.65 | 0.9412 | 38.978 |
| 0.50 | 17 | 0.9119 | 0.9446 | 0.9280 | 98.53 | 0.9141 | 13.280 |

### 表 C：POT① F1（mean ± sd，n=3）

| η | PESR gauss | **PESR mixed** | Δ(mixed−gauss) |
|---|---|---|---|
| 0 | 0.9272 ± 0.0017 | 0.9272 ± 0.0017（复用） | 0 |
| 0.05 | 0.9211 ± 0.0082 | **0.9344 ± 0.0098** | +0.0132 |
| 0.10 | 0.9265 ± 0.0116 | **0.9348 ± 0.0042** | +0.0083 |
| 0.15 | 0.9318 ± 0.0061 | **0.9370 ± 0.0012** | +0.0052 |
| 0.20 | 0.9368 ± 0.0036 | **0.9282 ± 0.0087** | −0.0086 |
| 0.50 | 0.9309 ± 0.0157 | **0.9294 ± 0.0139** | −0.0015 |
| **跨η平均** | **0.9291** | **0.9318** | **+0.0028** |

---

## 6. 成本

| 项 | 数量 | 单价 | 合计 |
|---|---|---|---|
| mixed 重训（含 POT①） | 15 | ~1.1 GPU·h | **~16.5 GPU·h** |
| 卡 | **0 1 2 3 4** | — | 每卡 3 个 run |
| 墙钟 | — | — | 预估 ~3.3 h；**实测 8.5 h**（GPU 0–2 与他人任务并发） |
| 磁盘 | 15 × 837 MB ckpt | — | ~12.5 GB |

---

## 7. 命名与产物（不覆盖任何现有文件）

| 类型 | 路径 |
|---|---|
| ckpt | `checkpoint/robust_ms_mixed/pesr_MSL_mixed_noise{η}_seed{s}/best.pt` |
| 结果 | `results/robust_ms_mixed/pesr_MSL_mixed_noise{η}_seed{s}.json` |
| POT① | `results/threshold_ms_mixed/SECOND_MSL_fixed_<run>.json` |
| 日志 | `nohp-out/robust_ms_mixed/` |
| 脚本 | `script/robust/pesr_mixed_bf.sh`（新增） |
| 汇总 | `results/robust_ms_mixed/summary.md`、`results/mixed_vs_gauss_bf.md` |

现有 `results/robust_ms/`、`checkpoint/robust_ms/`、`results/threshold_*`
**只读**。

---

## 8. 已知混叠与解释边界

1. **mixed 比 gauss 强得多**：同 η 下 mixed 含 ±5 常数行，gauss 是 N(0,1)。
   因此 mixed vs gauss 的差是**协议级**差异，不能拆成“分布形状”一项。
2. **η 是比例不是强度**：跨 η 比较的是“多少行被污染”，不是“噪声幅度”。
3. **bf 是 oracle 上界**：阈值用 test 标签选，两套噪声同样受益；绝对值偏乐观，
   但内部比较有效。
4. **val⊂test 未修**：与全部历史结果同病，不新增偏差。
5. **`best.pt` 恒为 epoch 0**（PRD6.2 已确认的既存 bug）：mixed 与 gauss
   都受同一 bug 影响，**不修**；修了会引入新的混淆。
6. **PESR 打分仍有 dropout**：bf 在训练进程内计算，与 gauss 存档路径一致；
   同 seed 跨进程不保证逐位相同（PRD6.2 §4.5），故报 3 seeds 的 mean±sd。
7. **F1 接近天花板**：gauss 的 0.94 已靠近 point-adjust 上限，任何下降都更容易
   被看见，但上升空间很小。

---

## 9. 预先登记的判据

| # | 观察 | 解释 |
|---|---|---|
| **J1** | mixed bf 与 gauss 的差在 seed-sd 内 | PESR 的 bf 对噪声分布不敏感；可强化“鲁棒”表述 |
| **J2** | mixed bf 显著下降（如 LSTM 那样到 0.4–0.6） | 现有 gauss 结论是**分布特定**的；论文必须写清训练噪声协议 |
| **J3** | mixed bf 明显高于 gauss | 先怀疑 point-adjust 伪影（分数塌缩抬高 recall），再解释 |
| **J4** | bf 摆幅大但 AUROC 平稳 | 与 PRD5.2 同型：F1 不稳、排序能力不变；主结论改用 AUROC |

无论结果如何，表 B 的逐 seed 值必须保留，不报“最好 seed”。

---

## 10. 需要审核的决定

**用户 2026-10-20 已确认，全部按下述执行：**

| # | 决定 | 结论 |
|---|---|---|
| **D1** | η=0 是否重训 | **复用 gauss 的 3 个 η=0 结果**，不重训、不加对照 |
| **D2** | 是否顺带记 POT① | **记录**：`--threshold-mode both --pot-fixed` |
| **D3** | 是否加 score-dump / AUROC | **不做**：不加 `--score-dump`，不报 AUROC/AUPRC |
| **D4** | 卡数与调度 | **GPU 0 1 2 3 4**（5 卡，每卡 3 个 run，墙钟 ~3.3 h） |
| **D5** | 是否保留 η=0.5 | **保留** |

---

## 11. 代码改动清单

1. **新增** `script/robust/pesr_mixed_bf.sh`：以 `lstmae_gauss_pot.sh`
   为模板，改五处——
   - `--noise-kind mixed`（原脚本写死 gauss）；
   - ckpt → `checkpoint/robust_ms_mixed/...`；
   - 结果 → `results/robust_ms_mixed/...`；
   - POT → `results/threshold_ms_mixed/...`；
   - `--backbone mamba`、η 网格去掉 0。
2. 复用现有 `--ckpt-dir`、`--result-name`、`--pot-dir`、`--pot-result-name`，
   **不改 `main.py`**。
3. **不改**：`add_noise`、模型、`bf_search`、`adjust_predicts`、数据划分、
   训练超参、dropout、POT 相关代码。
4. 新增汇总脚本 `bench/summarize_mixed_vs_gauss.py`（读两目录 JSON，
   产出表 A/B/C），不改动已有汇总脚本。

---

## 12. 执行顺序（审核通过后）

1. 写 `pesr_mixed_bf.sh`，先跑 **η=0.05 / seed42** 单点 smoke（65 min），
   确认：日志里 `noise_kind=mixed`、ckpt/结果路径正确、bf F1 可读出。
2. smoke 通过后并行铺满 15 runs（6–8 卡）。
3. 跑 `bench/summarize_mixed_vs_gauss.py`，产出表 A/B/C 与本 PRD 的“结果”节。
4. 只在 §9 的判据下解读，不改口历史结论。

---

## 13. 结果与读数（2026-10-20）

### 13.1 执行记录

- **15/15 runs 成功**（rc=0），bf JSON 15/15、POT① JSON 15/15；
  smoke（η=0.05 / seed42）于 02:43–03:54 通过后铺满。
- 墙钟 02:43–11:13（~8.5 h），长于预估的 3.3 h：GPU 0/1/2 与他人任务并发，
  单 run 从 65 min 拉长到 1.5–2 h。GPU·h 仍是 ~16.5（5 卡 × 3 run × ~1.1 h）。
- 产物：`results/robust_ms_mixed/`、`results/threshold_ms_mixed/`、
  `checkpoint/robust_ms_mixed/`（12 GB）、`nohp-out/robust_ms_mixed/`。
- 汇总：`results/robust_ms_mixed/summary.md`、`results/mixed_vs_gauss_bf.md`
  （由 `bench/summarize_mixed_vs_gauss.py` 生成）。

### 13.2 核心读数

1. **PESR 在 mixed 下没有退化。** 跨 η 平均 bf F1：mixed **0.9484** vs
   gauss **0.9447**（Δ=+0.0036）；POT①：**0.9318** vs **0.9291**
   （Δ=+0.0028）。差异全部落在 seed-sd（±0.002–0.015）量级。
2. **逐 η 没有一致方向。** 5 档中 4 档 mixed 高 0.004–0.010，η=0.5 低
   0.0097。η=0.05–0.15 的 +0.008–0.010 略高于该档 sd，但绝对量级仍只有
   约 1 个 F1 百分点，不能当作稳定增益。
3. **与 LSTM-AE 形成强对照。** 同一 mixed 协议、同一 η=0.05：
   LSTM-AE 从 clean 0.6765 掉到 **0.4238**（−0.25）；PESR 从 0.9467 到
   **0.9496**（+0.003）。但两者 clean F1 差 0.27（headroom 不同），
   只能写“PESR 在本协议下没测到退化”，不能写“PESR 更鲁棒”。
4. **bf 阈值百分位集中在 98.1–99.5**；POT① 的绝对阈值跨 seed/η 从
   **10.6 到 197.6**，再次说明 PESR 的分数尺度随 ckpt 大幅变化，
   POT① 的绝对阈值不可跨 run 复用（与 PRD6.2 结论一致）。
5. **Recall 被 point-adjust 顶满。** 15 个 mixed run 的 recall 为
   0.93–1.00，其中 η=0.05/seed42 的 FN=0；F1 主要由 precision 决定，
   ±0.01 的差很容易被单个 FP 段放大——这是解释边界，不是模型差异。

### 13.3 对 §9 判据的命中

| # | 判据 | 结果 |
|---|---|---|
| **J1** | mixed 与 gauss 的差在 seed-sd 内 | **命中**（bf +0.0036，POT① +0.0028） |
| **J2** | mixed bf 显著下降 | 未命中 |
| **J3** | mixed bf 明显更高 | 未命中（+0.0036 不算明显） |
| **J4** | bf 摆幅大而 AUROC 平稳 | 无法判定（本轮按决定未算 AUROC） |

### 13.4 能写 / 不能写

- **能写**：在 `script/robust` 的历史 mixed 协议（整行替换
  U(−2,2)/±5/random_walk，η≤0.5）下，PESR 的 bf F1 与 gauss 协议
  **无实质差异**（0.948 vs 0.945；POT① 0.932 vs 0.929）。
- **不能写**：PESR 对任意噪声鲁棒；也不能用这组 F1 证明“排序能力不变”
  ——本轮没有 AUROC。
- 与 PRD5.2 的 LSTM-AE 结果并列时，必须注明两者 clean F1 相差 0.27、
  headroom 不同，不能直接比较“退化幅度”。
