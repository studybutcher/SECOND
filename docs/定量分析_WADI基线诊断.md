# 定量分析 · WADI 基线诊断

> 起因为「A2 的 WADI 基线 ~0.951 偏低，历史最高到过 0.9839，怀疑设置有问题」。
> 本文只记录**可复核的证据与结论**；无法核实的部分明确标为「未决」，不写成结论。
> 诊断日期：2026-10-07。全部结论均可由仓库内的文件与一条命令复现。

---

## 0. 结论速览

| # | 结论 | 证据强度 |
|---|---|---|
| A | **0.9839 在仓库中不存在**，最接近是 0.9831（`attnmamba+blockize`）与 0.9859（MUTANT，见 `docs/两库阈值对照结果.md`） | 已核实（全仓扫描） |
| B | **评测管线是稳定的**：用当前代码重跑历史 ckpt，F1 与历史记录**逐位相同** → 历史数字不是评测口径漂移出来的 | 已核实（逐位复现） |
| C | `results/ablation/mamba_WADI_seed42.json` **不是 PRD1 的基线**：它是 `1-casestudy` 的 run 落在 ablation 目录里的残留，且用的是 **W=140**，而 PRD1 的 transformer/lstm 用的是 **W=70** | 已核实（config + ckpt 路径） |
| D | 所有可可靠归因的 **WADI F1 ≥ 0.9776 的记录，公共因子是 `window_length=140`**；当前代码对 WADI 的默认是 **70** | 已核实（逐块解析，94 字符分隔线） |
| E | `nohp-out/Robust/WADI.out` 中存在一个**被污染的日志块**（与 `no-patch` 的 W=140 run 的 th/TP/FP **逐位相同**但 header 写着 bs=24/W=70）。**据此撤回**"同设定（bs24/W70）也到过 0.9914"这一推断 | 已核实（逐位比对） |
| F | `--batch_size` 对 WADI **被静默覆盖**（`ExpConfig` 在 `parse_args` 之后无条件置 24），历史最优的 bs=32 在修复前**无法复现**。**已修复** | 已核实（实测 + 已改代码） |

**净结论**：你的「基线有问题」成立，但**问题不在"W/bs 选低了导致掉到 0.95"**，而在
① 目录里那份"mamba 基线"是别的 run 的残留（结论 C）；
② 高 WADI 分数的公共因子是 **W=140**，而当前默认是 70（结论 D）——这才是"设置"层面的真差异；
③ 一个 CLI 参数被静默吃掉（结论 F）。
"0.95 vs 0.98" 里有多少是**单次运行方差**，见 §8 的干净对照。

---

## 1. 方法与可复现命令

1. 全仓扫描所有含 WADI 指标的结果文件（`results/**`、`nohp-out/**`、`docs/**`）。
2. 解析 `nohp-out/**` 日志的 run 块。**注意**：`main.py` 打印的分隔线是 **94** 个 `=`，解析时必须按 94 切；按 90 切虽不致命，但任何按"第 N 个块"定位的做法都会错位。复核命令见 §7。
3. 用当前代码重跑历史 checkpoint 的 test，与历史记录逐位比对（结论 B）。

```bash
# 结论 B 的复现（同 seed / 同 noise / 同 W）
CUDA_VISIBLE_DEVICES=1 python main.py --dataset WADI --trainmode test --backbone mamba \
  --window_length 140 --batch_size 24 --seed 42 \
  --loadpath checkpoint/1-casestudy/WADI/best.pt \
  --result-dir results/evalcheck --result-name 1casestudy_WADI_W140_reeval
```

---

## 2. 结论 A：0.9839 查无此值

`docs/` + `results/` + `nohp-out/` 全量扫描，WADI 的最高 F1：

| F1 | 来源 | 模型 | bs | W |
|---|---|---|---|---|
| **0.9914** | `nohp-out/ablation/no-patch.out` | `Model-nopatch`（**MSSA 之外，MPA 也被删掉**） | 32 | **140** |
| 0.9859 | `docs/两库阈值对照结果.md:47` | **MUTANT** | — | 5 |
| 0.9831 | `nohp-out/attnmamba+blockize.out` | 旧 `attnmamba` | 32 | **140** |
| 0.9776 | `nohp-out/可视化retrain（找不到之前的了）.out` | — | 24 | **140** |
| 0.9749 | `nohp-out/重复测试/WADI.out` | — | 24 | **140** |
| 0.9722 | `nohp-out/ablation/patch中加了attention…out` | — | 24 | 70 |
| 0.9695 | `results/ablation/transformer_WADI_seed42.json` | PRD1 transformer | 24 | 70 |
| 0.9588 | `results/contam/nomssa_WADI_rho00_seed42.json` | **本次 A2 · nomssa** | 24 | 70 |
| 0.9510 | `results/ablation/mamba_WADI_seed42.json` | 见结论 C | 24 | **140** |

> **0.9830 作为 precision 出现过两次**（`no-patch` 的 WADI 块、`Robust/WADI.out`），容易被记成 0.9839。
> `results/threshold/scores/sweep.json` 里 WADI 的**oracle 上限**是 `ceil.f1 = 0.9511`，但那是针对 §3 那个 W=140 的 ckpt，不是全局上限。

---

## 3. 结论 B：评测管线稳定（排除"评测侧漂移"）

`checkpoint/1-casestudy/WADI/best.pt`（W=140）用**当前代码**重跑：

```
precision: 0.9066450398102311  recall: 0.9999999428514318  f_score 0.9510320390856472
```

与其 2026-09-14 的记录 `0.9510320390856472` **逐位相同**（阈值 `th` 也相同）。
→ 历史高分不是评测代码变化造成的；差异只能来自**模型/训练设置**。

---

## 4. 结论 C：ablation 目录里那份"mamba 基线"是张冠李戴

| 文件 | backbone | W | bs | ckpt | 时间 |
|---|---|---|---|---|---|
| `results/ablation/mamba_WADI_seed42.json` | mamba | **140** | 24 | `checkpoint/**1-casestudy**/WADI/best.pt` | 2026-09-14 |
| `results/ablation/transformer_WADI_seed42.json` | transformer | 70 | 24 | `checkpoint/ablation/transformer/WADI/best.pt` | 2026-09-12 |
| `results/ablation/lstm_WADI_seed42.json` | lstm | 70 | 24 | `checkpoint/ablation/lstm/WADI/best.pt` | 2026-09-12 |

`mamba` 那条的 ckpt 路径指向 **`1-casestudy`** 而不是 `ablation/`，且 W 与另两条不一致。
即：**PRD1 的 WADI 消融（两个变体都是 W=70）被与一个 W=140 的、来自别的实验的 run 放在同一目录里。**

- `results/ablation/summary.md` 已声明"对照的 mamba 基线数据由另处提供，不在本汇总内"，**所以该汇总本身没混**；
- 但 `docs/消融实验结果.md` §3 写的"训练设置……**全部沿用基线**"在 WADI 这一列**对不上**——同目录那份"基线"是 W=140。
- 附带风险：任何按 `results/ablation/*.json` 聚合的脚本都会把它当基线抓走。

---

## 5. 结论 D：高 WADI 分数的公共因子是 `window_length=140`

把可可靠归因的记录按 W 分组：

| W | 记录 | F1 区间 |
|---|---|---|
| **140** | `no-patch` 0.9914、`attnmamba+blockize` 0.9831、`可视化retrain` 0.9776、`重复测试` 0.9749、`1-casestudy` 0.9510 | 0.9510 ~ **0.9914** |
| 70 | `patch+attention` 0.9722、`transformer` 0.9695、`nomssa`(本次) 0.9588、`lstm` 0.8536 | 0.8536 ~ 0.9722 |

**≥0.9776 的每一条都是 W=140。** 而当前 `ExpConfig` 对 WADI 用的是默认 `window_length=70`（WADI 分支只覆盖 `batch_size`，不覆盖 `window_length`）。

> ⚠️ 这不是"W=140 必然更好"的因果证明——W=140 的高分记录同时换了模型（`no-patch`/`attnmamba`）。但它是**唯一同时出现在所有高分记录里的设置**，因此是最该优先排查的一项。
> §8 给出本次可做的干净对照。

---

## 6. 结论 E：一份日志块被污染（据此撤回一个推断）

`nohp-out/ablation/no-patch.out` 的 WADI 块与 `nohp-out/Robust/WADI.out` 的第 11 块：

| 来源 | header 里的设置 | th | TP | FP | P | F1 |
|---|---|---|---|---|---|---|
| `no-patch.out` | bs=**32**, W=**140** | 99.97000000000357 | 174.98250174982496 | 3.0174982501749854 | 0.983047707412067 | 0.9914463667777642 |
| `Robust/WADI.out` #11 | bs=**24**, W=**70**, noise=0.1 | 99.97000000000357 | 174.98250174982496 | 3.0174982501749854 | 0.983047707412067 | 0.9914463667777642 |

**th / TP / FP 到 16 位有效数字完全相同。** 不同模型不可能给出逐位相同的混淆计数，而 W=140 与 W=70 的 `simpembdding = DataEmbedding(W, 512)` 权重形状不同（`Conv1d(W→512,k=3)`），也不可能互相加载。故**其中一个块是重复/陈旧的输出**。

→ 本项目早前的一轮口头推断"同设定（bs24/W70）也到过 0.9914"**建立在被污染的块上，本文撤回**。`Robust/WADI.out` 的其余块（0.6769 ~ 0.9668）不受影响。

> 佐证：本仓存在 `nohp-out/重复测试/{WADI}.out` 这种文件名里带 `{WADI}` 的日志，说明结果文件有过手工搬运。此类文件的 provenance 需谨慎对待。

---

## 7. 结论 F：`--batch_size` 对 WADI 被静默覆盖（已修复）

`main.py` 的 `ExpConfig.__init__` 原先在 `parse_args` **之后**执行数据集的默认值覆盖：

```python
self.parse_args(namespace=self)
if self.dataset == 'SWaT':
    self.window_length = 35
elif self.dataset == 'WADI':
    self.batch_size = 24        # ← 无条件覆盖，命令行值失效
```

实测（修复前）：

```
CLI []                        -> window_length=70  batch_size=24
CLI ['--batch_size','32']     -> window_length=70  batch_size=24   ← 被吃掉
CLI ['--window_length','140'] -> window_length=140 batch_size=24
```

即 **`--batch_size` 对 WADI 完全无效，`--window_length` 有效**。历史最优用的 bs=32 在那个写法下无法复现。

**修复**：把两个参数的 argparse 默认值改为 `None`，只在命令行**没有**显式给出时才套数据集默认值。已验证：

- 不带参数时四个数据集的 `(W, bs)` 与修复前**完全一致**（SMAP/MSL 70/60、SWaT 35/60、WADI 70/24）→ 向后兼容；
- 显式传参现在生效（WADI `--batch_size 32` → 32；`--window_length 140 --batch_size 32` → 140/32）。

---

## 8. 一个干净的单变量对照（本次即可做）

A2 里 `mamba` 那条跑的是 **W=70, bs=24, seed=42, noise=0**；
`checkpoint/1-casestudy/WADI/best.pt` 是 **W=140, bs=24, seed=42, noise=0**（§3 已复现其 0.9510）。

→ **同一模型、同 seed、同 noise、同 bs，只差 W**。两条跑完即可得到 W=70 vs W=140 的单变量对照，无需额外成本。

| 单元格 | F1 |
|---|---|
| mamba, W=140, bs=24（1-casestudy，已复现） | 0.9510 |
| mamba, W=70, bs=24（A2，待出） | *待填* |

> 若 W=70 ≈ W=140 → 结论是"W 不重要，0.95↔0.98 的差是运行方差"；
> 若 W=140 明显更高 → 支持结论 D，WADI 的默认 W 需要重新选择。
> **在拿到这两个数之前，不要根据单次结果调整 WADI 的设定。**

---

## 9. 未决问题（需要你确认，本文不做推断）

1. **0.9839 究竟出自哪个 run？** 仓库里查无此值。若它在别处（另一个仓库 / 本地表格 / `TINA/` 等目录），给我路径我去核对；若就是 `attnmamba+blockize`(0.9831) 或 `no-patch`(0.9914) 那批，则结论是**"WADI 最高分来自没有 MPA 的模型"**——这对 PRD4text 的叙事方向影响很大，需要先定口径。
2. **`Robust/WADI.out` #11 那个块是谁写进去的？** 若是手工粘贴，说明该文件的其余块也需复核。
3. **WADI 的运行方差到底多大？** 现有记录里同设定跨 run 的 F1 跨度很大（`Robust` 那批 0.6769~0.9668），但样本量小且可能含污染块。需要专门的多 seed 实验才能定 `docs/消融实验结果.md` 那个"±0.005 噪声底"是否对 WADI 成立（该噪声底是从 SMAP 两个样本估的，对 WADI 很可能低估）。

---

## 10. 行动项

| # | 动作 | 状态 |
|---|---|---|
| 1 | 修 `--batch_size` 静默覆盖（结论 F） | ✅ 已完成，向后兼容 |
| 2 | 备份 `results/ablation/*.json`（13 个）并给污染实验加 `--result-dir`，避免覆盖 PRD1 结果 | ✅ 已完成 |
| 3 | 把 `results/ablation/mamba_WADI_seed42.json` 从基线口径中**剔除**（它是 W=140 的 1-casestudy 残留）；PRD1 的 WADI 基线需**重跑**（W=70 才算同口径） | ⏳ 待定 |
| 4 | 补 W=70 的 mamba WADI 基线（A2 已在跑，即 §8 对照的下半格） | ⏳ 在跑 |
| 5 | WADI 上任何"变体 vs 基线"的结论**必须多 seed 报 mean±std**；单次 0.95 vs 0.98 在 WADI 上不可区分 | ⏳ 未做 |
| 6 | 复核 `docs/消融实验结果.md` §3 的 WADI 行与"训练设置全部沿用基线"这句表述 | ⏳ 待定 |
