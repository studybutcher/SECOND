# PRD 2：推理成本统计（延时 / 算力 / 内存）

> 2026-09-12 ｜ 仓库：SECOND ↔ 对照：MUTANT（`anamolybench/MUTANT`）｜ 状态：待评审
> 关联文档：`PRD1_消融实验.md`（训练侧消融另开一份）

---

## 1. 目标

产出**延时 / 计算成本 / 内存**三个指标，使 SECOND 能与 MUTANT（及同目录 `AMSL / DTAAD / GDN / MTAD-GAT / USAD`）**放进同一张表**。

**F1 侧已经天然可比**（逐行 diff 确认）：同一份 npy 数据、`preprocess` 完全一致、`bf_search(93→100, step 0.01, 700)` 数学完全一致、切分与训练超参一致、都是每窗口 1 个分数。**所以只需要建成本侧协议。**

**各方法超参不同，不强行统一，但逐行披露**：

| | SECOND | MUTANT |
|---|---|---|
| `window_length` | 70（SMAP/MSL/WADI）、35（SWaT） | 20（SMAP/MSL/SWaT）、5（WADI） |
| `batch_size` | 60、24（WADI） | 120 |
| `latent_size` | 256 | 100 / 120 |
| 精度 | 全 FP32 | GCN 是 **FP64**（`Model.py:45`） |

---

## 2. 核心实测（决定了协议长什么样）

### 2.1 batch 曲线 —— 这是本 PRD 最重要的一张表

SMAP，同卡独占（RTX 5880 Ada），端到端推理路径：

| batch | **SECOND** ms/window | **MUTANT** ms/window |
|---|---|---|
| 1 | 34.80 | **5.32** |
| 8 | 4.21 | 4.03 |
| 32 | **1.08** | 3.75 |
| 60 / 64 | **0.87** | 3.72 |
| 120 | — | 3.72 |
| **提升幅度** | **40×** | **1.43×** |

**交叉点在 batch≈8。** 两个结论：

1. **batch=1 只测"单请求延迟"，会掩盖真实差异。** SECOND 在 batch=1 输 6.5×，在 batch≥32 赢 3.5~4.3×。
2. **SECOND 的并行性可以被 batch 摊薄（40×），MUTANT 不能（1.43×）** —— MUTANT 的 `returnA` 是逐样本在 CPU 上做 corrcoef 并读写磁盘，batch 再大也躲不掉。**这才是该写进论文的图。**
3. 附带现象：SECOND 在 **batch 1→32 之间总耗时几乎恒定**（34.8 → 34.5 ms/batch），加 batch 是纯赚，32 之后才线性增长。

**为什么 batch=1 这么慢**：中间块的"序列维"是**通道维 `l=25`**（不是时间维），Mamba 并行扫描需要长序列，`l=25` 时 pscan 只有 5 轮、张量极小；每层约 60~80 个 kernel，8 层 ≈ 500~600 个 kernel，每个 launch 10~20µs → 就是那 18.7ms。**是 launch-bound，不是算力问题**（有效算力 273 GFLOPs/s）。

### 2.2 torch.compile 验证了上述诊断

| batch | eager | +torch.compile | 加速 |
|---|---|---|---|
| 1 | 33.30 | **7.96** | **4.18×** |
| 32 | 31.73 | 27.93 | 1.14× |
| 60 | 49.10 | 53.67 | 0.92× |

batch=1 提速 4.2×、batch≥32 零收益 → 证明小 batch 全是 launch 开销。**编译只在 batch=1 有意义，大 batch 别开。**

### 2.3 算力差距（躲不掉，必须如实报；口径见 §4.1）

SMAP，batch=1 单窗口前向：

| | SECOND | MUTANT | 比值 |
|---|---|---|---|
| params | 193.52 M | 0.1256 M<sup>†</sup> | **1541×** |
| GFLOPs/window（计数器原值） | 5.1267 | 0.0001215 | 42,196× |
| **GFLOPs/window（修正后）** | **5.1746** | **0.0002215** | **23,361×** |
| **GFLOPs/时间步（修正后）** | **0.0739** | **1.11e-05** | **6,675×** |
| 窗口长度 W | 70 | 20 | 3.5× |

<sup>†</sup> MUTANT 的 `att_encoder.h_n/s_n` 是 `(1, batch, hidden)` 的 Parameter（`Model.py:82-83`），参数量随 batch 线性增长，故按部署 batch=120 折算（同一份 ckpt 在 batch=1 下 `sum(p.numel())` = 0.1018 M，两者都对，只是口径不同，须注明）；SECOND 的参数量与 batch 无关（`self.batch_size` 在 forward 中未被使用，用的是运行时 `len(x)`）。

**两个必须同时讲的比值**：

- 修正后仍有 **2.3 万倍算力差，而延时只差 3.5~6 倍** —— 说明两边都不是算力瓶颈。只报一个延时列，读者无法解释这张表；必须同时给 params / FLOPs（算法成本）与分解后的延时（部署成本）。
- **两库的 W 不同（70 vs 20）**，所以"差 2.3 万倍"里有一部分是窗口长度带来的。归一化到时间步后是 6,675×，这才是算法层面的可比数字。**跨库表必须同时给 per-window 与 per-timestep 两列**，理由见 §3-P6。

### 2.4 MUTANT 的 LSTM 实际只吃到 1 个时间步（结构性，不是计数问题）

`Model.py:122-128` 的循环里，LSTM 收到的不是聚合后的 `X_tilde`，而是**循环变量 `x_tilde`**（循环结束后只剩最后一轮的值）：

```python
for t in range(self.window_length):
    x_tilde = torch.mul(alpha, X[:, t, :])   # x_tilde: [B, C]
    X_tilde[:, t, :] = x_tilde               # 聚合结果写进 X_tilde
self.encoder_lstm.flatten_parameters()
_, final_state = self.encoder_lstm(x_tilde.unsqueeze(0), (self.h_n, self.s_n))
#                                  ^^^^^^^ 传的是 x_tilde，不是 X_tilde
```

实测 hook 到的 LSTM 输入形状为 **(seq=1, batch=B, input=C)**，时间步维被压成 1。后果（SMAP）：

| | as_shipped（seq=1） | 若按代码意图喂满 W 步 |
|---|---|---|
| LSTM 成本 | 100,000 FLOPs | 2,000,000 FLOPs |
| 修正后总量 | 221,500 | 2,121,500 |
| 相对计数器低估 | 1.82× | **17.5×** |

**处理口径**：主表按 **`as_shipped`** 报 —— 这才是该基线实际跑出的成本，也是它那一列 F1 的来源。附录注明"其 LSTM 实际只处理 1 个时间步；若按其论文意图喂满窗口，LSTM 成本 ×20"。**不要替基线"修 bug"后报数**，那等于和一个不存在的模型比较。同理适用于所有对照库：**只测、只披露、不改**。

---

## 3. 协议（7 条）

### P1 主口径 = batch 曲线，batch=1 只作"最坏情况"一行

主列用**吞吐**（batch ≥ 32 的 ms/window + windows/s），batch=1 单独一行如实报。公共测点集 `{1, 8, 32, 64}`（MUTANT 的 `h_n/s_n` 是 `(1, batch, hidden)` Parameter，上限 120；各自 operational batch 另标一行）。

### P2 延时分层，不报单一数字

```
T_total = T_prep + T_model + T_overhead
```
| 库 | `T_prep` 内容 |
|---|---|
| MUTANT | `returnA` 图构建（`np.corrcoef` + **写 tmp 文件到磁盘再读回**），实测占 **68.5%** |
| SECOND | 样本级 python 循环与 GPU→CPU 同步（`Model.py:177-180`） |

实现方式为**测量侧 hook，不改被测量代码**（MUTANT 用预计算邻接矩阵替换 `Model.returnA`，数学等价）。

### P3 双轨报告

| 变体 | 含义 |
|---|---|
| `as_shipped` | 按仓库原样跑，含实现性开销 |
| `core_only` | 用等数学替身绕过实现性开销，反映算法本身成本 |

两列都给，论文里明确区分 —— 既不被批"藏开销"，也不被批"拿实现差异当卖点"。

### P4 统一 schema + harness 指纹

`bench/benchmark_infer.py` **逐字节相同**地放进两个仓库，差异只由各自的 `bench/adapter_<model>.py` 承担：

```python
def build(dataset, ckpt_path, batch_size) -> nn.Module   # 已 .cuda().eval()
def forward_once(model, x) -> Tensor                     # 该库完整的单次推理路径
def core_only(model) -> nn.Module                        # 可选
def input_shape(dataset, batch_size) -> tuple            # [B, W, C]
```

每次测量把脚本的 **SHA256 写进 JSON**。论文可写"所有方法均使用同一份测量脚本（SHA256 `…`）在同一张卡上测量"。

输出目录与 MUTANT 的 `PRD_supplementary.md` 对齐：

```
<repo>/results/{dataset}/bench/{flops,latency_bs*,memory}.json
```

JSON 必备字段：`model` / `config`(各方法超参) / `cost`(`n_params`, `n_params_learned`, `n_params_state`, `macs_per_window`, **`flops_counter`**, **`flops_corrected`**, **`correction_factor`**, `flops_per_timestep_corrected`, `counter_ops_seen`, **`counter_ops_missing`（须为实测准确的算子名）**, `correction_formula_ref`) / `latency`(batch, mean/p50/p95, `breakdown`, `variant`) / `memory` / `protocol`(**`harness_sha256`**, input_shape, precision, `torch_num_threads`, warmup/iters/repeats, `cudnn_benchmark`) / `env`(gpu, 独占标志, cuda, torch) / `metrics`(F1/P/R) / `ckpt.sha256`。

### P5 共享环境变量（易被忽略的偏差源）

| 变量 | 为什么 |
|---|---|
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `torch.set_num_threads` | MUTANT 那 68.5% 在 CPU 上，**线程数直接决定耗时**（其 `run_SMAP.sh` 现设 32）。建议两库统一为 **1**（最可复现）并记录实际生效值 |
| **cwd 所在文件系统** | MUTANT 的 `returnA` 每样本读写磁盘，SSD/HDD/tmpfs 差异直接进延时 |
| 精度 | MUTANT 的 GCN 是 fp64，SECOND 全 fp32，分组件记录 |
| `CUDA_VISIBLE_DEVICES` + 独占断言 | 卡 0/2/5 空闲（1/4 卡 100%，3/7 卡显存近满） |

### P6 归一化：per-window + per-timestep 双列（**跨库可比性的核心**）

**两库的 W 不同（SECOND 70/35，MUTANT 20/5），所以"GFLOPs/窗口"这一列天生不可比** —— 它同时混进了算法成本和窗口长度两个量。协议要求：

| 列 | 用途 | 陷阱 |
|---|---|---|
| GFLOPs / window | **部署成本**（用户实际每次推理付多少） | 不可跨库直接比 |
| GFLOPs / timestep | **算法成本**（每步数据的处理代价） | 主对比列 |
| ms / window | 部署吞吐 | 主列（batch≥32） |
| ms / timestep | 与上一行配套 | 必须与 FLOPs 用同一归一化 |

**只报 per-window 会在两库 W 差 3.5× 时给出误导性的 42,196×；per-timestep 才是可比的 6,675×。** 同理，参数量列也须注明折算 batch（见 §2.3 †）。

### P7 修正项必须与原始项同表出现

任何"手工补算/口径修正"都不得只出现在脚注里 —— 必须与原始读数并列成列，让读者能自己复算。理由：本 PRD 实测的修正幅度从 1% 到 82% 不等，**量级差异过大，脚注承载不了**。

---

## 4. 三个指标怎么测

### 4.1 计算成本

- 工具：**PyTorch 自带 `torch.utils.flop_counter.FlopCounterMode`**（不引入 thop，零依赖；两库共用同一份计数器）
- 口径：**batch=1 单窗口前向**，`eval`，无 backward，FP32（MUTANT 的 GCN 走 FP64，按 §3-P5 单列）
- 实测 batch 无关性：SECOND bs=1/2/4/8 → 5.1267/5.1268/5.1268/5.1270 G per-sample，差 <0.01%

**覆盖率实测（2026-10-01 复测，逐库 instrument 验证）**：

| 库 | 计数器看到 | 漏掉 | 漏算占比 | 低估倍数 |
|---|---|---|---|---|
| SECOND | convolution / addmm / mm / bmm | `einsum`（广播乘，`mamba.py:489-490`）、`pscan`（in-place）、逐元素激活 | **0.93% ~ 1.15%** | 1.009 ~ 1.012× |
| MUTANT | addmm / **mm（含 `torch.spmm`）** | **`nn.LSTM`（`Model.py:88`）** | **30.4% ~ 45.1%** | **1.44 ~ 1.82×** |

**两条对原稿的纠正（均已实测，直接改结论）**：

1. **`spmm` 没有被漏掉。** `returnA` 返回的是 `nfadj.toarray()`（`Graph.py:64`）——**稠密** ndarray；`torch.spmm(dense.double(), support.double())`（`Model.py:45`）实际派发到 `aten.mm`，实测记为 6,250 FLOPs/sample（= 2·25·25·5，恰好是稠密邻接的代价）。与每样本 `mm` 总计数 11,250 = 5,000（`torch.mm`）+ 6,250（spmm）逐位吻合。
   > 附带结论：MUTANT 名为 sparse GCN，**实际付的是稠密 C²·OUT 的代价**，论文里不能替它写 "sparse"；也不存在"fp64 导致漏算"这回事。
2. **LSTM 确实全漏，而且不是零头。** `nn.LSTM` 走 ATen/cuDNN 融合算子，计数器对它完全不可见。

**MUTANT 漏算量化（四数据集，batch=1）**：

| 数据集 | C/W | 计数器 | LSTM 漏算 | 修正后 | 低估 |
|---|---|---|---|---|---|
| SMAP | 25/20 | 121,500 | 100,000 | 221,500 | **1.82×** |
| MSL | 55/20 | 283,800 | 124,000 | 407,800 | 1.44× |
| SWaT | 51/20 | 261,120 | 120,800 | 381,920 | 1.46× |
| WADI | 127/5 | 528,066 | 237,120 | 765,186 | 1.45× |

**这个修正幅度已经超过"脚注级"**：SMAP 的算力比从 42,196× 变成 23,361×，**接近腰斩**。且修正方向是**抬高 MUTANT 的成本**，即修正后对 SECOND 更有利 —— 不修正反而会被审稿人指为"用错误的计数器抬高自己"。

**报数规则（§3-P4 强制字段）**：每行同时给 `flops_counter`（原值）、`flops_corrected`（原值 + 附录 B 公式）、`counter_ops_missing`（**须为实测准确的算子名**）、`correction_factor`。**表主列用修正值，脚注给修正系数。**

### 4.2 延时

```python
def bench(fn, warmup=20, iters=100, reps=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)/iters*1000)
    return mean(ts), stdev(ts)
```

三个测点：**推理路径**（`reconstructed_probability`，与 MUTANT 对称）、**中间块 8 层微基准**、**全测试集总时**。报 mean / p50 / p95。batch ∈ {1,8,32,64} + 各自 operational batch。

### 4.3 内存

`empty_cache()` → `reset_peak_memory_stats()` → 记 `memory_allocated()`（基线）→ 推理 → `max_memory_allocated()` / `max_memory_reserved()`。
报三数：`after_load` / `inference_peak` / `activation_delta`。`max_memory_allocated` 是进程内计数，不受同卡其他进程影响。

---

## 5. 改动清单（A1~A6 仅 SECOND；A7~A8 为两库共用的 harness，落在本仓 `bench/`）

| # | 项 | 说明 |
|---|---|---|
| A1 | `bench/benchmark_infer.py` + `bench/adapter_second.py` | 通用 harness |
| A2 | `main.py` 加 `--trainmode profile` 分支 | 加载 ckpt → eval → FLOPs → 计时推理 → 显存 → 写 JSON。**`get_data`/`get_loader`/`is_anomaly`/`bf_search` 全部复用**。⚠️ `main.py:149/178` 是子串匹配 `'train' in ...`/`'test' in ...`，mode 名不能含这两串，`profile` 满足 |
| A3 | **P1**：推理前加 `model.eval()` | 现在没有 → `Dropout(0.1)` 推理时是激活的（`attentionblock.py:46,75,105`）。`test` 分支是否一并改，先做 A/B 量化 F1 差值再定（MUTANT 无 dropout，对它是 no-op，不影响对照） |
| A4 | **P2**：向量化逐样本循环 | `Model.py:177-180` 改为 `(x_a-mu).abs().sum(dim=list(range(1,x_a.dim())))`，数学等价。需 A/B 确认 F1 差 <1e-6。**这也是 `core_only` 变体的定义** |
| A5 | 补 seed 控制 | SECOND **当前完全没有**（MUTANT 有），对齐 `MUTANT/main.py:43-47`，~10 行 |
| A6 | 结果落盘 JSON | 现在只 `print`，加 `results/{dataset}/metrics_seed{S}.json` |
| A7 | **修 harness 的覆盖率元数据（两库都要）** | `collect_cost_mutant.py:157` 的 `counter_ops_missing` 现为 `['spmm(GCN, fp64)', 'LSTM(encoder_lstm)']` —— **`spmm` 一条是错的**（实测已计入，见 §4.1 纠正 1），须改为 `['LSTM(encoder_lstm)']`；`coverage_note` 引用的「PRD2 附录 B」现已存在（本文件），须同步 |
| A8 | **加 `flops_corrected` 计算并重出全部 JSON** | 现有 8 份 `results/bench_raw/*.json` 的 FLOPs 列均为未修正值，§2.3 / `summary.md` 的主表要改成修正值口径 → **旧 JSON 全部重跑**（MUTANT 侧 1.44~1.82× 的偏差不进主表） |

脚本：`script/addition/run_bench.sh`（调 harness）、`collect_stats.py`（扫 JSON → `summary.csv` + **`summary.md`**，可直接 UNION MUTANT 输出）。
⚠️ `.gitignore` 含 `*.csv`/`*.out` → 汇总结果必须同时产 `.md`。`conda activate loranomaly` 须改 `anomaly`。

---

## 6. 验收标准

1. `bench/benchmark_infer.py` 在两库下 SHA256 相同；两边 JSON 能被同一脚本 UNION。
2. 每份 JSON 含 §3-P4 全部字段（尤其 `harness_sha256` / `torch_num_threads` / `breakdown` / `counter_ops_missing`）。**缺字段视为不合格。**
3. 回归：`--trainmode profile` 的 F1 与 `--trainmode test` 一致（P1/P2 的 A/B 差值单独记录）。
4. P2 向量化前后 F1 差 <1e-6。
5. FLOPs 自洽：batch=1 vs batch=2 的 per-sample FLOPs 相对差 <0.1%（基线已满足）。
6. 中间块 FLOPs 占比回归：SMAP 复现 95.8%（±1%）。
7. 延时 repeats=5，`std/mean < 5%`；跨方法对比时 batch/W 逐行标注。
8. `flops_corrected` 与附录 B 公式**逐库可复算**，差异为 0（不是"有解释"而是"能对上"）。每份 JSON 的 `correction_factor` 落在实测区间内：SECOND 1.009~1.012×、MUTANT 1.44~1.82×；**超出区间说明公式或算子清单写错了**。
9. `counter_ops_missing` 必须经 instrument 复核（hook 出目标算子的真实输入形状），不得凭算子名猜测 —— 本次 `spmm` 的误报就是这么来的。
10. 跨库主表**同时含 per-window 与 per-timestep 两列**（§3-P6），且 params 列注明折算 batch。

---

## 7. 风险 / 待确认

| # | 项 |
|---|---|
| **R1** | **F1 态势未定**：日志里 SECOND 与 MUTANT 大致打平（SMAP 0.9751 vs 0.9794、MSL 0.9552 vs 0.9368、WADI 0.9831 vs 0.9914，MUTANT 是单 seed、SECOND 是不同超参的历史 run，均不可当结论）。**若受控对比后 F1 无优势，成本表就不该当卖点** —— 建议先跑受控 F1 对比再定成本表口径 |
| **R2** | 参数量落差 1541× 的叙事口径：成本表做"绝对时间"还是"F1-vs-成本散点" |
| **R3** | `OMP/MKL` 统一取值（建议 1，MUTANT 现为 32） |
| **R4** | 对照范围：只做 MUTANT，还是要适配其余 5 个库（GDN/DTAAD 可能各需 0.5~1d） |
| **R5** | MUTANT 侧需按 §3 补 4 项（延时分解 / 线程数 / harness SHA / 双轨），否则两张表合不到一起 |
| **R6** | 本机 GPU 被占用 → 固定 `CUDA_VISIBLE_DEVICES` ∈ {0,2,5} + 空闲断言 |
| **R7** | **修正后的算力比会从 42,196× 掉到 23,361×（SMAP），且方向对 SECOND 有利。**若先前的内部/汇报材料已用过 42,196× 或 "MUTANT 是 sparse GCN"，需要一并更正 —— 这是会被审稿人复算出来的数 |
| **R8** | **其余 5 个对照库（AMSL/DTAAD/GDN/MTAD-GAT/USAD）的漏算子清单尚未测**。它们同样可能含 LSTM/GRU/GNN 稀疏算子，**每一库都必须单独做 §4.1 的 instrument 复核**，不能套用 MUTANT 的结论。建议先各跑一次 `counter_ops_seen` 快照，再决定适配工作量 |

---

## 附录 A：复现本文实测数字

```bash
# 环境：conda activate anomaly (torch 2.14.0+cu126)，无需装 thop

# A. FOUR 数据集 FLOPs / params / 中间块占比
cd /home/sdb/yrlcd/anamolybench/SECOND
python - <<'EOF'
import torch
from torch.utils.flop_counter import FlopCounterMode
from model.Model import patchattmamba
from model.mamba import ModelArgs
for d,(C,W) in {'SMAP':(25,70),'MSL':(55,70),'SWaT':(51,35),'WADI':(127,70)}.items():
    m = patchattmamba(C, C*5, 256, 24, W, 5, args=ModelArgs()).eval()
    f = FlopCounterMode(display=False)
    with torch.no_grad(), f: m(torch.randn(1,W,C))
    tot = f.get_total_flops()
    f2 = FlopCounterMode(display=False)
    with torch.no_grad(), f2:
        for l in m.attmamba: l(torch.randn(1,C,512))
    print(f"{d:5s} C={C:3d} W={W:2d} {tot/1e9:7.3f}G/win {tot/W/1e9:7.4f}G/ts "
          f"params={sum(p.numel() for p in m.parameters())/1e6:6.2f}M block={100*f2.get_total_flops()/tot:.1f}%")
EOF

# B. MUTANT 跨库对照（独占卡；cwd 放 /tmp 以免污染 MUTANT 仓的 tmp_*.txt）
mkdir -p /tmp/mb && cd /tmp/mb && CUDA_VISIBLE_DEVICES=5 python - <<'EOF'
import sys, time, statistics, torch
sys.path.insert(0,'/home/sdb/yrlcd/anamolybench/MUTANT')
import Model as M
from Model import MUTANT
C,W,OUT = 25,20,5
sd = torch.load('/home/sdb/yrlcd/anamolybench/MUTANT/checkpoints/SMAP_model.pt', map_location='cpu')
def bench(m, x, n=50, warm=10):
    fn = lambda: m.reconstructed_probability(x)
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(3):
        t0=time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)/n*1000)
    return statistics.mean(ts)
for B in [1,8,32,64,120]:
    m = MUTANT(C, C*OUT, 100, 100, B, W, OUT)
    sd1 = dict(sd); sd1['att_encoder.h_n']=sd['att_encoder.h_n'][:,:B,:]; sd1['att_encoder.s_n']=sd['att_encoder.s_n'][:,:B,:]
    m.load_state_dict(sd1); m = m.cuda().eval()
    x = torch.randn(B,W,C)
    tt = bench(m,x)
    orig = M.returnA; A = torch.from_numpy(orig(x[0].cpu())).float(); M.returnA = lambda i: A
    tn = bench(m,x); M.returnA = orig
    print(f"batch={B:3d} {tt:8.3f} ms/batch  {tt/B:8.4f} ms/window  去returnA {tn/B:8.4f}")
EOF
```

---

## 附录 B：漏算子手工公式（两库统一口径）

> 本附录即 `collect_cost_mutant.py` 的 `correction_formula_ref` 所指内容。
> **统一约定**：MACs 记 1 次乘加；FLOPs = 2×MACs；逐元素乘/加记 1 FLOPs；`exp/silu/softplus` 等超越函数按 1 FLOPs 保守计（若按 4~8 计，SECOND 的修正量会翻几倍，但仍在 5% 以内，不改变结论）。**两库必须用同一套约定**，否则修正本身就成了新的不可比源。

### B.1 SECOND：`einsum` 广播乘 + `pscan` 逐元素

计数器已覆盖 `convolution / addmm / mm / bmm`（含 mutual-attn 的 attention），**未覆盖** `selective_scan` 与 `pscan`。单层 `attmambaBlock`，记 `b`=batch、`l`=**通道数 input_dim**（不是窗口长度！）、`d_in = d_inner = 1024`、`n = d_state = 16`：

| 操作 | 位置 | 漏算 FLOPs |
|---|---|---|
| ΔA 外积 `einsum('b l d_in, d_in n -> b l d_in n')` | `mamba.py:489` | `b·l·d_in·n` |
| `torch.exp(deltaA)` | 同上 | `b·l·d_in·n` |
| ΔB·u `einsum(..., 'b l d_in, b l n, b l d_in -> b l d_in n')` | `mamba.py:490` | `2·b·l·d_in·n` |
| `pscan` 上/下行扫（每轮 2 个 in-place op × ⌈log₂ npo2(l)⌉ 轮） | `pscan.py:54-92` | `2·⌈log₂ npo2(l)⌉·b·l·d_in·n` |
| `silu(x)` / `silu(res)` / `y*res` / `y + D*u` 等逐元素 | `mamba.py:407-411, 510` | `≈10·b·l·d_in` |

```
flops_corrected(SECOND) = flops_counter + Σ_{8 层} [ (4 + 2·⌈log₂ npo2(l)⌉)·b·l·d_in·n + 10·b·l·d_in ]
```

实测（b=1，括号内为**漏算占修正后总量**的比例）：SMAP 0.0479 G（0.93%）、MSL 0.1198 G（1.05%）、SWaT 0.1111 G（1.06%）、WADI 0.3100 G（1.15%）。

### B.2 MUTANT：`nn.LSTM`

计数器**完全看不到** `nn.LSTM`（ATen 融合算子）。记 `H`=hidden_size、`H_in`=input_size(=C)、`seq`/`B` 为 hook 到的**实际**输入形状 `(seq, batch, H_in)`：

```
每步每样本:  W_ih @ x_t   = 4H · H_in   MACs
             W_hh @ h_{t-1} = 4H · H    MACs
flops_lstm = seq · B · 2 · 4H · (H_in + H)
```

`seq` 必须**实测**（hook `encoder_lstm` 的输入形状），不能按 `window_length` 猜 —— 本库实际 `seq=1`（§2.4）。

```
flops_corrected(MUTANT) = flops_counter + flops_lstm
```

实测（batch=1，括号内为**漏算占修正后总量**的比例）：SMAP 100,000（45.1%）、MSL 124,000（30.4%）、SWaT 120,800（31.6%）、WADI 237,120（31.0%）。

### B.3 `spmm` 为何不在修正表内

`torch.spmm` 不在 `flop_registry` 里，但 `Graph.py:64` 的 `nfadj.toarray()` 把邻接矩阵**稠密化**了，`torch.spmm(dense.double(), dense.double())` 实际派发到 `aten.mm`，**已被计入**。直接验证：

```python
from torch.utils.flop_counter import FlopCounterMode
fc = FlopCounterMode(display=False)
with fc: torch.spmm(torch.randn(25,25,dtype=torch.float64), torch.randn(25,5,dtype=torch.float64))
print(fc.get_total_flops())   # 6250 = 2·25·25·5 ✓ 已计入
```

**判据不是"算子名在不在 registry 里"，而是"实测这个算子有没有进计数"。** 附录 B 的每一条都要求同样强度的验证。
