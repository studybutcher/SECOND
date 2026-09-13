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

### 2.3 算力差距（躲不掉，必须如实报）

| SMAP，batch=1 | SECOND | MUTANT | 比值 |
|---|---|---|---|
| params | 193.52 M | 0.126 M | **1541×** |
| FLOPs/window | 5.127 G | 0.000121 G | **42,372×** |

**算力差 4 万倍，延时只差 6 倍** —— 说明两边都不是算力瓶颈。只报一个延时列，读者无法解释这张表；必须同时给 params/FLOPs（算法成本）与分解后的延时（部署成本）。

---

## 3. 协议（5 条）

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

JSON 必备字段：`model` / `config`(各方法超参) / `cost`(n_params, macs_per_window, flops_per_window, `counter_ops_missing`, `macs_manual_formula`) / `latency`(batch, mean/p50/p95, `breakdown`, `variant`) / `memory` / `protocol`(**`harness_sha256`**, input_shape, precision, `torch_num_threads`, warmup/iters/repeats, `cudnn_benchmark`) / `env`(gpu, 独占标志, cuda, torch) / `metrics`(F1/P/R) / `ckpt.sha256`。

### P5 共享环境变量（易被忽略的偏差源）

| 变量 | 为什么 |
|---|---|
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `torch.set_num_threads` | MUTANT 那 68.5% 在 CPU 上，**线程数直接决定耗时**（其 `run_SMAP.sh` 现设 32）。建议两库统一为 **1**（最可复现）并记录实际生效值 |
| **cwd 所在文件系统** | MUTANT 的 `returnA` 每样本读写磁盘，SSD/HDD/tmpfs 差异直接进延时 |
| 精度 | MUTANT 的 GCN 是 fp64，SECOND 全 fp32，分组件记录 |
| `CUDA_VISIBLE_DEVICES` + 独占断言 | 卡 0/2/5 空闲（1/4 卡 100%，3/7 卡显存近满） |

---

## 4. 三个指标怎么测

### 4.1 计算成本

- 工具：**PyTorch 自带 `torch.utils.flop_counter.FlopCounterMode`**（不引入 thop，零依赖，覆盖面更好）
- 口径：**batch=1 单窗口前向**，`eval`，无 backward
- 实测 batch 无关性：bs=1/2/4/8 → 5.1267/5.1268/5.1268/5.1270 G per-sample，差 <0.01%
- **必须报覆盖声明**（单一工具一定漏算子，两边都证实了）：

| 库 | 计数器看到 | **漏掉** |
|---|---|---|
| SECOND | convolution / addmm / mm / bmm | `einsum`（广播乘，`mamba.py:489-490`）、`pscan`（in-place）。实测占 1~3%，**方向是低估 Mamba** |
| MUTANT | mm / addmm | **`spmm`（GCN 稀疏乘，且 fp64）**、**LSTM**（`Model.py:88`） |

→ 每行须有 `counter_ops_missing` + 手工公式兜底值交叉验证，**不一致必须查因后才能进论文**。

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

## 5. SECOND 侧改动

| # | 项 | 说明 |
|---|---|---|
| A1 | `bench/benchmark_infer.py` + `bench/adapter_second.py` | 通用 harness |
| A2 | `main.py` 加 `--trainmode profile` 分支 | 加载 ckpt → eval → FLOPs → 计时推理 → 显存 → 写 JSON。**`get_data`/`get_loader`/`is_anomaly`/`bf_search` 全部复用**。⚠️ `main.py:149/178` 是子串匹配 `'train' in ...`/`'test' in ...`，mode 名不能含这两串，`profile` 满足 |
| A3 | **P1**：推理前加 `model.eval()` | 现在没有 → `Dropout(0.1)` 推理时是激活的（`attentionblock.py:46,75,105`）。`test` 分支是否一并改，先做 A/B 量化 F1 差值再定（MUTANT 无 dropout，对它是 no-op，不影响对照） |
| A4 | **P2**：向量化逐样本循环 | `Model.py:177-180` 改为 `(x_a-mu).abs().sum(dim=list(range(1,x_a.dim())))`，数学等价。需 A/B 确认 F1 差 <1e-6。**这也是 `core_only` 变体的定义** |
| A5 | 补 seed 控制 | SECOND **当前完全没有**（MUTANT 有），对齐 `MUTANT/main.py:43-47`，~10 行 |
| A6 | 结果落盘 JSON | 现在只 `print`，加 `results/{dataset}/metrics_seed{S}.json` |

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
8. `macs_manual_formula` 与计数器值的差异有书面解释。

---

## 7. 风险 / 待确认

| # | 项 |
|---|---|
| **R1** | **F1 态势未定**：日志里 SECOND 与 MUTANT 大致打平（SMAP 0.9751 vs 0.9794、MSL 0.9552 vs 0.9368、WADI 0.9831 vs 0.9914，MUTANT 是单 seed、SECOND 是不同超参的历史 run，均不可当结论）。**若受控对比后 F1 无优势，成本表就不该当卖点** —— 建议先跑受控 F1 对比再定成本表口径 |
| **R2** | 参数量落差 1541× 的叙事口径：成本表做"绝对时间"还是"F1-vs-成本散点" |
| **R3** | `OMP/MKL` 统一取值（建议 1，MUTANT 现为 32） |
| **R4** | 对照范围：只做 MUTANT，还是要适配其余 5 个库（GDN/DTAAD 可能各需 0.5~1d） |
| **R5** | MUTANT 侧需按 §3 补 4 项（延时分解 / 线程数 / harness SHA / 双轨），否则两张表合不到一起 |
| **R6** | 本机 GPU 被占用 | 固定 `CUDA_VISIBLE_DEVICES` ∈ {0,2,5} + 空闲断言 |

---

## 附录：复现本文实测数字

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
