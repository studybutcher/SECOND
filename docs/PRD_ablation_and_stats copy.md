# PRD：推理成本统计 + 中间块消融实验

> 版本 v1 ｜ 日期 2026-09-11 ｜ 状态：待评审
> 需求来源：`main.py` / `model/Model.py` / `model/mamba.py` 现有实现的补充实验
> 范围：**只补"推理统计"与"中间块替换消融"两件事**，不重训基线、不改动现有指标口径

---

## 1. 背景与目标

现有仓库已有一条完整的 `train → test → bf_search(F1)` 流程，但：

1. **没有任何可用的成本数据**。`script/addition/timecost.sh` 只是把 `--name` 改成 `timecost` 又跑了一遍 train-test；`script/addition/memory.sh` 是空文件；`Model.py` 里手工加的 `time0=time.time()` 计时曾导致 `NameError: name 'time' is not defined`（见 `nohp-out/timecost.out` 的 traceback），现在被注释掉了。
2. **没有对核心模块的消融**。`model/attnssmblock.py` 是 0 字节空文件，`model/Model copy.py`、`model/Model back4sub.py`、`model/Model-nopatch(...).py` 都只是历史备份，不是受控消融。

本 PRD 的目标：

| 目标 | 可验证的产出 |
|---|---|
| G1 推理成本可量化 | 每个 (变体, 数据集) 一行：FLOPs/MACs、ms/window、峰值显存 MB |
| G2 中间块消融 | 用 Transformer / LSTM 替换中间块，其余结构完全不变，同一套脚本跑出 F1 对比 |
| G3 复用现有代码 | 改动集中在 1 个新工具文件 + `main.py` 增加 1 个分支 + 每个消融变体 ≤5 行 diff |

---

## 2. 现状盘点（只读代码得出的结论）

### 2.1 调用链

```
script/*.sh  →  main.py:main()
                 ├─ utils.get_data / get_loader        (滑窗 + padding)
                 ├─ patchattmamba(...)                 model/Model.py:57
                 │    ├─ pipline()                     model/Model.py:195
                 │    │    ├─ data_patch / DataEmbedding ×2 尺度 [5,7]
                 │    │    ├─ projection1 / projection2  → x_combined
                 │    │    ├─ self.attmamba  ← ★ 本次消融目标（8 层）
                 │    │    ├─ project4 → attention_sim → attention_ALL → projection3
                 │    │    └─ predict() → encoder/decoder (VAE)
                 │    └─ is_anomaly() / reconstructed_probability()   ← ★ 本次统计测点
                 └─ eval_method.bf_search  → F1 / precision / recall
```

### 2.2 关键张量形状（SMAP 为例，`latent_size=256`, `window_length=70`, `input_size=25`）

| 位置 | 形状 |
|---|---|
| 输入 | `[B, 70, 25]` |
| patch 双分支输出拼回 | `x_combined = [B, 25, 512]` （`dim = latent_size*2 = 512`, `Model.py:227`） |
| **中间块 8 层** | **`[B, 25, 512] → [B, 25, 512]`（形状不变）** |
| 展平后 | `[B, 25*512]` → `project4` → `[B, 512]` |
| VAE 输入 | `x_a = [B, 25*5 = 125]` |

**注意**：`self.attmamba` 的"序列维"是**通道维**（`l = input_size = 25/55/51/127`），不是时间维。代码里 `Model.py:230-234` 直接把它当序列送进 `ResidualBlock`，`mamba.py:403` 的注释「不用调换已经能获得很好的效果了」说明 conv1d/SSM 就作用在通道轴上（`mamba.py:399-402` 那个沿通道做 self-attn 的分支已被注释掉）。**这一点决定了后面第 6 节的风险判断。**

### 2.3 中间块的真实占比（本次实测，SMAP / bs=1 / window=70）

用 `torch.utils.flop_counter.FlopCounterMode` 实测：

```
全模型 forward FLOPs          : 5.127 GFLOPs
其中 self.attmamba 8 层       : 4.912 GFLOPs   → 95.8%
单层 attmambaBlock            : 0.614 GFLOPs
全模型参数量                  : 193.5 M
其中 self.attmamba 8 层       : 148.0 M        → 76.5%
```

**结论：中间块是全模型 95.8% 的算力、76.5% 的参数所在。** 以它为消融对象是"打到了要害"，成本指标一定会有可观测的差异（不是 5% 量级的噪声）。

### 2.4 现有统计代码的问题

| 位置 | 问题 |
|---|---|
| `nohp-out/timecost.out` | 历史崩溃：`Model.py:199 time0=time.time()` 未 `import time` |
| `Model.py:14` | 现在有 `import time`，但 200/229/233/245 行的计时全被注释 |
| `Model.py:177-180` | `reconstructed_probability` 逐样本 Python 循环 + `torch.tensor(list_of_gpu_scalar)` |
| `main.py:183` | 推理前没有 `model.eval()` |
| `script/addition/memory.sh` | 空文件 |
| 所有 `script/*.sh` | `conda activate loranomaly`，但**本机不存在该环境**（可用环境为 `anomaly`，torch 2.14.0+cu126） |

---

## 3. 范围

**做**
- 推理侧 FLOPs/MACs、延时、峰值显存三个指标的定义、采集、落盘、汇总
- 中间块 `attmambaBlock` → Transformer / LSTM 两个消融变体（复制文件法）
- 配套 shell 脚本与汇总脚本

**不做**
- 不改训练逻辑、超参、损失函数
- 不改现有 `test` 分支的 F1 计算口径（`bf_search` 保持原样）
- 不做训练侧成本统计（用户已明确"数据统计只写在推理部分"）
- 不动 `model/Model-nopatch(...)`、`Model copy.py` 等历史备份

---

## 4. Part A：推理成本统计

### 4.1 三个指标的口径（先把话说清楚）

| 指标 | 口径 | 单位 |
|---|---|---|
| FLOPs / MACs | 单窗口（batch=1）、`window_length` 全长的 **forward once**、`eval` 模式、不含 backward | GFLOPs / GMACs，FLOPs = 2×MACs |
| 推理延时 | 见 4.3，分「端到端 ms/window」与「中间块 ms/iter」两个数 | ms |
| 峰值显存 | 进程内 `torch.cuda.max_memory_allocated()`，从「模型加载完成」到「全测试集推理结束」 | MB |

**必须与变体无关地固定**：`batch_size`、`window_length`、`latent_size`、`input_dim`、patch 尺度 `[5,7]`、`n_layer`。
（注意 `main.py:40-44` 会按数据集强制覆盖：`SWaT → window=35`、`WADI → batch=24`。统计表里必须把这两个数一起打出来，否则跨数据集不可比。）

### 4.2 FLOPs/MACs 怎么做（已实测选型）

**方案：用 PyTorch 自带的 `torch.utils.flop_counter.FlopCounterMode`，不引入 thop/fvcore。**
理由（实测）：thop 在本机未安装（可 `pip` 装，网络通），而 torch 自带的不仅能覆盖，还免依赖。

实测覆盖面（本模型的算子在 bs=1 下的全局统计）：

```
aten.convolution : 25,174,016
aten.addmm       : 8,796,223,488
aten.mm          : 1,336,934,400
aten.bmm         :    95,174,656
```

**未覆盖的部分（必须在论文表格里加脚注）**：

- `selective_scan` 里的两个 `einsum`（`mamba.py:489-490`）是**广播逐元素乘**，不是矩阵乘 → 走 `mul`，不计入
- `pscan`（`model/pscan.py`）全程是 in-place 的 `add_`/`mul_` → 完全不计入
- `exp/silu/softplus` 等激活 → 不计入

解析式补算（单层 `attmambaBlock`，记 `b`=batch, `l`=通道数, `d_in=d_inner=1024`, `n=d_state=16`）：

```
丢失 FLOPs ≈ (ΔA 外积 1 + exp 1 + ΔB·u 2 + y·C 1 + pscan 2·⌈log2 l⌉) · b·l·d_in·n
```

SMAP（l=25, npo2(25)=32, ⌈log2⌉=5）单层约 `15·b·l·d_in·n`，实测占比 **1%~3%**。
→ **口径要求**：FLOPs 表里同时给「计数器读数」+「解析补算值」两列，或至少在脚注写明「Mamba 变体的 FLOPs 被低估约 1~3%，Transformer 变体的 attention 走 bmm 已计入」。**这个偏差方向对 Mamba 有利，必须披露，否则审稿人会抓。**

**batch 维度的坑（已实测澄清）**：`tabular_encoder/decoder` 的 `attentionstacking` 和 `attention_sim` 收到的是 `[B, 512]` 的二维张量，`nn.MultiheadAttention` 会把 **B 当作序列维**。理论上这让 per-window FLOPs 随 batch 二次增长。实测 bs ∈ {1,2,4,8}：

```
bs=1  5.1267 G    per-sample 5.1267 G
bs=2 10.2535 G    per-sample 5.1268 G
bs=4 20.5073 G    per-sample 5.1268 G
bs=8 41.0158 G    per-sample 5.1270 G
```

每样本差异 <0.01%（因为 E=512、B 小，二次项可忽略）。
→ **口径：FLOPs 一律按 batch=1 报**，这是通行做法，且本模型恰好也等价于 per-sample 成本。

### 4.3 推理延时怎么做（回答"batch 要不要相同"）

**结论：必须相同；但不要只报一个 ms/window，要报三个数。**

为什么不能只报 ms/window：

1. `ms/window = 总时间 / 窗口数`，这个数**只有在 GPU 被喂饱时才与 batch 无关**。batch=60 在 5880 Ada（48GB）上跑这个 194M 的小模型，大概率**喂不饱**，ms/window 会随 batch 增大而下降（launch overhead 被摊薄）。用不同 batch 测出来的 ms/window 放同一张表，等于比的是 batch 不是模型。
2. 上面 4.2 已经证明 encoder/decoder 的 attention 走的是 batch 维，严格说 per-window 成本不是与 batch 严格无关的量。

**采集方案（两级）**：

**A) 端到端**：在 `is_anomaly()` 外层包计时，跑完整测试集。

```
warmup: 前 5 个 batch 不计时（cuDNN/MemoryPool 预热）
计时  : 每个 batch 前后 torch.cuda.synchronize()
输出  : e2e_ms_per_batch, e2e_ms_per_window = 总时间 / 有效窗口数
        （有效窗口数 = num_t*batch + con_t，要扣掉 get_loader 补的零 padding）
```

**B) 中间块微基准（重点）**：只跑 `self.attmamba` 那 8 层，输入用真实数据形状 `[B, input_dim, 512]`。

```
warmup 20 iter → 计时 100 iter，重复 5 次取 mean ± std
```

这是本 PRD 最推荐的一个数，理由：
- 消融只改这 8 层，其余 4.2% 的算力完全相同，端到端数字会被"不变量"稀释；
- **更严重的是 `Model.py:177-180` 的逐样本 Python 循环**：`for i in range(x.shape[0])` 里做 `torch.sum` 再 `torch.tensor(p)`，每个样本一次 GPU→CPU 同步。测试集 1~5 万窗口就是 1~5 万次同步，估计 1s 量级——它会直接淹没变体之间真实的差异。微基准完全绕开这个污染项。

**（可选，推荐做）batch 扫描**：`batch ∈ {1, 8, 16, 60}` 各测一遍 ms/window，画一条曲线。既能证明"结论不依赖某个特定 batch"，也是论文里很好的效率图。

**环境纪律（强制）**：
- 测量必须独占 GPU。当前 `nvidia-smi`：卡 1/4 利用率 100%、卡 3/7 显存 46GB 已占满，**卡 0/2/5 空闲**。
- 脚本里固定 `CUDA_VISIBLE_DEVICES` ∈ {0,2,5}，并在跑之前断言该卡 `memory.used < 1000 MiB`。
- 记录 `torch.__version__`、CUDA 版本、GPU 型号、batch_size 到 JSON 里。

### 4.4 峰值显存怎么做

```python
model.load_state_dict(torch.load(save_path))
model.cuda()
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
mem_after_load = torch.cuda.memory_allocated() / 2**20        # 参数 + context
score = model.is_anomaly(test_loader, num_t, con_t)           # 复用现有函数，不改
peak  = torch.cuda.max_memory_allocated() / 2**20             # 推理峰值
reserved = torch.cuda.max_memory_reserved() / 2**20
```

报三个数：`after_load`（≈参数+context 基线）、`inference_peak`、`activation_delta = peak - after_load`。
`max_memory_allocated` 是**进程内**计数，不受同卡其他进程影响（这点比延时可靠）；但仍要固定 batch_size，因为激活峰值随 batch 线性增长。

顺带把静态参数指标也报了：`sum(p.numel())`、`param_mem_MB = numel*4/2**20`（FP32）。

### 4.5 落盘与汇总

**每跑一次产出一个 JSON**：`nohp-out/stats/{backbone}_{dataset}.json`

```json
{
  "backbone": "mamba", "dataset": "SMAP", "name": "ablation/mamba",
  "batch_size": 60, "window_length": 70, "latent_size": 256, "input_dim": 25, "n_layer": 8,
  "n_params": 193521527, "param_mem_MB": 738.3,
  "num_windows": 34338,
  "flops_per_window_G": 5.1267, "macs_per_window_G": 2.5634,
  "flop_counter_ops": ["convolution","addmm","mm","bmm"],
  "flops_uncounted_note": "einsum/pscan/elementwise 未计入，解析补算约 +0.10 G (2.0%)",
  "e2e_ms_total": 0.0, "e2e_ms_per_batch": 0.0, "e2e_ms_per_window": 0.0,
  "block_ms_per_iter": {"mean": 0.0, "std": 0.0, "n": 5, "batch": 60},
  "mem_MB": {"after_load": 0.0, "inference_peak": 0.0, "activation_delta": 0.0},
  "f1": 0.0, "precision": 0.0, "recall": 0.0,
  "env": {"gpu": "RTX 5880 Ada", "torch": "2.14.0+cu126", "cuda": "12.6", "eval_mode": true}
}
```

**汇总脚本** `script/addition/collect_stats.py`：读 `nohp-out/stats/*.json` → 输出
`nohp-out/stats/summary.csv` 和 `nohp-out/stats/summary.md`（markdown 表格，可直接贴论文）。

> ⚠️ `.gitignore` 里有 `*.csv`、`*.out`、`*.npy`、`*.pt`。汇总结果若要进版本库，**同时产一份 `.md`**（或 `git add -f`）。

`f1/precision/recall` 直接复用 `eval_method.bf_search` 的返回值，让"精度 + 成本"出现在同一张表里。

### 4.6 需要顺带修的三个坑（都影响统计可信度）

| # | 问题 | 建议 | 影响面 |
|---|---|---|---|
| P1 | 推理时没有 `model.eval()`，`nn.Dropout(0.1)`（`attentionblock.py:46,75,105`）**在推理时是激活的** | `profile` 分支里显式 `model.eval()`；把是否 eval 记进 JSON | 只改统计分支，**不动 `test` 分支**（否则历史 F1 数字会变，需用户决定） |
| P2 | `reconstructed_probability` 逐样本同步（`Model.py:177-180`） | 数学等价改写：`p = (x_a - mu).abs().sum(dim=list(range(1, x_a.dim())))`，一次搞定 | 去掉 1~5 万次同步，e2e 延时才有意义。**会改变浮点求和顺序**，需确认 F1 是否完全一致（预期一致） |
| P3 | 中间块 FLOPs 被低估 1~3% | 见 4.2，脚注 + 解析补算列 | 论文表格 |

P1/P2 都涉及"是否改变现有结果"，列为待确认项（见第 7 节）。

---

## 5. Part B：中间块消融（Mamba → Transformer / LSTM）

> 术语对齐：本 PRD 把"MSSA 中间部分"理解为 `Model.py:120` 的
> `self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])`
> 即 8 层 `ResidualBlock(mixer=attmambaBlock)`，`attmambaBlock` = in_proj / conv1d / SSM(selective scan) / mutual-attn（`mamba.py:340-417`）。若你指的只是其中某个子模块（例如只换 SSM 或只换 mutual-attn），方案同样适用，改动点变成 `mamba.py` 内部一行。

### 5.1 公平性约定（先定死，否则消融没意义）

| 项 | 约定 |
|---|---|
| 替换范围 | **只换中间 8 层的 mixer**，patch 双分支、`project4`、`attention_sim`、`attention_ALL`、VAE 一字不改 |
| 维度 | `d_model = 512`（= `latent_size*2`，`Model.py:119` 就地赋值）、`d_inner = 2*d_model = 1024`、`n_layer = 8` 三者全变体一致 |
| 残差骨架 | 统一用 `ResidualBlock` 的 `output = mixer(norm(x)) + x`（`mamba.py:177`），norm 统一 `RMSNorm(d_model)` |
| 训练设置 | 优化器、lr、epoch、`N=256` 梯度累积步、数据切分、`val=0.35` 全部沿用 `main.py` 现值 |
| 参数量 | **不强行对齐**（对齐会引入额外自由度），改为在结果表里同时报参数量，让读数自己说话 |
| 输出 | 每个变体都必须产出 4.2 节的 JSON，含 `n_params` 与 F1 |

### 5.2 文件组织（复制法，采纳用户偏好）

```
model/backbone.py            # 新增：只放两个 mixer 实现 + 通用 ResidualBlock
model/Model.py               # 不动（mamba 基线）
model/Model_trans.py         # 复制 Model.py，改 2 行
model/Model_lstm.py          # 复制 Model.py，改 2 行
```

`Model_trans.py` 相对 `Model.py` 的**全部 diff**（预期 ≤5 行）：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import TransformerResidualBlock
...
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([TransformerResidualBlock(args) for _ in range(args.n_layer)])
```

**类名保持 `patchattmamba`** —— 这样 `main.py` 里除了 import 那一行，其余代码（`model = patchattmamba(...)`、`is_anomaly`、`bf_search`）全部零改动。

### 5.3 `model/backbone.py` 接口契约（关键）

`attmambaBlock` 的调用签名是 `__init__(args: ModelArgs)` + `forward(x: [b,l,d_model]) -> [b,l,d_model]`（形状不变）。新 mixer 必须严格照这个契约写，才能直接塞进 `ResidualBlock` 骨架：

```python
import torch
from torch import nn
from einops import repeat
from .mamba import ModelArgs, RMSNorm

class ResidualBlockGeneric(nn.Module):
    """与 mamba.py:146 ResidualBlock 完全同构，只把 mixer 参数化出来。"""
    def __init__(self, args: ModelArgs, mixer_cls):
        super().__init__()
        self.mixer = mixer_cls(args)
        self.norm = RMSNorm(args.d_model)
    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class TransformerMixer(nn.Module):
    """(b,l,d) -> (b,l,d)，与 attmambaBlock 同接口。"""
    def __init__(self, args: ModelArgs, nhead=4, ffn_dim=1024, dropout=0.1):
        super().__init__()
        d = args.d_model
        assert d % nhead == 0
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, d))
    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]   # need_weights=False: 省掉 l×l 权重回传
        return x + self.ffn(self.ln2(x))

class LSTMMixer(nn.Module):
    """(b,l,d) -> (b,l,d)。双向 LSTM 输出 2d，Linear 投回 d。"""
    def __init__(self, args: ModelArgs, dropout=0.1):
        super().__init__()
        d = args.d_model
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d, d)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        y, _ = self.lstm(x)
        return x + self.proj(y)

def TransformerResidualBlock(args): return ResidualBlockGeneric(args, TransformerMixer)
def LSTMMixerBlock(args):           return ResidualBlockGeneric(args, LSTMMixer)
```

`ffn_dim` 默认 1024（对齐本仓库既有 `feeddim=1024` 惯例，`Model.py:105`），`nhead=4`（对齐 `attmambaBlock` 里 `h=4`，`mamba.py:345`）。这两个值作为可调项记录进 JSON。

### 5.4 `main.py` 改动（1 个参数 + 1 个分支）

```python
self.add_argument('--backbone', default='mamba',
                  choices=['mamba', 'transformer', 'lstm'])

BACKBONE_REGISTRY = {
    'mamba':       'model.Model',
    'transformer': 'model.Model_trans',
    'lstm':        'model.Model_lstm',
}
...  # 替换 main.py:137 的构建
import importlib
_MOD = importlib.import_module(BACKBONE_REGISTRY[config.backbone])
model = _MOD.patchattmamba(config.input_dim, w_size, config.latent_size,
                           config.batch_size, config.window_length, config.out_dim, args=argsa)
```

`checkpoint` 路径沿用 `checkpoint/{name}/{dataset}/best.pt`（`main.py:144-147`），消融用 `--name ablation/{backbone}`，天然隔离。

### 5.5 统计分支（复用现有流程）

在 `main.py` 增加：

```python
if 'profile' in config.trainmode:
    ...
```

⚠️ 坑：`main.py:149/178` 用的是**子串匹配** `'train' in config.trainmode`、`'test' in ...`。所以新 mode 字符串必须**既不包含 `train` 也不包含 `test`**——`profile` 满足。

用法：`python main.py --dataset SMAP --name ablation/mamba --trainmode profile --backbone mamba`
该分支内部依次：加载 checkpoint → `model.eval()` → FLOPs 计数 → 冷启动计时跑 `is_anomaly` → 显存读数 → 中间块微基准 → `bf_search` 取 F1 → 写 JSON。**`get_data` / `get_loader` / `is_anomaly` / `bf_search` 全部直接复用。**

### 5.6 脚本

| 脚本 | 作用 |
|---|---|
| `script/addition/ablation.sh` | 训练两个变体 × 4 数据集：`--trainmode train-test --name ablation/$BB --backbone $BB` |
| `script/addition/stats.sh` | 三个变体 × 4 数据集跑 `--trainmode profile`，产出 JSON（**取代现有空白的 `timecost.sh` / `memory.sh`**，两个旧文件删掉或改成调用本脚本） |
| `script/addition/collect_stats.py` | JSON → `summary.csv` + `summary.md` |

三个脚本都要改掉 `conda activate loranomaly`（本机无此环境）→ `conda activate anomaly`，并在开头断言所选 GPU 空闲。

---

## 6. 风险与预期

| # | 风险 | 说明 / 应对 |
|---|---|---|
| R1 | **中间块的"序列"其实是通道维（l=25~127），不是时间维** | SSM 对长序列的优势在这个设置下**本来就发挥不出来**。期望管理：Transformer 变体很可能打平甚至小胜 F1。**若打平，论文的叙事必须从"精度"转向"成本/显存/长通道可扩展性"**——这正是 Part A 存在的意义。需要提前决定叙事，否则消融做完不好写。 |
| R2 | Transformer 是置换等变的，Mamba/conv1d 沿通道轴是有序的 | 默认**不加位置编码**（纯算子替换，最严苛也最公平）。可加 `--pos_emb` 开关做一张补充表：若"Transformer+PE"明显更好，说明差异来自顺序归纳偏置而非 SSM 本身。**取向需用户确认。** |
| R3 | LSTM 走 cuDNN，batch 依赖的算法选择 + 首次调用预热 | 微基准必须 warmup ≥20 iter、重复 5 次报 mean±std。 |
| R4 | 端到端延时被 P2 的逐样本同步污染 | 先解决 P2，或至少在报告里给出"中间块占比"，否则 e2e 数字讲不出结论。 |
| R5 | FLOPs 计数器不覆盖 einsum/pscan，低估 Mamba | 4.2 的脚注 + 解析补算列。 |
| R6 | 参数量不可比（Mamba 8 层 148M vs Transformer/LSTM 更小或更大） | 表里并列 `n_params`，必要时补一张"参数量对齐"的补充实验。 |
| R7 | 本机 GPU 被占用（1/4/7 卡在跑） | 固定 `CUDA_VISIBLE_DEVICES` ∈ {0,2,5} + 空闲断言。 |
| R8 | 复制文件后三份 `Model*.py` 漂移（已有 `Model copy.py` 等前车之鉴） | 用 `diff` 作为验收项（见 7.2），并把"只允许改哪几行"写进文件头注释。 |

---

## 7. 交付物与验收标准

### 7.1 交付物

| 交付物 | 路径 |
|---|---|
| 统计工具 | `profiler.py`（或 `utils_stats.py`）|
| 消融 backbone | `model/backbone.py` |
| 消融变体 | `model/Model_trans.py`、`model/Model_lstm.py` |
| main.py 改动 | `--backbone` 参数 + `profile` 分支 |
| 脚本 | `script/addition/{ablation.sh, stats.sh, collect_stats.py}` |
| 结果 | `nohp-out/stats/{backbone}_{dataset}.json` × 12、`summary.csv`、`summary.md` |
| PRD 文档 | `docs/PRD_ablation_and_stats.md`（本文）|

### 7.2 验收标准

1. `diff "model/Model.py" "model/Model_trans.py"` 的差异行数 **≤ 5 行**，且全部落在 import 与 `self.attmamba` 两处。（`Model_lstm.py` 同理）
2. `--backbone mamba --trainmode profile` 报出的 F1 与现有 `--trainmode test` 分支一致（同一 checkpoint，容差见 P1/P2 决策）。
3. FLOPs 自洽性：`batch=1` 与 `batch=2` 的 per-sample FLOPs 相对差 **< 0.1%**（基线实测 5.1267 vs 5.1268 G，已满足）。
4. 中间块 FLOPs 占比与本文 2.3 节实测（95.8%）一致（回归保护）。
5. 延时：同一配置重复 5 次，`std/mean < 5%`；跨变体对比时 `batch_size` / `window_length` 完全相同。
6. 显存读数与 `nvidia-smi` 观测同量级（进程内 allocated ≤ nvidia-smi used）。
7. 12 份 JSON 全部产出，`summary.md` 能直接贴进论文。

### 7.3 任务拆分（粗估）

| # | 任务 | 文件 | 估时 |
|---|---|---|---|
| T1 | 统计工具（FLOPs/延时/显存/JSON） | `profiler.py` | 0.5d |
| T2 | main.py：`--backbone` + `profile` 分支 | `main.py` | 0.3d |
| T3 | 两个 mixer + 两个变体文件 | `model/backbone.py`, `Model_trans.py`, `Model_lstm.py` | 0.5d |
| T4 | 三个脚本 | `script/addition/*` | 0.3d |
| T5 | 跑基线统计（3 变体 × 4 数据集）并回归验证 | — | 0.5d |
| T6 | 消融训练（Transformer/LSTM × 4 数据集） | — | 1~2d（GPU） |
| T7 | 汇总表 + 效率曲线图 | `collect_stats.py`, `plotting.py` | 0.3d |

依赖：T2 依赖 T1；T5 依赖 T1~T4；T6 可与 T5 并行（不同 GPU）。

---

## 8. 待确认问题（阻塞项）

| # | 问题 | 影响 |
|---|---|---|
| Q1 | **"MSSA" 的确切所指**：是整个 8 层 `attmamba` 栈，还是只换其中的 SSM / mutual-attn 子模块？ | 决定 T3 改在哪一层 |
| Q2 | `profile` 分支是否用 `model.eval()`？**现有 `test` 分支是否也一并加**（会改变历史 F1 数字）？ | P1，影响所有对比数字 |
| Q3 | 是否允许把 `reconstructed_probability` 改成向量化（数学等价但浮点求和顺序变）？ | P2，端到端延时是否可用 |
| Q4 | Transformer 变体默认**不加**位置编码（严苛公平）是否接受？是否要额外跑一版 +PE 的补充实验？ | R2，可能改变消融结论的解读 |
| Q5 | 消融结果若与 Mamba 打平，论文叙事是否切到"效率/显存/成本"？（R1） | 决定 Part A 的地位是"配角附录"还是"主结果" |
| Q6 | 实验规模：4 数据集 × 2 变体 × 是否 3 次重复（现有 `script/重复测试/repeat.sh` 有 3 次重复的先例） | GPU 预算 |

---

## 附录 A：Mamba 中间块丢失 FLOPs 的解析式

单层 `attmambaBlock`，`b`=batch，`l`=通道数，`d_in=1024`，`n=d_state=16`，`L_pad=npo2(l)`：

| 操作 | 位置 | 丢失 FLOPs |
|---|---|---|
| ΔA 外积 `einsum('b l d_in, d_in n -> b l d_in n')` | `mamba.py:489` | `b·l·d_in·n` |
| `torch.exp(deltaA)` | 同上 | `b·l·d_in·n` |
| ΔB·u `einsum(...,'b l d_in, b l n, b l d_in -> b l d_in n')` | `mamba.py:490` | `2·b·l·d_in·n` |
| `(hs @ C)` | `mamba.py:509` | 已计入（bmm） |
| `y + D*u` | `mamba.py:510` | `b·l·d_in` |
| `pscan` 上/下行扫 | `pscan.py` | `≈ 2·log2(L_pad)·b·l·d_in·n` |

SMAP：`l=25 → L_pad=32 → log2=5`，单层 ≈ `15·b·l·d_in·n`。

## 附录 B：复现本文所有实测数字的命令

```bash
# 环境：conda activate anomaly (torch 2.14.0+cu126)，无需装 thop
cd /home/sdb/yrlcd/anamolybench/SECOND
python - <<'EOF'
import torch
from torch.utils.flop_counter import FlopCounterMode
from model.Model import patchattmamba
from model.mamba import ModelArgs
args = ModelArgs()
m = patchattmamba(25, 125, 256, 60, 70, 5, args=args).eval()   # SMAP
x = torch.randn(1, 70, 25)
f = FlopCounterMode(display=False)
with torch.no_grad(), f: m(x)
print("total:", f.get_total_flops()/1e9, "G")                   # 5.127
z = torch.randn(1, 25, 512)
f2 = FlopCounterMode(display=False)
with torch.no_grad(), f2:
    for layer in m.attmamba: layer(z)
print("mid stack:", f2.get_total_flops()/1e9, "G")              # 4.912 (95.8%)
EOF
```
