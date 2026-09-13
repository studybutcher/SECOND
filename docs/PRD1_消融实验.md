# PRD 1：中间块消融实验（训练侧）

> 2026-09-12 ｜ 仓库：SECOND ｜ 状态：待评审（仅剩 1 个待确认项）
> 关联文档：`PRD2_推理成本统计.md`（成本表另开一份）

---

## 1. 目标

把 `self.attmamba`（8 层，即"MSSA"整体）替换为 **Transformer** 和 **LSTM**，其余结构一字不改，验证 Mamba/SSM 主体的必要性。

**为什么是这块**（SMAP 实测）：

| | 全模型 | `self.attmamba` 8 层 | 占比 |
|---|---|---|---|
| FLOPs | 5.127 G | 4.912 G | **95.8%** |
| 参数 | 193.5 M | 148.0 M | **76.5%** |

它是模型的计算与参数重心，换掉它消融才有信号。

**消融对象**：`model/Model.py:120` 的 `nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])`。
其中 `ResidualBlock`（`mamba.py:146`）= `RMSNorm` + `attmambaBlock`（`mamba.py:340`，含 in_proj / conv1d / SSM selective-scan / mutual-attn）。

**已确认**：整个 8 层栈一起换；Transformer 变体**不加位置编码**（纯算子替换，最严苛公平）。

---

## 2. 方案：复制文件法，每个变体只改 2 行

```
model/backbone.py        # 新增：两个 mixer + 通用 ResidualBlock
model/Model.py           # 不动（mamba 基线）
model/Model_trans.py     # 复制 Model.py，改 2 行
model/Model_lstm.py      # 复制 Model.py，改 2 行
```

`Model_trans.py` 相对 `Model.py` 的**全部 diff**：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import TransformerResidualBlock
...
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([TransformerResidualBlock(args) for _ in range(args.n_layer)])
```

**类名保持 `patchattmamba`** → `main.py` 除 import 外零改动。

---

## 3. 接口契约（关键）

`attmambaBlock` 的签名是 `__init__(args: ModelArgs)` + `forward(x:[b,l,d]) -> [b,l,d]`（形状不变）。新 mixer 必须严格照抄，才能直接塞进残差骨架。

```python
import torch
from torch import nn
from .mamba import ModelArgs, RMSNorm

class ResidualBlockGeneric(nn.Module):
    """与 mamba.py:146 ResidualBlock 同构，只把 mixer 参数化。"""
    def __init__(self, args: ModelArgs, mixer_cls):
        super().__init__()
        self.mixer = mixer_cls(args)
        self.norm = RMSNorm(args.d_model)
    def forward(self, x):
        return self.mixer(self.norm(x)) + x

class TransformerMixer(nn.Module):
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
        x = x + self.attn(h, h, h, need_weights=False)[0]   # need_weights=False：省掉 l×l 权重回传
        return x + self.ffn(self.ln2(x))

class LSTMMixer(nn.Module):
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

超参对齐现有惯例：`ffn_dim=1024`（= `Model.py:105` 的 `feeddim`）、`nhead=4`（= `mamba.py:345` 的 `h=4`）。

---

## 4. main.py 改动

```python
self.add_argument('--backbone', default='mamba', choices=['mamba','transformer','lstm'])

BACKBONE_REGISTRY = {'mamba': 'model.Model',
                     'transformer': 'model.Model_trans',
                     'lstm': 'model.Model_lstm'}
# 替换 main.py:137
import importlib
_MOD = importlib.import_module(BACKBONE_REGISTRY[config.backbone])
model = _MOD.patchattmamba(config.input_dim, w_size, config.latent_size,
                           config.batch_size, config.window_length, config.out_dim, args=argsa)
```

ckpt 走现有 `checkpoint/{name}/{dataset}/best.pt`（`main.py:144-147`），消融用 `--name ablation/{backbone}` 天然隔离。

---

## 5. 公平性约定

| 项 | 约定 |
|---|---|
| 替换范围 | **只换中间 8 层的 mixer**；patch 双分支、`project4`、`attention_sim`、`attention_ALL`、VAE 一字不改 |
| 维度 | `d_model=512`（`Model.py:119`）、`d_inner=1024`、`n_layer=8` 三变体一致 |
| 残差骨架 | 统一 `mixer(norm(x)) + x`，norm 统一 `RMSNorm` |
| 位置编码 | 不加 |
| 训练设置 | 优化器 / lr / epoch / `N=256` / `val=0.35` 全沿用现有值 |
| 参数量 | **不强行对齐**（会引入额外自由度），在结果表里并列 `n_params` 让读数说话 |

---

## 6. 脚本

`script/addition/ablation.sh`：三变体 × 4 数据集

```bash
for BB in mamba transformer lstm; do
  for data in SMAP MSL SWaT WADI; do
    python main.py --dataset $data --name ablation/$BB --trainmode train-test --backbone $BB
  done
done
```

沿用现有约定（`nohup` + `nohp-out/`），但 `conda activate loranomaly` **须改为 `anomaly`**（本机无 `loranomaly` 环境）。

---

## 7. 验收标准

1. `diff model/Model.py model/Model_trans.py` **≤5 行**，且只落在 import 与 `self.attmamba` 两处（`Model_lstm.py` 同理）。
2. 三变体在同一数据集上的 `window_length` / `batch_size` / `latent_size` / epoch / lr 完全一致。
3. 每个变体产出 4 个数据集的 F1，并同时给出 `n_params`。
4. 结果表能回答："去掉 SSM 换成 attention / 换成 LSTM，F1 掉多少、参数量变多少"。

---

## 8. 风险

| # | 风险 | 应对 |
|---|---|---|
| **R1** | 中间块的"序列维"是**通道维**（`l = input_size = 25/55/51/127`），不是时间维（`Model.py:230-234` 直接把它当序列送进 `ResidualBlock`，`mamba.py:403` 注释确认 conv1d/SSM 作用在通道轴）。**SSM 的长序列优势在这个设置下本就发挥不出来** → Transformer 很可能打平甚至小胜 | 提前定叙事：若打平，卖点转向成本 / 显存 / 长通道可扩展性，或补一版 +PE 的补充表（若 +PE 明显更好，说明差异来自**顺序归纳偏置**而非 SSM 本身） |
| R2 | LSTM 走 cuDNN，batch 依赖的算法选择 + 首调预热 | 同 batch、同 warmup、多重复，报 mean±std |
| R3 | 三份 `Model*.py` 漂移（仓库已有 `Model copy.py` 等前车之鉴） | 以验收项 1 的 `diff` 行数守住；把"只允许改哪几行"写进文件头注释 |

---

## 9. 待确认

| # | 问题 |
|---|---|
| **Q1** | 四数据集 × 2 变体，是否需要 3 次重复（`script/重复测试/repeat.sh` 有先例）？ |
