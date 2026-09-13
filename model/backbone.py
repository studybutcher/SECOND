"""消融用中间块（mixer）实现。

接口契约与 `model/mamba.py:340` 的 `attmambaBlock` 完全一致：
    __init__(args: ModelArgs)
    forward(x: [b, l, d_model]) -> [b, l, d_model]      # 形状不变

因此可以直接塞进 `mamba.py:146` 的 `ResidualBlock` 骨架
（`output = mixer(norm(x)) + x`），只需替换 mixer 的构造。
"""
from torch import nn

from .mamba import ModelArgs, RMSNorm


class ResidualBlockGeneric(nn.Module):
    """与 `mamba.py:146` 的 ResidualBlock 同构，只把 mixer 参数化。

    基线 mamba.py 里是 `self.mixer = attmambaBlock(args)`；
    这里改成由外部传入 mixer_cls，其余（RMSNorm + 残差）完全一致。
    """
    def __init__(self, args: ModelArgs, mixer_cls):
        super().__init__()
        self.args = args
        self.mixer = mixer_cls(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return self.mixer(self.norm(x)) + x


class TransformerMixer(nn.Module):
    """标准 pre-norm Transformer block（attention + FFN，各带一个 LayerNorm）。"""

    def __init__(self, args: ModelArgs, nhead: int = 4, ffn_dim: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        d = args.d_model
        assert d % nhead == 0, f'd_model={d} 不能被 nhead={nhead} 整除'
        self.nhead = nhead
        self.ffn_dim = ffn_dim
        self.attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ffn_dim, d),
        )

    def forward(self, x):
        h = self.ln1(x)
        # need_weights=False：省掉 l×l 注意力权重的回传
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x


class LSTMMixer(nn.Module):
    """双向 LSTM：输出 2*d，再线性投影回 d。"""

    def __init__(self, args: ModelArgs, dropout: float = 0.1):
        super().__init__()
        d = args.d_model
        self.lstm = nn.LSTM(d, d, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * d, d)

    def forward(self, x):
        y, _ = self.lstm(x)
        return x + self.proj(y)


class TransformerResidualBlock(ResidualBlockGeneric):
    def __init__(self, args: ModelArgs):
        super().__init__(args, TransformerMixer)


class LSTMMixerBlock(ResidualBlockGeneric):
    def __init__(self, args: ModelArgs):
        super().__init__(args, LSTMMixer)
