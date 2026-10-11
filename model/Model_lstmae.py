"""LSTM-AE —— 供 `main.py` 直接调用（PRD5.2）。

**目的**：在 `script/robust/*` 那套加噪协议下，把模型换成传统 LSTM-AE，
其余（评测协议、训练调度、W/bs、噪声档）全部沿用 `main.py` 现状。

⚠️ **类名必须叫 `patchattmamba`**：`main.py` 对四个消融变体都是同一个类名
（`main.py:292` 直接 `_mod.patchattmamba(...)`），换名字就接不上。

接口契约（照 `model/Model.py` 的 `patchattmamba` 逐条对齐）：
    __init__(input_size, w_size, latent_size, batch_size, window_length, out_dim,
             dropout=0.0, args=None)
        注意 `w_size` 在本模型里**不使用**（它是原模型 CRR 的目标维度），
        保留只是为了让 main.py 的调用点不用改。
    forward(x)                      [B,W,C] → **标量 loss**（训练用）
    is_anomaly(loader, num_t, con_t)         → [N] 分数（测试/验证用）
    reconstructed_probability(x)    [B,W,C] → [B] 分数

    `attmamba` 属性故意留成空 ModuleList —— `main.py:296` 会统计它的参数量，
    没有这个属性会 AttributeError。

**异常分数**：原模型是 `|Σ(x_a − recon_mu)|`（`x_a` 是进 CRR 之前的表示）。
LSTM-AE 没有那个中间张量，故取标准做法 —— **对原始输入的重构误差**：
    score_i = Σ_{t,c} |X[i,t,c] − X̂[i,t,c]|
排序类指标（F1 的百分位阈值、AUPRC）都只依赖排序，量纲不影响。
"""
import torch
import torch.nn as nn


class patchattmamba(nn.Module):

    def __init__(self, input_size: int, w_size, latent_size, batch_size, window_length,
                 out_dim, dropout=0.0, args=None):
        super().__init__()
        self.input_size = input_size          # C
        self.window_length = window_length    # W
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.w_size = w_size
        h = getattr(args, 'lstm_hidden', 128) if args is not None else 128

        # ── encoder：整窗 → 末隐状态 → latent
        self.encoder = nn.LSTM(input_size, h, num_layers=1, batch_first=True)
        self.to_latent = nn.Linear(h, latent_size)
        # ── decoder：latent 广播到每个时间步作输入 → 逐拍展开 → 投回 C 维
        self.decoder = nn.LSTM(latent_size, h, num_layers=1, batch_first=True)
        self.out = nn.Linear(h, input_size)

        # main.py:296 会 sum(model.attmamba.parameters())；本模型没有那个中间块
        self.attmamba = nn.ModuleList()
        self._loss_fn = nn.MSELoss()
        self._last_recon = None      # 最近一次 forward 的输入重构（便于诊断）

    # ────────────────────────────────────────────── 前向与重构
    def _reconstruct(self, x):
        """x [B,W,C] → X̂ [B,W,C]。"""
        _, (hn, _) = self.encoder(x)
        z = self.to_latent(hn[-1])                                   # [B, latent]
        dec_in = z.unsqueeze(1).expand(-1, self.window_length, -1)   # [B, W, latent]
        dec_out, _ = self.decoder(dec_in)
        return self.out(dec_out)                                     # [B, W, C]

    def forward(self, x):
        """训练：返回标量 MSE（与 `Model.py` 一样只返回 loss）。"""
        x_hat = self._reconstruct(x)
        self._last_recon = x_hat
        return self._loss_fn(x_hat, x)

    # ────────────────────────────────────────────── 打分（照 Model.py 的写法定式）
    def reconstructed_probability(self, x):
        with torch.no_grad():
            x_hat = self._reconstruct(x)
            # 与原模型同构：对所有维度求和（原式 sum(x_a − recon_mu)，此处用绝对差）
            s = (x - x_hat).abs().sum(dim=(1, 2))
        return s.cpu()          # main.py 的 bf_search 用 np.percentile，需要 CPU/ndarray 友好

    def is_anomaly(self, x, num_t, con_t):
        """x 是一个可迭代的 batch 序列（DataLoader）。逐条与 Model.py 对齐。"""
        score = None
        for i, inputs in enumerate(x, 0):
            inputs = inputs.cuda()
            p = self.reconstructed_probability(inputs)
            if i == num_t:                       # 最后一个不满的 batch
                p = p[:con_t]
            score = p if score is None else torch.cat((score, p), dim=0)
        return score
