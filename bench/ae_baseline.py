#!/usr/bin/env python
"""PRD5.1 r3 · 三个主干模型（§5）+ MPA 前端复刻 + Gate 检查。

    WindowAE      展平窗口 → MLP 自编码器（主）
    WindowVAE     同结构 + KL；**评价一律用 reconstruction mean**（§5.2，避开采样噪声）
    MPAWindowAE   MPA → 通道维池化 → adapter → 同一套 VectorAE

关键设计（PRD §1.3 / §5.3）：

* **MPA 的输出没有时间轴**（`model/Model.py:206-227` 把 `patchnum×latent` 用 `projection1/2`
  压成了 `latent`），所以对 MLP-AE 只需**在通道维池化**即可接，**不需要**时间步语义。
* **三种模型共用同一份 `VectorAE` 定义**（相同 encoder/decoder 结构、相同 latent=256），
  差别只在喂进去的向量：`flatten(X)` vs `adapter(pool(MPA(X)))`。
* **MPA 前端按 `Model.py` 原样复刻**，子模块名刻意与父模型一致
  （`embedding_inpatch` / `embedding_patchwise` / `projection1` / `projection2`），
  以便用 `verify_mpa_equivalence()` 把权重灌进来做**逐值等价性检验**。

⚠️ 复刻保留了 `Model.py` 的一个既存行为：`x_combined`（即 H）与 `loss_c` **都只来自最后一个
patch 尺度**（循环里被覆盖），`pre_inpatch_mean/pre_patchwise_mean` 算了但从未被使用。
`verify_mpa_equivalence()` 会把这一点一并验掉。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embed import DataEmbedding                     # noqa: E402


# ─────────────────────────────────────────────────────────── MPA 前端（复刻）
class MPAFrontend(nn.Module):
    """`model/Model.py:201-227` 那段 patch 循环的等价复刻。返回 (H, loss_c)。"""

    def __init__(self, input_size, window_length, latent_size, patch_sizes=(5, 7), dropout=0.0):
        super().__init__()
        self.patch_size = list(patch_sizes)
        self.input_size = input_size
        self.latent_size = latent_size
        self.embedding_inpatch = nn.ModuleList()
        self.embedding_patchwise = nn.ModuleList()
        self.projection1 = nn.ModuleList()
        self.projection2 = nn.ModuleList()
        for ps in self.patch_size:
            patchnum = window_length // ps
            self.embedding_inpatch.append(DataEmbedding(ps, latent_size, dropout))
            self.embedding_patchwise.append(DataEmbedding(patchnum, latent_size, dropout))
            self.projection1.append(nn.Linear(patchnum * latent_size, latent_size))
            self.projection2.append(nn.Linear(ps * latent_size, latent_size))
        self.criterion = nn.CosineSimilarity(dim=-1)

    @staticmethod
    def _data_patch(x, patchsize):
        x = rearrange(x, 'b l m -> b m l')                    # Batch channel win
        return (rearrange(x, 'b m (n p) -> (b m) n p', p=patchsize),
                rearrange(x, 'b m (p n) -> (b m) p n', p=patchsize))

    def forward(self, x):
        """x [B,W,C] → H [B,C,2·latent]（= Model.py 里喂给 MSSA 的 x_combined）, loss_c 标量。"""
        crit = self.criterion
        for i, ps in enumerate(self.patch_size):
            xi, xp = self._data_patch(x, ps)
            xi = self.embedding_inpatch[i](xi)                 # [B*C, patchnum, L]
            xp = self.embedding_patchwise[i](xp)               # [B*C, patchsize, L]
            xi = self.projection1[i](rearrange(xi, 'b p q -> b (p q)'))
            xp = self.projection2[i](rearrange(xp, 'b p q -> b (p q)'))
            loss_c = (crit(xi, xp).mean() + crit(xp, xi).mean()) * 0.5
        H = torch.cat((xi, xp), dim=-1)                        # 只用最后一个尺度（与原实现一致）
        return rearrange(H, '(a b) c -> a b c', b=self.input_size), loss_c


# ─────────────────────────────────────────────────────────── AE 主干
class VectorAE(nn.Module):
    """展平向量上的 MLP 自编码器。encoder/decoder 分开，三个模型共用同一份定义。"""

    def __init__(self, in_dim, hidden=(1024, 512), latent=256):
        super().__init__()
        enc, prev = [], in_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        enc.append(nn.Linear(prev, latent))
        self.enc = nn.Sequential(*enc)
        dec, prev = [], latent
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        dec.append(nn.Linear(prev, in_dim))
        self.dec = nn.Sequential(*dec)

    def encode(self, v):
        return self.enc(v)

    def decode(self, z):
        return self.dec(z)


class WindowAE(nn.Module):
    kind = 'ae'

    def __init__(self, window, n_channel, hidden=(1024, 512), latent=256, **kw):
        super().__init__()
        self.window, self.n_channel = window, n_channel
        self.ae = VectorAE(window * n_channel, hidden, latent)

    def encode_input(self, x):
        return x.flatten(1)

    def forward(self, x):
        v = self.encode_input(x)
        z = self.ae.encode(v)
        return self.ae.decode(z).view_as(x), z

    @staticmethod
    def loss(x, recon, **kw):
        return F.mse_loss(recon, x)


class WindowVAE(nn.Module):
    """与 WindowAE 同结构同 latent；评价用 reconstruction mean（§5.2）。"""
    kind = 'vae'

    def __init__(self, window, n_channel, hidden=(1024, 512), latent=256, beta=1.0, **kw):
        super().__init__()
        self.window, self.n_channel, self.beta = window, n_channel, beta
        self.body = VectorAE(window * n_channel, hidden, latent * 2)
        self.dec = VectorAE(latent, hidden, window * n_channel)
        self.latent = latent

    def forward(self, x):
        v = x.flatten(1)
        mu, logvar = self.body.encode(v).chunk(2, dim=1)
        logvar = logvar.clamp(-10, 10)
        z = mu if not self.training else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec.decode(z).view_as(x), (mu, logvar)

    def loss(self, x, recon, aux=None):
        mu, logvar = aux
        rec = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return rec + self.beta * kl


class MPAWindowAE(nn.Module):
    """MPA → 通道维池化 → adapter → 同一套 VectorAE。重建目标始终是**原始 X**。

    ⚠️ **池化前必须处理常量成分。** `position_embedding` 是固定 buffer（每个窗口相同），
    通道维平均会让它**存活**而各通道的变化互相抵消 —— 实测未训练时
    `H` 未池化 sd/|v|=1.03，池化后掉到 0.195。`mpa_norm` 用**训练集统计量**（只从 train 拟合）
    先逐通道中心化/标准化再池化，把常量去掉：
        'none'        直接池化（默认，用于暴露问题）
        'center'      (H − μ_H)
        'standardize' (H − μ_H)/σ_H
    """
    kind = 'mpa_ae'

    def __init__(self, window, n_channel, hidden=(1024, 512), latent=256,
                 mpa_latent=256, patch_sizes=(5, 7), lam=1.0, mpa_norm='none',
                 mpa_pool='mean', **kw):  # noqa: D417
        super().__init__()
        self.window, self.n_channel, self.lam = window, n_channel, lam
        self.mpa_norm = mpa_norm
        self.mpa = MPAFrontend(n_channel, window, mpa_latent, patch_sizes)
        self.mpa_pool = mpa_pool
        # 'mean'   : 通道维平均 → [B, 2L]（会抹掉通道间差异；MSL 的 55 列是 1 信号 + 54 二值指示，尤其可疑）
        # 'concat' : 通道维拼接 → [B, C·2L]（保留通道身份，参数量大 55×）
        ad_in = (2 * mpa_latent) if mpa_pool == 'mean' else (n_channel * 2 * mpa_latent)
        self.adapter = nn.Linear(ad_in, window * n_channel)
        # ⚠️ MPA→AE 接口必须做尺度归一化：不归一化时 adapter 输出量级可达 1e2~1e3，
        # 而 `flatten(X)` 的量级是 1e-2（MinMax 后数据）——差 4~5 个数量级，
        # 同一个 lr 下 latent 会炸到 1e3 量级、decoder 退化成常量（见 §5.3 与实测）。
        self.out_norm = nn.LayerNorm(window * n_channel) if mpa_norm == 'layernorm' else nn.Identity()
        self.ae = VectorAE(window * n_channel, hidden, latent)
        self.register_buffer('h_mu', torch.zeros(1, n_channel, 2 * mpa_latent))
        self.register_buffer('h_sd', torch.ones(1, n_channel, 2 * mpa_latent))
        self._h_stats_set = False

    @torch.no_grad()
    def fit_h_stats(self, win, batch=256, device='cuda'):
        """用**训练窗**估计 H 的逐通道均值/标准差（只从 train 拟合，无泄漏）。"""
        self.eval()
        ws = win
        s = n = None
        for i in range(0, len(ws), batch):
            x = torch.as_tensor(ws[i:i + batch], device=device)
            H, _ = self.mpa(x)
            s = H.sum(0, keepdim=True) if s is None else s + H.sum(0, keepdim=True)
            n = (H * H).sum(0, keepdim=True) if n is None else n + (H * H).sum(0, keepdim=True)
        cnt = len(ws)
        mu = s / cnt
        sd = (n / cnt - mu * mu).clamp_min(1e-12).sqrt()
        self.h_mu.copy_(mu)
        self.h_sd.copy_(sd)
        self._h_stats_set = True
        return float(sd.mean())

    def _norm_h(self, H):
        if self.mpa_norm in ('none', 'layernorm'):
            return H
        if self.mpa_norm == 'standardize':
            return (H - self.h_mu) / self.h_sd
        return H - self.h_mu

    def _pool(self, H):
        return H.mean(dim=1) if self.mpa_pool == 'mean' else H.flatten(1)

    def encode_input(self, x):
        H, _ = self.mpa(x)
        return self.out_norm(self.adapter(self._pool(self._norm_h(H))))

    def forward(self, x):
        H, loss_c = self.mpa(x)
        v = self.out_norm(self.adapter(self._pool(self._norm_h(H))))
        z = self.ae.encode(v)
        return self.ae.decode(z).view_as(x), z, loss_c

    def loss(self, x, recon, aux=None):
        return F.mse_loss(recon, x) + self.lam * aux


class MPAWindowVAE(WindowVAE):
    """VAE 主干 + MPA 前端（可选第二主干）。"""
    kind = 'mpa_vae'

    def __init__(self, window, n_channel, hidden=(1024, 512), latent=256, beta=1.0,
                 mpa_latent=256, patch_sizes=(5, 7), lam=1.0, **kw):
        super().__init__(window, n_channel, hidden, latent, beta)
        self.mpa = MPAFrontend(n_channel, window, mpa_latent, patch_sizes)
        self.adapter = nn.Linear(2 * mpa_latent, window * n_channel)
        # 替换 body 的输入分支：先把 MPA 输出适配成 in_dim，再走原来的 encoder
        self.lam = lam

    def forward(self, x):
        H, loss_c = self.mpa(x)
        v = self.adapter(H.mean(dim=1))
        mu, logvar = self.body.encode(v).chunk(2, dim=1)
        logvar = logvar.clamp(-10, 10)
        z = mu if not self.training else mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec.decode(z).view_as(x), (mu, logvar), loss_c

    def loss(self, x, recon, aux=None):
        out = super().loss(x, recon, aux[0])
        return out + self.lam * aux[1]


REGISTRY = {'ae': WindowAE, 'vae': WindowVAE,
            'mpa_ae': MPAWindowAE, 'mpa_vae': MPAWindowVAE}


def loss_of(model, x, out):
    """统一取损失（四种模型的 forward 返回元组长度不同，集中在这里）。"""
    k = getattr(model, 'kind', 'ae')
    if k == 'ae':
        return F.mse_loss(out[0], x)
    if k == 'vae':
        return model.loss(x, out[0], out[1])
    if k == 'mpa_ae':
        return model.loss(x, out[0], out[2])
    if k == 'mpa_vae':
        return model.loss(x, out[0], (out[1], out[2]))
    raise SystemExit(f'未知 kind={k}')


def mse_of(recon, x):
    """§5.1 的逐元素 MSE（1/TC ‖X−X̂‖²），评价口径统一用这个。"""
    return F.mse_loss(recon, x).item()


def build(name, window, n_channel, **kw):
    if name not in REGISTRY:
        raise SystemExit(f'未知模型 {name}；可选 {list(REGISTRY)}')
    return REGISTRY[name](window, n_channel, **kw)


def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    ad = sum(p.numel() for p in m.adapter.parameters()) if hasattr(m, 'adapter') else 0
    mpa = sum(p.numel() for p in m.mpa.parameters()) if hasattr(m, 'mpa') else 0
    return dict(total=total, total_M=total / 1e6, adapter=ad, mpa=mpa)


# ─────────────────────────────────────────────────────────── Gate 检查
@torch.no_grad()
def gate_n(model, x_real, tol=1e-3):
    """Gate N（§7）：**非常量**检查。真实 / 全零 / 随机 三类输入放同一 batch。

    ⚠️ 这条是 2026-10-10 事故的直接产物：SECOND 的所有 ckpt 输出常量，
    而当时只查了"重复前向一致"——**常量也满足那一条**。

    判据：重建张量的跨样本 sd > 取值量级的 tol 倍。返回 (ok, 详情)。
    """
    was_training = model.training
    model.eval()
    B = x_real.shape[0]
    zero = torch.zeros_like(x_real[:B])
    rand = torch.randn_like(x_real[:B])
    batch = torch.cat([x_real, zero, rand], 0)
    n = x_real.shape[0]
    with torch.no_grad():
        out = model(batch)
        recon = out[0]
        # ① 重建的跨样本 sd
        scale = recon.abs().mean().item()
        sd = recon.std(dim=0).mean().item()
        d_rz = (recon[:n] - recon[n:2 * n]).abs().max().item()
        # ② **喂给 AE encoder 的向量**的跨样本 sd —— 比重建更直接：
        #    MPA 输出若被固定 positional embedding 主导，这一步就会暴露出来
        vin = model.encode_input(batch)
        v_scale = vin.abs().mean().item()
        v_sd = vin.std(dim=0).mean().item()
        v_drz = (vin[:n] - vin[n:2 * n]).abs().max().item()
    detail = dict(recon_abs_mean=scale, recon_cross_sample_sd=sd, ratio=sd / (scale + 1e-30),
                  real_vs_zero_max_abs_diff=d_rz,
                  enc_input_abs_mean=v_scale, enc_input_cross_sample_sd=v_sd,
                  enc_input_ratio=v_sd / (v_scale + 1e-30),
                  enc_input_real_vs_zero_max_abs_diff=v_drz, tol=tol)
    if was_training:
        model.train()
    return bool(detail['ratio'] > tol), detail


def verify_mpa_equivalence(seed=0, verbose=True):
    """把 `patchattmamba` 的 MPA 权重灌进 `MPAFrontend`，比对是否逐值相等。

    做法：构造一个**小尺寸**的 `patchattmamba`，hook `attmamba[0]` 的输入（那正是
    `pipline` 喂给 MSSA 的 `x_combined`），再与 `MPAFrontend` 的输出比。
    """
    import importlib
    from model.mamba import ModelArgs
    import main as mainmod
    C, W, L = 8, 35, 16
    mainmod.set_seed(seed)
    mod = importlib.import_module(mainmod.BACKBONE_REGISTRY['mamba'])
    args = ModelArgs()
    full = mod.patchattmamba(C, C * 5, L, 4, W, 5, args=args).eval()
    mpa = MPAFrontend(C, W, L).eval()

    sd = {k: v for k, v in full.state_dict().items()
          if k.split('.')[0] in ('embedding_inpatch', 'embedding_patchwise',
                                 'projection1', 'projection2')}
    missing = mpa.load_state_dict(sd, strict=True)          # strict：键必须完全对上

    cap = {}
    h = full.attmamba[0].register_forward_pre_hook(lambda _m, i: cap.setdefault('H', i[0].detach()))
    x = torch.randn(6, W, C)
    with torch.no_grad():
        full.pipline(x)
        H_ref = cap['H']
        H_my, loss_c = mpa(x)
    h.remove()
    d = (H_ref - H_my).abs().max().item()
    ok = d == 0.0
    info = dict(H_shape=tuple(H_ref.shape), max_abs_diff=d, identical=ok,
                loss_c=float(loss_c), keys_loaded=len(sd), strict_load_missing=missing)
    if verbose:
        print(f'  [Gate M] MPA 等价性: H{tuple(H_ref.shape)}  max|Δ|={d:.3e}  '
              f'→ {"逐值相同 ✅" if ok else "不一致 ❌"}   loss_c={float(loss_c):.6f}')
    return ok, info


if __name__ == '__main__':
    verify_mpa_equivalence()
