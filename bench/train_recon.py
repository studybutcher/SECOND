#!/usr/bin/env python
"""PRD5.2 r2 §3–§4 · **AE 与 LSTM-AE 共用**的训练与评价驱动。

一个 run 同时产出**两套口径**：

    F1   —— 项目现有协议：`bf_search(score, label, step_num=700)` → `adjust_predicts`
            （注意 `adjust_predicts` 的 threshold **是百分位**，`np.percentile(score, t)`
             后再 `score > thresh`，见 eval_method.py:44 的注释，故对任意量纲的分数都适用）
    MSE  —— 有分辨力的那一套：E_N（val_normal）、E_A（**评测侧**异常窗）、E_A/E_N

公平性（PRD §3）：两模型**共用本驱动** —— 同划分、同注入、同调度、同评价窗、**同一分数定义**
（`score_i = Σ_{t,c} |X − X̂|`，两模型一致）。

⚠️ **Gate N 用 `真实vs全零 max|Δ|` 相对数据尺度判**，不用 `sd/|mean|`
（后者在常量输出的 MPA+AE 上误判为通过，见 PRD §8）。

用法：
    python bench/train_recon.py --model ae     --dataset SWaT --rho 0    --seed 42 --epochs 100
    python bench/train_recon.py --model lstmae --dataset MSL  --rho 0.20 --seed 42 --epochs 100
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_method import bf_search, calc_point2point, adjust_predicts   # noqa: E402
from bench.donor_split import load_split, OUT as ART                   # noqa: E402
from bench.ae_data import windows_from                                 # noqa: E402

RES = os.path.join('results', 'recon')


def P(*args, **kw):
    """带 flush 的 print：单 run 主要耗时在评估阶段，不 flush 就看不出被杀在哪一步。"""
    print(*args, **kw, flush=True)


def build_model(name, W, C, latent, bs, dev):
    if name == 'ae':
        from bench.ae_baseline import build as build_ae
        return build_ae('ae', W, C, latent=latent).to(dev)
    if name == 'lstmae':
        from model.Model_lstmae import patchattmamba
        return patchattmamba(C, C * 5, latent, bs, W, 5).to(dev)
    raise SystemExit(f'未知模型 {name}')


def recon_of(model, x):
    r = getattr(model, '_reconstruct', None)      # LSTM-AE（forward 按 main.py 契约返回标量）
    return r(x) if r is not None else model(x)[0]


@torch.no_grad()
def forward_all(model, win, batch=256, dev='cuda'):
    """→ [N,W,C]。"""
    model.eval()
    out = []
    for i in range(0, len(win), batch):
        out.append(recon_of(model, torch.as_tensor(win[i:i + batch], device=dev)).cpu())
    return torch.cat(out)


def pw_mse(r, x):
    return ((r - x) ** 2).mean(dim=(1, 2)).numpy()


def pw_score(r, x):
    return (r - x).abs().sum(dim=(1, 2)).numpy()


def gate_n(model, x_real, tol=1e-3, dev='cuda'):
    """非常量检查：`真实vs全零 max|Δ|` **相对数据尺度**。"""
    n = min(64, len(x_real))
    xr = torch.as_tensor(x_real[:n], device=dev)
    with torch.no_grad():
        rr = recon_of(model, xr)
        rz = recon_of(model, torch.zeros_like(xr))
    d = float((rr - rz).abs().max())
    scale = float(xr.abs().mean())
    return dict(ok=bool(d / (scale + 1e-30) > tol), real_vs_zero_max_abs_diff=d,
                data_scale=scale, ratio=d / (scale + 1e-30), tol=tol)


def oracle_f1(score, label, step_num=200):
    """复用项目协议。返回 (f1, best_pct, p, r)。"""
    t, th, _ = bf_search(score, label, step_num=step_num)
    return float(t[0]), float(th), float(t[1]), float(t[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ae', choices=['ae', 'lstmae'])
    ap.add_argument('--dataset', default='SWaT')
    ap.add_argument('--rho', type=float, default=0.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contam-seed', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--latent', type=int, default=256)
    ap.add_argument('--bf-steps', type=int, default=200,
                    help='bf_search 的阈值搜索步数。main.py 用 700；**F1 只是参照列**，'
                         '200 与 700 的最优点几乎相同，但快 3.5×（bf_search 是逐元素的 '
                         'Python 循环，是单 run 的主要耗时来源）')
    ap.add_argument('--out-dir', default=RES)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--no-save-ckpt', action='store_true')
    a = ap.parse_args()
    cs = a.seed if a.contam_seed is None else a.contam_seed
    dev = 'cuda'

    meta = load_split(a.dataset)
    W, C = meta['window'], meta['dim']
    tr_arr = np.load(f'{ART}/{a.dataset}_train_normal.npy')
    if a.rho > 0:
        tag_c = f'{a.dataset}_train_rho{int(round(a.rho*100)):02d}_seed{cs}'
        tr_arr = np.load(f'{ART}/{tag_c}.npy')
    va_arr = np.load(f'{ART}/{a.dataset}_val_normal.npy')
    te_arr = np.load(f'{ART}/{a.dataset}_test_norm.npy')
    x_tr, _, _ = windows_from(tr_arr, meta['blocks_train'], W)
    x_va, _, _ = windows_from(va_arr, meta['blocks_val'], W)
    ei = np.array(meta['eval_win_idx']); el = np.array(meta['eval_win_label'])
    tst = np.lib.stride_tricks.sliding_window_view(te_arr, W, axis=0).transpose(0, 2, 1)
    x_te, y_te = tst[ei].astype(np.float32), el.astype(int)
    print(f'{a.model} | {a.dataset} ρ={a.rho:.0%} seed={a.seed} | train {x_tr.shape} '
          f'val {x_va.shape} test(eval) {x_te.shape} 异常率 {y_te.mean():.4f}')

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = build_model(a.model, W, C, a.latent, a.batch_size, dev)
    n_par = sum(p.numel() for p in model.parameters())
    gate_v = dict(in_dim=W * C, latent=a.latent, ratio=(W * C) / a.latent,
                  ok=True, params_M=n_par / 1e6)
    print(f'  参数 {n_par/1e6:.2f}M  [Gate V] 瓶颈 {W*C}→{a.latent} ({gate_v["ratio"]:.1f}×)')

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.min_lr)
    n, curve, t0 = len(x_tr), [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, a.batch_size):
            x = torch.as_tensor(x_tr[perm[i:i + a.batch_size]], device=dev)
            r = recon_of(model, x)
            loss = torch.nn.functional.mse_loss(r, x)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(x); cnt += len(x)
        sched.step()
        e_n = pw_mse(forward_all(model, x_va), torch.as_tensor(x_va)).mean()
        curve.append(dict(epoch=ep, train_mse=tot / cnt, val_mse=float(e_n)))
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f'    ep{ep:4d}  train {tot/cnt:.6f}  val_MSE {e_n:.6f}  ({time.time()-t0:.0f}s)',
                  flush=True)

    # ── 指标 ─────────────────────────────────────────────────────────
    def mse_on(win, mask=None):
        r = forward_all(model, win if mask is None else win[mask])
        x = torch.as_tensor(win if mask is None else win[mask])
        return pw_mse(r, x)

    print('  [阶段] 训练结束，开始评估...', flush=True)
    v = mse_on(x_va)
    tn, ta = y_te == 0, y_te == 1
    # **测试集只前向一次**（原来 E_A / E_A_clip / score 各跑一遍，白花 2× 时间与内存）。
    x_te_t = torch.as_tensor(x_te)
    r_te = forward_all(model, x_te)
    score = pw_score(r_te, x_te_t)
    m_te = pw_mse(r_te, x_te_t)
    # clip 版只在**真有超出训练范围的值**时才需要（否则它恒等于 E_A，跳过第二次前向）
    lo, hi = float(x_te.min()), float(x_te.max())
    if lo >= 0.0 and hi <= 1.0:
        m_te_c = m_te
        print(f'  [clip] 测试值域 [{lo:.4f}, {hi:.4f}] ⊂ [0,1] ⇒ E_A_clip ≡ E_A，跳过第二次前向')
    else:
        r_te_c = forward_all(model, np.clip(x_te, 0.0, 1.0))
        m_te_c = pw_mse(r_te_c, torch.as_tensor(np.clip(x_te, 0.0, 1.0)))
        print(f'  [clip] 测试值域 [{lo:.4f}, {hi:.4f}] 越界 ⇒ 已另跑一次 clip 前向')
    print('  [阶段] bf_search 开始（本 run 主要耗时）...', flush=True)
    f1, pct, p, r_ = oracle_f1(score, y_te, a.bf_steps)
    # 未调整的旁证
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = (score > np.percentile(score, pct)).astype(int)
    u = calc_point2point(pred, y_te)
    auprc, auroc = average_precision_score(y_te, score), roc_auc_score(y_te, score)
    E_N, E_A = float(v.mean()), float(m_te[ta].mean())
    E_A_c = float(m_te_c[ta].mean())
    gn = gate_n(model, x_va)
    tail = sorted(x['val_mse'] for x in curve[-10:])
    conv = max(tail) / max(min(tail), 1e-30)
    P(f'  E_N={E_N:.6f}  E_N(test)={m_te[tn].mean():.6f}  E_A={E_A:.6f}  '
          f'E_A/E_N={E_A/E_N:.3f}  E_A_clip/E_N={E_A_c/E_N:.3f}')
    P(f'  F1={f1:.4f} (P={p:.4f} R={r_:.4f} @pct={pct:.2f})  未调整 F1={u[0]:.4f}  '
          f'AUPRC={auprc:.4f} AUROC={auroc:.4f}  (base={y_te.mean():.4f})')
    P(f'  [Gate N] 真实vs全零 max|Δ|/scale={gn["ratio"]:.3e} (tol={gn["tol"]}) → '
          f'{"✅" if gn["ok"] else "❌ 常量！"}   [收敛] 末10轮极差比={conv:.2f}')

    tag = a.tag or f'{a.model}_{a.dataset}_rho{int(round(a.rho*100)):02d}_seed{a.seed}'
    os.makedirs(a.out_dir, exist_ok=True)
    if not a.no_save_ckpt:
        ck = f'checkpoint/recon/{tag}.pt'
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save(model.state_dict(), ck)
    rec = dict(model=a.model, dataset=a.dataset, rho=a.rho, seed=a.seed, contam_seed=cs,
               window=W, dim=C, latent=a.latent, epochs=a.epochs, lr=a.lr,
               batch_size=a.batch_size, params_M=n_par / 1e6,
               n_train_win=int(len(x_tr)), n_val_win=int(len(x_va)), n_test_win=int(len(x_te)),
               E_N=E_N, E_N_test_normal=float(m_te[tn].mean()), E_A=E_A, E_A_clip=E_A_c,
               ratio=float(E_A / E_N), ratio_clip=float(E_A_c / E_N),
               F1=f1, F1_pct=pct, F1_P=p, F1_R=r_, F1_unadj=float(u[0]), P_unadj=float(u[1]),
               R_unadj=float(u[2]), AUPRC=float(auprc), AUROC=float(auroc),
               base_rate=float(y_te.mean()), curve=curve,
               gates=dict(V=gate_v, N=gn, conv_tail_ratio=float(conv),
                          conv_ok=bool(conv < 1.2)),
               ckpt=None if a.no_save_ckpt else os.path.abspath(ck),
               wall_sec=round(time.time() - t0, 1),
               timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(rec, open(f'{a.out_dir}/{tag}.json', 'w'), indent=1, ensure_ascii=False)
    np.savez_compressed(f'{a.out_dir}/{tag}_scores.npz', score=score, label=y_te,
                        bf_pct=pct, f1=f1)
    print(f'  saved: {a.out_dir}/{tag}.json')
    if not (gn['ok'] and conv < 1.2):
        print('  [WARN] Gate N 或收敛未过 —— 该 run 的 MSE 列不可解读')


if __name__ == '__main__':
    main()
