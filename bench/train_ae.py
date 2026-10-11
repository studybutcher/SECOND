#!/usr/bin/env python
"""PRD5.1 r3 · AE / VAE / MPA+AE 的训练与评价驱动（§5–§7）。

为什么**不**复用 `main.py`（PRD r3 §11）：
  * `main.py` 绑定 `patchattmamba` 的接口、自带 `val ⊂ test` 的历史缺陷、且推理不调 `eval()`；
  * 本实验只需要「训练 + 在固定 clean-normal val 上算 MSE」，另写一个小驱动更干净可控。

口径（全部来自 PRD r3）：
  * 训练窗来自 `ae_contam.py` 产出的污染数组（ρ=0 即 `*_train_normal.npy`）；
  * **固定轮数**，不早停、不按模型选不同策略（§4.3）；
  * 评价一律 `model.eval()` + `no_grad`（§4.6），VAE 用 reconstruction mean（§5.2）；
  * $E_N$ = val_normal 上的逐元素 MSE；$E_A$ = anomaly probe 上的逐元素 MSE。

用法：
    python bench/train_ae.py --model ae --dataset MSL --rho 0    --seed 42 --epochs 50
    python bench/train_ae.py --model mpa_ae --dataset MSL --rho 0.2 --seed 42 --epochs 50
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.ae_data import load, windows_from                    # noqa: E402
from bench import ae_baseline as B                              # noqa: E402

OUT_DIR = os.path.join('results', 'ae_mpa')
ART = os.path.join('artifacts', 'ae_mpa')


@torch.no_grad()
def evaluate(model, win, batch=512, device='cuda'):
    """→ 逐元素 MSE（§5.1 的 1/TC‖X−X̂‖²），eval 口径。"""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(win), batch):
        x = torch.as_tensor(win[i:i + batch], device=device)
        recon = model(x)[0]
        tot += torch.nn.functional.mse_loss(recon, x, reduction='sum').item()
        n += x.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ae', choices=list(B.REGISTRY))
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window', type=int, default=70)
    ap.add_argument('--rho', type=float, default=0.0, help='0 = clean')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contam-seed', type=int, default=None, help='默认 = --seed')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-schedule', default='cosine', choices=['none', 'cosine'],
                    help='Adam lr=1e-3 跑满 50 epoch 会在最小值附近抖动（实测 val_MSE 出现 '
                         '0.007 的尖峰，一个 seed 直接跳出好盆地）。cosine 衰减是统一的、'
                         '不涉及按模型选型的修正')
    ap.add_argument('--min-lr', type=float, default=1e-5)
    ap.add_argument('--latent', type=int, default=256)
    ap.add_argument('--lam', type=float, default=1.0, help='MPA 一致性损失权重 λ')
    ap.add_argument('--mpa-norm', default='none', choices=["none","center","standardize","layernorm"],
                    help='池化前对 H 的逐通道处理（用训练集统计量）；见 ae_baseline.MPAWindowAE')
    ap.add_argument('--mpa-pool', default='mean', choices=['mean','concat'], help='MPA 输出怎么进 AE')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--no-save-ckpt', action='store_true',
                    help='默认存 ckpt（分析"局部损害"要逐窗重建误差，必须能重载模型）')
    a = ap.parse_args()
    cs = a.contam_seed if a.contam_seed is not None else a.seed
    dev = 'cuda'

    _, val_arr, meta = load(a.dataset)
    C = {'MSL': 55, 'SMAP': 25, 'SWaT': 51, 'WADI': 127}[a.dataset]
    if a.rho > 0:
        tag_c = f'{a.dataset}_train_rho{int(round(a.rho * 100)):02d}_seed{cs}'
        train_arr = np.load(f'{ART}/{tag_c}.npy')
        man = json.load(open(f'{ART}/{tag_c}.json'))
    else:
        train_arr = np.load(f'{ART}/{a.dataset}_train_normal.npy')
        man = dict(rho=0.0, note='clean')

    x_tr, _, _ = windows_from(train_arr, meta['blocks_train'], a.window)
    x_va, _, _ = windows_from(val_arr, meta['blocks_val'], a.window)
    probe = np.load(f'{ART}/{a.dataset}_probe.npz')
    print(f'{a.model} | {a.dataset} ρ={a.rho:.0%} seed={a.seed} | '
          f'train {x_tr.shape} val {x_va.shape} probe(anom) {probe["x_anom"].shape}')

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    kw = dict(latent=a.latent, lam=a.lam)
    if a.model.startswith('mpa'):
        kw['mpa_norm'] = a.mpa_norm; kw['mpa_pool'] = a.mpa_pool
    model = B.build(a.model, a.window, C, **kw).to(dev)
    par = B.count_params(model)
    print(f'  参数: total={par["total_M"]:.2f}M  adapter={par["adapter"]/1e6:.3f}M  '
          f'mpa={par["mpa"]/1e6:.3f}M  mpa_norm={a.mpa_norm}')
    if a.mpa_norm != 'none' and hasattr(model, 'fit_h_stats'):
        mu_sd = model.fit_h_stats(x_tr[:4096], device=dev)   # 只用**训练窗**拟合
        print(f'  H 统计量已从训练窗拟合（{min(len(x_tr), 4096)} 条）：σ_H 均值={mu_sd:.4f}）')

    # ── Gate V：瓶颈有效（latent ≪ 输入维，不是恒等映射）──────────────
    gate_v = dict(in_dim=a.window * C, latent=a.latent,
                  ratio=(a.window * C) / a.latent)
    gate_v['ok'] = bool(a.latent * 2 <= a.window * C)
    print(f'  [Gate V] 瓶颈 {gate_v["in_dim"]} → {a.latent} '
          f'({gate_v["ratio"]:.1f}×)  → {"✅" if gate_v["ok"] else "❌"}')

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.min_lr)
             if a.lr_schedule == 'cosine' else None)
    n = len(x_tr)
    curve = []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot, tot_rec, cnt = 0.0, 0.0, 0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            x = torch.as_tensor(x_tr[idx], device=dev)
            out = model(x)
            loss = B.loss_of(model, x, out)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            # **重建项单独记**：MPA 的 `loss_c` 是余弦相似度（目标 −1 量级），
            # 合并后的 total 可能为负，与 MSE 不可比，不能混报（详见 §6）
            tot_rec += torch.nn.functional.mse_loss(out[0], x).item() * len(idx)
            cnt += len(idx)
        e_n = evaluate(model, x_va)
        curve.append(dict(epoch=ep, train_loss=tot / cnt, train_recon_mse=tot_rec / cnt,
                          val_mse=e_n, lr=opt.param_groups[0]['lr']))
        if sched is not None:
            sched.step()
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f'    epoch {ep:3d}  train_MSE {tot_rec/cnt:.5f}  (total {tot/cnt:+.5f})  '
                  f'val_MSE {e_n:.5f}  ({time.time()-t0:.0f}s)', flush=True)

    E_N = evaluate(model, x_va)                 # clean-normal val（主指标）
    E_Ntest = evaluate(model, probe['x_normal'])
    E_A = evaluate(model, probe['x_anom'])
    # ⚠️ MinMax scaler 对 test 中超出 train 范围的值会外推：实测 MSL 异常 probe max=46，
    # 而 train 只在 [0,1]。E_A 因此被极少数极端值主导 → 同时报一个 clip 到训练范围的版本。
    E_A_clip = evaluate(model, np.clip(probe['x_anom'], 0.0, 1.0))
    ok_n, det_n = B.gate_n(model, torch.as_tensor(x_va[:64], device=dev))
    print(f'  E_N={E_N:.6f}  E_N(test normal)={E_Ntest:.6f}  E_A(anomaly)={E_A:.6f} '
          f'(clip 到 [0,1] 后 {E_A_clip:.6f})  比值 E_N/E_A_clip={E_N/max(E_A_clip,1e-30):.3f}')
    print(f'  [Gate N] 非常量: ratio={det_n["ratio"]:.3e} (tol={det_n["tol"]}) '
          f'真实vs全零 max|Δ|={det_n["real_vs_zero_max_abs_diff"]:.3e} → '
          f'{"✅" if ok_n else "❌ 常量！"}')
    print(f'  [诊断] 喂给 AE encoder 的向量: ratio={det_n["enc_input_ratio"]:.3e}  '
          f'真实vs全零 max|Δ|={det_n["enc_input_real_vs_zero_max_abs_diff"]:.3e}'
          + ('   ⚠️ 输入近似常量（MPA 输出被固定 positional 主导？）'
             if det_n['enc_input_ratio'] < 1e-2 else ''))

    # 末段稳健量：最后 10 轮的 val MSE 中位/最小（固定轮数下用来看"末点是否只是个抖动"）
    tail = sorted(x['val_mse'] for x in curve[-10:])
    v_med10 = tail[len(tail) // 2]
    v_min = min(x['val_mse'] for x in curve)
    ep_min = min(range(len(curve)), key=lambda i: curve[i]['val_mse'])

    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or f'{a.model}_{a.dataset}_rho{int(round(a.rho*100)):02d}_seed{a.seed}'
    rec = dict(model=a.model, dataset=a.dataset, rho=a.rho, seed=a.seed, contam_seed=cs,
               window=a.window, n_channel=C, latent=a.latent, lam=a.lam,
               epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
               lr_schedule=a.lr_schedule, min_lr=a.min_lr,
               E_N_med10=v_med10, E_N_min=v_min, E_N_min_epoch=ep_min,
               n_train_win=int(len(x_tr)), n_val_win=int(len(x_va)),
               n_probe_normal=int(len(probe['x_normal'])), n_probe_anom=int(len(probe['x_anom'])),
               params=par, E_N=E_N, E_N_test_normal=E_Ntest, E_A=E_A, E_A_clip=E_A_clip,
               ratio_EN_EA=E_N / E_A, ratio_EN_EA_clip=E_N / max(E_A_clip, 1e-30),
               curve=curve, gates=dict(V=gate_v, N=det_n | {'ok': ok_n}),
               contam=({k: man[k] for k in ('rho', 'rho_win', 'n_contaminated', 'n_segments',
                                            'mean_seg_len', 'alpha')} if a.rho > 0 else man),
               wall_sec=round(time.time() - t0, 1),
               timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
    if not a.no_save_ckpt:
        ck = f'checkpoint/ae_mpa/{tag}.pt'
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save(model.state_dict(), ck)
        rec['ckpt'] = os.path.abspath(ck)
        print(f'  ckpt: {ck}')
    with open(f'{a.out_dir}/{tag}.json', 'w') as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f'  saved: {a.out_dir}/{tag}.json')
    if not (gate_v['ok'] and ok_n):
        raise SystemExit('[FAIL] Gate V/N 未通过，结果不可用')


if __name__ == '__main__':
    main()
