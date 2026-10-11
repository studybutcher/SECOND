#!/usr/bin/env python
"""PRD5.1 r3 §6 · 一张图：Normal Reconstruction Learning Curve。

    Figure X  左：(a) 固定 clean-normal val 的 MSE vs epoch，按 ρ 一条线（seed 均值 + 带）
              右：(b) E_N 与 ΔE_N 随 ρ 的点图（mean ± std）

产出：figs/ae_mpa/normal_mse_vs_epoch.pdf（+ .png 便于快速查看尺寸）

用法：python bench/plot_ae.py [--model ae]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')                                      # 无显示环境
import matplotlib.pyplot as plt                            # noqa: E402
import numpy as np                                         # noqa: E402

OUT = os.path.join('figs', 'ae_mpa')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ae')
    ap.add_argument('--min-epochs', type=int, default=20, help='忽略未跑完的 run')
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()

    by_rho = {}
    for p in sorted(glob.glob(f'results/ae_mpa/{a.model}_*_rho*_seed*.json')):
        d = json.load(open(p))
        if d['epochs'] < a.min_epochs:
            continue
        by_rho.setdefault(d['rho'], []).append(d)
    if not by_rho:
        raise SystemExit('[FAIL] 没有可用的结果 json')

    os.makedirs(a.out_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap('viridis')
    rhos = sorted(by_rho)
    colors = {r: cmap(i / max(len(rhos) - 1, 1)) for i, r in enumerate(rhos)}

    for r in rhos:
        runs = by_rho[r]
        L = min(len(d['curve']) for d in runs)
        M = np.array([[d['curve'][e]['val_mse'] for e in range(L)] for d in runs])
        mu, sd = M.mean(0), M.std(0, ddof=1) if len(runs) > 1 else np.zeros(L)
        ep = np.arange(L)
        ax1.plot(ep, mu, color=colors[r], label=f'ρ={r:.0%}  (n={len(runs)})', lw=1.8)
        ax1.fill_between(ep, mu - sd, mu + sd, color=colors[r], alpha=0.22, lw=0)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('clean-normal validation MSE')
    ax1.set_title('(a) Normal reconstruction learning curve')
    ax1.set_yscale('log')
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25, ls=':')

    mean = [np.mean([d['E_N_med10'] for d in by_rho[r]]) for r in rhos]
    std = [np.std([d['E_N_med10'] for d in by_rho[r]], ddof=1) if len(by_rho[r]) > 1 else 0
           for r in rhos]
    ax2.errorbar([100 * r for r in rhos], mean, yerr=std, marker='o', capsize=4,
                 color='#333', lw=1.6, ms=6)
    base = mean[0] if rhos and rhos[0] == 0.0 else np.nan
    for r, m in zip(rhos, mean):
        if r > 0 and not np.isnan(base):
            ax2.annotate(f'Δ={100*(m-base)/base:+.1f}%', (100 * r, m),
                         textcoords='offset points', xytext=(4, 7), fontsize=8)
    ax2.set_xlabel('training contamination ρ (%)')
    ax2.set_ylabel('$E_N$ (median of last 10 epochs)')
    ax2.set_title('(b) Normal reconstruction quality vs ρ')
    ax2.grid(alpha=0.25, ls=':')

    fig.suptitle(f"PRD5.1 r3 - {a.model.upper()} - MSL (W=70) - structured anomaly-segment injection, alpha=1.5 sigma",
                 fontsize=10)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'normal_mse_vs_epoch.{ext}'), dpi=140,
                    bbox_inches='tight')
    print(f'saved: {a.out_dir}/normal_mse_vs_epoch.pdf (+.png)')
    for r in rhos:
        runs = by_rho[r]
        e = [d['E_N'] for d in runs]
        m = [d['E_N_med10'] for d in runs]
        print(f'  ρ={r:.2f}: n={len(runs)}  E_N末={np.mean(e):.6f}±{np.std(e,ddof=1) if len(e)>1 else 0:.6f}'
              f'  med10={np.mean(m):.6f}')


if __name__ == '__main__':
    main()
