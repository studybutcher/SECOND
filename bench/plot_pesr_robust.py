#!/usr/bin/env python
"""PRD6 · PESR 在基础高斯噪声下的多 seed 结果(MSL)。

版式对齐 `bench/plot_lstmae_robust.py::lstmae_robust_two_rulers`:

    左:F1(point-adjust, oracle 阈值) vs η —— 每 seed 一个点 + 均值±std
    右:Precision / Recall vs η —— 同一批 run

⚠️ 原图右栏是 AUROC,但本轮没有 `--score-dump`,JSON 里没有分数,
   无法计算未调整 AUROC。右栏因此改为 P/R,用来检查 F1 的"平"
   是否来自两者的反向抵消。要恢复 AUROC 栏,需带 --score-dump 重跑。

用法:python bench/plot_pesr_robust.py
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402

OUT = os.path.join('figs', 'robust_ms')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='MSL')
    ap.add_argument('--json-glob', default=None,
                    help='默认 results/robust_ms/pesr_<data>_gauss_noise*_seed*.json')
    ap.add_argument('--out-dir', default=OUT)
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    if a.json_glob is None:
        a.json_glob = (f'results/robust_ms/pesr_{a.data}_gauss_noise'
                       f'*_seed*.json')
    if a.name is None:
        a.name = f'pesr_{a.data}_robust_two_rulers'

    rows = []
    for p in sorted(glob.glob(a.json_glob)):
        d = json.load(open(p))
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        if not m:
            continue
        mtr = d['metrics']
        rows.append(dict(nz=float(m.group(1)), seed=int(m.group(2)),
                         f1=mtr['f1'], precision=mtr['precision'],
                         recall=mtr['recall']))
    if not rows:
        raise SystemExit(f'no json matched: {a.json_glob}')
    nzs = sorted({r['nz'] for r in rows})

    def stats(key):
        mu = [np.mean([r[key] for r in rows if r['nz'] == nz]) for nz in nzs]
        sd = [np.std([r[key] for r in rows if r['nz'] == nz], ddof=1) for nz in nzs]
        return mu, sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)

    # ---- 左:F1 ----
    mu, sd = stats('f1')
    for r in rows:
        ax1.scatter(100 * r['nz'], r['f1'], s=18, color='#888',
                    zorder=2, alpha=0.85)
    ax1.errorbar([100 * nz for nz in nzs], mu, yerr=sd, marker='o',
                 color='#c0392b', lw=1.8, capsize=4, ms=6, zorder=3,
                 label='mean ± std')
    ax1.axhline(mu[0], color='k', ls='--', lw=0.9, alpha=0.6,
                label='clean mean')
    ax1.set_ylabel('F1')
    ax1.set_title('F1 (point-adjust, oracle threshold)', fontsize=10)
    lo = min(r['f1'] for r in rows)
    hi = max(r['f1'] for r in rows)
    ax1.annotate(f'18-run range = {lo:.4f} – {hi:.4f}\n'
                 f'spread = {hi - lo:.4f} ({(hi / lo - 1) * 100:.1f}%)',
                 xy=(0.03, 0.05), xycoords='axes fraction', fontsize=8,
                 color='#444')

    # ---- 右:Precision / Recall ----
    for key, color, label in (('precision', '#2c7fb8', 'Precision'),
                              ('recall', '#e67e22', 'Recall')):
        mu, sd = stats(key)
        for r in rows:
            ax2.scatter(100 * r['nz'], r[key], s=14, color=color,
                        zorder=2, alpha=0.45)
        ax2.errorbar([100 * nz for nz in nzs], mu, yerr=sd, marker='o',
                     color=color, lw=1.8, capsize=4, ms=6, zorder=3,
                     label=label)
    mp, _ = stats('precision')
    ax2.axhline(mp[0], color='#2c7fb8', ls=':', lw=0.9, alpha=0.6)
    ax2.set_ylabel('Precision / Recall')
    ax2.set_title('Precision & Recall (same runs)', fontsize=10)
    ax2.set_ylim(0.86, 1.02)

    for ax in (ax1, ax2):
        ax.set_xlabel('training noise ratio η (%)')
        ax.grid(alpha=0.25, ls=':')
        ax.legend(frameon=False, fontsize=8)
    ax1.set_ylim(0.92, 0.975)

    fig.suptitle(f'PRD6 - PESR under Gaussian row-replacement noise '
                 f'({a.data}, 6 levels x 3 seeds)', fontsize=10)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'{a.name}.{ext}'), dpi=140,
                    bbox_inches='tight')
    print(f'saved: {a.out_dir}/{a.name}.pdf (+.png)')

    # ---- 终端摘要 ----
    n = len(rows)
    f1s = [r['f1'] for r in rows]
    print(f'  runs={n}  F1 极差 = {max(f1s) - min(f1s):.4f}')
    for key in ('f1', 'precision', 'recall'):
        mu, sd = stats(key)
        print(f'  {key:10s} clean={mu[0]:.4f}  eta50={mu[-1]:.4f}  '
              f'diff={mu[-1] - mu[0]:+.4f}  '
              f'sd_clean={sd[0]:.4f}  between-eta range={max(mu) - min(mu):.4f}')


if __name__ == '__main__':
    main()
