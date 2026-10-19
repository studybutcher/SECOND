#!/usr/bin/env python
"""PRD6 · 主模型 PESR 在四个数据集上的高斯噪声多 seed 结果(2x2 小图矩阵)。

每格 = 一个数据集,F1(point-adjust, oracle 阈值) vs η;
每 seed 一个散点 + 均值±std,灰色虚线为 η=0 的均值。

坐标轴与 `lstmae_robust_two_rulers` / `robust_combined_two_rulers` 统一:
x ∈ [-3, 53] %,y ∈ [0.2, 1.0],便于并排阅读。

⚠️ 与 LSTM-AE 的图只比**形状**:加噪分布不同(mixed vs 高斯 N(0,1)),
绝对高低不可解释为模型差异。F1 本身是 point-adjust 的产物(PRD5.2 §2.3)。

用法:python bench/plot_pesr_robust_all.py [--datasets MSL SMAP SWaT WADI]
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
COLOR = {'MSL': '#c0392b', 'SMAP': '#e67e22',
         'SWaT': '#27ae60', 'WADI': '#8e44ad'}


def load(ds, res_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(
            res_dir, f'pesr_{ds}_gauss_noise*_seed*.json'))):
        d = json.load(open(p))
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        if not m:
            continue
        rows.append(dict(nz=float(m.group(1)), seed=int(m.group(2)),
                         f1=d['metrics']['f1'],
                         precision=d['metrics'].get('precision'),
                         recall=d['metrics'].get('recall')))
    return rows


def series(rows, key):
    nzs = sorted({r['nz'] for r in rows})
    mu = [np.nanmean([r[key] for r in rows if r['nz'] == nz]) for nz in nzs]
    sd = [np.nanstd([r[key] for r in rows if r['nz'] == nz], ddof=1)
          if sum(1 for r in rows if r['nz'] == nz) > 1 else 0.0
          for nz in nzs]
    return nzs, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['MSL', 'SMAP', 'SWaT', 'WADI'])
    ap.add_argument('--res-dir', default='results/robust_ms')
    ap.add_argument('--out-dir', default=OUT)
    ap.add_argument('--name', default='pesr_robust_all_datasets')
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True)
    axes = axes.ravel()
    lines = []
    for ax, ds in zip(axes, a.datasets):
        rows = load(ds, a.res_dir)
        if not rows:
            ax.text(0.5, 0.5, f'{ds}: no results yet', ha='center',
                    va='center', transform=ax.transAxes, color='gray')
            ax.set_title(ds, fontsize=11)
            lines.append(f'  {ds:5s} —')
            continue
        nzs, mu, sd = series(rows, 'f1')
        c = COLOR.get(ds, '#333333')
        for r in rows:
            ax.scatter(100 * r['nz'], r['f1'], s=13, marker='s',
                       color=c, alpha=0.35, zorder=2)
        ax.errorbar([100 * n for n in nzs], mu, yerr=sd, marker='s',
                    color=c, lw=1.8, capsize=3.5, ms=5, zorder=3)
        clean = mu[nzs.index(0.0)] if 0.0 in nzs else None
        if clean is not None:
            ax.axhline(clean, color='gray', ls='--', lw=1, zorder=1)
        ax.set_title(ds, fontsize=11)
        ax.grid(alpha=0.25, ls=':')
        n_by = {nz: sum(1 for r in rows if r['nz'] == nz) for nz in nzs}
        ntxt = ','.join(str(n_by[nz]) for nz in nzs)
        ax.text(0.03, 0.05, f'n per η: {ntxt}', transform=ax.transAxes,
                fontsize=7.5, color='0.35')
        lines.append(f'  {ds:5s} F1 mean range {min(mu):.4f}–{max(mu):.4f} '
                     f'(spread {max(mu) - min(mu):.4f}), max seed-sd {max(sd):.4f}')

    for ax in axes[len(a.datasets):]:
        ax.axis('off')
    for ax in axes[-2:]:
        ax.set_xlabel('training noise ratio η (%)')
    for ax in (axes[0], axes[2]):
        ax.set_ylabel('F1 (point-adjust, oracle threshold)')
    for ax in axes:
        ax.set_xlim(-3, 53)
        ax.set_ylim(0.2, 1.0)

    fig.suptitle('PESR, Gaussian $N(0,1)$ whole-row replacement — '
                 '6 noise levels x 3 seeds\n'
                 'F1 is a point-adjust artifact (see PRD5.2 §2.3): '
                 'a flat line is not evidence of robustness', fontsize=9.5)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, a.name)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{base}.{ext}', dpi=140, bbox_inches='tight')
    print(f'saved: {base}.pdf (+.png)')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
