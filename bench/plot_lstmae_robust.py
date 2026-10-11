#!/usr/bin/env python
"""PRD5.2 · 一张图：同一批 run 上，"现有评测协议(F1)" vs "好尺子(AUROC)"。

    左：F1(point-adjust, oracle 阈值) vs η —— 每 seed 一个点 + 均值线
    右：AUROC(未调整) vs η              —— 同格式

两栏用**同一批 18 个 run**，所以左右差异只能归因于指标本身。

用法：python bench/plot_lstmae_robust.py
"""
import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402
import numpy as np                                                    # noqa: E402
from sklearn.metrics import roc_auc_score                             # noqa: E402

OUT = os.path.join('figs', 'ae_mpa')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='results/lstmae_robust/*_scores.npz')
    ap.add_argument('--json-glob', default='results/lstmae_robust/lstmae_*_seed*.json')
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()

    f1 = {}
    for p in glob.glob(a.json_glob):
        d = json.load(open(p))
        f1[(d.get('noise', d['config'].get('noise', 0.0)), d['seed'])] = d['metrics']['f1']
    rows = []
    for p in sorted(glob.glob(a.glob)):
        z = np.load(p)
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        nz, sd = float(m.group(1)), int(m.group(2))
        rows.append(dict(nz=nz, seed=sd,
                         auroc=roc_auc_score(z['test_label'].astype(int), z['test_score']),
                         f1=f1.get((nz, sd), np.nan)))
    nzs = sorted({r['nz'] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)
    for ax, key, name in ((ax1, 'f1', 'F1 (point-adjust, oracle threshold)'),
                          (ax2, 'auroc', 'AUROC (unadjusted)')):
        mu = [np.nanmean([r[key] for r in rows if r['nz'] == nz]) for nz in nzs]
        sd = [np.nanstd([r[key] for r in rows if r['nz'] == nz], ddof=1) for nz in nzs]
        for r in rows:                                   # 每 seed 一个点
            ax.scatter(100 * r['nz'], r[key], s=18, color='#888', zorder=2, alpha=0.85)
        ax.errorbar([100 * nz for nz in nzs], mu, yerr=sd, marker='o', color='#c0392b',
                    lw=1.8, capsize=4, ms=6, zorder=3, label='mean ± std')
        ax.set_xlabel('training noise ratio η (%)')
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25, ls=':')
        ax.legend(frameon=False, fontsize=8)
    ax1.set_ylabel('F1')
    ax2.set_ylabel('AUROC')
    ax1.set_ylim(0.2, 0.85)
    ax2.set_ylim(0.2, 0.85)
    ax2.axhline(0.5, color='k', ls='--', lw=1)
    ax2.text(1, 0.51, 'chance', fontsize=8, color='k')
    fig.suptitle('PRD5.2 - LSTM-AE under script/robust noise protocol (MSL, 6 levels x 3 seeds)',
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'lstmae_robust_two_rulers.{ext}'), dpi=140,
                    bbox_inches='tight')
    print(f"saved: {a.out_dir}/lstmae_robust_two_rulers.pdf (+.png)")
    print(f'  F1    极差 = {max(r["f1"] for r in rows) - min(r["f1"] for r in rows):.4f}')
    print(f'  AUROC 极差 = {max(r["auroc"] for r in rows) - min(r["auroc"] for r in rows):.4f}')


if __name__ == '__main__':
    main()
