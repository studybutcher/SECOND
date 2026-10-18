#!/usr/bin/env python
"""PRD6 · 把 LSTM-AE(PRD5.2)与 PESR(PRD6)画进**同一坐标系**。

    左:F1(point-adjust, oracle 阈值) vs η —— 两个模型,每 seed 一个点 + 均值±std
    右:AUROC(未调整) vs η —— 仅 LSTM-AE(有 --score-dump);
        PESR 本轮没有分数,明确标注 unavailable

两栏 y 轴统一 0.2–1.0、x 轴统一 0–50%,与 `lstmae_robust_two_rulers` 同版式。
⚠️ 两个模型用的**加噪分布不同**(LSTM-AE = mixed U/±5/walk;PESR = 高斯 N(0,1)),
所以只能比较**随 η 的形状**,不能把绝对高低读成模型差异。

用法:python bench/plot_robust_combined.py
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
from sklearn.metrics import roc_auc_score                             # noqa: E402

OUT = os.path.join('figs', 'robust_ms')


def load_lstmae(json_glob, npz_glob):
    rows = []
    for p in sorted(glob.glob(json_glob)):
        d = json.load(open(p))
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        if not m:
            continue
        rows.append(dict(nz=float(m.group(1)), seed=int(m.group(2)),
                         f1=d['metrics']['f1'], auroc=np.nan))
    auc = {}
    for p in glob.glob(npz_glob):
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        if not m:
            continue
        z = np.load(p)
        auc[(float(m.group(1)), int(m.group(2)))] = roc_auc_score(
            z['test_label'].astype(int), z['test_score'])
    for r in rows:
        r['auroc'] = auc.get((r['nz'], r['seed']), np.nan)
    return rows


def load_pesr(json_glob):
    rows = []
    for p in sorted(glob.glob(json_glob)):
        d = json.load(open(p))
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        if not m:
            continue
        rows.append(dict(nz=float(m.group(1)), seed=int(m.group(2)),
                         f1=d['metrics']['f1']))
    return rows


def series(rows, key):
    nzs = sorted({r['nz'] for r in rows})
    mu = [np.nanmean([r[key] for r in rows if r['nz'] == nz]) for nz in nzs]
    sd = [np.nanstd([r[key] for r in rows if r['nz'] == nz], ddof=1)
          for nz in nzs]
    return nzs, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lstmae-json', default='results/lstmae_robust/lstmae_*_seed*.json')
    ap.add_argument('--lstmae-npz', default='results/lstmae_robust/*_scores.npz')
    ap.add_argument('--pesr-json', default='results/robust_ms/pesr_MSL_gauss_noise*_seed*.json')
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()

    lae = load_lstmae(a.lstmae_json, a.lstmae_npz)
    pesr = load_pesr(a.pesr_json)
    if not lae or not pesr:
        raise SystemExit('missing json: check globs')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True)

    # ---- 左:F1,两个模型 ----
    for rows, color, name, mk in (
            (lae, '#2c7fb8', 'LSTM-AE (mixed noise, PRD5.2)', 'o'),
            (pesr, '#c0392b', 'PESR (Gaussian $N(0,1)$, PRD6)', 's')):
        nzs, mu, sd = series(rows, 'f1')
        for r in rows:
            ax1.scatter(100 * r['nz'], r['f1'], s=14, marker=mk,
                        color=color, alpha=0.35, zorder=2)
        ax1.errorbar([100 * n for n in nzs], mu, yerr=sd, marker=mk,
                     color=color, lw=1.8, capsize=4, ms=5, zorder=3,
                     label=name)
    ax1.set_ylabel('F1 (point-adjust, oracle threshold)')
    ax1.set_title('F1 — bad ruler, same 3-seed design', fontsize=10)

    # ---- 右:AUROC,LSTM-AE 有;PESR 无 ----
    nzs, mu, sd = series(lae, 'auroc')
    for r in lae:
        ax2.scatter(100 * r['nz'], r['auroc'], s=14, marker='o',
                    color='#2c7fb8', alpha=0.35, zorder=2)
    ax2.errorbar([100 * n for n in nzs], mu, yerr=sd, marker='o',
                 color='#2c7fb8', lw=1.8, capsize=4, ms=5, zorder=3,
                 label='LSTM-AE (mixed noise, PRD5.2)')
    ax2.axhline(0.5, color='k', ls='--', lw=1)
    ax2.text(1, 0.51, 'chance', fontsize=8, color='k')
    ax2.text(0.52, 0.42,
             'PESR: AUROC unavailable\n(no --score-dump in PRD6)',
             transform=ax2.transAxes, fontsize=8, color='#c0392b',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#c0392b',
                       alpha=0.8))
    ax2.set_ylabel('AUROC (unadjusted)')
    ax2.set_title('AUROC — good ruler (LSTM-AE only)', fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel('training noise ratio η (%)')
        ax.set_xlim(-3, 53)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.25, ls=':')
        ax.legend(frameon=False, fontsize=8, loc='lower left')

    fig.suptitle('Two rulers, two models — MSL, 6 noise levels x 3 seeds\n'
                 'note: different noise distributions (mixed vs Gaussian); '
                 'compare shapes, not absolute levels', fontsize=9.5)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'robust_combined_two_rulers.{ext}'),
                    dpi=140, bbox_inches='tight')
    print(f'saved: {a.out_dir}/robust_combined_two_rulers.pdf (+.png)')

    for name, rows in (('LSTM-AE', lae), ('PESR', pesr)):
        nzs, mu, sd = series(rows, 'f1')
        print(f'  {name:8s} F1  mean range = {min(mu):.4f} – {max(mu):.4f}  '
              f'(spread {max(mu) - min(mu):.4f})  max seed-sd = {max(sd):.4f}')
    nzs, mu, sd = series(lae, 'auroc')
    print(f'  LSTM-AE  AUROC mean range = {min(mu):.4f} – {max(mu):.4f}')


if __name__ == '__main__':
    main()
