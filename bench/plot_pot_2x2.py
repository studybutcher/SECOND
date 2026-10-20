#!/usr/bin/env python
"""PRD6.1 · MSL：2 模型 × 2 阈值口径的合并图。

  * 左：bf（oracle 百分位搜索，现有口径；取存档结果）
  * 右：POT（pot_dtaad，--pot-fixed：q=1e-3, level=0.98, scale=1.0）

每格：3 个 seed 的散点 + mean±sd。两栏共享横轴与纵轴。

代码结构、配色、轴范围与 `bench/plot_robust_combined.py` 一致。

用法：python bench/plot_pot_2x2.py
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
MODEL_LABEL = {'lstmae': 'LSTM-AE (Gaussian $N(0,1)$)',
               'pesr': 'PESR (Gaussian $N(0,1)$)'}
MODEL_STYLE = {'lstmae': ('#2c7fb8', 'o'), 'pesr': ('#c0392b', 's')}
STORED = {
    'lstmae': 'results/lstmae_robust_gauss/lstmae_MSL_gauss_noise{nz}_seed{s}.json',
    'pesr':   'results/robust_ms/pesr_MSL_gauss_noise{nz}_seed{s}.json',
}
_NZ_STR = {0.0: '0', 0.05: '0.05', 0.1: '0.10', 0.15: '0.15',
           0.2: '0.2', 0.5: '0.5'}


def load(pot_glob):
    rows = []
    for p in sorted(glob.glob(pot_glob)):
        m = re.search(r'(lstmae|pesr)_MSL_noise([\d.]+)_seed(\d+)\.json$', os.path.basename(p))
        if not m:
            continue
        model, nz, seed = m.group(1), float(m.group(2)), int(m.group(3))
        d = json.load(open(p))
        bf = d['bf_search_oracle']['f1']
        pot_d = d['pot']['results'].get('pot_dtaad', {})
        sp = STORED[model].format(
            nz=_NZ_STR.get(round(nz, 4), f'{nz:g}'), s=seed)
        bf_stored = json.load(open(sp))['metrics']['f1'] if os.path.exists(sp) else np.nan
        rows.append(dict(model=model, nz=nz, seed=seed, bf=bf_stored,
                         bf_reeval=bf, pot=pot_d.get('f1', np.nan)))
    return rows


def series(rows, model, key):
    nzs = sorted({r['nz'] for r in rows if r['model'] == model})
    mu = [np.nanmean([r[key] for r in rows if r['model'] == model and r['nz'] == nz]) for nz in nzs]
    sd = [np.nanstd([r[key] for r in rows if r['model'] == model and r['nz'] == nz], ddof=1)
          for nz in nzs]
    return nzs, mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pot-json', default='results/threshold_ms/SECOND_MSL_fixed_*_seed*.json')
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()

    rows = load(a.pot_json)
    if not rows:
        raise SystemExit('no POT re-eval json found')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
    panels = (('bf', 'bf (oracle percentile)', 'F1 — oracle threshold'),
              ('pot', 'POT (fixed: $q$=1e-3, level=0.98, scale=1)',
               'F1 — label-free POT threshold'))

    for ax, (key, ylab, title) in zip(axes, panels):
        for model in ('lstmae', 'pesr'):
            color, mk = MODEL_STYLE[model]
            sub = [r for r in rows if r['model'] == model]
            nzs, mu, sd = series(rows, model, key)
            ax.scatter([100 * r['nz'] for r in sub], [r[key] for r in sub],
                       s=14, marker=mk, color=color, alpha=0.35, zorder=2)
            ax.errorbar([100 * n for n in nzs], mu, yerr=sd, marker=mk,
                        color=color, lw=1.8, capsize=4, ms=5, zorder=3,
                        label=MODEL_LABEL[model])
        ax.set_xlabel('training noise ratio η (%)')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3, 53)
        # POT 可能把某些模型压到 0.05 以下，统一放到 0–1，避免截断
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25, ls=':')
        ax.legend(frameon=False, fontsize=8, loc='lower left')

    fig.suptitle('MSL, 6 noise levels × 3 seeds — same checkpoints, two threshold rules\n'
                 'left: oracle search uses test labels; right: POT never sees labels',
                 fontsize=9.5)
    fig.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out_dir, f'pot_2models_2thresholds.{ext}'),
                    dpi=140, bbox_inches='tight')
    print(f'saved: {a.out_dir}/pot_2models_2thresholds.pdf (+.png)')

    for model in ('lstmae', 'pesr'):
        for key in ('bf', 'pot'):
            nzs, mu, sd = series(rows, model, key)
            print(f'  {model:7s} {key:3s}: ' +
                  ' '.join(f'{100*n:.0f}%:{m:.4f}±{s:.4f}' for n, m, s in zip(nzs, mu, sd)))
    d = np.abs([r['bf_reeval'] - r['bf'] for r in rows if np.isfinite(r['bf'])])
    print(f'  bf re-eval vs stored: mean|Δ|={d.mean():.4f} max|Δ|={d.max():.4f}')


if __name__ == '__main__':
    main()
