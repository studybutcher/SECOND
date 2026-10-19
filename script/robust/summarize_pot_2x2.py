#!/usr/bin/env python
"""PRD6.1 · 汇总 MSL 上 2 模型 × 2 阈值口径的结果。

数据来源
----------------
* `results/threshold_ms/SECOND_MSL_fixed_<model>_MSL_noise<nz>_seed<s>.json`
  —— 本次对已有 ckpt 的复评，**同一次前向**同时给出 bf 与 pot_dtaad。
* `results/robust_ms/pesr_MSL_gauss_noise<nz>_seed<s>.json`
  `results/lstmae_robust/lstmae_MSL_noise<nz>_seed<s>.json`
  —— 现有（PRD6 / PRD5.2）的 bf 结果，用作主表的 bf 列与一致性校验。

产出
----------------
* `results/threshold_ms/summary_pot_2x2.md`
* `results/threshold_ms/summary_pot_2x2.csv`（逐 seed 长表）
* `results/threshold_ms/table_pot_2x2.tex`

用法：python script/robust/summarize_pot_2x2.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

OUT_DIR = 'results/threshold_ms'
DS = 'MSL'

MODEL_LABEL = {'lstmae': 'LSTM-AE', 'pesr': 'PESR'}
MODEL_STORED = {
    # LSTM-AE 已按 PRD6.1 D5 换成 gauss 重训（旧 mixed 结果保留但不进本表）
    'lstmae': 'results/lstmae_robust_gauss/lstmae_{ds}_gauss_noise{nz}_seed{s}.json',
    'pesr':   'results/robust_ms/pesr_{ds}_gauss_noise{nz}_seed{s}.json',
}


def parse_tag(path):
    m = re.search(r'(lstmae|pesr)_MSL_noise([\d.]+)_seed(\d+)\.json$', os.path.basename(path))
    if not m:
        return None
    return m.group(1), float(m.group(2)), int(m.group(3))


def load_rows():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, f'SECOND_{DS}_fixed_*_seed*.json'))):
        tag = parse_tag(p)
        if tag is None:
            continue
        model, nz, seed = tag
        d = json.load(open(p))
        pot = d['pot']['results'].get('pot_dtaad', {})
        bf = d['bf_search_oracle']
        stored_path = MODEL_STORED[model].format(ds=DS, nz=('' if False else _fmt(nz)), s=seed)
        stored = None
        if os.path.exists(stored_path):
            stored = json.load(open(stored_path))['metrics']
        rows.append(dict(
            model=model, eta=nz, seed=seed,
            bf_stored=stored['f1'] if stored else np.nan,
            bf_reeval=bf['f1'],
            pot=pot.get('f1', np.nan),
            pot_p=pot.get('precision', np.nan),
            pot_r=pot.get('recall', np.nan),
            pot_th=pot.get('threshold', np.nan),
            bf_th_pct=bf.get('threshold_pct', np.nan),
        ))
    return rows


_NZ_STR = {0.0: '0', 0.05: '0.05', 0.1: '0.10', 0.15: '0.15',
           0.2: '0.2', 0.5: '0.5'}


def _fmt(nz):
    """与磁盘文件名一致：0 / 0.05 / 0.10 / 0.15 / 0.2 / 0.5。

    注意 `f'{0.1:g}'` 会得到 '0.1'，而脚本生成的文件名是 '0.10'，
    故用显式映射，不能用格式化默认值。
    """
    return _NZ_STR.get(round(float(nz), 4), f'{nz:g}')


def agg(rows, model, eta, key):
    v = [r[key] for r in rows if r['model'] == model and r['eta'] == eta
         and np.isfinite(r[key])]
    if not v:
        return np.nan, np.nan, 0
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit('no POT re-eval json found under results/threshold_ms')

    etas = sorted({r['eta'] for r in rows})
    models = [m for m in ('lstmae', 'pesr') if any(r['model'] == m for r in rows)]

    lines = []
    lines.append(f'# PRD6.1 · {DS}：2 模型 × 2 阈值口径（mean ± sd，n=3 seeds）\n')
    lines.append('阈值口径：**bf** = `bf_search` oracle 百分位（现有口径）；'
                 '**POT** = `pot_dtaad`，`--pot-fixed`（q=1e-3, level=0.98, scale=1.0）。\n')
    lines.append('| η | ' + ' | '.join(
        f'{MODEL_LABEL[m]} {k}' for m in models for k in ('bf', 'POT')) + ' |')
    lines.append('|---|' + '---|' * (2 * len(models)))
    for eta in etas:
        cells = []
        for m in models:
            for key in ('bf_stored', 'pot'):
                mu, sd, n = agg(rows, m, eta, key)
                cells.append('—' if n == 0 else f'{mu:.4f} ± {sd:.4f}')
        lines.append(f'| {eta:g} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    # 覆盖度
    lines.append('## 覆盖度\n')
    for m in models:
        for key in ('bf_stored', 'pot'):
            n = sum(1 for r in rows if r['model'] == m and np.isfinite(r[key]))
            lines.append(f'- {MODEL_LABEL[m]} / {key}: {n} runs')
    lines.append('')

    # 一致性：复评 bf vs 存档 bf（dropout 开着，两次前向不同）
    lines.append('## 一致性校验：复评 bf vs 存档 bf\n')
    lines.append('| 模型 | 平均 \\|Δ\\| | 最大 \\|Δ\\| | 说明 |')
    lines.append('|---|---|---|---|')
    for m in models:
        d = np.abs([r['bf_reeval'] - r['bf_stored'] for r in rows
                    if r['model'] == m and np.isfinite(r['bf_stored'])])
        lines.append(f'| {MODEL_LABEL[m]} | {d.mean():.4f} | {d.max():.4f} | '
                     f'dropout 开启，分数不可逐位复现（PRD3.1 F6） |')
    lines.append('')

    # POT 的 P/R
    lines.append('## POT 的 Precision / Recall（mean，n=3）\n')
    lines.append('| η | ' + ' | '.join(f'{MODEL_LABEL[m]} P | {MODEL_LABEL[m]} R'
                                       for m in models) + ' |')
    lines.append('|---|' + '---|' * (2 * len(models)))
    for eta in etas:
        cells = []
        for m in models:
            p, _, _ = agg(rows, m, eta, 'pot_p')
            r, _, _ = agg(rows, m, eta, 'pot_r')
            cells += [f'{p:.4f}', f'{r:.4f}']
        lines.append(f'| {eta:g} | ' + ' | '.join(cells) + ' |')
    lines.append('')

    md = '\n'.join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'summary_pot_2x2.md'), 'w') as f:
        f.write(md)
    print(md)

    # 逐 seed CSV
    with open(os.path.join(OUT_DIR, 'summary_pot_2x2.csv'), 'w') as f:
        cols = ['model', 'eta', 'seed', 'bf_stored', 'bf_reeval', 'pot',
                'pot_p', 'pot_r', 'pot_th', 'bf_th_pct']
        f.write(','.join(cols) + '\n')
        for r in sorted(rows, key=lambda x: (x['model'], x['eta'], x['seed'])):
            f.write(','.join(str(r[c]) for c in cols) + '\n')
    print(f'wrote {OUT_DIR}/summary_pot_2x2.csv')

    # LaTeX（booktabs，均值±sd）
    tex = []
    tex.append('\\begin{table}[t]\\centering')
    tex.append('\\caption{MSL: two models under oracle vs.\\ POT threshold '
               '(mean$\\pm$std over 3 seeds).}')
    tex.append('\\label{tab:pot-2x2}')
    tex.append('\\begin{tabular}{c' + 'c' * (2 * len(models)) + '}')
    tex.append('\\toprule')
    head = ['$\\eta$']
    for m in models:
        head += [f'{MODEL_LABEL[m]} bf', f'{MODEL_LABEL[m]} POT']
    tex.append(' & '.join(head) + ' \\\\')
    tex.append('\\midrule')
    for eta in etas:
        cells = [f'{eta:g}']
        for m in models:
            for key in ('bf_stored', 'pot'):
                mu, sd, n = agg(rows, m, eta, key)
                cells.append('--' if n == 0 else f'{mu:.4f}$\\pm${sd:.4f}')
        tex.append(' & '.join(cells) + ' \\\\')
    tex.append('\\bottomrule')
    tex.append('\\end{tabular}')
    tex.append('\\end{table}')
    with open(os.path.join(OUT_DIR, 'table_pot_2x2.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f'wrote {OUT_DIR}/table_pot_2x2.tex')


if __name__ == '__main__':
    main()
