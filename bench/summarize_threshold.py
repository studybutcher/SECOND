"""汇总阈值选择（bf_search vs POT）结果

读取 results/threshold/<model>_<dataset>_<paramset>.json，产出
    results/threshold/summary.md
    results/threshold/summary.csv

用法：
    python bench/summarize_threshold.py
"""
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(REPO_ROOT, 'results', 'threshold')
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']
POT_MODES = ['pot_dtaad', 'pot_evt', 'pot_dynamic']


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(IN_DIR, '*.json'))):
        try:
            with open(p) as f:
                out.append(json.load(f))
        except Exception:
            continue
    return out


def f(x, n=4):
    return '-' if x is None else f'{x:.{n}f}'


def main():
    recs = load()
    if not recs:
        print('没有找到阈值结果，请先跑 script/addition/threshold_eval.sh')
        return

    md = ['# 阈值选择结果：bf_search (oracle) vs POT', '',
          '> `bf_search` 在测试集标签上搜最优阈值（oracle，不可部署）；',
          '> POT 系列只由分数分布决定（标签无关）。',
          '> 三个 POT 口径共用同一次 `initialize()`，只是取不同的量当阈值。', '']

    # ---------- 表 1：主结果 ----------
    md += ['## 表 1  各数据集 F1（同一 ckpt）', '',
           '| 数据集 | 参数集 | bf_search (oracle) | pot_dtaad | pot_evt | pot_dynamic |',
           '|---|---|---|---|---|---|']
    for d in DATASETS:
        rows = [r for r in recs if r.get('dataset') == d]
        if not rows:
            md.append(f'| {d} | - | - | - | - | - |')
            continue
        for r in sorted(rows, key=lambda x: x['pot']['paramset']):
            bf = r['bf_search_oracle']['f1']
            cells = [d, r['pot']['paramset'], f'**{f(bf)}**']
            for m in POT_MODES:
                v = r['pot']['results'].get(m)
                cells.append(f(v['f1']) if v else '-')
            md.append('| ' + ' | '.join(cells) + ' |')
    md.append('')

    # ---------- 表 2：POT 阈值与 oracle 阈值对比 ----------
    md += ['## 表 2  阈值取值', '',
           '| 数据集 | 参数集 | (q, level, scale) | init_threshold | 真EVT | pot_dtaad 阈值 | pot_evt 阈值 | pot_dynamic 阈值 |',
           '|---|---|---|---|---|---|---|---|']
    for d in DATASETS:
        for r in sorted([x for x in recs if x.get('dataset') == d],
                        key=lambda x: x['pot']['paramset']):
            p = r['pot']
            cells = [d, p['paramset'], f"({p['q']}, {f(p['level'],3)}, {f(p['scale'],2)})",
                     f(p.get('init_threshold'), 4), f(p.get('extreme_quantile_raw'), 4)]
            for m in POT_MODES:
                v = p['results'].get(m)
                cells.append(f(v['threshold'], 4) if v else '-')
            md.append('| ' + ' | '.join(cells) + ' |')
    md.append('')

    # ---------- 表 3：Precision / Recall ----------
    md += ['## 表 3  Precision / Recall（同一 ckpt）', '',
           '| 数据集 | 方法 | F1 | P | R |', '|---|---|---|---|---|']
    for d in DATASETS:
        for r in sorted([x for x in recs if x.get('dataset') == d],
                        key=lambda x: x['pot']['paramset']):
            bf = r['bf_search_oracle']
            md.append(f"| {d} | bf_search ({r['pot']['paramset']}) | {f(bf['f1'])} "
                      f"| {f(bf['precision'])} | {f(bf['recall'])} |")
            for m in POT_MODES:
                v = r['pot']['results'].get(m)
                if v:
                    md.append(f"| {d} | {m} ({r['pot']['paramset']}) | {f(v['f1'])} "
                              f"| {f(v['precision'])} | {f(v['recall'])} |")
    md.append('')

    # ---------- 表 4：测量元数据 ----------
    md += ['## 表 4  测量元数据', '',
           '| 数据集 | 参数集 | init_score n | fit_ok | 隐层 W | ckpt |',
           '|---|---|---|---|---|---|']
    for d in DATASETS:
        for r in sorted([x for x in recs if x.get('dataset') == d],
                        key=lambda x: x['pot']['paramset']):
            p, c = r['pot'], r.get('config', {})
            md.append(f"| {d} | {p['paramset']} | {p['init_score']['n']} | {p['fit_ok']} "
                      f"| {c.get('window_length','-')} | {r.get('ckpt',{}).get('path','-')} |")
    md.append('')

    # ---------- 一致性检查 ----------
    miss = [f'{d}/{ps}' for d in DATASETS for ps in ['fixed', 'dtaad']
            if not any(r.get('dataset') == d and r['pot']['paramset'] == ps for r in recs)]
    if miss:
        md += ['## ⚠️ 缺失的组合', ''] + [f'- {x}' for x in miss] + ['']
    else:
        md += ['> ✅ 4 数据集 × 2 参数集 齐全。', '']

    with open(os.path.join(IN_DIR, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print('written:', os.path.join(IN_DIR, 'summary.md'))

    lines = ['dataset,paramset,q,level,scale,f1_bf_oracle,f1_pot_dtaad,f1_pot_evt,'
             'f1_pot_dynamic,init_threshold,evt_threshold,fit_ok']
    for d in DATASETS:
        for r in sorted([x for x in recs if x.get('dataset') == d],
                        key=lambda x: x['pot']['paramset']):
            p = r['pot']
            g = lambda m, k: (p['results'][m][k] if m in p['results'] else '')
            lines.append(','.join(str(x) for x in [
                d, p['paramset'], p['q'], p['level'], p['scale'],
                round(r['bf_search_oracle']['f1'], 6),
                g('pot_dtaad', 'f1'), g('pot_evt', 'f1'), g('pot_dynamic', 'f1'),
                p.get('init_threshold', ''), p.get('extreme_quantile_raw', ''), p['fit_ok']]))
    with open(os.path.join(IN_DIR, 'summary.csv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('written:', os.path.join(IN_DIR, 'summary.csv'))


if __name__ == '__main__':
    main()
