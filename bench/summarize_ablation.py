"""汇总中间块消融结果

读取 results/ablation/<backbone>_<dataset>_seed<seed>.json，产出
    results/ablation/summary.md
    results/ablation/summary.csv

用法：
    python bench/summarize_ablation.py
"""
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(REPO_ROOT, 'results', 'ablation')
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']
BACKBONES = ['transformer', 'lstm']   # 本次消融的两个结构（mamba 参照由用户自有数据提供）

# 噪声底：单次运行的 F1 波动幅度。用 mamba/SMAP 的两个同协议样本估计：
#   历史 run（casestudy补充, 无种子）F1=0.9751  vs  本次 run（seed=42）F1=0.9703  → 差 0.0048
# 两者的数据切分逐位相同（test_data (277952,25) / val_data (149665,25)）。
# 差值绝对值小于该阈值的，视为"单 seed 下不可区分"，不应解读为谁更好。
NOISE_FLOOR = 0.005


def load():
    recs = {}
    for path in sorted(glob.glob(os.path.join(IN_DIR, '*.json'))):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue
        if 'backbone' in r and 'metrics' in r:
            recs[(r['backbone'], r['dataset'])] = r
    return recs


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def f(x, n=4):
    return '-' if x is None else f'{x:.{n}f}'


def main():
    recs = load()
    if not recs:
        print('没有找到消融结果，请先跑 script/addition/ablation.sh')
        return

    n_rep = sum(1 for k in recs if k[0] in BACKBONES)
    md = ['# 中间块消融结果', '',
          f'共 {n_rep} 条记录（{len(BACKBONES)} 个结构 × {len(DATASETS)} 个数据集）。'
          f'变体定义与生成的 diff 见 `model/ABLATION_MANIFEST.md`。', '',
          '消融对象：`self.attmamba` 的 8 层中间块（其余结构一字不改）。', '',
          '对照的 mamba 基线数据由另处提供，不在本汇总内。', '']

    # ---------- 表 1：主结果 ----------
    md += ['## 表 1  主结果（F1 / 参数量）', '',
           '| 数据集 | 变体 | F1 | Precision | Recall | threshold | params(M) | 中间块(M) | 中间块占比 |',
           '|---|---|---|---|---|---|---|---|---|']
    for d in DATASETS:
        for bb in BACKBONES:
            r = recs.get((bb, d))
            if not r:
                md.append(f'| {d} | {bb} | - | - | - | - | - | - | - |')
                continue
            m, p = r['metrics'], r.get('params', {})
            md.append(f"| {d} | {bb} | **{f(m['f1'])}** | {f(m['precision'])} | {f(m['recall'])} "
                      f"| {f(m['threshold'],2)} | {f(p.get('n_params_M'),2)} | "
                      f"{f(p.get('n_midstack_M'),2)} | {f(p.get('midstack_share'),4)} |")
    md.append('')

    # ---------- 表 2：两个变体之间的差异 ----------
    md += ['## 表 2  transformer 与 lstm 的差异', '',
           f'> 差值绝对值 ≤ {NOISE_FLOOR:.3f} 视为不可区分（标记 **≈**）。'
           f'噪声底来源见脚本内注释。', '',
           '| 数据集 | transformer | lstm | Δ (lstm − trans) |',
           '|---|---|---|---|']
    for d in DATASETS:
        a = recs.get(('transformer', d))
        b = recs.get(('lstm', d))
        fa = a['metrics']['f1'] if a else None
        fb = b['metrics']['f1'] if b else None
        if fa is not None and fb is not None:
            dlt = fb - fa
            mark = ' **≈**' if abs(dlt) <= NOISE_FLOOR + 1e-9 else ''
            cell = f'{dlt:+.4f}{mark}'
        else:
            cell = '-'
        md.append(f'| {d} | {f(fa)} | {f(fb)} | {cell} |')
    md.append('')

    # ---------- 表 3：参数量对比 ----------
    md += ['## 表 3  参数量对比（M）', '',
           '| 数据集 | transformer | lstm |', '|---|---|---|']
    for d in DATASETS:
        cells = [d]
        for bb in BACKBONES:
            r = recs.get((bb, d))
            cells.append(f(r['params']['n_params_M'], 2) if r and 'params' in r else '-')
        md.append('| ' + ' | '.join(cells) + ' |')
    md.append('')

    # ---------- 表 4：运行配置 ----------
    md += ['## 表 4  运行配置', '',
           '| 数据集 | 变体 | seed | C | W | latent | batch | 时间戳 |',
           '|---|---|---|---|---|---|---|---|']
    for d in DATASETS:
        for bb in BACKBONES:
            r = recs.get((bb, d))
            if not r:
                continue
            c = r['config']
            md.append(f"| {d} | {bb} | {r.get('seed','-')} | {c['input_dim']} | {c['window_length']} "
                      f"| {c['latent_size']} | {c['batch_size']} | {r.get('timestamp','-')} |")
    md.append('')

    # ---------- 一致性检查 ----------
    miss = [f'{d}/{bb}' for d in DATASETS for bb in BACKBONES if (bb, d) not in recs]
    if miss:
        md += ['## ⚠️ 缺失的运行', ''] + [f'- {x}' for x in miss] + ['']
    else:
        md += [f'> ✅ 2 个结构 × 4 数据集 共 8 组结果齐全。', '']

    with open(os.path.join(IN_DIR, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    print('written:', os.path.join(IN_DIR, 'summary.md'))

    lines = ['dataset,backbone,seed,f1,precision,recall,threshold,n_params_M,n_midstack_M,midstack_share']
    for d in DATASETS:
        for bb in BACKBONES:
            r = recs.get((bb, d))
            if not r:
                continue
            m, p = r['metrics'], r.get('params', {})
            lines.append(','.join(str(x) for x in [
                d, bb, r.get('seed', ''), round(m['f1'], 6), round(m['precision'], 6),
                round(m['recall'], 6), round(m['threshold'], 4),
                round(p.get('n_params_M', 0), 4), round(p.get('n_midstack_M', 0), 4),
                round(p.get('midstack_share', 0), 6)]))
    with open(os.path.join(IN_DIR, 'summary.csv'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('written:', os.path.join(IN_DIR, 'summary.csv'))


if __name__ == '__main__':
    main()
