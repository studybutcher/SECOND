"""两库阈值结果并排：SECOND vs MUTANT，同一套 POT 口径。

前提：两库的 `pot_eval` 逐字节相同（SHA256 见 PRD3.1 验收项 3），
且分数已由各自的 `--dump-scores` / `cache_scores.py` 落盘为同 schema 的 npz：
    SECOND/results/threshold/scores/SECOND_<dataset>.npz
    MUTANT/results/threshold/scores/MUTANT_<dataset>.npz
    （键：init_score, test_score, test_label）

产出：
    results/threshold/two_libs.md
    results/threshold/two_libs.json

用法：
    python bench/compare_two_libs.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from sweep_threshold import f1_at, adjust_fast          # noqa: E402

MUTANT_REPO = os.path.join(os.path.dirname(REPO), 'MUTANT')
SCORE_DIR = {
    'SECOND': os.path.join(REPO, 'results', 'threshold', 'scores'),
    'MUTANT': os.path.join(MUTANT_REPO, 'results', 'threshold', 'scores'),
}
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']

# 全数据集统一的 label-free 对照参数（不做任何调参）
FIXED = (0.98, 1.0)
# 主口径 q（对齐 DTAAD 公开代码）
Q = 1e-5

LEVELS = [0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.997, 0.998, 0.999]
SCALES = [0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]


def bf_search_fast(score, label, step_num=700):
    """与 `eval_method.bf_search` **等价但向量化**的版本。

    原版对每个百分位点调用 `calc_seq` → `adjust_predicts`，后者是**逐点 Python 循环**；
    SMAP 有 277932 个点 × 700 步 = 1.9 亿次迭代，慢到不可用。
    这里搜索**同一组**百分位点（93.00 → 99.99，步长 0.01），但 point-adjust 用 `adjust_fast`。
    两者的 `score > np.percentile(score, p)` 语义一致，结果逐位相同。
    """
    best = (-1.0, -1.0, -1.0)
    best_p = 0.0
    for i in range(step_num):
        p = 93 + 0.01 * i
        th = float(np.percentile(score, p))
        f, pr, rc = f1_at(score, label, th)
        if f > best[0]:
            best, best_p = (f, pr, rc), p
    return best, best_p


def oracle(score, label):
    """绝对阈值上限（bf_search 的语义：在测试分数上搜最优绝对阈值）。"""
    m, th = bf_search_fast(score, label, step_num=700)
    return float(m[0]), float(th), float(m[1]), float(m[2])


def pot_at(init, score, label, level, scale):
    th = float(np.quantile(init, level)) * scale
    f, p, r = f1_at(score, label, th)
    return float(f), float(p), float(r), th


def plateau(score, label, lvl=0.95, n=1200):
    lo, hi = float(score.min()), float(score.max())
    ths = np.linspace(lo, hi, n)
    cur = np.array([f1_at(score, label, float(t))[0] for t in ths])
    sel = ths[cur >= lvl]
    if sel.size == 0:
        return None
    return {'th_lo': float(sel.min()), 'th_hi': float(sel.max()),
            'width': float(sel.max() - sel.min()),
            'pct': 100 * float(sel.max() - sel.min()) / (hi - lo)}


def analyse(lib, ds):
    p = os.path.join(SCORE_DIR[lib], f'{lib}_{ds}.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p)
    init = d['init_score'].astype(float).ravel()
    score = d['test_score'].astype(float).ravel()
    label = d['test_label'].astype(float).ravel()

    o_f1, o_th, o_p, o_r = oracle(score, label)
    fx = pot_at(init, score, label, *FIXED)

    grid = []
    for lv in LEVELS:
        for sc in SCALES:
            f, pr, rc, th = pot_at(init, score, label, lv, sc)
            grid.append({'level': lv, 'scale': sc, 'threshold': th,
                         'f1': f, 'precision': pr, 'recall': rc})
    best = max(grid, key=lambda g: g['f1'])

    return {'init_n': int(init.size), 'test_n': int(score.size),
            'oracle': {'f1': o_f1, 'threshold': o_th, 'precision': o_p, 'recall': o_r},
            'fixed': {'level': FIXED[0], 'scale': FIXED[1], 'f1': fx[0],
                      'precision': fx[1], 'recall': fx[2], 'threshold': fx[3]},
            'calibrated': best,
            'plateau_f1_90': plateau(score, label, 0.90),
            'plateau_f1_95': plateau(score, label, 0.95)}


def main():
    res = {}
    missing = []
    for ds in DATASETS:
        res[ds] = {}
        for lib in ('SECOND', 'MUTANT'):
            r = analyse(lib, ds)
            if r is None:
                missing.append(f'{lib}/{ds}')
            res[ds][lib] = r

    # ---------- 主表 ----------
    md = ['# 两库对照：同一套 POT 口径下的阈值结果', '',
          f'> `q={Q}`、`dynamic=False`、`init_score` = 全训练集、POT 实现两库逐字节相同',
          '> 固定参数列 = 全数据集统一 `(level, scale) = (0.98, 1.0)`，**不做任何调参**', '',
          '## 表 1  主结果', '',
          '| 数据集 | 库 | ① 固定无调参 | ② 标定最优 | ③ oracle | ②−① |',
          '|---|---|---|---|---|---|']
    for ds in DATASETS:
        for lib in ('SECOND', 'MUTANT'):
            r = res[ds][lib]
            if r is None:
                md.append(f'| {ds} | {lib} | - | - | - | - |')
                continue
            md.append(f"| {ds} | {lib} | {r['fixed']['f1']:.4f} | "
                      f"**{r['calibrated']['f1']:.4f}** | {r['oracle']['f1']:.4f} | "
                      f"{r['calibrated']['f1'] - r['fixed']['f1']:+.4f} |")
    md += ['']

    # ---------- 标定参数与阈值 ----------
    md += ['## 表 2  标定参数与阈值', '',
           '| 数据集 | 库 | 标定 (level, scale) | 标定阈值 | ② P | ② R |',
           '|---|---|---|---|---|---|']
    for ds in DATASETS:
        for lib in ('SECOND', 'MUTANT'):
            r = res[ds][lib]
            if r is None:
                continue
            c = r['calibrated']
            md.append(f"| {ds} | {lib} | ({c['level']}, {c['scale']}) | "
                      f"{c['threshold']:.4f} | {c['precision']:.4f} | {c['recall']:.4f} |")
    md += ['']

    # ---------- 可用阈值带宽度 ----------
    md += ['## 表 3  可用阈值带宽度（标定稳不稳）', '',
           '| 数据集 | 库 | F1≥0.95 带宽 | 占全长 | F1≥0.90 带宽 | 占全长 |',
           '|---|---|---|---|---|---|']
    for ds in DATASETS:
        for lib in ('SECOND', 'MUTANT'):
            r = res[ds][lib]
            if r is None:
                continue
            a, b = r['plateau_f1_95'], r['plateau_f1_90']
            ca = f"{a['width']:.3f} | {a['pct']:.1f}%" if a else '无 | -'
            cb = f"{b['width']:.3f} | {b['pct']:.1f}%" if b else '无 | -'
            md.append(f"| {ds} | {lib} | {ca} | {cb} |")
    md += ['']

    if missing:
        md += ['## ⚠️ 缺失', ''] + [f'- {m}' for m in missing] + ['']
    else:
        md += ['> ✅ 4 数据集 × 2 库 齐全。', '']

    out_md = os.path.join(REPO, 'results', 'threshold', 'two_libs.md')
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('written:', out_md)

    with open(os.path.join(REPO, 'results', 'threshold', 'two_libs.json'), 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print('written:', os.path.join(REPO, 'results', 'threshold', 'two_libs.json'))

    # 控制台摘要
    print(f"\n{'数据集':<7}{'库':<8}{'固定':>8}{'标定':>9}{'oracle':>9}{'F1≥0.95带宽':>14}")
    for ds in DATASETS:
        for lib in ('SECOND', 'MUTANT'):
            r = res[ds][lib]
            if r is None:
                continue
            a = r['plateau_f1_95']
            w = f"{a['pct']:.1f}%" if a else '无'
            print(f'{ds:<7}{lib:<8}{r["fixed"]["f1"]:>8.4f}{r["calibrated"]["f1"]:>9.4f}'
                  f'{r["oracle"]["f1"]:>9.4f}{w:>14}')


if __name__ == '__main__':
    main()
