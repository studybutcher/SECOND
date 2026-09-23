"""离线标定 (level, scale)：用缓存的分数扫阈值网格，不跑模型。

依据 docs/阈值选择结果.md F5：`pot_dtaad` 的阈值 ≡ `quantile(init_score, level) * scale`。
所以标定 = 在「训练分数分位数」这条曲线上找最优点，纯 numpy 即可。

同时回答一个更本质的问题：**训练分数的分位数能不能爬到 oracle 阈值的高度**——
即 WADI 那 −0.324 是「参数没调好」还是「分布上就够不着」。

用法：
    python bench/sweep_threshold.py                 # 全部数据集
    python bench/sweep_threshold.py --dataset WADI  # 单个
"""
import argparse
import glob
import json
import os

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(REPO_ROOT, 'results', 'threshold', 'scores')
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']

LEVELS = [0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
SCALES = [0.98, 1.00, 1.02, 1.04]


# --------------------------------------------------------------------------
# 向量化 point-adjust：与 eval_method.adjust_predicts 的循环逐点等价
# --------------------------------------------------------------------------
def adjust_fast(score, label, thresh):
    predict = np.asarray(score) > thresh
    actual = np.asarray(label) > 0.1
    d = np.diff(np.concatenate(([False], actual, [False])).view(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    cs = np.cumsum(predict, dtype=np.int64)
    for s, e in zip(starts, ends):
        before = cs[s - 1] if s > 0 else 0
        if cs[e - 1] - before > 0:          # 段内至少有一个点被判异常
            predict[s:e] = True
    return predict


def f1_at(score, label, thresh):
    p = adjust_fast(score, label, thresh)
    a = np.asarray(label) > 0.1
    tp = np.sum(p & a); fp = np.sum(p & ~a); fn = np.sum(~p & a)
    prec = tp / (tp + fp + 1e-5); rec = tp / (tp + fn + 1e-5)
    return 2 * prec * rec / (prec + rec + 1e-5), prec, rec


def verify():
    """用真实的 calc_seq_abs 校验向量化实现（少了这步后面的数都不能信）。"""
    import sys
    sys.path.insert(0, REPO_ROOT)
    from eval_method import calc_seq_abs
    rng = np.random.default_rng(0)
    n = 5000
    score = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n // 5)])
    label = np.zeros_like(score); label[n:] = 1
    # 再掺几段分散的异常段
    label[100:150] = 1; label[900:1000] = 1
    bad = 0
    for th in np.percentile(score, [10, 50, 80, 95, 99]):
        ref, _ = calc_seq_abs(score, label, float(th))
        got = f1_at(score, label, float(th))
        if not (np.allclose(ref[:3], got, atol=1e-9)):
            bad += 1
            print(f'  ✗ th={th:.4f}  ref={ref[:3]}  got={got}')
    return bad == 0


def analyse(npz_path, levels=LEVELS, scales=SCALES):
    d = np.load(npz_path)
    init = np.asarray(d['init_score'], dtype=float).ravel()
    score = np.asarray(d['test_score'], dtype=float).ravel()
    label = np.asarray(d['test_label']).ravel()

    # ---- 1. oracle：测试分数上的绝对阈值上限 ----
    # F1 关于阈值分段常数，断点在分数取值处；用分位点采样即可覆盖
    cand = np.unique(np.quantile(score, np.linspace(0, 1, 3000)))
    ceil_f1, ceil_th = -1.0, None
    for th in cand:
        v = f1_at(score, label, float(th))[0]
        if v > ceil_f1:
            ceil_f1, ceil_th = v, float(th)

    # ---- 2. 训练分数分位数能否够到 oracle 阈值 ----
    # level_needed = init_score 中小于 oracle 阈值的比例
    level_needed = float(np.mean(init < ceil_th))
    reachable = ceil_th <= init.max()

    # ---- 3. (level, scale) 网格 ----
    grid = []
    for lv in levels:
        base = float(np.quantile(init, lv))
        for sc in scales:
            th = base * sc
            f1, p, r = f1_at(score, label, th)
            grid.append({'level': lv, 'scale': sc, 'threshold': th,
                         'f1': float(f1), 'precision': float(p), 'recall': float(r)})
    best = max(grid, key=lambda g: g['f1'])

    return {'n_init': int(init.size), 'n_test': int(score.size),
            'init': {'min': float(init.min()), 'max': float(init.max()),
                     'q98': float(np.quantile(init, 0.98)),
                     'q999': float(np.quantile(init, 0.999)),
                     'q9999': float(np.quantile(init, 0.9999))},
            'test': {'min': float(score.min()), 'max': float(score.max())},
            'ceil': {'f1': float(ceil_f1), 'threshold': ceil_th},
            'level_needed': level_needed, 'reachable': bool(reachable),
            'grid': grid, 'best': best}


def fine_curve(npz_path, n_pts=1200):
    """阈值响应曲线：F1(绝对阈值)。

    回答的是「可用阈值带有多宽」——带子越窄，标定越脆弱。
    输出 plateau（F1>=0.9 / >=0.94 / >=0.95）的宽度与占分数全长的比例。
    """
    d = np.load(npz_path)
    init = np.asarray(d['init_score'], dtype=float).ravel()
    score = np.asarray(d['test_score'], dtype=float).ravel()
    label = np.asarray(d['test_label']).ravel()

    lo, hi = float(score.min()), float(score.max())
    ths = np.linspace(lo, hi, n_pts)
    rows = []
    for th in ths:
        f1, p, r = f1_at(score, label, float(th))
        rows.append((float(th), float(f1), float(p), float(r)))

    span = hi - lo
    plateaus = {}
    for name, lvl in [('f1>=0.90', 0.90), ('f1>=0.94', 0.94), ('f1>=0.95', 0.95)]:
        sel = [r for r in rows if r[1] >= lvl]
        if sel:
            plateaus[name] = {
                'th_lo': min(r[0] for r in sel), 'th_hi': max(r[0] for r in sel),
                'width': max(r[0] for r in sel) - min(r[0] for r in sel),
                'pct_of_span': 100 * (max(r[0] for r in sel) - min(r[0] for r in sel)) / span,
            }
        else:
            plateaus[name] = None

    # 训练分数能落进 plateau 的 level 区间
    level_span = {}
    for name, pl in plateaus.items():
        if pl:
            level_span[name] = {
                'level_lo': float(np.mean(init < pl['th_lo'])),
                'level_hi': float(np.mean(init < pl['th_hi'])),
            }
            level_span[name]['level_width'] = (level_span[name]['level_hi']
                                               - level_span[name]['level_lo'])
    best = max(rows, key=lambda r: r[1])
    return {'span': span, 'score_range': [lo, hi], 'curve': rows,
            'plateaus': plateaus, 'level_span': level_span,
            'best': {'th': best[0], 'f1': best[1], 'p': best[2], 'r': best[3]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None, choices=DATASETS)
    ap.add_argument('--out', default=os.path.join(IN_DIR, 'sweep.json'))
    ap.add_argument('--fine', action='store_true',
                    help='额外输出阈值响应曲线（可用阈值带的宽度）')
    a = ap.parse_args()

    print('== 校验向量化 point-adjust ==')
    print('   与 calc_seq_abs 一致' if verify() else '   ⚠️ 不一致，结果不可用')
    print()

    ds_list = [a.dataset] if a.dataset else DATASETS
    allres = {}
    for ds in ds_list:
        p = os.path.join(IN_DIR, f'SECOND_{ds}.npz')
        if not os.path.exists(p):
            print(f'{ds}: 缺 {p}，先跑 bench/cache_scores.py')
            continue
        r = analyse(p)
        allres[ds] = r
        print(f'===== {ds} =====')
        print(f'  init_score  n={r["n_init"]}  range=[{r["init"]["min"]:.4f}, {r["init"]["max"]:.4f}]  '
              f'q98={r["init"]["q98"]:.4f}  q99.9={r["init"]["q999"]:.4f}  q99.99={r["init"]["q9999"]:.4f}')
        print(f'  test_score  n={r["n_test"]}  range=[{r["test"]["min"]:.4f}, {r["test"]["max"]:.4f}]')
        print(f'  绝对阈值上限 F1={r["ceil"]["f1"]:.4f} @ th={r["ceil"]["threshold"]:.4f}')
        print(f'  复现该阈值需要的 level={r["level_needed"]:.6f}  '
              f'{"✅ 可达（<= max(init)）" if r["reachable"] else "❌ 不可达（> max(init)）"}')
        b = r['best']
        print(f'  网格最优 level={b["level"]} scale={b["scale"]}  →  '
              f'th={b["threshold"]:.4f}  F1={b["f1"]:.4f} P={b["precision"]:.4f} R={b["recall"]:.4f}')
        # 按 level 汇总（scale=1）看曲线形状
        print('  level 曲线 (scale=1.00):')
        for g in r['grid']:
            if abs(g['scale'] - 1.0) < 1e-9:
                print(f'    level={g["level"]:<8} th={g["threshold"]:>10.4f}  '
                      f'F1={g["f1"]:.4f}  P={g["precision"]:.4f}  R={g["recall"]:.4f}')
        print()

        if a.fine:
            fc = fine_curve(p)
            r = allres[ds]
            r['fine'] = {k: v for k, v in fc.items() if k != 'curve'}
            print(f'  --- 阈值响应曲线（分数全长 {fc["span"]:.2f}）---')
            print(f'  最优 F1={fc["best"]["f1"]:.4f} @ th={fc["best"]["th"]:.4f}')
            for name, pl in fc['plateaus'].items():
                if pl:
                    ls = fc['level_span'][name]
                    print(f'  {name}: th∈[{pl["th_lo"]:.3f}, {pl["th_hi"]:.3f}]  '
                          f'宽度={pl["width"]:.3f} ({pl["pct_of_span"]:.1f}% of 全长)  '
                          f'← 对应 level∈[{ls["level_lo"]:.4f}, {ls["level_hi"]:.4f}] '
                          f'(宽 {ls["level_width"]:.4f})')
                else:
                    print(f'  {name}: 无')
            print()

    with open(a.out, 'w') as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)
    print('saved:', a.out)


if __name__ == '__main__':
    main()
