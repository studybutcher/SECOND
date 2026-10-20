#!/usr/bin/env python
# coding=utf-8
"""汇总 PRD6.2 的“不重训 ③ 标定”结果。

两种粒度的 ③ 都报告：

  A. per-run（推荐，部署语义）：每个 (η, seed) 是独立模型，各自在自己的
     val 上选 s*，再在 test 上评测；
  B. per-η shared（PRD §2.1.1 原定口径）：同 η 的 3 个 seed 共用
     s*(η)=argmax_s mean_seed val_F1。
     平分取较大的 scale。

为了看清 val⊂test 的乐观程度，每条 run 同时给：
  - test F1：完整 65% 测试集（与 PRD6.1 相同，约 54% 窗口与 val 重叠）；
  - held-out F1：去掉与 val 重叠窗口后的 22,049 个窗口；
  - bf_free：在 held-out 上用 bf_search 选阈值（oracle 上界，同一子集）；
  - ①（s=1.0）的 test / held-out F1。
"""
import argparse
import csv
import json
import os

import numpy as np

ETAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.50]
SEEDS = [42, 7, 17]
MODEL_DIR = {
    'LSTM-AE': ('results/threshold_lstmae_potcal', 'lstmae'),
    'PESR':    ('results/threshold_pesr_potcal', 'pesr'),
}
ETA_TAG = {0.0: '0', 0.05: '0.05', 0.10: '0.10',
           0.15: '0.15', 0.2: '0.2', 0.5: '0.5'}


def f4(x):
    return f'{x:.4f}'


def pm(m, s, nd=4):
    return f'{m:.{nd}f} ± {s:.{nd}f}'


def load_model(model):
    d, stem = MODEL_DIR[model]
    runs = {}
    for eta in ETAS:
        for seed in SEEDS:
            p = os.path.join(d, f'{stem}_MSL_noise{ETA_TAG[eta]}_seed{seed}.json')
            if not os.path.exists(p):
                raise SystemExit(f'[FAIL] 缺文件 {p}')
            with open(p) as f:
                runs[(eta, seed)] = json.load(f)
    return runs


def pick(run, s):
    for r in run['scales']:
        if abs(r['scale'] - s) < 1e-12:
            return r
    raise KeyError(s)


def scale_axis(runs, eta):
    ref = runs[(eta, SEEDS[0])]['scales']
    scales = [r['scale'] for r in ref]
    for seed in SEEDS[1:]:
        cur = runs[(eta, seed)]['scales']
        assert [r['scale'] for r in cur] == scales, (eta, seed)
    return scales


def plateau_scale(curve, scales, delta=0.005):
    """敏感性分析：val 曲线平台中心。

    δ 内（max−δ）的、包含 argmax 的连续区间取中位 scale。若平台宽度为
    偶数，取较低的一侧（更小的 scale，离高阈值悬崖更远）。
    """
    j = int(np.argmax(curve))
    lo = hi = j
    while lo > 0 and curve[lo - 1] >= curve[j] - delta:
        lo -= 1
    while hi + 1 < len(curve) and curve[hi + 1] >= curve[j] - delta:
        hi += 1
    return scales[(lo + hi) // 2]


def per_run_rows(runs, model):
    rows = []
    for eta in ETAS:
        for seed in SEEDS:
            run = runs[(eta, seed)]
            scales = scale_axis(runs, eta)
            curve = np.array([r['val_f1'] for r in run['scales']])
            # 平分取较大的 scale
            j = max(range(len(scales)), key=lambda k: (curve[k], scales[k]))
            s = scales[j]
            r = run['scales'][j]
            r1 = pick(run, 1.0)
            spl = plateau_scale(curve, scales)
            rpl = pick(run, spl)
            rows.append({
                'model': model, 'eta': eta, 'seed': seed, 's_star': s,
                'th_star': r['threshold'],
                'val_f1': r['val_f1'],
                'test_f1': r['test_f1'], 'test_p': r['test_p'], 'test_r': r['test_r'],
                'free_f1': r['free_f1'], 'free_p': r['free_p'], 'free_r': r['free_r'],
                'bf_test_f1': run.get('bf_test', {}).get('f1', float('nan')),
                'bf_free_f1': run.get('bf_free', {}).get('f1', float('nan')),
                'test_f1_scale1': r1['test_f1'],
                'free_f1_scale1': r1['free_f1'],
                's_plateau': spl,
                'test_f1_plateau': rpl['test_f1'],
                'free_f1_plateau': rpl['free_f1'],
                'val_f1_plateau': rpl['val_f1'],
            })
    return rows


def shared_rows(runs, model):
    rows = []
    for eta in ETAS:
        scales = scale_axis(runs, eta)
        curves = np.array([[r['val_f1'] for r in runs[(eta, sd)]['scales']]
                           for sd in SEEDS])
        mean_curve = curves.mean(0)
        j = max(range(len(scales)), key=lambda k: (mean_curve[k], scales[k]))
        s = scales[j]
        got = [pick(runs[(eta, sd)], s) for sd in SEEDS]
        get1 = [pick(runs[(eta, sd)], 1.0) for sd in SEEDS]
        th = np.array([g['threshold'] for g in got])
        rows.append({
            'model': model, 'eta': eta, 's_star': s,
            'th_mean': th.mean(), 'th_sd': th.std(),
            'val_f1': mean_curve[j],
            'test_f1_mean': np.mean([g['test_f1'] for g in got]),
            'test_f1_sd': np.std([g['test_f1'] for g in got]),
            'free_f1_mean': np.mean([g['free_f1'] for g in got]),
            'free_f1_sd': np.std([g['free_f1'] for g in got]),
            'bf_free_mean': np.mean([runs[(eta, sd)].get('bf_free', {})
                                     .get('f1', float('nan')) for sd in SEEDS]),
            'test_f1_scale1': np.mean([g['test_f1'] for g in get1]),
            'free_f1_scale1': np.mean([g['free_f1'] for g in get1]),
        })
    return rows


def stability_rows(runs):
    rows = []
    for eta in ETAS:
        scales = scale_axis(runs, eta)
        curves = np.array([[r['val_f1'] for r in runs[(eta, sd)]['scales']]
                           for sd in SEEDS])
        arg = [scales[int(np.argmax(c))] for c in curves]
        # 逐 seed：argmax 与次优的 val 差、以及 argmax 附近 ±1 格的 val 变化
        detail = []
        for c in curves:
            k = int(np.argmax(c))
            nxt = np.sort(c)[-2]
            detail.append((scales[k], c[k], c[k] - nxt))
        rows.append({'eta': eta, 'argmax': arg, 'detail': detail,
                     'scales': scales, 'curves': curves})
    return rows


def write_model_md(model, runs, out_path):
    L = []
    L.append(f'## {model}（现有 ckpt，不重训）\n')

    # ── A. per-run ────────────────────────────────────────────────────
    pr = per_run_rows(runs, model)
    L.append('### A. per-run ③（每个 seed 用自己的 val 选 s*）\n')
    L.append('| η | seed | s* | th* | val F1 | test F1 | held-out F1 | '
             'bf held-out (oracle) | ① test F1 | ① held-out F1 | '
             's* (平台中心) | test F1 (平台) | held-out F1 (平台) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in pr:
        L.append(f'| {r["eta"]:g} | {r["seed"]} | {r["s_star"]:g} | '
                 f'{r["th_star"]:.2f} | {f4(r["val_f1"])} | {f4(r["test_f1"])} | '
                 f'{f4(r["free_f1"])} | {f4(r["bf_free_f1"])} | '
                 f'{f4(r["test_f1_scale1"])} | {f4(r["free_f1_scale1"])} | '
                 f'{r["s_plateau"]:g} | {f4(r["test_f1_plateau"])} | '
                 f'{f4(r["free_f1_plateau"])} |')
    L.append('')
    L.append('| η | s* (mean±sd) | th* (mean±sd) | val F1 | test F1 | '
             'held-out F1 | bf held-out | ① test F1 | ① held-out F1 | '
             'test F1 (平台) | held-out F1 (平台) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for eta in ETAS:
        g = [r for r in pr if r['eta'] == eta]
        arr = lambda k: np.array([r[k] for r in g])
        L.append(f'| {eta:g} | {pm(arr("s_star").mean(), arr("s_star").std(), 4)} | '
                 f'{pm(arr("th_star").mean(), arr("th_star").std(), 2)} | '
                 f'{arr("val_f1").mean():.4f} | '
                 f'{pm(arr("test_f1").mean(), arr("test_f1").std())} | '
                 f'{pm(arr("free_f1").mean(), arr("free_f1").std())} | '
                 f'{arr("bf_free_f1").mean():.4f} | '
                 f'{arr("test_f1_scale1").mean():.4f} | '
                 f'{arr("free_f1_scale1").mean():.4f} | '
                 f'{arr("test_f1_plateau").mean():.4f} | '
                 f'{arr("free_f1_plateau").mean():.4f} |')
    L.append('')
    L.append('> “平台”列是选择规则的敏感性分析：在 val 曲线 max−0.005 内、'
             '包含 argmax 的连续区间取中位 scale。δ=0.005 是先验设定，'
             '只用于判断 argmax 是否在噪声上追涨，不替代主口径。\n')

    # ── B. shared ─────────────────────────────────────────────────────
    sr = shared_rows(runs, model)
    L.append('### B. per-η shared ③（3 seeds 共用 s*，PRD §2.1.1 原定口径）\n')
    L.append('| η | s*(η) | th (mean±sd) | val F1 | test F1 | held-out F1 | '
             'bf held-out | ① test F1 | ① held-out F1 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for r in sr:
        L.append(f'| {r["eta"]:g} | {r["s_star"]:g} | '
                 f'{pm(r["th_mean"], r["th_sd"], 2)} | {f4(r["val_f1"])} | '
                 f'{pm(r["test_f1_mean"], r["test_f1_sd"])} | '
                 f'{pm(r["free_f1_mean"], r["free_f1_sd"])} | '
                 f'{r["bf_free_mean"]:.4f} | {r["test_f1_scale1"]:.4f} | '
                 f'{r["free_f1_scale1"]:.4f} |')
    L.append('')

    # ── C. stability ──────────────────────────────────────────────────
    L.append('### C. 标定稳定性（逐 seed val argmax）\n')
    L.append('| η | seed42 | seed7 | seed17 | val 曲线是否单峰 |')
    L.append('|---|---|---|---|---|')
    for st in stability_rows(runs):
        curve = st['curves'].mean(0)
        j = int(np.argmax(curve))
        lohi = bool(np.all(np.diff(curve[:j + 1]) >= -1e-12) and
                    np.all(np.diff(curve[j:]) <= 1e-12))
        L.append(f'| {st["eta"]:g} | ' +
                 ' | '.join(f'{x:g}' for x in st['argmax']) +
                 f' | {"是" if lohi else "否"} |')
    L.append('')
    L.append('逐 seed 的 (argmax s, val F1, 与次优的 val 差)：\n')
    for st in stability_rows(runs):
        cells = []
        for sd, (s, v, gap) in zip(SEEDS, st['detail']):
            cells.append(f'seed{sd}: {s:g} ({v:.4f}, Δ={gap:+.4f})')
        L.append(f'- η={st["eta"]:g}：' + '；'.join(cells))
    L.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    return pr, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/pot_cal_no_retrain_summary.md')
    ap.add_argument('--csv', default='results/pot_cal_no_retrain_summary.csv')
    ap.add_argument('--models', default='LSTM-AE,PESR',
                    help='逗号分隔，例如 LSTM-AE')
    args = ap.parse_args()

    md = ['# PRD6.2 · 不重训的 POT ③ 标定结果（MSL，3 seeds）\n',
          '> POT 口径：`q=1e-3`、`level=0.98`、`init_score=全训练集窗口分数` 固定；',
          '> 只标定 `scale`；`th = base_th × s`。标定只用 val，评测只用 test。\n',
          '> 完整 test = 与 PRD6.1 相同的 65% 测试集（约 54% 的窗口与 val 重叠）；',
          '> held-out test = 去掉与 val 重叠的 22,049 个窗口。',
          '> `bf held-out` 是在同一 held-out 子集上用 bf_search 选阈值的 oracle 上界。\n']
    rows = []
    for model in [m.strip() for m in args.models.split(',') if m.strip()]:
        runs = load_model(model)
        d, _ = MODEL_DIR[model]
        pr, sr = write_model_md(model, runs, os.path.join(d, 'scale_table.md'))
        rows += pr
        with open(os.path.join(d, 'scale_table.md')) as f:
            md.append(f.read())
    with open(args.out, 'w') as f:
        f.write('\n'.join(md) + '\n')
    if rows:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f'[ok] {args.out}  {args.csv}')


if __name__ == '__main__':
    main()
