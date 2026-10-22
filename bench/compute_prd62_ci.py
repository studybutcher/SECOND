#!/usr/bin/env python
# coding=utf-8
"""PRD6.2 §5.1 主表：完整 test F1 的 95% 置信区间（seed 级，n=3）。

数据来源（全部是已落盘的逐 seed 结果，不重新推理）：
  LSTM-AE bf      results/lstmae_robust_gauss/lstmae_MSL_gauss_noise{η}_seed{s}.json
  LSTM-AE POT③    results/threshold_lstmae_potcal/lstmae_MSL_noise{η}_seed{s}.json
  PESR bf         results/robust_ms/pesr_MSL_gauss_noise{η}_seed{s}.json
  PESR POT①       results/threshold_ms/SECOND_MSL_fixed_pesr_MSL_noise{η}_seed{s}.json
  PESR POT③       results/threshold_pesr_potcal/pesr_MSL_noise{η}_seed{s}.json

③ 的逐 run 值取该 run 自己 val argmax 对应的 scale 行（与 PRD6.2 主口径一致）。

区间：t 分布，df = n−1 = 2，95% 双侧
      mean ± t_{0.975,2} · sd_sample / sqrt(n)
      t = 4.302653，sd 用样本标准差（ddof=1）。
      同时给出 seed 的 min–max（无分布假设）。

输出：
  results/prd62_main_table_ci.md
  results/prd62_main_table_ci.csv
"""
import csv
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETAS = ['0', '0.05', '0.10', '0.15', '0.2', '0.5']
SEEDS = [42, 7, 17]
T975_DF2 = 4.302652729696142   # scipy.stats.t.ppf(0.975, 2)


def _p(*a):
    return os.path.join(ROOT, *a)


def _json(*a):
    with open(_p(*a)) as f:
        return json.load(f)


def lstmae_bf(eta, seed):
    return _json('results/lstmae_robust_gauss',
                 f'lstmae_MSL_gauss_noise{eta}_seed{seed}.json')['metrics']['f1']


def lstmae_pot3(eta, seed):
    j = _json('results/threshold_lstmae_potcal',
              f'lstmae_MSL_noise{eta}_seed{seed}.json')
    return max(j['scales'], key=lambda r: r['val_f1'])['test_f1']


def pesr_bf(eta, seed):
    return _json('results/robust_ms',
                 f'pesr_MSL_gauss_noise{eta}_seed{seed}.json')['metrics']['f1']


def pesr_pot1(eta, seed):
    j = _json('results/threshold_ms',
              f'SECOND_MSL_fixed_pesr_MSL_noise{eta}_seed{seed}.json')
    return j['pot']['results']['pot_dtaad']['f1']


def pesr_pot3(eta, seed):
    j = _json('results/threshold_pesr_potcal',
              f'pesr_MSL_noise{eta}_seed{seed}.json')
    return max(j['scales'], key=lambda r: r['val_f1'])['test_f1']


METRICS = [
    ('LSTM-AE bf', lstmae_bf),
    ('LSTM-AE POT③', lstmae_pot3),
    ('PESR bf', pesr_bf),
    ('PESR POT①', pesr_pot1),
    ('PESR POT③', pesr_pot3),
]

# 同一 run 内这些口径共享同一份 test 分数，逐 seed 相减是配对比较。
PAIRS = [
    ('PESR POT③ − POT①', pesr_pot3, pesr_pot1),
    ('PESR bf − POT①', pesr_bf, pesr_pot1),
    ('LSTM-AE bf − POT③', lstmae_bf, lstmae_pot3),
]


def summarize(vals):
    m = st.mean(vals)
    sd = st.stdev(vals)                     # ddof=1
    hw = T975_DF2 * sd / math.sqrt(len(vals))
    return m, sd, m - hw, m + hw, min(vals), max(vals)


def main():
    rows = []
    for eta in ETAS:
        for name, fn in METRICS:
            vals = [fn(eta, s) for s in SEEDS]
            m, sd, lo, hi, vmin, vmax = summarize(vals)
            rows.append({'eta': eta, 'metric': name,
                         **{f'seed{s}': v for s, v in zip(SEEDS, vals)},
                         'mean': m, 'sd': sd, 'ci95_lo': lo, 'ci95_hi': hi,
                         'min': vmin, 'max': vmax})

    # 配对差分：同 run 内共享分数，Δ = A(seed) − B(seed) 逐 seed 计算
    pair_rows = []
    for eta in ETAS:
        for name, fa, fb in PAIRS:
            diffs = [fa(eta, s) - fb(eta, s) for s in SEEDS]
            m, sd, lo, hi, vmin, vmax = summarize(diffs)
            pair_rows.append({'eta': eta, 'pair': name,
                              **{f'seed{s}': v for s, v in zip(SEEDS, diffs)},
                              'mean': m, 'sd': sd, 'ci95_lo': lo, 'ci95_hi': hi,
                              'min': vmin, 'max': vmax,
                              'contains_0': 'yes' if lo <= 0 <= hi else 'no'})

    os.makedirs(_p('results'), exist_ok=True)
    with open(_p('results', 'prd62_paired_diff_ci.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    with open(_p('results', 'prd62_main_table_ci.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    L = ['# PRD6.2 §5.1 主表 · 完整 test F1 的 95% 置信区间', '',
         '完整 test = 47,854 窗口；n=3 seeds。区间为 seed 级 t 区间'
         '（df=2，t=4.303，样本 sd/ddof=1），不是测试窗口级 bootstrap。', '',
         '| η | 口径 | seed 42 | seed 7 | seed 17 | mean ± sd | 95% CI |',
         '|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f'| {r["eta"]} | {r["metric"]} | {r["seed42"]:.4f} | '
                 f'{r["seed7"]:.4f} | {r["seed17"]:.4f} | '
                 f'{r["mean"]:.4f} ± {r["sd"]:.4f} | '
                 f'[{r["ci95_lo"]:.4f}, {r["ci95_hi"]:.4f}] |')
    L += ['', '## 宽表（mean [95% CI]）', '',
          '| η | LSTM-AE bf | LSTM-AE POT③ | PESR bf | PESR POT① | PESR POT③ |',
          '|---|---|---|---|---|---|']
    by = {(r['eta'], r['metric']): r for r in rows}
    for eta in ETAS:
        cells = []
        for name, _ in METRICS:
            r = by[(eta, name)]
            cells.append(f'{r["mean"]:.4f} [{r["ci95_lo"]:.4f}, {r["ci95_hi"]:.4f}]')
        L.append(f'| {eta} | ' + ' | '.join(cells) + ' |')
    L += ['',
          '> 说明：η=0.5 的 LSTM-AE bf 只有 3 个 seed，其中 seed7=0.2960，'
          't 区间为 [0.0255, 1.0334]，上界超过 1；这是 n=3 + 离群点的直接后果，'
          '不应把该区间当作精确的 95% 覆盖。',
          '> 同理 PESR POT③ 在 η=0.2 的上界 1.0016 也略超 1。',
          '> 若论文需要更窄、只反映测试集规模的区间，需要保存逐窗口预测后做 '
          'block bootstrap；当前结果未落盘逐窗口分数。', '',
          '## 配对差分（同一 run 内共享 test 分数）', '',
          'Δ = A − B 逐 seed 计算后取 t 区间；`contains_0=no` 表示 95% 区间'
          '不含 0。', '',
          '| η | 比较 | Δ mean [95% CI] | 含 0 |',
          '|---|---|---|---|']
    for r in pair_rows:
        L.append(f'| {r["eta"]} | {r["pair"]} | '
                 f'{r["mean"]:+.4f} [{r["ci95_lo"]:+.4f}, {r["ci95_hi"]:+.4f}] | '
                 f'{r["contains_0"]} |')
    with open(_p('results', 'prd62_main_table_ci.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
