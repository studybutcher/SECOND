#!/usr/bin/env python
# coding=utf-8
"""PRD6.3：汇总 PESR 的 mixed vs gauss bf / POT① 结果（MSL）。

输入：
  results/robust_ms_mixed/  + results/threshold_ms_mixed/   新跑的 mixed
  results/robust_ms/        + results/threshold_ms/         存档 gauss
  其中 η=0 直接复用 gauss（add_noise 在 η=0 不消耗 RNG，两条路径逐位相同）。

输出：
  results/robust_ms_mixed/summary.md      mixed 逐 seed + 按 η
  results/mixed_vs_gauss_bf.md            表 A/B/C
"""
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETAS = ['0', '0.05', '0.10', '0.15', '0.2', '0.5']
SEEDS = [42, 7, 17]
MIX_BF = 'results/robust_ms_mixed'
MIX_POT = 'results/threshold_ms_mixed'
GAUSS_BF = 'results/robust_ms'
GAUSS_POT = 'results/threshold_ms'


def _p(*a):
    return os.path.join(ROOT, *a)


def load_bf(d, eta, seed, mixed):
    if eta == '0':
        d, tag = (GAUSS_BF if mixed else GAUSS_BF), 'gauss'
        name = f'pesr_MSL_{tag}_noise0_seed{seed}.json'
    else:
        name = (f'pesr_MSL_mixed_noise{eta}_seed{seed}.json' if mixed
                else f'pesr_MSL_gauss_noise{eta}_seed{seed}.json')
    with open(_p(d, name)) as f:
        return json.load(f)['metrics']


def load_pot(d, eta, seed, mixed):
    if eta == '0':
        d = GAUSS_POT
        name = f'SECOND_MSL_fixed_pesr_MSL_noise0_seed{seed}.json'
    else:
        name = (f'SECOND_MSL_fixed_pesr_MSL_mixed_noise{eta}_seed{seed}.json'
                if mixed else
                f'SECOND_MSL_fixed_pesr_MSL_noise{eta}_seed{seed}.json')
    with open(_p(d, name)) as f:
        return json.load(f)['pot']['results']['pot_dtaad']


def ms(v):
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    data = {}
    for eta in ETAS:
        for seed in SEEDS:
            data[(eta, seed)] = {
                'mb': load_bf(MIX_BF, eta, seed, True),
                'mp': load_pot(MIX_POT, eta, seed, True),
                'gb': load_bf(GAUSS_BF, eta, seed, False),
                'gp': load_pot(GAUSS_POT, eta, seed, False),
            }

    lines_b = ['# PESR mixed 逐 seed（MSL）', '',
               '| η | seed | bf P | bf R | bf F1 | bf 百分位 | POT① F1 | POT① th |',
               '|---|---|---|---|---|---|---|---|']
    lines_a = ['# PRD6.3 表 A/B/C：PESR mixed vs gauss（MSL）', '',
               '## 表 A · bf F1（mean ± sd, n=3）', '',
               '| η | PESR gauss | PESR mixed | Δ(mixed−gauss) |',
               '|---|---|---|---|']
    lines_c = ['', '## 表 C · POT① F1（mean ± sd, n=3）', '',
               '| η | PESR gauss | PESR mixed | Δ(mixed−gauss) |',
               '|---|---|---|---|']
    for eta in ETAS:
        rows = [data[(eta, s)] for s in SEEDS]
        for s in SEEDS:
            r = data[(eta, s)]
            b, p = r['mb'], r['mp']
            note = '（复用 gauss）' if eta == '0' else ''
            lines_b.append(
                f'| {eta}{note} | {s} | {b["precision"]:.4f} | {b["recall"]:.4f} | '
                f'{b["f1"]:.4f} | {b["threshold"]:.2f} | {p["f1"]:.4f} | '
                f'{p["threshold"]:.3f} |')
        mb = ms([r['mb']['f1'] for r in rows])
        gb = ms([r['gb']['f1'] for r in rows])
        mp = ms([r['mp']['f1'] for r in rows])
        gp = ms([r['gp']['f1'] for r in rows])
        lines_a.append(f'| {eta} | {gb[0]:.4f} ± {gb[1]:.4f} | '
                       f'{mb[0]:.4f} ± {mb[1]:.4f} | {mb[0]-gb[0]:+.4f} |')
        lines_c.append(f'| {eta} | {gp[0]:.4f} ± {gp[1]:.4f} | '
                       f'{mp[0]:.4f} ± {mp[1]:.4f} | {mp[0]-gp[0]:+.4f} |')

    allm = [data[(e, s)]['mb']['f1'] for e in ETAS for s in SEEDS]
    allg = [data[(e, s)]['gb']['f1'] for e in ETAS for s in SEEDS]
    allmp = [data[(e, s)]['mp']['f1'] for e in ETAS for s in SEEDS]
    allgp = [data[(e, s)]['gp']['f1'] for e in ETAS for s in SEEDS]
    lines_a.append(f'| **跨η平均** | **{st.mean(allg):.4f}** | '
                   f'**{st.mean(allm):.4f}** | **{st.mean(allm)-st.mean(allg):+.4f}** |')
    lines_c.append(f'| **跨η平均** | **{st.mean(allgp):.4f}** | '
                   f'**{st.mean(allmp):.4f}** | **{st.mean(allmp)-st.mean(allgp):+.4f}** |')
    lines_b.append('')
    lines_b.append('> η=0 行复用 gauss 存档（`results/robust_ms/`），'
                   '加噪函数在 η=0 直接返回，两条路径逐位相同。')

    os.makedirs(_p(MIX_BF), exist_ok=True)
    with open(_p(MIX_BF, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines_b) + '\n')
    with open(_p('results', 'mixed_vs_gauss_bf.md'), 'w') as f:
        f.write('\n'.join(lines_a + lines_c) + '\n')
    print('\n'.join(lines_a))
    print()
    print('\n'.join(lines_c))
    print()
    print('\n'.join(lines_b))


if __name__ == '__main__':
    main()
