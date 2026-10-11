#!/usr/bin/env python
"""PRD5.1 r3 · AE 实验的最终汇总（§7 主表 + §9 判据）。

    E_N       = 固定 clean-normal val 上的逐元素 MSE（主指标）
    E_N_med10 = 末 10 轮 val MSE 的中位（末点抖动时的稳健量）
    ΔE_N(ρ)   = (E_N(ρ) − E_N(0)) / E_N(0)，**同 seed 配对后再平均**
    E_A_clip  = anomaly probe MSE（clip 到训练范围 [0,1]；原始版被 MinMax 外推的极端值主导）

判据（PRD §9，预注册）：
    ΔE_N 随 ρ 单调上升 → D1「正常模式重建质量受可测量影响」
    不单调/≈0        → D2「本协议下未检测到影响」，不得声称已定量证明
    ΔE_N < 0         → D3「可能正则化」，不得写成污染有害

用法：python bench/summarize_ae.py --model ae [--out results/ae_mpa/summary.md]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ae')
    ap.add_argument('--out', default='results/ae_mpa/summary.md')
    ap.add_argument('--min-epochs', type=int, default=20)
    a = ap.parse_args()

    rs = {}
    for p in sorted(glob.glob(f'results/ae_mpa/{a.model}_*_rho*_seed*.json')):
        d = json.load(open(p))
        if d['epochs'] < a.min_epochs:
            continue
        rs[(d['rho'], d['seed'])] = d
    if not rs:
        raise SystemExit('[FAIL] 没有可用结果')
    rhos = sorted({r for r, _ in rs})
    seeds = sorted({s for _, s in rs})
    base = {s: rs[(rhos[0], s)] for s in seeds if (rhos[0], s) in rs}

    L = [f'# PRD5.1 r3 · AE 实验汇总（{a.model}）', '',
         f'MSL, W=70，结构化异常段注入 α=1.5σ，污染 seed = 训练 seed。',
         f'ρ ∈ {rhos}，seed ∈ {seeds}（**n={len(seeds)}**），'
         f'每个 run {rs[(rhos[0], seeds[0])]["epochs"]} epoch、'
         f'lr_schedule={rs[(rhos[0], seeds[0])].get("lr_schedule")}。', '',
         '## 逐 seed 的 $E_N$', '',
         '| ρ | ' + ' | '.join(f'seed {s}' for s in seeds) + ' | 均值 | std |',
         '|' + '---|' * (len(seeds) + 3)]
    for r in rhos:
        v = [rs[(r, s)]['E_N'] for s in seeds if (r, s) in rs]
        L.append(f'| {r:.0%} | ' + ' | '.join(f'{x:.6f}' for x in v) +
                 f' | {np.mean(v):.6f} | {np.std(v, ddof=1):.6f} |')

    L += ['', '## 主表：$E_N$ / $\\Delta E_N$ / $E_A$', '',
          '| ρ | $E_N$ (mean ± std) | **$\\Delta E_N$** (配对) | 逐 seed $\\Delta E_N$ | 同号 | '
          '$E_N$ 配对 t 检验 | $E_{A,clip}$ | $E_N/E_{A,clip}$ |',
          '|' + '---|' * 8]
    for r in rhos:
        v = np.array([rs[(r, s)]['E_N'] for s in seeds if (r, s) in rs])
        ea = np.array([rs[(r, s)].get('E_A_clip', np.nan) for s in seeds if (r, s) in rs])
        row = f'| {r:.0%} | {np.mean(v):.6f} ± {np.std(v, ddof=1):.6f} |'
        if r > rhos[0] and base:
            pr = [(rs[(r, s)]['E_N'] - base[s]['E_N']) / base[s]['E_N']
                  for s in seeds if (r, s) in rs and s in base]
            xs = [base[s]['E_N'] for s in seeds if (r, s) in rs and s in base]
            ys = [rs[(r, s)]['E_N'] for s in seeds if (r, s) in rs and s in base]
            t, pv = stats.ttest_rel(ys, xs)
            same = '是' if all(x > 0 for x in pr) or all(x < 0 for x in pr) else '否'
            row += (f' **{np.mean(pr):+.4f}** ± {np.std(pr, ddof=1):.4f} | '
                    + ' '.join(f'{x:+.3f}' for x in pr)
                    + f' | {same} | t={t:.2f}, p={pv:.3f} |')
        else:
            row += ' （基准） | — | — | — |'
        row += f' {np.mean(ea):.6f} | {np.mean(v) / np.mean(ea):.3f} |'
        L.append(row)

    # ── 未收敛 run（机械规则，事先写死，不是事后挑）────────────────────
    allmap = np.array([d['E_N'] for d in rs.values()])
    med = float(np.median(allmap))
    bad = {k for k, d in rs.items() if d['E_N'] > 2 * med}
    L += ['', f'## 未收敛 run（规则：$E_N$ > 2×全部 run 的中位 {med:.6f}）', '']
    if bad:
        L.append('| ρ | seed | $E_N$ | 说明 |')
        L.append('|---|---|---|---|')
        for (r, s) in sorted(bad):
            d = rs[(r, s)]
            e = [x['val_mse'] for x in d['curve']]
            L.append(f'| {r:.0%} | {s} | {d["E_N"]:.6f} | '
                     f'最优在 ep{d["E_N_min_epoch"]}（{d["E_N_min"]:.6f}）之后反而变差，'
                     f'末 10 轮极点比 {max(e[-10:]) / min(e[-10:]):.2f} |')
        L.append('')
        L.append(f'共 {len(bad)}/{len(rs)} 个 run（{100 * len(bad) / len(rs):.0f}%）落入坏盆地。'
                 '**下面是剔除它们之后的同一张表**（两个口径都报，不择其一）。')
        L += ['', '| ρ | $E_N$ (mean ± std) | $\\Delta E_N$ (配对) | 逐 seed $\\Delta E_N$ | n | '
                  '配对 t 检验 |', '|' + '---|' * 6]
        for r in rhos:
            keep = [s for s in seeds if (r, s) in rs and (r, s) not in bad]
            keepb = [s for s in keep if s in base and (rhos[0], s) not in bad]
            v = np.array([rs[(r, s)]['E_N'] for s in keep])
            row = f'| {r:.0%} | {np.mean(v):.6f} ± {np.std(v, ddof=1):.6f} |'
            if r > rhos[0] and keepb:
                pr = [(rs[(r, s)]['E_N'] - base[s]['E_N']) / base[s]['E_N'] for s in keepb]
                t, pv = stats.ttest_rel([rs[(r, s)]['E_N'] for s in keepb],
                                        [base[s]['E_N'] for s in keepb])
                same = '是' if all(x > 0 for x in pr) or all(x < 0 for x in pr) else '否'
                row += (f' **{np.mean(pr):+.4f}** ± {np.std(pr, ddof=1):.4f} | '
                        + ' '.join(f'{x:+.3f}' for x in pr)
                        + f' | {len(pr)} | t={t:.2f}, p={pv:.3f} | 同号={same} |')
            else:
                row += ' （基准） | — | — | — |'
            L.append(row)
    else:
        L.append('无。')

    # 单调性
    L += ['', '## 单调性（逐 seed）', '',
          '| seed | ' + ' → '.join(f'{r:.0%}' for r in rhos) + ' | 单调递增 |',
          '|' + '---|' * (len(rhos) + 2)]
    nmono = 0
    for s in seeds:
        v = [rs[(r, s)]['E_N'] for r in rhos if (r, s) in rs]
        mono = all(v[i] < v[i + 1] for i in range(len(v) - 1))
        nmono += mono
        L.append(f'| {s} | ' + ' → '.join(f'{x:.6f}' for x in v) + f' | {"✅" if mono else "❌"} |')

    # 判据
    d10 = [(rs[(rhos[-1], s)]['E_N'] - base[s]['E_N']) / base[s]['E_N']
           for s in seeds if (rhos[-1], s) in rs and s in base]
    L += ['', '## 按 §9 的判据', '',
          f'- 严格单调递增的 seed：**{nmono}/{len(seeds)}**',
          f'- $\\Delta E_N$ 在最大 ρ（{rhos[-1]:.0%}）上：**{np.mean(d10):+.4f} ± '
          f'{np.std(d10, ddof=1):.4f}**',
          f'- 配对 t 检验 p 值见主表；**n={len(seeds)}，若 p>0.05 则不能下显著结论**', '']
    mono_all = nmono == len(seeds)
    if mono_all and np.mean(d10) > 0:
        L.append('- ⇒ **D1 成立**：随 ρ 上升 $E_N$ 单调上升')
    elif np.mean(d10) > 0:
        L.append('- ⇒ **介于 D1 与 D2 之间**：最大 ρ 上 $E_N$ 上升，但**非严格单调**、'
                 '且样本量有限 → 只能说"观察到一致方向的弱退化"，不能声称已定量证明')
    else:
        L.append('- ⇒ **D2/D3**：未观察到随 ρ 的退化；**不得**写成污染损害了学习')

    L += ['', '> 注：$E_{A,clip}$ 是 anomaly probe 的 MSE（clip 到训练范围 [0,1]）。'
              '未 clip 的版本因 MinMax 对 test 的外推（实测 max=46）被极端值主导，不用于比较。',
          '> 注：`E_N_med10`（末 10 轮中位）与 `E_N` 结论一致时才可放心用末点。']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nsaved: {a.out}')


if __name__ == '__main__':
    main()
