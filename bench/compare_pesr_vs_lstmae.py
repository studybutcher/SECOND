#!/usr/bin/env python
# coding=utf-8
"""PRD6.2 §5.1：PESR vs LSTM-AE 的显著性检验（完整 test，n=3 seeds）。

为什么可以按 seed 配对：
  main.py 的顺序是 set_seed(seed) → get_data → add_noise(train_data)
  → build_model。两个 backbone 走同一条数据路径，同一 seed 下
  **训练集噪声实现与 test 窗口完全相同**；模型初始化在 add_noise 之后，
  不影响数据。因此 (同 η, 同 seed) 的两个模型共享数据难度，配对合理。
  但两个模型的初始化 RNG 消耗不同，配对只共享“数据”这一半，故同时报
  不配对的 Welch t 作为敏感性分析。

检验口径：
  - 逐 η：paired t（df=2）与 Welch t；效应量 dz = mean(Δ)/sd(Δ)。
  - 6 个 η 上做 Holm 校正（每个比较族内），另给 Bonferroni 同时 95% CI
    （m=6，df=2，t=10.886）作为与多重校正配套的区间；Holm p 与单项 CI
    不严格对偶，不能混配。
  - 跨 η 汇总：先对每个 seed 在 η 网格上取平均，再做 n=3 的 paired t。
    这只是“对这个 η 网格平均”的描述性检验，不应外推到新 η。
  - n=3 下 p 值完全依赖正态假设；非参数检验在 n=3 时最小双侧 p=0.25，
    无法显著，故未用。

输出：
  results/prd62_pesr_vs_lstmae_test.md
  results/prd62_pesr_vs_lstmae_test.csv
"""
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_prd62_ci import (  # noqa: E402
    ETAS, SEEDS, lstmae_bf, lstmae_pot3, pesr_bf, pesr_pot1, pesr_pot3,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPARISONS = [
    ('PESR bf − LSTM-AE bf', pesr_bf, lstmae_bf),
    ('PESR POT① − LSTM-AE POT③', pesr_pot1, lstmae_pot3),
    ('PESR POT③ − LSTM-AE POT③', pesr_pot3, lstmae_pot3),
]


def holm(pvals):
    """Holm-Bonferroni；返回与输入同序的校正 p。"""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adj[i] = running
    return adj


def ci_t(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    tcrit = stats.t.ppf(0.975, n - 1)
    hw = tcrit * sd / math.sqrt(n)
    return m, sd, m - hw, m + hw


def paired_row(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, sd, lo, hi = ci_t(d)
    t, p = stats.ttest_rel(a, b)
    dz = m / sd if sd > 0 else float('inf')
    return {'diff': d, 'mean': m, 'sd': sd, 'se': sd / math.sqrt(len(d)),
            'lo': lo, 'hi': hi, 't': t, 'p': p, 'dz': dz}


def bonf_ci(row, m):
    """Bonferroni 同时 95% CI（m 个检验，df=n−1）。比 Holm 保守。"""
    tc = stats.t.ppf(1 - 0.05 / (2 * m), len(row['diff']) - 1)
    hw = tc * row['se']
    return row['mean'] - hw, row['mean'] + hw


def welch_row(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return t, p


def fmt_p(p):
    return '<1e-4' if p < 1e-4 else f'{p:.4f}'


def main():
    L = ['# PESR vs LSTM-AE 显著性检验（完整 test，47,854 窗口，n=3）', '',
         '配对单位 = seed；同一 seed 下两模型共享训练噪声实现与 test 窗口'
         '（`set_seed → get_data → add_noise → build_model`）。', '',
         '效应量 dz = mean(Δ)/sd(Δ)（配对）。Holm 校正在每个比较族内'
         '对 6 个 η 做。n=3 时 p 值依赖正态假设，非参数检验最小 p=0.25，'
         '因此同时看效应量与 Δ 的 95% CI。', '',
         '**p 与 CI 的对偶关系**：双侧 α=0.05 下，`raw p < 0.05` ⟺'
         ' 0 不在**单项** 95% CI 内（同一个 t 统计量）。Holm p 是校正后的，'
         '不能配单项 CI；表中另给 `Bonferroni 95% CI（m=6）` 作为'
         '同时区间。Bonferroni 比 Holm step-down 更保守，两者不严格对偶。'
         '跨 η 平均是单次检验，单项 CI 即可。', '']

    csv_rows = []
    for name, fa, fb in COMPARISONS:
        L += [f'## {name}', '',
              '| η | PESR | LSTM-AE | Δ mean | 95% CI（单项） | '
              'Bonferroni 95% CI（m=6） | dz | raw p | Holm p | Welch p |',
              '|---|---|---|---|---|---|---|---|---|---|']
        rows = []
        for eta in ETAS:
            a = [fa(eta, s) for s in SEEDS]
            b = [fb(eta, s) for s in SEEDS]
            r = paired_row(a, b)
            wt, wp = welch_row(a, b)
            r.update({'eta': eta, 'a': np.mean(a), 'b': np.mean(b),
                      'welch_t': wt, 'welch_p': wp})
            r['bonf_lo'], r['bonf_hi'] = bonf_ci(r, len(ETAS))
            rows.append(r)
        hp = holm([r['p'] for r in rows])
        for r, h in zip(rows, hp):
            L.append(f'| {r["eta"]} | {r["a"]:.4f} | {r["b"]:.4f} | '
                     f'{r["mean"]:+.4f} | '
                     f'[{r["lo"]:+.4f}, {r["hi"]:+.4f}] | '
                     f'[{r["bonf_lo"]:+.4f}, {r["bonf_hi"]:+.4f}] | '
                     f'{r["dz"]:.2f} | {fmt_p(r["p"])} | {fmt_p(h)} | '
                     f'{fmt_p(r["welch_p"])} |')
            csv_rows.append({
                'comparison': name, 'eta': r['eta'],
                'pesr': r['a'], 'lstmae': r['b'],
                'diff_mean': r['mean'], 'diff_sd': r['sd'], 'diff_se': r['se'],
                'ci95_lo': r['lo'], 'ci95_hi': r['hi'],
                'bonf95_lo': r['bonf_lo'], 'bonf95_hi': r['bonf_hi'],
                'dz': r['dz'], 'p_raw': r['p'], 'p_holm': h,
                'p_welch': r['welch_p'],
            })
        # 跨 η：每个 seed 先在网格上平均（单次检验，无需校正）
        for label, etas in [('跨 η 平均（0–0.5）', ETAS),
                            ('跨 η 平均（0–0.2，去掉崩溃点）',
                             [e for e in ETAS if e != '0.5'])]:
            a = [np.mean([fa(e, s) for e in etas]) for s in SEEDS]
            b = [np.mean([fb(e, s) for e in etas]) for s in SEEDS]
            r = paired_row(a, b)
            L.append(f'| **{label}** | {np.mean(a):.4f} | {np.mean(b):.4f} | '
                     f'{r["mean"]:+.4f} | '
                     f'[{r["lo"]:+.4f}, {r["hi"]:+.4f}] | '
                     f'（单检验，同上） | {r["dz"]:.2f} | '
                     f'{fmt_p(r["p"])} | — | — |')
            csv_rows.append({
                'comparison': name, 'eta': label,
                'pesr': float(np.mean(a)), 'lstmae': float(np.mean(b)),
                'diff_mean': r['mean'], 'diff_sd': r['sd'], 'diff_se': r['se'],
                'ci95_lo': r['lo'], 'ci95_hi': r['hi'],
                'bonf95_lo': r['lo'], 'bonf95_hi': r['hi'],
                'dz': r['dz'], 'p_raw': r['p'], 'p_holm': '',
                'p_welch': '',
            })
        L.append('')

    L += ['> 解读限制：',
          '> 1. n=3 的 p 值只说明“在正态 + 配对差正态假设下”，'
          '不能替代效应量；Δ 的 CI 是更稳的表述。',
          '> 2. 6 个 η 不是独立样本，不能把 18 个 run 当 n=18 做检验'
          '（伪重复）。跨 η 汇总只做 seed 级平均后的 n=3 检验。',
          '> 3. 若审稿要求非参数或更稳的 p，需要把 seed 加到 ≥6'
          '（符号检验/Wilcoxon 在 n=6 才能到双侧 0.031）。',
          '> 4. 若要做窗口级检验，需要保存逐窗口分数并在时间块上做'
          'block bootstrap（point-adjust 使窗口不独立）。当前未落盘。',
          '> 5. `p ≥ 0.05` 只表示“没有足够证据拒绝”，不等于两模型相同；'
          '要主张等价需 TOST + 预设 margin。`p < 0.05` 也不表示差异大，'
          '应同时看 Δ 的 CI 与效应量。',
          '> 6. 建议主比较只保留两条：`PESR bf − LSTM bf` 与'
          ' `PESR POT① − LSTM POT③` 的跨 η 平均（0–0.2），各一次检验；'
          '逐 η 表作为探索性/敏感性分析，报 raw p + 单项 CI。']

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    out = os.path.join(ROOT, 'results', 'prd62_pesr_vs_lstmae_test.md')
    with open(out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    csv_out = os.path.join(ROOT, 'results', 'prd62_pesr_vs_lstmae_test.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
