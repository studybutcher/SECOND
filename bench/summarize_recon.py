#!/usr/bin/env python
"""PRD5.2 r2 §4.4 · 主表：AE 与 LSTM-AE 在真实异常注入下的 **F1** 与 **MSE**。

未过 Gate N（重建近似常量）的 run 的 MSE 列标 `—` 并在表注里说明 —— 不是"结果差"，是"没测到东西"。

用法：python bench/summarize_recon.py --dataset SWaT
"""
import argparse
import glob
import json
import os
import statistics as st

RANK = {'ae': 0, 'lstmae': 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='SWaT')
    ap.add_argument('--glob', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    pat = a.glob or f'results/recon/*_{a.dataset}_rho*_seed*.json'
    out = a.out or f'results/recon/summary_{a.dataset}.md'

    runs = {}
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p))
        runs.setdefault((d['model'], d['rho']), {})[d['seed']] = d
    if not runs:
        raise SystemExit(f'[FAIL] 没匹配到 {pat}')

    def agg(rs, key, ok_only=False):
        v = [r[key] for r in rs if (r['gates']['N']['ok'] or not ok_only)]
        return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else (None, None, 0)

    L = [f'# PRD5.2 r2 · 真实异常注入：AE vs LSTM-AE 的 F1 与 MSE（{a.dataset}）', '',
         '| 模型 | ρ | **F1**（项目协议） | F1(未调整) | **E_N** | E_A | **E_A/E_N** | AUROC | n | Gate N |',
         '|' + '---|' * 10]
    for (m, rho), sd in sorted(runs.items(), key=lambda kv: (RANK.get(kv[0][0], 9), kv[0][1])):
        rs = list(sd.values())
        ok = all(r['gates']['N']['ok'] for r in rs)
        f1 = agg(rs, 'F1'); f1u = agg(rs, 'F1_unadj')
        en = agg(rs, 'E_N'); ea = agg(rs, 'E_A'); rt = agg(rs, 'ratio')
        au = agg(rs, 'AUROC')
        def f(t, nd=4):
            return '—' if t[0] is None else f'{t[0]:.{nd}f}' + (f' ± {t[1]:.{nd}f}' if t[2] > 1 else '')
        L.append(f"| {m} | {rho:.0%} | {f(f1)} | {f(f1u)} | "
                 + (f'{f(en)} | {f(ea)} | **{f(rt,3)}** | {f(au)} | {len(rs)} | '
                    f'{"✅" if ok else "❌"} |' if ok else '— | — | — | — | '
                    f'{len(rs)} | ❌ 常量 |'))
    L += ['', '> `E_A/E_N` 是**唯一可跨模型比较**的量（各自归一化不同，绝对 MSE 不可比）。',
          '> Gate N ❌ 的行表示该 run 的重建对"真实 vs 全零"输入几乎不变 ⇒ **MSE 列不可解读**，',
          '> 必须剔除（见 PRD §8：MPA+AE 的 `E_A/E_N=30.9` 正是这种假象）。',
          '> F1 是 point-adjust + oracle 百分位阈值口径，**只作参照**，不用于比较。']
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nsaved: {out}')


if __name__ == '__main__':
    main()
