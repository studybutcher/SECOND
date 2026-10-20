#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD7: 汇总 HAI 交叉泛化的四个格子（bf_search F1）。

读取 results/hai/ 下的
    HAI22.json, HAI23.json, HAI22to23.json, HAI23to22.json
打印 A→A / B→B / A→B / B→A 的 P/R/F1。

用法:
    python bench/summarize_hai_cross.py [results/hai]
"""
import glob
import json
import os
import sys

CELLS = [
    ('A→A', 'HAI22'),
    ('B→B', 'HAI23'),
    ('A→B', 'HAI22to23'),
    ('B→A', 'HAI23to22'),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join('results', 'hai')
    rows, missing = [], []
    for label, name in CELLS:
        p = os.path.join(out_dir, name + '.json')
        if not os.path.exists(p):
            missing.append(p)
            continue
        d = load(p)
        m = d['metrics']
        rows.append((label, name, d.get('test_dataset', d.get('dataset')),
                     m['precision'], m['recall'], m['f1'], m['threshold']))

    if not rows:
        print(f'[FAIL] {out_dir} 下没有可汇总的结果', file=sys.stderr)
        for p in missing:
            print(f'  missing: {p}', file=sys.stderr)
        return 1

    print(f'{"cell":6s} {"run":12s} {"test":8s} {"P":>8s} {"R":>8s} {"F1":>8s} {"th":>10s}')
    for label, name, test_ds, p, r, f1, th in rows:
        print(f'{label:6s} {name:12s} {str(test_ds):8s} '
              f'{p:8.4f} {r:8.4f} {f1:8.4f} {th:10.6f}')

    if 'A→A' in [r[0] for r in rows] and 'A→B' in [r[0] for r in rows]:
        d = {r[0]: r for r in rows}
        print(f'\nΔ_A = F1(A→A) - F1(A→B) = {d["A→A"][5] - d["A→B"][5]:+.4f}')
        if 'B→B' in d and 'B→A' in d:
            print(f'Δ_B = F1(B→B) - F1(B→A) = {d["B→B"][5] - d["B→A"][5]:+.4f}')

    if missing:
        print('\n[incomplete] 缺少:', file=sys.stderr)
        for p in missing:
            print('  ' + p, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
