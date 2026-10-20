#!/usr/bin/env python
# coding=utf-8
"""比较 PESR η=0 的重复前向结果（同 ckpt、同 seed）。

读取：
  results/threshold_pesr_potcal/           原始一次
  results/threshold_pesr_repeat/rep{1..4}  重复四次
对每个 seed 打印 base_th、s=1.0 的 val/test/free F1、val-argmax 的
s* 与 test/free F1，并给出整份 JSON 的 md5（判断是否逐位一致）。

用法：
  python script/robust/compare_pesr_repeat.py
"""
import glob
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(ROOT, 'results/threshold_pesr_potcal')
REPS = sorted(glob.glob(os.path.join(ROOT, 'results/threshold_pesr_repeat/rep*')))
SEEDS = [42, 7, 17]
NOISE_TAG = '0'


def load(d, seed):
    p = os.path.join(d, f'pesr_MSL_noise{NOISE_TAG}_seed{seed}.json')
    if not os.path.exists(p):
        return None, None
    with open(p) as f:
        j = json.load(f)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()[:12]
    return j, h


def pick(j, scale):
    for r in j['scales']:
        if abs(r['scale'] - scale) < 1e-12:
            return r
    return None


def main():
    for seed in SEEDS:
        print(f'\n=== seed {seed} ===')
        d0 = BASE
        j, h = load(d0, seed)
        rows = [(os.path.basename(d0), j, h)]
        for d in REPS:
            j, h = load(d, seed)
            if j is not None:
                rows.append((os.path.basename(d), j, h))
        for name, j, h in rows:
            one = pick(j, 1.0)
            best = max(j['scales'], key=lambda r: r['val_f1'])
            print(f'{name:>10}  md5={h}  base_th={j["base_threshold_scale1"]:9.4f}  '
                  f'bf_test={j["bf_test"]["f1"]:.6f}  '
                  f's=1: test={one["test_f1"]:.6f} free={one["free_f1"]:.6f}  '
                  f'| argmax s={best["scale"]:.4f} test={best["test_f1"]:.6f} '
                  f'free={best["free_f1"]:.6f}')


if __name__ == '__main__':
    main()
