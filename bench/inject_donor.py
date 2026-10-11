#!/usr/bin/env python
"""PRD5.2 r2 §2.3 · 把 **test 的真实异常片段**按行替换进正常训练集。

与 `gen_contam.py`（合成段）的关键差别：**整行替换成真实异常行** —— 不做 ασ 扰动、
不限定只改 col0（因为注入的是真实观测，指示列/离散列也应当随之而来）。

嵌套：一次性为 ρ_max 定址 → 各档取前缀 ⇒ `A_5% ⊂ A_10% ⊂ A_20%` 构造性成立。

产出：`artifacts/donor/<ds>_train_rho{05,10,20}_seed{S}.npy` + manifest

用法：python bench/inject_donor.py --dataset SWaT --seed 42
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.donor_split import load_split, OUT                      # noqa: E402

RHOS = [0.05, 0.10, 0.20]


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def place(n_rows, donor_segs, rho_max, rng):
    """为 ρ_max 一次性定址：[(target_start, length, donor_start)]。

    候选起点用 `rng.permutation` 打乱后线性扫，取第一个放得下的 —— 确定性、避免
    "随机重试 2000 次失败"那种脆弱写法。嵌套由"按 donor 段起始序 + 前缀"保证。
    """
    need = int(rho_max * n_rows)
    used = np.zeros(n_rows, bool)
    out, acc = [], 0
    for s, e in donor_segs:
        if acc >= need:
            break
        L = e - s
        if L >= n_rows:
            raise SystemExit(f'[FAIL] donor 段长 {L} ≥ 训练行数 {n_rows}')
        # 用卷积一次性找出所有「连续 L 行空闲」的起点（C 层，避免 Python 逐个试）
        free = np.convolve(used.astype(np.int32), np.ones(L, np.int32), 'valid') == 0
        cand = np.flatnonzero(free)
        if len(cand) == 0:
            raise SystemExit(f'[FAIL] 放不下 len={L} 的段（已用 {int(used.sum())}/{n_rows}）'
                             f'—— 调小 ρ_max 或加长训练集')
        t = int(cand[rng.integers(len(cand))])
        used[t:t + L] = True
        out.append((t, L, s))
        acc += L
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='SWaT')
    ap.add_argument('--seed', type=int, default=42)
    # ⚠️ 用 str 收、自己解析：`--rhos "$RHOS"`（带引号）时 argparse 会把整串当成**一个**值，
    #    配 type=float 直接报 invalid float。这里同时兼容 "0.05 0.10"、"0.05,0.10"、分开传三种写法。
    ap.add_argument('--rhos', nargs='+', default=[str(r) for r in RHOS])
    ap.add_argument('--out-dir', default=OUT)
    a = ap.parse_args()
    a.rhos = sorted({float(x) for tok in a.rhos
                     for x in str(tok).replace(',', ' ').replace('"', '').replace("'", '').split()
                     if x})
    a.rhos = [r for r in a.rhos if r > 0]          # ρ=0 就是干净的 train_normal，不生成文件
    assert a.rhos, '--rhos 里没有 >0 的档'

    meta = load_split(a.dataset, a.out_dir)
    tr = np.load(f'{a.out_dir}/{a.dataset}_train_normal.npy')
    te = np.load(f'{a.out_dir}/{a.dataset}_test_norm.npy')
    N, W = len(tr), meta['window']
    donor_segs = [tuple(x) for x in meta['donor_segs']]
    assert donor_segs == sorted(donor_segs), 'donor 段必须按起始位置有序（嵌套依赖它）'
    placed = place(N, donor_segs, meta['rho_max'], np.random.default_rng(a.seed))
    print(f'{a.dataset} N_tr={N}  donor 段={len(donor_segs)}  定址 {len(placed)} 处  '
          f'（ρ_max={meta["rho_max"]:.0%} 累计 {sum(p[1] for p in placed)} 行）')

    for rho in sorted(a.rhos):
        target = int(rho * N)
        recs, acc = [], 0
        for t, L, ds in placed:                       # 前缀嵌套
            if acc >= target:
                break
            take = min(L, target - acc)               # **截断最后一段** ⇒ 实际 ρ 精确等于名义值
            recs.append((t, take, ds)); acc += take
        out = tr.copy()
        for t, L, ds in recs:
            out[t:t + L] = te[ds:ds + L]

        # ── 断言 ─────────────────────────────────────────────────────
        inj = np.zeros(N, bool)
        for t, L, _ in recs:
            inj[t:t + L] = True
        n_inj = int(inj.sum())
        assert n_inj == acc, f'注入行 {n_inj} != 累计 {acc}'
        assert np.array_equal(out[~inj], tr[~inj]), '非注入行不一致'
        for t, L, ds in recs:                          # 正检：donor 内容真的落进去了
            assert np.array_equal(out[t:t + L], te[ds:ds + L]), 'donor 内容未落位'

        win = np.zeros(N, bool)
        for t, L, _ in recs:
            win[max(0, t - W + 1):min(N, t + L)] = True
        rho_win = float(win[W:].mean())

        tag = f'{a.dataset}_train_rho{int(round(rho*100)):02d}_seed{a.seed}'
        np.save(f'{a.out_dir}/{tag}.npy', out)
        json.dump(dict(dataset=a.dataset, rho_nominal=rho, rho_achieved=n_inj / N,
                       seed=a.seed, n_rows=N, n_injected=n_inj, rho_win=rho_win,
                       n_placements=len(recs),
                       donor_segments_used=[[int(t), int(L), int(ds)] for t, L, ds in recs],
                       donor_seg_lengths=[int(L) for _, L, _ in recs],
                       eval_segs=meta['eval_segs'], budget=meta['budget'],
                       clean_sha256=sha256(tr), out_sha256=sha256(out),
                       gen_script='bench/inject_donor.py'),
                  open(f'{a.out_dir}/{tag}.json', 'w'), indent=1, ensure_ascii=False)
        print(f'  [OK] ρ={rho:.0%}: 注 {n_inj} 行（实际 {100*n_inj/N:.2f}%）/ {len(recs)} 段  '
              f'ρ_win={rho_win:.4f}  → {tag}.npy')

    # 嵌套断言
    sets, prev = [], None
    for rho in sorted(a.rhos):
        m = json.load(open(f'{a.out_dir}/{a.dataset}_train_rho{int(round(rho*100)):02d}'
                           f'_seed{a.seed}.json'))
        s = set()
        for t, L, _ in m['donor_segments_used']:
            s |= set(range(t, t + L))
        sets.append(s)
    for lo, hi in zip(sets, sets[1:]):
        assert lo <= hi, '[FAIL] 嵌套不成立'
    print('  [OK] 嵌套断言通过：' + ' ⊂ '.join(f'{r:.0%}' for r in sorted(a.rhos)))


if __name__ == '__main__':
    main()
