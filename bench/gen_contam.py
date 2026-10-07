#!/usr/bin/env python
"""污染训练集生成器（PRD4text · MSL）

把「真实异常混进正常训练集」物化成确定的 `.npy` 字节流 + manifest，供
`main.py --contam-file` 加载。设计依据见 `docs/PRD4text_定量分析_r2.md` §5/§6。

## 关键设计决定（都有理由，不是随手选的）

1. **注入空间 = 模型输入空间**（§6.3 方案 (A)）。由 §6.1 的恒等式，(A) 与「原始空间
   注入再归一化」对扰动本身**完全等价**（实测 max|A−B|=2.2e-16），但 (A) 不需要原始
   物理量、也不会让 σ 与训练数据不同源。流程：读 clean → `preprocess`(fit on clean)
   → 注入 → 存盘。因此 **`main.py --contam-file` 不得再 preprocess 一次**。
   α 的口径是「该通道在模型输入空间 σ 的倍数」。

2. **替换而非增补**（§5.2）：训练集总长恒为 N，ρ 只是「被改写的行数占比」。

3. **MSL 是 27 个通道纵向拼接**（`bench/channel_split.py` 的 stacked 布局）：每块
   第 0 列是信号、第 1..54 列是二值指示列。所以
   - 段只落在**单个通道块内**、只落在**可注入块**（col0 非常数）；
   - 段**只改第 0 列**（往指示列注入 ασ 会产生非 0/1 的非法状态）；
   - 「段内随机抽 1~3 个连续通道」这套多通道设计对 MSL 无意义，已删除。

4. **嵌套与精确性同时满足**（§5.3 + §5.5 断言 1，r2 没处理两者的冲突）：
   只按一种方式做——**先把最大 ρ 的所有段一次性定长定址，再取前缀**。
   - 长度：各 ρ 档的累计目标 `C(ρ)=round(ρN)`；相邻目标之间放
     `n_seg(ρ)−n_seg(ρ_prev)` 段，该组长度重标定到恰好填满 `C(ρ)−C(ρ_prev)`。
     于是 `plan[ρ]` 是 `plan[ρ_max]` 的**长度前缀**，且前缀和精确等于 round(ρN)。
   - 位置与类型：对 `plan[ρ_max]` 只做**一次**放置与抽类型，小 ρ 直接取前 k 个。
   两层前缀合起来，`A_{1%} ⊂ A_{5%} ⊂ A_{10%} ⊂ A_{20%}` 构造性成立。

5. **段数上限**：r2 的 §5.3 公式与它自己的冻结检查表（段数 ≤20）在 SWaT 上自相矛盾；
   MSL 上 `N_min=3, L̄_typ=583` 是唯一让 4 个 ρ 档全过的组合（已验证），故取为默认。

用法：
    python bench/gen_contam.py --all --alpha 1.5          # 四个 ρ 档，共享嵌套
    python bench/gen_contam.py --rho 0.20 --alpha 1.5     # 单档
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_dim, preprocess                    # noqa: E402
from bench.channel_split import stacked_blocks                # noqa: E402

RAW_DIR = '/data/sdc/yrlbkp/anomaly/data/SMAP_MSL'
TYPES = ('level_shift', 'amplitude_scaling', 'perturbation')
MULT = (0.33, 1.0, 3.0)
MULT_P = np.array((5, 3, 2)) / 10.0
WINDOW = 70          # MSL 的 W，用于统计 ρ_win


def load_clean(dataset):
    """读 clean train → 模型输入空间（preprocess，scaler 拟合于 clean）。"""
    K = get_data_dim(dataset)
    raw = np.load(f'/data/sdc/yrlbkp/anomaly/{dataset}/{dataset}_train.npy').reshape(-1, K)
    return preprocess(raw.astype(np.float64))


def plan_lengths(N, rhos, lbar, n_min, rng):
    """→ {rho: [L,...]}，plan[ρ] 是 plan[max ρ] 的长度前缀，且 sum == round(ρN)。"""
    targets = {r: int(round(r * N)) for r in rhos}
    nseg = {}
    prev_k = 0
    for r in sorted(rhos):
        k = max(n_min, int(round(r * N / lbar)))
        k = max(k, prev_k + 1)                      # 必须严格递增，否则取前缀会倒退
        nseg[r] = k
        prev_k = k

    plan, full, prev_r, prev_target = {}, [], None, 0
    for r in sorted(rhos):
        k_new = nseg[r] - (nseg[prev_r] if prev_r is not None else 0)
        span = targets[r] - prev_target
        # 按 5:3:2 抽**相对权重**，再归一化到总量恰好 = span。
        # 不能写成「前 k-1 段取整、最后一段补差」——权重抽到大值时差为负，
        # max(1, …) 会把它夹成 1，总量反被放大（曾出现 3 段合计 778 ≠ 583）。
        w = rng.choice(MULT, size=k_new, p=MULT_P)
        if span < k_new:                       # 退化：段数比行数还多 -> 每段 1 行
            L = np.ones(k_new, int)
            L[0] += span - k_new
        else:
            L = np.maximum(1, np.floor(w / w.sum() * span).astype(int))
            resid = span - int(L.sum())
            # 残差（可正可负）加在最长的一段上；最长段必然够减，不会变负
            L[int(np.argmax(L))] += resid
        assert L.min() >= 1 and int(L.sum()) == span, \
            f'ρ={r}: 段长和 {int(L.sum())} != 目标 {span}'
        full.extend(int(x) for x in L)
        prev_r, prev_target = r, targets[r]
        plan[r] = list(full)                        # 前缀
    assert len({len(plan[r]) for r in rhos}) == len(rhos)
    return plan, targets, nseg


def place_all(lengths, blocks, rng):
    """一次性放置全部段：不跨块、不重叠、只落在可注入块内。"""
    placed = []
    for L in lengths:
        cand = [i for i, b in enumerate(blocks) if b['injectable'] and b['length'] >= L + 2]
        if not cand:
            longest = max(b['length'] for b in blocks if b['injectable'])
            raise SystemExit(f'[FAIL] 无可容纳 L={L} 的可注入块（最长 {longest}）'
                             f'—— 调小 --lbar-typ')
        for _ in range(500):
            i = int(rng.choice(cand))
            b = blocks[i]
            s = int(rng.integers(b['t_start'], b['t_end'] - L))
            if all(s + L + 1 < t or t + l + 1 < s for t, l, _ in placed):
                placed.append((s, L, i))
                break
        else:
            raise SystemExit('[FAIL] 500 次放不下（段太多/太重）—— 调小 --lbar-typ')
    return placed


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--rho', type=float, default=0.20)
    ap.add_argument('--all', action='store_true', help='生成 --rhos 的全部档（共享嵌套）')
    ap.add_argument('--rhos', default='0.01,0.05,0.10,0.20')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--lbar-typ', type=float, default=583.0)
    ap.add_argument('--n-min', type=int, default=3)
    ap.add_argument('--out-dir', default='artifacts/contam')
    a = ap.parse_args()

    rhos = [float(x) for x in a.rhos.split(',')] if a.all else [a.rho]
    rhos = sorted(rhos)
    arr = load_clean(a.dataset)
    N = arr.shape[0]
    blk = stacked_blocks(a.dataset, RAW_DIR)
    blocks = blk['blocks']
    print(f'{a.dataset}: N={N}  通道块={len(blocks)}  可注入={len(blk["injectable_blocks"])}  '
          f'ρ={rhos}  seed={a.seed}  α={a.alpha}')

    # ── 长度计划 + 一次性放置（嵌套的两个前缀层）────────────────────────
    len_rng = np.random.default_rng(a.seed)
    plan, targets, nseg = plan_lengths(N, rhos, a.lbar_typ, a.n_min, len_rng)
    placed = place_all(plan[rhos[-1]], blocks, np.random.default_rng(a.seed + 1))
    type_rng = np.random.default_rng(a.seed + 2)
    types = [str(t) for t in type_rng.choice(TYPES, size=len(placed))]

    for r in rhos:
        k = len(plan[r])
        print(f'  ρ={r:.0%}: {nseg[r]} 段, 目标 {targets[r]} 行, 均值长 '
              f'{np.mean(plan[r]):.0f}')

    os.makedirs(a.out_dir, exist_ok=True)
    written = []
    for r in rhos:
        k = len(plan[r])
        recs = [dict(t_start=int(s), L=int(L), block=blocks[i]['name'], block_idx=int(i),
                     type=types[j], alpha=float(a.alpha))
                for j, (s, L, i) in enumerate(placed[:k])]
        out = arr.copy()
        for x in recs:
            b = blocks[x['block_idx']]
            ref = arr[b['t_start']:b['t_end'], 0]           # σ/μ 用该段所属块的 col0
            sd, mu = float(ref.std()), float(ref.mean())
            s, L = x['t_start'], x['L']
            if x['type'] == 'level_shift':
                out[s:s + L, 0] = out[s:s + L, 0] + a.alpha * sd
            elif x['type'] == 'amplitude_scaling':
                out[s:s + L, 0] = mu + a.alpha * (out[s:s + L, 0] - mu)
            else:
                out[s:s + L, 0] = out[s:s + L, 0] + \
                    a.alpha * sd * len_rng.standard_normal(L)

        # ── 断言（§5.5）──────────────────────────────────────────────
        n_cont = int((out[:, 0] != arr[:, 0]).sum())
        assert n_cont == targets[r], f'ρ={r}: 污染行数 {n_cont} != {targets[r]}'
        bad = [c for c in range(1, arr.shape[1]) if not np.array_equal(out[:, c], arr[:, c])]
        assert not bad, f'ρ={r}: 指示列 {bad[:5]} 被改动'
        mask = np.ones(N, bool)
        for x in recs:
            mask[x['t_start']:x['t_start'] + x['L']] = False
        assert np.array_equal(out[mask], arr[mask]), f'ρ={r}: 非污染行不一致'

        # ── ρ_win（§5.6，直接统计而非公式估算）───────────────────────
        win = np.zeros(N, bool)
        for x in recs:
            win[max(0, x['t_start'] - WINDOW + 1):min(N, x['t_start'] + x['L'])] = True
        rho_win = float(win[WINDOW:].mean())

        tag = f'{a.dataset}_rho{int(round(r*100)):02d}_seed{a.seed}'
        saved = out.astype(np.float32)
        np.save(os.path.join(a.out_dir, tag + '.npy'), saved)
        man = dict(dataset=a.dataset, rho=r, seed=a.seed, alpha=a.alpha,
                   n_rows_total=N, n_contaminated=n_cont, rho_win=rho_win,
                   n_segments=len(recs), mean_seg_len=float(np.mean([x['L'] for x in recs])),
                   lbar_typ=a.lbar_typ, n_min=a.n_min, window=WINDOW,
                   injection_space='model_input_space',
                   injection_note='preprocess 已在本脚本内完成；main.py --contam-file 不得再 preprocess',
                   injection_col=0, blocks_injectable=blk['injectable_blocks'],
                   segments=recs, nested_key=f'{a.dataset}_s{a.seed}_v1',
                   clean_sha256=sha256(arr), out_sha256=sha256(saved),
                   gen_script='bench/gen_contam.py')
        with open(os.path.join(a.out_dir, tag + '.json'), 'w') as f:
            json.dump(man, f, indent=1, ensure_ascii=False)
        written.append(tag)
        print(f'  [OK] ρ={r:.0%}  {n_cont} 行 / {len(recs)} 段  均值段长 '
              f'{np.mean([x["L"] for x in recs]):.0f}  ρ_win={rho_win:.4f} '
              f'(/ρ = {rho_win/r:.3f})  -> {tag}.npy')

    # ── 跨档嵌套断言（§5.5 断言 2）──────────────────────────────────
    sets = {}
    for r in rhos:
        sets[r] = {x['t_start'] for x in json.load(
            open(os.path.join(a.out_dir, f'{a.dataset}_rho{int(round(r*100)):02d}_seed{a.seed}.json')))
            ['segments']}
    for lo, hi in zip(rhos, rhos[1:]):
        assert sets[lo] <= sets[hi], f'[FAIL] 嵌套不成立：{lo} ⊄ {hi}'
    print(f'  [OK] 嵌套断言通过：' + ' ⊂ '.join(f'{r:.0%}' for r in rhos))


if __name__ == '__main__':
    main()
