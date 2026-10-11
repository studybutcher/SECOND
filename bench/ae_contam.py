#!/usr/bin/env python
"""PRD5.1 r3 · 结构化异常段注入（§3.1），注入到 `ae_data.py` 划出的**正常训练行**上。

复用 `bench/gen_contam.py` 的**长度计划与放置逻辑**（`plan_lengths` / `place_all` / `TYPES`），
保证：
  * ρ 的定义 = **被替换的训练行占比**（替换而非增补，行数不变）；
  * 四档 ρ **共享嵌套前缀**（`A_5% ⊂ A_10% ⊂ A_20%`，构造性成立）；
  * 段只落在**可注入块**（col0 std>0）内，只改 **col0**，二值指示列逐位不动。

与 `gen_contam.py` 的差别只有一处：这里作用在 **train_normal 子数组**上（val 已单独切出），
所以 val 天然不受污染（Gate B），scaler 也只在 train 上拟合过（§4.2）。

⚠️ 本脚本目前只支持 **stacked 布局（MSL/SMAP，col0=信号、1..K-1=二值指示列）**。
SWaT 需要 flat 布局的支持（注入连续通道）——见 PRD r3 §3.2，属 Phase 2。

用法：
    python bench/ae_contam.py --dataset MSL --window 70 --all --alpha 1.5 --seed 42
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.ae_data import load                                  # noqa: E402
from bench.gen_contam import plan_lengths, place_all, TYPES     # noqa: E402

OUT_DIR = os.path.join('artifacts', 'ae_mpa')

# 归一化后 col0 的 σ 低于这个值的块不算可注入。
# 理由：MSL 归一化后 col0 ∈ [0,1]，σ=1.4e-5 意味着整块几乎恒定（P-14），
# 而且 α·σ·ε 的增量会小到被 float32 舍入吃掉 —— 实测因此出现 9477 vs 9478 的 off-by-one。
# 这种块上的"异常"本来就不可辩，排除比容忍更诚实。
MIN_INJECT_SD = 1e-4


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_blocks(arr, meta_blocks):
    """把 split.json 的块元数据转成 place_all 需要的格式，并判定 injectable。

    `injectable = col0_std > MIN_INJECT_SD`（σ=0 的恒值块与 σ≈1e-5 的近恒值块都被排除）。
    """
    out = []
    for b in meta_blocks:
        seg = arr[b['off']:b['off'] + b['n']]
        sd0 = float(seg[:, 0].std())
        out.append(dict(name=b['name'], t_start=int(b['off']), t_end=int(b['off'] + b['n']),
                        length=int(b['n']), col0_std=sd0,
                        injectable=bool(sd0 > MIN_INJECT_SD)))
    return out


def inject_one(arr, blocks, placed, types, alpha, rng):
    """按给定段注入，返回新数组 + 段记录。

    三种类型都在 col0 上做（level_shift / amplitude_scaling 是改值，gaussian 是加噪），
    与 `gen_contam.py` 保持一致。
    """
    out = arr.copy()
    recs = []
    for j, (s, L, bi) in enumerate(placed):
        b = blocks[bi]
        ref = arr[b['t_start']:b['t_end'], 0]
        sd, mu = float(ref.std()), float(ref.mean())
        t = types[j]
        if t == 'level_shift':
            out[s:s + L, 0] = out[s:s + L, 0] + alpha * sd
        elif t == 'amplitude_scaling':
            out[s:s + L, 0] = mu + alpha * (out[s:s + L, 0] - mu)
        else:
            out[s:s + L, 0] = out[s:s + L, 0] + alpha * sd * rng.standard_normal(L)
        recs.append(dict(t_start=int(s), L=int(L), block=b['name'], block_idx=int(bi),
                         type=t, alpha=float(alpha), col0_std=sd, col0_mean=mu))
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window', type=int, default=70)
    ap.add_argument('--rho', type=float, default=0.20)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--rhos', default='0.05,0.10,0.20')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--alpha', type=float, default=1.5)
    ap.add_argument('--lbar-typ', type=float, default=583.0)
    ap.add_argument('--n-min', type=int, default=3)
    ap.add_argument('--out-dir', default=OUT_DIR)
    a = ap.parse_args()
    assert a.dataset in ('MSL', 'SMAP'), '本脚本只支持 stacked 布局（见文件头说明）'

    rhos = sorted([float(x) for x in a.rhos.split(',')] if a.all else [a.rho])
    arr, _, meta = load(a.dataset)
    N = arr.shape[0]
    blocks = build_blocks(arr, meta['blocks_train'])
    inj = [b['name'] for b in blocks if b['injectable']]
    print(f'{a.dataset}: N_train={N}  块={len(blocks)}  可注入={len(inj)}  '
          f'ρ={rhos}  seed={a.seed}  α={a.alpha}')

    len_rng = np.random.default_rng(a.seed)
    plan, targets, nseg = plan_lengths(N, rhos, a.lbar_typ, a.n_min, len_rng)
    placed = place_all(plan[rhos[-1]], blocks, np.random.default_rng(a.seed + 1))
    types = [str(t) for t in np.random.default_rng(a.seed + 2).choice(TYPES, size=len(placed))]
    print(f'  段数: ' + ', '.join(f'ρ={r:.0%}→{nseg[r]}' for r in rhos))

    os.makedirs(a.out_dir, exist_ok=True)
    sets = {}
    for r in rhos:
        k = len(plan[r])
        out, recs = inject_one(arr, blocks, placed[:k], types, a.alpha,
                               np.random.default_rng(a.seed + 3))
        sets[r] = {x['t_start'] for x in recs}

        # ── Gate A 的四项断言（与 gen_contam 一致）──────────────────────
        n_cont = int((out[:, 0] != arr[:, 0]).sum())
        # 允许 ≤1 行的 float32 舍入 no-op（增量小到加不上），但要如实记录、不静默
        short = targets[r] - n_cont
        assert short <= 1, f'ρ={r}: 污染行 {n_cont} != 目标 {targets[r]}（差 {short}）'
        if short:
            print(f'  [warn] ρ={r}: {short} 行的增量被 float32 舍入吃掉（no-op），已记账')
        bad = [c for c in range(1, arr.shape[1]) if not np.array_equal(out[:, c], arr[:, c])]
        assert not bad, f'ρ={r}: 指示列 {bad[:5]} 被改动'
        m = np.ones(N, bool)
        for x in recs:
            m[x['t_start']:x['t_start'] + x['L']] = False
        assert np.array_equal(out[m], arr[m]), f'ρ={r}: 非污染行不一致'

        # ρ_win：窗口级污染率（≠ ρ）
        win = np.zeros(N, bool)
        for x in recs:
            win[max(0, x['t_start'] - a.window + 1):min(N, x['t_start'] + x['L'])] = True
        rho_win = float(win[a.window:].mean())

        tag = f'{a.dataset}_train_rho{int(round(r * 100)):02d}_seed{a.seed}'
        np.save(f'{a.out_dir}/{tag}.npy', out.astype(np.float32))
        json.dump(dict(dataset=a.dataset, rho=r, seed=a.seed, alpha=a.alpha, window=a.window,
                       n_rows_total=N, n_contaminated=n_cont, n_contaminated_target=targets[r],
                       n_contaminated_shortfall=short, rho_win=rho_win,
                       n_segments=len(recs), mean_seg_len=float(np.mean([x['L'] for x in recs])),
                       lbar_typ=a.lbar_typ, n_min=a.n_min, layout='stacked', injection_col=0,
                       min_inject_sd=MIN_INJECT_SD,
                       injection_space='model_input_space (scaler fitted on train_normal only)',
                       val_untouched=True, blocks_injectable=inj, segments=recs,
                       clean_sha256=sha256(arr), out_sha256=sha256(out.astype(np.float32)),
                       gen_script='bench/ae_contam.py'),
                  open(f'{a.out_dir}/{tag}.json', 'w'), indent=1, ensure_ascii=False)
        print(f'  [OK] ρ={r:.0%}: {n_cont} 行 / {len(recs)} 段  均值段长 '
              f'{np.mean([x["L"] for x in recs]):.0f}  ρ_win={rho_win:.4f} '
              f'(ρ_win/ρ={rho_win / r:.3f})')

    # ── Gate A⑤：跨档嵌套 ────────────────────────────────────────────────
    for lo, hi in zip(rhos, rhos[1:]):
        assert sets[lo] <= sets[hi], f'[FAIL] 嵌套不成立：{lo} ⊄ {hi}'
    # ── Gate D：最轻档之外，ρ=0 未生成文件，故只需验证 clean 定义 ────────
    print(f'  [OK] 嵌套断言通过：' + ' ⊂ '.join(f'{r:.0%}' for r in rhos))
    print(f'  [OK] Gate D: ρ=0 即 `{a.dataset}_train_normal.npy`（不生成污染文件），'
          f'sha256={sha256(arr)[:16]}…')


if __name__ == '__main__':
    main()
