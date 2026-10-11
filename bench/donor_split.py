#!/usr/bin/env python
"""PRD5.2 r2 §2 · donor / 评测的**事件级**切分 + 训练集截断（预算由 donor 供给倒推）。

硬要求（PRD §1、§2）：

* **评测侧必须保留 ≥60% 的 test 异常点** ⇒ donor 上限 $A_{donor}\\le0.4A_{total}$；
* $\\rho\\cdot N_{tr}\\le A_{donor}$ ⇒ 由 ρ_max 倒推 $N_{tr}$（截断）；
* donor 与评测**按整段划分**，跨边界留 ≥W 隔离区；
* 同一份切分服务三档 ρ，且 **donor 是前缀嵌套**（5% ⊂ 10% ⊂ 20%）——按**起始位置**排序后取前缀。

**SWaT**：最长段（35,900 点 = 异常池 65.7%）判给**评测侧**（评测才有统计意义、donor 才有多样性）。
**MSL**：纵向拼接布局；**允许跨实体注入** —— 本实验的 AE/LSTM-AE 是**实体无关**的
（输入 `[W,55]` 里没有实体标识），故 donor 窗可以落在任何实体的训练行上。
（逐实体注入不可行：27 个实体里 8 个在 test 里**没有**异常，ρ=20% 时只有 4 个够用。）

产出（`artifacts/donor/`）：
    <ds>_split.json          事件表 + donor/eval 归属 + 预算记账 + 评测窗索引
    <ds>_train_normal.npy    截断后的正常训练行（已归一化）
    <ds>_val_normal.npy      固定正常验证行（用 train 的 scaler 变换）
    <ds>_test_norm.npy       已归一化、按 main.py 截断后的 test 行（供 F1/E_A 用）

用法：
    python bench/donor_split.py --dataset SWaT
    python bench/donor_split.py --dataset MSL
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import _clean                                  # noqa: E402
from bench.extract_repr import entity_blocks              # noqa: E402

RAW = '/data/sdc/yrlbkp/anomaly'
OUT = os.path.join('artifacts', 'donor')

# **小片断**（用户 2026-07-11 决定）：训练集固定用一小段，全体配置共享同一份 ⇒
# 欠拟合在所有臂上相同，**档间比较仍然有效**；绝对值写进 limitation。
# 好处：ρ·N_tr 小 ⇒ donor 需求小 ⇒ 评测侧几乎保住全部异常（消除"评测只有一个事件"）。
CFG = {
    'SWaT': dict(dim=51, window=35, n_tr=5000, val_frac=0.15),
    'MSL':  dict(dim=55, window=70, n_tr=5000, val_frac=0.15),
}
RHO_MAX = 0.20


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def segments_of(label):
    d = np.diff(np.concatenate([[0], label, [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(st, en)]


def assign(segs, window, n_tr, rho_max):
    """返回 (donor_segs, eval_segs, 记账)。规则见模块 docstring，全确定性、无 RNG。"""
    A_total = sum(e - s for s, e in segs)
    cap = 0.4 * A_total                      # donor 上限
    longest = max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
    side = {i: 'eval' for i in range(len(segs))}
    side[longest] = 'eval'                   # 最长段判给评测侧
    target = rho_max * n_tr
    acc = 0
    for i in sorted((i for i in range(len(segs)) if i != longest),
                    key=lambda i: segs[i][0]):
        L = segs[i][1] - segs[i][0]
        if acc >= target or acc + L > cap:
            break
        side[i] = 'donor'
        acc += L
    # 隔离区**后置检查**：与任何最终 eval 段距离 < W 的 donor 段降级为 eval。
    # （不能在选前缀时判：那时还没分配完，"其它段是 eval"是错的假设 —— 第一版就这么错的。）
    demoted = []
    for i in [i for i in range(len(segs)) if side[i] == 'donor']:
        s, e = segs[i]
        if any(not (e + window <= segs[j][0] or segs[j][1] + window <= s)
               for j in range(len(segs)) if j != i and side[j] == 'eval'):
            side[i] = 'eval'; demoted.append(i); acc -= e - s
    donor = [segs[i] for i in range(len(segs)) if side[i] == 'donor']
    ev = [segs[i] for i in range(len(segs)) if side[i] == 'eval']
    acc = dict(A_total=A_total, A_donor=int(acc), A_eval=int(A_total - acc),
               n_demoted=len(demoted),
               frac_eval=float((A_total - acc) / A_total), n_tr=int(n_tr),
               rho_max=rho_max, donor_cap=int(cap),
               max_rho_supported=float(acc / n_tr) if n_tr else 0.0,
               longest_seg_len=int(segs[longest][1] - segs[longest][0]),
               longest_side='eval')
    return donor, ev, acc, side


def build(dataset, out_dir=OUT, n_tr=None):
    c = CFG[dataset]
    K, W, N_TR = c['dim'], c['window'], (c['n_tr'] if n_tr is None else int(n_tr))
    raw_tr = np.load(f'{RAW}/{dataset}/{dataset}_train.npy').reshape(-1, K).astype(np.float64)
    raw_te = np.load(f'{RAW}/{dataset}/{dataset}_test.npy').reshape(-1, K).astype(np.float64)
    lab = (np.load(f'{RAW}/{dataset}/{dataset}_test_label.npy').reshape(-1) > 0.1).astype(int)

    segs = segments_of(lab)
    donor_segs, eval_segs, acc, side = assign(segs, W, N_TR, RHO_MAX)
    # **N_tr 由实际到手的 donor 倒推**（而不是先写死 N_tr 再希望 donor 够）：
    N_TR_eff = min(N_TR, int(acc['A_donor'] / RHO_MAX))
    acc['n_tr'] = int(N_TR_eff)
    acc['n_tr_target'] = int(N_TR)
    print(f'{dataset}: 段 {len(segs)}  A_total={acc["A_total"]}  donor={acc["A_donor"]} 点 / '
          f'{len(donor_segs)} 段  eval={acc["A_eval"]} 点 (保留 {100*acc["frac_eval"]:.1f}%)'
          + (f'  降级 {acc["n_demoted"]} 段' if acc['n_demoted'] else ''))
    if N_TR_eff < N_TR:
        print(f'  [调整] donor 只有 {acc["A_donor"]} 点，ρ_max={RHO_MAX:.0%} 下 '
              f'N_tr 由 {N_TR} 降到 {N_TR_eff}')
    N_TR = N_TR_eff
    print(f'  ρ_max={RHO_MAX:.0%} 需 {RHO_MAX*N_TR:.0f} 点 ≤ donor {acc["A_donor"]} ✓')

    # ── 训练集截断（逐块按同一比例取前段），val 取各块尾部 ────────────
    tr_parts, va_parts, tr_meta, va_meta = [], [], [], []
    off_t = off_v = 0
    keep = N_TR / (len(raw_tr) - 0)
    for name, s, e in entity_blocks(dataset, 'train', len(raw_tr)):
        n = e - s
        cut = s + max(int(n * keep), W + 2)             # train = [s, cut)
        # val 取块尾 15%，但**封顶 5000 行**（单实体下 15% = 74k 行，每个 epoch 都评估太浪费）
        t_start = e - min(max(int(n * 0.15), W + 2), 5000)
        t_start = max(t_start, cut + W)                 # 隔离区 ≥W
        if t_start + W + 2 > e:
            continue
        a_t, a_v = raw_tr[s:cut], raw_tr[t_start:e]
        if len(a_t) < W + 2 or len(a_v) < W + 2:
            continue
        tr_parts.append(a_t); va_parts.append(a_v)
        tr_meta.append(dict(name=name, off=int(off_t), n=int(len(a_t)), src_start=int(s)))
        va_meta.append(dict(name=name, off=int(off_v), n=int(len(a_v)), src_start=int(t_start)))
        off_t += len(a_t); off_v += len(a_v)

    tr_raw, va_raw = np.concatenate(tr_parts), np.concatenate(va_parts)
    sc = MinMaxScaler().fit(_clean(tr_raw))
    tr = sc.transform(_clean(tr_raw)).astype(np.float32)
    va = sc.transform(_clean(va_raw)).astype(np.float32)
    te = sc.transform(_clean(raw_te)).astype(np.float32)      # test 用 train 的 scaler（r2 统一口径）

    # ── 评测窗：test 里**不与任何 donor 段重叠**的窗 ──────────────────
    ov = np.zeros(len(raw_te), bool)
    for s, e in donor_segs:
        ov[s:e] = True
    idx = np.arange(W)[None, :] + np.arange(len(raw_te) - W)[:, None]
    ok = ~ov[idx].any(axis=1)
    lab_w = lab[idx].max(axis=1)
    print(f'  评测窗 {int(ok.sum())} / {len(ok)}（排除 donor 重叠 {int((~ok).sum())} 窗）'
          f'  其中异常窗 {int((lab_w[ok] == 1).sum())}  (异常率 {lab_w[ok].mean():.4f})')

    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/{dataset}_train_normal.npy', tr)
    np.save(f'{out_dir}/{dataset}_val_normal.npy', va)
    np.save(f'{out_dir}/{dataset}_test_norm.npy', te)
    meta = dict(dataset=dataset, dim=K, window=W, n_tr_target=N_TR,
                n_train_rows=int(len(tr)), n_val_rows=int(len(va)),
                blocks_train=tr_meta, blocks_val=va_meta,
                donor_segs=[[int(s), int(e)] for s, e in donor_segs],
                eval_segs=[[int(s), int(e)] for s, e in eval_segs],
                events=[dict(id=i, start=int(s), end=int(e), length=int(e - s), side=side[i])
                        for i, (s, e) in enumerate(segs)],
                budget=acc, guard_rows=W, rho_max=RHO_MAX,
                scaler_fit_on='train_normal only', scaler_min=sc.min_.tolist(),
                scaler_scale=sc.scale_.tolist(),
                eval_win_idx=np.where(ok)[0].tolist(), eval_win_label=lab_w[ok].tolist(),
                sha256_train=sha256(tr), sha256_val=sha256(va))
    with open(f'{out_dir}/{dataset}_split.json', 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f'  saved: {out_dir}/{dataset}_split.json  (train {tr.shape}, val {va.shape})')
    return meta


def load_split(dataset, out_dir=OUT):
    with open(f'{out_dir}/{dataset}_split.json') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='SWaT', choices=list(CFG))
    ap.add_argument('--out-dir', default=OUT)
    ap.add_argument('--n-tr', type=int, default=None,
                    help='覆盖默认训练行数（片断大小）。调小 ⇒ donor 需求小 ⇒ 评测侧保留更多异常，'
                         '但太小会欠训练（实测欠训练时 E_A/E_N 会虚高，见 PRD §0）')
    a = ap.parse_args()
    build(a.dataset, a.out_dir, a.n_tr)


if __name__ == '__main__':
    main()
