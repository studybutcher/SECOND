#!/usr/bin/env python
"""PRD5.4 · 方案 B：**局部损害**分析 —— 按"窗口到最近注入段的距离"分层看正常窗重建误差。

**动机**：全局平均 $E_N$ 会被 95% 未污染的正常窗稀释。若"污染损害正常模式学习"真存在，
它最可能先出现在**贴着注入段的正常窗**上。

**设计要点（扣掉混淆）**：每个距离桶里的窗口，同时用 **ρ 模型**与**同 seed 的 ρ=0 模型**
各算一遍 MSE，报**桶内配对差 $\Delta$ = MSE(ρ) − MSE(0)**。
这样"贴着段的窗口本来就难/易修"这个与污染无关的窗口属性被扣掉。

**距离**定在**原始 train 数组坐标**（段在 train_normal 子数组里的下标先换算回原坐标）。

两组候选窗各报：
  * `val`   —— 实体后段 15%，**held-out**
  * `train` —— 污染数组里**未被注入**的正常窗（fit 口径，但紧邻注入段、数量足）

**桶内 n 必须一起报**：若 val 的近段桶太小，结论只能靠 train 那组。

用法：python bench/local_damage.py --rhos 0.05 0.10 0.20 --seeds 42 7 17 0 1 2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.ae_data import load, windows_from                    # noqa: E402
from bench import ae_baseline as B                              # noqa: E402

ART = os.path.join('artifacts', 'ae_mpa')
CKPT = os.path.join('checkpoint', 'ae_mpa')
#        lo     hi     label
BUCKETS = [(None, 0.0, 'd=0 重叠/紧邻'), (0.0, 1.0, '(0,1]W'),
           (1.0, 2.0, '(1,2]W'), (2.0, 5.0, '(2,5]W'), (5.0, None, '>5W')]


def to_orig(ridx, eid, blocks):
    """子数组行号 → 原始 train 数组行号。"""
    off = np.array([b['off'] for b in blocks])
    src = np.array([b['src_start'] for b in blocks])
    return src[eid] + (ridx - off[eid])


def seg_orig(man, blocks_train):
    """⚠️ `segments[].t_start` 是**拼接后 train_normal 数组**的全局下标（= 块元数据的 off
    基准），不是块内偏移。映射回原始数组要减掉 off，否则会多加一个 off、
    把段挪到别的块里去（这正是第一版把 val 打出假 d=0 桶的原因）。"""
    out = []
    for s in man['segments']:
        b = blocks_train[s['block_idx']]
        a = b['src_start'] + (s['t_start'] - b['off'])
        assert b['src_start'] <= a and a + s['L'] <= b['src_start'] + b['n'], \
            f"段 {s['t_start']} 映射后越出块 {b['name']} 的范围"
        out.append((a, a + s['L']))
    return out


def gap(a, w, segs):
    """窗口 [a, a+w) 到最近段的行距离；重叠或相接 = 0。"""
    return min(0 if (a < e and s < a + w) else
               (a - e if a >= e else s - (a + w)) for s, e in segs)


@torch.no_grad()
def per_win_mse(model, win, batch=512, dev='cuda'):
    model.eval()
    out = []
    for i in range(0, len(win), batch):
        x = torch.as_tensor(win[i:i + batch], device=dev)
        r = model(x)[0]
        out.append(((r - x) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out)


def load_ae(tag, W, C, dev='cuda'):
    p = f'{CKPT}/{tag}.pt'
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    m = B.build('ae', W, C, latent=256).to(dev)
    m.load_state_dict(torch.load(p, map_location='cpu'))
    return m.eval()


def analyse(dataset, rho, seed, W, C, dev='cuda'):
    _, val_arr, meta = load(dataset)
    ri = int(round(rho * 100))
    man = json.load(open(f'{ART}/{dataset}_train_rho{ri:02d}_seed{seed}.json'))
    tr_arr = np.load(f'{ART}/{dataset}_train_rho{ri:02d}_seed{seed}.npy')
    segs = seg_orig(man, meta['blocks_train'])

    cand = {}
    xv, ev, rv = windows_from(val_arr, meta['blocks_val'], W)
    cand['val'] = (xv, to_orig(rv, ev, meta['blocks_val']))
    xt, et, rt = windows_from(tr_arr, meta['blocks_train'], W)
    ot = to_orig(rt, et, meta['blocks_train'])
    g_t = np.array([gap(o, W, segs) for o in ot])
    m = g_t > 0                                   # 未注入的正常窗
    cand['train'] = (xt[m], ot[m])

    m_rho = load_ae(f'ae_{dataset}_rho{ri:02d}_seed{seed}', W, C, dev)
    m_0 = load_ae(f'ae_{dataset}_rho00_seed{seed}', W, C, dev)

    # 桶边界向量化：(-inf,0], (0,1], (1,2], (2,5], (5,inf)
    edges = [-np.inf, 0.0, 1.0, 2.0, 5.0, np.inf]

    rows = []
    for name, (win, orig) in cand.items():
        d = np.array([gap(o, W, segs) for o in orig]) / W
        bi = np.digitize(d, edges[1:-1], right=True)          # 0..4
        mr = per_win_mse(m_rho, win, dev=dev)
        m0 = per_win_mse(m_0, win, dev=dev)
        for i, (_, _, label) in enumerate(BUCKETS):
            sel = bi == i
            n = int(sel.sum())
            rows.append(dict(set=name, bucket=label, n=n,
                             mse_rho=float(mr[sel].mean()) if n else None,
                             mse_0=float(m0[sel].mean()) if n else None,
                             delta=float(mr[sel].mean() - m0[sel].mean()) if n else None,
                             delta_rel=float((mr[sel].mean() - m0[sel].mean()) / m0[sel].mean())
                             if n else None))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window', type=int, default=70)
    ap.add_argument('--channel', type=int, default=55)
    ap.add_argument('--rhos', nargs='+', type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 17, 0, 1, 2])
    ap.add_argument('--out', default='results/ae_mpa/local_damage.json')
    a = ap.parse_args()

    out = {}
    for rho in a.rhos:
        for seed in a.seeds:
            try:
                r = analyse(a.dataset, rho, seed, a.window, a.channel)
            except FileNotFoundError as e:
                print(f'  跳过 ρ={rho:.2f} seed={seed}: {os.path.basename(str(e))}')
                continue
            out[f'{rho}_{seed}'] = r
            v = [x for x in r if x['set'] == 'val']
            print(f"  ρ={rho:.2f} seed={seed:<3} val n: "
                  + '  '.join(f"{x['bucket']}={x['n']}" for x in v))
        print()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('saved:', a.out)


if __name__ == '__main__':
    main()
