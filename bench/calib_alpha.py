#!/usr/bin/env python
"""α 标定（PRD4text §5.4）

给定一个**干净训练**的模型，调整污染强度 α，使注入段的 ≥ 目标比例的时间戳
被该模型判为异常。α 太小 = 什么都没加（曲线全平）；α 太大 = 数据集破坏，与
「真实异常混入正常训练」的动机不符。所以 α 不能拍，要标定。

## 为什么可以复用已有 manifest
段的位置只由 (seed, ρ, 几何) 决定，**与 α 无关**（gen_contam 的 plan/place 不碰 α）。
所以本脚本直接读某个已生成 manifest 的 `segments`，只把注入幅度换掉，
不重跑几何 —— 保证标定用的污染结构与主实验**逐位同址**。

## 判据 —— 本模型的检测是**纯排序**的（踩过两次坑，记在这里）
`eval_method.py:48`：`thresh = np.percentile(score, float(threshold))`，
即 `bf_search` 扫的 93~100 是**百分位**，不是分数。两个后果：

1. **不能用 `metrics.threshold` 直接跟原始分数比大小**（曾导致所有 α 检出率都是 0）。
2. **分数的绝对值不可解释**：实测干净训练/测试/真实异常段/正常段四种窗口的分数
   都是 `48.2937 ± 1e-5`，跨度 1e-4 量级。F1 之所以成立，是因为**只有排序**起作用。
   所以标定必须用**排序口径**（分位数），不能用绝对差值。

因此：阈值取**干净训练窗口分数的 `threshold_pct` 分位** —— 直接复用模型自己的
告警率（`threshold_pct` 从干净 run 的结果 JSON 读，MSL 上是 98.85，即只标红前 1.15%
的窗口）。这样标定与 pipeline 的判定口径**逐字一致**，而不引入任何自定义常数。

「注入段被检出」= 与注入区间**有交叠**的窗口中，分数超过该分位的占比。

用法：
    python bench/calib_alpha.py \
        --ckpt checkpoint/contam/mamba/MSL/best.pt \
        --clean-json results/contam/mamba_MSL_rho00_seed42.json \
        --manifest artifacts/contam/MSL_rho20_seed7.json \
        --target 0.90 --out artifacts/contam/calib_MSL.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_loader, preprocess                       # noqa: E402
from bench.gen_contam import load_clean                         # noqa: E402


def build_at_alpha(base, segments, alpha, blocks_by_name):
    """按 manifest 的段位置，以给定 α 重新注入。返回 (array, 注入区间列表)。"""
    out = base.copy()
    spans = []
    for x in segments:
        b = blocks_by_name[x['block']]
        ref = base[b['t_start']:b['t_end'], 0]
        sd, mu = float(ref.std()), float(ref.mean())
        s, L, t = x['t_start'], x['L'], x['type']
        if t == 'level_shift':
            out[s:s + L, 0] = out[s:s + L, 0] + alpha * sd
        elif t == 'amplitude_scaling':
            out[s:s + L, 0] = mu + alpha * (out[s:s + L, 0] - mu)
        else:
            rng = np.random.default_rng(hash((alpha, s, L)) % (2**31))
            out[s:s + L, 0] = out[s:s + L, 0] + alpha * sd * rng.standard_normal(L)
        spans.append((s, s + L))
    return out, spans


def score(model, arr, W, bs, C):
    """复刻 main.py 的口径：滑窗 → is_anomaly → 每窗口一个分数。"""
    win = arr[np.arange(W)[None, :] + np.arange(arr.shape[0] - W)[:, None]]
    loader = get_loader(win, batch_size=bs, window_length=W, input_size=C, shuffle=False)
    n, c = int(win.shape[0] / bs), win.shape[0] % bs
    s = model.is_anomaly(loader, n, c)
    return np.asarray(s.detach().cpu().numpy() if hasattr(s, 'detach') else s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--clean-json', required=True, help='干净 run 的结果 JSON（提供 t_0）')
    ap.add_argument('--manifest', required=True, help='提供段位置（α 无关）')
    ap.add_argument('--target', type=float, default=0.90, help='目标检出率')
    ap.add_argument('--pct', type=float, default=None,
                    help='检出用的百分位；默认取干净 run 的 metrics.threshold（模型自己的告警率）')
    ap.add_argument('--alphas', default='0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0')
    ap.add_argument('--latent-size', type=int, default=256)
    ap.add_argument('--out', default='artifacts/contam/calib.json')
    a = ap.parse_args()

    cj = json.load(open(a.clean_json))
    W, bs = cj['config']['window_length'], cj['config']['batch_size']
    C = cj['config']['input_dim']
    print(f'干净模型: W={W} bs={bs} C={C}  (backbone={cj["backbone"]}, '
          f'来自 {os.path.basename(a.clean_json)})')

    man = json.load(open(a.manifest))
    segs = man['segments']
    print(f'段位置来自 {os.path.basename(a.manifest)}: {len(segs)} 段, '
          f'共 {sum(x["L"] for x in segs)} 行（α 无关）')

    import importlib
    from model.mamba import ModelArgs
    from main import BACKBONE_REGISTRY
    mod = importlib.import_module(BACKBONE_REGISTRY[cj['backbone']])
    args = ModelArgs()
    model = mod.patchattmamba(C, C * 5, a.latent_size, bs, W, 5, args=args).cuda()
    model.load_state_dict(torch.load(a.ckpt, map_location='cuda'))
    model.eval()
    print(f'载入 ckpt: {a.ckpt}  (backbone={cj["backbone"]})')

    from bench.channel_split import stacked_blocks
    blk = stacked_blocks(a.dataset, '/data/sdc/yrlbkp/anomaly/data/SMAP_MSL')
    byname = {b['name']: b for b in blk['blocks']}

    base = load_clean(a.dataset)

    # 阈值：干净训练窗口分数的 pct 分位（与 eval_method.py:48 的判定口径逐字一致）
    sc_clean = score(model, base, W, bs, C)
    pct = a.pct if a.pct is not None else float(cj['metrics']['threshold'])
    t0 = float(np.percentile(sc_clean, pct))
    print(f'\n干净训练窗口分数: n={len(sc_clean)}  min={sc_clean.min():.6f} '
          f'p50={np.median(sc_clean):.6f} max={sc_clean.max():.6f} '
          f'std={sc_clean.std():.2e} 跨度={sc_clean.ptp() if hasattr(sc_clean,"ptp") else np.ptp(sc_clean):.2e}')
    print(f'检出阈值 = 干净训练分数的 {pct:.4f} 分位 = {t0:.6f}'
          f'（即标红前 {100-pct:.2f}% 的窗口；分位来自模型自己的告警率）')
    print('  ⚠️ 分数绝对值退化是**正常**的：eval_method.py:48 用 np.percentile 定阈，'
          '检测是纯排序的')

    print(f'\n{"α":>7}{"检出率":>10}{"注入段窗口分位(p50/p90)":>26}  判定')
    rows, best = [], None
    for al in [float(x) for x in a.alphas.split(',')]:
        arr, spans = build_at_alpha(base, segs, al, byname)
        sc = score(model, arr, W, bs, C)
        m = np.zeros(len(sc), bool)
        for s, e in spans:
            m[max(0, s - W + 1):min(len(sc), e)] = True
        hit = float((sc[m] > t0).mean())
        p50, p90 = np.percentile(sc[m], [50, 90])
        ok = hit >= a.target
        print(f'{al:>7.2f}{hit:>10.3f}{p50:>14.4f} /{p90:>10.4f}  {"✓ 达标" if ok else ""}')
        rows.append(dict(alpha=al, detection=hit, p50=float(p50), p90=float(p90)))
        if ok and best is None:
            best = al

    if best is None:
        print(f'\n[FAIL] 扫描范围内没有 α 达到 {a.target:.0%} —— 扩大 --alphas 上限')
        sys.exit(1)
    print(f'\n>>> α* = {best}  （最小的达标 α）')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(dataset=a.dataset, target=a.target, alpha_star=best,
                   t0=t0, clean_ckpt=a.ckpt, clean_json=a.clean_json,
                   manifest=a.manifest, scan=rows,
                   note='段位置取自 manifest（与 α 无关）；注入在模型输入空间，见 gen_contam.py'),
              open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('->', a.out)


if __name__ == '__main__':
    main()
