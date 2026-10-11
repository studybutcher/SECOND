#!/usr/bin/env python
"""PRD5.1 r3 · 为 AE 实验准备数据（§4）。

做三件事，都按 PRD 的硬要求：

1. **正常 train / 正常 val 划分**：逐实体切，val 取每个实体的**后段**，两者之间留 `GAP=window`
   行隔离区（§4.1「时间段互不重叠，跨边界留一个窗口的隔离区」）。
2. **scaler 只在 train_normal 上拟合**（§4.2），再用它 transform val。
   —— 注意 `utils.preprocess` 的默认行为是在**自己**上面拟合，所以 val 必须显式传 `fit_data`。
3. **逐实体切窗**（不跨实体接缝），窗口化在注入之后做（§4.3），故本模块只产出**行数组 + 块元数据**。

产出（`artifacts/ae_mpa/`）：
    <ds>_train_normal.npy   正常训练行（已归一化，float32）
    <ds>_val_normal.npy     固定正常验证行（用 train 的 scaler 变换，float32）
    <ds>_split.json         块元数据（每块在新数组里的 [off, off+n)）+ 行数/gap/val_frac + sha256

用法：
    python bench/ae_data.py --dataset MSL --window 70
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

OUT_DIR = os.path.join('artifacts', 'ae_mpa')
RAW = '/data/sdc/yrlbkp/anomaly'


def sha256(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def transform(arr, meta):
    """用 train_normal 上拟合的 scaler 变换任意同维数组。

    与 `utils.preprocess` 逐位等价（`_clean` → `MinMaxScaler`），只是把 scaler 显式留下来，
    好让 anomaly probe 也能用**同一把** scaler（§4.2：scaler 只从 train_normal 拟合）。
    """
    return _clean(np.asarray(arr, dtype=np.float64)) @ np.diag(meta['scaler_scale']) \
        + meta['scaler_min']


def build(dataset, window, val_frac=0.15, out_dir=OUT_DIR):
    K = {'MSL': 55, 'SMAP': 25, 'SWaT': 51, 'WADI': 127}[dataset]
    raw = np.load(f'{RAW}/{dataset}/{dataset}_train.npy').reshape(-1, K).astype(np.float64)
    blocks = entity_blocks(dataset, 'train', len(raw))

    tr_parts, va_parts, tr_meta, va_meta = [], [], [], []
    off_t = off_v = 0
    for name, s, e in blocks:
        n = e - s
        n_val = int(n * val_frac)
        cut = e - n_val
        t_end = cut - window if cut - window > s else s          # 隔离区
        a_t, a_v = raw[s:t_end], raw[cut:e]
        if len(a_t) < window + 1 or len(a_v) < window + 1:
            print(f'  [skip] {name}: train {len(a_t)} / val {len(a_v)} 太短')
            continue
        tr_parts.append(a_t); va_parts.append(a_v)
        tr_meta.append(dict(name=name, off=int(off_t), n=int(len(a_t)), src_start=int(s)))
        va_meta.append(dict(name=name, off=int(off_v), n=int(len(a_v)), src_start=int(cut)))
        off_t += len(a_t); off_v += len(a_v)
        assert t_end <= cut, f'{name}: train 与 val 重叠'

    tr_raw = np.concatenate(tr_parts, 0)
    va_raw = np.concatenate(va_parts, 0)
    # §4.2：scaler 只在 train_normal 上拟合；val 与后续 probe 都用它变换
    sc = MinMaxScaler().fit(_clean(tr_raw))
    tr = sc.transform(_clean(tr_raw)).astype(np.float32)
    va = sc.transform(_clean(va_raw)).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(f'{out_dir}/{dataset}_train_normal.npy', tr)
    np.save(f'{out_dir}/{dataset}_val_normal.npy', va)
    meta = dict(dataset=dataset, window=window, val_frac=val_frac, gap_rows=window,
                n_train_rows=int(len(tr)), n_val_rows=int(len(va)),
                n_blocks=len(tr_meta), blocks_train=tr_meta, blocks_val=va_meta,
                scaler_fit_on='train_normal only',
                scaler_min=sc.min_.tolist(), scaler_scale=sc.scale_.tolist(),
                sha256_train=sha256(tr), sha256_val=sha256(va),
                source=f'{RAW}/{dataset}/{dataset}_train.npy')
    with open(f'{out_dir}/{dataset}_split.json', 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f'{dataset}: train {tr.shape}  val {va.shape}  块={len(tr_meta)}  '
          f'({tr.shape[0]} / {va.shape[0]} = {tr.shape[0]/va.shape[0]:.2f}:1)')
    for m in tr_meta[:3] + tr_meta[-1:]:
        print(f'    {m["name"]:<6} train off={m["off"]:<7} n={m["n"]:<6}')
    return meta


def build_probe(dataset, window, meta, out_dir=OUT_DIR):
    """从 **test** 构造固定 probe（§4.4：只用于评价与作图，不参与梯度与模型选择）。

    x_normal / x_anom = test 的正常窗 / 异常窗（标签口径 = **any-in-window**，预注册）。
    与 train/val 使用**同一把** scaler。
    """
    K = {'MSL': 55, 'SMAP': 25, 'SWaT': 51, 'WADI': 127}[dataset]
    raw = np.load(f'{RAW}/{dataset}/{dataset}_test.npy').reshape(-1, K).astype(np.float64)
    lab = (np.load(f'{RAW}/{dataset}/{dataset}_test_label.npy').reshape(-1) > 0.1).astype(int)
    arr = transform(raw, meta)
    xn, xa = [], []
    for name, s, e in entity_blocks(dataset, 'test', len(raw)):
        n = e - s - window
        if n <= 0:
            continue
        idx = (s + np.arange(n))[:, None] + np.arange(window)[None, :]
        y = lab[idx].max(axis=1)                       # any-in-window
        w = arr[idx]
        if (y == 0).any():
            xn.append(w[y == 0])
        if (y == 1).any():
            xa.append(w[y == 1])
    xn = np.concatenate(xn).astype(np.float32)
    xa = np.concatenate(xa).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f'{out_dir}/{dataset}_probe.npz', x_normal=xn, x_anom=xa)
    print(f'{dataset} probe: normal {xn.shape}  anomaly {xa.shape}  '
          f'(异常窗占比 {len(xa) / (len(xa) + len(xn)):.4f})'
          f'  → {out_dir}/{dataset}_probe.npz')
    return xn, xa


def load(dataset, out_dir=OUT_DIR):
    """→ (train_arr, val_arr, meta)。"""
    with open(f'{out_dir}/{dataset}_split.json') as f:
        meta = json.load(f)
    return (np.load(f'{out_dir}/{dataset}_train_normal.npy'),
            np.load(f'{out_dir}/{dataset}_val_normal.npy'), meta)


def windows_from(arr, blocks, window):
    """逐块切窗（不跨接缝）→ (win [N,W,C], ent_id [N], row_idx [N])。"""
    win, eid, ridx = [], [], []
    for i, b in enumerate(blocks):
        n = b['n'] - window
        if n <= 0:
            continue
        r = b['off'] + np.arange(n)
        win.append(arr[r[:, None] + np.arange(window)[None, :]])
        eid.append(np.full(n, i, dtype=np.int16))
        ridx.append(r)
    return np.concatenate(win), np.concatenate(eid), np.concatenate(ridx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window', type=int, default=70)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--probe', action='store_true', help='同时构造 test 的 normal/anomaly probe')
    ap.add_argument('--out-dir', default=OUT_DIR)
    a = ap.parse_args()
    meta = build(a.dataset, a.window, a.val_frac, a.out_dir)
    if a.probe:
        build_probe(a.dataset, a.window, meta, a.out_dir)


if __name__ == '__main__':
    main()
