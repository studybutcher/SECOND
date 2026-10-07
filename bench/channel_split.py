#!/usr/bin/env python
"""连续 / 离散通道划分表 —— 供污染注入排除不可注入的通道。

`docs/PRD4text_定量分析_r2.md` §5.3 要求"binary/discrete 执行器变量不注入"
（`x + ασ` 会产生物理上不存在的状态），并引用了本表，但表中此前没有内容。
本脚本用**可复核的规则**生成它，而不是硬编码一份名单。

判定优先级（先到先得）：
  1. `dead`       —— 训练切片上 std == 0（恒为零的通道，`σ=0` 使 `ασ` 无效果；
                     含原始 CSV 全 NaN、被 `preprocess` 填 0 的通道）
  2. `discrete`   —— 训练切片上唯一值数 ≤ `--max-unique`（默认 4），或 tag 名以
                     `_STATUS` / `_AL` 结尾（执行器/报警）
  3. `continuous` —— 其余（= **可注入**）

注意 tag 名判定不完整：`2_LS_101_AH` 这类 `_AH`（高位报警）名字像模拟量、实为二值。
所以**以实测唯一值数为准**，tag 名只作交叉校验。WADI 上两者会差 7 个通道，
本脚本会把差异显式打印出来。

数据来源（`utils.prefix` 下的布局）：
  * 训练切片：`<prefix>/<ds>/<ds>_train.npy`  ← 训练实际看到的数据
  * 原始 CSV（可选，`--raw-csv`）：用于检查「全 NaN 通道」与物理单位

用法：
    python bench/channel_split.py WADI
    python bench/channel_split.py WADI --raw-csv /data/.../WADI_14days.csv
    python bench/channel_split.py WADI --out artifacts/contam/channel_split_WADI.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_dim, prefix          # noqa: E402

DISCRETE_TAG_RE = re.compile(r'_(STATUS|AL)$')

# SMAP / MSL 的 `<ds>_train.npy` 不是"时间戳×通道"，而是把**逐通道文件纵向拼接**起来的：
# 第 0 列是该通道的信号，第 1..K-1 列是二值指示列。因此"可注入"的单位是
# **(通道块, 第 0 列)**，不是整个数组的某一列 —— 对整个数组按唯一值数判定会得出
# "MSL 只有 1 个可注入列"的错误结论（实际是 22/27 个通道块）。
STACKED_SETS = {'SMAP', 'MSL'}


def stacked_blocks(dataset, raw_dir, split='train'):
    """SMAP/MSL：按 labeled_anomalies.csv 的通道顺序切出块，统计每块第 0 列。

    `split` 决定**块边界与统计量取自哪个划分**：train 与 test 的每块长度不同
    （如 MSL 的 C-1 是 train 2158 / test 2264），所以往测试集注入时必须用
    `split='test'`，否则段会落在错误的块里。
    """
    import csv
    import ast
    lab_csv = os.path.join(raw_dir, 'labeled_anomalies.csv')
    rows = sorted(list(csv.reader(open(lab_csv)))[1:], key=lambda k: k[0])
    info = [r for r in rows if r[1] == dataset and r[0] != 'P-2']

    K = get_data_dim(dataset)
    arr = np.load(os.path.join(prefix, dataset, f'{dataset}_{split}.npy')).reshape(-1, K)
    lab = np.load(os.path.join(prefix, dataset, f'{dataset}_test_label.npy')).reshape(-1)

    blocks, b, bt = [], 0, 0
    for r in info:
        name = r[0]
        L = np.load(os.path.join(raw_dir, split, name + '.npy')).shape[0]
        Lte = np.load(os.path.join(raw_dir, 'test', name + '.npy')).shape[0]
        c0 = arr[b:b + L, 0]
        sd = float(c0.std())
        blocks.append(dict(
            name=name, t_start=int(b), t_end=int(b + L), length=int(L),
            col0_mean=float(c0.mean()), col0_std=sd,
            col0_n_unique=int(np.unique(c0).size),
            n_constant_indicator_cols=int((arr[b:b + L, 1:].std(0) == 0).sum()),
            n_anomaly_intervals=len(ast.literal_eval(r[2])),
            n_anomalous_points=int(lab[bt:bt + Lte].sum()),
            test_length=int(Lte),
            injectable=bool(sd > 0),
        ))
        b += L
        bt += Lte

    inj = [blk['name'] for blk in blocks if blk['injectable']]
    return dict(dataset=dataset, layout='stacked', split=split, n_channels=K,
                n_timestamps=int(arr.shape[0]),
                npy=os.path.join(prefix, dataset, f'{dataset}_{split}.npy'),
                rule=dict(signal_col=0,
                          injectable='该通道块第 0 列的 std > 0',
                          other_cols='二值指示列，不注入'),
                raw_csv=lab_csv, n_blocks=len(blocks),
                injectable_blocks=inj, blocks=blocks)



def tag_names_from_csv(raw_csv, skiprows, first_col, n_cols):
    """从原始 CSV 的表头取通道名（去掉 Windows 路径前缀与引号）。"""
    import pandas as pd
    hdr = list(pd.read_csv(raw_csv, skiprows=skiprows, nrows=0).columns)
    return [str(c).strip().strip('"').split('\\')[-1] for c in hdr[first_col:first_col + n_cols]]


def split(dataset, max_unique=4, raw_csv=None, raw_skiprows=None, raw_first_col=None):
    C = get_data_dim(dataset)
    path = os.path.join(prefix, dataset, f'{dataset}_train.npy')
    arr = np.load(path).reshape(-1, C)

    tags = None
    if raw_csv:
        skiprows = 4 if raw_skiprows is None else raw_skiprows
        first_col = 3 if raw_first_col is None else raw_first_col
        tags = tag_names_from_csv(raw_csv, skiprows, first_col, C)
        if len(tags) != C:
            print(f'[warn] CSV 通道名 {len(tags)} 个 != 数据维度 {C}，放弃 tag 交叉校验')
            tags = None
    if tags is None:
        tags = [f'ch{j}' for j in range(C)]

    rows = []
    for j in range(C):
        col = arr[:, j]
        nu = int(np.unique(col).size)
        sd = float(col.std())
        tag = tags[j]
        tag_discrete = bool(DISCRETE_TAG_RE.search(tag))
        if sd == 0.0:
            kind = 'dead'
        elif nu <= max_unique:
            kind = 'discrete'
        else:
            kind = 'continuous'
        rows.append(dict(idx=j, tag=tag, kind=kind, n_unique=nu, std=sd,
                         tag_suffix='discrete' if tag_discrete else 'analog'))

    injectable = [r['idx'] for r in rows if r['kind'] == 'continuous']
    return dict(dataset=dataset, n_channels=C, n_timestamps=int(arr.shape[0]),
                train_npy=path, rule=dict(std_eq_0='dead',
                                          n_unique_le=max_unique, rest='continuous'),
                raw_csv=raw_csv, injectable_idx=injectable, channels=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['WADI', 'SWaT', 'MSL', 'SMAP'])
    ap.add_argument('--max-unique', type=int, default=4)
    ap.add_argument('--raw-csv', default=None, help='原始 CSV，用于 tag 名与交叉校验')
    ap.add_argument('--raw-dir', default='/data/sdc/yrlbkp/anomaly/data/SMAP_MSL',
                    help='SMAP/MSL 的逐通道原始目录（含 labeled_anomalies.csv）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if a.raw_csv is None:      # 已知布局的默认值
        default = {'WADI': '/data/sdc/yrlbkp/anomaly/data/WADI/WADI_14days.csv'}
        a.raw_csv = default.get(a.dataset)

    if a.dataset in STACKED_SETS:
        r = stacked_blocks(a.dataset, a.raw_dir)
        print(f"{a.dataset}（stacked 布局）: {r['n_blocks']} 个通道块 / "
              f"{r['n_channels']} 列 / {r['n_timestamps']} 时间戳")
        print(f"  >>> 可注入通道块（第 0 列非常数）= {len(r['injectable_blocks'])}/{r['n_blocks']}")
        print(f"      {r['injectable_blocks']}")
        skip = [b['name'] for b in r['blocks'] if not b['injectable']]
        print(f"  恒值通道块（不可注入）: {skip}")
        print(f"\n  {'通道':<7}{'块长':>7}{'col0_σ':>10}{'col0_μ':>9}{'异常段':>7}{'异常点':>7}")
        for b in r['blocks']:
            flag = '' if b['injectable'] else '  ← 恒值'
            print(f"  {b['name']:<7}{b['length']:>7}{b['col0_std']:>10.4f}{b['col0_mean']:>9.3f}"
                  f"{b['n_anomaly_intervals']:>7}{b['n_anomalous_points']:>7}{flag}")
        if a.out:
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            with open(a.out, 'w') as f:
                json.dump(r, f, indent=1, ensure_ascii=False)
            print('\n  ->', a.out)
        return

    r = split(a.dataset, a.max_unique, a.raw_csv)
    from collections import Counter
    cnt = Counter(c['kind'] for c in r['channels'])
    print(f"{a.dataset}: {r['n_channels']} 通道 / {r['n_timestamps']} 时间戳")
    print(f"  dead={cnt['dead']}  discrete={cnt['discrete']}  continuous={cnt['continuous']}")
    print(f"  >>> 可注入（continuous）= {len(r['injectable_idx'])} 个: {r['injectable_idx']}")

    mism = [c for c in r['channels']
            if c['kind'] == 'continuous' and c['tag_suffix'] == 'discrete']
    if mism:
        print(f'  [warn] tag 像离散但数值连续：{[(c["idx"], c["tag"]) for c in mism]}')
    nobin = [c for c in r['channels']
             if c['tag_suffix'] == 'analog' and c['kind'] != 'continuous']
    if nobin:
        print(f'  [info] tag 像模拟量但不可注入（{len(nobin)} 个，注入前须排除）：'
              f'{[(c["idx"], c["tag"]) for c in nobin]}')

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, 'w') as f:
            json.dump(r, f, indent=1, ensure_ascii=False)
        print('  ->', a.out)


if __name__ == '__main__':
    main()
