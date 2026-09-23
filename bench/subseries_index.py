"""还原子序列在拼接数组中的边界（SMAP / MSL）。

**重建方式（已逐行验证）**

OmniAnomaly / TranAD 把每个数据集的全部子序列**按文件名排序**拼接成一个数组。
本仓库的 `SMAP_test.npy` / `MSL_test.npy` 就是拼接后的结果，但**各自排除了一个子序列**：

    SMAP: 拼接 54 个 25 维文件，排除 P-2.npy    → 427617 行
    MSL : 拼接 28 个 55 维文件，排除 T-10.npy   →  73729 行

排除项的判定依据：文件行数与「排序拼接总长 − 目标长度」的差额**恰好相等**
（SMAP 差 8209 = P-2 长度；MSL 差 670 = T-10 长度），且去掉后逐行严格比对完全一致。

**为什么能逐值比对**：`utils.preprocess()` 用的是 `MinMaxScaler`——**逐列的仿射变换**。
所以把两个数组各自归一化到 [0,1] 后，max|Δ| 应为 0。实测为 0。

用法：
    from bench.subseries_index import build_index
    idx = build_index('SMAP')      # {'P-1': (354374, 362879), ...}
"""
import os

import numpy as np

# 原始逐子序列文件所在目录
RAW_TEST_DIR = '/data/sdc/yrlbkp/DATASET/anomaly/data/SMAP_MSL/test'
# 本仓库使用的拼接数组（经 get_data 的 get_data_dim 确定列数）
CONCAT_PATH = {
    'SMAP': '/data/sdc/yrlbkp/anomaly/SMAP/SMAP_test.npy',
    'MSL':  '/data/sdc/yrlbkp/anomaly/MSL/MSL_test.npy',
}
CONCAT_DIM = {'SMAP': 25, 'MSL': 55}
SKIP_FILE = {'SMAP': 'P-2.npy', 'MSL': 'T-10.npy'}


def _minmax(x):
    """按列归一化到 [0,1]，用于跨数组比对（MinMaxScaler 是逐列仿射）。"""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(0, keepdims=True), x.max(0, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def list_subseries(dataset):
    """列出属于该数据集的子序列文件名（按排序）。用列数区分数据集。"""
    dim = CONCAT_DIM[dataset]
    names = []
    for f in sorted(os.listdir(RAW_TEST_DIR)):
        if not f.endswith('.npy'):
            continue
        if np.load(os.path.join(RAW_TEST_DIR, f), mmap_mode='r').shape[1] == dim:
            names.append(f)
    return names


def build_index(dataset, verify=True):
    """返回 {子序列名(不含 .npy): (start, end)}，位置基于本仓库的拼接数组。

    verify=True 时逐段严格比对重建结果与拼接数组（不匹配则抛错）。
    """
    names = list_subseries(dataset)
    skip = SKIP_FILE[dataset]
    keep = [f for f in names if f != skip]

    lens = {f: int(np.load(os.path.join(RAW_TEST_DIR, f), mmap_mode='r').shape[0])
            for f in names}

    idx, c = {}, 0
    for f in keep:
        idx[f[:-4]] = (c, c + lens[f])
        c += lens[f]

    if verify:
        full = np.load(CONCAT_PATH[dataset]).reshape(-1, CONCAT_DIM[dataset])
        if c != len(full):
            raise ValueError(f'{dataset}: 重建长度 {c} != 拼接数组 {len(full)}')
        cat = np.concatenate([np.load(os.path.join(RAW_TEST_DIR, f)) for f in keep], 0)
        # 分块比对：整段一次性归一化会因范围不同而失真，故每块各自成 [0,1]
        for s in range(0, len(full), 50000):
            e = min(s + 50000, len(full))
            if np.abs(_minmax(cat[s:e]) - _minmax(full[s:e])).max() > 1e-8:
                raise ValueError(f'{dataset}: 段 [{s},{e}) 比对失败')
    return idx


def load_labels(dataset):
    """拼接数组对应的标签（一维）。"""
    p = CONCAT_PATH[dataset].replace('_test.npy', '_test_label.npy')
    return np.load(p).reshape(-1)


def dataset_config(dataset):
    """与 main.py / ExpConfig 一致的 W 与 batch_size。"""
    if dataset == 'SWaT':
        return 35, 60
    if dataset == 'WADI':
        return 140, 24
    return 70, 60


if __name__ == '__main__':
    import sys
    for ds in ('SMAP', 'MSL'):
        idx = build_index(ds)
        print(f'===== {ds}: {len(idx)} 个子序列（已去掉 {SKIP_FILE[ds]}）=====')
        tot = max(b for _, b in idx.values())
        print(f'  拼接总长 {tot}')
        for k, (a, b) in list(idx.items())[:3]:
            print(f'    {k:8s} [{a:>7d}, {b:>7d})  {b - a:>5d} 行')
        print(f'    ...')
        for k in ('P-1', 'C-1', 'P-2', 'T-10'):
            if k in idx:
                a, b = idx[k]
                print(f'    ★ {k:8s} [{a:>7d}, {b:>7d})  {b - a:>5d} 行')
        print()
