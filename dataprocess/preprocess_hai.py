#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PRD7: 把 HAI 22.04 / 23.05 处理成统一 schema 的 npy。

产物（默认写到 /data/sdc/yrlbkp/anomaly）:

    HAI22/HAI22_{train,test}.npy            float32 [N, 78]，清洗后**原始值**
    HAI22/HAI22_{train,test}_label.npy      int8    [N]
    HAI22/HAI22_{train,test}_segment.npy    int16   [N]
    HAI22/HAI22_{train,test}_ts.npy         int64   [N]，纳秒
    HAI22/scale.json                        本域 train 的逐列 min/max/degenerate
    HAI22/schema.json                       78 列名 + sha256
    HAI22/meta.json                         行数/片段/逐文件统计
    HAI23/...                               同上

设计要点（见 docs/PRD7_HAI双数据集_交叉泛化预处理与实验.md）:
- 78 列全部保留；20 个常量列不删除，靠加载时“源域零范围 → 输出 0”保护。
- npy 不预归一化：交叉评测要在加载时用**源域**的 scale.json 变换目标域 test。
- 23.05 test 的标签按**行号**对齐（label-test2 的 timestamp 被截断到分钟）。
- 同一 split 内按 timestamp 稳定升序排序，断言严格递增且片段内间隔 1 秒。

用法:
    python dataprocess/preprocess_hai.py \
        [--src /data/sdc/yrlbkp/anomaly] [--out /data/sdc/yrlbkp/anomaly] [--force]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

# 附录 A：78 个模型输入列（字典序）
EXPECTED_COLUMNS = [
    'P1_FCV01D', 'P1_FCV01Z', 'P1_FCV02D', 'P1_FCV02Z', 'P1_FCV03D', 'P1_FCV03Z',
    'P1_FT01', 'P1_FT01Z', 'P1_FT02', 'P1_FT02Z', 'P1_FT03', 'P1_FT03Z',
    'P1_LCV01D', 'P1_LCV01Z', 'P1_LIT01', 'P1_PCV01D', 'P1_PCV01Z', 'P1_PCV02D',
    'P1_PCV02Z', 'P1_PIT01', 'P1_PIT01_HH', 'P1_PIT02',
    'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR', 'P1_PP02D', 'P1_PP02R',
    'P1_PP04', 'P1_PP04SP', 'P1_SOL01D', 'P1_SOL03D', 'P1_STSP',
    'P1_TIT01', 'P1_TIT02', 'P1_TIT03',
    'P2_24Vdc', 'P2_ATSW_Lamp', 'P2_AutoGO', 'P2_AutoSD', 'P2_Emerg',
    'P2_MASW', 'P2_MASW_Lamp', 'P2_ManualGO', 'P2_ManualSD',
    'P2_OnOff', 'P2_RTR', 'P2_SCO', 'P2_SCST', 'P2_SIT01', 'P2_TripEx',
    'P2_VIBTR01', 'P2_VIBTR02', 'P2_VIBTR03', 'P2_VIBTR04', 'P2_VT01',
    'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04',
    'P3_FIT01', 'P3_LCP01D', 'P3_LCV01D', 'P3_LH01', 'P3_LIT01', 'P3_LL01', 'P3_PIT01',
    'P4_HT_FD', 'P4_HT_PO', 'P4_HT_PS', 'P4_LD', 'P4_ST_FD', 'P4_ST_GOV', 'P4_ST_LD',
    'P4_ST_PO', 'P4_ST_PS', 'P4_ST_PT01', 'P4_ST_TT01',
]

# 附录 B：在 4 个 split 的任一 split 里 nunique<=1 的 20 列
EXPECTED_CONST = [
    'P1_PIT01_HH', 'P1_PP01AD', 'P1_PP01AR', 'P1_PP01BD', 'P1_PP01BR',
    'P1_PP02D', 'P1_PP02R', 'P1_SOL01D', 'P1_SOL03D', 'P1_STSP',
    'P2_Emerg', 'P2_OnOff', 'P2_RTR', 'P2_TripEx',
    'P2_VTR01', 'P2_VTR02', 'P2_VTR03', 'P2_VTR04',
    'P3_LH01', 'P3_LL01',
]

DOMAINS = {
    'HAI22': {
        'raw_dir': 'hai-22.04',
        'train_files': [f'train{i}.csv' for i in range(1, 7)],
        'test_files': [f'test{i}.csv' for i in range(1, 5)],
        'inline_label': True,            # 每行最后一列 Attack
        'label_files': None,
    },
    'HAI23': {
        'raw_dir': 'hai-23.05',
        'train_files': [f'hai-train{i}.csv' for i in range(1, 5)],
        'test_files': [f'hai-test{i}.csv' for i in range(1, 3)],
        'inline_label': False,           # 无 Attack 列，标签在 label-test*.csv
        'label_files': {f'hai-test{i}.csv': f'label-test{i}.csv' for i in range(1, 3)},
    },
}


def die(msg):
    print(f'[FAIL] {msg}', file=sys.stderr)
    raise SystemExit(1)


def clean_float(X):
    """与 utils._clean 等价：NaN→0、±inf→100。"""
    X = np.asarray(X, dtype=np.float32)
    bad = np.isnan(X)
    if bad.any():
        print(f'  [clean] NaN {int(bad.sum())} → 0')
        X[bad] = 0.0
    bad = np.isinf(X)
    if bad.any():
        print(f'  [clean] inf {int(bad.sum())} → 100')
        X[bad] = 100.0
    return X


def read_columns(path):
    return pd.read_csv(path, nrows=0).columns.tolist()


def load_label_by_row(src_dir, data_file, label_file, n_rows):
    """23.05 test: 标签与数据**按行号**对齐，时间戳只做分钟级校验。"""
    lf = os.path.join(src_dir, label_file)
    lab = pd.read_csv(lf)
    if 'label' not in lab.columns:
        die(f'{lf} 缺少 label 列: {lab.columns.tolist()}')
    if len(lab) != n_rows:
        die(f'{label_file} 行数 {len(lab)} != {data_file} 行数 {n_rows}')

    # 标签集合检查
    vals = set(np.unique(lab['label'].to_numpy()))
    if not vals <= {0, 1}:
        die(f'{label_file} 标签取值不是 {{0,1}}: {vals}')

    # 分钟级校验：两边的 timestamp 都 floor('min') 后逐行相等
    dt = pd.read_csv(os.path.join(src_dir, data_file), usecols=['timestamp'])
    t_data = pd.to_datetime(dt['timestamp'])
    t_lab = pd.to_datetime(lab['timestamp'])
    m_data = t_data.dt.floor('min').to_numpy()
    m_lab = t_lab.dt.floor('min').to_numpy()
    n_eq = int((m_data == m_lab).sum())
    if n_eq != n_rows:
        die(f'{label_file} 分钟级对齐失败: {n_eq}/{n_rows}')
    print(f'  [label] {data_file}: 行号对齐 OK, floor(min) {n_eq}/{n_rows}, '
          f'pos={int(lab["label"].sum())}')
    return lab['label'].to_numpy(dtype=np.int8)


def load_split(src_dir, files, columns, inline_label, label_files):
    """逐文件读取 → 清洗 → 拼接 → 稳定排序 → 一致性断言。"""
    Xs, ys, ts, segs = [], [], [], []
    per_file = []
    for seg_id, fname in enumerate(files):
        path = os.path.join(src_dir, fname)
        header = read_columns(path)
        missing = [c for c in columns if c not in header]
        if missing:
            die(f'{path} 缺少列: {missing[:5]} ...')
        usecols = ['timestamp'] + columns + (['Attack'] if inline_label else [])
        df = pd.read_csv(path, usecols=usecols)
        n = len(df)

        t = pd.to_datetime(df['timestamp'])
        dts = t.diff().dropna()
        if len(dts) and not (dts == pd.Timedelta(seconds=1)).all():
            bad = int((dts != pd.Timedelta(seconds=1)).sum())
            die(f'{path} 片段内存在 {bad} 处非 1 秒间隔')

        X = clean_float(df[columns].to_numpy())
        if inline_label:
            y = df['Attack'].to_numpy(dtype=np.int8)
            if not set(np.unique(y)) <= {0, 1}:
                die(f'{path} Attack 取值不是 {{0,1}}')
        else:
            mapping = (label_files or {}).get(fname)
            if mapping is None:
                y = np.zeros(n, dtype=np.int8)   # train：无标签，全 0
            else:
                y = load_label_by_row(src_dir, fname, mapping, n)

        Xs.append(X)
        ys.append(y)
        ts.append(t.to_numpy(dtype='datetime64[ns]').astype(np.int64))
        segs.append(np.full(n, seg_id, dtype=np.int16))
        per_file.append({'file': fname, 'rows': int(n),
                         'pos': int(y.sum()),
                         'ts_min': str(t.iloc[0]), 'ts_max': str(t.iloc[-1])})
        print(f'  [read] {fname}: rows={n} pos={int(y.sum())}')

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys)
    ts = np.concatenate(ts)
    seg = np.concatenate(segs)

    # 稳定升序排序，同时携带 label / segment
    order = np.argsort(ts, kind='mergesort')
    X, y, ts, seg = X[order], y[order], ts[order], seg[order]

    # 严格递增、无重复
    dup = int((np.diff(ts) == 0).sum())
    if dup:
        die(f'排序后仍有 {dup} 个重复时间戳（split 内不允许重复）')
    if np.any(np.diff(ts) < 0):
        die('排序后时间戳非单调')

    return X, y, ts, seg, per_file


def check_schema(columns, expected):
    if columns != expected:
        die('列交集与附录 A 不一致\n'
            f'  actual  ({len(columns)}): {columns}\n'
            f'  expected({len(expected)}): {expected}')
    if len(set(columns)) != len(columns):
        die('列名有重复')


def main():
    ap = argparse.ArgumentParser(description='PRD7 HAI 22.04/23.05 预处理')
    ap.add_argument('--src', default='/data/sdc/yrlbkp/anomaly')
    ap.add_argument('--out', default='/data/sdc/yrlbkp/anomaly')
    ap.add_argument('--force', action='store_true', help='覆盖已存在的输出目录')
    args = ap.parse_args()

    # ---- 1. 列交集 ----
    cols22 = read_columns(os.path.join(args.src, 'hai-22.04', 'train1.csv'))
    cols23 = read_columns(os.path.join(args.src, 'hai-23.05', 'hai-train1.csv'))
    common = sorted((set(cols22) & set(cols23)) - {'timestamp', 'Attack'})
    check_schema(common, EXPECTED_COLUMNS)
    schema_sha = hashlib.sha256('\n'.join(common).encode()).hexdigest()
    print(f'[schema] {len(common)} columns, sha256={schema_sha}')

    out_dirs = {d: os.path.join(args.out, d) for d in DOMAINS}
    for d, p in out_dirs.items():
        if os.path.isdir(p) and os.listdir(p) and not args.force:
            die(f'{p} 已存在且非空；加 --force 覆盖')
        os.makedirs(p, exist_ok=True)

    # ---- 2. 逐域处理 ----
    all_stats = {}
    const_union = set()
    for dname, spec in DOMAINS.items():
        src_dir = os.path.join(args.src, spec['raw_dir'])
        out = out_dirs[dname]
        print(f'\n=== {dname}: {src_dir} -> {out} ===')

        splits = {}
        for split, files in (('train', spec['train_files']),
                             ('test', spec['test_files'])):
            print(f'[{split}]')
            X, y, ts, seg, per_file = load_split(
                src_dir, files, common, spec['inline_label'], spec['label_files'])
            splits[split] = (X, y, ts, seg, per_file)

            # 常量列检测（清洗后无 NaN，min==max 等价于 nunique<=1）
            nun = (X.max(axis=0) == X.min(axis=0))
            const_union |= {c for c, k in zip(common, nun) if k}

            np.save(os.path.join(out, f'{dname}_{split}.npy'), X)
            np.save(os.path.join(out, f'{dname}_{split}_label.npy'), y)
            np.save(os.path.join(out, f'{dname}_{split}_segment.npy'), seg)
            np.save(os.path.join(out, f'{dname}_{split}_ts.npy'), ts)

            stats = {
                'rows': int(X.shape[0]), 'cols': int(X.shape[1]),
                'segments': len(files), 'pos': int(y.sum()),
                'pos_rate': float(y.mean()) if len(y) else 0.0,
                'nan': int(np.isnan(X).sum()), 'inf': int(np.isinf(X).sum()),
                'dup_ts': int((np.diff(ts) == 0).sum()),
                'ts_min': pd.Timestamp(ts[0]).isoformat() if len(ts) else None,
                'ts_max': pd.Timestamp(ts[-1]).isoformat() if len(ts) else None,
                'per_file': per_file,
            }
            all_stats[f'{dname}/{split}'] = stats
            print(f'  [save] {dname}_{split}.npy {X.shape} '
                  f'{X.dtype}, pos={stats["pos"]} ({stats["pos_rate"]:.3%})')

        # ---- 3. scale.json（只用本域 train） ----
        Xtr = splits['train'][0]
        mn, mx = Xtr.min(axis=0), Xtr.max(axis=0)
        scale = mx - mn
        stats = []
        for j, c in enumerate(common):
            s = float(scale[j])
            stats.append({
                'column': c,
                'min': float(mn[j]),
                'max': float(mx[j]),
                'scale': s,
                'degenerate': bool(s == 0.0),
            })
        with open(os.path.join(out, 'scale.json'), 'w') as f:
            json.dump({'dataset': dname, 'columns': common, 'stats': stats},
                      f, indent=2, ensure_ascii=False)
        n_deg = sum(s['degenerate'] for s in stats)
        print(f'[scale] degenerate={n_deg} columns -> scale.json')

    # ---- 4. 常量列断言 ----
    const_union = sorted(const_union)
    if const_union != sorted(EXPECTED_CONST):
        die('常量列集合与附录 B 不一致\n'
            f'  actual  ({len(const_union)}): {const_union}\n'
            f'  expected({len(EXPECTED_CONST)}): {sorted(EXPECTED_CONST)}')
    print(f'\n[const] {len(const_union)} columns match appendix B')

    # ---- 5. schema / meta ----
    for dname, out in out_dirs.items():
        with open(os.path.join(out, 'schema.json'), 'w') as f:
            json.dump({'dataset': dname, 'columns': common,
                       'count': len(common), 'sha256': schema_sha},
                      f, indent=2, ensure_ascii=False)
        meta = {
            'dataset': dname,
            'columns': common,
            'column_count': len(common),
            'schema_sha256': schema_sha,
            'const_columns': const_union,
            'splits': {k.split('/')[1]: v for k, v in all_stats.items()
                       if k.startswith(dname + '/')},
        }
        with open(os.path.join(out, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'[meta] {out}/schema.json, meta.json')

    print('\n[OK] done')
    print(f'  HAI22 rows: train={all_stats["HAI22/train"]["rows"]} '
          f'test={all_stats["HAI22/test"]["rows"]}')
    print(f'  HAI23 rows: train={all_stats["HAI23/train"]["rows"]} '
          f'test={all_stats["HAI23/test"]["rows"]}')


if __name__ == '__main__':
    main()
