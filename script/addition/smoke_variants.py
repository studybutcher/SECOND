#!/usr/bin/env python
"""A0 验收：四个变体 + 基线在 CPU 上前向一次，打印参数量与输出形状。

不训练、不读数据，只验证：
  1. 每个变体都能 import、构造、前向（形状与基线一致）；
  2. 参数量符合预期（尤其 nompa 必须清掉 MPA 死参数、nomssa 中间块为 0）。

用法：
    python script/addition/smoke_variants.py            # 全部数据集
    python script/addition/smoke_variants.py SMAP       # 单个
"""
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import get_data_dim                      # noqa: E402
from model.mamba import ModelArgs                   # noqa: E402

BACKBONES = ['mamba', 'nomssa', 'nompa']

# 与 main.py 的 ExpConfig 保持一致
DATASET_CFG = {
    'SMAP': dict(window_length=70, batch_size=60),
    'MSL':  dict(window_length=70, batch_size=60),
    'SWaT': dict(window_length=35, batch_size=60),
    'WADI': dict(window_length=70, batch_size=24),
}
OUT_DIM, LATENT = 5, 256


def build(dataset, backbone):
    cfg = DATASET_CFG[dataset]
    input_dim = get_data_dim(dataset)
    w_size = input_dim * OUT_DIM
    mod = importlib.import_module({
        'mamba':  'model.Model',
        'nomssa': 'model.Model_nomssa',
        'nompa':  'model.Model_nompa',
    }[backbone])
    model = mod.patchattmamba(input_dim, w_size, LATENT, cfg['batch_size'],
                              cfg['window_length'], OUT_DIM, args=ModelArgs())
    return model, input_dim, cfg


def main():
    torch.manual_seed(0)
    datasets = sys.argv[1:] or list(DATASET_CFG)
    rows, fails = [], []

    for ds in datasets:
        for bb in BACKBONES:
            torch.manual_seed(0)
            model, input_dim, cfg = build(ds, bb)
            n_all = sum(p.numel() for p in model.parameters())
            n_mid = sum(p.numel() for p in model.attmamba.parameters())
            x = torch.randn(2, cfg['window_length'], input_dim)
            try:
                model.eval()
                with torch.no_grad():
                    score = model.reconstructed_probability(x)
                shape_ok = tuple(score.shape) == (2,)
                err = ''
            except Exception as e:                      # noqa: BLE001
                shape_ok, err = False, f'{type(e).__name__}: {e}'
            rows.append((ds, bb, input_dim, n_all / 1e6, n_mid / 1e6,
                         n_all / 1e6 and 100 * n_mid / n_all, shape_ok, err))
            if err or not shape_ok:
                fails.append(f'{ds}/{bb}: {err or "score shape != (b,)"}')

    print(f'\n{"数据集":<6}{"变体":<9}{"C":>4}{"总参数/M":>11}{"中间块/M":>11}{"中间块占比":>10}{"前向":>6}')
    base = {}
    for ds, bb, c, n_all, n_mid, share, ok, err in rows:
        print(f'{ds:<6}{bb:<9}{c:>4}{n_all:>11.2f}{n_mid:>11.2f}{share:>9.1f}%{"OK" if ok else "FAIL":>7}')
        if bb == 'mamba':
            base[ds] = n_all
    print()
    for ds, bb, c, n_all, n_mid, share, ok, err in rows:
        if bb != 'mamba' and ds in base:
            print(f'  {ds:<6}{bb:<9} 相对基线 {n_all - base[ds]:+.2f} M '
                  f'({100 * (n_all - base[ds]) / base[ds]:+.1f}%)')

    if fails:
        print('\n[FAIL] ' + '\n       '.join(fails))
        return 1
    print('\n[OK] 全部分布式前向通过，形状与基线一致')
    return 0


if __name__ == '__main__':
    sys.exit(main())
