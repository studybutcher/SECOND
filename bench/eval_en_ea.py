#!/usr/bin/env python
"""PRD5.2 · 事后给 LSTM-AE 的每个 ckpt 算 E_N / E_A（`main.py` 不产出这两个量）。

定义（与 `PRD5.1_r3` 的 AE 实验对齐，便于并排）：

    E_N = 测试集里**正常窗**的逐元素 MSE（held-out、干净）
    E_A = 测试集里**异常窗**的逐元素 MSE
    标签口径 = any-in-window（两套实验一致）
    两者都在**该模型自己的归一化**里 ⇒ 跨模型只能比 **比值 E_A/E_N**（无量纲），
    不能比绝对 MSE（AE 用 train-fitted scaler，LSTM-AE 走 `utils.get_data` 的
    train/test 各自拟合 —— 后者是本仓/OmniAnomaly 的历史口径）。

窗口按实体切（不跨 stacked 接缝），与 AE 实验同源；test 按 main.py 的 35% 截断。
`E_A_clip` = 把输入与目标都 clip 到 [0,1] 再算（对付 MinMax 对 test 的外推）。

用法：python bench/eval_en_ea.py --ckpt-glob 'checkpoint/lstmae_robust/*/best.pt'
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_data, get_data_dim, preprocess              # noqa: E402
from bench.extract_repr import entity_blocks                      # noqa: E402
from bench.ae_data import windows_from                            # noqa: E402

REG = {'mamba': 'model.Model', 'nompa': 'model.Model_nompa', 'nomssa': 'model.Model_nomssa',
       'lstmae': 'model.Model_lstmae'}


@torch.no_grad()
def per_window_mse(model, win, batch=256, device='cuda', clip=None):
    """→ [N] 每窗的逐元素 MSE。clip 非 None 时把输入与目标都 clip 到该区间。"""
    model.eval()
    # forward 按 main.py 契约返回**标量 loss**；重建张量走 `_reconstruct`（本模型专有）。
    # AE 那一族的 forward 返回 (recon, ...) 元组，故两种都兼容。
    recon = getattr(model, '_reconstruct', None)
    out = []
    for i in range(0, len(win), batch):
        x = torch.as_tensor(win[i:i + batch], device=device)
        if clip is not None:
            x = x.clamp(*clip)
        r = recon(x) if recon is not None else model(x)[0]
        out.append(((r - x) ** 2).mean(dim=(1, 2)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-glob', default='checkpoint/lstmae_robust/*/best.pt')
    ap.add_argument('--backbone', default='lstmae')
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window', type=int, default=70)
    ap.add_argument('--out', default='results/lstmae_robust/en_ea.json')
    a = ap.parse_args()
    C = get_data_dim(a.dataset)
    W = a.window

    (train, _), (test, test_label) = get_data(a.dataset, None, None, train_start=0, test_start=0)
    ncut = int(test.shape[0] * 0.35)                 # main.py 的 val 截断
    test, test_label = test[:-ncut], test_label[:-ncut]
    blocks = [(nm, s, min(e, len(test)))
              for nm, s, e in entity_blocks(a.dataset, 'test', len(test) + ncut)]
    blocks = [b for b in blocks if b[2] - b[1] > W]
    meta = {'blocks_val': [dict(name=nm, off=s, n=e - s) for nm, s, e in blocks]}
    win, eid, ridx = windows_from(test, meta['blocks_val'], W)
    lab = (test_label[np.clip(ridx[:, None] + np.arange(W)[None, :], 0, len(test) - 1)]
           > 0.1).max(axis=1).astype(int)                # any-in-window
    print(f'{a.dataset}: test(65%) 窗={len(win)}  正常={int((lab == 0).sum())} '
          f'异常={int((lab == 1).sum())}  异常率={lab.mean():.4f}')

    import importlib
    mod = importlib.import_module(REG[a.backbone])
    rows = []
    for ck in sorted(glob.glob(a.ckpt_glob)):
        m = re.search(r'noise([\d.]+)_seed(\d+)', ck)
        if not m:
            continue
        nz, sd = float(m.group(1)), int(m.group(2))
        model = mod.patchattmamba(C, C * 5, 256, 60, W, 5).cuda()
        model.load_state_dict(torch.load(ck, map_location='cpu'))
        mse = per_window_mse(model, win)
        mse_c = per_window_mse(model, win, clip=(0.0, 1.0))
        e_n, e_a = mse[lab == 0].mean(), mse[lab == 1].mean()
        e_nc, e_ac = mse_c[lab == 0].mean(), mse_c[lab == 1].mean()
        rows.append(dict(eta=nz, seed=sd, ckpt=ck, E_N=float(e_n), E_A=float(e_a),
                         E_A_minus_E_N=float(e_a - e_n), ratio_E_A_over_E_N=float(e_a / e_n),
                         E_N_clip=float(e_nc), E_A_clip=float(e_ac),
                         E_A_minus_E_N_clip=float(e_ac - e_nc),
                         ratio_clip=float(e_ac / e_nc)))
        print(f'  η={nz:<5} seed={sd:<3} E_N={e_n:.6f} E_A={e_a:.6f} '
              f'E_A−E_N={e_a - e_n:+.6f} E_A/E_N={e_a / e_n:.3f} | '
              f'clip: E_N={e_nc:.6f} E_A={e_ac:.6f} 比值={e_ac / e_nc:.3f}')
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, 'w'), indent=1, ensure_ascii=False)
    print('saved:', a.out)


if __name__ == '__main__':
    main()
