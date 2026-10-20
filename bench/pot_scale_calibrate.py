#!/usr/bin/env python
# coding=utf-8
"""PRD6.2：现有 ckpt 的 POT ③（验证集标定）——不重训。

对每个 (backbone, eta, seed)：
  1. 完全复现 main.py 的数据读取 / 加噪 / 划分 / 窗口化；
  2. 加载现有 best.pt（epoch 0）；
  3. init_score = 全训练集窗口分数（与 compute_init_score 一致）；
     val_score / test_score = 同一次前向（LSTM-AE 确定性；PESR 仍有 dropout）；
  4. base_th = pot_eval(init_score, test_score, level=0.98, scale=1.0) 的
     pot_dtaad 阈值；对 scale 网格 th = base_th * s；
  5. 算 val / test（完整 65%）/ held-out test（不含与 val 重叠的窗口）的
     point-adjust F1。

**只做前向，不训练、不改任何已有文件。** 结果写 --out-dir。

用法（单 run）：
  python bench/pot_scale_calibrate.py --model lstmae --noise 0.1 --seed 42
"""
import argparse
import importlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch.utils.data import DataLoader                       # noqa: E402
from utils import get_data                                    # noqa: E402
from eval_method import pot_eval, calc_seq_abs, bf_search     # noqa: E402
from main import add_noise, set_seed, BACKBONE_REGISTRY       # noqa: E402
from model.mamba import ModelArgs                             # noqa: E402


def make_loader(win, bs):
    """与 utils.get_loader 输出等价，但**不做 pad 拷贝**。

    get_loader 为对齐 batch 会反复 np.concatenate 整个数组（实测 MSL
    三个 split 合计 ≈70 s/run）；这里直接交给 DataLoader，最后一个
    batch 自然就是 con_* 大小，is_anomaly 的 p[:con_*] 是 no-op。
    """
    return DataLoader(torch.from_numpy(np.ascontiguousarray(win)),
                      batch_size=bs, shuffle=False)

# checkpoint 目录里的 η 写法（与历史产物一致：0.10 与 0.2 都不是 :g）
ETA_DIR = {0.0: '0', 0.05: '0.05', 0.10: '0.10',
           0.15: '0.15', 0.2: '0.2', 0.5: '0.5'}

CKPT_ROOT = {
    'lstmae': ('checkpoint/lstmae_robust_gauss',
               'lstmae_MSL_gauss_noise{eta}_seed{seed}/best.pt'),
    'pesr':   ('checkpoint/robust_ms',
               'pesr_MSL_gauss_noise{eta}_seed{seed}/best.pt'),
}


def parse_scales(s):
    """默认网格：对数均匀。

    base_th 随 η 从 ~149 涨到 ~2150，而“有用”的绝对阈值大致固定在
    O(10^2)，所以最优点 s* 会横跨 ~0.05–0.9。线性网格在低 s 端分辨率
    不够（η=0.10 的 s* 落在 0.1 与 0.2 之间），故用 geomspace：
    0.03–1.20 共 81 点（步长 ≈4.3%），外侧再放 0.01/0.02/1.5/2.0
    用于识别边界。
    """
    if s is None:
        g = [round(float(x), 4) for x in np.geomspace(0.03, 1.20, 81)]
        # 1.0 必须显式加入：它是 ①（固定参数）的对照点。
        return sorted(set(g + [0.01, 0.02, 1.0, 1.5, 2.0]))
    return [float(x) for x in s.split(',')]


def build_model(model_name, input_dim, batch_size, window_length, latent_size):
    """与 main.py:346-352 完全一致；**不调用 model.eval()**。

    注意：产物名是 `pesr`，但 BACKBONE_REGISTRY 里 PESR 的键是 `mamba`
    （它的模型类就是 `model.Model`）。
    """
    backbone = 'mamba' if model_name == 'pesr' else model_name
    argsa = ModelArgs()
    mod = importlib.import_module(BACKBONE_REGISTRY[backbone])
    model = mod.patchattmamba(input_dim, input_dim * 5, latent_size,
                              batch_size, window_length, 5, args=argsa)
    return model.cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['lstmae', 'pesr'])
    ap.add_argument('--noise', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--window-length', type=int, default=70)
    ap.add_argument('--batch-size', type=int, default=60)
    ap.add_argument('--latent-size', type=int, default=256)
    ap.add_argument('--val-ratio', type=float, default=0.35)
    ap.add_argument('--scales', default=None)
    ap.add_argument('--out-dir', default='results/pot_cal_no_retrain')
    ap.add_argument('--ckpt', default=None, help='显式指定 ckpt（默认按命名规则）')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    if torch.cuda.is_available() and args.device == 'cuda':
        torch.cuda.set_device(0)          # 由 CUDA_VISIBLE_DEVICES 选卡

    eta_s = ETA_DIR[args.noise]
    ckpt_root, ckpt_pat = CKPT_ROOT[args.model]
    ckpt = args.ckpt or os.path.join(
        ROOT, ckpt_root, ckpt_pat.format(eta=eta_s, seed=args.seed))
    if not os.path.exists(ckpt):
        raise SystemExit(f'[FAIL] ckpt 不存在: {ckpt}')

    cfg = SimpleNamespace(batch_size=args.batch_size,
                          window_length=args.window_length,
                          input_dim=55)
    W, bs = args.window_length, args.batch_size
    scales = parse_scales(args.scales)

    # ── 1. 与 main.py 相同的数据流程 ────────────────────────────────────
    set_seed(args.seed)
    (train_data, _), (test_data, test_label) = get_data(args.dataset)
    train_data = add_noise(train_data, args.noise, 'gauss')

    orig_test_len = test_data.shape[0]
    n = int(orig_test_len * args.val_ratio)
    test_data = test_data[:-n]
    test_label = test_label[:-n]
    val_data = test_data[-n:]
    val_label = test_label[-n:]
    truncated_len = test_data.shape[0]
    val_start_row = truncated_len - val_data.shape[0]   # val 在截断 test 中的起点

    train_win = train_data[np.arange(W)[None, :] +
                           np.arange(train_data.shape[0] - W)[:, None]]
    val_win = val_data[np.arange(W)[None, :] +
                       np.arange(val_data.shape[0] - W)[:, None]]
    test_win = test_data[np.arange(W)[None, :] +
                         np.arange(test_data.shape[0] - W)[:, None]]
    del train_data, val_data, test_data

    num_val, con_val = val_win.shape[0] // bs, val_win.shape[0] % bs
    num_t, con_t = test_win.shape[0] // bs, test_win.shape[0] % bs

    # ⚠️ val_loader 用 shuffle=False：main.py 的 val_loader 是 shuffle=True，
    #    但随后直接拿 val_score 对 val_label 算 F1（顺序错位）。标定必须用
    #    对齐的分数/标签，故这里只修正这一点，窗口与数据完全不变。
    val_loader = make_loader(val_win, bs)
    test_loader = make_loader(test_win, bs)
    train_loader = make_loader(train_win, bs)
    num_tr, con_tr = train_win.shape[0] // bs, train_win.shape[0] % bs

    # ── 2. 模型与 ckpt ─────────────────────────────────────────────────
    model = build_model(args.model, cfg.input_dim, bs, W, args.latent_size)
    state = torch.load(ckpt, map_location='cuda')
    model.load_state_dict(state)

    # ── 3. 分数 ────────────────────────────────────────────────────────
    init_score = model.is_anomaly(train_loader, num_tr, con_tr).detach().cpu().numpy()
    val_score = model.is_anomaly(val_loader, num_val, con_val).detach().cpu().numpy()
    test_score = model.is_anomaly(test_loader, num_t, con_t).detach().cpu().numpy()

    val_label_a = np.asarray(val_label[-len(val_score):]).ravel()
    test_label_a = np.asarray(test_label[-len(test_score):]).ravel()
    if len(val_score) != len(val_label_a) or len(test_score) != len(test_label_a):
        raise SystemExit('[FAIL] score/label 长度不一致')

    # held-out test：窗口 i 覆盖截断 test 的行 [i, i+W-1]；main.py 把窗口 i 的
    # 分数与标签行的第 i 个（从 W 开始）配对，故要同时保证窗口与配对标签都不
    # 落在 val 区间 [val_start_row, truncated_len)：
    #   i + W - 1 < val_start_row  AND  i + W < val_start_row
    # ⇒ n_free = val_start_row - W。
    n_free = max(0, val_start_row - W)
    free_score = test_score[:n_free]
    free_label = test_label_a[:n_free]

    # ── 4. base threshold（scale=1.0，与 PRD6.1 完全相同的 pot_eval 路径）──
    pres = pot_eval(init_score, test_score, test_label_a,
                    q=1e-3, level=0.98, scale=1.0,
                    modes=('pot_dtaad',))
    base_th = float(pres['modes']['pot_dtaad']['threshold'])
    init_thr = float(pres.get('init_threshold', float('nan')))
    base_metrics = pres['modes']['pot_dtaad']['metrics']

    # bf（oracle）对照：完整 test 复核历史值；held-out test 给出与 ③ 同一
    # 数据子集上的 oracle 上界（阈值仍由该子集标签选出，故是上界）。
    bf_test, bf_test_pct, _ = bf_search(test_score, test_label_a, step_num=700)
    if n_free:
        bf_free, bf_free_pct, _ = bf_search(free_score, free_label, step_num=700)
    else:
        bf_free, bf_free_pct = (float('nan'),) * 3, float('nan')

    # ── 5. scale 网格 ──────────────────────────────────────────────────
    rows = []
    for s in scales:
        th = base_th * s
        vm, _ = calc_seq_abs(val_score, val_label_a, th)
        tm, _ = calc_seq_abs(test_score, test_label_a, th)
        fm, _ = calc_seq_abs(free_score, free_label, th)
        rows.append({
            'scale': s,
            'threshold': th,
            'val_f1': float(vm[0]), 'val_p': float(vm[1]), 'val_r': float(vm[2]),
            'val_tp': int(vm[3]), 'val_fp': int(vm[5]), 'val_fn': int(vm[6]),
            'test_f1': float(tm[0]), 'test_p': float(tm[1]), 'test_r': float(tm[2]),
            'test_tp': int(tm[3]), 'test_fp': int(tm[5]), 'test_fn': int(tm[6]),
            'free_f1': float(fm[0]), 'free_p': float(fm[1]), 'free_r': float(fm[2]),
            'free_tp': int(fm[3]), 'free_fp': int(fm[5]), 'free_fn': int(fm[6]),
        })

    def _qs(x):
        qs = [0, 50, 90, 93, 95, 97, 99, 99.5, 99.9, 100]
        return {str(q): float(np.percentile(x, q)) for q in qs}

    out = {
        'model': args.model, 'dataset': args.dataset,
        'noise': args.noise, 'seed': args.seed,
        'ckpt': os.path.relpath(ckpt, ROOT),
        'window_length': W, 'batch_size': bs,
        'n_train_win': int(train_win.shape[0]),
        'n_val_win': int(val_win.shape[0]),
        'n_test_win': int(test_win.shape[0]),
        'n_free_win': int(n_free),
        'orig_test_len': int(orig_test_len),
        'truncated_test_len': int(truncated_len),
        'val_start_row': int(val_start_row),
        'init_threshold': init_thr,
        'base_threshold_scale1': base_th,
        'base_metrics_scale1': [float(x) for x in base_metrics],
        'bf_test': {'f1': float(bf_test[0]), 'precision': float(bf_test[1]),
                    'recall': float(bf_test[2]), 'percentile': float(bf_test_pct)},
        'bf_free': {'f1': float(bf_free[0]), 'precision': float(bf_free[1]),
                    'recall': float(bf_free[2]), 'percentile': float(bf_free_pct)},
        'init_score_size': int(pres['init_score_size']),
        'fit_ok': bool(pres['fit_ok']),
        'score_quantiles': {'val': _qs(val_score), 'test': _qs(test_score),
                            'free': _qs(free_score) if n_free else {}},
        'scales': rows,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    name = f'{args.model}_{args.dataset}_noise{eta_s}_seed{args.seed}.json'
    path = os.path.join(args.out_dir, name)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'[ok] {path}  init_thr={init_thr:.6f}  base_th={base_th:.6f}  '
          f'val_n={len(val_score)} test_n={len(test_score)} free_n={n_free}')
    best = max(rows, key=lambda r: r['val_f1'])
    print(f'     per-run val argmax: s={best["scale"]:g}  '
          f'valF1={best["val_f1"]:.4f}  testF1={best["test_f1"]:.4f}')


if __name__ == '__main__':
    main()
