"""缓存每个数据集的 init_score / test_score，供离线阈值扫描复用。

为什么要缓存：`pot_dtaad` 的阈值 ≡ `quantile(init_score, level) * scale`
（见 docs/阈值选择结果.md F5），因此扫 (level, scale) 网格**不需要重跑模型**。
把两组分数落盘一次，后续所有标定都是纯 numpy 运算。

产出：
    results/threshold/scores/SECOND_<dataset>.npz   init_score, test_score, test_label
    results/threshold/scores/SECOND_<dataset>.json  元数据（ckpt / W / batch / 形状）

用法：
    CUDA_VISIBLE_DEVICES=5 python bench/cache_scores.py --dataset SMAP
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as mainmod                                    # noqa: E402
from utils import get_data, get_loader                    # noqa: E402
from model.mamba import ModelArgs                         # noqa: E402

# 与 script/addition/threshold_eval.sh 完全一致的 ckpt 选择
CKPTS = {
    'SMAP': ('checkpoint/casestudy补充/SMAP/best.pt', []),
    'MSL':  ('checkpoint/casestudy/MSL/best.pt', []),
    'SWaT': ('checkpoint/casestudy补充/SWaT/best.pt', []),
    'WADI': ('checkpoint/1-casestudy/WADI/best.pt', ['--window_length', '140']),
}


def make_config(dataset):
    """复用 main.ExpConfig，保证 W / batch / latent 等默认值与原流程逐字一致。"""
    ckpt, extra = CKPTS[dataset]
    argv = sys.argv
    sys.argv = ['cache_scores', '--dataset', dataset, '--trainmode', 'test',
                '--loadpath', ckpt] + extra
    try:
        cfg = mainmod.ExpConfig()
    finally:
        sys.argv = argv
    return cfg, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(CKPTS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--init-n', type=int, default=-1,
                    help='init_score 取训练集前多少个窗口；<=0 = 全部（与 DTAAD 一致）')
    a = ap.parse_args()

    cfg, ckpt = make_config(a.dataset)
    mainmod.set_seed(a.seed)
    cfg.input_dim = mainmod.get_data_dim(cfg.dataset)

    (train_data, _), (test_data, test_label) = get_data(
        cfg.dataset, cfg.max_train_size, cfg.max_test_size,
        train_start=cfg.train_start, test_start=cfg.test_start)
    train_data = mainmod.add_noise(train_data, cfg.noise)

    # ---- 与 main.py 完全相同的切分 / 窗口化 ----
    n = int(test_data.shape[0] * cfg.val)
    test_data = test_data[:-n]
    test_label = test_label[:-n]
    W = cfg.window_length
    idx = np.arange(W)[None, :] + np.arange(train_data.shape[0] - W)[:, None]
    train_data = train_data[idx]
    idx = np.arange(W)[None, :] + np.arange(test_data.shape[0] - W)[:, None]
    test_data = test_data[idx]

    num_t = int(test_data.shape[0] / cfg.batch_size)
    con_t = test_data.shape[0] % cfg.batch_size

    test_loader = get_loader(test_data, batch_size=cfg.batch_size, window_length=W,
                             input_size=cfg.input_dim, shuffle=False)

    argsa = ModelArgs()
    import importlib
    _mod = importlib.import_module(mainmod.BACKBONE_REGISTRY[cfg.backbone])
    model = _mod.patchattmamba(cfg.input_dim, cfg.input_dim * cfg.out_dim,
                               cfg.latent_size, cfg.batch_size, W, cfg.out_dim, args=argsa)
    model = model.cuda()
    model.load_state_dict(torch.load(cfg.loadpath))
    # 注意：main.py 推理时从未调用 model.eval()，Dropout 是开的。
    # 这里保持同样的行为，才能与 docs/阈值选择结果.md 的数字对齐。

    print(f'--- {a.dataset}: W={W} batch={cfg.batch_size} test_windows={test_data.shape[0]} ---')
    t0 = time.time()
    test_score = model.is_anomaly(test_loader, num_t, con_t).detach().cpu().numpy()
    print(f'test_score: n={test_score.size}  {time.time()-t0:.1f}s  '
          f'range=[{test_score.min():.4f}, {test_score.max():.4f}]')

    init_score = mainmod.compute_init_score(model, train_data, cfg, a.init_n)
    init_score = np.asarray(init_score).ravel()
    print(f'init_score: n={init_score.size}  '
          f'range=[{init_score.min():.4f}, {init_score.max():.4f}]')

    out_dir = os.path.join('results', 'threshold', 'scores')
    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f'SECOND_{a.dataset}.npz')
    np.savez_compressed(npz, init_score=init_score, test_score=test_score,
                        test_label=np.asarray(test_label[-len(test_score):]))
    meta = {
        'model': 'SECOND', 'dataset': a.dataset, 'ckpt': os.path.abspath(cfg.loadpath),
        'window_length': W, 'batch_size': cfg.batch_size, 'latent_size': cfg.latent_size,
        'input_dim': cfg.input_dim, 'val': cfg.val, 'seed': a.seed,
        'init_score_n': int(init_score.size), 'init_score_requested': a.init_n,
        'test_score_n': int(test_score.size),
        'dropout_active': True,
        'note': 'main.py 未调用 model.eval()，分数含 Dropout 噪声，与阈值结果文档口径一致',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, f'SECOND_{a.dataset}.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print('saved:', npz)


if __name__ == '__main__':
    main()
