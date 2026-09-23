"""子序列粒度评测：把全数据集的评测切回子序列，产出可与 DTAAD 并排的行。

**背景**：DTAAD 的 `load_dataset` 硬编码只加载一个子序列（SMAP→`P-1_`、MSL→`C-1_`），
而本仓库在全数据集上评测。两者口径不同，数字无法并列（PRD3.1 §2 / G2）。

**两种子序列的取法**（结果等价，因为窗口建在同一段连续数据上）：

| 类别 | 判定 | 取法 |
|---|---|---|
| `full`    | `b <= evalL`           | 直接从缓存的 `test_score` 切片（零 GPU） |
| `partial` | `a < evalL < b`        | 补一次前向，只覆盖评估范围内的窗口 |
| `absent`  | `a >= evalL`           | 从**原始拼接数组**切片后补前向 |

`evalL = L - int(L * val)`，即 `main.py` 砍掉 val（35%）之后的评测长度。

**标签对齐**（与 `main.py` 一致）：窗口 `i` 覆盖行 `[i, i+W)`，预测的是**下一行** `i+W`，
故 `main.py` 用 `test_label[-len(score):]` = `test_label[W:]`。本脚本对子序列 `[a,b)` 用
`labels[a+W : b]` 与 `scores[a : b-W]`。

用法：
    CUDA_VISIBLE_DEVICES=2 python bench/subseries_eval.py --dataset SMAP
    CUDA_VISIBLE_DEVICES=2 python bench/subseries_eval.py --dataset SMAP --all
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import main as mainmod                                       # noqa: E402
from eval_method import bf_search                            # noqa: E402
from utils import get_loader                                 # noqa: E402
from model.mamba import ModelArgs                            # noqa: E402
from subseries_index import (build_index, CONCAT_PATH,        # noqa: E402
                             CONCAT_DIM, load_labels, dataset_config)

SCORES_DIR = os.path.join('results', 'threshold', 'scores')
OUT_DIR = os.path.join('results', 'threshold', 'subseries')
CKPTS = {
    'SMAP': ('checkpoint/casestudy补充/SMAP/best.pt', []),
    'MSL':  ('checkpoint/casestudy/MSL/best.pt', []),
}
VAL = 0.35


def f1_point(score, label, thresh):
    """带 point-adjust 的 F1（复用 eval_method 的语义）。"""
    from eval_method import calc_seq_abs
    met, _ = calc_seq_abs(score, label, float(thresh))
    return met


def load_model(dataset):
    ckpt, extra = CKPTS[dataset]
    argv = sys.argv
    sys.argv = ['subseries_eval', '--dataset', dataset, '--trainmode', 'test',
                '--loadpath', ckpt] + extra
    try:
        cfg = mainmod.ExpConfig()
    finally:
        sys.argv = argv
    mainmod.set_seed(cfg.seed)
    cfg.input_dim = mainmod.get_data_dim(cfg.dataset)
    W, bs = dataset_config(dataset)
    argsa = ModelArgs()
    mod = importlib.import_module(mainmod.BACKBONE_REGISTRY[cfg.backbone])
    model = mod.patchattmamba(cfg.input_dim, cfg.input_dim * cfg.out_dim,
                              cfg.latent_size, bs, W, cfg.out_dim, args=argsa).cuda()
    model.load_state_dict(torch.load(ckpt))
    return model, cfg, W, bs


def forward_slice(model, data_slice, cfg, W, bs):
    """对一段连续数据做窗口化前向，返回窗口分数。

    注意：拼接数组存的是 float64，而正常流程里 `utils.preprocess()` 会转 float32。
    这里直接加载绕过了那一步，故显式转换——否则 conv1d 会因 dtype 不匹配报错。
    """
    n = data_slice.shape[0] - W
    if n <= 0:
        return np.empty(0)
    data_slice = np.asarray(data_slice, dtype=np.float32)
    win = data_slice[np.arange(W)[None, :] + np.arange(n)[:, None]]
    loader = get_loader(win, batch_size=bs, window_length=W,
                        input_size=cfg.input_dim, shuffle=False)
    num_i, con_i = n // bs, n % bs
    s = model.is_anomaly(loader, num_i, con_i)
    return s.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['SMAP', 'MSL'])
    ap.add_argument('--all', action='store_true', help='评全部子序列（否则只评 DTAAD 那一个）')
    ap.add_argument('--level', type=float, default=None, help='POT level（默认取 v2 标定值）')
    ap.add_argument('--scale', type=float, default=None, help='POT scale（默认取 v2 标定值）')
    a = ap.parse_args()

    # v2 标定出的 (level, scale)（docs/阈值选择结果_对齐DTAAD.md §1）
    CAL = {'SMAP': (0.98, 1.04), 'MSL': (0.90, 1.04)}
    level = CAL[a.dataset][0] if a.level is None else a.level
    scale = CAL[a.dataset][1] if a.scale is None else a.scale

    L = np.load(CONCAT_PATH[a.dataset]).reshape(-1, CONCAT_DIM[a.dataset]).shape[0]
    evalL = L - int(L * VAL)
    print(f'--- {a.dataset}: 拼接总长 L={L}  评测范围 [0, {evalL})  val={VAL} ---')

    idx = build_index(a.dataset)
    npz = np.load(os.path.join(SCORES_DIR, f'SECOND_{a.dataset}.npz'))
    cached_score = npz['test_score'].astype(float)
    label_full = load_labels(a.dataset)
    init_score = npz['init_score'].astype(float)
    W, bs = dataset_config(a.dataset)
    pot_th = float(np.quantile(init_score, level)) * scale
    print(f'    level={level} scale={scale} → POT 阈值 = {pot_th:.6f}')
    print(f'    缓存分数长度 {cached_score.size}（应 = evalL - W = {evalL - W}）')

    # DTAAD 评的那一个子序列
    target = 'P-1' if a.dataset == 'SMAP' else 'C-1'
    names = list(idx) if a.all else [target]

    need_fwd = []
    for nm in names:
        st, en = idx[nm]
        if st >= evalL:
            need_fwd.append((nm, st, en - W))          # absent：全段窗口都在范围外
    model = None
    if need_fwd:
        print(f'    需补前向的子序列: {len(need_fwd)} 个')
        model, cfg, W_, bs_ = load_model(a.dataset)
        assert (W_, bs_) == (W, bs)

    full_arr = np.load(CONCAT_PATH[a.dataset]).reshape(-1, CONCAT_DIM[a.dataset])
    rows, t0 = [], time.time()
    sc_by_name, lb_by_name = {}, {}
    for nm in names:
        st, en = idx[nm]
        if en <= evalL:                                  # full：离线切片
            sc = cached_score[st:en - W]
            lb = label_full[st + W:en]
            how = 'cache'
        else:                                            # partial / absent：补前向
            hi = min(en, evalL) if st < evalL else en
            sc = forward_slice(model, full_arr[st:hi], cfg, W, bs)
            lb = label_full[st + W:hi]
            how = 'fwd'
        if sc.size == 0:
            continue
        sc_by_name[nm], lb_by_name[nm] = sc, lb
        orc, orc_th, _ = bf_search(sc, lb, step_num=700)
        pot_met = f1_point(sc, lb, pot_th)
        rows.append({
            'subseries': nm, 'range': [st, en], 'rows': en - st,
            'windows': int(sc.size), 'how': how,
            'anomaly_rate': float((lb > 0.1).mean()),
            'oracle': {'f1': float(orc[0]), 'precision': float(orc[1]),
                       'recall': float(orc[2]), 'threshold_pct': float(orc_th)},
            'pot': {'f1': float(pot_met[0]), 'precision': float(pot_met[1]),
                    'recall': float(pot_met[2]), 'threshold': pot_th},
        })
        print(f'  {nm:8s} rows={en-st:>5d} win={sc.size:>5d} [{how}] '
              f'anom={100*(lb>0.1).mean():5.2f}%  '
              f'oracle F1={orc[0]:.4f}  POT F1={pot_met[0]:.4f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = 'all' if a.all else target
    # 分数落盘：标定需要它，之后所有扫描都是纯 numpy
    npz_out = os.path.join(OUT_DIR, f'SECOND_{a.dataset}_{suffix}_scores.npz')
    np.savez_compressed(npz_out,
                        init_score=init_score,
                        **{r['subseries'] + '__score': np.asarray(sc_by_name[r['subseries']])
                           for r in rows},
                        **{r['subseries'] + '__label': np.asarray(lb_by_name[r['subseries']])
                           for r in rows})
    out = os.path.join(OUT_DIR, f'SECOND_{a.dataset}_{suffix}.json')
    with open(out, 'w') as fh:
        json.dump({'model': 'SECOND', 'dataset': a.dataset,
                   'eval_range': [0, evalL], 'concat_len': int(L),
                   'pot': {'level': level, 'scale': scale, 'threshold': pot_th},
                   'target': target if not a.all else None,
                   'subseries': rows,
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')},
                  fh, indent=2, ensure_ascii=False)
    print(f'\n耗时 {time.time()-t0:.1f}s   saved: {out}')


if __name__ == '__main__':
    main()
