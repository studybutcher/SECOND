#!/usr/bin/env python
"""抽取 PESR 三个阶段的表示，供表示层定量分析（PRD5 §3.3 / §5）。

三个 tap（**模型文件零改动**，全部走 hook）：
    Z_p  = 进入 MSSA 的张量      [B, C, 512]    —— attmamba[0] 的 forward_pre_hook
           （nomssa 的 attmamba 是空 ModuleList → 退回 project4 的 pre-hook）
    V_t  = 进入 CRR 之前的表示   [B, 275]      —— projection3 的 forward_hook
           （= C·out_dim = 55×5；三变体同维同义，是跨变体比较的唯一合法位置）
    mu   = CRR 编码器均值        [B, 256]      —— encoder 的 forward_hook，取 chunk(2,dim=1) 的前半

以及窗口级异常分数 score = |sum(x_a - recon_mu)|（与 Model.reconstructed_probability 同一式子）。

三条实现层约束（PRD5 §3.4），缺一条数就不可复现：
    1. model.eval()            —— selfattentionblock1d 里 Dropout(0.1) 是硬编码的
    2. batch_size=60, shuffle=False —— [B,512] 进 nn.MultiheadAttention 会被当成 (L=B, E)，
                                      表示依赖同 batch 的其它窗口
    3. 按实体切窗（MSL/SMAP）  —— 窗口不跨 stacked 布局的实体接缝

用法：
    CUDA_VISIBLE_DEVICES=5 python bench/extract_repr.py \
        --ckpt checkpoint/repr/nomssa_MSL_seed42/best.pt --backbone nomssa \
        --dataset MSL --split test --out results/repr/nomssa_MSL_seed42_test.npz
"""
import argparse
import hashlib
import importlib
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

RAW_DIR = '/data/sdc/yrlbkp/anomaly/data/SMAP_MSL'
STACKED = {'SMAP', 'MSL'}


def make_config(dataset, backbone, ckpt, window_length=None):
    """复用 main.ExpConfig，保证 W / batch / latent / val 的默认值与原流程逐字一致。"""
    argv = sys.argv
    sys.argv = ['extract_repr', '--dataset', dataset, '--trainmode', 'test',
                '--loadpath', ckpt, '--backbone', backbone]
    if window_length:
        sys.argv += ['--window_length', str(window_length)]
    try:
        cfg = mainmod.ExpConfig()
    finally:
        sys.argv = argv
    cfg.input_dim = mainmod.get_data_dim(cfg.dataset)
    return cfg


def entity_blocks(dataset, split, n_rows):
    """返回 [(name, t_start, t_end)]；非 stacked 数据集返回单实体。

    MSL/SMAP 的 .npy 是逐通道文件纵向拼接，块边界由 labeled_anomalies.csv 的通道顺序 +
    每通道在**该 split** 下的文件长度决定（train 与 test 的块长不同）。
    """
    if dataset not in STACKED:
        return [('all', 0, n_rows)]
    from bench.channel_split import stacked_blocks
    r = stacked_blocks(dataset, RAW_DIR, split=split)
    out = [(b['name'], b['t_start'], b['t_end']) for b in r['blocks']]
    if out[-1][2] != n_rows:
        raise SystemExit(f'[FAIL] 实体块末端 {out[-1][2]} != 数组行数 {n_rows}，'
                         f'块划分与数据不匹配')
    return out


def build_windows(arr, W, blocks):
    """逐实体切窗（不跨接缝）。返回 win_idx[N]（每窗起始行）、ent_id[N]、win[N,W,K]。"""
    idx, eid, names = [], [], []
    for name, s, e in blocks:
        n = e - s - W
        if n <= 0:
            print(f'  [skip] 实体 {name}: 长度 {e - s} < W={W}')
            continue
        i = np.arange(s, s + n)
        idx.append(i)
        eid.append(np.full(n, len(names), dtype=np.int16))
        names.append(name)
    if not idx:
        raise SystemExit('[FAIL] 没有任何实体能切出窗口')
    idx = np.concatenate(idx)
    eid = np.concatenate(eid)
    return idx, eid, names, arr[idx[:, None] + np.arange(W)[None, :]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backbone', default='mamba', choices=list(mainmod.BACKBONE_REGISTRY))
    ap.add_argument('--dataset', default='MSL')
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--window-length', type=int, default=None)
    ap.add_argument('--test-region', default='full', choices=['full', 'trimmed'],
                    help='test 用哪一段：full=整个测试集（本分析主口径）；'
                         'trimmed=main.py 的 65%% 区（与该表的 F1 同区，仅作对照）')
    ap.add_argument('--no-repeat-check', action='store_true',
                    help='跳过 Gate 0 的重复前向一致性检查')
    ap.add_argument('--train-save-stride', type=int, default=1,
                    help='train 只落盘每 N 行（前向仍跑全量，保住 batch 组成）；'
                         'SWaT 这类单实体的全量 train 是 49 万窗，落盘 3.8 GB 没必要')
    a = ap.parse_args()

    cfg = make_config(a.dataset, a.backbone, a.ckpt, a.window_length)
    W, bs = cfg.window_length, cfg.batch_size
    mainmod.set_seed(42)

    (train_data, _), (test_data, test_label) = get_data(
        a.dataset, None, None, train_start=cfg.train_start, test_start=cfg.test_start)

    if a.split == 'train':
        arr, label, blocks = train_data, None, entity_blocks(a.dataset, 'train', len(train_data))
    else:
        cut = int(len(test_data) * cfg.val) if a.test_region == 'trimmed' else 0
        arr = test_data[:len(test_data) - cut] if cut else test_data
        label = test_label[:len(arr)]
        blocks = entity_blocks(a.dataset, 'test', len(test_data))
        if cut:                       # 截断后最后一个实体被切短
            blocks = [(n, s, min(e, len(arr))) for n, s, e in blocks]
        blocks = [b for b in blocks if b[2] > b[1]]

    win_idx, ent_id, ent_names, win = build_windows(arr, W, blocks)
    N = win.shape[0]
    print(f'--- {a.dataset} {a.split} ({a.test_region})  W={W} bs={bs} '
          f'实体={len(ent_names)} 窗口={N} ---')

    # ---------- 模型与 hook ----------
    argsa = ModelArgs()
    _mod = importlib.import_module(mainmod.BACKBONE_REGISTRY[a.backbone])
    model = _mod.patchattmamba(cfg.input_dim, cfg.input_dim * cfg.out_dim, cfg.latent_size,
                               bs, W, cfg.out_dim, args=argsa).cuda()
    sd = torch.load(a.ckpt, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()                     # §3.4-2：必须关 dropout
    del sd

    for m in ('projection3', 'encoder'):
        if not hasattr(model, m):
            raise SystemExit(f'[FAIL] 模型缺 {m}，hook 无法挂载')
    if len(model.attmamba) > 0:
        zp_target, zp_kind = model.attmamba[0], 'attmamba[0]'
    elif hasattr(model, 'project4'):
        zp_target, zp_kind = model.project4, 'project4(flattened)'
    else:
        raise SystemExit('[FAIL] 既无 attmamba 也无 project4，无法取 Z_p')
    path = model.pipline if hasattr(model, 'embedding_inpatch') else model.simpip
    print(f'    hook: Z_p←{zp_kind}  V_t←projection3  mu←encoder   前向路径="{path.__name__}"')

    cap = {}

    def pre_zp(_m, inp):
        cap['zp'] = inp[0].detach()

    def post_vt(_m, _i, out):
        cap['vt'] = out.detach()

    def post_mu(_m, _i, out):
        cap['mu'] = out.detach()

    hs = [zp_target.register_forward_pre_hook(pre_zp),
          model.projection3.register_forward_hook(post_vt),
          model.encoder.register_forward_hook(post_mu)]

    n_batches = int(np.ceil(N / bs))
    loader = get_loader(win, batch_size=bs, window_length=W,
                        input_size=cfg.input_dim, shuffle=False)
    store = {k: [] for k in ('zp', 'vt', 'mu', 'score')}

    def run(loader_iter, keep):
        for bi, inputs in enumerate(loader_iter):
            cap.clear()
            x = inputs.cuda()
            with torch.no_grad():
                x_a, pred, _ = path(x, 'test')
            zp = cap['zp']
            if zp.dim() == 2:                       # project4 的输入是 [B, C*512]
                zp = zp.view(zp.shape[0], cfg.input_dim, -1)
            # §5.2 主口径：Z_p 按实体通道维 mean pooling → [B, 512]
            # （不池化是 [N,55,512] ≈ 6~8 GB，且近 28k 维 kNN 有严重距离集中问题）
            store['zp'].append(zp.float().mean(dim=1).cpu().numpy())
            store['vt'].append(cap['vt'].float().cpu().numpy())
            store['mu'].append(cap['mu'][:, :cfg.latent_size].float().cpu().numpy())
            if a.split == 'test':
                s = torch.abs((x_a - pred['recon_mu']).sum(dim=1))
                store['score'].append(s.float().cpu().numpy())
            if not keep and bi + 1 >= 2:
                break

    t0 = time.time()
    with torch.no_grad():
        run(loader, keep=True)
    el = time.time() - t0
    print(f'    前向完成 {el:.1f}s（{n_batches} batch）')

    out = {k: np.concatenate(v, 0)[:N] for k, v in store.items() if v}
    out['zp'] = out['zp'].reshape(N, -1)                     # [N, 512]（已按通道维池化）
    # ---------- Gate 0：重复前向一致性 ----------
    repeat_ok, repeat_max = None, None
    if not a.no_repeat_check:
        ref_vt = out['vt'][:min(2 * bs, N)].copy()
        store_backup = {k: list(v) for k, v in store.items()}
        for k in store:
            store[k] = []
        with torch.no_grad():
            run(get_loader(win, batch_size=bs, window_length=W,
                           input_size=cfg.input_dim, shuffle=False), keep=False)
        rep_vt = np.concatenate(store['vt'], 0)
        repeat_max = float(np.abs(rep_vt - ref_vt).max())
        repeat_ok = bool(repeat_max == 0.0)
        for k in store:
            store[k] = store_backup[k]
        print(f'    [Gate 0] 重复前向 max|ΔV_t| = {repeat_max:.3e}  '
              f'→ {"一致 ✅" if repeat_ok else "不一致 ❌（dropout 没关？）"}')

    for h in hs:
        h.remove()

    # ---------- 标签 ----------
    meta = dict(ckpt=os.path.abspath(a.ckpt), backbone=a.backbone, dataset=a.dataset,
                split=a.split, test_region=a.test_region, window_length=W, batch_size=bs,
                latent_size=cfg.latent_size, input_dim=cfg.input_dim, out_dim=cfg.out_dim,
                n_windows=int(N), n_entities=len(ent_names), entity_names=ent_names,
                entity_id=ent_id.astype(int).tolist(),
                win_start=int(win_idx[0]) if N else 0,
                eval_mode=True, shuffle=False, seed=42,
                zp_hook=zp_kind, zp_pool='mean over channel dim (done in analyze)',
                forward_path=path.__name__, forward_sec=round(el, 1),
                repeat_check_max_abs_diff=repeat_max, repeat_check_ok=repeat_ok,
                ckpt_bytes=os.path.getsize(a.ckpt),
                timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.split == 'test':
        lab_end = label[win_idx + W - 1].astype(np.int8)         # 窗口内最后一个点
        lab_any = np.array([label[i:i + W].max() for i in win_idx], dtype=np.int8)
        out.update(label_last=lab_end, label_any=lab_any)
        meta.update(prevalence_last=float(lab_end.mean()), prevalence_any=float(lab_any.mean()),
                    n_pos_last=int(lab_end.sum()), n_pos_any=int(lab_any.sum()))
        per = [(ent_names[e], int((ent_id == e).sum()),
                float(lab_any[ent_id == e].mean())) for e in range(len(ent_names))]
        meta['per_entity'] = [dict(name=n, n_win=nw, rate_any=rt) for n, nw, rt in per]
    if a.split == 'train' and a.train_save_stride > 1:
        sel = np.arange(0, N, a.train_save_stride)
        out = {k: v[sel] for k, v in out.items()}
        meta['n_windows_saved'] = int(len(sel))
        meta['train_save_stride'] = a.train_save_stride
        print(f'    train 落盘 stride={a.train_save_stride}: {N} → {len(sel)} 行')
    else:
        meta['n_windows_saved'] = int(N)
    out['win_start'] = win_idx.astype(np.int64)[:len(out['vt'])] if a.split == 'train' \
        else win_idx.astype(np.int64)
    out['entity_id'] = ent_id[:len(out['vt'])] if a.split == 'train' else ent_id

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    with open(os.path.splitext(a.out)[0] + '.json', 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'    saved: {a.out}')
    for k in ('zp', 'vt', 'mu', 'score'):
        if k in out:
            arr_ = out[k]
            print(f'    {k}: {arr_.shape}  std={arr_.std():.4g}  '
                  f'nan={int(np.isnan(arr_).sum())}  inf={int(np.isinf(arr_).sum())}')
    if a.split == 'test':
        print(f'    prevalence any={meta["prevalence_any"]:.4f} last={meta["prevalence_last"]:.4f}')


if __name__ == '__main__':
    main()
