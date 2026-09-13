"""SECOND 推理成本原始数据采集

对每个数据集产出 results/bench_raw/SECOND_{dataset}.json，字段遵循
docs/PRD2_推理成本统计.md §3-P4 的 schema。

用法（在 SECOND/ 根目录）：
    CUDA_VISIBLE_DEVICES=<空闲卡> python bench/collect_cost_second.py
"""
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import torch

# 允许从任意 cwd 直接 `python bench/collect_cost_second.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Model import patchattmamba   # noqa: E402
from model.mamba import ModelArgs       # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# window_length 由 main.py:40-44 决定（SWaT 强制 35，其余 70）
# ckpt 说明：仓库内 20 个 ckpt 中只有 4 个与当前 model/Model.py 兼容且 W 一致；
#           WADI 唯一可用的 ckpt 是 W=140，与默认 W=70 形状不符，故用随机初始化。
#           权重取值不影响 FLOPs / 延时 / 显存（三者只由网络形状决定），已如实记录。
CFG = {
    'SMAP':      dict(C=25,  W=70,  op_batch=60, ckpt='checkpoint/casestudy补充/SMAP/best.pt'),
    'MSL':       dict(C=55,  W=70,  op_batch=60, ckpt='checkpoint/casestudy/MSL/best.pt'),
    'SWaT':      dict(C=51,  W=35,  op_batch=60, ckpt='checkpoint/casestudy补充/SWaT/best.pt'),
    'WADI':      dict(C=127, W=70,  op_batch=24, ckpt=None),
    # 补充记录：仓库里唯一存活的 WADI ckpt 是 W=140 的，实测 W 对成本影响 <2%（见文档），
    # 保留该记录以便和主表（W=70）交叉验证；不进 summary 主表。
    'WADI_W140': dict(C=127, W=140, op_batch=24, ckpt='checkpoint/1-casestudy/WADI/best.pt'),
}
OUT_DIR = os.path.join(REPO_ROOT, 'results', 'bench_raw')
OUT_DIM = 5
LATENT = 256
# 曲线采样点：必须与 MUTANT 完全一致（collect_cost_mutant.py 用同一套）。
# 120 是合法公共点：MUTANT 的 ckpt 训练时 batch=120，h_n/s_n 形状正好支持；
# SECOND 的 self.batch_size 在 forward 里未被使用（用的是运行时 len(x)），任意 batch 可跑。
# operational batch（SMAP/MSL/SWaT=60、WADI=24）两边不同，故不作为曲线点，只在 config 里记录。
BATCH_SWEEP = [1, 8, 32, 64, 120]
WARMUP, REPS = 10, 3
TARGET_MS = 2000.0      # 每个 rep 至少跑这么久；迭代数按实测单次耗时自适应


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_self():
    return sha256_file(os.path.abspath(__file__))


def mb(x):
    return x / 2 ** 20


def bench(fn, warmup=WARMUP, reps=REPS, target_ms=TARGET_MS):
    """迭代数按实测单次耗时自适应，保证每个 rep 至少跑 target_ms。返回 (各 rep 的 ms/iter, iters)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    single_ms = max((time.perf_counter() - t0) * 1000, 1e-3)
    iters = int(max(10, min(100, target_ms / single_ms)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1000)
    return ts, iters


def summarize(ts, iters=None):
    ts_sorted = sorted(ts)
    d = {
        'mean': statistics.mean(ts),
        'std': statistics.stdev(ts) if len(ts) > 1 else 0.0,
        'min': min(ts),
        'p50': ts_sorted[len(ts_sorted) // 2],
        'p95': ts_sorted[min(len(ts_sorted) - 1, int(0.95 * len(ts_sorted)))],
        'reps_ms': ts,
    }
    if iters is not None:
        d['timed_iters'] = iters
    return d


def core_only_forward(model, x):
    """P2 向量化版：数学等价，去掉逐样本 GPU->CPU 同步。"""
    with torch.no_grad():
        x_a, pred, _ = model.pipline(x, 'test')
        mu = pred['recon_mu']
        return (x_a - mu).abs().sum(dim=list(range(1, x_a.dim())))


def measure_dataset(dataset, device='cuda'):
    cfg = CFG[dataset]
    C, W, op_batch = cfg['C'], cfg['W'], cfg['op_batch']
    w_size = C * OUT_DIM
    ckpt = os.path.join(REPO_ROOT, cfg['ckpt']) if cfg['ckpt'] else None
    rec = {
        'model': 'SECOND',
        'dataset': dataset,
        'config': {'input_dim': C, 'out_dim': OUT_DIM, 'window_length': W,
                   'latent_size': LATENT, 'n_layer': 8, 'batch_size': op_batch,
                   'patch_size': [5, 7]},
        'protocol': {
            'harness_path': os.path.abspath(__file__),
            'harness_sha256': sha256_self(),
            'input_shape_b1': [1, W, C],
            'unit': 'per_window_forward',
            'include_vae_head': True,
            'include_dataloader': False,
            'include_padding_batch': False,
            'eval_mode': True,
            'precision': 'fp32',
            'warmup_iters': WARMUP, 'timed_iters': 'adaptive(target_ms=%.0f)' % TARGET_MS,
            'repeats': REPS, 'batch_sweep': BATCH_SWEEP,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
            'torch_num_threads': torch.get_num_threads(),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
            'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
            'cwd': os.getcwd(),
        },
        'env': {
            'gpu': torch.cuda.get_device_name(0),
            'cuda': torch.version.cuda,
            'torch': torch.__version__,
            'platform': platform.platform(),
        },
    }

    # ---------- 建模 + 加载 ----------
    args = ModelArgs()
    model = patchattmamba(C, w_size, LATENT, op_batch, W, OUT_DIM, args=args)
    if ckpt:
        torch.manual_seed(0)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        rec['ckpt'] = {'path': os.path.abspath(ckpt), 'sha256': sha256_file(ckpt),
                       'weights_loaded': True}
    else:
        rec['ckpt'] = {'path': None, 'sha256': None, 'weights_loaded': False,
                       'note': ('无与当前 Model.py 形状兼容的 ckpt（仓库内 WADI ckpt 为 W=140，'
                                '默认配置为 W=70）→ 随机初始化。权重取值不影响 FLOPs/延时/显存，'
                                '三者只由网络形状决定。')}
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    rec['cost'] = {
        'n_params': n_params,
        'n_params_M': n_params / 1e6,
        'weight_mem_MB': mb(sum(p.numel() * p.element_size() for p in model.parameters())),
    }

    # ---------- FLOPs / MACs (batch=1) ----------
    from torch.utils.flop_counter import FlopCounterMode
    x1 = torch.randn(1, W, C, device=device)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        model(x1)
    total = fc.get_total_flops()
    ops_seen = sorted({str(k).split('.')[-1].rstrip("'>") for k in fc.get_flop_counts()['Global']})
    blk = FlopCounterMode(display=False)
    with torch.no_grad(), blk:
        for layer in model.attmamba:
            layer(torch.randn(1, C, LATENT * 2, device=device))
    blk_flops = blk.get_total_flops()
    rec['cost'].update({
        'flops_per_window': total,
        'macs_per_window': total / 2,
        'flops_per_window_G': total / 1e9,
        'macs_per_window_M': total / 2 / 1e6,
        'flops_per_timestep_G': total / W / 1e9,
        'midstack_flops_per_window_G': blk_flops / 1e9,
        'midstack_flops_share': blk_flops / total,
        'counter': 'torch.utils.flop_counter.FlopCounterMode',
        'counter_ops_seen': ops_seen,
        'counter_ops_missing': ['einsum(broadcast)', 'pscan(inplace)', 'exp/silu/softplus'],
        'coverage_note': ('einsum 是广播逐元素乘、pscan 全程 in-place，均不计入；'
                          '解析补算见 PRD2 附录 A，约占 1~3%，方向为低估 Mamba'),
    })

    # ---------- 延时曲线 ----------
    batches = sorted(BATCH_SWEEP)   # 与 MUTANT 逐点一致；op_batch 只在 config 里记录
    lat = {'as_shipped': {}, 'core_only': {}}
    for B in batches:
        try:
            x = torch.randn(B, W, C, device=device)
            with torch.no_grad():
                ts, it = bench(lambda: model.reconstructed_probability(x))
                lat['as_shipped'][str(B)] = summarize(ts, it)
                ts, it = bench(lambda: core_only_forward(model, x))
                lat['core_only'][str(B)] = summarize(ts, it)
            del x
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            for v in lat:
                lat[v][str(B)] = {'error': type(e).__name__, 'detail': str(e)[:120]}
        finally:
            torch.cuda.empty_cache()
    for variant in lat:
        for B, s in lat[variant].items():
            if 'mean' in s:
                s['ms_per_window'] = s['mean'] / int(B)
                s['windows_per_s'] = 1000.0 / s['ms_per_window']
    rec['latency'] = lat

    # ---------- 中间块微基准 ----------
    blk_lat = {}
    for B in batches:
        try:
            z = torch.randn(B, C, LATENT * 2, device=device)
            with torch.no_grad():
                ts, it = bench(lambda: [layer(z) for layer in model.attmamba])
            s = summarize(ts, it)
            s['ms_per_window'] = s['mean'] / B
            blk_lat[str(B)] = s
            del z
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            blk_lat[str(B)] = {'error': type(e).__name__, 'detail': str(e)[:120]}
        finally:
            torch.cuda.empty_cache()
    rec['midstack_latency'] = blk_lat

    # ---------- 显存 ----------
    mem = {}
    for B in batches:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            after_load = torch.cuda.memory_allocated()
            x = torch.randn(B, W, C, device=device)
            with torch.no_grad():
                for _ in range(3):
                    model.reconstructed_probability(x)
            mem[str(B)] = {
                'after_load_MB': mb(after_load),
                'inference_peak_MB': mb(torch.cuda.max_memory_allocated()),
                'reserved_peak_MB': mb(torch.cuda.max_memory_reserved()),
                'activation_delta_MB': mb(torch.cuda.max_memory_allocated()) - mb(after_load),
            }
            del x
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            mem[str(B)] = {'error': type(e).__name__, 'detail': str(e)[:120]}
        finally:
            torch.cuda.empty_cache()
    rec['memory'] = mem

    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('GPU:', torch.cuda.get_device_name(0))
    wanted = sys.argv[1:] or list(CFG)          # 用法: python collect_cost_second.py WADI_W140
    for dataset in wanted:
        print(f'--- {dataset} ---', flush=True)
        rec = measure_dataset(dataset)
        out = os.path.join(OUT_DIR, f'SECOND_{dataset}.json')
        with open(out, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f'  params={rec["cost"]["n_params_M"]:.2f}M '
              f'FLOPs/win={rec["cost"]["flops_per_window_G"]:.4f}G '
              f'-> {out}', flush=True)


if __name__ == '__main__':
    main()
