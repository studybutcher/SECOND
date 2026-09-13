"""MUTANT 推理成本原始数据采集（跨代码库对照）

产出 results/bench_raw/MUTANT_{dataset}.json，schema 与 collect_cost_second.py 一致。

注意：
  * MUTANT 的 att_encoder.h_n / s_n 是 (1, batch, hidden) 的 Parameter，
    换 batch 需切片加载。
  * returnA 会在 cwd 写 tmp_{pid}.txt / tmp2_{pid}.txt，故必须从临时目录运行，
    且 cwd 写入记录（文件系统差异直接影响延时）。
  * core_only 用「预计算邻接矩阵」替换 returnA：数学等价于计时目的
    （GCN 的 spmm 代价与邻接矩阵取值无关），只在计时窗口外做重活。

用法（从任意临时目录，例如 /tmp/mb）：
    CUDA_VISIBLE_DEVICES=<空闲卡> python /path/to/SECOND/bench/collect_cost_mutant.py
"""
import hashlib
import itertools
import json
import os
import platform
import statistics
import sys
import time

import torch

MUTANT_ROOT = '/home/sdb/yrlcd/anamolybench/MUTANT'
OUT_DIR = '/home/sdb/yrlcd/anamolybench/SECOND/results/bench_raw'
sys.path.insert(0, MUTANT_ROOT)

import Model as M                       # noqa: E402  MUTANT 的模型
from Model import MUTANT                # noqa: E402

# dataset -> (input_dim, window_length, out_dim, hidden, latent)  取自 MUTANT/main.py DATASET_CONFIG
CFG = {
    'SMAP': (25, 20, 5, 100, 100),
    'MSL':  (55, 20, 5, 100, 100),
    'SWaT': (51, 20, 5, 100, 100),
    'WADI': (127, 5, 3, 120, 120),
}
TRAIN_BATCH = 120
# 必须与 SECOND 逐点一致（collect_cost_second.py 用同一套）。
# 120 = MUTANT ckpt 的训练 batch，h_n/s_n 形状 (1,120,hidden) 刚好支持，无法再大。
BATCH_SWEEP = [1, 8, 32, 64, 120]
WARMUP, REPS = 5, 3
TARGET_MS = 2000.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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
    iters = int(max(5, min(100, target_ms / single_ms)))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1000)
    return ts, iters


def summarize(ts, iters=None):
    s = sorted(ts)
    d = {'mean': statistics.mean(ts),
         'std': statistics.stdev(ts) if len(ts) > 1 else 0.0,
         'min': min(ts),
         'p50': s[len(s) // 2],
         'p95': s[min(len(s) - 1, int(0.95 * len(s)))],
         'reps_ms': ts}
    if iters is not None:
        d['timed_iters'] = iters
    return d


def load_model(dataset, batch, ckpt, device='cuda'):
    C, W, OUT, H, L = CFG[dataset]
    model = MUTANT(C, C * OUT, H, L, batch, W, OUT)
    sd = torch.load(ckpt, map_location='cpu')
    sd = dict(sd)
    for k in ('att_encoder.h_n', 'att_encoder.s_n'):
        sd[k] = sd[k][:, :batch, :]
    model.load_state_dict(sd)
    return model.to(device).eval()


def measure_dataset(dataset):
    C, W, OUT, H, L = CFG[dataset]
    ckpt = os.path.join(MUTANT_ROOT, 'checkpoints', f'{dataset}_model.pt')
    rec = {
        'model': 'MUTANT',
        'source_repo': MUTANT_ROOT,
        'dataset': dataset,
        'config': {'input_dim': C, 'out_dim': OUT, 'window_length': W,
                   'hidden_size': H, 'latent_size': L, 'batch_size': TRAIN_BATCH},
        'protocol': {
            'harness_path': os.path.abspath(__file__),
            'harness_sha256': sha256_file(os.path.abspath(__file__)),
            'input_shape_b1': [1, W, C],
            'unit': 'per_window_forward',
            'include_vae_head': True,
            'include_dataloader': False,
            'include_padding_batch': False,
            'eval_mode': True,
            'precision': 'fp32 + fp64(GCN spmm)',
            'warmup_iters': WARMUP, 'timed_iters': 'adaptive(target_ms=%.0f)' % TARGET_MS,
            'repeats': REPS, 'batch_sweep': BATCH_SWEEP,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'tf32_enabled': torch.backends.cuda.matmul.allow_tf32,
            'torch_num_threads': torch.get_num_threads(),
            'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
            'mkl_num_threads': os.environ.get('MKL_NUM_THREADS'),
            'cwd': os.getcwd(),
            'note_returnA': 'returnA 在 cwd 写 tmp_{pid}.txt/tmp2_{pid}.txt 并读回',
        },
        'env': {'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                'torch': torch.__version__, 'platform': platform.platform()},
        'ckpt': {'path': ckpt, 'sha256': sha256_file(ckpt)},
    }

    # ---------- FLOPs (batch=1) ----------
    from torch.utils.flop_counter import FlopCounterMode
    m1 = load_model(dataset, 1, ckpt)
    x1 = torch.randn(1, W, C)
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        m1.reconstructed_probability(x1)
    total = fc.get_total_flops()
    ops_seen = sorted({str(k).split('.')[-1].rstrip("'>") for k in fc.get_flop_counts()['Global']})
    n_params = sum(p.numel() for p in m1.parameters())
    rec['cost'] = {
        'n_params': n_params, 'n_params_M': n_params / 1e6,
        'weight_mem_MB': mb(sum(p.numel() * p.element_size() for p in m1.parameters())),
        'flops_per_window': total, 'macs_per_window': total / 2,
        'flops_per_window_G': total / 1e9, 'macs_per_window_M': total / 2 / 1e6,
        'flops_per_timestep_G': total / W / 1e9,
        'counter': 'torch.utils.flop_counter.FlopCounterMode',
        'counter_ops_seen': ops_seen,
        'counter_ops_missing': ['spmm(GCN, fp64)', 'LSTM(encoder_lstm)'],
        'coverage_note': 'nn.LSTM 与稀疏 spmm 均不计入；手工公式见 PRD2 附录 B',
    }
    del m1
    torch.cuda.empty_cache()

    # ---------- 延时曲线 ----------
    lat = {'as_shipped': {}, 'core_only': {}}
    for B in BATCH_SWEEP:
        try:
            model = load_model(dataset, B, ckpt)
            x = torch.randn(B, W, C)
            with torch.no_grad():
                ts, it = bench(lambda: model.reconstructed_probability(x))
                lat['as_shipped'][str(B)] = summarize(ts, it)
            # core_only: 预计算邻接矩阵替换 returnA（计时窗口外做重活，数学等价）
            orig = M.returnA
            try:
                As = [torch.from_numpy(orig(x[i].cpu())) for i in range(B)]
                cyc = itertools.cycle(range(B))
                M.returnA = lambda i: As[next(cyc)]
                with torch.no_grad():
                    ts, it = bench(lambda: model.reconstructed_probability(x))
                    lat['core_only'][str(B)] = summarize(ts, it)
            finally:
                M.returnA = orig
            del model, x
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            for v in lat:
                lat[v].setdefault(str(B), {'error': type(e).__name__, 'detail': str(e)[:120]})
        finally:
            torch.cuda.empty_cache()
    for variant in lat:
        for B, s in lat[variant].items():
            if 'mean' in s:
                s['ms_per_window'] = s['mean'] / int(B)
                s['windows_per_s'] = 1000.0 / s['ms_per_window']
    rec['latency'] = lat

    # ---------- 显存 ----------
    mem = {}
    for B in BATCH_SWEEP:
        try:
            model = load_model(dataset, B, ckpt)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            after_load = torch.cuda.memory_allocated()
            x = torch.randn(B, W, C)
            with torch.no_grad():
                for _ in range(3):
                    model.reconstructed_probability(x)
            mem[str(B)] = {
                'after_load_MB': mb(after_load),
                'inference_peak_MB': mb(torch.cuda.max_memory_allocated()),
                'reserved_peak_MB': mb(torch.cuda.max_memory_reserved()),
                'activation_delta_MB': mb(torch.cuda.max_memory_allocated()) - mb(after_load),
            }
            del model, x
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            mem[str(B)] = {'error': type(e).__name__, 'detail': str(e)[:120]}
        finally:
            torch.cuda.empty_cache()
    rec['memory'] = mem

    return rec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print('GPU:', torch.cuda.get_device_name(0), '| cwd:', os.getcwd(), flush=True)
    for dataset in CFG:
        print(f'--- {dataset} ---', flush=True)
        rec = measure_dataset(dataset)
        out = os.path.join(OUT_DIR, f'MUTANT_{dataset}.json')
        with open(out, 'w') as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        print(f'  params={rec["cost"]["n_params_M"]:.4f}M '
              f'FLOPs/win={rec["cost"]["flops_per_window_G"]:.9f}G -> {out}', flush=True)


if __name__ == '__main__':
    main()
