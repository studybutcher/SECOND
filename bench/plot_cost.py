"""从 results/bench_raw/*.json 画推理成本曲线

产出（写到 results/bench_raw/figs/）：
    latency_{dataset}.png   ms/窗口 vs batch（SECOND vs MUTANT，as_shipped / core_only 两栏）
    memory.png              峰值显存 vs batch（全数据集）
    flops_vs_C.png          FLOPs/窗口 & 参数量 vs 通道数 C

用法：
    python bench/plot_cost.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')                     # 无显示环境
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, 'results', 'bench_raw')
FIG_DIR = os.path.join(RAW_DIR, 'figs')
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']

try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'ieee'])
except Exception:
    pass

# 必须在 style.use 之后设置：scienceplots 会把 font.serif 覆盖成 Times（本机无此字体）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'DejaVu Sans'],
    'text.usetex': False,
    'font.size': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_records():
    recs = {}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, '*.json'))):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue
        if 'model' in r and 'cost' in r:
            recs[(r['model'], r['dataset'])] = r
    return recs


def curve(rec, variant):
    """返回 (batches, ms_per_window)，跳过出错点"""
    xs, ys = [], []
    for b, s in sorted(rec['latency'].get(variant, {}).items(), key=lambda kv: int(kv[0])):
        if 'ms_per_window' in s:
            xs.append(int(b))
            ys.append(s['ms_per_window'])
    return xs, ys


def setup_logx(ax, all_batches):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(all_batches))))
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('batch size')


def plot_latency(recs):
    made = []
    for d in DATASETS:
        a, b = recs.get(('SECOND', d)), recs.get(('MUTANT', d))
        if not a and not b:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=False)
        all_b = []
        for ax, variant in zip(axes, ['as_shipped', 'core_only']):
            for rec, label, style in ((a, 'SECOND', '-o'), (b, 'MUTANT', '-s')):
                if not rec:
                    continue
                xs, ys = curve(rec, variant)
                if not xs:
                    continue
                all_b += xs
                ax.plot(xs, ys, style, ms=3, lw=1.0, label=label)
            ax.set_yscale('log')
            ax.set_ylabel('ms / window')
            ax.set_title(variant)
            ax.legend(frameon=False, fontsize=7)
        if all_b:
            setup_logx(axes[0], all_b)
            setup_logx(axes[1], all_b)
        cfg = (a or b)['config']
        fig.suptitle(f'{d}   (C={cfg["input_dim"]}, W={cfg["window_length"]})', fontsize=9)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f'latency_{d}.png')
        fig.savefig(out)
        plt.close(fig)
        made.append(out)
    return made


def plot_memory(recs):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    all_b = []
    for ax, model in zip(axes, ['SECOND', 'MUTANT']):
        for d in DATASETS:
            rec = recs.get((model, d))
            if not rec:
                continue
            xs, ys = [], []
            for b, s in sorted(rec.get('memory', {}).items(), key=lambda kv: int(kv[0])):
                if 'inference_peak_MB' in s:
                    xs.append(int(b))
                    ys.append(s['inference_peak_MB'])
            if xs:
                all_b += xs
                ax.plot(xs, ys, '-o', ms=3, lw=1.0, label=d)
        ax.set_yscale('log')
        ax.set_ylabel('peak GPU mem (MB)')
        ax.set_title(model)
    if all_b:
        for ax in axes:
            setup_logx(ax, all_b)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'memory.png')
    fig.savefig(out)
    plt.close(fig)
    return [out]


def plot_flops_vs_C(recs):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ax, key, ylab in ((axes[0], 'flops_per_window_G', 'GFLOPs / window'),
                          (axes[1], 'n_params_M', 'params (M)')):
        for model, style in (('SECOND', '-o'), ('MUTANT', '-s')):
            xs, ys = [], []
            for d in DATASETS:
                rec = recs.get((model, d))
                if not rec:
                    continue
                xs.append(rec['config']['input_dim'])
                ys.append(rec['cost'][key])
            if xs:
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                ax.plot([xs[i] for i in order], [ys[i] for i in order],
                        style, ms=3, lw=1.0, label=model)
        ax.set_xlabel('channels C')
        ax.set_ylabel(ylab)
        ax.set_yscale('log')
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'flops_vs_C.png')
    fig.savefig(out)
    plt.close(fig)
    return [out]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    recs = load_records()
    if not recs:
        print('没有原始记录，请先跑 collect_cost_*.py')
        return
    made = plot_latency(recs) + plot_memory(recs) + plot_flops_vs_C(recs)
    for p in made:
        print('written:', p, f'({os.path.getsize(p)} bytes)')


if __name__ == '__main__':
    main()
