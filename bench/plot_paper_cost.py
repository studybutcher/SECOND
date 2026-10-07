"""Fig. X (延时) / Fig. Y (峰值显存) —— 论文正文图

规格来源：docs/PRD4_FigX_FigY_绘图规格.md（v1，2026-09-23）
数据来源：results/bench_raw/*.json（不含 SECOND_WADI_W140.json）
产出：plots/paper/figX_latency_vs_batch.{pdf,png}
      plots/paper/figY_memory_vs_batch.{pdf,png}

与 bench/plot_cost.py 的区别：那个是内部检查图（scienceplots/Times/网格线），
本脚本是投稿图（Arial 系 / 无网格 / 去上右边框 / restrained 调色板）。**不要合并两者。**

用法：
    python bench/plot_paper_cost.py
"""
# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) 参数扫描折线图 (latency vs batch) → assets/figures/LineTrend/plot_sweep.py → param inherit
# (b) 参数扫描折线图 (memory vs batch) → assets/figures/LineTrend/plot_sweep.py → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')

# ============================================================================
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
# ============================================================================
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# ============================================================================
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
# ============================================================================
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# ============================================================================
# Academic Figure Skill Export Baseline — COPY VERBATIM
# ============================================================================
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
    # 默认 pad_inches=0.1 会把成品推到 185.5 mm，超出期刊 183 mm 上限；
    # 收紧到 0.02 in（0.5 mm/侧）后成品 181.4 mm，留 1.6 mm 余量。
    "savefig.pad_inches": 0.02,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


def save_paper_figure(fig, filename):
    """本图对上面那个函数的偏离：**故意不用 bbox_inches='tight'**。

    Fig. X 与 Fig. Y 在正文里上下叠放，两者宽度必须逐像素一致。tight bbox 按实际
    内容裁剪，而两图的 y 轴刻度标签宽度不同（Fig. X 是 '1 / 10'，Fig. Y 是
    '10 / 100 / 1000 / 10000'），裁剪后实测 137.9 mm 与 136.7 mm —— 差 1.2 mm，
    并排看会明显不齐。
    改用固定画布 + 显式 subplots_adjust：两图都恰好 140 x 92 mm，与 figsize 一致，
    也符合 skill 的「尺寸在创建时定死，禁止后期裁剪缩放」。
    """
    # 注意：savefig(bbox_inches=None) 不等于"不裁剪" —— matplotlib 会把它回落到
    # rcParams['savefig.bbox']，而基线里那格正是 'tight'。必须用 rc_context 真正关掉。
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(f"{filename}.pdf", dpi=300)
        fig.savefig(f"{filename}.png", dpi=300)


# ============================================================================
# 本图规格覆盖：画布收窄拉高后字号整体上调
# 基线只锁字族（Arial 系）与风格（无网格/去边框），字号属于 skill 的 Class B
# 比例参数 —— 画布尺寸变了就该跟着变，不是对基线的破坏。
# ============================================================================
mpl.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
})
FS_ANNOT = 8                     # 图内标注字号

# ============================================================================
# 本图规格（PRD4 §5）
# ============================================================================
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.ticker import (FixedLocator, NullFormatter,        # noqa: E402
                               ScalarFormatter)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, 'results', 'bench_raw')
OUT_DIR = os.path.join(REPO_ROOT, 'plots', 'paper')

DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']
BATCHES = [1, 8, 32, 64, 120]
# 源数据/JSON 里的模型名是 SECOND，正文与图里统一叫 PESR（PRD4 §10 R5）
MODEL_ALIAS = {'SECOND': 'PESR', 'MUTANT': 'MUTANT'}

# 颜色 = 模型；marker = 数据集（两个正交通道，PRD4 §5）
MODEL_COLOR = {'PESR': CATEGORICAL[0], 'MUTANT': CATEGORICAL[1]}
DATASET_MARKER = {'SMAP': 'o', 'MSL': 's', 'SWaT': '^', 'WADI': 'D'}
# PESR 用空心 marker：MSL/SWaT 两条曲线几乎重合，空心才能看出是两条（PRD4 §10 R1）
MODEL_STYLE = {
    'PESR':   dict(linewidth=1.7, markersize=5.0, markerfacecolor='none',
                   markeredgewidth=1.1, alpha=1.00, zorder=3),
    'MUTANT': dict(linewidth=1.4, markersize=4.5, markerfacecolor=CATEGORICAL[1],
                   markeredgewidth=0.8, alpha=0.95, zorder=2),
}

MM = 1 / 25.4
FIGSIZE = (140 * MM, 92 * MM)            # 收窄拉高，比双栏 183 mm 窄 1/4
# 固定布局：显式边距，保证两图轴区位置与成品尺寸逐像素一致（配合 save_paper_figure）
SUBPLOTS = dict(left=0.155, right=0.985, top=0.95, bottom=0.26)

# Fig. Y 的权重基线带（after_load_MB 的跨数据集跨度，PRD4 §2.4）
BASELINE_BAND = (782.1, 886.5)
# Fig. Y 的逐窗口激活斜率（PRD4 §7.3）
ACT_SLOPES = {'SMAP': 8.7, 'MSL': 18.2, 'SWaT': 17.6, 'WADI': 38.9}


# ============================================================================
# 数据段 —— 用全部数据，不采样、不插值、不剔除（PRD4 §11 QA-9）
# ============================================================================
def load_records():
    """{('PESR'|'MUTANT', dataset): record}；剔除 WADI W=140 的交叉验证记录。"""
    recs = {}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, '*.json'))):
        if 'W140' in os.path.basename(path):
            continue                      # 交叉验证记录，不进主图（PRD4 §2.1）
        with open(path) as f:
            r = json.load(f)
        if 'model' not in r or 'cost' not in r:
            continue
        model = MODEL_ALIAS.get(r['model'])
        if model is None:
            continue
        recs[(model, r['dataset'])] = r
    return recs


def latency_series(rec):
    """(batches, ms_per_window) —— 注意取 ms_per_window，不是 mean（mean 是 ms/batch）。"""
    xs, ys = [], []
    lat = rec['latency']['as_shipped']
    for b in BATCHES:
        s = lat.get(str(b))
        if s and 'ms_per_window' in s:
            xs.append(b)
            ys.append(s['ms_per_window'])
    return xs, ys


def memory_series(rec):
    """(batches, inference_peak_MB)"""
    xs, ys = [], []
    mem = rec.get('memory', {})
    for b in BATCHES:
        s = mem.get(str(b))
        if s and 'inference_peak_MB' in s:
            xs.append(b)
            ys.append(s['inference_peak_MB'])
    return xs, ys


def setup_batch_axis(ax):
    """横轴 log2 + 五个采样点全打刻度（PRD4 §4.1）"""
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(BATCHES))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.85, 145)
    ax.set_xlabel('Batch size')


def add_marker_legend(fig, ax):
    """图例放在坐标区下方通栏横排 —— 保证任何布局下都不压数据（PRD4 §5.2 的稳健替代）。

    6 项：2 条线（模型=颜色）+ 4 个 marker（数据集）。
    """
    handles = [
        Line2D([], [], color=MODEL_COLOR['PESR'], linewidth=1.4, label='PESR'),
        Line2D([], [], color=MODEL_COLOR['MUTANT'], linewidth=1.1, label='MUTANT'),
    ]
    handles += [
        Line2D([], [], color=GREY, linestyle='none', marker=DATASET_MARKER[d],
               markersize=6, label=d)
        for d in DATASETS
    ]
    # 画布收窄到 140 mm 后 6 项单排会溢出行宽，改 3 列 2 行。
    # 用 figure 坐标而非 axes 坐标定位 —— 固定画布下这样才和 subplots_adjust 对得上。
    # y 锚点 0.138：x 轴标签底边在 figure 的 ~0.152 处，0.155 会与之贴住（实测重叠 1px）
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, 0.138), bbox_transform=fig.transFigure,
              ncol=3, frameon=False, fontsize=9, borderaxespad=0,
              handletextpad=0.6, columnspacing=2.0, labelspacing=0.5)


# ============================================================================
# Fig. X —— 端到端推理延时 vs batch
# ============================================================================
def fig_latency(recs):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for model in ('MUTANT', 'PESR'):          # MUTANT 先画，PESR 压在上层
        st = MODEL_STYLE[model]
        for d in DATASETS:
            rec = recs.get((model, d))
            if not rec:
                continue
            xs, ys = latency_series(rec)
            ax.plot(xs, ys, color=MODEL_COLOR[model], marker=DATASET_MARKER[d],
                    linestyle='-', label=f'{model}-{d}', **st)

    setup_batch_axis(ax)
    ax.set_yscale('log', base=10)
    ax.set_ylim(0.6, 60)
    ax.set_ylabel('Latency (ms / window)')

    # 交叉点锚线：实测交叉发生在 batch ∈ (1, 8]，x=8 是该区间右端点（PRD4 §7.1）
    ax.axvline(8, color=GREY, linestyle='--', linewidth=0.5, alpha=0.5, zorder=1)
    ax.text(9.0, 52, 'crossover', fontsize=FS_ANNOT, color=GREY,
            ha='left', va='top')

    add_marker_legend(fig, ax)
    fig.subplots_adjust(**SUBPLOTS)
    save_paper_figure(fig, os.path.join(OUT_DIR, 'figX_latency_vs_batch'))
    plt.close(fig)


# ============================================================================
# Fig. Y —— 峰值显存 vs batch
# ============================================================================
def fig_memory(recs):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # 权重基线带：截距是权重、斜率才是并行代价（PRD4 §7.2）
    ax.axhspan(*BASELINE_BAND, color=GREY, alpha=0.15, zorder=0, linewidth=0)
    ax.text(1.05, 725, 'weight baseline (PESR)', fontsize=FS_ANNOT, color=GREY,
            ha='left', va='top')

    for model in ('MUTANT', 'PESR'):
        st = MODEL_STYLE[model]
        for d in DATASETS:
            rec = recs.get((model, d))
            if not rec:
                continue
            xs, ys = memory_series(rec)
            ax.plot(xs, ys, color=MODEL_COLOR[model], marker=DATASET_MARKER[d],
                    linestyle='-', label=f'{model}-{d}', **st)

    # 逐窗口激活斜率 —— log 轴会掩盖"线性增长"，用数字把它补回来（PRD4 §4.3 / §7.3）
    # 单行横排而非分列：分列需要等宽字体对齐，会引入 DejaVuSansMono，违反全图统一 Arial 系
    # x 从 10 挪到 6、单位上移到标题行：8 pt 下原文字实测溢出画布右边界
    ax.text(6, 560, 'activation growth (MB / window)', fontsize=FS_ANNOT,
            color=CATEGORICAL[0], ha='left', va='top', fontweight='bold')
    ax.text(6, 350,
            '  '.join(f'{d} +{ACT_SLOPES[d]:.1f}' for d in DATASETS),
            fontsize=FS_ANNOT, color=CATEGORICAL[0], ha='left', va='top')

    setup_batch_axis(ax)
    ax.set_yscale('log', base=10)
    ax.set_ylim(8, 12000)
    ax.set_ylabel('Peak GPU memory (MB)')

    add_marker_legend(fig, ax)
    fig.subplots_adjust(**SUBPLOTS)
    save_paper_figure(fig, os.path.join(OUT_DIR, 'figY_memory_vs_batch'))
    plt.close(fig)


# ============================================================================
def print_verification_table(recs):
    """打印 PRD4 附录 A 的核对表 —— 出图前后各核一遍，防取数错。"""
    print('batch =', BATCHES)
    for name, fn in (('latency_ms_per_window (as_shipped)', latency_series),
                     ('memory_inference_peak_MB', memory_series)):
        print(f'\n{name}')
        for model in ('PESR', 'MUTANT'):
            for d in DATASETS:
                rec = recs.get((model, d))
                if not rec:
                    continue
                _, ys = fn(rec)
                print(f'  {model:6s} {d:5s} [' + ', '.join(f'{v:8.3f}' for v in ys) + ']')
    print('\nmemory_after_load_MB (权重基线)')
    for model in ('PESR', 'MUTANT'):
        vals = []
        for d in DATASETS:
            rec = recs.get((model, d))
            if rec:
                vals.append(f'{d} {rec["memory"]["1"]["after_load_MB"]:.1f}')
        print(f'  {model:6s} ' + ' | '.join(vals))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = load_records()
    if len(recs) != 8:
        raise SystemExit(f'期望 8 条记录（2 模型 x 4 数据集），实得 {len(recs)}：{sorted(recs)}')

    print_verification_table(recs)

    fig_latency(recs)
    fig_memory(recs)
    for stem in ('figX_latency_vs_batch', 'figY_memory_vs_batch'):
        for ext in ('pdf', 'png'):
            p = os.path.join(OUT_DIR, f'{stem}.{ext}')
            print(f'written: {p} ({os.path.getsize(p)} bytes)')


if __name__ == '__main__':
    main()
