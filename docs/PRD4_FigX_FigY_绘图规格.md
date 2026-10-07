# PRD 4：Fig. X / Fig. Y 绘图规格（推理延时 & 峰值显存 vs batch）

> 2026-09-23 ｜ 状态：**待评审**（评审通过后才进入绘图实现）
> 数据源：`docs/推理成本统计结果.md` §2.1（延时）、§3（显存）
> 原始数据：`results/bench_raw/SECOND_{SMAP,MSL,SWaT,WADI}.json`、`results/bench_raw/MUTANT_{SMAP,MSL,SWaT,WADI}.json`
> 关联文档：`docs/PRD2_推理成本统计.md`（测量协议）
> 画图 skill：`academic-figure-skill`（图型路由：`assets/figures/LineTrend/`，参数扫描折线）

---

## 0. 本 PRD 解决什么

在画之前把**风格、坐标系、视觉编码、辅助元素、导出规格**全部钉死，避免"改一版问一版"。评审通过后，实现脚本应当能一次性产出定稿。

**本 PRD 不定两件事**（已在本次决策中确认，直接沿用）：

| 决策项 | 结论 |
|---|---|
| Fig. X 延时口径 | **只画 `as_shipped`**（部署真实开销）。`core_only` 不进正文图，见 §10 风险 R3 |
| Fig. Y 显存口径 | **`inference_peak_MB` 峰值**，另加权重基线参考带（不画 `activation_delta_MB`） |
| 布局 | **单面板**，一张图内画完全部曲线（不分子图） |
| 风格 | **academic-figure-skill 基线**：Arial 无衬线、去上/右边框、无网格线、restrained 调色板 |

---

## 1. 两张图的科学主张（one figure, one message）

### Fig. X — 延时

> **主张**：PESR 在单窗口（batch=1）推理时比 MUTANT 慢，但 batch 并行收益大一个数量级，batch≥8 后反超。

支撑数字（`docs/推理成本统计结果.md` §2.3、§4）：

| 数据集 | b=1 慢多少 | b=120 快多少 | batch 收益 PESR | batch 收益 MUTANT |
|---|---|---|---|---|
| SMAP | 6.61× | 4.51× | 41.7× | 1.40× |
| MSL | 5.59× | 2.89× | 20.9× | 1.30× |
| SWaT | 5.62× | 2.90× | 21.3× | 1.31× |
| WADI | 3.98× | 2.02× | 9.5× | 1.18× |

**3 秒可读性检查**：图上必须一眼看出「左边红在上、右边蓝在上」——即两条曲线族在 batch≈8 处交叉。若做不到，这张图就失败了。

### Fig. Y — 峰值显存

> **主张**：PESR 的峰值显存随 batch 线性增长（并发窗口越多、激活越大），MUTANT 几乎持平——这是上图中 batch 并行收益的**代价**。

支撑数字（逐窗口激活斜率，由 §3 表算得）：

| 数据集 | PESR b=1→120 | 斜率 (MB/窗口) | MUTANT b=1→120 | 斜率 (MB/窗口) | 斜率比 |
|---|---|---|---|---|---|
| SMAP | 790.9 → 1831.0 | **+8.74** | 16.7 → 18.7 | +0.0162 | 539× |
| MSL | 830.4 → 2996.5 | **+18.20** | 17.1 → 19.9 | +0.0234 | 778× |
| SWaT | 822.6 → 2915.8 | **+17.59** | 17.0 → 19.4 | +0.0196 | 898× |
| WADI | 925.6 → 5556.6 | **+38.92** | 17.9 → 38.2 | +0.1711 | 228× |

> 斜率 = (peak@b=120 − peak@b=1) / 119，即每多并发一个窗口的激活增量。

**3 秒可读性检查**：蓝族四条明显向右上爬升，红族四条几乎水平压在底部；灰色基线带横在 782~887 MB。三者形成"截距—斜率"的清晰分层。

---

## 2. 数据溯源（逐点，绘图时照抄）

### 2.1 横轴采样点（两图共用）

```
batch = [1, 8, 32, 64, 120]
```

来源：`bench/collect_cost_*.py` 的 `BATCHES`；`summarize_cost.py` 已校验两库逐点一致（`docs/推理成本统计结果.md` §0 第 2 条）。

> ⚠️ **不含 operational batch**（PESR 为 60、WADI 24；MUTANT 为 120）。§0 第 3 条已明确"不作为曲线点"。

### 2.2 Fig. X 纵值 —— `latency.as_shipped.{batch}.ms_per_window`

| 模型 | 数据集 | b=1 | b=8 | b=32 | b=64 | b=120 |
|---|---|---|---|---|---|---|
| PESR | SMAP | 34.505 | 4.154 | 1.074 | 0.853 | **0.828** |
| PESR | MSL | 35.251 | 4.285 | 1.792 | 1.694 | **1.683** |
| PESR | SWaT | 34.574 | 4.189 | 1.674 | 1.640 | **1.623** |
| PESR | WADI | 35.751 | 4.306 | 3.769 | **3.691** | 3.768 |
| MUTANT | SMAP | 5.217 | 3.982 | 3.785 | 3.742 | **3.734** |
| MUTANT | MSL | 6.307 | 5.042 | 4.911 | 4.896 | **4.857** |
| MUTANT | SWaT | 6.153 | 4.899 | 4.748 | 4.745 | **4.704** |
| MUTANT | WADI | 8.976 | 7.747 | 7.648 | 7.619 | **7.621** |

单位 ms/窗口。**粗体 = 该行最小**（与 §2.1 一致）。

> ⚠️ JSON 里的 `latency.*.{b}.mean` 是**每 batch 总耗时**（ms/batch）；`ms_per_window` = mean / batch。绘图取 `ms_per_window`，**不要取 `mean`**。

### 2.3 Fig. Y 纵值 —— `memory.{batch}.inference_peak_MB`

| 模型 | 数据集 | b=1 | b=8 | b=32 | b=64 | b=120 |
|---|---|---|---|---|---|---|
| PESR | SMAP | 790.9 | 852.3 | 1061.7 | 1340.9 | 1831.0 |
| PESR | MSL | 830.4 | 959.3 | 1394.9 | 1977.0 | 2996.5 |
| PESR | SWaT | 822.6 | 946.3 | 1367.5 | 1931.3 | 2915.8 |
| PESR | WADI | 925.6 | 1199.0 | 2131.9 | 3378.1 | **5556.6** |
| MUTANT | SMAP | 16.7 | 16.8 | 17.2 | 17.8 | 18.7 |
| MUTANT | MSL | 17.1 | 17.2 | 17.7 | 18.3 | 19.9 |
| MUTANT | SWaT | 17.0 | 17.2 | 17.6 | 18.3 | 19.4 |
| MUTANT | WADI | 17.9 | 18.0 | 18.6 | 24.8 | 38.2 |

单位 MB，进程内 `torch.cuda.max_memory_allocated()`。

### 2.4 权重基线（Fig. Y 的参考带）

来自 `memory.{batch}.after_load_MB`（模型加载完成、尚未推理时的显存）：

| 数据集 | PESR 基线 (MB) | MUTANT 基线 (MB) |
|---|---|---|
| SMAP | 782.1 | 8.5 |
| MSL | 812.1 | 8.8 |
| SWaT | 804.5 | 8.8 |
| WADI | 886.5 | 9.3 |

> **PESR 基线带 = [782.1, 886.5] MB**（跨数据集不等，因为参数量随 C 变化）。MUTANT 基线 8.5~9.3 MB，与它的峰位（16.7~38.2）基本重合，**不单独画线**，否则底部两条灰线叠在一起。

**为什么必须画这条带**：PESR 在 b=1 的 790.9 MB 里，782.1 MB 是权重——**激活只占 8.8 MB**。不标基线，读者会误以为"PESR 单窗口就要 790 MB 激活"，这对你不公平；标了基线，才有"截距是权重、斜率才是并行代价"的正确读法。

---

## 3. 画布与布局

| 项 | 规格 |
|---|---|
| 版式 | 双栏通栏（double column） |
| 画布尺寸 | **183 mm × 75 mm**（`figsize=(183/25.4, 75/25.4)` ≈ 7.20 × 2.95 in） |
| 面板数 | 各 1 个（两张图都是单面板） |
| 边距 | `constrained_layout=True` 或 `tight_layout(pad=0.5)` |
| 图序 | Fig. X 在上、Fig. Y 在下，**两图等宽等高**，便于在正文中堆叠 |

> 高度 75 mm 是初值。定稿前按 §11 QA-6 在 100% 缩放下核一次标签是否碰撞；若 PESR 四条曲线在图内仍显拥挤，可放宽到 85 mm。

---

## 4. 坐标系规格

### 4.1 两张图共用的横轴

| 项 | 规格 |
|---|---|
| 轴类型 | **log，base = 2** |
| 刻度位置 | `FixedLocator([1, 8, 32, 64, 120])` —— 五个点全部打刻度，不自动采样 |
| 刻度标签 | `ScalarFormatter()` 原样输出 `1 / 8 / 32 / 64 / 120`（**不要**显示成 `2^0 2^3 …`） |
| 次刻度 | `NullFormatter()` 关闭 |
| 轴标签 | `Batch size`，8 pt |
| 轴范围 | `xlim = (0.85, 145)` —— 两端留白，避免 b=1 和 b=120 的 marker 被边框切掉 |

> **为什么用 log₂ 而不用等距 ordinal**：b=1→8 的陡降本身是结论（§2「batch 1→32 总耗时几乎恒定」）。等距轴会把这段压扁，"加 batch 是纯赚"的观感消失。

### 4.2 Fig. X 纵轴

| 项 | 规格 |
|---|---|
| 轴类型 | **log，base = 10** |
| 轴标签 | `Latency (ms / window)`，8 pt |
| 轴范围 | `ylim = (0.6, 60)` |
| 主刻度 | `1, 10`（外加自动次要刻度 `2,3,5,7`） |
| 数据跨度 | 0.828 ~ 35.751（约 43×） |

> y 轴用 log 的理由：PESR 在 b=1 是 35 ms、MUTANT 在 b=120 是 3.7 ms，线性轴上 b=1 那一段会把右侧全部压扁，交叉点看不出来。

### 4.3 Fig. Y 纵轴

| 项 | 规格 |
|---|---|
| 轴类型 | **log，base = 10** |
| 轴标签 | `Peak GPU memory (MB)`，8 pt |
| 轴范围 | `ylim = (8, 12000)` |
| 主刻度 | `10, 100, 1000, 10000` |
| 数据跨度 | 16.7 ~ 5556.6（约 333×） |

> ⚠️ **log 轴会掩盖"线性增长"**。补偿手段见 §7.3：在每条 PESR 曲线末端直接标注斜率 `+8.7 MB/win` 等——把"线性"这个结论用**数字**给出，不依赖轴的视觉斜率。
>
> **备选方案**（若作者坚持视觉上呈现"直线增长"）：Fig. Y 改线性 y、`ylim=(0, 6000)`。代价是 MUTANT 四条（16.7~38.2）全部压成贴 x 轴的一条线，**"MUTANT 几乎不涨"这一半信息丢失**。本 PRD 推荐 log。

---

## 5. 视觉编码方案

单面板 8 条曲线，用 **颜色编码模型 + marker 编码数据集**（两个正交通道，互不干扰）：

| 通道 | 编码对象 | 取值 |
|---|---|---|
| **颜色** | 模型 | PESR → `#2166AC`（蓝）｜ MUTANT → `#B2182B`（红） |
| **Marker** | 数据集 | SMAP → `o`(圆) ｜ MSL → `s`(方) ｜ SWaT → `^`(↑三角) ｜ WADI → `D`(菱) |
| **线型** | — | 两族均实线（不区分） |

**为什么蓝/红而不是红/绿**：蓝-红是全色盲类型下都可分辨的组合；红-绿对红绿色盲不可用。同时二者亮度差足够，**灰度打印下仍可分**。

**语义角色**：蓝 `#2166AC` 在本 skill 调色板中定义为「primary / 本文方法」——PESR 用它；红 `#B2182B` 定义为「强调 / 强对比」——MUTANT 作为对照用它。

### 5.1 逐条线的绘制参数

| 模型 | color | linewidth | markersize | markerfacecolor | alpha | zorder |
|---|---|---|---|---|---|---|
| PESR | `#2166AC` | 1.4 | 3.5 | **`none`（空心）** | 1.0 | 3 |
| MUTANT | `#B2182B` | 1.1 | 3.0 | `#B2182B`（实心） | 0.95 | 2 |

> **PESR 用空心 marker 是硬要求**，理由见 §10 风险 R1：PESR 的 MSL 与 SWaT 两条曲线几乎完全重合，空心 marker 能让人分辨出"这里其实有两条"。
>
> MUTANT 用实心、且线更细，是为了**压低红色饱和面积**（skill 规则：单一强调色不宜占大面积）。红族四条本来就平，细线足够。

### 5.2 直接标注优先于图例

按 skill 规则「legend inside plot → 建议 outside/direct labeling」：

- **图内直接标注**：在曲线族末端（b=120 右侧）放 `PESR` 与 `MUTANT` 两个文字标签，颜色同线色，8 pt bold，`ha='left'`。需要把 `xlim` 右端留到 145（§4.1 已留）。
- **图例**：仅用于说明 marker ↔ 数据集。位置 `loc='lower left'`，`frameon=False`，`fontsize=7`，`ncol=4`，只放四个 marker、无文字线——形如 `○ SMAP  □ MSL  △ SWaT  ◇ WADI`。

> 图例只放 marker 这一条很关键：如果把 8 条线全塞进图例，图例会占掉 1/4 画面。

---

## 6. 风格基线（COPY VERBATIM，放在脚本最顶部）

以下三块**逐字复制**，不得改写、不得省略、不得"写个差不多的"。

### 6.1 排版基线

```python
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
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
```

### 6.2 配色基线

```python
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
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
```

**取色**：PESR = `CATEGORICAL[0]`（`#2166AC`）；MUTANT = `CATEGORICAL[1]`（`#B2182B`）；辅助线/参考带 = `GREY`。

### 6.3 导出基线

```python
# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)
```

### 6.4 明确禁止

- ✗ `scienceplots` / `plt.style.use(['science','ieee'])`（本仓库 `bench/plot_cost.py` 目前在用，**论文图不用**）
- ✗ `tab10` / `jet` / `rainbow` / seaborn 默认色板 / Excel 默认色
- ✗ 四边框（top/right spine 必须去掉）
- ✗ 网格线（两图都不加）
- ✗ Times / 衬线字体

> 注意：`bench/plot_cost.py` 产出的 `results/bench_raw/figs/*.png` 是**内部检查图**，风格与本文档规格**不一致**——不要直接拿去投稿，也不要"为了统一"而改本规格。论文图另写脚本（§9）。

---

## 7. 辅助元素

### 7.1 交叉点标记（Fig. X）

在 `x = 8` 处画一条竖直虚线：`color=GREY, linestyle='--', linewidth=0.5, alpha=0.5, zorder=1`。

在上方加注文字 `crossover`（7 pt，`GREY`），位置 `x=8`、`y=55`，`ha='center'`。

**依据**：§2.3 实测比值——b=1 时 PESR 慢 3.98~6.61×；b=8 时比值降到 0.56~1.04（SMAP 最晚交叉，1.04）；b=32 已全部反超（0.28~0.49）。故交叉发生在 `(1, 8]` 区间内，`x=8` 是该区间的右端点，作为视觉锚点最保守、最不会误导。

### 7.2 权重基线带（Fig. Y）

横向灰带：`ax.axhspan(782.1, 886.5, color=GREY, alpha=0.15, zorder=0, linewidth=0)`。

右端标注文字 `weight baseline (PESR)`，7 pt，`color=GREY`，贴近带的下沿（`y≈760`，`va='top'`），避免与 SWaT 曲线（822.6）重叠。

**画法说明**：跨数据集基线略有差异（782.1/812.1/804.5/886.5），画成**一条带**而非四条线——四条细线在 75 mm 高、log 轴上会挤成一条，且徒增图例负担。带宽本身（104 MB）相对 y 轴跨度（8~12000）只有 1.7% 高度，视觉上就是"一条带"，诚实且不误导。

### 7.3 曲线末端斜率标注（Fig. Y）

在四条 PESR 曲线末端（b=120 右侧）直接标注逐窗口激活斜率：

| 数据集 | 标注文字 |
|---|---|
| SMAP | `+8.7 MB/win` |
| MSL | `+18.2 MB/win` |
| SWaT | `+17.6 MB/win` |
| WADI | `+38.9 MB/win` |

7 pt，`color=#2166AC`，`ha='left'`，垂直位置按曲线末值错开（SMAP 1831 / MSL 2996 / SWaT 2916 / WADI 5557，在 log 轴上足够分开，无需额外错位逻辑）。

> **MSL 与 SWaT 会重叠**（2996.5 vs 2915.8，log 轴上仅差 0.012 dex）。处理：给 SWaT 的标注加 `y` 方向 +25% 偏移，并用一条 0.4 pt 的细引线连回数据点。

### 7.4 误差棒

**两图都不画误差棒**，理由：

- 重复测量 `std/mean` 绝大多数 <1%（见 QA-4），误差棒长度会**小于 marker 直径**，画出来只是一团黑，纯噪声。
- 唯一例外：MUTANT SMAP @ b=8 的 `std/mean = 3.03%`。

处理：在 caption 或 methods 中一句话交代「repeated measurements, std/mean < 1%; the single exception (MUTANT, SMAP, b=8) is 3.0%」。

---

## 8. 导出规格

| 项 | 规格 |
|---|---|
| 矢量母版 | `figX_latency_vs_batch.pdf`、`figY_memory_vs_batch.pdf`（`pdf.fonttype=42`，字体嵌入） |
| 位图预览 | 同名 `.png`，300 dpi |
| 色彩模式 | RGB（不做 CMYK 转换） |
| 输出目录 | `plots/paper/`（与 `plots/` 下的 case-study 可视化分开；不写进 `results/bench_raw/figs/`，那里是内部检查图） |
| 命名 | 与正文图号对齐，投稿时按目标期刊改 `figX` → `fig3` 之类 |

**单栏备选**：若目标期刊或排版要求单栏（89 mm），把 `figsize` 改为 `(89/25.4, 70/25.4)` 并把字号整体降至 6~7 pt；**不要**把 183 mm 的图缩小使用（skill 硬规则：尺寸在创建时定死，禁止后期缩放）。

---

## 9. 绘图脚本结构

新建 `bench/plot_paper_cost.py`（**不改** `bench/plot_cost.py`，那个是内部检查图）。

```
1. 本规格 §6.1/6.2/6.3 三个 COPY VERBATIM 基线块（顺序：排版 → 配色 → 导出）
2. Asset Confirmation Table 注释块（见下）
3. 数据段：glob results/bench_raw/*.json → 剔除 SECOND_WADI_W140.json → 建 recs[(model, dataset)]
4. 取数：ms_per_window（注意 §2.2 的 mean/ms_per_window 陷阱）、inference_peak_MB、after_load_MB
5. figX() — 单面板双曲线族
6. figY() — 单面板双曲线族 + 基线带 + 斜率标注
7. save_cns_figure(fig, "plots/paper/figX_latency_vs_batch")
```

**脚本首部必须有**（`academic-figure-skill` 硬规则）：

```python
# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) 参数扫描折线图 (latency vs batch) → assets/figures/LineTrend/plot_sweep.py → param inherit
# (b) 参数扫描折线图 (memory vs batch) → assets/figures/LineTrend/plot_sweep.py → param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.
```

**为什么是 `param inherit` 而不是 `native run`**：`assets/figures/LineTrend/plot_sweep.py` 的**布局逻辑**（x = 扫描值、多序列、marker+line、单面板多曲线）与需求一致，但它的视觉参数（`fontsize=16`、`linewidth=3`、`usetex=True`、`helvetica`、无本规格的调色板）**与本规格冲突**，且数据是硬编码的另一篇论文的。故继承其**结构**，参数全部走本规格。

| 继承类别 | 从 `plot_sweep.py` 继承 | 被本规格覆盖 |
|---|---|---|
| Class A（硬参数） | — | 全部按 §5.1 重定 |
| Class B（比例参数） | 多序列同面板、marker+line 组合、`MaxNLocator(nbins=5)` 的刻度密度思路 | 字号 16→8；`linewidth` 3→1.4/1.1；`figsize` → 183×75 mm |
| Class C（逻辑参数） | `frameon=False` 图例、`tight_layout` | `usetex` 关闭、字体改 Arial、去掉 `ylim=[0,100]` 硬编码 |

---

## 10. 已知风险与待决项

### R1 — MSL 与 SWaT 曲线几乎重合（**高风险，必须处理**）

PESR 的 MSL 与 SWaT 两条线在全部 5 个 batch 点上相差 **<4%**（b=120：1.683 vs 1.623）；MUTANT 同样（4.857 vs 4.704）。原因是两个数据集的配置接近（C=55 vs C=51，§6.2 已证明 FLOPs 几乎严格正比于 C），**曲线接近是预期内的、不是测量错误**。

**为什么这是选单面板的代价**：若按数据集分 2×2 子图，每张子图各取各的 y 轴，重合问题不存在。

**处理（本 PRD 已内置）**：
1. PESR 用**空心** marker、MUTANT 用实心 —— 重合处仍能看出 marker 形状不同（§5.1）。
2. 曲线末端直接标注（§7.3），重合处标注错开 + 引线。
3. 在 caption 或正文补一句：MSL 与 SWaT 的 C 接近，故曲线重合属预期。

**兜底**：若终稿仍显缠绕，退回 2×2 分数据集布局（本 PRD 保留该备选，改动仅限 §3 与 §5.2，其余规格不变）。

### R2 — 单面板 8 条线的认知负荷

8 条线在一张图里，读者需要先建立"颜色=模型、marker=数据集"的解码规则。**缓解**：靠 §1 的"两族分离"结构——PESR 四条陡降、MUTANT 四条近乎水平，视觉上天然聚成两簇，读者的第一眼是"两簇"而非"八条"。这个结构成立，单面板就成立；若定稿时发现两簇没有分离开，说明设计失败，走 R1 的兜底方案。

### R3 — `as_shipped` 单口径与 `docs/推理成本统计结果.md` §5⑦ 冲突（**必须在正文处理**）

§5⑦ 明确要求双侧披露：只报 `as_shipped` 是"拿实现差异当卖点"，只报 `core_only` 是"藏掉部署时真实发生的开销"。本图按你的决策**只画 `as_shipped`**。

**这不是不能做，但不能只做图了事**，需要在正文/附录补齐：

| 位置 | 补什么 |
|---|---|
| Fig. X caption | 明示口径：「Latency is measured on the end-to-end deployment path (`as_shipped`), including per-sample scoring and, for MUTANT, on-the-fly graph construction I/O.」 |
| 正文 | 用 §4 的 `core_only @b=120` 比值行（2.95× ~ 11.83×，MUTANT 更快）**主动交代**：剥离实现性开销后 MUTANT 延时可低至 0.27~0.32 ms/窗口。 |
| 附录/SI | 放 `core_only` 的双栏对照图（可复用 `bench/plot_cost.py` 的 `latency_{dataset}.png` 逻辑，按 §6 风格重绘）。 |

**理由**：审稿人一定会问"你把 MUTANT 的磁盘 I/O 算进去公平吗"。主动在正文给出 `core_only` 数字，比被质疑后再补要好，且不削弱 Fig. X 的结论——因为 §5⑦ 的核心观点是「`as_shipped` 反映**部署现实**」，而部署时那部分 I/O **确实会发生**。

### R4 — Fig. Y 的 log 轴与"线性增长"表述

见 §4.3。用 §7.3 的斜率标注补偿。若审稿人仍质疑，改用线性 y 的出图（备选方案已给出）。

### R5 — PESR = SECOND 的命名映射

源数据与 JSON 里模型名是 `SECOND`，正文/caption 里是 `PESR`。**绘图脚本里做一层映射**（`{'SECOND': 'PESR', 'MUTANT': 'MUTANT'}`），不要改 JSON，也不要在图里混用两个名字。

### R6 — WADI 用随机初始化权重（`weights_loaded: false`）

§7 第 1 条记录：WADI 用的是随机初始化权重。**对张量形状决定的量（延时 / 显存）无影响**，但需要在 caption 或 methods 脚注说明，否则是审稿人可攻击点。

标准表述：`Delays and memory are shape-determined; WADI results use randomly initialized weights (see Methods), which does not affect FLOPs, latency, or peak memory.`

### 待决项（需你或评审确认）

| # | 待决 | 建议 |
|---|---|---|
| D1 | 图号最终是 `Fig. X` / `Fig. Y` 还是具体编号 | 投稿前统一替换，脚本里用变量 `FIGX_STEM` / `FIGY_STEM` 参数化 |
| D2 | 是否需要 Fig. Y 的线性 y 备选版一起产出 | 建议一次产出两版（log 主 + linear 备），投稿时二选一 |
| D3 | 是否加 `torch.compile` 的对照线（PRD2 §2.2：b=1 提速 4.18×） | 本图不含（数据只有 SMAP 单点）；若要用，另开 Fig. |
| D4 | 输出目录 `plots/paper/` 是否合适 | 可改；规格其余部分不依赖该路径 |

---

## 11. QA 自检清单（出图后逐项打勾）

| # | 检查项 | 判据 |
|---|---|---|
| QA-1 | 字体 | 全部 Arial 或 Liberation Sans，无回退到 DejaVu；无衬线 |
| QA-2 | 边框 | 只有 left + bottom spine，0.6 pt；无 top/right |
| QA-3 | 网格线 | 两图均无 |
| QA-4 | 取数正确性 | 抽样核对：PESR/SMAP/b=120 应为 `0.828`（不是 `99.379`）；PESR/WADI/b=120 显存应为 `5556.6` |
| QA-5 | 曲线条数 | 每图 8 条（4 数据集 × 2 模型），无遗漏、无重复 |
| QA-6 | 印刷尺寸可读性 | 在 183 mm × 75 mm、100% 缩放下，7 pt 刻度标签互不碰撞；PESR 末端标注不与曲线重叠 |
| QA-7 | 色盲可用性 | 灰度转换后，蓝族与红族仍可区分（PESR 空心 vs MUTANT 实心是冗余通道） |
| QA-8 | log 轴刻度 | 横轴五点在位、标签为 `1/8/32/64/120`（非 `2^n`）；纵轴主刻度为 10 的幂 |
| QA-9 | 无数据丢失 | 8×5 = 40 个延时点、40 个显存点、8 个基线点，全部进图；**无重采样、无插值、无剔除** |
| QA-10 | 导出 | PDF 母版字体已嵌入（`pdf.fonttype=42`）；PNG 为 300 dpi；两者内容一致 |
| QA-11 | 辅助元素 | Fig. X 的 `x=8` 竖线在位；Fig. Y 的 782.1~886.5 灰带与斜率标注在位 |
| QA-12 | 命名 | 图中模型名统一为 `PESR` / `MUTANT`，未出现 `SECOND` |

---

## 附录 A：数据核对表（绘图脚本应打印此表自检）

```
batch = [1, 8, 32, 64, 120]

latency_ms_per_window (as_shipped)
  PESR   SMAP  [34.505, 4.154, 1.074, 0.853, 0.828]
  PESR   MSL   [35.251, 4.285, 1.792, 1.694, 1.683]
  PESR   SWaT  [34.574, 4.189, 1.674, 1.640, 1.623]
  PESR   WADI  [35.751, 4.306, 3.769, 3.691, 3.768]
  MUTANT SMAP  [ 5.217, 3.982, 3.785, 3.742, 3.734]
  MUTANT MSL   [ 6.307, 5.042, 4.911, 4.896, 4.857]
  MUTANT SWaT  [ 6.153, 4.899, 4.748, 4.745, 4.704]
  MUTANT WADI  [ 8.976, 7.747, 7.648, 7.619, 7.621]

memory_inference_peak_MB
  PESR   SMAP  [ 790.9,  852.3, 1061.7, 1340.9, 1831.0]
  PESR   MSL   [ 830.4,  959.3, 1394.9, 1977.0, 2996.5]
  PESR   SWaT  [ 822.6,  946.3, 1367.5, 1931.3, 2915.8]
  PESR   WADI  [ 925.6, 1199.0, 2131.9, 3378.1, 5556.6]
  MUTANT SMAP  [  16.7,   16.8,   17.2,   17.8,   18.7]
  MUTANT MSL   [  17.1,   17.2,   17.7,   18.3,   19.9]
  MUTANT SWaT  [  17.0,   17.2,   17.6,   18.3,   19.4]
  MUTANT WADI  [  17.9,   18.0,   18.6,   24.8,   38.2]

memory_after_load_MB (权重基线)
  PESR   SMAP 782.1 | MSL 812.1 | SWaT 804.5 | WADI 886.5
  MUTANT SMAP   8.5 | MSL   8.8 | SWaT   8.8 | WADI   9.3
```

**来源**：由 `results/bench_raw/*.json` 现场提取（`SECOND_WADI_W140.json` 已剔除，它是交叉验证记录，不进主图）；数值与 `docs/推理成本统计结果.md` §2.1 / §3 逐格一致。

---

## 附录 B：与给定 caption 的一致性核对

| Caption 句子 | 图中哪一处支撑 | 是否成立 |
|---|---|---|
| "Latency is reported as milliseconds per window" | y 轴标签 `Latency (ms / window)` | ✅ |
| "PESR incurs higher latency for isolated single-window inference" | b=1 处红族在蓝族之上（5.22~8.98 vs 34.5~35.8） | ✅ |
| "but benefits substantially more from batched parallel execution" | 蓝族 b=1→120 降 9.5~41.7×，红族仅降 1.18~1.40× | ✅ |
| "as the batch size increases" | 横轴 1→120 | ✅ |
| "PESR exhibits increasing memory consumption as more windows are processed concurrently" | 蓝族单调上升（斜率 +8.7~+38.9 MB/win） | ✅ |
| "illustrating the memory cost associated with its improved batch-level parallel scalability" | 灰基线带 + 斜率标注共同说明：截距是权重、增量是并行代价 | ✅ |

> ⚠️ **建议补充的半句**：当前 Fig. Y 的 caption 没有交代"PESR 的显存上升伴随更低的单窗口延时"——即**代价换来的收益**。建议在 Fig. Y caption 末补一句指向 Fig. X，例如 `This memory growth is the cost of the batch-level parallelism that yields the latency advantage in Fig. X.` 否则 Fig. Y 单独看像是 PESR 的缺点陈列。

---

## 下一步

本 PRD 评审通过后：

1. 按 §9 新建 `bench/plot_paper_cost.py`
2. 跑通后按 §11 逐项 QA，把 QA 结果附在交付说明里
3. 产出 `plots/paper/figX_latency_vs_batch.{pdf,png}`、`plots/paper/figY_memory_vs_batch.{pdf,png}`
4. 附带**统计与可复现性报告**：n 定义（每个点是 1 次测量的 mean/batch，重复测量 std/mean < 1%）、GPU 型号（RTX 5880 Ada，独占串行）、环境变量（`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`）、权重来源（`ckpt.sha256`，WADI 为随机初始化）
