# PRD7 · HAI 22.04 + HAI 23.05 双数据集预处理与交叉泛化实验

> 目标:把 HAI 22.04 与 HAI 23.05 处理成**同一套列 / 同一套标签 / 同一套行序**，
> 使模型只吃两个数据集的公共部分；在此基础上做
> **A→A / B→B / A→B / B→A** 的交叉泛化实验，回答“在一年期数据上训练，
> 能否直接迁移到另一年期”的问题。
> 日期:2026-10-19
> 状态:**设计稿**（未开始实现）
> 修订:按 2026-10-19 确认的口径定稿——78 列全保留、主指标只用 bf F1、
> 只训 A/B 各一次（不设多 seed）、只跑 PESR。

---

## 0. 决策记录

| 项 | 决定 | 理由 |
|---|---|---|
| 数据版本 | `hai-22.04`、`hai-23.05`（**不用** `haiend-23.05`） | 两者都有 86 个过程量、同一套工艺对象；HAIEnd 多 139 个 DCS 内部控制量，与 22.04 无公共子集 |
| 输入空间 | 列名交集 **78 列全部保留**；其中 20 列在某一域内恒为常数，用“源域零范围 → 全零”保护，不删除 | 保持两域完全相同的 78 维 schema；若直接对常数列做源域 MinMax，目标域常数会被放大到 `5.2e5` 量级（如 `P1_PIT01_HH` 源域 0、目标域 524320），必须屏蔽 |
| 标签 | 统一为独立的 `label` 向量，`0=正常, 1=攻击`，只用于 test | 22.04 的标签是 CSV 内联 `Attack` 列，23.05 是独立的 `label-test*.csv` |
| 行序 | 按 `timestamp` 升序稳定排序；同一 split 内不得有重复时间戳 | 多文件拼接后必须保证时间单调；当前数据本身已单调，但契约仍要显式检查 |
| 列序 | 传感器列按**字典序**排列（`timestamp` 不在 npy 中，`label` 单独存） | 字典序与文件原始顺序无关，任何脚本/机器上都能复现同一 schema |
| 归一化 | **源域训练集 fit 的 MinMax**，应用到源域 train/test 和目标域 test；**不裁剪**；源域零范围列输出 0 | 交叉泛化要保留域偏移；零范围保护避免常数列把目标域输入放大数个数量级 |
| 验证集 | 沿用现有 `main.py` 的 **test 尾部 35%** | 现有 `best.pt` 就是按 val 上的 bf F1 选的；本轮不引入新的 val 协议，保持与旧表一致 |
| 主指标 | **bf_search F1**（与现有论文表格完全同一口径） | 用户要求统一;point-adjust 与 oracle 阈值的影响在结果里统一承受，不单独换指标 |
| 实验 | **训练 A、训练 B 各一次** → 得到 2 个 `best.pt`；4 次评测:A→A、B→B、A→B、B→A | 交叉评测只换 test 数据，不重新训练;不设多 seed，`seed=42` 固定 |
| 模型 | 只跑主模型 PESR（`--backbone mamba`） | 用户明确只跑这个模型 |

---

## 1. 数据现状（已实测，2026-10-19）

### 1.1 原始文件

两个目录均从 `icsdataset/hai` 以 Git LFS 拉取，18 个 CSV 已用 LFS
`oid sha256` 逐文件校验通过。

| 数据集 | split | 文件 | 行数 | 时间范围 |
|---|---|---|---|---|
| HAI 22.04 | train | `train1..6.csv` | **1,004,402** | 2021-07-11 10:00:00 → 2021-08-09 09:00:00 |
| HAI 22.04 | test | `test1..4.csv` | **361,200** | 2021-07-10 00:00:01 → 2021-07-17 00:00:00 |
| HAI 23.05 | train | `hai-train1..4.csv` | **896,400** | 2022-08-04 18:00:01 → 2022-08-25 00:00:00 |
| HAI 23.05 | test | `hai-test1..2.csv` | **284,400** | 2022-08-12 16:00:01 → 2022-08-19 16:00:00 |

标签:

| 数据集 | 标签位置 | test 正样本 | 正样本率 |
|---|---|---|---|
| HAI 22.04 | 每行最后一列 `Attack ∈ {0,1}` | 12,030 / 361,200 | **3.331%** |
| HAI 23.05 | `label-test1/2.csv` 的 `label ∈ {0,1}` | 11,384 / 284,400 | **4.003%** |

训练集两侧标签**全为 0**（已逐文件核对）。

### 1.2 结构与质量

- 列数:22.04 每个文件 **88 列** = `timestamp` + 86 过程量 + `Attack`；
  23.05 每个文件 **87 列** = `timestamp` + 86 过程量。
- 采样率:全部为 **1 秒/行**；每个文件内部严格单调、无重复时间戳、无 NaN、无 inf。
- 每个 CSV 是一个**连续采集片段**（segment），片段之间可能有数小时到数天的间隔。
  拼接时不能把片段边界当成连续时间。

### 1.3 片段（写窗口时必须知道）

| 数据集/split | 片段数 | 各片段起止 |
|---|---|---|
| 22.04 train | 6 | 07-11 10:00→07-12 12:00；07-17 00:00→07-19 08:00；07-23 18:00→07-25 05:00；07-26 12:00→07-27 12:00；08-02 15:00→08-05 09:00；08-06 09:00→08-09 09:00 |
| 22.04 test | 4 | 07-10 00:00→07-11 00:00；07-13 16:00→07-14 15:00；07-14 17:40→07-15 11:00；07-15 12:00→07-17 00:00 |
| 23.05 train | 4 | 08-04 18:00→08-08 00:00；08-13 07:00→08-16 16:00；08-19 16:00→08-21 03:00；08-22 17:00→08-25 00:00 |
| 23.05 test | 2 | 08-12 16:00→08-13 07:00；08-17 00:00→08-19 16:00 |

### 1.4 标签对齐的一个坑（23.05 test2）

`hai-test2.csv` 的时间戳是**秒级**（`2022-08-17 00:00:01`），
而 `label-test2.csv` 的时间戳被截断到了**分钟级**（`2022-08-17 0:00`），
且分钟值大量重复（230,400 行只有 3,841 个唯一时间戳）。

已验证:两者**行数相同、逐行按位置对齐**；把两边的 `timestamp` 都
`floor('min')` 后 230,400/230,400 完全一致。因此对齐规则必须是
**按行号对齐**，而不是按原始时间戳做 `merge`；时间戳只用于分钟级校验。

---

## 2. 预处理规格

### 2.1 列交集（78 列）

22.04 的 86 个过程量 ∩ 23.05 的 86 个过程量 = **78 列**。

- 仅 22.04 有（8 列，被排除）:
  `P1_B2004, P1_B2016, P1_B3004, P1_B3005, P1_B4002, P1_B4005, P1_B400B, P1_B4022`
- 仅 23.05 有（8 列，被排除）:
  `P1_PP04D, x1001_05_SETPOINT_OUT, x1001_15_ASSIGN_OUT, x1002_07_SETPOINT_OUT,
  x1002_08_SETPOINT_OUT, x1003_10_SETPOINT_OUT, x1003_18_SETPOINT_OUT, x1003_24_SUM_OUT`
- `timestamp` 单独作为索引，不进入模型；`Attack` / `label` 单独作为标签。

> 注意:22.04 的 8 个 `P1_B*` 列是锅炉侧量，23.05 没有；23.05 的 7 个
> `x100x_*` 是 HAIEnd/控制逻辑相关量。这些列**不能**用 0 或均值补齐后喂给模型，
> 因为那会向模型注入一个“域指示位”。直接丢弃是正确的交集语义。

### 2.2 常量列处理（78 列全部保留，零范围列置 0）

判定规则:**在 4 个 split（22-train / 22-test / 23-train / 23-test）中的任意一个里
`nunique() <= 1`**，就标记为“常量列”。常量列**不删除**，仍占模型输入的一维，
但必须做零范围保护（见 §2.6），否则源域 MinMax 会把目标域的常数放大成异常值。

被标记的 20 列及其常数取值:

| 列 | 22.04 常数 | 23.05 常数 |
|---|---:|---:|
| `P1_PIT01_HH` | 0 | 524320 |
| `P1_SOL01D` | 0 | 524320 |
| `P1_SOL03D` | 0 | 524320 |
| `P1_PP01AD` | 1 | 0 |
| `P1_PP01AR` | 1 | 0 |
| `P1_PP01BD` | 0 | 1 |
| `P1_PP01BR` | 0 | 1 |
| `P1_PP02D` | 1 | 1 |
| `P1_PP02R` | 1 | 1 |
| `P1_STSP` | 1 | 1 |
| `P2_Emerg` | 0 | 0 |
| `P2_OnOff` | 1 | 1 |
| `P2_RTR` | 2880 | 2880 |
| `P2_TripEx` | 1 | 1 |
| `P2_VTR01..04` | 50 | 50 |
| `P3_LH01` | 22.04 train 有 {70,80}，23.05 恒为 70 | — |
| `P3_LL01` | 22.04 train 有 {20,25}，23.05 恒为 30 | — |

说明:
- 前 18 列在**两个域里都**是常数；后 2 列只在 23.05 里是常数。
- `P1_PIT01_HH / P1_SOL01D / P1_SOL03D` 在两个域里的常数分别是 `0` 和 `524320`。
  若直接用源域 MinMax:源域 `max-min=0`（退化），目标域会被变换成 `524320`，
  比其余列的 `[0,1]` 大 5 个数量级，模型会立刻失稳。**所以必须有零范围保护。**
- 输入维度固定为 **78**，完整清单见附录 A。20 个常量列在附录 B 列出。
- 保护规则等价于“在该方向上把这 20 维恒置 0”，但仍保留 78 维 schema，
  使 A→B 和 B→A 能共用同一份模型配置与同一个 `input_dim=78`。

### 2.3 标签统一

输出 schema:

```
timestamp (仅索引) | <78 个传感器列，字典序> | label ∈ {0,1}
```

- 22.04:直接取 `Attack` 列，`label = Attack.astype(int8)`。
- 23.05:对每个 test 文件读取同名的 `label-test*.csv`，
  **按行号**与数据文件对齐；校验条件:
  1. `len(label) == len(data)`；
  2. 两边 `timestamp` 做 `floor('min')` 后逐行相等（test1 秒级、test2 分钟级都能过）；
  3. `set(label) ⊆ {0,1}`。
- 训练集标签由数据文件内部生成（全 0），不读外部 label 文件。

### 2.4 排序、拼接与分段

按 split 执行:

1. 逐个文件读入 → 保留 `file_id` 作为该行的 `segment_id`。
2. 纵向拼接所有文件。
3. 对 `timestamp` 做**稳定升序排序**（`kind='mergesort'`），
   同时携带 `segment_id` 与 `label`。
4. 断言:
   - `timestamp` 严格单调递增（无重复）；若有重复，保留首次出现并记录到日志，
     同时让验收失败——当前实测 4 个 split 均无重复。
   - 每个 `segment_id` 内部时间连续（相邻行差 1 秒），片段之间允许任意大间隔。
5. 传感器列按字典序重排；输出 npy 中**不包含** `timestamp` 列，
   时间信息单独存为 `*_timestamps.npy`（int64 纳秒），供审计使用。

**片段边界与滑窗**:

`main.py` 当前对整个 2D 数组直接做滑窗，
因此每个片段边界会产生 `W-1` 个跨边界窗口。W=70 时:

| split | 边界数 | 跨边界窗口数 | 占全部窗口 |
|---|---:|---:|---:|
| 22.04 train | 5 | 345 | 0.034% |
| 22.04 test | 3 | 207 | 0.057% |
| 23.05 train | 3 | 207 | 0.023% |
| 23.05 test | 1 | 69 | 0.024% |

**v1 决定:接受这些窗口**，因为它们占比 <0.06%，且只影响边界附近的训练/评分；
在结果里记录数量即可。v2 若要严格，应在 `main.py` 窗口化之前按
`segment_id` 切分，逐段窗口化后再拼接（每个窗口保留其 segment_id）。

### 2.5 数值清洗

当前数据实测:NaN=0、inf=0、非数值=0，因此清洗是**防御性**的:

- 读入后先 `to_numeric(errors='coerce')`；
- NaN → 0、±inf → 100（与 `utils._clean` 现有语义一致，不另造规则）；
- 任一列清洗后仍全为 NaN → 直接报错，不静默通过；
- 记录每列的 `[min,max]`、`nunique`、`nan_count` 到 `meta.json`。

### 2.6 归一化

**归一化在加载时执行，不在预处理脚本里烘焙进 npy。**

原因:交叉评测要拿**源域**的 min/max 去变换**目标域** test。若预处理阶段就把
每个域按自己的统计量归一化好，`HAI23_test.npy` 里存的是“用 23.05 统计量归一化”
的结果，A→B 时无法退回源域口径。因此:

```
npy 中存:   清洗后的原始数值（float32, 未归一化）
scale.json: 源域 train 的逐列 min/max/degenerate
加载时:     utils.get_data 读取源域 scale.json 后再变换
```

**主协议（P1，源域归一化）**:

```
fit:        X_source_train 的逐列 min/max      （预处理脚本计算并写入 scale.json）
transform:  训练时：source train、source test
            交叉时：target test 也用 source 的 scale.json
```

逐列公式:

```
scale_j = max_j - min_j
if scale_j == 0:   y_j = 0                        # degenerate_j = True
else:              y_j = (x_j - min_j) / scale_j
```

- **零范围保护**:源域训练集上 `max==min` 的列，在所有 split、所有域里一律输出 0。
  这一条对 78 列逐列执行;A→B 时 18 列命中，B→A 时 20 列命中（含 `P3_LH01/LL01`）。
- 不做裁剪。MinMax 在目标域上会产生 `[0,1]` 之外的值，这是被允许的，
  因为它保留了域偏移。
- 保存 `scale.json`:
  `{col: {"min": ..., "max": ..., "scale": ..., "degenerate": true/false}}`。
- 归一化统计量**只**在源域 train 上拟合；test（任何域）都不参与 fit。

实测目标域超出源域 `[min,max]` 的比例（只统计 58 个非常量列）:均值 **6.4%**，中位数 **0%**，
9 列超过 10%，最高的是 `P1_LIT01` **78.6%**、`P1_PCV02Z` **60.1%**、
`P2_SCST` **53.1%**。这组数字说明“源域 MinMax 直接套目标域”会产生明显的外推，
必须在论文里明确写出，并把外推比例作为诊断指标（§5.3）一起报。

**对照协议（P2，逐域归一化）—— 本轮不跑**:

- 每个域用自己的 train 统计量做 MinMax，再各自应用。
- 该协议把“边际分布差异”消掉，只留“结构/时序模式差异”，用于判断
  跨域性能下降有多少来自量纲、多少来自真正的模式变化。
- 它需要额外 4 次训练（每个域 × 2 个方向），与“只训 A、只训 B”的本轮范围冲突。
  **本轮只记录设计，不产出结果**;若 A→B / B→A 的 F1 接近 0，再作为后续轮次启动。

### 2.7 验证集与评测区间

**唯一协议（沿用现有 `main.py`）**:每个 split 的 test 尾部 35% 作 val，
前 65% 作最终评测。`best.pt` 在**训练阶段**按 val 上的 bf F1 选出。

四个评测格子统一使用**被测 test 的前 65%**（域内 = 本域 test，交叉 = 目标域 test）:

| 格子 | 模型 | 评测数据 |
|---|---|---|
| A→A | ckpt_A | A-test 前 65% |
| A→B | ckpt_A | B-test 前 65% |
| B→B | ckpt_B | B-test 前 65% |
| B→A | ckpt_B | A-test 前 65% |

> 交叉评测是 `trainmode=test`，不再选 checkpoint；目标域 test 的尾部 35%
> 仍然按代码逻辑被切走，但它不参与任何指标，只是保持与域内评测**同样的评测区间**。
>
> 已知代价:域内格子（A→A、B→B）的 `best.pt` 是用各自 test 尾部选的，
> 存在 val 泄漏。为了与现有论文表格完全同一口径，本轮接受这一点;
> 交叉格子（A→B、B→A）的 checkpoint 只用**源域 test 尾部**选择，
> 选择过程不接触目标域任何数据。

### 2.8 输出产物

统一放在 `/data/sdc/yrlbkp/anomaly/`，沿用现有“一数据集一目录”的约定:

```
HAI22/
  HAI22_train.npy          # float32, [N_train, 78]，清洗后原始值
  HAI22_test.npy           # float32, [N_test, 78]，清洗后原始值
  HAI22_test_label.npy     # int8,    [N_test]
  HAI22_train_segment.npy  # int16,   [N_train]
  HAI22_test_segment.npy   # int16,   [N_test]
  HAI22_train_ts.npy       # int64 纳秒, [N_train]
  HAI22_test_ts.npy        # int64 纳秒, [N_test]
  scale.json               # 本域 train 的逐列 min/max/degenerate（留待加载时使用）
  meta.json                # 行数/片段/列清单/校验和/生成时间
  schema.json              # 78 列名 + 排序 + sha256
HAI23/                     # 同上
```

- **只生成 HAI22 / HAI23 两个目录**，不再生成 `HAI22to23` 这类别名目录。
  交叉评测时用 `--test-dataset` 换测试集，模型和 checkpoint 仍来自源域。
- **npy 不预归一化**（见 §2.6）;加载时由 `get_data` 用**源域**的 `scale.json` 变换。
  A→B 时 `HAI23_test.npy` 会被 HAI22 的 min/max 变换，B→A 反之。
- 所有 npy 一律 `float32`；标签 `int8`；时间戳 `int64`。
- 目录名直接作为 `--dataset` / `--test-dataset` 的取值，交给 `utils.get_data` 读取。

---

## 3. 加载器与代码改动

### 3.1 `utils.py`

1. `get_data_dim()` 增加:

```python
HAI_DATASETS = {'HAI22', 'HAI23'}
if dataset in HAI_DATASETS:
    return 78
```

2. `get_data()` 增加 `minmax_from=None, load_train=True`:HAI 数据集**跳过 `preprocess`**，
   改为在加载后用 `scale.json` 做源域 MinMax:

```python
if dataset in HAI_DATASETS:
    scale_path = os.path.join(prefix, minmax_from or dataset, 'scale.json')
    scale = load_scale(scale_path)          # {col: {min,max,scale,degenerate}}
    train_data = apply_minmax(train_data, scale)
    test_data  = apply_minmax(test_data, scale)
elif do_preprocess:
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
```

`apply_minmax` 必须实现 §2.6 的零范围保护:

```python
def apply_minmax(X, scale):
    mn = np.array([s['min'] for s in scale], dtype=np.float32)
    mx = np.array([s['max'] for s in scale], dtype=np.float32)
    dg = mx == mn
    den = np.where(dg, 1.0, mx - mn).astype(np.float32)
    Y = (X - mn) / den
    Y[:, dg] = 0.0
    return Y
```

原因:原始 npy 不能再用 `preprocess` 分别对 train/test 拟合 min/max，
否则会破坏“只用源域 train 拟合”的协议。

3. 保持 `_clean` 的 NaN→0 / inf→100 语义不变；HAI 数据实测无缺失，属于兜底。
4. `max_train_size` 当前是 MSL 专用缩放（`int(x * 58317)`）。对 HAI
   **不要使用该参数**，正式实验一律全量训练；若确需子集冒烟测试，
   用 `train_start/train_end` 的绝对行号，或在 `get_data` 里按
   `len(train_data)` 缩放。

### 3.2 `main.py`

新增参数（默认值保持旧行为）:

| 参数 | 默认 | 作用 |
|---|---|---|
| `--no-preprocess` | `False` | 显式跳过 `get_data` 内的归一化；HAI22/HAI23 自动开启 |
| `--test-dataset` | `None` | 若给出，test/label 从该数据集目录读取，train 仍来自 `--dataset`。`trainmode=test` 时用它做交叉评测，**不重新训练** |
| `--max-test-size` | `None` | 只取 test 前 N 行，供冒烟;正式结果不传 |
| `--loadpath` | `None` | 已有；`trainmode=test` 时指向源域的 `best.pt` |
| `--score-dump` | `None` | 可选；落盘 `test_score`/`test_label` 供诊断。主指标不依赖它 |

不使用 `--val-from`：验证集仍是现有 `main.py` 的 **test 尾部 35%**，
源码中的切分位置与旧实验完全一致，不新增分支。

`trainmode=test` 时 `main.py` 仍会先切走 test 的尾部 35% 再评分，
因此交叉评测的评分子集是**目标域 test 的前 65%**，与域内评测区间一致（§2.7）。

实现位置（`main.py` 的 `get_data` 调用点，约 225 行）:

```python
(train_data, _), (test_data, test_label) = get_data(
    config.dataset, config.max_train_size, config.max_test_size,
    train_start=config.train_start, test_start=config.test_start)
if config.test_dataset:                      # 交叉评测：只换 test/label
    _, (test_data, test_label) = get_data(
        config.test_dataset, None, config.max_test_size,
        train_start=config.train_start, test_start=config.test_start,
        minmax_from=config.dataset)          # ← 关键：用源域 scale.json
```

- `input_dim` 必须由 `--dataset` 决定（两域都是 78，但仍显式校验相等）。
- `train_data` 在 `trainmode=test` 下不参与前向，保留原样只为复用现有加载流程。
- `minmax_from=config.dataset` 是交叉评测的正确性前提:目标域 test 必须用
  **源域**的 min/max 变换，而不是它自己的。
- 可选优化:`get_data(..., load_train=False)` 让 test 模式跳过 train 的读取
  （HAI22 train 窗口化后 >20 GB）。交叉评测已默认使用；
  域内 test-only（无 `--test-dataset`）仍加载 train，以保持 POT 复评可用。

### 3.3 新增 `dataprocess/preprocess_hai.py`

职责:把 §2 的全部规则物化。建议 CLI:

```bash
python dataprocess/preprocess_hai.py \
  --src /data/sdc/yrlbkp/anomaly \
  --out /data/sdc/yrlbkp/anomaly
```

伪代码:

```python
COMMON = [c for c in COLS22 if c in COLS23 and c not in ('timestamp', 'Attack')]  # 78
CONST  = [...20 列...]          # 由 nunique<=1 在 4 个 split 上求并集，脚本自动判定并断言等于本表
USE    = sorted(COMMON)         # 78 列全部保留，常量列不删除

def load_split(files, inline_label):
    parts, seg = [], []
    for i, f in enumerate(sorted(files)):
        df = pd.read_csv(f)
        X  = df[USE].to_numpy('float32')
        y  = df['Attack'] if inline_label else load_label_by_row(f)
        parts.append((df['timestamp'], X, y, i))
    # 处理 label 的按行对齐检查 → 拼接 → mergesort → 去重检查
    return X_all, y_all, ts_all, seg_all

def fit_scale(X_train):          # 只用本域 train 全量
    mn, mx = X_train.min(0), X_train.max(0)
    scale = mx - mn
    return [{'min': float(a), 'max': float(b), 'scale': float(s),
             'degenerate': bool(s == 0)} for a, b, s in zip(mn, mx, scale)]

# 关键：npy 直接写 load_split 返回的原始 X（float32, 未归一化）
#       fit_scale 的结果只写进 scale.json，供 utils.get_data 在加载时使用
```

脚本必须在结束时打印并写入:
- 每个 split 的 `(rows, cols, segments, pos_rate, nan, inf, dup_ts)`；
- `schema.json` 的列名 sha256；
- 与本文档附录 A/B 的列清单做**逐项断言**（防止上游文件变动后静默漂移）；
- 每个域 `scale.json` 中 `degenerate=true` 的列数（HAI22：18 列；HAI23：20 列）。

### 3.4 运行脚本

新增 `script/hai/cross.sh`。**只训练两次**，两次训练各得到一份 `best.pt`；
交叉评测是 `trainmode=test`，用 `--loadpath` 指回源域 checkpoint:

```bash
PY=/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python
COMMON="--backbone mamba --window_length 70 --batch_size 60 --latent_size 256 \
        --epochs 10 --N 256 --seed 42"

# 训练 A（一次）
$PY main.py --dataset HAI22 --trainmode train-test --name hai/HAI22 \
  --ckpt-dir checkpoint/hai/HAI22 \
  --result-dir results/hai --result-name HAI22 $COMMON

# 训练 B（一次）
$PY main.py --dataset HAI23 --trainmode train-test --name hai/HAI23 \
  --ckpt-dir checkpoint/hai/HAI23 \
  --result-dir results/hai --result-name HAI23 $COMMON

# 交叉评测：A→B
$PY main.py --dataset HAI22 --test-dataset HAI23 --trainmode test \
  --loadpath checkpoint/hai/HAI22/best.pt \
  --result-dir results/hai --result-name HAI22to23 $COMMON

# 交叉评测：B→A
$PY main.py --dataset HAI23 --test-dataset HAI22 --trainmode test \
  --loadpath checkpoint/hai/HAI23/best.pt \
  --result-dir results/hai --result-name HAI23to22 $COMMON
```

`trainmode=train-test` 本身已经吃掉域内格子（A→A、B→B），所以四次评测无需额外命令:
训练结束时打印的 bf F1 就是域内结果，交叉两格由上面的两条 test 命令补齐。

---

## 4. 实验矩阵

### 4.1 核心 run

| ID | 来源 | train | test | 含义 |
|---|---|---|---|---|
| E1 | 训练 A 的附带评测 | HAI22 train | HAI22 test | 域内基线 A→A |
| E2 | 训练 B 的附带评测 | HAI23 train | HAI23 test | 域内基线 B→B |
| E3 | 评测 A 的 ckpt | HAI22 train | HAI23 test | **交叉 A→B** |
| E4 | 评测 B 的 ckpt | HAI23 train | HAI22 test | **交叉 B→A** |

- **2 次训练**（A、B），**4 次评测**（E1–E4）。E1/E3 共用 `ckpt_A`，E2/E4 共用 `ckpt_B`。
- `seed=42` 固定，**不做多 seed**，不报 mean ± sd。
- 不设合并域训练（E5）。

### 4.2 超参数

沿用现有主模型配置，不因数据集改动:

```
window_length = 70
batch_size    = 60
latent_size   = 256
lr            = 0.01
epochs        = 10
N             = 256
val           = 0.35（test 尾部）
backbone      = mamba
seed          = 42
```

### 4.3 成本估算

按 MSL 实测 `0.394 s/step`（高负载）外推，仅作数量级参考。训练集为全量（不切 65%）:

| 源域 | 训练行数 | windows | steps/epoch | 10 epoch |
|---|---:|---:|---:|---:|
| HAI 22.04 | 1,004,402 | 1,004,332 | 16,739 | **≈18.3 h** |
| HAI 23.05 | 896,400 | 896,330 | 14,939 | **≈16.3 h** |

- 评测（forward + bf_search）只用 test 前 65%:HAI22 ≈234,710 窗 / 3,912 步、
  HAI23 ≈184,790 窗 / 3,080 步，按 0.394 s/step 各 ≈0.35–0.45 h；
  4 次评测合计 ≈ **1.5 GPU·h**。
- 总计 ≈ **36 GPU·h**。两张卡各跑一个源域，墙钟 ≈ **19–20 h**（含交叉评测）;
  若机器空载（按 0.13 s/step 量级），可降到 ≈6 h。
- 正式结果必须用全量训练。子集只用于冒烟（见 §7）。
  ⚠️ 现有 `get_data` 的 `max_train_size` 是 **MSL 专用**写法
  (`int(max_train_size * 58317)`)，对 HAI 会算错，**不要传该参数**。

---

## 5. 评测协议

### 5.1 指标

**唯一报告指标:`bf_search` 的 F1**（oracle 百分位阈值 + `adjust_predicts`），
与现有论文表格完全同一口径。

| 项 | 取值 |
|---|---|
| 阈值搜索 | `bf_search(score, label, step_num=700)` |
| 后处理 | `adjust_predicts`（point-adjust） |
| 输出 | Precision / Recall / F1，论文表只填 **F1** |
| 评测区间 | 被测 test 的前 65%（§2.7） |

说明:
- `--score-dump` 不是必需项;若顺手打开，只用于审计与画曲线，**不改变主指标**。
- F1 是 point-adjust 的产物（跨域低召回时会被回填抬高），这一点在结果文档里
  必须写明;跨域数值只用于本 PRD 内部的 A→B / B→A 对照，不与无阈值指标混排。

### 5.2 交叉泛化报告方式

单 seed，不报 mean ± sd。四个格子直接列:

| 格子 | F1 | 来源 ckpt |
|---|---|---|
| A→A | | `ckpt_A` |
| B→B | | `ckpt_B` |
| A→B | | `ckpt_A` |
| B→A | | `ckpt_B` |

派生量（只用于描述，不做显著性检验）:

```
Δ_A = F1(A→A) - F1(A→B)
Δ_B = F1(B→B) - F1(B→A)
```

- 两个 Δ 同号且量级相近 ⇒ 域偏移是双向对称的;
- 只有一个 Δ 大 ⇒ 说明目标域难度或源域覆盖不对称，需要在结果里单独讨论;
- 由于没有重复 run，**不声称任何差异超出随机波动**;若需置信区间，需另开多 seed 轮次（本轮不做）。

### 5.3 诊断指标（解释“为什么掉”）

1. **逐列 SMD**（标准化均值差）:`(μ22 - μ23) / sqrt((σ22²+σ23²)/2)`。
   实测 58 个非常量列中 **13 列 |SMD|>1**、22 列 |SMD|>0.5；最大的是
   `P1_LIT01` 3.48、`P2_SCST` -2.47、`P1_PCV02Z` 2.32、`P2_VIBTR04` 2.12。
   （20 个常量列无方差，SMD 无定义，不计入。）
2. **域分类器 AUC**:把 78 列按源域 MinMax 标准化后用逻辑回归区分 22.04 / 23.05 train
   （各抽 10 万行），实测 **AUROC = 1.0000**。这说明两域在边缘分布上
   完全可分;交叉性能下降应优先归因于域偏移，而不是模型容量。
3. **外推比例**:目标域落在源域 `[min,max]` 之外的行列占比（§2.6）。
4. 可选:按片段统计误差，检查是否某个采集时段的偏移特别大。

---

## 6. 风险与边界

| 风险 | 影响 | 处理 |
|---|---|---|
| 两域边缘分布完全可分（域分类 AUC=1.0） | A→B 可能接近随机;结果“难看”是真实结论，不要因此调参 | 主结果照实报;P2 逐域归一化本轮不跑，仅在结果异常时作为后续对照 |
| 源域 MinMax 外推（`P1_LIT01` 78.6%） | 目标域输入大量落在 [0,1] 外，模型可能饱和 | 不裁剪;记录外推比例;P2 对照留待后续轮次 |
| val 来自目标域 test（旧协议） | 域内格子存在 val 泄漏 | 沿用旧协议以保证与既有表格同口径;交叉格子的 ckpt 只用源域 test 尾部选择，不受目标域影响 |
| bf_search / point-adjust | F1 不可跨设置比较 | 主指标仍用 bf F1（用户要求统一）;只在 PRD 内部四格之间比较，并注明 point-adjust |
| 片段边界跨窗 | <0.06% 窗口语义错误 | v1 接受并记录;v2 做 segment-aware 滑窗 |
| 23.05 test2 标签时间戳被截断到分钟 | 按时间 merge 会错位 | 按行号对齐 + `floor('min')` 校验 |
| 20 个常量列保留 | 其中 `P3_LH01/LL01` 只在 23.05 退化为常数 | 源域零范围 → 全零保护，78 维 schema 不变;不做 58/60/62 列变体 |
| 单 seed | 无法区分“域偏移”与“训练随机性” | 本轮按用户要求只训 A/B 各一次;不声称超出随机波动的差异 |
| 训练集无攻击样本 | 无法做“源域有监督异常分类” | 本实验是正常性建模（AE/重构式），与现有主模型一致 |

**本轮已确定的边界**（不再作为开放问题）:

1. 交叉评测使用 `--test-dataset`，不重新训练，也不生成别名目录。
2. 不纳入合并域训练（E5）。
3. 输入固定 78 列，不做 58/60/62 列变体。
4. 只跑 PESR（`--backbone mamba`），不跑 `lstmae` 第二模型。

---

## 7. 验收清单

预处理脚本产出后，逐项断言:

- [ ] 4 个 split 的行数分别等于 `1,004,402 / 361,200 / 896,400 / 284,400`。
- [ ] 22.04 与 23.05 的 `schema.json` 中 **78 列名逐项相同**，sha256 相同。
- [ ] 所有 npy 的 `dtype` 正确（数据 `float32`、标签 `int8`、段号 `int16`）。
- [ ] 每个 split 内 `timestamp` 严格递增、无重复；片段时间差为 1 秒。
- [ ] 标签只含 `{0,1}`;test 正样本数分别为 `12,030` 和 `11,384`。
- [ ] 23.05 test2 的标签:按行号对齐，`floor('min')` 后 230,400/230,400 一致。
- [ ] 任一 split 的 NaN / inf 计数为 0。
- [ ] `scale.json` 只含本域 train（全量）的统计量，不包含任何 test 行。
- [ ] `degenerate=true` 列数:HAI22 为 **18**、HAI23 为 **20**;加载后这些列在
      所有 split、所有域中都恒为 0（npy 里仍是原始常数，置零发生在 `apply_minmax`）。
- [ ] 交叉评测确实使用源域统计量:目标域 test 加载后的数值，应与“用源域
      `scale.json` 变换原始 npy”的离线结果逐位一致，而不是按目标域自身 min/max 重缩放。
- [ ] 用 1000 行子集做 smoke run:能完成 `train` 和 `test`，打印出 bf F1。
- [ ] 评测的 `test_score` 长度 = `(N_test − int(0.35 × N_test)) − W`
      （`main.py` 先切走尾部 35% val，再做窗口化）。

---

## 8. 产物与实现状态

| 类型 | 路径 | 状态 |
|---|---|---|
| 预处理脚本 | `dataprocess/preprocess_hai.py` | ✅ 已实现并跑通 |
| 数据产物 | `/data/sdc/yrlbkp/anomaly/HAI22/`、`HAI23/` | ✅ 421 MB + 364 MB |
| 加载器 | `utils.py`:`HAI_DATASETS`、`get_data_dim=78`、`load_hai_scale`、`apply_minmax`、`get_data(minmax_from, load_train)` | ✅ |
| 交叉评测 | `main.py`:`--test-dataset`、`--max-test-size`、test-only 跳过 train 加载 | ✅ |
| 调度脚本 | `script/hai/cross.sh` | ✅ |
| 汇总脚本 | `bench/summarize_hai_cross.py` | ✅ |
| 冒烟验证 | HAI22 子集训练 + A→B 交叉评测各跑通一次 | ✅ |

冒烟命令（不产生正式结果）:

```bash
CUDA_VISIBLE_DEVICES=7 python -u main.py --dataset HAI22 --trainmode train-test \
  --max_train_size 2000 --max_test_size 12000 --epochs 1 --N 10 \
  --ckpt-dir checkpoint/hai_smoke/HAI22 --result-dir results/hai_smoke \
  --result-name HAI22_smoke

CUDA_VISIBLE_DEVICES=7 python -u main.py --dataset HAI22 --test-dataset HAI23 \
  --trainmode test --max_test_size 12000 \
  --loadpath checkpoint/hai_smoke/HAI22/best.pt \
  --result-dir results/hai_smoke --result-name HAI22to23_smoke
```

> HAI 下 `--max_train_size` / `--max_test_size` 都解释为**绝对行数**；
> 正式结果不要传这两个参数。

---

## 9. 正式结果（2026-10-19）

### 9.1 四格 bf F1

评价区间 = test 前 65%（尾部 35% 作 val）；交叉格用**源域** `scale.json` 归一化目标域 test。

| 格 | run | 源 ckpt | 测试域 | P | R | F1 | th | TP / FP / FN |
|---|---|---|---|---|---|---|---|---|
| A→A | `HAI22` | HAI22 | HAI22 | 0.7786 | 0.8705 | **0.8220** | 99.16 | 6656 / 1893 / 990 |
| B→B | `HAI23` | HAI23 | HAI23 | 0.8003 | 0.9154 | **0.8540** | 99.14 | 6112 / 1525 / 565 |
| A→B | `HAI22to23` | HAI22 | HAI23 | 0.7620 | 0.8926 | **0.8221** | 98.96 | 5960 / 1862 / 717 |
| B→A | `HAI23to22` | HAI23 | HAI22 | 0.7255 | 0.9162 | **0.8098** | 98.83 | 7005 / 2650 / 641 |

- Δ_A = F1(A→A) − F1(A→B) = **−0.0001**
- Δ_B = F1(B→B) − F1(B→A) = **+0.0442**

一致性检查：同一测试域的两次评测正样本数完全相同
（HAI22：6656+990 = 7005+641 = 7646；HAI23：6112+565 = 5960+717 = 6677），
说明标签对齐与评测区间一致。

### 9.2 训练轮数说明（重要）

`main.py` 的 checkpoint 逻辑（`main.py:389-390`）在每次 val 前
`load_state_dict(best.pt)`，而 `flag` 从 epoch 0 后恒为 1，因此
**`best.pt` 恒为 epoch-0 权重，epoch≥1 的训练全部被丢弃**（项目 memory
`checkpoint-best-is-epoch0`；该写法自上游首次提交 `9353909` 就存在）。

本次在 epoch 0 的 val + 落盘完成后即终止训练，跳过 9 个必然被丢弃的 epoch：

| 域 | epoch 0 耗时 | steps |
|---|---|---|
| HAI22 | 12,523 s | 16,739 |
| HAI23 | 8,261 s | 14,939 |

这与跑满 `--epochs 10` 得到的 ckpt 和指标**逐位相同**。因此本表是
**1 个有效 epoch** 的 PESR 结果，与 PRD6 系列的历史口径一致。

### 9.3 解读边界

- `bf_search` 是逐格在测试标签上搜 oracle 阈值，交叉格不是"固定阈值泛化"，
  只反映分数排序能达到的上限。
- 单 seed（42）、只跑 PESR（`--backbone mamba`）、无重复 run。
- 结果文件：`results/hai/{HAI22,HAI23,HAI22to23,HAI23to22}.json`；
  ckpt：`checkpoint/hai/{HAI22,HAI23}/best.pt`；日志：`nohp-out/hai/`。

---

## 附录 A · 78 个模型输入列（字典序，`*` = 常量列）

```
P1_FCV01D, P1_FCV01Z, P1_FCV02D, P1_FCV02Z, P1_FCV03D, P1_FCV03Z,
P1_FT01, P1_FT01Z, P1_FT02, P1_FT02Z, P1_FT03, P1_FT03Z,
P1_LCV01D, P1_LCV01Z, P1_LIT01, P1_PCV01D, P1_PCV01Z, P1_PCV02D,
P1_PCV02Z, P1_PIT01, P1_PIT01_HH*, P1_PIT02,
P1_PP01AD*, P1_PP01AR*, P1_PP01BD*, P1_PP01BR*, P1_PP02D*, P1_PP02R*,
P1_PP04, P1_PP04SP, P1_SOL01D*, P1_SOL03D*, P1_STSP*,
P1_TIT01, P1_TIT02, P1_TIT03,
P2_24Vdc, P2_ATSW_Lamp, P2_AutoGO, P2_AutoSD, P2_Emerg*,
P2_MASW, P2_MASW_Lamp, P2_ManualGO, P2_ManualSD,
P2_OnOff*, P2_RTR*, P2_SCO, P2_SCST, P2_SIT01, P2_TripEx*,
P2_VIBTR01, P2_VIBTR02, P2_VIBTR03, P2_VIBTR04, P2_VT01,
P2_VTR01*, P2_VTR02*, P2_VTR03*, P2_VTR04*,
P3_FIT01, P3_LCP01D, P3_LCV01D, P3_LH01*, P3_LIT01, P3_LL01*, P3_PIT01,
P4_HT_FD, P4_HT_PO, P4_HT_PS, P4_LD, P4_ST_FD, P4_ST_GOV, P4_ST_LD,
P4_ST_PO, P4_ST_PS, P4_ST_PT01, P4_ST_TT01
```

计数核对:P1 = 36、P2 = 24、P3 = 7、P4 = 11，合计 **78**；
其中带 `*` 的常量列 20 个（P1 = 10、P2 = 8、P3 = 2）。

## 附录 B · 20 个常量列与零范围保护

这 20 列**包含在附录 A 的 78 列中**，不删除。判定规则是“在 4 个 split 的任一
`nunique() <= 1`”；处理规则是“源域 train 上 `max == min` ⇒ 该列全零输出”。

| 列 | 22.04 常数 | 23.05 常数 | 源域 HAI22 下退化 | 源域 HAI23 下退化 |
|---|---:|---:|---|---|
| `P1_PIT01_HH` | 0 | 524320 | ✔ | ✔ |
| `P1_SOL01D` | 0 | 524320 | ✔ | ✔ |
| `P1_SOL03D` | 0 | 524320 | ✔ | ✔ |
| `P1_PP01AD` | 1 | 0 | ✔ | ✔ |
| `P1_PP01AR` | 1 | 0 | ✔ | ✔ |
| `P1_PP01BD` | 0 | 1 | ✔ | ✔ |
| `P1_PP01BR` | 0 | 1 | ✔ | ✔ |
| `P1_PP02D` | 1 | 1 | ✔ | ✔ |
| `P1_PP02R` | 1 | 1 | ✔ | ✔ |
| `P1_STSP` | 1 | 1 | ✔ | ✔ |
| `P2_Emerg` | 0 | 0 | ✔ | ✔ |
| `P2_OnOff` | 1 | 1 | ✔ | ✔ |
| `P2_RTR` | 2880 | 2880 | ✔ | ✔ |
| `P2_TripEx` | 1 | 1 | ✔ | ✔ |
| `P2_VTR01..04` | 50 | 50 | ✔ | ✔ |
| `P3_LH01` | train 有 {70,80} | 恒 70 | ✘ | ✔ |
| `P3_LL01` | train 有 {20,25} | 恒 30 | ✘ | ✔ |

合计退化列数:**HAI22 源域 18 列，HAI23 源域 20 列**。
注意 `P1_PIT01_HH / P1_SOL01D / P1_SOL03D` 的常数差为 `524320`，
若不做零范围保护，目标域会被放大 5 个数量级。

## 附录 C · 逐文件统计

| 文件 | 行数 | 标签正例 | NaN | inf | 重复时间戳 |
|---|---:|---:|---:|---:|---:|
| 22.04 train1 | 93,601 | 0 | 0 | 0 | 0 |
| 22.04 train2 | 201,600 | 0 | 0 | 0 | 0 |
| 22.04 train3 | 126,000 | 0 | 0 | 0 | 0 |
| 22.04 train4 | 86,401 | 0 | 0 | 0 | 0 |
| 22.04 train5 | 237,600 | 0 | 0 | 0 | 0 |
| 22.04 train6 | 259,200 | 0 | 0 | 0 | 0 |
| 22.04 test1 | 86,400 | 885 | 0 | 0 | 0 |
| 22.04 test2 | 82,800 | 2,881 | 0 | 0 | 0 |
| 22.04 test3 | 62,400 | 3,841 | 0 | 0 | 0 |
| 22.04 test4 | 129,600 | 4,423 | 0 | 0 | 0 |
| 23.05 hai-train1 | 280,800 | 0 | 0 | 0 | 0 |
| 23.05 hai-train2 | 291,600 | 0 | 0 | 0 | 0 |
| 23.05 hai-train3 | 126,000 | 0 | 0 | 0 | 0 |
| 23.05 hai-train4 | 198,000 | 0 | 0 | 0 | 0 |
| 23.05 hai-test1 | 54,000 | 2,981 | 0 | 0 | 0 |
| 23.05 hai-test2 | 230,400 | 8,403 | 0 | 0 | 0 |

## 附录 D · 复现入口

```bash
PY=/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python
cd /home/sdb/yrlcd/anamolybench/SECOND

# 0) 原始数据（已下载并校验）
ls /data/sdc/yrlbkp/anomaly/hai-22.04 /data/sdc/yrlbkp/anomaly/hai-23.05

# 1) 预处理：生成 HAI22/ 与 HAI23/ 两个目录（78 列、原始值 + scale.json）
$PY dataprocess/preprocess_hai.py

# 2) 训练 A（GPU0）→ ckpt_A + 域内 A→A 指标
CUDA_VISIBLE_DEVICES=0 $PY -u main.py --dataset HAI22 --trainmode train-test \
  --name hai/HAI22 --ckpt-dir checkpoint/hai/HAI22 \
  --result-dir results/hai --result-name HAI22 \
  --backbone mamba --window_length 70 --batch_size 60 --latent_size 256 \
  --epochs 10 --N 256 --seed 42

# 3) 训练 B（GPU1）→ ckpt_B + 域内 B→B 指标
CUDA_VISIBLE_DEVICES=1 $PY -u main.py --dataset HAI23 --trainmode train-test \
  --name hai/HAI23 --ckpt-dir checkpoint/hai/HAI23 \
  --result-dir results/hai --result-name HAI23 \
  --backbone mamba --window_length 70 --batch_size 60 --latent_size 256 \
  --epochs 10 --N 256 --seed 42

# 4) 交叉评测 A→B（不训练，加载 ckpt_A）
CUDA_VISIBLE_DEVICES=0 $PY -u main.py --dataset HAI22 --test-dataset HAI23 \
  --trainmode test --loadpath checkpoint/hai/HAI22/best.pt \
  --result-dir results/hai --result-name HAI22to23 \
  --backbone mamba --window_length 70 --batch_size 60 --latent_size 256 \
  --epochs 10 --N 256 --seed 42

# 5) 交叉评测 B→A（不训练，加载 ckpt_B）
CUDA_VISIBLE_DEVICES=1 $PY -u main.py --dataset HAI23 --test-dataset HAI22 \
  --trainmode test --loadpath checkpoint/hai/HAI23/best.pt \
  --result-dir results/hai --result-name HAI23to22 \
  --backbone mamba --window_length 70 --batch_size 60 --latent_size 256 \
  --epochs 10 --N 256 --seed 42

# 6) 汇总四个格子的 bf F1
$PY bench/summarize_hai_cross.py results/hai/    # 读 4 个 JSON 出表
```
