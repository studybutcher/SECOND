# PRD7 · HAI 22.04 + HAI 23.05 双数据集预处理与交叉泛化实验

> 目标:把 HAI 22.04 与 HAI 23.05 处理成**同一套列 / 同一套标签 / 同一套行序**，
> 使模型只吃两个数据集的公共部分；在此基础上做
> **A→A / B→B / A→B / B→A** 的交叉泛化实验，回答“在一年期数据上训练，
> 能否直接迁移到另一年期”的问题。
> 日期:2026-10-19
> 状态:**设计稿**（未开始实现）

---

## 0. 决策记录

| 项 | 决定 | 理由 |
|---|---|---|
| 数据版本 | `hai-22.04`、`hai-23.05`（**不用** `haiend-23.05`） | 两者都有 86 个过程量、同一套工艺对象；HAIEnd 多 139 个 DCS 内部控制量，与 22.04 无公共子集 |
| 输入空间 | 先取列名交集（78 列），再删零方差列（20 列）→ **58 列** | 交集保证同一模型可跨域；零方差列在某一域内恒为常数，不携带时序信息，且部分列的常数在两年份间完全不同（域指纹） |
| 标签 | 统一为独立的 `label` 向量，`0=正常, 1=攻击`，只用于 test | 22.04 的标签是 CSV 内联 `Attack` 列，23.05 是独立的 `label-test*.csv` |
| 行序 | 按 `timestamp` 升序稳定排序；同一 split 内不得有重复时间戳 | 多文件拼接后必须保证时间单调；当前数据本身已单调，但契约仍要显式检查 |
| 列序 | 传感器列按**字典序**排列（`timestamp` 不在 npy 中，`label` 单独存） | 字典序与文件原始顺序无关，任何脚本/机器上都能复现同一 schema |
| 归一化 | **源域训练集 fit 的 MinMax**，应用到源域 train/test 和目标域 test；**不裁剪** | 交叉泛化要保留域偏移；用目标域统计量归一化会把“域偏移”这个问题本身抹掉 |
| 验证集 | 主协议从**源域训练集尾部**切 35%（P1）；现有“从 test 尾部切”仅用于与旧结果对照（P0） | 从目标域 test 切 val 会让阈值选择看到目标域数据，交叉实验会虚高 |
| 主指标 | **未 point-adjust 的逐点 AUROC / AUPRC**；F1@POT（标签无关阈值）为次指标；F1@bf（oracle）只作参考 | bf_search 用测试标签搜阈值，且 F1 经 point-adjust 放大；跨域比较必须用无阈值指标 |
| 实验 | 4 个核心 run（A→A、B→B、A→B、B→A）+ 可选 pooled run | 最小可回答问题；seed={42,7,17} |
| 模型 | 不改结构，仍用主模型 PESR（`--backbone mamba`） | 本轮只验证数据协议与泛化现象 |

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

### 2.2 零方差裁剪（删 20 列，留 58 列）

判定规则:**在 4 个 split（22-train / 22-test / 23-train / 23-test）中的任意一个里
`nunique() <= 1`，就删除该列**。也就是说，只要它在某一域内恒为常数，
就不进入公共输入空间。

被删除的 20 列及其常数取值:

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
- 前 18 列在**两个域里都**是常数；后 2 列只在一个域里变化。
- `P1_PIT01_HH / P1_SOL01D / P1_SOL03D` 在两个域里的常数分别是 `0` 和 `524320`，
  是最强的“域指纹”。若不删除，源域 MinMax 会对目标域产生退化缩放。
- 删除后剩 **58 个模型输入列**，完整清单见附录 A。

### 2.3 标签统一

输出 schema:

```
timestamp (仅索引) | <58 个传感器列，字典序> | label ∈ {0,1}
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

**主协议（P1，源域归一化）**:

```
fit:        X_source_train 的逐列 min/max
transform:  X_source_train、X_source_test、X_target_test
```

- 不做裁剪。MinMax 在目标域上会产生 `[0,1]` 之外的值，这是被允许的，
  因为它保留了域偏移。
- 保存 `scale.json`:`{col: {"min": ..., "max": ..., "scale": ..., "offset": ...}}`。
- 源域 train 若按 §2.7 的方式切了 val，**scaler 只用剩下的训练部分拟合**
  （val 也不参与 fit），避免 val 信息进入归一化。

实测目标域超出源域 `[min,max]` 的比例（58 列）:均值 **6.4%**，中位数 **0%**，
9 列超过 10%，最高的是 `P1_LIT01` **78.6%**、`P1_PCV02Z` **60.1%**、
`P2_SCST` **53.1%**。这组数字说明“源域 MinMax 直接套目标域”会产生明显的外推，
必须在论文里明确写出，并配合下面的对照协议。

**对照协议（P2，逐域归一化）**:

- 每个域用自己的 train 统计量做 MinMax，再各自应用。
- 该协议把“边际分布差异”消掉，只留“结构/时序模式差异”，用于判断
  跨域性能下降有多少来自量纲、多少来自真正的模式变化。
- P2 只作为消融，不作为主结果。

### 2.7 验证集切分

| 协议 | val 来源 | 用途 |
|---|---|---|
| **P1（主）** | 源域 **train** 的尾部 35% | 阈值选择/早停只看源域，保证交叉实验无目标域泄漏 |
| P0（兼容） | 目标域 test 的尾部 35%（现有 `main.py` 行为） | 只用于和旧结果对齐；交叉 run 不报 P0 |

> 现有 `main.py:290-296` 是 `test_data[:-n]` / `val_data = test_data[-n:]`。
> 交叉泛化若沿用该逻辑，等于用目标域 35% 的数据（含标签）参与阈值/验证，
> A→B 的指标会虚高。PRD7 要求把 val 来源做成显式参数。

### 2.8 输出产物

统一放在 `/data/sdc/yrlbkp/anomaly/`，沿用现有“一数据集一目录”的约定:

```
HAI22/
  HAI22_train.npy          # float32, [N_train, 58]
  HAI22_test.npy           # float32, [N_test, 58]
  HAI22_test_label.npy     # int8,    [N_test]
  HAI22_train_segment.npy  # int16,   [N_train]
  HAI22_test_segment.npy   # int16,   [N_test]
  HAI22_train_ts.npy       # int64 纳秒, [N_train]
  HAI22_test_ts.npy        # int64 纳秒, [N_test]
  scale.json               # 归一化统计量（fit 于 HAI22 train 的 65%）
  meta.json                # 行数/片段/列清单/校验和/生成时间
  schema.json              # 58 列名 + 排序 + sha256
HAI23/                     # 同上
HAI22to23/                 # train=HAI22 训练集(同一份)，test=HAI23 test
HAI23to22/                 # train=HAI23 训练集，test=HAI22 test
```

- `HAI22to23` / `HAI23to22` 的 train 与源域目录中的 train **逐位相同**，
  可以用硬链接（`os.link`）节省约 230 MB/份。
- 所有 npy 一律 `float32`；标签 `int8`；时间戳 `int64`。
- 目录名直接作为 `--dataset` 的取值，交给现有 `utils.get_data` 读取。

---

## 3. 加载器与代码改动

### 3.1 `utils.py`

1. `get_data_dim()` 增加:

```python
HAI_ALIASES = {'HAI22', 'HAI23', 'HAI22to23', 'HAI23to22',
               'HAIpool22', 'HAIpool23'}
if dataset in HAI_ALIASES:
    return 58
```

2. `get_data()` 对上述别名**跳过 `preprocess`**:

```python
if do_preprocess and dataset not in HAI_ALIASES:
    train_data = preprocess(train_data)
    test_data = preprocess(test_data)
```

原因:这些 npy 已经按 §2.6 用源域统计量归一化过；再调用 `preprocess`
会分别用 train/test 自己的 min/max 二次归一化，把 P1 协议破坏掉。

3. 保持 `_clean` 的 NaN→0 / inf→100 语义不变；HAI 数据实测无缺失，属于兜底。
4. `max_train_size` 当前是 MSL 专用缩放（`int(x * 58317)`）。对 HAI 别名
   改为按 `len(train_data)` 缩放，或直接使用 `train_start/train_end` 的绝对行号；
   否则子集实验的截断位置会完全错误。

### 3.2 `main.py`

新增参数（默认值保持旧行为）:

| 参数 | 默认 | 作用 |
|---|---|---|
| `--no-preprocess` | `False` | 显式跳过 `get_data` 内的归一化；HAI 别名会自动开启 |
| `--val-from {test,train}` | `test` | `train` = P1，从 `train_data` 尾部切 35% 作 val；`test` = P0 现状 |
| `--test-dataset` | `None` | 若给出，test/label 从该别名目录读取，train 仍来自 `--dataset`。用于“训练一次、评测两个目标域”，避免重复训练 |
| `--score-dump` | 已有 | PRD7 要求每个 run 必须落盘，供离线算 AUROC/AUPRC |

`--val-from train` 的切分位置在 `add_noise` / contam 逻辑**之后**、
窗口化**之前**，且 scaler 拟合必须在这之前完成（见 §2.6），
否则 val 数据会进入 min/max。

### 3.3 新增 `dataprocess/preprocess_hai.py`

职责:把 §2 的全部规则物化。建议 CLI:

```bash
python dataprocess/preprocess_hai.py \
  --src /data/sdc/yrlbkp/anomaly \
  --out /data/sdc/yrlbkp/anomaly \
  --normalization source_minmax \
  --val-from train --val-ratio 0.35 \
  --aliases HAI22 HAI23 HAI22to23 HAI23to22
```

伪代码:

```python
COMMON = [c for c in COLS22 if c in COLS23 and c not in ('timestamp', 'Attack')]
DROP   = [...20 列...]          # 由 nunique<=1 在 4 个 split 上求并集，脚本自动判定并断言等于本表
USE    = sorted(set(COMMON) - set(DROP))

def load_split(files, inline_label):
    parts, seg = [], []
    for i, f in enumerate(sorted(files)):
        df = pd.read_csv(f)
        X  = df[USE + (['Attack'] if inline_label else [])]
        y  = df['Attack'] if inline_label else load_label_by_row(f)
        parts.append((df['timestamp'], X[USE].to_numpy('float32'), y, i))
    # 处理 label 的按行对齐检查 → 拼接 → mergesort → 去重检查
    return X_all, y_all, ts_all, seg_all

def fit_minmax(X):  # 只用 source train 的前 65%
    return X.min(0), X.max(0)

def transform(X, mn, mx):
    scale = np.where(mx > mn, mx - mn, 1.0)
    return (X - mn) / scale
```

脚本必须在结束时打印并写入:
- 每个 split 的 `(rows, cols, segments, pos_rate, nan, inf, dup_ts)`;
- `schema.json` 的列名 sha256；
- 与本文档附录 A/B 的列清单做**逐项断言**（防止上游文件变动后静默漂移）。

### 3.4 运行脚本

新增 `script/hai/cross.sh`，按“源域 → 目标域”组织:

```bash
# 训练一次,评测源域 test + 目标域 test
python main.py --dataset HAI22 --test-dataset HAI23 \
  --val-from train --score-dump results/hai/HAI22to23_seed42.npz \
  --seed 42 --window_length 70 --batch_size 60 ...
```

若不实现 `--test-dataset`，退化为别名方案:
`--dataset HAI22to23`（train 与 HAI22 相同，test 是 HAI23 test）。

---

## 4. 实验矩阵

### 4.1 核心 run

| ID | train | test | 含义 |
|---|---|---|---|
| E1 | HAI 22.04 train | HAI 22.04 test | 域内基线 A→A |
| E2 | HAI 23.05 train | HAI 23.05 test | 域内基线 B→B |
| E3 | HAI 22.04 train | HAI 23.05 test | **交叉 A→B** |
| E4 | HAI 23.05 train | HAI 22.04 test | **交叉 B→A** |
| E5（可选） | 22.04 + 23.05 合并 train | 分别测两个 test | 判断“多域训练是否比单域更稳” |

seed: `{42, 7, 17}`。核心 4 组 × 3 seed = **12 次评测**。

若实现 `--test-dataset`，训练只需 **6 次**（2 个源域 × 3 seed），
E1/E3 共享同一个 checkpoint，E2/E4 同理；否则按别名方案需要 12 次训练。
**推荐实现 `--test-dataset`**，理由是交叉实验的变量应该只有“测试域”，
而不是“重新训练一次”。

### 4.2 超参数

沿用现有主模型配置，不因数据集改动:

```
window_length = 70
batch_size    = 60
latent_size   = 256
lr            = 0.01
epochs        = 10
N             = 256
val           = 0.35
backbone      = mamba
```

### 4.3 成本估算

按 MSL 实测 `0.394 s/step`（高负载）外推，仅作数量级参考:

| 源域 | P1 训练行数（65%） | windows | steps/epoch | 10 epoch |
|---|---:|---:|---:|---:|
| HAI 22.04 | 652,861 | 652,791 | 10,879 | **≈11.9 h** |
| HAI 23.05 | 582,660 | 582,590 | 9,709 | **≈10.6 h** |

- 6 次训练 ≈ **67.5 GPU·h**；12 次评测（每次 ≈0.3–0.5 h）≈ **5 GPU·h**。
- 总 ≈ **73 GPU·h**。8 卡各跑 1 个 run，墙钟 ≈ 2 波 ≈ **24 h**；
  若机器空载（按 0.13 s/step 量级），可降到 ≈8 h。
- 若算力紧张，先用子集跑通协议，但正式结果必须用全量。
  ⚠️ 现有 `get_data` 的 `max_train_size` 是 **MSL 专用**写法
  (`int(max_train_size * 58317)`)，对 HAI 会算错。要么改用
  `train_start/train_end` 的绝对行号，要么在 `get_data` 里按
  `len(train_data)` 缩放。本节不把它当作可用参数。

---

## 5. 评测协议

### 5.1 指标分层

| 优先级 | 指标 | 阈值来源 | 说明 |
|---|---|---|---|
| **主** | 逐点 **AUROC / AUPRC**（未 point-adjust） | 无阈值 | 跨域比较的唯一公平指标;由 `--score-dump` 的 `test_score`/`test_label` 离线计算 |
| 次 | **F1@POT** | 源域 train 的 init score 拟合 POT（标签无关） | 部署语义最接近;沿用 `pot_eval` |
| 参考 | F1@bf_search | 目标域 test 标签（oracle） | 只用于和既有论文表格对齐，必须标注 oracle |
| 参考 | point-adjust F1 | 同上 | point-adjust 会按异常段回填预测，跨域/低召回时虚高;不作为结论依据 |

### 5.2 交叉泛化报告方式

对每个指标报:

```
in-domain baseline:  E1/E2
cross-domain:        E3/E4
Δ_cross = mean(in-domain) - mean(cross)   # 在 3 个 seed 上分别算再平均
```

必须同时报 **mean ± sd**（3 seed），并明确 sd 只反映训练随机性
（本轮不叠加噪声，seed 不改变数据）。

### 5.3 诊断指标（解释“为什么掉”）

1. **逐列 SMD**（标准化均值差）:`(μ22 - μ23) / sqrt((σ22²+σ23²)/2)`。
   实测 58 列中 **13 列 |SMD|>1**、22 列 |SMD|>0.5；最大的是
   `P1_LIT01` 3.48、`P2_SCST` -2.47、`P1_PCV02Z` 2.32、`P2_VIBTR04` 2.12。
2. **域分类器 AUC**:把 58 列标准化后用逻辑回归区分 22.04 / 23.05 train
   （各抽 10 万行），实测 **AUROC = 1.0000**。这说明两域在边缘分布上
   完全可分;交叉性能下降应优先归因于域偏移，而不是模型容量。
3. **外推比例**:目标域落在源域 `[min,max]` 之外的行列占比（§2.6）。
4. 可选:按片段统计误差，检查是否某个采集时段的偏移特别大。

---

## 6. 风险与开放问题

| 风险 | 影响 | 处理 |
|---|---|---|
| 两域边缘分布完全可分（域分类 AUC=1.0） | A→B 可能接近随机;结果“难看”是真实结论，不要因此调参 | 主结果照实报;P2 逐域归一化作对照，区分“量纲差”与“模式差” |
| 源域 MinMax 外推（`P1_LIT01` 78.6%） | 目标域输入大量落在 [0,1] 外，模型可能饱和 | 不裁剪;另报 P2 对照;必要时对单列做 log/分位数变换（需单独 PRD） |
| val 来自目标域 test（旧协议） | 交叉指标虚高 | P1 强制 val 来自源域 train |
| bf_search / point-adjust | F1 不可跨设置比较 | 主指标改为无阈值 AUROC/AUPRC |
| 片段边界跨窗 | <0.06% 窗口语义错误 | v1 接受并记录;v2 做 segment-aware 滑窗 |
| 23.05 test2 标签时间戳被截断到分钟 | 按时间 merge 会错位 | 按行号对齐 + `floor('min')` 校验 |
| 20 个零方差列被删 | 其中 `P3_LH01/LL01` 在 22.04 内有变化 | 为保证公共输入空间必须删;若只做 22.04 域内实验，可另设 60 列变体，但不进入交叉主表 |
| 训练集无攻击样本 | 无法做“源域有监督异常分类” | 本实验是正常性建模（AE/重构式），与现有主模型一致 |

**开放问题（需要用户确认）**:

1. 是否实现 `--test-dataset`（推荐），还是先用 4 个别名目录跑 12 次训练？
2. E5（合并两域训练）是否纳入本轮？若纳入，建议只报“合并 vs 单域”的对比，
   不参与交叉主表排名。
3. `P3_LH01/LL01` 是否保留（62/60 列变体），还是坚持 58 列公共空间？
   本文档默认 58 列。
4. 主模型之外，是否需要同时跑 `lstmae` 作为交叉泛化的第二模型？
   （会再增加约 1/3 成本。）

---

## 7. 验收清单

预处理脚本产出后，逐项断言:

- [ ] 4 个 split 的行数分别等于 `1,004,402 / 361,200 / 896,400 / 284,400`。
- [ ] 22.04 与 23.05 的 `schema.json` 中 **58 列名逐项相同**，sha256 相同。
- [ ] 所有 npy 的 `dtype` 正确（数据 `float32`、标签 `int8`、段号 `int16`）。
- [ ] 每个 split 内 `timestamp` 严格递增、无重复；片段时间差为 1 秒。
- [ ] 标签只含 `{0,1}`;test 正样本数分别为 `12,030` 和 `11,384`。
- [ ] 23.05 test2 的标签:按行号对齐，`floor('min')` 后 230,400/230,400 一致。
- [ ] 任一 split 的 NaN / inf 计数为 0。
- [ ] `HAI22to23/HAI22to23_train.npy` 与 `HAI22/HAI22_train.npy` 逐位相同（`np.array_equal`）。
- [ ] `scale.json` 只含源域 train（P1 的 65% 部分）的统计量。
- [ ] 用 1000 行子集做 smoke run:能完成 `train` 和 `test`，`score-dump` 非空，
      AUROC 可计算（不要求数值）。
- [ ] 交叉 run 的 `test_score` 长度 = 目标域 test 行数 − W（`main.py` 的窗口数口径;
      若 `--val-from test`，还要再减去 val 的 35%）。

---

## 附录 A · 58 个模型输入列（字典序）

```
P1_FCV01D, P1_FCV01Z, P1_FCV02D, P1_FCV02Z, P1_FCV03D, P1_FCV03Z,
P1_FT01, P1_FT01Z, P1_FT02, P1_FT02Z, P1_FT03, P1_FT03Z,
P1_LCV01D, P1_LCV01Z, P1_LIT01, P1_PCV01D, P1_PCV01Z, P1_PCV02D,
P1_PCV02Z, P1_PIT01, P1_PIT02, P1_PP04, P1_PP04SP, P1_TIT01,
P1_TIT02, P1_TIT03, P2_24Vdc, P2_ATSW_Lamp, P2_AutoGO, P2_AutoSD,
P2_MASW, P2_MASW_Lamp, P2_ManualGO, P2_ManualSD, P2_SCO, P2_SCST,
P2_SIT01, P2_VIBTR01, P2_VIBTR02, P2_VIBTR03, P2_VIBTR04, P2_VT01,
P3_FIT01, P3_LCP01D, P3_LCV01D, P3_LIT01, P3_PIT01, P4_HT_FD,
P4_HT_PO, P4_HT_PS, P4_LD, P4_ST_FD, P4_ST_GOV, P4_ST_LD,
P4_ST_PO, P4_ST_PS, P4_ST_PT01, P4_ST_TT01
```

## 附录 B · 78 列交集（未裁剪，供审计）

即在附录 A 的基础上加回以下 20 列:

```
P1_PIT01_HH, P1_PP01AD, P1_PP01AR, P1_PP01BD, P1_PP01BR,
P1_PP02D, P1_PP02R, P1_SOL01D, P1_SOL03D, P1_STSP,
P2_Emerg, P2_OnOff, P2_RTR, P2_TripEx,
P2_VTR01, P2_VTR02, P2_VTR03, P2_VTR04,
P3_LH01, P3_LL01
```

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
# 0) 原始数据（已下载并校验）
ls /data/sdc/yrlbkp/anomaly/hai-22.04 /data/sdc/yrlbkp/anomaly/hai-23.05

# 1) 预处理
/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python \
  dataprocess/preprocess_hai.py --normalization source_minmax --val-from train

# 2) 域内基线
python main.py --dataset HAI22 --val-from train --seed 42 \
  --score-dump results/hai/HAI22_seed42.npz
python main.py --dataset HAI23 --val-from train --seed 42 \
  --score-dump results/hai/HAI23_seed42.npz

# 3) 交叉泛化（推荐 --test-dataset 路径）
python main.py --dataset HAI22 --test-dataset HAI23 --val-from train --seed 42 \
  --score-dump results/hai/HAI22to23_seed42.npz
python main.py --dataset HAI23 --test-dataset HAI22 --val-from train --seed 42 \
  --score-dump results/hai/HAI23to22_seed42.npz

# 4) 离线指标
python bench/eval_hai_cross.py results/hai/*.npz   # 待实现:AUROC/AUPRC/F1@POT
```
