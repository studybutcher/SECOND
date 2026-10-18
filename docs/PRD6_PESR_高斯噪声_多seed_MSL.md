# PRD6 · 主模型在**基础高斯加噪**下的多 seed 结果(MSL 先行)

> 目标:把论文 `robust` 表从"单次点估计"升级成 **mean ± sd**,并保证同一套
> η 档、同一组 seed、同一套评测协议,使结果可与 `PRD5.2`(LSTM-AE)并排阅读。
> **本轮只跑 MSL**,其余三个数据集原样复用同一脚本。
> 日期:2026-10-17

---

## 0. 决策记录(2026-10-17,用户确认)

| 项 | 决定 |
|---|---|
| 加噪实现 | **基础版 B:整行替换为标准正态 `N(0,1)`**;不用现在 `main.py` 里生效的混合噪声 |
| 模型 | **主模型 PESR**(`--backbone mamba`,即 `model.Model` / attmambaBlock) |
| 数据集 | **只跑 MSL**;SMAP/SWaT/WADI 留给后续,脚本不变 |
| 种子 | **3 个:{42, 7, 17}**(与 PRD5.2 完全一致) |
| 指标 | **只要 F1**;不落 score、不算 AUROC/AUPRC |
| 评测协议 | 沿用现状:`bf_search` oracle 百分位阈值 + `adjust_predicts`(point-adjust) |

> ⚠️ **先澄清一个事实**:`script/robust/{msl,SMAP,SWaT,WADI}.sh` **四个数据集并没有
> 各自的噪声结构**。它们只有 `--dataset` 和 GPU 不同,加噪全部走 `main.py::add_noise`。
> 当前生效的是"混合噪声"(每行随机三选一:uniform U(−2,2) / spike ±5 / random_walk)。
> 本轮要用的基础版 B 是同一函数里**被注释掉的那一行**:
>
> ```python
> # noise = np.random.normal(0, 1, data[indices].shape)
> ```
>
> 两者的差别不是"数据集不同",而是"噪声分布不同"。本 PRD 的名字里因此写
> **高斯噪声**,而不是"四个数据集各自的结构"。

---

## 1. 加噪协议(唯一要改的代码)

### 1.1 定义

对训练集 `train_data`(已 `MinMaxScaler` 到 [0,1]):

```
num_samples = int(N_train * η)
indices     = 无放回随机抽 num_samples 行
train_data[indices] = N(0,1)  # 整行、逐元素独立采样
```

- 加噪点在 `main.py:231`,在 `get_data`(内部已 `preprocess`)**之后**、切窗之前。
- **只加在训练集**;test/val 保持干净。因此本实验测的是
  "**用部分损坏的数据训练,还能不能学到正常模式**",不是"含噪输入下的检测能力"。
- `N(0,1)` 与 [0,1] 的干净数据量纲差异很大 —— 这是"基础版"的原始语义,不要额外缩放。

### 1.2 代码改动(最小、不破坏旧行为)

`main.py`:

```python
# argparse: 新增
self.add_argument('--noise-kind', type=str, default='mixed',
                  choices=['mixed', 'gauss'],
                  help='mixed=现有混合噪声(默认,保持旧结果可复现);'
                       'gauss=基础版整行 N(0,1)')

# add_noise: 增加 kind 形参,其余分支原样保留
def add_noise(data, noise_ratio, kind='mixed'):
    num_samples = int(data.shape[0] * noise_ratio)
    if num_samples <= 0:
        return data
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    if kind == 'gauss':
        data[indices] = np.random.normal(0, 1, data[indices].shape)
        return data
    # ---- 以下为现有 mixed 分支,不动 ----
    ...

# 调用点
train_data = add_noise(train_data, config.noise, config.noise_kind)
```

**默认值 `mixed`** ⇒ 所有旧脚本、旧结果、旧命令的行为完全不变。
新脚本显式传 `--noise-kind gauss`。

### 1.3 seed 与噪声实现的关系(必须在论文里写明)

`set_seed(config.seed)` 在 `add_noise` **之前**调用(`main.py:191` vs `:231`),所以
**每个 seed 同时改变两件事**:

1. 抽到哪些行、每行采什么噪声(噪声实现);
2. 模型初始化、采样顺序(训练随机性)。

因此本轮报出的 **sd 是"噪声实现 + 训练随机性"的合并方差**,不是纯粹的训练方差。
若审稿人要求把两者分离,见 §6 的可选控制组;本轮不做。

---

## 2. 实验矩阵

| 维度 | 取值 | 数量 |
|---|---|---|
| 数据集 | MSL | 1 |
| 加噪 | `gauss`,η ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.50} | 6 |
| seed | 42, 7, 17 | 3 |
| **合计** | | **18 runs** |

- η=0 也走脚本(不调用 `add_noise`),作为表里的 `-` 行,同时是 sd 的 clean 基准。
- 其余超参全部沿用 `script/robust` 现状:W=70、bs=60、latent=256、lr=0.01、
  10 epoch、val=0.35、`bf_search` 阈值。

### 成本

MSL 单 run 实测 **≈ 65 min**(382.9 s/epoch × 10 + 评测,2026-10-17 探路实测;
早期 127.9 s/epoch 的估计对应机器空载,当日 load≈366/144 核)。

| 并行度 | 墙钟 |
|---|---|
| 6 卡,每卡 3 run | **≈ 3.3 h** |
| 8 卡,每卡 2–3 run | ≈ 3.3 h(17 runs 仍需 3 波) |

### 2.1 后续扩展(2026-10-18):SMAP / SWaT / WADI

同一脚本、同一协议(`gauss`、6 档、3 seeds、只报 F1),新增 **54 runs**:

| 数据集 | W | bs | 训练窗数 | steps/epoch | 估计 epoch | 估计 run | 18 runs |
|---|---|---|---|---|---|---|---|
| SWaT | 35 | 60 | 494,965 | 8,249 | ≈54 min | **≈9 h** | ≈162 GPU·h |
| SMAP | 70 | 60 | 135,113 | 2,252 | ≈15 min | **≈2.5 h** | ≈45 GPU·h |
| WADI | 70 | 24 | 19,930 | 831 | ≈6 min | **≈1 h** | ≈18 GPU·h |

估计口径:用 MSL 实测 0.394 s/step(负载 265–380 时)按 steps 外推,
并用旧日志的 `nompamssa` per-step 标定(W 与维度对 per-step 影响很小)。
**6 张空闲卡 ⇒ 总墙钟 ≈36–40 h**(若机器负载回落到 76 s/epoch 水平,可缩短到 ≈8 h)。

```bash
DATASETS="SWaT SMAP WADI" GPUS="1 2 4 5 6 7" \
  bash script/robust/robust_multiseed.sh
```

队列按 SWaT → SMAP → WADI 排序、轮转分配,长任务先起跑。

显存实测约 13.6 GB/run(48 GB 卡),每卡放 1 个 run 最稳。

---

## 3. 执行脚本

新增 `script/robust/robust_multiseed.sh`(**不改原有四个 `.sh`**):

```bash
#!/bin/bash
# PRD6 · 主模型 + 基础高斯噪声,多 seed
# 用法: bash script/robust/robust_multiseed.sh
#       DATA=MSL SEEDS="42 7 17" NOISES="0 0.05 0.10 0.15 0.2 0.5" \
#         GPUS="0 1 2 3 4 5" bash script/robust/robust_multiseed.sh
set -u
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# 重装系统后 env 位于 miniconda3（2026-10-17 实测）
PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}

DATA=${DATA:-MSL}
BB=${BB:-mamba}                 # 主模型 PESR
SEEDS=${SEEDS:-"42 7 17"}
NOISES=${NOISES:-"0 0.05 0.10 0.15 0.2 0.5"}
GPUS=(${GPUS:-0 1 2 3 4 5})

mkdir -p results/robust_ms nohp-out/robust_ms

tasks=()
for nz in $NOISES; do for s in $SEEDS; do tasks+=("$nz $s"); done; done

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      read -r nz s <<< "${tasks[$ti]}"
      t="pesr_${DATA}_gauss_noise${nz}_seed${s}"
      [ -f "results/robust_ms/${t}.json" ] && { echo "跳过 $t"; continue; }
      echo "[GPU ${GPUS[$gi]}] $t"
      CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY -u main.py \
        --dataset "$DATA" --name "robust_ms/${DATA}" --trainmode train-test \
        --backbone "$BB" --noise "$nz" --noise-kind gauss --seed "$s" \
        --result-dir results/robust_ms --result-name "$t" \
        --ckpt-dir "checkpoint/robust_ms/${t}" \
        > "nohp-out/robust_ms/${t}.log" 2>&1
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
```

关键点:

- `--ckpt-dir` **按 run 唯一化** —— 原 `script/robust/*.sh` 只用 `--name` 决定 ckpt 路径,
  同变体不同 η/seed 会互相覆盖 `best.pt`(PRD4 §2 已记录过一次并发读坏文件)。
- `--result-name` 唯一化,便于聚合。
- 断点续跑:`[ -f ...json ] && continue`。

---

## 4. 结果呈现

### 4.1 逐 run

`results/robust_ms/pesr_MSL_gauss_noise{η}_seed{s}.json`,
字段:`metrics.{precision, recall, f1, threshold}`。

### 4.2 按 η 汇总(mean ± sd,n=3)

| η | Precision | Recall | **F1** | n |
|---|---|---|---|---|
| 0.00(clean) | | | | 3 |
| 0.05 | | | | 3 |
| 0.10 | | | | 3 |
| 0.15 | | | | 3 |
| 0.20 | | | | 3 |
| 0.50 | | | | 3 |

聚合脚本(附在 PRD 末尾或在 `bench/` 下新增 `summarize_robust_ms.py`):

```python
import json, glob, statistics, collections
rows = collections.defaultdict(list)
for p in glob.glob('results/robust_ms/pesr_MSL_gauss_noise*_seed*.json'):
    d = json.load(open(p))
    rows[d['config']['noise']].append(d['metrics'])
for nz in sorted(rows):
    ms = rows[nz]
    def msd(k):
        v = [m[k] for m in ms]
        return f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}" \
               if len(v) > 1 else f"{v[0]:.4f}"
    print(nz, len(ms), msd('precision'), msd('recall'), msd('f1'))
```

### 4.3 目标 LaTeX 表(MSL 列先填,其余留 `—`)

```latex
\begin{table*}[!h]
\caption{Model Evaluation with Replace Noise on Four Datasets
         (mean\,$\pm$\,std over 3 seeds; MSL only in this round)}
\label{robust}
\resizebox{\textwidth}{!}{
\begin{tabular}{lllllllllllll}
\toprule
\multicolumn{1}{c}{\multirow{2}{*}{$\varepsilon$\%}} & \multicolumn{3}{c}{MSL}
& \multicolumn{3}{c}{SMAP} & \multicolumn{3}{c}{SWaT} & \multicolumn{3}{c}{WADI} \\
\cmidrule(r){2-4} \cmidrule(r){5-7} \cmidrule(r){8-10} \cmidrule(r){11-13}
\multicolumn{1}{c}{} & Precision & Recall & F1-score
& Precision & Recall & F1-score & Precision & Recall & F1-score
& Precision & Recall & F1-score \\
\midrule
-    & $p\pm s$ & $r\pm s$ & $f\pm s$ & — & — & — & — & — & — & — & — & — \\
5\%  & ... \\
10\% & ... \\
15\% & ... \\
20\% & ... \\
50\% & ... \\
\bottomrule
\end{tabular}}
\end{table*}
```

---

## 5. 与 PRD5.2 的可比性

| 维度 | PRD5.2(LSTM-AE) | 本 PRD(PESR) | 可比? |
|---|---|---|---|
| η 档 | {0, 0.05, 0.10, 0.15, 0.2, 0.5} | **同** | ✔ |
| seed | {42, 7, 17} | **同** | ✔ |
| 评测协议 | bf_search + point-adjust | **同** | ✔ |
| 加噪 | 混合噪声 | **基础高斯** | ✘ **协议不同** |
| 指标 | F1 + AUROC/AUPRC | **仅 F1** | 部分 |
| 数据 | MSL | MSL | ✔ |

**结论:可以在"档位 × seed × 评测"上与 PRD5.2 并排,但不能把两者的绝对数值
解释为"模型差异"** —— 加噪分布和指标口径都不同。正确的读法是:

- 同一张表里比较**各自随 η 的走势**(形状),而不是跨模型的绝对高低;
- 若要和 PRD5.2 做严格模型对比,需要把加噪切回 `mixed` 再跑一遍 PESR,
  或用 `gauss` 重跑 LSTM-AE。**两者至少有一边要补,本轮先不做。**

---

## 6. 判据(预先写死)

| # | 判据 | 通过条件 |
|---|---|---|
| **D1** | 多 seed 可复现 | 18 个 JSON 齐全;同 η 下 3 个 seed 的 F1 不全部相同 |
| **D2** | sd 的量级 | 报告每个 η 的 seed-sd;若 sd 与 η 间差值同量级,则**不能**声称"噪声导致退化" |
| **D3** | 与 PRD5.2 形状一致 | 两边 F1-vs-η 都平(或都单调)才算一致;不一致时优先怀疑协议差异 |
| **D4** | clean 基准可复核 | η=0 的 mean 应落在既有 `attnmamba+blockize` clean 值(F1≈0.9577)的 seed 波动内 |

> ⚠️ **已知风险(来自 `docs/PRD5.2...` §2.3)**:F1 是 point-adjust 的产物,
> 同一批 run 上 F1 可摆 2.6× 而 AUROC 只摆 0.03。用户本轮决定只看 F1,
> 因此 **D2 不通过时不能反推"模型不鲁棒"**。若后续需要可信结论,
> 加 `--score-dump` 一行即可补 AUROC/AUPRC(不改变任何已跑结果)。

### 可选控制组(本轮不做)

固定噪声实现、只变训练 seed:
先以 seed=42 生成一次噪声索引并落盘,其余 seed 复用同一份 `indices` 与噪声。
这样 sd 只反映训练随机性,能把 §1.3 的合并方差拆开。需要给 `add_noise`
再加一个 `--noise-file` 参数。

---

## 7. 边界

- **只跑 MSL**。SMAP/SWaT/WADI 复用同一脚本(`DATA=SMAP ...`),预计各 1–2 h。
- **噪声 ≠ 异常**:测的是"用部分损坏的数据训练",不是测试时的含噪输入。
- **val ⊂ test** 的既存缺陷沿用未改,所有 η 同病,不影响"随 η 的走势"这一结论。
- **`N(0,1)` 是绝对量纲**:数据已被 MinMax 到 [0,1],噪声标准差 1 相对很大,
  η=0.5 时训练集一半是 [-3,3] 的垃圾行。这是"基础版"的原意,不做强度重标定。
- **不改 `script/robust` 原有四个脚本**;新实验单独放 `robust_ms/`。

---

## 8. 产物

| 类型 | 路径 |
|---|---|
| 代码改动 | `main.py`:新增 `--noise-kind {mixed,gauss}`,默认 `mixed` |
| 调度脚本 | `script/robust/robust_multiseed.sh` |
| 聚合脚本 | `script/robust/summarize_robust_ms.py` |
| 逐 run 结果 | `results/robust_ms/pesr_MSL_gauss_noise<η>_seed<s>.json` |
| 日志 | `nohp-out/robust_ms/*.log` |
| 聚合表 | `results/robust_ms/summary_<DS>.{md,csv}` |
| 目标表 | `results/robust_ms/table_<ds>.tex` |
| ckpt | `checkpoint/robust_ms/<run>/best.pt` |

---

## 9. 执行清单

- [x] 改 `main.py`(`--noise-kind` + `add_noise(kind)`),默认 `mixed` ⇒ 旧命令行为不变
- [x] 新增 `script/robust/robust_multiseed.sh`
- [x] **模型身份核对**:探路 run 打印 `params=201.6167M  中间块=147.9926M (73.40%)`,
      与 `ablation/mamba_MSL.out` 一致
- [x] 单点探路:η=0.5 / seed=42 / GPU1 → P=0.9390 R=0.9505 F1=0.9447,
      JSON 含 `noise_kind: gauss`,实测 382.9 s/epoch
- [x] 铺满 18 runs(6 卡,17 个剩余,2026-10-17 12:55 → 17:09,≈4.2 h)
- [x] 聚合出 §4.2 表 + §4.3 LaTeX(`results/robust_ms/summary_MSL.{md,csv}` + `table_msl.tex`)
- [x] 核对 D1–D4 → 结论见
      `docs/PRD6_PESR_高斯噪声_多seed_MSL_结果_20261017.md`
