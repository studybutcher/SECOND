#!/bin/bash
# =============================================================
# 阶段 B0 · WADI 基线配置网格（完整模型 mamba，只扫 W × bs）
#
# 目的：给「更好的基线」一个可辩护的出处——不是挑最好的一次 run，
#       而是把完整模型在候选配置上**全部跑一遍**，再选。
#
# 已有格子（不重跑）：
#   W=140 bs=24  -> 0.9510  (checkpoint/1-casestudy，已用当前代码逐位复现)
#   W=70  bs=24  -> A2 contam/mamba 在跑
# 本脚本补：W ∈ {35,70,140} × bs=32  以及  W=35 × bs=24
#
# 为什么 bs=32 之前跑不了：ExpConfig 在 parse_args 之后无条件把 WADI 的
# batch_size 置 24，命令行值被静默吃掉（已修，见 docs/定量分析_WADI基线诊断.md §7）。
# 历史最优（no-patch 0.9914 / attnmamba 0.9831）用的正是 W=140, bs=32。
#
# 产出：
#   results/baseline/mamba_WADI_W<W>_bs<bs>_seed42.json
#   checkpoint/baseline/<W>x<bs>/WADI/best.pt
#   nohp-out/baseline/mamba_WADI_W<W>_bs<bs>.out
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

SEED=42
# "W bs gpu"
CELLS=(
  "35  24 1"
  "35  32 2"
  "70  32 3"
  "140 32 7"
)

mkdir -p nohp-out/baseline results/baseline

echo "GPU 状态："
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

pids=()
for cell in "${CELLS[@]}"; do
    set -- $cell
    W=$1; BS=$2; gpu=$3
    out="nohp-out/baseline/mamba_WADI_W${W}_bs${BS}.out"
    json="results/baseline/mamba_WADI_W${W}_bs${BS}_seed${SEED}.json"
    if [ -f "$json" ]; then
        echo "  跳过 W=$W bs=$BS （已有结果）"
        continue
    fi
    echo "  启动 W=$W bs=$BS → GPU $gpu   日志 $out"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main.py \
        --dataset WADI \
        --name "baseline/${W}x${BS}" \
        --trainmode train-test \
        --backbone mamba \
        --window_length $W \
        --batch_size $BS \
        --noise 0 \
        --seed $SEED \
        --result-dir results/baseline \
        --result-name "mamba_WADI_W${W}_bs${BS}_seed${SEED}" \
        > "$out" 2>&1 &
    pids+=($!)
done

for p in "${pids[@]:-}"; do
    [ -n "$p" ] && wait "$p"
done

echo "=================== 汇总 ==================="
$PYTHON - <<'PY'
import json, glob, os
rows = []
for p in glob.glob('results/baseline/*.json') + ['results/contam/mamba_WADI_rho00_seed42.json']:
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    c, m, pr = d['config'], d['metrics'], d['params']
    rows.append((m['f1'], m['precision'], m['recall'], c['window_length'],
                 c['batch_size'], pr['n_params_M'], os.path.basename(p)))
rows.sort(reverse=True)
print(f'{"F1":>8}{"P":>9}{"R":>9}{"W":>5}{"bs":>5}{"params/M":>10}  文件')
for r in rows:
    print(f'{r[0]:>8.4f}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>5}{r[4]:>5}{r[5]:>10.2f}  {r[6]}')
PY
echo "=================== DONE ==================="
