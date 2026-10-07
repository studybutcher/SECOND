#!/bin/bash
# =============================================================
# 阶段 B1 · MSL ρ=0%（干净）· 三变体
#
#   mamba   PESR 完整模型（基线）
#   nompa   PESR − MPA （切 simpip 管线 + 清死参数）
#   nomssa  PESR − MSSA（删掉 8 层中间块）
#
# 配置依据（均已由现有材料确定，不再扫描）：
#   W=70, bs=60   <- nohp-out/Hyperpara/windowlength.out 在 MSL 上扫过
#                    W ∈ {35,70,105,175} -> F1 {0.9502, 0.9705, 0.9558, 0.9359}
#   latent=256, lr=0.01, N=256, epochs=10, val=0.35, noise=0（沿用基线）
#
# 这一批同时产出两个东西：
#   1) MSL 上三个变体的干净基线点（ρ=0）
#   2) **实测每 epoch 耗时**（写进 JSON 的 timing 字段）——用来给后续
#      (变体 × ρ × seed) 的并行队列排批次，MSL 训练窗口 58,247、bs=60 → 971 步/epoch
#
# 产出：
#   results/contam/<bb>_MSL_rho00_seed42.json
#   checkpoint/contam/<bb>/MSL/best.pt
#   nohp-out/contam/<bb>_MSL_rho00.out
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

GPUS=(0 2 5)
SEED=42
DATASET=MSL
W=70
BS=60
BACKBONES=(mamba nompa nomssa)

mkdir -p nohp-out/contam results/contam

echo "GPU 状态："
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo "=================== $DATASET  W=$W bs=$BS  rho=0% ==================="
pids=()
i=0
for bb in "${BACKBONES[@]}"; do
    gpu=${GPUS[$i]}
    out="nohp-out/contam/${bb}_${DATASET}_rho00.out"
    json="results/contam/${bb}_${DATASET}_rho00_seed${SEED}.json"
    if [ -f "$json" ]; then
        echo "  跳过 $bb （已有结果 JSON）"
        i=$((i+1)); continue
    fi
    echo "  启动 $bb  → GPU $gpu   日志 $out"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main.py \
        --dataset "$DATASET" \
        --name "contam/${bb}" \
        --trainmode train-test \
        --backbone "$bb" \
        --window_length $W \
        --batch_size $BS \
        --noise 0 \
        --seed $SEED \
        --result-dir results/contam \
        --result-name "${bb}_${DATASET}_rho00_seed${SEED}" \
        --ckpt-dir "checkpoint/contam_runs/${bb}_${DATASET}_rho00_seed${SEED}" \
        > "$out" 2>&1 &
    pids+=($!)
    i=$((i+1))
done

for p in "${pids[@]:-}"; do
    [ -n "$p" ] && wait "$p"
done

echo "=================== 结果汇总 ==================="
$PYTHON - <<'PY'
import json, glob, os
print(f'{"变体":<9}{"F1":>8}{"P":>8}{"R":>7}{"总参数/M":>10}{"中间块/M":>10}'
      f'{"s/epoch":>9}{"窗口数":>8}{"步/epoch":>9}')
for p in sorted(glob.glob('results/contam/*MSL_rho00*.json')):
    d = json.load(open(p)); m, pr, t = d['metrics'], d['params'], d.get('timing') or {}
    print(f'{d["backbone"]:<9}{m["f1"]:>8.4f}{m["precision"]:>8.4f}{m["recall"]:>7.4f}'
          f'{pr["n_params_M"]:>10.2f}{pr["n_midstack_M"]:>10.2f}'
          f'{t.get("mean_epoch_sec", float("nan")):>9.1f}{t.get("n_windows", 0):>8}'
          f'{t.get("n_steps_per_epoch", 0):>9}')
PY
echo "=================== DONE ==================="
