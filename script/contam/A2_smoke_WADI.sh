#!/bin/bash
# =============================================================
# PRD4text_r2 阶段 A · A2 冒烟测试（WADI，ρ=0%）
#
# 目的：在**未加任何污染**的原始 WADI 训练集上，确认三个变体都能训出
#       同量级 F1，并拿到每 epoch 实测耗时（供 §12.2 替换估算）。
#
#   mamba   PESR 基线
#   nomssa  PESR − MSSA（删掉 8 层中间块）
#   nompa   PESR − MPA （走历史遗留的 simpip 管线）← 唯一有实质不确定性
#
# 调度：三个变体并行跑在 3 张空卡上（条件一致）。已有结果 JSON 则跳过（可重入）。
#
# 判据（§14 验收清单 2–4）：
#   * nomssa 落盘 JSON 的 params.n_midstack == 0
#   * nompa  落盘 JSON 的 params.n_params_M ≈ 218.66（不是 221.05，后者=死参数没清）
#   * 三者 F1 同量级；若 nompa 显著偏低 → 不拿它当基线，退回两变体（§13-R5）
#
# 产出：
#   checkpoint/contam/<backbone>/WADI/best.pt
#   results/contam/<backbone>_WADI_rho00_seed42.json
#   nohp-out/contam/<backbone>_WADI_rho00.out
# =============================================================
set -u
cd "$(dirname "$0")/../.."          # 回到仓库根目录

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1           # 否则 tqdm 进度会被块缓冲，日志里看不到训练进度
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

GPUS=(0 2 5)
SEED=42
DATASET=WADI
BACKBONES=(mamba nomssa nompa)

mkdir -p nohp-out/contam results/contam

echo "GPU 状态："
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo "=================== $DATASET (rho=0%) ==================="
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
for bb in "${BACKBONES[@]}"; do
    json="results/contam/${bb}_${DATASET}_rho00_seed${SEED}.json"
    if [ -f "$json" ]; then
        $PYTHON - "$json" "$bb" <<'PY'
import json, sys
p, bb = sys.argv[1], sys.argv[2]
r = json.load(open(p))
m, pr = r['metrics'], r['params']
tm = r.get('timing') or {}
print(f"  {bb:<8} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f}  "
      f"params={pr['n_params_M']:.2f}M mid={pr['n_midstack_M']:.2f}M  "
      f"epoch={tm.get('mean_epoch_sec', float('nan')):.1f}s")
PY
    else
        echo "  $bb    缺失结果 JSON —— 看 nohp-out/contam/${bb}_${DATASET}_rho00.out"
    fi
done

echo "=================== DONE ==================="
