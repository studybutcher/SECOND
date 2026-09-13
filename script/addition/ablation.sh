#!/bin/bash
# =============================================================
# 中间块消融：3 个 backbone × 4 个数据集
#   mamba       基线（attmambaBlock，MSSA 中间块）
#   transformer 换成 Transformer mixer
#   lstm        换成双向 LSTM mixer
#
# 调度：同一数据集的 3 个 backbone 并行跑在 3 张空卡上（条件一致），
#       数据集之间串行。已有结果 JSON 的运行会被跳过（可重入）。
#
# 产出：
#   checkpoint/ablation/<backbone>/<dataset>/best.pt
#   results/ablation/<backbone>_<dataset>_seed<seed>.json
#   nohp-out/ablation/<backbone>_<dataset>.out
# =============================================================
set -u
cd "$(dirname "$0")/../.."          # 回到仓库根目录

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1        # 否则 tqdm 进度会被块缓冲，日志里看不到训练进度
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

GPUS=(0 2 5)                        # 三张空闲卡
SEED=42
BACKBONES=(mamba transformer lstm)
DATASETS=(SMAP MSL SWaT WADI)

mkdir -p nohp-out/ablation results/ablation

echo "GPU 状态："
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

for data in "${DATASETS[@]}"; do
    echo "=================== $data ==================="
    pids=()
    i=0
    for bb in "${BACKBONES[@]}"; do
        gpu=${GPUS[$i]}
        out="nohp-out/ablation/${bb}_${data}.out"
        if [ -f "results/ablation/${bb}_${data}_seed${SEED}.json" ]; then
            echo "  跳过 $bb （已有结果 JSON）"
            i=$((i+1)); continue
        fi
        echo "  启动 $bb  → GPU $gpu   日志 $out"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON main.py \
            --dataset "$data" \
            --name "ablation/${bb}" \
            --trainmode train-test \
            --backbone "$bb" \
            --seed $SEED \
            >> "$out" 2>&1 &
        pids+=($!)
        i=$((i+1))
    done
    for p in "${pids[@]:-}"; do
        [ -n "$p" ] && wait "$p"
    done
    echo "  $data 完成"
done

echo "=================== ALL DONE ==================="
