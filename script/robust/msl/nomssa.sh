#!/bin/zsh
# =============================================================
# MSL · 噪声鲁棒性 · 变体 nomssa（PESR − MSSA，删掉 8 层中间块）
#
# 只跑这一个变体，5 个噪声档**顺序**执行。**不等待其他变体**。
# 本变体每 epoch 只要 ~85 s（中间块参数为 0），5 档约 70 分钟。
#
# 其余说明见同目录 nompa.sh。
#
# 用法：  zsh script/robust/msl/nomssa.sh
#         gpu=7 zsh script/robust/msl/nomssa.sh
# =============================================================
cd "$(dirname "$0")/../../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

data=MSL
bb=nomssa
seed=42
gpu=${gpu:-5}

echo "[$(date +%H:%M:%S)] $data / $bb  GPU=$gpu  开始"
for nz in {0.05,0.10,0.15,0.2,0.5}; do
    name="${bb}_${data}_noise${nz}_seed${seed}"
    if [ -f "results/robust/${name}.json" ]; then
        echo "[$(date +%H:%M:%S)] 跳过 $name（已有结果）"
        continue
    fi
    echo "[$(date +%H:%M:%S)] 启动 $name"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main.py \
        --dataset $data \
        --name "Robust/${data}_${bb}" \
        --trainmode train-test \
        --backbone $bb \
        --noise $nz \
        --seed $seed \
        --result-dir results/robust \
        --result-name "$name" \
        --ckpt-dir "checkpoint/robust/${name}" \
        >> "nohp-out/Robust/${data}_${bb}.out" 2>&1
    echo "[$(date +%H:%M:%S)] 完成 $name"
done
echo "[$(date +%H:%M:%S)] $data / $bb 全部完成"
