#!/bin/zsh
# =============================================================
# SWaT · 噪声鲁棒性 · 变体 nompamssa（MPA 与 MSSA 都删掉）· 分片 b
#
# 与 nompamssa_a.sh 配对，把 5 档拆到两张卡：
#   nompamssa_a.sh          noise 0.05 / 0.10 / 0.15   GPU 5
#   nompamssa_b.sh（本文件） noise 0.2  / 0.5           GPU 7
# 两个脚本互不等待。
#
# 其余说明见 nompamssa_a.sh。
#
# 用法：  zsh script/robust/SWaT/nompamssa_b.sh
# =============================================================
cd "$(dirname "$0")/../../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

data=SWaT
bb=nompamssa
part=b
seed=42
gpu=${gpu:-7}

echo "[$(date +%H:%M:%S)] $data / $bb / $part  GPU=$gpu  开始"
for nz in {0.2,0.5}; do
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
        >> "nohp-out/Robust/${data}_${bb}_${part}.out" 2>&1
    echo "[$(date +%H:%M:%S)] 完成 $name"
done
echo "[$(date +%H:%M:%S)] $data / $bb / $part 全部完成"
