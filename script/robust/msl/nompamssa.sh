#!/bin/zsh
# =============================================================
# MSL · 噪声鲁棒性 · 变体 nompamssa（**MPA 与 MSSA 都删掉**）
#
# 剩余路径：simpembdding → project4 → attention_sim → attention_ALL
#           → projection3 → predict        （只剩 CRR 头与两个 attention）
#
# 参数量（MSL）：51.24 M = 201.62（基线）− 147.99（MSSA）− 2.39（MPA）
# 中间块 0.00 M。与 nomssa 一样快（~80 s/epoch），5 档约 70 分钟。
#
# 与 script/robust/msl.sh 同一套协议（--noise 训练侧加噪），只换了模型；
# 变体经 main.py 的 BACKBONE_REGISTRY 接入，模型代码零改动。
#
# 用法：  zsh script/robust/msl/nompamssa.sh
#         gpu=2 zsh script/robust/msl/nompamssa.sh
# =============================================================
cd "$(dirname "$0")/../../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

data=MSL
bb=nompamssa
seed=42
gpu=${gpu:-1}

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
