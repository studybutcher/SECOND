#!/bin/zsh
# =============================================================
# MSL · 噪声鲁棒性 · 变体 nompa（PESR − MPA）· **上半档**
#
# nompa 单卡跑 5 档要 5 小时，拆成两张卡并行（与 nompa_b.sh 配对）：
#   nompa_a.sh（本文件） noise 0.05 / 0.10 / 0.15   GPU 3   约 3 h
#   nompa_b.sh          noise 0.2  / 0.5            GPU 7   约 2 h
# 两个脚本互不等待，各自跑完各自退出。
#
# 与 script/robust/msl.sh 同一套协议（--noise 训练侧加噪），只换了模型；
# 变体经 main.py 的 BACKBONE_REGISTRY 接入，模型代码零改动。
#
# 日志带变体名与分片名，避免覆盖既有的 nohp-out/Robust/msl.out；
# 分片各写各的日志，防止两个进程的 tqdm 输出交叉串行。
#
# 用法：  zsh script/robust/msl/nompa_a.sh
#         gpu=6 zsh script/robust/msl/nompa_a.sh
# =============================================================
cd "$(dirname "$0")/../../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

data=MSL
bb=nompa
part=a
seed=42
gpu=${gpu:-3}

echo "[$(date +%H:%M:%S)] $data / $bb / $part  GPU=$gpu  开始"
for nz in {0.05,0.10,0.15}; do
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
