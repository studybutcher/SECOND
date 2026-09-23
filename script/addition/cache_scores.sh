#!/bin/bash
# =============================================================
# 缓存四个数据集的 init_score / test_score，供离线阈值标定复用。
# 跑一次即可；之后 bench/sweep_threshold.py 是纯 numpy，不占 GPU。
#
# 产出：results/threshold/scores/SECOND_<dataset>.{npz,json}
#       nohp-out/threshold/cache_<dataset>.log
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python
GPU=${CUDA_VISIBLE_DEVICES:-2}

mkdir -p results/threshold/scores nohp-out/threshold

# 可用参数指定数据集（默认四个全跑）
DATASETS=${@:-SMAP MSL SWaT WADI}

for ds in $DATASETS; do
    echo "=================== $ds ==================="
    CUDA_VISIBLE_DEVICES=$GPU $PY bench/cache_scores.py --dataset "$ds" \
        2>&1 | tee "nohp-out/threshold/cache_${ds}.log" | tail -8
done

echo "=================== ALL DONE ==================="
