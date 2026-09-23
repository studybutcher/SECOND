#!/bin/bash
# =============================================================
# POT 阈值评估：对**原始代码的已有 ckpt** 跑四种阈值口径（不重训）
#
#   参数集：--pot-fixed  → q=1e-3, level=0.98, scale=1
#            （不加 --pot-fixed 则用 DTAAD 的 lm_d 表，见 eval_method.py:DTAAD_LM）
#   口径  ：--threshold-mode all → bf(oracle) + pot_dtaad + pot_evt + pot_dynamic
#
# 产出：
#   results/threshold/SECOND_<dataset>_{fixed,dtaad}.json
#   nohp-out/threshold/<dataset>.log
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python
GPU=${CUDA_VISIBLE_DEVICES:-5}

mkdir -p results/threshold nohp-out/threshold

run() {
    local ds=$1 ckpt=$2
    shift 2
    echo "=================== $ds ==================="
    CUDA_VISIBLE_DEVICES=$GPU $PY main.py \
        --dataset "$ds" --name "threshold_$ds" --trainmode test \
        --loadpath "$ckpt" --threshold-mode all --pot-fixed "$@" \
        2>&1 | tee "nohp-out/threshold/${ds}.log" | tail -14
}

# WADI 的 ckpt 是 W=140 训的（仓库里唯一可加载的一个），必须显式指定 window_length
run SMAP checkpoint/casestudy补充/SMAP/best.pt
run MSL  checkpoint/casestudy/MSL/best.pt
run SWaT checkpoint/casestudy补充/SWaT/best.pt
run WADI checkpoint/1-casestudy/WADI/best.pt --window_length 140

echo "=================== ALL DONE ==================="
