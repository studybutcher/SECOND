#!/bin/bash
# PRD5.4 · 重跑 AE 的 4 档 ρ × 6 seed，**产出 ckpt**（方案 B 的局部损害分析要用逐窗重建误差）。
# 与 ae_sweep.sh 的唯一差别：train_ae.py 现在会存 checkpoint/ae_mpa/<tag>.pt。
set -u
cd "$(dirname "$0")/../.."
source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python
EPOCHS=${EPOCHS:-100}
mkdir -p results/ae_mpa nohp-out/ae_mpa checkpoint/ae_mpa

run_one() {
  local gpu=$1 rho=$2 seed=$3
  local ri
  ri=$(printf "%02d" "$($PY -c "print(int(round($rho*100)))")")
  local tag="ae_MSL_rho${ri}_seed${seed}"
  if [ -f "checkpoint/ae_mpa/${tag}.pt" ] && [ -f "results/ae_mpa/${tag}.json" ]; then
    echo "skip $tag"; return 0
  fi
  echo "start $tag on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u bench/train_ae.py --model ae --dataset MSL \
      --rho "$rho" --seed "$seed" --epochs "$EPOCHS" \
      > "nohp-out/ae_mpa/${tag}.log" 2>&1
  echo "done  $tag (exit $?)"
}

GPUS=(5 6 7)
i=0
for rho in 0 0.05 0.10 0.20; do
  for seed in 42 7 17 0 1 2; do
    g=${GPUS[$((i % 3))]}
    ( run_one "$g" "$rho" "$seed" ) &
    i=$((i+1))
    if [ $((i % 3)) -eq 0 ]; then wait; fi
  done
done
wait
echo "ALL DONE  ckpt="$(ls checkpoint/ae_mpa/*.pt 2>/dev/null | wc -l)
