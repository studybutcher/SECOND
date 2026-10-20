#!/bin/bash
# PRD7 · HAI 22.04 / 23.05 交叉泛化
#   训练 A、训练 B 各一次（各自产出 best.pt + 域内指标），
#   再用同一份 ckpt 交叉评测另一个域（trainmode=test，不重新训练）。
#
# 用法:
#   bash script/hai/cross.sh
#   GPUS="0 1" bash script/hai/cross.sh
set -u
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}
GPUS=(${GPUS:-0 1})
OUT=${OUT:-results/hai}
LOG=${LOG:-nohp-out/hai}

mkdir -p "$OUT" "$LOG" checkpoint/hai

COMMON=(--backbone mamba --window_length 70 --batch_size 60 --latent_size 256
        --epochs 10 --N 256 --seed 42)

train_one () {   # $1=gpu  $2=dataset
  local gpu=$1 ds=$2
  local json="$OUT/${ds}.json"
  if [ -f "$json" ]; then echo "跳过训练 $ds（已有 $json）"; return 0; fi
  echo "[GPU $gpu] 训练 $ds"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py \
    --dataset "$ds" --trainmode train-test --name "hai/$ds" \
    --ckpt-dir "checkpoint/hai/$ds" \
    --result-dir "$OUT" --result-name "$ds" \
    "${COMMON[@]}" > "$LOG/${ds}.log" 2>&1
}

eval_one () {    # $1=gpu  $2=源域  $3=目标域
  local gpu=$1 src=$2 tgt=$3
  local name="${src}to${tgt#HAI}"
  local json="$OUT/${name}.json"
  if [ -f "$json" ]; then echo "跳过评测 $name（已有 $json）"; return 0; fi
  if [ ! -f "checkpoint/hai/$src/best.pt" ]; then
    echo "[FAIL] 缺少 checkpoint/hai/$src/best.pt" >&2; return 1
  fi
  echo "[GPU $gpu] 交叉评测 $name"
  CUDA_VISIBLE_DEVICES=$gpu $PY -u main.py \
    --dataset "$src" --test-dataset "$tgt" --trainmode test \
    --loadpath "checkpoint/hai/$src/best.pt" \
    --result-dir "$OUT" --result-name "$name" \
    "${COMMON[@]}" > "$LOG/${name}.log" 2>&1
}

# 两个源域并行训练（只有一张卡时退化为串行）
if [ ${#GPUS[@]} -ge 2 ]; then
  train_one "${GPUS[0]}" HAI22 &
  p1=$!
  train_one "${GPUS[1]}" HAI23 &
  p2=$!
  wait $p1; s1=$?
  wait $p2; s2=$?
else
  train_one "${GPUS[0]}" HAI22; s1=$?
  train_one "${GPUS[0]}" HAI23; s2=$?
fi

if [ $s1 -eq 0 ] && [ $s2 -eq 0 ]; then
  eval_one "${GPUS[0]}" HAI22 HAI23
  eval_one "${GPUS[$((${#GPUS[@]} - 1))]}" HAI23 HAI22
else
  echo "[FAIL] 训练失败（HAI22=$s1 HAI23=$s2），跳过交叉评测" >&2
  exit 1
fi

$PY -I bench/summarize_hai_cross.py "$OUT"
