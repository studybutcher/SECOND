#!/bin/bash
# PRD6 · 主模型(PESR)+ 基础高斯噪声,多 seed
#
# 用法:
#   bash script/robust/robust_multiseed.sh                     # MSL, 6 卡, 18 runs
#   DATASETS="SWaT SMAP WADI" GPUS="1 2 4 5 6 7" \
#       bash script/robust/robust_multiseed.sh                 # 其余三集, 54 runs
#   DATA=MSL SEEDS="42" NOISES="0.5" GPUS="0" \
#       bash script/robust/robust_multiseed.sh                 # 单点探路
#
# 与 script/robust/*.sh 的区别:
#   1. 显式 --noise-kind gauss(整行 N(0,1)),旧脚本仍是 mixed;
#   2. 每个 run 单独的 --ckpt-dir / --result-name,并行不互相覆盖 best.pt;
#   3. 已存在结果 JSON 时跳过,可断点续跑;
#   4. 不改动原有四个脚本。
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1

PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}
DATASETS=${DATASETS:-${DATA:-MSL}}    # 空格分隔;DATA 为旧的单数据集写法
BB=${BB:-mamba}                       # 主模型 PESR
SEEDS=${SEEDS:-"42 7 17"}
NOISES=${NOISES:-"0 0.05 0.10 0.15 0.2 0.5"}
GPUS=(${GPUS:-0 1 2 3 4 5})

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

RES_DIR=results/robust_ms
LOG_DIR=nohp-out/robust_ms
mkdir -p "$RES_DIR" "$LOG_DIR"

if [ ! -x "$PY" ]; then
  echo "python not found: $PY" >&2
  exit 1
fi

tasks=()
for ds in $DATASETS; do
  for nz in $NOISES; do
    for s in $SEEDS; do
      tasks+=("$ds $nz $s")
    done
  done
done
echo "datasets=[$DATASETS] backbone=$BB gpus=${GPUS[*]} runs=${#tasks[@]}"

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      read -r ds nz s <<< "${tasks[$ti]}"
      t="pesr_${ds}_gauss_noise${nz}_seed${s}"
      if [ -s "$RES_DIR/${t}.json" ]; then
        echo "[skip] $t"
        continue
      fi
      # 同一 run 可能由另一个队列实例启动（多数据集并行 / 卡复用）：
      # 已在跑就跳过，避免两个进程写同一个 best.pt。
      if pgrep -f -- "--result-name ${t}( |$)" >/dev/null 2>&1; then
        echo "[running] $t"
        continue
      fi
      echo "[GPU ${GPUS[$gi]}] $t"
      CUDA_VISIBLE_DEVICES=${GPUS[$gi]} "$PY" -u main.py \
        --dataset "$ds" --name "robust_ms/${ds}" --trainmode train-test \
        --backbone "$BB" --noise "$nz" --noise-kind gauss --seed "$s" \
        --result-dir "$RES_DIR" --result-name "$t" \
        --ckpt-dir "checkpoint/robust_ms/${t}" \
        > "$LOG_DIR/${t}.log" 2>&1
      rc=$?
      [ $rc -ne 0 ] && echo "[FAIL rc=$rc] $t  see $LOG_DIR/${t}.log"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "---- done ----"
for ds in $DATASETS; do
  printf '%s: ' "$ds"
  ls -1 "$RES_DIR"/pesr_${ds}_gauss_noise*_seed*.json 2>/dev/null | wc -l
done
