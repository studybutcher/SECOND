#!/bin/bash
# =============================================================
# PRD6.3 · PESR(MSL) 在历史 mixed 噪声协议下的 bf F1（顺带 POT①）
#
#   噪声：main.py 的 add_noise(kind='mixed')，整行替换，
#         U(-2,2) / ±5 / random_walk 各 1/3。**不改加噪代码。**
#   协议：W=70、bs=60、latent=256、lr=0.01、10 epoch、每 256 步 step、val=0.35。
#   阈值：bf（oracle）+ POT①（--pot-fixed: q=1e-3, level=0.98, scale=1.0）。
#   η：0.05 0.10 0.15 0.2 0.5（η=0 复用 gauss 结果，见 PRD6.3 §3.2）
#   seeds：42 7 17  → 15 runs
#
#   用法：
#     GPUS="0 1 2 3 4" bash script/robust/pesr_mixed_bf.sh
#     NOISES="0.05" SEEDS="42" GPUS="3" bash script/robust/pesr_mixed_bf.sh  # smoke
# =============================================================
set -u
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_PESR_MIXED_BASH:-}" ]; then
  export _PESR_MIXED_BASH=1; exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.." || exit 1

PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

RES_DIR=results/robust_ms_mixed
POT_DIR=results/threshold_ms_mixed
LOG_DIR=nohp-out/robust_ms_mixed
CKPT_DIR=checkpoint/robust_ms_mixed
mkdir -p "$RES_DIR" "$POT_DIR" "$LOG_DIR" "$CKPT_DIR"

GPUS=(${GPUS:-0 1 2 3 4})
NOISES=${NOISES:-"0.05 0.10 0.15 0.2 0.5"}
SEEDS=${SEEDS:-"42 7 17"}

tasks=()
for nz in $NOISES; do for s in $SEEDS; do tasks+=("$nz $s"); done; done
echo "PRD6.3 PESR mixed: ${#tasks[@]} runs  GPUs=${GPUS[*]}"

run_one () {   # run_one <gpu> <nz> <seed>
  local gpu=$1 nz=$2 s=$3
  local t="pesr_MSL_mixed_noise${nz}_seed${s}"
  local out_json="${RES_DIR}/${t}.json"
  local pot_json="${POT_DIR}/SECOND_MSL_fixed_${t}.json"

  if [ -s "$out_json" ] && [ -s "$pot_json" ]; then
    echo "  [skip] $t"; return 0
  fi
  if pgrep -f -- "--result-name ${t}( |$)" >/dev/null 2>&1; then
    echo "  [running] $t"; return 0
  fi

  echo "  [GPU$gpu] $t  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u main.py \
    --dataset MSL --name "robust_ms_mixed/MSL" --trainmode train-test \
    --backbone mamba --noise "$nz" --noise-kind mixed --seed "$s" \
    --threshold-mode both --pot-fixed \
    --result-dir "$RES_DIR" --result-name "$t" \
    --pot-dir "$POT_DIR" --pot-result-name "$t" \
    --ckpt-dir "${CKPT_DIR}/${t}" \
    > "${LOG_DIR}/${t}.log" 2>&1
  local rc=$?
  echo "  [done rc=$rc] $t  $(date '+%F %T')"
  return $rc
}

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      set -- ${tasks[$ti]}
      run_one "${GPUS[$gi]}" "$1" "$2"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=================== summary ==================="
"$PY" - <<'PY'
import glob, json, os
bf = sorted(glob.glob('results/robust_ms_mixed/pesr_MSL_mixed_noise*_seed*.json'))
pot = sorted(glob.glob('results/threshold_ms_mixed/SECOND_MSL_fixed_pesr_MSL_mixed_noise*_seed*.json'))
print(f'bf json: {len(bf)} / 15   pot json: {len(pot)} / 15')
for pb in bf:
    b = json.load(open(pb))['metrics']['f1']
    key = os.path.basename(pb)[:-5]
    pp = f'results/threshold_ms_mixed/SECOND_MSL_fixed_{key}.json'
    q = float('nan')
    if os.path.exists(pp):
        q = json.load(open(pp))['pot']['results'].get('pot_dtaad', {}).get('f1', float('nan'))
    print(f'{key}  bf={b:.4f}  pot={q:.4f}')
PY
echo "=================== DONE ==================="
