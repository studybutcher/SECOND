#!/bin/bash
# =============================================================
# PRD6.1 · LSTM-AE(MSL) 用 **gauss N(0,1)** 重训，并同时产出 bf + POT
#
#   为什么重训：用户 2026-10-19 决定把 LSTM-AE 的加噪分布从历史 mixed
#   （U(-2,2)/±5/random_walk）改成与 PESR 一致的整行 N(0,1)。
#
#   协议：完全沿用 PESR 的训练设置 —— W=70、bs=60、latent=256、lr=0.01、
#         10 epoch、每 256 步一次 optimizer.step()、val=0.35。
#   阈值：训练结束后同一次流程里算 bf（oracle）与
#         pot_dtaad（--pot-fixed：q=1e-3, level=0.98, scale=1.0）。
#
#   η：0 0.05 0.10 0.15 0.2 0.5      seeds：42 7 17  → 18 runs
#
#   产出：
#     results/lstmae_robust_gauss/lstmae_MSL_gauss_noise<nz>_seed<s>.json   （bf）
#     results/threshold_ms/SECOND_MSL_fixed_lstmae_MSL_noise<nz>_seed<s>.json（POT）
#
#   用法：zsh script/robust/lstmae_gauss_pot.sh
#         GPUS="1 4 6 7" bash script/robust/lstmae_gauss_pot.sh
# =============================================================
set -u
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_LSTMAE_GAUSS_POT_BASH:-}" ]; then
  export _LSTMAE_GAUSS_POT_BASH=1; exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."

PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

RES_DIR=results/lstmae_robust_gauss
POT_DIR=results/threshold_ms
LOG_DIR=nohp-out/lstmae_robust_gauss
mkdir -p "$RES_DIR" "$POT_DIR" "$LOG_DIR"

GPUS=(${GPUS:-1 4 6 7})
NOISES=${NOISES:-"0 0.05 0.10 0.15 0.2 0.5"}
SEEDS=${SEEDS:-"42 7 17"}

tasks=()
for nz in $NOISES; do for s in $SEEDS; do tasks+=("$nz $s"); done; done
echo "PRD6.1 LSTM-AE gauss retrain: ${#tasks[@]} runs  GPUs=${GPUS[*]}"

run_one () {   # run_one <gpu> <nz> <seed>
  local gpu=$1 nz=$2 s=$3
  local t="lstmae_MSL_gauss_noise${nz}_seed${s}"
  local out_json="${RES_DIR}/${t}.json"
  local pot_json="${POT_DIR}/SECOND_MSL_fixed_lstmae_MSL_noise${nz}_seed${s}.json"

  if [ -s "$out_json" ] && [ -s "$pot_json" ]; then echo "  [skip] $t"; return 0; fi
  if pgrep -f -- "--result-name ${t}( |$)" >/dev/null 2>&1; then
    echo "  [running] $t"; return 0
  fi

  echo "  [GPU$gpu] $t  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u main.py \
    --dataset MSL --name "lstmae_robust_gauss/MSL" --trainmode train-test \
    --backbone lstmae --noise "$nz" --noise-kind gauss --seed "$s" \
    --threshold-mode both --pot-fixed \
    --result-dir "$RES_DIR" --result-name "$t" \
    --pot-dir "$POT_DIR" --pot-result-name "lstmae_MSL_noise${nz}_seed${s}" \
    --ckpt-dir "checkpoint/lstmae_robust_gauss/${t}" \
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
bf = sorted(glob.glob('results/lstmae_robust_gauss/lstmae_MSL_gauss_noise*_seed*.json'))
pot = sorted(glob.glob('results/threshold_ms/SECOND_MSL_fixed_lstmae_MSL_noise*_seed*.json'))
print(f'bf json: {len(bf)} / 18   pot json: {len(pot)} / 18')
for pb, pp in zip(bf, pot):
    b = json.load(open(pb))['metrics']['f1']
    q = json.load(open(pp))['pot']['results'].get('pot_dtaad', {}).get('f1', float('nan'))
    print(os.path.basename(pb), 'bf=%.4f' % b, 'pot=%.4f' % q)
PY
echo "=================== DONE ==================="
