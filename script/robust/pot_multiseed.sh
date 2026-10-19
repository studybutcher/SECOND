#!/bin/bash
# =============================================================
# PRD6.1 · PESR(MSL) 的 bf + POT 复评（**不重训**）
#
#   模型   ：mamba = PESR（PRD6，gauss 噪声）
#   η      ：0 0.05 0.10 0.15 0.2 0.5      seeds：42 7 17  → 18 runs
#   阈值   ：bf（oracle 百分位搜索）+ pot_dtaad
#            --pot-fixed → q=1e-3, level=0.98, scale=1.0
#            （docs/阈值选择结果.md §1 的既有 POT 设置）
#
#   做法：--trainmode test --loadpath <已有 best.pt> --threshold-mode both
#         → 扫一次测试分数，同时得到 bf 与 POT 的 F1。
#         init_score = 训练集分数；为复现原 run 的训练集，必须传同样的
#         --noise / --noise-kind / --seed（add_noise 在 set_seed 之后、训练之前执行）。
#
#   ⚠️ LSTM-AE 不在本脚本里：它要换成 gauss 重训，见 lstmae_gauss_pot.sh。
#
#   产出：results/threshold_ms/SECOND_MSL_fixed_pesr_MSL_noise<nz>_seed<s>.json
#
#   用法：zsh script/robust/pot_multiseed.sh
#         GPUS="1 4 6 7" bash script/robust/pot_multiseed.sh
# =============================================================
set -u
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_POT_MULTISEED_BASH:-}" ]; then
  export _POT_MULTISEED_BASH=1; exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."

PY=${PY:-/home/sdb/yiruolei/miniconda3/envs/anomaly/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

OUT_DIR=results/threshold_ms
LOG_DIR=nohp-out/pot_eval
mkdir -p "$OUT_DIR" "$LOG_DIR" checkpoint/pot_eval

GPUS=(${GPUS:-1 4 6 7})
NOISES=${NOISES:-"0 0.05 0.10 0.15 0.2 0.5"}
SEEDS=${SEEDS:-"42 7 17"}

tasks=()
for nz in $NOISES; do for s in $SEEDS; do tasks+=("$nz $s"); done; done
echo "PRD6.1 PESR POT re-eval: ${#tasks[@]} runs  GPUs=${GPUS[*]}"

run_one () {   # run_one <gpu> <nz> <seed>
  local gpu=$1 nz=$2 s=$3
  local t="pesr_MSL_noise${nz}_seed${s}"
  local out_json="${OUT_DIR}/SECOND_MSL_fixed_${t}.json"
  local ckpt="checkpoint/robust_ms/pesr_MSL_gauss_noise${nz}_seed${s}/best.pt"
  local tag="results/robust_ms/pesr_MSL_gauss_noise${nz}_seed${s}.json"

  if [ -s "$out_json" ]; then echo "  [skip] $t"; return 0; fi
  if pgrep -f -- "--result-name ${t}( |$)" >/dev/null 2>&1; then
    echo "  [running] $t"; return 0
  fi
  if [ ! -s "$ckpt" ] || [ ! -s "$tag" ]; then
    echo "  [MISSING] $t  ckpt=$ckpt  tag=$tag"; return 1
  fi

  echo "  [GPU$gpu] $t  $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u main.py \
    --dataset MSL --name "pot_eval/MSL" --trainmode test \
    --backbone mamba --noise "$nz" --noise-kind gauss --seed "$s" \
    --loadpath "$ckpt" \
    --threshold-mode both --pot-fixed \
    --result-dir "$OUT_DIR" --result-name "$t" \
    --pot-dir "$OUT_DIR" --pot-result-name "$t" \
    --ckpt-dir "checkpoint/pot_eval/${t}" \
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
paths = sorted(glob.glob('results/threshold_ms/SECOND_MSL_fixed_pesr_MSL_noise*_seed*.json'))
print(f'POT json: {len(paths)} / 18')
for p in paths:
    d = json.load(open(p))
    b = d['bf_search_oracle']; q = d['pot']['results'].get('pot_dtaad', {})
    print(os.path.basename(p), 'bf=%.4f' % b['f1'], 'pot=%.4f' % q.get('f1', float('nan')))
PY
echo "=================== DONE ==================="
