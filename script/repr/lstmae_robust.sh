#!/bin/bash
# =============================================================
# PRD5.2 · LSTM-AE 在**现有 robust 加噪协议**下的训练效果
#
#   **只换模型，其余全沿用现有设置**（用户 2026-10-10 决定）：
#     * 加噪：main.py --noise（add_noise，整行替换 U(-2,2)/±5/random_walk）
#     * 评测：main.py 现状 —— bf_search 的 oracle 百分位阈值 + point-adjust F1
#     * 调度：10 epoch、lr=0.01、每 256 步 step、seed 42、W/bs 走 main.py 默认
#     * 档位：{0.05,0.10,0.15,0.2,0.5}（script/robust/*.sh 的原档）+ 0（clean 基准）
#
#   ⚠️ F1 已实测无分辨力（docs/PRD5_T0结果_20261010.md §0），曲线平了也不能下结论。
#      故本脚本**同时把 score 存盘**（--dump-score），事后可算未调整 AUPRC。
#
# 用法：
#     zsh script/repr/lstmae_robust.sh
#     SEEDS="42" GPUS="5 6 7" zsh script/repr/lstmae_robust.sh
# =============================================================
set -u
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_LSTMAE_ROBUST_BASH:-}" ]; then
  export _LSTMAE_ROBUST_BASH=1; exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python

DATA=${DATA:-MSL}
BB=${BB:-lstmae}
SEEDS=${SEEDS:-"42"}
NOISES=${NOISES:-"0 0.05 0.10 0.15 0.2 0.5"}
GPUS=(${GPUS:-5 6 7})

mkdir -p results/lstmae_robust nohp-out/lstmae_robust

tasks=()
for nz in $NOISES; do for s in $SEEDS; do tasks+=("$nz $s"); done; done
echo "$DATA / $BB : ${#tasks[@]} 个 run   η={$NOISES}  seeds={$SEEDS}  GPU=${GPUS[*]}"

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      read -r nz s <<< "${tasks[$ti]}"
      t="lstmae_${DATA}_noise${nz}_seed${s}"
      if [ -f "results/lstmae_robust/${t}.json" ]; then echo "  跳过 $t"; continue; fi
      echo "  [GPU ${GPUS[$gi]}] $t"
      CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY -u main.py \
        --dataset "$DATA" --name "lstmae_robust/${DATA}" --trainmode train-test \
        --backbone "$BB" --noise "$nz" --seed "$s" \
        --result-dir results/lstmae_robust --result-name "$t" \
        --ckpt-dir "checkpoint/lstmae_robust/${t}" \
        --score-dump "results/lstmae_robust/${t}_scores.npz" \
        > "nohp-out/lstmae_robust/${t}.log" 2>&1
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=================== 汇总（F1 = 现有评测协议）==================="
$PY - <<'PY'
import json, glob, os
rows = {}
for p in sorted(glob.glob('results/lstmae_robust/lstmae_*_noise*_seed*.json')):
    d = json.load(open(p))
    rows[(d.get('noise', d['config'].get('noise', 0)), d['seed'])] = d
seeds = sorted({s for _, s in rows})
print(f'{"η":>6} ' + ' '.join(f'{"seed"+str(s):>10}' for s in seeds) + f'{"均值":>10}{"std":>9}')
for nz in sorted({n for n, _ in rows}):
    v = [rows[(nz, s)]['metrics']['f1'] for s in seeds if (nz, s) in rows]
    print(f'{nz:>6.2f} ' + ' '.join(f'{x:>10.4f}' for x in v)
          + f'{sum(v)/len(v):>10.4f}' + (f'{(max(v)-min(v)):>9.4f}' if len(v) > 1 else f'{"—":>9}'))
print('\n注：F1 是 point-adjust 口径，已实测无分辨力；平了不等于没影响。')
PY
echo "=================== DONE ==================="
