#!/bin/bash
# =============================================================
# PRD5.1 r3 · AE 主实验队列（§8 Phase 1）
#
#   实验 A（motivation）：同一 AE 在 4 档污染下，固定正常验证集 MSE 是否变化
#     主指标  E_N   = clean-normal val 上的逐元素 MSE
#     ΔE_N(ρ) = (E_N(ρ) − E_N(0)) / E_N(0)
#     辅助    E_A   = anomaly probe MSE（另报 clip 到 [0,1] 的版本）
#
#   矩阵：ρ ∈ {0, 5%, 10%, 20%} × seed ∈ {42, 7, 17} = **12 runs**
#   污染：bench/ae_contam.py 的结构化异常段注入，α=1.5σ，**contam_seed = training seed**
#         （同一 seed 的模型共享同一份污染；不同 seed 是污染的独立实现）
#
# 用法：
#     zsh script/repr/ae_sweep.sh
#     MODEL=ae EPOCHS=50 GPUS="5 6 7" zsh script/repr/ae_sweep.sh
#     RHOS="0 0.10" SEEDS="42" zsh script/repr/ae_sweep.sh     # 子集
# =============================================================
set -u
# ⚠️ zsh 不对**未加引号**的变量做词分割（bash 会），`for r in $RHOS` 在 zsh 下会整串当一项。
# 本仓历史脚本多用 `zsh script/xxx.sh` 调用，故这里强制用 bash 重入，两种调用方式都成立。
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_AE_SWEEP_BASH:-}" ]; then
  export _AE_SWEEP_BASH=1
  exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python

MODEL=${MODEL:-ae}
EPOCHS=${EPOCHS:-50}
DATASET=${DATASET:-MSL}
RHOS=${RHOS:-"0 0.05 0.10 0.20"}
SEEDS=${SEEDS:-"42 7 17"}
GPUS=(${GPUS:-5 6 7})

mkdir -p results/ae_mpa nohp-out/ae_mpa

# ── 污染文件是否齐（ρ>0 的档必须有对应 seed 的文件）─────────────────
for r in $RHOS; do
  [ "$r" = "0" ] && continue
  ri=$(printf "%02d" $($PY -c "print(int(round($r*100)))"))
  for s in $SEEDS; do
    f="artifacts/ae_mpa/${DATASET}_train_rho${ri}_seed${s}.npy"
    if [ ! -f "$f" ]; then
      echo "!! 缺污染文件 $f"
      echo "   先跑: $PY bench/ae_contam.py --dataset $DATASET --window 70 --all --seed $s"
      exit 1
    fi
  done
done

echo "GPU: ${GPUS[*]}   模型=$MODEL  epochs=$EPOCHS  ρ={$RHOS}  seeds={$SEEDS}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

tasks=()
for r in $RHOS; do for s in $SEEDS; do tasks+=("$r $s"); done; done
echo "共 ${#tasks[@]} 个 run"

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      read -r r s <<< "${tasks[$ti]}"
      ri=$(printf "%02d" $($PY -c "print(int(round($r*100)))"))
      out="nohp-out/ae_mpa/${MODEL}_rho${ri}_seed${s}.log"
      echo "  [GPU ${GPUS[$gi]}] $MODEL ρ=$r seed=$s → $out"
      CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY -u bench/train_ae.py \
        --model "$MODEL" --dataset "$DATASET" --rho "$r" --seed "$s" \
        --epochs "$EPOCHS" > "$out" 2>&1
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=================== 汇总 ==================="
$PY - <<'PY'
import json, glob, statistics as st, os
rows, meta = {}, {}
MODEL = os.environ.get('MODEL', 'ae')
for p in sorted(glob.glob(f'results/ae_mpa/{MODEL}_*_rho*_seed*.json')):
    d = json.load(open(p))
    rows.setdefault(d['rho'], {})[d['seed']] = d
    meta['epochs'] = d['epochs']; meta['sched'] = d.get('lr_schedule')

print(f"模型={MODEL}  epochs={meta.get('epochs')}  lr_schedule={meta.get('sched')}")
print(f'{"ρ":>6}{"n":>4}{"E_N 均值":>13}{"E_N std":>11}{"ΔE_N":>10}'
      f'{"med10 均值":>13}{"Δmed10":>10}{"E_A(clip)":>12}')
base = rows.get(0.0, {})


def d_of(key, rho, sd):
    if rho == 0 or not base:
        return ''
    pair = [(sd[s][key] - base[s][key]) / base[s][key] for s in sd if s in base]
    return f'{sum(pair)/len(pair):+.4f}' if pair else ''


for rho, sd in sorted(rows.items()):
    e = [v['E_N'] for v in sd.values()]
    m10 = [v.get('E_N_med10', v['E_N']) for v in sd.values()]
    ea = [v.get('E_A_clip', float('nan')) for v in sd.values()]
    sd_ = st.stdev(e) if len(e) > 1 else float('nan')
    print(f'{rho:>6.2f}{len(e):>4}{sum(e)/len(e):>13.6f}{sd_:>11.6f}{d_of("E_N", rho, sd):>10}'
          f'{sum(m10)/len(m10):>13.6f}{d_of("E_N_med10", rho, sd):>10}'
          f'{sum(ea)/len(ea):>12.6f}')
print()
print('提示：ΔE_N < 0 = 污染后正常重建反而更好 → 按 PRD §9 只能记"正则化/未观察到损害"，')
print('      不得写成"污染损害了学习"。末点抖动时以 med10 为准。')
PY
echo "=================== DONE ==================="
