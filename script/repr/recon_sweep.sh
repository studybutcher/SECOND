#!/bin/bash
# =============================================================
# PRD5.2 r2 的一键队列：真实异常片段注入 + AE/LSTM-AE 的 F1 与 MSE 对比
#
#   2 模型 (ae, lstmae) × 4 档 ρ (0/5/10/20%) × 3 seed = 24 runs
#
# 依赖（脚本会先自检、缺了就自己补）：
#   artifacts/donor/<ds>_split.json             ← 由 bench/donor_split.py 生成
#   artifacts/donor/<ds>_train_rho*_seed<S>.npy ← 由 bench/inject_donor.py 生成
#
# 用法：
#     zsh script/repr/recon_sweep.sh                        # 默认 MSL（快，先探路）
#     DATA=SWaT zsh script/repr/recon_sweep.sh               # SWaT 主实验
#     DATA=SWaT MODELS="lstmae" SEEDS="42" zsh script/repr/recon_sweep.sh   # 单跑一个
# =============================================================
set -u
if [ -n "${ZSH_VERSION:-}" ] && [ -z "${_RECON_SWEEP_BASH:-}" ]; then
  export _RECON_SWEEP_BASH=1; exec bash "$0" "$@"
fi
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY=/home/yiruolei/anaconda3/envs/anomaly/bin/python

DATA=${DATA:-MSL}
MODELS=${MODELS:-"ae lstmae"}
SEEDS=${SEEDS:-"42 7 17"}
RHOS=${RHOS:-"0 0.05 0.10 0.20"}
EPOCHS=${EPOCHS:-100}
GPUS=(${GPUS:-5 6 7})
# 片断大小（训练行数）。用小片断能顺带保住评测侧的异常（donor 需求小），
# 但太小会欠训练 ⇒ E_A/E_N 虚高。**必须先跑标定确认收敛**，见 PRD §1.3。
N_TR=${N_TR:-}

mkdir -p results/recon nohp-out/recon checkpoint/recon artifacts/donor

echo "===== 数据准备 ====="
# ⚠️ **无条件重生成**：切分是确定性的、且很便宜（~20 s）。原来"文件存在就跳过"会
#    在改了 --n-tr 之后**留着旧切分**，静默跑在错误的数据上。
if [ -z "${SKIP_DATA:-}" ]; then
  $PY -u bench/donor_split.py --dataset "$DATA" ${N_TR:+--n-tr $N_TR} || exit 1
  for s in $SEEDS; do
    $PY -u bench/inject_donor.py --dataset "$DATA" --seed "$s" --rhos "$RHOS" || exit 1
  done
else
  echo "  SKIP_DATA=1：沿用现有 artifacts/donor/"
fi

echo "===== 训练队列：$DATA  ${MODELS} × ρ{$RHOS} × seed{$SEEDS}  epochs=$EPOCHS  GPU=${GPUS[*]} ====="
tasks=()
for m in $MODELS; do for r in $RHOS; do for s in $SEEDS; do tasks+=("$m $r $s"); done; done; done
echo "共 ${#tasks[@]} 个 run"

pids=()
for gi in "${!GPUS[@]}"; do
  (
    for ti in "${!tasks[@]}"; do
      (( ti % ${#GPUS[@]} == gi )) || continue
      read -r m r s <<< "${tasks[$ti]}"
      ri=$(printf "%02d" "$($PY -c "print(int(round($r*100)))")")
      tag="${m}_${DATA}_rho${ri}_seed${s}"
      if [ -f "results/recon/${tag}.json" ]; then echo "  skip $tag"; continue; fi
      # 重试：本机 user-1009.slice 有 400 GB 上限，同一 UID 下别的 session 冲高时
      # 内核的 cgroup-OOM 会 **SIGKILL 我们进程**（`Killed`、无 traceback、间歇性）。
      # 失败重试是最实际的对策；连续 3 次失败才放弃。
      for attempt in 1 2 3; do
        [ -f "results/recon/${tag}.json" ] && break
        echo "  [GPU ${GPUS[$gi]}] $tag  (attempt $attempt)"
        CUDA_VISIBLE_DEVICES=${GPUS[$gi]} $PY -u bench/train_recon.py \
            --model "$m" --dataset "$DATA" --rho "$r" --seed "$s" --epochs "$EPOCHS" \
            >> "nohp-out/recon/${tag}.log" 2>&1
        if [ ! -f "results/recon/${tag}.json" ]; then
          echo "    !! $tag 未产出 JSON（疑似被外部 SIGKILL），sleep 20s 后重试"
          sleep 20
        fi
      done
      [ -f "results/recon/${tag}.json" ] || echo "    ## $tag 连续 3 次失败"
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "===== 汇总 ====="
$PY -u bench/summarize_recon.py --dataset "$DATA"
echo "===== DONE ====="
