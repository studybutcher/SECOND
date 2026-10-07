#!/bin/bash
# =============================================================
# PRD4text 主实验队列 · MSL
#
#   变体 × ρ × seed，全部配对：
#     变体  mamba（PESR）/ nompa（PESR − MPA）/ nomssa（PESR − MSSA）
#     ρ     0 / 1 / 5 / 10 / 20 %   （0% 走干净路径，不需要污染文件）
#     seed  7                        （**同时**决定污染数据与模型初始化）
#
#   ⚠️ 配对性：同一 (ρ, seed) 下三个变体载入**同一个**污染 .npy，
#      因此变体间差异不含数据方差。
#
# ⚠️ 已知局限：**单种子**（使用方 2026-10-07 明确要求「先不改变」）。
#    MSL 的单次运行 std ≈ 0.0075（由 nohp-out/重复测试.out 的 10 次同配置重复测得），
#    而要检验的效应量级约 0.01~0.03。因此：
#      - ρ 主效应（ρ=20% 应掉 >0.05）：够看
#      - 变体主效应 / **交互效应**：只有点估计，**无误差棒、无法判定显著性**
#    论文里写「消融后劣化更严重」时必须带上「单次运行、未做重复」的说明。
#    若需补：只需在 ρ∈{0,20%} 端点把 seed 加到 3 个（+12 个任务）——
#    交互效应由端点斜率差决定，中间档不必补。
#
# 要检验的假设（用户定义，三条）：
#   1) 变体的主效应：消融导致劣化
#   2) ρ 的主效应：异常增加导致劣化
#   3) **交互**：消融后，劣化随 ρ 增加而更严重   <- 最有说服力的那条
#
# 调度：worker-pool。最多 ${#GPUS[@]} 个并发，任务用 mkdir 原子加锁，
#       所以可以安全地启动**多个实例**（例如先用 3 张卡验证，再用满 8 张）。
#       已有结果 JSON 的任务自动跳过（可重入）。
#
# 用法：
#   bash script/contam/B2_grid.sh              # 用默认卡池
#   GPUS="1 3 7" bash script/contam/B2_grid.sh # 只指定这几张
#
# 产出：
#   results/contam/<bb>_MSL_rho<RR>_seed<S>.json
#   nohp-out/contam/grid/<bb>_MSL_rho<RR>_seed<S>.out
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
DATASET=MSL
W=70
BS=60
ALPHA=1.5
# 变体与 ρ 档可用环境变量覆盖：
#   variants="nompamssa" rhos="01 05 10 20" bash script/contam/B2_grid.sh
if [ -n "${variants:-}" ]; then read -ra VARIANTS <<< "$variants"; else
    VARIANTS=(mamba nompa nomssa nompamssa); fi
if [ -n "${rhos:-}" ]; then read -ra RHOS <<< "$rhos"; else
    RHOS=(00 01 05 10 20); fi
SEEDS=(7)          # 单种子（使用方决定，见文件头「已知局限」）

mkdir -p nohp-out/contam/grid results/contam .locks/contam

# 收集待跑任务（TASK_FILTER 可选：只跑 task id 里含该子串的，便于分批验证）
TASK_FILTER="${TASK_FILTER:-}"
tasks=()
skipped=0
for bb in "${VARIANTS[@]}"; do
  for rr in "${RHOS[@]}"; do
    for s in "${SEEDS[@]}"; do
      id="${bb}:${rr}:${s}"
      if [ -n "$TASK_FILTER" ] && [[ "$id" != *"$TASK_FILTER"* ]]; then continue; fi
      json="results/contam/${bb}_${DATASET}_rho${rr}_seed${s}.json"
      if [ -f "$json" ]; then skipped=$((skipped+1)); else tasks+=("$id"); fi
    done
  done
done
echo "待跑任务 ${#tasks[@]} 个（已跳过 ${skipped:-0} 个已有结果）${TASK_FILTER:+  过滤=${TASK_FILTER}}"
echo "并发卡池: ${GPUS[*]}"
[ ${#tasks[@]} -eq 0 ] && { echo "无待跑任务"; exit 0; }

pids=(); busy=()
launch() {                       # $1=task  $2=gpu
  # 注意：`local a=... b=${a}` 不行 —— 同一个 local 语句的所有右侧会**先于**赋值展开，
  # 在 `set -u` 下会报 "rest: unbound variable"。必须拆成两条。
  local bb=${1%%:*} rest=${1#*:}
  local rr=${rest%%:*} s=${rest##*:}
  local out="nohp-out/contam/grid/${bb}_${DATASET}_rho${rr}_seed${s}.out"
  local name="${bb}_${DATASET}_rho${rr}_seed${s}"
  # ⚠️ ckpt 必须按 run 唯一化：否则同变体的并行/后续 run 会并发写同一个 best.pt
  #    （读到半截 → PytorchStreamReader 报损坏），并互相覆盖。
  local args=(--dataset "$DATASET" --name "contam/${bb}" --trainmode train-test
              --backbone "$bb" --window_length $W --batch_size $BS --seed "$s"
              --result-dir results/contam --result-name "$name"
              --ckpt-dir "checkpoint/contam_runs/${name}")
  if [ "$rr" != "00" ]; then
    local npy="artifacts/contam/${DATASET}_rho${rr}_seed${s}.npy"
    [ -f "$npy" ] || { echo "  [SKIP] $name 缺污染文件 $npy" >&2; return 1; }
    args+=(--contam-file "$npy")
  else
    args+=(--noise 0)
  fi
  # ⚠️ 不能写成 `pid=$(launch ...)`：命令替换会开子 shell，`&` 起的作业就不是
  #    本 shell 的子进程，`wait` 会立刻返回（作业被 reparent 掉）。
  #    所以这里把 pid 写进全局变量 LAST_PID，由调用方读取。
  echo "  启动 $name → GPU $2" >&2
  CUDA_VISIBLE_DEVICES=$2 $PYTHON main.py "${args[@]}" > "$out" 2>&1 &
  LAST_PID=$!
}

for t in "${tasks[@]}"; do
  # 占坑：mkdir 原子，避免多实例重复跑同一任务
  lock=".locks/contam/$(echo "$t" | tr ':' '_')"
  mkdir "$lock" 2>/dev/null || { echo "  跳过 $t（已被其他实例占坑）"; continue; }
  # 等一个空位
  while [ ${#pids[@]} -ge ${#GPUS[@]} ]; do
    wait -n 2>/dev/null || true
    alive=()
    for i in "${!pids[@]}"; do
      kill -0 "${pids[$i]}" 2>/dev/null && alive+=("$i")
    done
    newp=(); newb=()
    for i in "${alive[@]}"; do newp+=("${pids[$i]}"); newb+=("${busy[$i]}"); done
    pids=("${newp[@]:-}"); busy=("${newb[@]:-}")
  done
  used=" ${busy[*]:-} "
  gpu=""
  for g in "${GPUS[@]}"; do [[ "$used" == *" $g "* ]] || { gpu=$g; break; }; done
  [ -z "$gpu" ] && gpu=${GPUS[0]}
  LAST_PID=""
  launch "$t" "$gpu"
  if [ -n "$LAST_PID" ]; then
    pids+=("$LAST_PID"); busy+=("$gpu")
  else
    rmdir "$lock" 2>/dev/null || true      # 没起起来就把坑还回去
  fi
done

echo "全部任务已投递，等待收尾..."
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null || true; done

echo "=================== 汇总 ==================="
$PYTHON - <<'PY'
import json, glob, os
rows = []
for p in glob.glob('results/contam/*MSL_rho*.json'):
    d = json.load(open(p)); c, m = d['config'], d['metrics']
    cont = d.get('contam') or {}
    rows.append((d['backbone'], c.get('noise'), cont.get('rho'),
                 d['seed'], m['f1'], m['precision'], m['recall'], os.path.basename(p)))
rows.sort(key=lambda r: (str(r[0]), r[1] if r[2] is None else r[2], r[3]))
print(f'{"变体":<9}{"ρ":>7}{"seed":>6}{"F1":>9}{"P":>9}{"R":>9}  文件')
for bb, nz, rho, s, f1, pr, rc, f in rows:
    tag = f'{rho:.0%}' if rho is not None else f'0%(noise={nz})'
    print(f'{bb:<9}{tag:>7}{s:>6}{f1:>9.4f}{pr:>9.4f}{rc:>9.4f}  {f}')
print(f'\n共 {len(rows)} 条')
PY
echo "=================== DONE ==================="
