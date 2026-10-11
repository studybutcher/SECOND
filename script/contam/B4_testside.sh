#!/bin/bash
# =============================================================
# PRD4 · 路 B：**测试侧**扰动（干净训练 → 含异常检测）
#
# 与训练侧污染（B1/B2）的区别：
#   * 训练集**干净**，模型只训一次
#   * 污染加在 **test_data** 上（`--contam-test-file`），test_label 不变
#   * 因此评测可复用同一个 ckpt —— 这次是"模型在受扰输入下的检测能力"，
#     即「抗扰」的本义
#
# 变体：nompamssa（MPA 与 MSSA 都删，四个变体里最脆）
# 档位：ρ ∈ {0, 1, 5, 10, 20}%   （0% = 干净 test 基线，必须报）
#
# 已知口径（写进结果 JSON 的说明里）：
#   val 是从 test 尾部切出来的（main.py 的既存缺陷），所以替换 test_data 会同时
#   影响 val 与 bf_search 的阈值选择 —— 语义是"部署时输入一直有扰"，
#   代价是阈值会自适应，可能部分掩盖退化。这一条必须与结果一起报。
#
# 用法：bash script/contam/B4_testside.sh [GPU...]
# =============================================================
set -u
cd "$(dirname "$0")/../.."

source /home/yiruolei/anaconda3/etc/profile.d/conda.sh
conda activate anomaly
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PYTHON=/home/yiruolei/anaconda3/envs/anomaly/bin/python

GPUS=(${@:-6 7})
DATASET=MSL
BB=nompamssa
W=70; BS=60; SEED=7; ALPHA=1.5
RHOS=(00 01 05 10 20)

TAG="${BB}_${DATASET}_testrho"
CKPT_DIR="checkpoint/contam_runs/${BB}_${DATASET}_clean_seed${SEED}"
# ⚠️ --ckpt-dir 是**整体替换** checkpoint 目录（不是在其下再套一层 <dataset>），
#    所以 best.pt 就在 $CKPT_DIR/best.pt。曾因此误判"训练未产出 ckpt"而没启动评测。
CKPT="${CKPT_DIR}/best.pt"
mkdir -p nohp-out/testside results/testside artifacts/contam

echo "GPU: ${GPUS[*]}"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

# ── 1) 干净训练（只一次）─────────────────────────────────────────
if [ -f "$CKPT" ]; then
    echo "[1/2] 跳过训练（已有 ckpt $CKPT）"
else
    echo "[1/2] 干净训练 $BB on $DATASET  (W=$W bs=$BS seed=$SEED)"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PYTHON main.py \
        --dataset $DATASET --name "contam/${BB}_clean" --trainmode train \
        --backbone $BB --window_length $W --batch_size $BS --seed $SEED \
        --result-dir results/testside --result-name "${BB}_${DATASET}_clean_seed${SEED}" \
        --ckpt-dir "$CKPT_DIR" > nohp-out/testside/${BB}_${DATASET}_clean.out 2>&1
    [ -f "$CKPT" ] || { echo "[FAIL] 训练未产出 ckpt，见日志"; exit 1; }
    echo "      ckpt 就绪: $CKPT"
fi

# ── 2) 各 ρ 档评测（并行，只前向）────────────────────────────────
echo "[2/2] 评测 5 个档位"
pids=(); i=0
for rr in "${RHOS[@]}"; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}
    out="nohp-out/testside/${BB}_${DATASET}_testrho${rr}.out"
    name="${TAG}${rr}_seed${SEED}"
    json="results/testside/${name}.json"
    if [ -f "$json" ]; then echo "  跳过 ρ=${rr}%（已有）"; i=$((i+1)); continue; fi
    args=(--dataset $DATASET --name "contam/${BB}_test" --trainmode test
          --backbone $BB --window_length $W --batch_size $BS --seed $SEED
          --loadpath "$CKPT"
          --result-dir results/testside --result-name "$name"
          --ckpt-dir "$CKPT_DIR")
    if [ "$rr" != "00" ]; then
        npy="artifacts/contam/${DATASET}_test_rho${rr}_seed${SEED}.npy"
        [ -f "$npy" ] || { echo "  [SKIP] 缺 $npy（先跑 gen_contam.py --split test）"; i=$((i+1)); continue; }
        args+=(--contam-test-file "$npy")
    fi
    echo "  评测 ρ=${rr}% → GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON main.py "${args[@]}" > "$out" 2>&1 &
    pids+=($!); i=$((i+1))
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null || true; done

# ── 3) 汇总 ───────────────────────────────────────────────────
echo "=================== nompamssa · MSL · 测试侧扰动 ==================="
$PYTHON - <<'PY'
import json, glob
rows = []
for p in glob.glob('results/testside/nompamssa_MSL_testrho*_seed7.json'):
    d = json.load(open(p)); c = d.get('contam_test') or {}; m = d['metrics']
    rho = c.get('rho', 0.0)
    rows.append((rho, c.get('rho_win', 0.0), m['f1'], m['precision'], m['recall'],
                 m.get('threshold', 0.0)))
rows.sort()
base = rows[0][2] if rows else None
print(f'{"ρ":>7}{"ρ_win":>9}{"F1":>9}{"P":>9}{"R":>9}{"th":>8}{"Δ vs ρ=0":>11}')
for rho, rw, f1, pr, rc, th in rows:
    d = f'{f1-base:+.4f}' if base is not None else ''
    print(f'{rho:>7.0%}{rw:>9.4f}{f1:>9.4f}{pr:>9.4f}{rc:>9.4f}{th:>8.2f}{d:>11}')
PY
echo "=================== DONE ==================="
