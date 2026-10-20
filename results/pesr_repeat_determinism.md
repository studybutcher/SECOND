# PESR η=0 重复 4 次：F1 确定性校验

固定三个 seed（42 / 7 / 17），对同一 ckpt 用完全相同的命令重复 4 次完整前向
（独立进程、独立输出目录 `results/threshold_pesr_repeat/rep{1..4}/`），
与原始 `results/threshold_pesr_potcal/` 对比。生成命令见
`script/robust/pot_cal_no_retrain.py`（`--noises 0`），对比脚本
`script/robust/compare_pesr_repeat.py`。

| seed | 5 次 JSON md5 | base_th | ③ test (val-argmax) | ③ held-out | ① test (s=1.0) | ① held-out | bf test |
|---|---|---|---|---|---|---|---|
| 42 | 全部相同 (`48dc288d4093`) | 48.2218 | 0.9290 | 0.8980 | 0.9290 | 0.8980 | 0.9443 |
| 7  | 全部相同 (`6b2061a20ab0`) | 80.8864 | 0.9369 | 0.9232 | 0.9369 | 0.9232 | 0.9509 |
| 17 | 全部相同 (`b63e712e99a9`) | 24.4988 | 0.9065 (s\*=1.045) | 0.8982 | 0.9253 | 0.8964 | 0.9350 |

5 次（原始 + 4 次重复）输出**逐位一致**，包括 `base_th`、`bf_test` 与全部
scale 行的 F1。

## 结论

1. PESR 未调用 `model.eval()`，dropout 仍开启；但 `set_seed(seed)`（含
   `torch.cuda.manual_seed_all`）加固定前向顺序使 dropout mask 可复现，
   **同一打分路径下结果确定**。
2. PRD6.1 §5.4 的 |Δ|≤0.0268 是跨打分路径的差异（训练进程内打分 vs 新进程
   重新前向），不是 run-to-run 随机性。
3. 同一 run 内 ③ 与 ① 共享分数，差异只来自阈值选择，不含 dropout 噪声。
