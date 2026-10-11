# 表示层定量分析 —— 单 seed 汇总（k=10, bank=400, label_any）

来源文件：1 个 test npz。**主数是实体级 macro mean**（跨实体离散度），pooled 与"不含 σ=0 实体"另列。

| 模型 | repr-AUPRC(V_t) | repr-AUROC(V_t) | repr-AUPRC(μ_t) | repr-AUPRC(Z_p) | D_N | D_A | Sep | **score-AUPRC(原生)** | 实体数 | base rate |
|---|---|---|---|---|---|---|---|---|---|---|
| mamba | 0.1496 ± — | 0.5601 | 0.1238 | 0.7128 | 0.357 | 0.368 | 0.192 | **0.1299** | 1 | 0.1241 |

> 注：`Sep` 与 `D_N/D_A` 只在同一样本内可比；主结论看 AUPRC/AUROC（排序类）。`score-AUPRC` 是同一批窗口上**原生重构分数**的无阈值 AUPRC（未 point-adjust）。

## pooled 口径（次要）

| 模型 | V_t AUPRC | V_t AUROC | Z_p AUPRC | μ_t AUPRC | score AUPRC |
|---|---|---|---|---|---|
| mamba | 0.1625 | 0.5892 | 0.7128 | 0.1241 | 0.1299 |

`σ=0 实体`（MSL: C-2, D-14, M-6, S-2, T-5，信号恒值，重构类方法原理上无法检出）单独列，不进上面的 macro mean：

| 模型 | 实体 | 窗数 | 异常率 | V_t AUPRC | score AUPRC |
|---|---|---|---|---|---|

## 逐实体明细（V_t）

| 实体 | 窗数 | 异常率 | bank | dim | NN时间间隔 | mamba |
|---|---|---|---|---|---|---|---|
| all | 449884 | 0.1241 | 400 | 197 | 222357.0 | 0.1496 |

## 有效性 Gate（PRD5 §11）

### mamba

- **Gate 0** 重复前向 `max|ΔV_t|` = `0.0` → 一致 ✅；eval 模式 `True`；前向路径 `pipline`
- **Gate 1** bank 大小 400（min 400）、k=10；最近邻时间间隔中位 222357.0 窗（≤1 说明 bank 近似查询窗的拷贝）
- **Gate 2** prevalence(any)=0.1241 / (last)=0.1214；异常方向 D_A>D_N 的实体占比 1.00

