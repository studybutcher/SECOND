# 推理成本原始数据汇总

共 9 条记录，来自 `results/bench_raw/*.json`。

> ✅ 跨库 batch 采样点逐点一致。

## 表 1  计算成本（batch=1 单窗口前向，FP32）

| 模型 | 数据集 | C/W | params(M)<sup>†</sup> | 权重(MB) | GFLOPs/窗口 | GFLOPs/时间步 | 中间块占比 |
|---|---|---|---|---|---|---|---|
| SECOND | SMAP | 25/70 | 193.5215 | 738.23 | 5.126716 | 0.073239 | 0.9581 |
| SECOND | MSL | 55/70 | 201.6167 | 769.11 | 11.299808 | 0.161426 | 0.9659 |
| SECOND | SWaT | 51/35 | 199.6879 | 761.75 | 10.378730 | 0.296535 | 0.9739 |
| SECOND | WADI | 127/70 | 221.0451 | 843.22 | 26.596525 | 0.379950 | 0.9701 |
| MUTANT | SMAP | 25/20 | 0.1256 | 0.39 | 1.215e-04 | 6.075e-06 | - |
| MUTANT | MSL | 55/20 | 0.1979 | 0.66 | 2.838e-04 | 1.419e-05 | - |
| MUTANT | SWaT | 51/20 | 0.1882 | 0.63 | 2.611e-04 | 1.306e-05 | - |
| MUTANT | WADI | 127/5 | 0.3325 | 1.16 | 5.281e-04 | 1.056e-04 | - |

> <sup>†</sup> params 按各自部署 batch 折算。MUTANT 的 `att_encoder.h_n/s_n` 是 `(1,batch,hidden)` 的 Parameter（`Model.py:82-83`），参数量随 batch 线性增长，故按训练/部署 batch（120）计；SECOND 的 `self.batch_size` 在 forward 中未被使用（用的是运行时 `len(x)`），参数量与 batch 无关。

## 表 2  推理延时（ms/窗口，端到端推理路径）

### as_shipped

| 模型 | 数据集 | b=1 | b=8 | b=32 | b=64 | b=120 |
|---|---|---|---|---|---|---|
| SECOND | SMAP | 34.505 | 4.154 | 1.074 | 0.853 | 0.828 |
| SECOND | MSL | 35.251 | 4.285 | 1.792 | 1.694 | 1.683 |
| SECOND | SWaT | 34.574 | 4.189 | 1.674 | 1.640 | 1.623 |
| SECOND | WADI | 35.751 | 4.306 | 3.769 | 3.691 | 3.768 |
| MUTANT | SMAP | 5.217 | 3.982 | 3.785 | 3.742 | 3.734 |
| MUTANT | MSL | 6.307 | 5.042 | 4.911 | 4.896 | 4.857 |
| MUTANT | SWaT | 6.153 | 4.899 | 4.748 | 4.745 | 4.704 |
| MUTANT | WADI | 8.976 | 7.747 | 7.648 | 7.619 | 7.621 |

### core_only

| 模型 | 数据集 | b=1 | b=8 | b=32 | b=64 | b=120 |
|---|---|---|---|---|---|---|
| SECOND | SMAP | 34.119 | 4.092 | 1.025 | 0.808 | 0.785 |
| SECOND | MSL | 35.209 | 4.232 | 1.751 | 1.653 | 1.640 |
| SECOND | SWaT | 34.492 | 4.141 | 1.630 | 1.597 | 1.582 |
| SECOND | WADI | 35.719 | 4.255 | 3.735 | 3.650 | 3.732 |
| MUTANT | SMAP | 1.702 | 0.443 | 0.301 | 0.277 | 0.266 |
| MUTANT | MSL | 1.706 | 0.464 | 0.325 | 0.300 | 0.290 |
| MUTANT | SWaT | 1.701 | 0.463 | 0.322 | 0.297 | 0.284 |
| MUTANT | WADI | 1.676 | 0.485 | 0.349 | 0.325 | 0.315 |

## 表 3  峰值显存（MB）

| 模型 | 数据集 | b=1 | b=8 | b=32 | b=64 | b=120 |
|---|---|---|---|---|---|---|
| SECOND | SMAP | 790.9 | 852.3 | 1061.7 | 1340.9 | 1831.0 |
| SECOND | MSL | 830.4 | 959.3 | 1394.9 | 1977.0 | 2996.5 |
| SECOND | SWaT | 822.6 | 946.3 | 1367.5 | 1931.3 | 2915.8 |
| SECOND | WADI | 925.6 | 1199.0 | 2131.9 | 3378.1 | 5556.6 |
| MUTANT | SMAP | 16.7 | 16.8 | 17.2 | 17.8 | 18.7 |
| MUTANT | MSL | 17.1 | 17.2 | 17.7 | 18.3 | 19.9 |
| MUTANT | SWaT | 17.0 | 17.2 | 17.6 | 18.3 | 19.4 |
| MUTANT | WADI | 17.9 | 18.0 | 18.6 | 24.8 | 38.2 |

## 表 4  跨库比值（SECOND / MUTANT）

> 延时比 < 1 表示 SECOND 更快（原文标注 **粗体**）。

| 数据集 | 参数量比 | FLOPs 比 | 权重显存比 | 延时比 b=1 | 延时比 b=120<br>(as_shipped) | 延时比 b=120<br>(core_only) |
|---|---|---|---|---|---|---|
| SMAP | 1541.26x | 42195.20x | 1901.73x | 6.61x | **0.22x** | 2.95x |
| MSL | 1018.98x | 39816.10x | 1158.31x | 5.59x | **0.35x** | 5.66x |
| SWaT | 1060.92x | 39746.98x | 1214.49x | 5.62x | **0.35x** | 5.57x |
| WADI | 664.87x | 50365.91x | 727.35x | 3.98x | **0.49x** | 11.83x |

## 表 5  计数器覆盖声明

| 模型 | 数据集 | 已计入 | 未计入 |
|---|---|---|---|
| SECOND | SMAP | addmm, bmm, convolution, mm | einsum(broadcast), pscan(inplace), exp/silu/softplus |
| SECOND | MSL | addmm, bmm, convolution, mm | einsum(broadcast), pscan(inplace), exp/silu/softplus |
| SECOND | SWaT | addmm, bmm, convolution, mm | einsum(broadcast), pscan(inplace), exp/silu/softplus |
| SECOND | WADI | addmm, bmm, convolution, mm | einsum(broadcast), pscan(inplace), exp/silu/softplus |
| MUTANT | SMAP | addmm, mm | spmm(GCN, fp64), LSTM(encoder_lstm) |
| MUTANT | MSL | addmm, mm | spmm(GCN, fp64), LSTM(encoder_lstm) |
| MUTANT | SWaT | addmm, mm | spmm(GCN, fp64), LSTM(encoder_lstm) |
| MUTANT | WADI | addmm, mm | spmm(GCN, fp64), LSTM(encoder_lstm) |

## 表 6  测量元数据

| 模型 | 数据集 | GPU | torch | torch线程 | OMP | 载入权重 | ckpt sha256 |
|---|---|---|---|---|---|---|---|
| SECOND | SMAP | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | True | 66adccc3af12 |
| SECOND | MSL | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | True | a09401b7827c |
| SECOND | SWaT | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | True | 486c3ee6677d |
| SECOND | WADI | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | False | - |
| MUTANT | SMAP | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | None | a2e8e215d498 |
| MUTANT | MSL | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | None | 8dcdc2c0a89d |
| MUTANT | SWaT | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | None | 64d8859893ae |
| MUTANT | WADI | RTX 5880 Ada Generation | 2.14.0+cu126 | 1 | 1 | None | ace739735af0 |

