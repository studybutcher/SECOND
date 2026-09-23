# 阈值选择结果：bf_search (oracle) vs POT

> `bf_search` 在测试集标签上搜最优阈值（oracle，不可部署）；
> POT 系列只由分数分布决定（标签无关）。
> 三个 POT 口径共用同一次 `initialize()`，只是取不同的量当阈值。

## 表 1  各数据集 F1（同一 ckpt）

| 数据集 | 参数集 | bf_search (oracle) | pot_dtaad | pot_evt | pot_dynamic |
|---|---|---|---|---|---|
| SMAP | fixed | **0.9684** | 0.9445 | 0.8884 | 0.9504 |
| MSL | dtaad | **0.9408** | 0.0000 | 0.0000 | 0.0000 |
| MSL | fixed | **0.9408** | 0.9338 | 0.4751 | 0.7125 |
| SWaT | fixed | **0.9836** | 0.9516 | 0.9449 | 0.9693 |
| WADI | fixed | **0.9510** | 0.6272 | 0.0000 | 0.0000 |

## 表 2  阈值取值

| 数据集 | 参数集 | (q, level, scale) | init_threshold | 真EVT | pot_dtaad 阈值 | pot_evt 阈值 | pot_dynamic 阈值 |
|---|---|---|---|---|---|---|---|
| SMAP | fixed | (0.001, 0.980, 1.00) | 27.0296 | 30.4886 | 27.0303 | 30.4886 | 29.9198 |
| MSL | dtaad | (1e-05, 0.999, 1.04) | 99.8059 | 106.0054 | 103.9572 | 110.2456 | 109.4046 |
| MSL | fixed | (0.001, 0.980, 1.00) | 95.3998 | 100.1013 | 95.4000 | 100.1013 | 99.6948 |
| SWaT | fixed | (0.001, 0.980, 1.00) | 13.2162 | 18.1090 | 13.2173 | 18.1090 | 17.5044 |
| WADI | fixed | (0.001, 0.980, 1.00) | 166.1108 | 173.6613 | 166.1489 | 173.6613 | 173.3395 |

## 表 3  Precision / Recall（同一 ckpt）

| 数据集 | 方法 | F1 | P | R |
|---|---|---|---|---|
| SMAP | bf_search (fixed) | 0.9684 | 0.9573 | 0.9797 |
| SMAP | pot_dtaad (fixed) | 0.9445 | 0.8989 | 0.9950 |
| SMAP | pot_evt (fixed) | 0.8884 | 0.9946 | 0.8026 |
| SMAP | pot_dynamic (fixed) | 0.9504 | 0.9919 | 0.9122 |
| MSL | bf_search (dtaad) | 0.9408 | 0.9006 | 0.9848 |
| MSL | pot_dtaad (dtaad) | 0.0000 | 0.0000 | 0.0000 |
| MSL | pot_evt (dtaad) | 0.0000 | 0.0000 | 0.0000 |
| MSL | pot_dynamic (dtaad) | 0.0000 | 0.0000 | 0.0000 |
| MSL | bf_search (fixed) | 0.9408 | 0.9006 | 0.9848 |
| MSL | pot_dtaad (fixed) | 0.9338 | 0.8879 | 0.9848 |
| MSL | pot_evt (fixed) | 0.4751 | 0.9844 | 0.3131 |
| MSL | pot_dynamic (fixed) | 0.7125 | 0.9869 | 0.5574 |
| SWaT | bf_search (fixed) | 0.9836 | 0.9752 | 0.9922 |
| SWaT | pot_dtaad (fixed) | 0.9516 | 0.9076 | 1.0000 |
| SWaT | pot_evt (fixed) | 0.9449 | 0.9950 | 0.8996 |
| SWaT | pot_dynamic (fixed) | 0.9693 | 0.9928 | 0.9468 |
| WADI | bf_search (fixed) | 0.9510 | 0.9066 | 1.0000 |
| WADI | pot_dtaad (fixed) | 0.6272 | 0.4569 | 1.0000 |
| WADI | pot_evt (fixed) | 0.0000 | 0.0000 | 0.0000 |
| WADI | pot_dynamic (fixed) | 0.0000 | 0.0000 | 0.0000 |

## 表 4  测量元数据

| 数据集 | 参数集 | init_score n | fit_ok | 隐层 W | ckpt |
|---|---|---|---|---|---|
| SMAP | fixed | 20000 | True | 70 | /home/sdb/yrlcd/anamolybench/SECOND/checkpoint/casestudy补充/SMAP/best.pt |
| MSL | dtaad | 20000 | True | 70 | /home/sdb/yrlcd/anamolybench/SECOND/checkpoint/casestudy/MSL/best.pt |
| MSL | fixed | 20000 | True | 70 | /home/sdb/yrlcd/anamolybench/SECOND/checkpoint/casestudy/MSL/best.pt |
| SWaT | fixed | 20000 | True | 35 | /home/sdb/yrlcd/anamolybench/SECOND/checkpoint/casestudy补充/SWaT/best.pt |
| WADI | fixed | 19860 | True | 140 | /home/sdb/yrlcd/anamolybench/SECOND/checkpoint/1-casestudy/WADI/best.pt |

## ⚠️ 缺失的组合

- SMAP/dtaad
- SWaT/dtaad
- WADI/dtaad

