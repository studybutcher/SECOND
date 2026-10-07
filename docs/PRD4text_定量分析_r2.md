# PRD4 · 异常污染动机定量实验（r2）

> r1 见 `docs/PRD4text_定量分析.md`。r2 的改动集中在四处：**(a) 把术语钉到代码行**（r1 的 `MPA`/`CRR` 在代码里不存在）、**(b) 补上 r1 完全缺失的执行流水线**（先造数据、再训练）、**(c) 修掉三个会让结论无法归因的方法论漏洞**（数据量混淆、重建指标的循环论证、图表与自身原则不一致）、**(d) 先跑 WADI pilot 再决定规模**（不预先承诺 430 GPU·h）。
>
> **执行起点：阶段 A（WADI pilot，约 2 GPU·h）**，不要直接开全量。见 §4 与 §14 的验收清单。
>
> 依赖：**PRD1**（MSSA 替换型消融，**已实现**）、**PRD3.1**（non-oracle 阈值协议）。两个**删除型**变体（`PESR − MPA` / `PESR − MSSA`）**已接线并通过 A0 验收**（§3.2），剩下唯一有实质不确定性的环节是 A2：`PESR − MPA` 走的 `simpip` 是历史遗留路径，能否训出同量级 F1 未知。产出图规格见 `docs/PRD4_FigX_FigY_绘图规格.md`。

---

## 0. r2 相对 r1 的变更清单

| # | r1 的问题 | r2 的处理 |
|---|---|---|
| 1 | `MPA`/`CRR` 在代码中**无对应标识符**；`MSSA` 未与 `--backbone` 对应 | §2 给术语→代码映射表（带 `file:line`） |
| 2 | 完全**没有执行流水线**：没说先造数据再训练，没有目录/命名/顺序/产物 | §4 分阶段流水线（A: WADI pilot → B: WADI 全量 → C: SWaT）+ §12 预算 + §14 交付物与验收清单 |
| 3 | **数据量混淆**：ρ=20% 时若总长固定，则正常样本少了 20%，性能下降无法归因于"污染" | §5.2 增加 **length-matched clean 对照**（`clean-L80`） |
| 4 | `E_A(ρ)` 在 **test 集**上测，但污染只加在 **train** 集——"模型学会重建异常"在 test 上**原理上测不出来** | §8.3 增加 `E_A^inj`（在被注入的训练段上测），`E_A` 保留为 test 侧泛化指标 |
| 5 | 图 (a) 画**绝对** AUPRC，但三个变体的 ρ=0 基线不同 → 混了"起点低"与"退化快" | §9.1 改画**归一化退化** `ΔAUPRC(ρ)`；绝对曲线仅作附录 |
| 6 | `--noise` 注入发生在 MinMax 缩放**之后**，注入值 ±2/±5 落在 [0,1] 空间 → 物理无意义 | §6 明确注入空间的选择及对应代码改动 |
| 7 | 未提**窗口级污染率**：ρ 定义在时间戳级，但模型训练在窗口上，短段会放大曝光 | §5.6 强制报告 `ρ_win` |
| 8 | 异常强度 `α` 只说"保持不变"，未给值、未说怎么定 | §5.4 加 **α 标定流程**（以 clean 模型的检出率为准） |
| 9 | seed 同时扰动数据与初始化，两个方差源**混在一起** | §9.3 数据 seed × 初始化 seed 的方差分解 |
| 10 | 只有 mean±std，没有显著性检验 | §9.2 配对检验（同 seed 同数据 → 可配对） |
| 11 | 假设 H1–H4 没有**证伪条件** | §11 每个假设加"什么结果会推翻它" |
| 12 | `loss_c`（视图一致性损失）r1 从未提及，却与 MPA 结构性耦合 | §2.1 明确其归属，§3 变体设计给出处置 |
| 13 | 大量重复（ρ 网格出现 4 次、`E_N/E_A/Gap` 出现 4 次）；894 行里约 2/3 是重复与排版 | r2 全文 600 余行，同一事实只出现一次 |
| 14 | 未说明与 `script/robust/*.sh`（现役 ρ=0.5 随机替换实验）的关系 | §15 明确处置 |
| **15** | **误以为 `PESR − MPA` 已有设置**：实际只有一条被注释掉的 `simpip`，**没有开关能打开**（§2.0） | §3.2 给出 2 处补丁（已落地）+ A2 冒烟测试；失败则退回两变体 |
| **16** | r1 默认直接开跑全套（估算 430 GPU·h，其中 SWaT 占 91%） | §4 改为 **WADI pilot 优先**（2 GPU·h），§12 阶段化，SWaT 规模在 A5 用**实测**预算决定 |
| **17** | 段几何是隐藏自由参数，r1 完全未给值 | §5.3 参数化 + **A5 冻结检查表**（ρ_win 落在 [1.0,1.5] 等 4 项） |
| **18** | 未发现 `Model-nopatch*.py` 是历史备份而非受控消融 | §2.0 显式排除，防止误当 `w/o MPA` 的结果 |
| **19** | r1 默认 `w/o MSSA` = **替换**（沿用 PRD1），与 PRD1 的 `w/o MSSA` 撞名但模型不同 | §3.1/§3.2 改为**删除型**并改名 `PESR − MSSA`；与 PRD1 的替换型显式区分（§13-R12） |
| **20** | 未注意两个消融的**容量削减不对称**（−76.5% vs −1.2%） | §3.3 保留**一句限定**即可。本实验的目的是"说明模块有用"，不做跨变体排名，因此**不做容量对齐、不补 FLOPs 表** |
| **21** | 未发现 **val ⊂ test** 的既存切分缺陷（`main.py:175-182`） | §7.4 三种处置，推荐"AUPRC 为主 + F1 披露" |
| **22** | r1 的 `w/o MPA` 直接沿用 `simpip`，其嵌入是论文式的 `DataEmbedding`（Informer 风格 `Conv1d(k=3,circular)` + 正弦位置编码 + dropout，`embed.py:45`）——曾考虑一并换成单层 `nn.Linear`，但那会引入"输入侧被换弱"的混淆 | §3.2 **保留 `DataEmbedding` 不变**，只移掉 MPA 本身（切管线 + 清死参数）。若日后 `simpip` 路径效果不佳再单独评估换嵌入 |
| **23** | PRD1 与本 PRD 都叫 `w/o MSSA`，但是两个不同模型（**替换** vs **删除**） | 已定名（§3.2/§13-R12）：PRD1 → **`Mamba→Transformer`** / **`Mamba→LSTM`**；本 PRD → **`PESR − MPA`** / **`PESR − MSSA`** |
| **24** | r1 的接线方案停在纸面 | **已落地**：`make_ablation_models.py` 扩为两类消融、一次生成 4 个变体；`smoke_variants.py` 验收 **12/12 通过**（§3.2 末） |
| **25** | §5.3 引用了一张"continuous/discrete 划分表"，但**表根本不存在**——而"执行器不注入"正是 §5.5 断言 4 的依据 | 已实现 `bench/channel_split.py`（按实测唯一值数判定，tag 名仅作交叉校验）+ §5.3 的划分表：WADI 66/127 可注入、SWaT 25/51 可注入 |
| **26** | 选 (B) 的理由是"MinMax 是逐通道仿射 → 有效 α 不同"。**该理由不成立**：(B) 的 `ασ_j` 也是逐通道的 | §6.1 给出代数恒等式 + 实测（`max|A−B|=2.2e-16`）：(A)/(B) 对扰动本身**等价**，只在 α 的报告单位与依赖上有别 |
| **27** | 未发现 **WADI 的 `.npy` 已被归一化到 `[0,1]`**，而 SWaT 的是原始物理单位 | §6.2 记录；§6.3 据此改选 (A)——省掉 777 MB CSV 依赖，且避免 `σ_j` 与训练数据不同源 |
| **28** | 未发现污染 run 的结果路径与 PRD1 消融**撞车**（`main.py` 硬编码 `results/ablation/`，而两者 backbone 名相同） | §13-R16：已备份 `results/ablation/*.json`；`main.py` 增 `--result-dir` / `--result-name`，污染 run 一律写 `results/contam/` |

---

## 1. 目的

回应两条审稿意见：

- **R1｜动机不明确**：需定量说明"训练集中的异常观测会干扰正常模式学习"这一现象的真实影响。
- **R2｜传统重构方法局限性说明不足**：需说明 PESR 的模块设计为何能缓解该干扰。

**方法**：**controlled training contamination** —— 固定其余一切，只调训练集的异常污染比例 ρ，观察检测性能与重建行为如何随 ρ 变化。

**不做**：不重跑 SOTA 对比。跨模型比较（架构、重构目标、分数定义都不同）无法归因到 PESR 的内部机制，故本实验**只与自身消融比**（理由见 §3.4）。

---

## 2. 术语 → 代码映射（r1 缺失，必读）

r1 通篇使用 `MPA`/`MSSA`/`CRR`，但**这三个词在代码中没有任何对应标识符**（全仓 grep 仅命中该 PRD 自身）。实现前必须先钉死。

下面这张表的每一行都来自实际检索（检索范围见本节末），**不是推测**：

| 论文术语 | 代码实体 | 位置 | 消融方式 | 现状 |
|---|---|---|---|---|
| **MPA**（第一个模块，多 patch 分块聚合） | `data_patch` + `embedding_inpatch` + `embedding_patchwise` + `projection1/2`，`patch_size=[5,7]` | `model/Model.py:201-223` | **删除** → `PESR − MPA`（`nompa`），见 §3.2 | ✅ **已接线** |
| **MSSA**（8 层中间块的 mixer） | `self.attmamba` = 8 × `attmambaBlock` | `model/Model.py:120`，`model/mamba.py:340` | **删除** → `PESR − MSSA`（`nomssa`），1 行补丁<br>替换版见 PRD1（§3.2 命名） | ✅ **已接线** |
| **CRR** | 重构头：`encoder`/`decoder`（tabular VAE）+ `predict`（`recon_mu` 出分） | `model/Model.py:111-112, 269-294` | **不消融**（所有变体保留） | ✅ 已实现 |
| ⚠️ **（r1 未提及）** | **视图一致性损失 `loss_c`** = `x_inpatch` 与 `x_patchwise` 的余弦相似度 | `model/Model.py:218` | 无法单独消融，见 §2.1 | ✅ 已实现 |

### 2.0 两个模块的消融现状（这是 r2 的关键订正）

> ⚠️ **先分清两种消融**，否则整节都会读错：
> - **替换型**（PRD1 做的）：`attmamba` 的 mixer 换成 Transformer/LSTM —— 测"SSM mixer 是否不可替代"；
> - **删除型**（**本 PRD 做的**）：整块拿掉 —— 测"模块是否需要存在"。
>
> **r1 定稿时**：代码里只有替换型接好，删除型两个都没接（这正是 r1 不可直接执行的原因）。
> **r2 现状**：删除型两个都已接线并生成变体文件（A0 完成）。

**MSSA 的替换型：已就绪，但那不是本实验要的。** 证据链完整——`model/ABLATION_MANIFEST.md`、`make_ablation_models.py`、`main.py:20` 的 `BACKBONE_REGISTRY`、PRD1 标题"中间块消融实验"。替换是最小侵入的：只换 mixer 类，共用 `backbone.py` 的 `ResidualBlockGeneric` 骨架，各只有 **5 行 diff**；中间块 147.99 M → 16.83 M（transformer）/ 37.82 M（lstm）。命名见 §3.2。

**MSSA 的删除型：1 行补丁，已生成 `model/Model_nomssa.py`**——把 `self.attmamba` 建成空 `ModuleList` 即可（`pipline` 的循环变成 no-op，形状不变）。**已实测**：SMAP 总参数 193.52 M → **45.53 M**，四个数据集前向全部通过。详见 §3.2。

**MPA：2 处补丁，已生成 `model/Model_nompa.py`。** 代码里**存在**一条等价的删除路径 `simpip`（`model/Model.py:251-267`）：它把多 patch 嵌入换成整窗单嵌入，其余（`attmamba`→`project4`→`attention_sim`→`attention_ALL`→`projection3`→`predict`）与 `pipline` 完全一致。r1 定稿时它的 4 个调用点**全部被注释掉**（`Model.py:133`、`Model.py:174`），没有任何 flag 能打开；r2 已切到该路径（2 处），并清掉 2.39 M 死参数。`simpembdding` 仍是论文式 `DataEmbedding`，**未改**。详见 §3.2。

> ⚠️ `model/Model-nopatch(目前效果最好的了).py` 虽然文件名暗示"无 patch 且效果最好"，但它是**历史备份，不是受控消融**——本项目自己的文档已明确排除它（`docs/PRD_ablation_and_stats copy.md:14`：`Model copy.py`、`Model back4sub.py`、`Model-nopatch*.py` "都只是历史备份，不是受控消融"）。**不要拿它当 `PESR − MPA` 的结果。**

**本节检索范围**（便于复核）：全仓 `*.py`/`*.md`/`*.sh` 搜 `MPA|CRR|simpip|simp|nopatch|patchwise`；通读 `model/ABLATION_MANIFEST.md`、`script/addition/make_ablation_models.py`、`docs/替换网络`、`model/backbone.py` 的类清单、`docs/PRD1_消融实验.md` 的章节结构。

### 2.1 `loss_c` 的归属必须显式声明

`loss_c` 定义在**两个 patch 视图之间**（`criterion(x_inpatch, x_patchwise)`）。因此：

> **移除 MPA 必然同时移除 `loss_c`** —— 该损失在数学上没有 MPA 就不存在。

这在论文里必须写清楚，否则审稿人会问"你的消融到底是只去了多尺度分块，还是连一致性正则也去了"。r2 的处置（见 §3.2）是**承认二者不可分**，并把 `PESR − MPA` 定义为"去掉多视图分块聚合及其一致性正则"，在正文明确披露。若论文坚持要把 `loss_c` 作为独立的 **CRR** 来消融，则 r1 的"所有模型均保留 CRR"这一行必须删掉——二者矛盾，r1 没有察觉。

---

## 3. 实验变体

### 3.1 主实验（3 个变体，**两个消融都是"删除"**）

| 变体 | MPA | MSSA | CRR | 注册名 | 参数（SMAP） | 现状 |
|---|:-:|:-:|:-:|---|---|---|
| **PESR** | ✓ | ✓ | ✓ | `mamba` | 193.52 M | 完整模型 |
| **PESR − MPA** | ✗ | ✓ | ✓ | `nompa` | 191.13 M（−1.2%） | ✅ 已生成（§3.2）：删多视图分块 + `loss_c`（§2.1） |
| **PESR − MSSA** | ✓ | ✗ | ✓ | `nomssa` | 45.53 M（−76.5%） | ✅ 已生成（§3.2）：**直接删掉 8 层中间块** |

> `CRR` 列在所有变体中恒为 ✓（`predict`/VAE 路径一字不改），该列在论文表格中可省略，只在正文用一句话说明"未做 AE 对照"。
>
> 两个消融削减的容量差得远（−76.5% vs −1.2%），但对本实验**不构成问题**：要说明的是"各模块是否有用"，不是给两个模块排名。见 §3.3。

### 3.2 两个变体都是"删除"，各需一处生成补丁

> **命名（与 PRD1 的区别，必读）**：PRD1 的消融是**替换**（`attmamba` 的 mixer 换成 Transformer/LSTM，测"SSM mixer 是否不可替代"），本 PRD 是**删除**（整块拿掉，测"模块是否需要存在"）。**是两个不同模型，必须不同名**，否则同一张表会出现两个不一致的数。定名：
>
> | 来源 | 类型 | 定名 | 注册名 |
> |---|---|---|---|
> | PRD1 | 替换 | **`Mamba→Transformer`** / **`Mamba→LSTM`** | `transformer` / `lstm` |
> | 本 PRD | 删除 | **`PESR − MPA`** / **`PESR − MSSA`** | `nompa` / `nomssa` |
>
> 任何地方都不再出现 `w/o MSSA` 这个叫法（见 §13-R12）。

**实现**：`script/addition/make_ablation_models.py` 已扩展为覆盖**两类**消融，一次生成 4 个变体文件（`Model_{trans,lstm,nomssa,nompa}.py`），全部是"复制 `Model.py` + 精确改动"，每处补丁必须恰好命中一次否则报错退出；生成清单落在 `model/ABLATION_MANIFEST.md`。

**`PESR − MSSA`（删除）：1 行补丁。**

`pipline` 的中间块调用是 `for layer in self.attmamba: x_combined = layer(x_combined)`（`Model.py:231-232`）。把它换成空 `ModuleList`，循环自然变成 no-op，`x_combined` 保持 `[b, C, 512]` 直接进入 `project4`——形状完全不变。补丁：
```
'        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])'
→ '        self.attmamba = nn.ModuleList([])   # 消融：删除中间块（PRD4text_r2 §3.2）'
```
**实测**（`script/addition/smoke_variants.py`，四个数据集前向全通过，输出 shape 与基线一致）：

| 数据集 | C | baseline | `PESR − MSSA` | 总参数降幅 |
|---|---|---|---|---|
| SMAP | 25 | 193.52 M | **45.53 M** | −76.5% |
| MSL | 55 | 201.62 M | **53.62 M** | −73.4% |
| SWaT | 51 | 199.69 M | **51.70 M** | −74.1% |
| WADI | 127 | 221.05 M | **73.05 M** | −67.0% |

（中间块恒定 147.99 M，故降幅随 C 变化。）

> 副作用（无害）：`main.py:216` 会打印 `中间块=0.0000M (0.00%)`，与"已删除"一致。落盘 JSON 里 `n_midstack=0` 可直接用作自检。

**`PESR − MPA`（删除）：2 处补丁。**

`simpip`（`Model.py:251-267`）与 `pipline` 的差异恰好就是 MPA + `loss_c`——它把多 patch 嵌入换成整窗单嵌入，其余完全一致：

| | `pipline`（PESR） | `simpip`（`PESR − MPA`） |
|---|---|---|
| 分块嵌入 | `embedding_inpatch` + `embedding_patchwise`（`patch_size=[5,7]` 两尺度） | 单个 `simpembdding`（整窗，`Model.py:107`） |
| 一致性损失 | `loss_c ≠ 0`（`Model.py:218`） | `return ..., 0`（`Model.py:267`） |
| 其余（`attmamba`→`project4`→`attention_sim`→`attention_ALL`→`projection3`→`predict`） | — | 完全相同 |

1. **切管线**：`forward`（`Model.py:132`）与 `reconstructed_probability`（`Model.py:173`）两处 `self.pipline(` → `self.simpip(`。
2. **清掉死参数**：删掉 `Model.py:77-98` 的四个 `ModuleList`（`embedding_inpatch` / `embedding_patchwise` / `projection1` / `projection2`，共 22 行），它们在 `simpip` 路径上不再被调用。

> **`simpembdding` 保持论文式 `DataEmbedding` 不变**（`Conv1d(k=3, circular)` + 正弦 `PositionalEmbedding` + `Dropout`，`embed.py:45`）。消融只移掉被考察的模块，**不改其余组件的设置**——否则"性能差"就无法归因到"少了 MPA"还是"输入侧被换弱了"。若日后发现 `simpip` 路径本身效果不佳，**再**考虑退化成单层 `nn.Linear(window_length, indim)`，不作为一上来就做的改动。

**实测**：

| 数据集 | 切管线但保留死参数 | **清掉后（正式口径）** |
|---|---|---|
| SMAP | 193.52 M | **191.13 M**（−1.2%） |
| MSL | 201.62 M | **199.23 M**（−1.2%） |
| SWaT | 199.69 M | **198.10 M**（−0.8%） |
| WADI | 221.05 M | **218.66 M**（−1.1%） |

> **参数量必须报"清掉后"的数**。若沿用"只切管线"版，`sum(p.numel())` 会把从未执行的 2.39 M 算进去，审稿人一比对参数就会发现问题。
>
> **前置冒烟测试（A2）**：`simpip` 是历史遗留路径，能训到什么水平未知。用它在 **WADI ρ=0%** 上训练，确认能训出与 `mamba` 同量级的 F1。
> - 达标 → 纳入三变体；
> - 显著偏低 → **不要**拿它当 `PESR − MPA` 的基线（那等于"拿坏掉的路径当对照"），退回两变体（`PESR` vs `PESR − MSSA`），论文中把 MPA 消融列为 future work。

**A0 验收（已通过）**：`python script/addition/smoke_variants.py` 对 4 数据集 × {`mamba`,`nomssa`,`nompa`} 共 **12 组**构造 + 前向（`reconstructed_probability` 返回 `[b]` 分数），全部 OK；`nomssa` 的中间块参数为 0.00 M，`nompa` 相对基线削减量与上表逐项一致。

### 3.3 两个消融的容量不对称（一句限定即可）

`PESR − MSSA` 砍掉 147.99 M（−76.5%），`PESR − MPA` 只砍 2.39 M（−1.2%）。**本实验的目的是说明"这两个模块各自有用"，不是给它们排名**——所以不需要做容量对齐、不需要补 FLOPs 表、不需要再补一个"1 层中间块"的可比容量对照。

唯一要守住的一条：**结论只陈述"各自相对自身 ρ=0% 的退化"（§9.1 的归一化退化），不写"MPA 比 MSSA 更重要"这类跨变体排名**——那种说法会被容量差异解释掉。§9.1 的归一化口径天然满足这一条。

### 3.4 为什么只与自己比

`PESR − MPA` 与 `PESR − MSSA` 就是 R2 要回答的对象（PESR 的**具体设计**是否有用）。外部重建基线（AE/USAD 等）的重构目标与分数定义都不同，即使它们在污染下表现不同，也无法归因到 PESR 的内部机制。故不引入外部 baseline；若审稿人要求 external validity，**另做**一个附录实验，不混入本篇。

### 3.5 其它消融（不做）

`w/o Mutual Attention`、`w/o Global Attention`、`w/o Local Correlation`、`w/o Temporal State Tracker` 等**不纳入**。否则 motivation experiment 会膨胀成第二套完整 ablation study，反而稀释 R1/R2 的针对性。

---

## 4. 执行流水线

r1 完全没有这一节——这是它最不可实现的地方。

**总策略：先在 WADI 上把整条流水线跑通，再决定 SWaT 的规模。** 理由：WADI 训练集只有 20k 窗口（SWaT 的 1/25），跑一期只要几小时；而它会让下面四件事同时落地——(a) 验证生成器与断言正确，(b) **实测每个 epoch 的真实耗时**（§12 的预算现在是估算，pilot 后变成实测），(c) 看 ρ_win 等诊断量是否合理从而**冻结段几何参数**（§5.3），(d) 确认两条删除路径都能训（§3.2）。**在拿到这四件事之前不要开 SWaT**，否则等于用 400 GPU·h 试错。

```
── 阶段 A：WADI pilot（跑通 + 标定，≈5 GPU·h）──────────────────
A0  接线两个删除变体     →  model/Model_nomssa.py（1 行补丁）              ✅ 已完成
                            model/Model_nompa.py（切管线 + 清 2.39 M 死参数）✅ 已完成
                            BACKBONE_REGISTRY += {'nomssa','nompa'}          ✅ 已完成
                            smoke_variants.py 12/12 通过                     ✅ 已完成
A1  α 标定              →  artifacts/contam/calib_WADI.json
A2  冒烟测试            →  WADI ρ=0% × 3 变体，各 1 seed
                            （校验：nomssa 的 n_midstack==0；nompa 走历史遗留的
                              simpip 能否训出同量级 F1；三者计时）
A3  污染数据生成        →  artifacts/contam/WADI_rho<RR>_seed<S>.{npy,json}
A4  生成器校验          →  断言 nested / ρ 精确 / 非污染区逐元素相等 / ρ_win
A5  冻结决策            →  段几何定版（§5.3）+ 实测预算外推（§12）+ SWaT 规模选定

── 阶段 B：WADI 全量（≈37 GPU·h）─────────────────────────────
B1  训练+评测           →  results/contam/<bb>_WADI_rho<RR>_seed<S>.json
B2  聚合出表出图        →  results/contam/summary.{md,csv} + figs/contam_WADI.pdf

── 阶段 C：SWaT（规模由 A5 决定，69~406 GPU·h）───────────────
C1..C3  同 B，数据集换成 SWaT，全部参数与 A5 冻结值一致
```

**为什么必须先把数据物化、再训练**（用户的直觉是对的，但理由要写实）：

1. **可审计**：`.npy` 是确定的字节流，可算 SHA256、可在论文附录里给；进程内 `np.random` 生成的数据无法事后核验。
2. **跨变体同一份数据**：虽然 `set_seed` + `add_noise` 在模型创建之前（`main.py:155` vs `173`）理论上能让三个变体抽到同一串随机数，但这依赖"`main.py` 第 173 行之前不再引入任何随机调用"——**任何上游改动都会静默改变污染数据**，且无法察觉。物化后这个耦合消失。
3. **省算力**：生成一次（秒级），训练 58 次/数据集（见 §12）。进程内生成等于每次训练都要重新生成。
4. **能校验**：nested 性质（§5.3）只能在生成后断言，不能在训练时顺带检查。

### 4.1 目录与命名

```
artifacts/contam/<dataset>_rho<RR>_seed<S>.npy     # 污染后的原始尺度训练数组，[N, C] float32
artifacts/contam/<dataset>_rho<RR>_seed<S>.json    # manifest：见 §5.5
artifacts/contam/calib_<dataset>.json              # α 标定结果：见 §5.4
results/contam/<backbone>_<dataset>_rho<RR>_seed<S>.json   # 复用 main.py 现有落盘 schema
checkpoint/contam/<backbone>/<dataset>/rho<RR>_seed<S>/best.pt
nohp-out/contam/<backbone>_<dataset>_rho<RR>.out
```

`<RR>` 用两位整数（`00`,`01`,`05`,`10`,`20`），保证字典序 = 数值序，`ls` 出来天然有序。

---

## 5. Stage 0–2：污染数据生成

### 5.1 ρ 的定义

$$\rho=\frac{\text{被污染的原始时间戳数}}{\text{训练集原始时间戳总数}}$$

定义在 **raw timestamp level**（不是窗口级）——因为一个异常点会进入多个重叠窗口，按窗口统计会重复计数、无法解释。取值范围：

| ρ | 角色 |
|---|---|
| 0% | clean 参照（**必须复现论文主表 F1/AUPRC**，见 §7.3） |
| 1% | 轻度 |
| 5% | 较低 |
| 10% | 中等 |
| 20% | 压力测试 |

训练集规模（实测，`/data/sdc/yrlbkp/anomaly/`）：

| 数据集 | train 时间戳 | ρ=20% 的污染点数 | test 时间戳 |
|---|---|---|---|
| SWaT | 495,000 | 99,000 | 449,919 |
| WADI | 20,000 | 4,000 | 17,281 |

### 5.2 长度口径 —— r1 的混淆必须在这里切断

「污染 20%」有两种实现，**结论完全不同**：

| 方案 | 训练集长度 | ρ=20% 时正常样本量 | 混淆 |
|---|---|---|---|
| **(a) 替换**（固定总长 N） | N | **0.8N** | 性能下降可能是"正常数据少了"，而非"被污染了" |
| **(b) 增补**（clean ∪ anomaly） | N + 0.2N | N | 训练样本变多，与 (a) 不可比 |

**r2 采用 (a) 替换**（保持训练规模恒定，符合"污染"的字面含义），**并强制增加一个对照**：

```
clean-L80   =  把 clean 训练集确定性截断到 0.8N（同一时刻区间，无污染），ρ 记为 0%-L80
```

于是退化可以被三方分解：

$$\underbrace{P(0) - P(\rho)}_{\text{总退化}} = \underbrace{\big[P(0)-P(\text{0\%-L80})\big]}_{\text{数据量效应}} + \underbrace{\big[P(\text{0\%-L80})-P(\rho)\big]}_{\text{污染效应}}$$

**论文只需要在 ρ=20% 这一档报告这个分解**（其余档位不必）。代价是每 dataset 多 3 次训练，但没有它，"污染影响性能"这句话就立不住——这是 r1 最大的方法论缺口。

> 截断方式：取前 0.8N 个时间戳（确定性），避免再引入一个随机源。

### 5.3 异常注入形态

**连续异常段**，不是随机独立替换单点（r1 的判断正确，且旧的 `add_noise` 正是后者）：

- 每段 $[t_s,\ t_s+L]$，$L$ 从固定的段长分布抽取；
- 段内随机选一部分**连续传感器通道**施加扰动；
- 段与段不重叠（生成时去重，写进 manifest）。

**Nested 集合**（保证 ρ 增加时"只有数量变了"，而不是"每档重抽一套更难的数据"）：

$$\mathcal{A}_{1\%}\subset\mathcal{A}_{5\%}\subset\mathcal{A}_{10\%}\subset\mathcal{A}_{20\%}$$

实现：先一次性**生成 20% 这么多段**并排序；1%/5%/10% 分别取前若干段。这样嵌套是构造性成立的，且生成后可用断言验证（§5.5）。同时固定：异常类型分布、强度分布、段长分布、受影响通道分布。

**段的几何参数**（r1 未给值的自由参数，直接决定 §5.6 的 ρ_win）。**这些参数在 pilot 阶段保持可调，由阶段 A5 冻结**：

| 参数 | 初值 | 冻结时机 |
|---|---|---|
| 段数 $n_{seg}(\rho)$ | $\max\left(N_{min},\ \mathrm{round}\left(\rho N/\bar L_{typ}\right)\right)$，$N_{min}=5$，$\bar L_{typ}=300$ s | A5 |
| 实际段长 | $({\rho N})/{n_{seg}(\rho)}$ 为均值，逐段从 {0.33, 1, 3}×均值 按 5:3:2 抽 | A5 |
| 每段受影响通道数 | 从 {1, 2, 3} 按 5:3:2 抽，再从连续通道集合中均匀抽该数量的**连续**下标 | A5 |
| 异常类型比例 | 三类均匀（1/3 各） | A5 |

生成脚本把以上全部**暴露为命令行参数**（带上面的初值），不改代码就能调。冻结后写入 manifest，后续所有 ρ/seed/数据集共用同一套值。

**A5 的冻结检查表**（pilot 跑完必须逐项看，不达标就调参数重生成）：

| # | 检查 | 通过条件 |
|---|---|---|
| 1 | `ρ_win` 与标称 ρ 的比值 | 落在 [1.0, 1.5]；超出说明段太短，增大 $\bar L_{typ}$ |
| 2 | 每档段数 | $\ge N_{min}$ 且 $\le$ 20；过多说明段太碎 |
| 3 | 每档被污染窗口占比 | 严格随 ρ 单调（nested 的直接推论），否则生成器有 bug |
| 4 | 段长分布的实际均值/分位 | 与设计值偏差 <10% |

> ⚠️ **WADI 的 $\rho=1\%$ 档需要特殊处理**：WADI 训练集只有 20,000 个时间戳，$1\%=200$ s，若沿用 $\bar L=300$ s 则**不足一段**，该档位在物理上不可表示。按上面的规则 WADI 各档 $\bar L$ 会变成 40/200/200/308 s（ρ=1/5/10/20%），段数恒为 5/5/5/13。
> 但短段会**放大窗口级曝光**：ρ=1%、$\bar L=40$ s、W=70 时 $\rho_{win}\approx0.01\times(1+69/40)=2.7\%$，是标称值的 2.7 倍，违反上表检查项 1。
> **建议在 A5 直接删掉 WADI 的 1% 档**（改为 ρ∈{0,5,10,20}%），而不是用短段硬凑一个名不副实的"1%"。

冻结后附录另做一组"段长减半"的敏感性（§13-R8）。

**三类异常**（只作用在 `bench/channel_split.py` 判为 `continuous` 的通道上；`α` 为强度，`σ_j`/`μ_j` 为该通道在 **clean 训练集、模型输入空间**上的 std/mean——见 §6.3）：

| 类型 | 公式 | 模拟 |
|---|---|---|
| Level shift | $x'_{t,j}=x_{t,j}+\alpha\sigma_j$ | 传感器偏置、系统性漂移 |
| Amplitude scaling | $x'_{t,j}=\mu_j+\alpha(x_{t,j}-\mu_j)$ | 动态范围被压缩/放大 |
| Short-duration perturbation | $x'_{t,j}=x_{t,j}+\delta_{t,j}$ | 瞬时冲击、局部异常 |

**binary/discrete 执行器变量不注入**（`x + ασ` 会产生物理上不存在的状态）。若需扩展，另设 binary flip / 状态驻留实验，**不进主实验**。

**划分表**（§5.5 断言 4 与 §5.3 都要用它；已实现为 `bench/channel_split.py`，可复核、非硬编码）：

判定优先级：训练切片上 `std == 0` → `dead`；唯一值数 ≤ 4 → `discrete`；其余 → `continuous`（**= 可注入**）。

| 数据集 | C | `dead` | `discrete` | `continuous`（可注入） |
|---|---|---|---|---|
| WADI | 127 | 34 | 27 | **66** |
| SWaT | 51 | 11 | 15 | **25** |

> SWaT 的 **25 连续 = 已知的 25 个连续传感器**，15+11 = 26 = 已知的 26 个执行器，与数据集公开构造吻合——这是对该判定规则的独立验证。
>
> 三点实施注意：
> 1. **tag 名判定不完整**：`2_LS_101_AH`（高位报警）名字像模拟量、实为二值。WADI 上有 **10 个** tag 像模拟量却不可注入（6 个 `_AH` + `2_MCV_007_CO` + `2_PIC_003_SP` + `PLANT_START_STOP_LOG` + `3_AIT_001_PV`）。**必须以实测唯一值数为准**，tag 名只作交叉校验，`channel_split.py` 会把差异打印出来。
> 2. **WADI 有 4 个通道在原始 CSV 里整列是 NaN**（`2_LS_001_AL`、`2_LS_002_AL`、`2_P_001_STATUS`、`2_P_002_STATUS`），被 `preprocess` 的 NaN→0 填成 0，因而落入 `dead`。它们不是"恒定的执行器"，是**缺失传感器**——正文若提到"34 个恒零通道"，需要区分这两种成因。
> 3. WADI 的 `dead` 占比高（34/127 = 27%）会稀释污染：可注入通道只有 66 个（52%）。段内抽"连续通道"时**只能从 `injectable_idx` 里抽**，否则会往恒零通道注入 `ασ=0` 的空操作，使实际 ρ 低于标称值。

### 5.4 强度 α 的标定（Stage 0）

r1 只说"强度保持不变"，**但没给值，也没说怎么定**——这是审稿人一定会问的自由参数。r2 用**可检出来标定**：

1. 取 **clean-trained 的 PESR**（直接复用 A2 产出的 `results/contam/mamba_<ds>_rho00_seed42.json` 对应的 ckpt——ρ=0% 就是 clean 训练）；
2. 用它在 clean 数据上的 bf 阈值 $t_0$（`main.py:250` 的 `bf_search` 产出）；
3. 调整 α，使**注入段的 ≥90% 时间戳被 clean 模型判为异常**；
4. 记下 $\alpha^\*$（每 dataset 一个），写入 `calib_<ds>.json`。

**理由**：污染必须与真实异常"同等可检出"，否则要么 α 太小（等于什么都没加，曲线全平），要么 α 太大（污染变成了数据集破坏，与"真实异常混入正常训练"的动机不符）。

**敏感性**：主实验用 $\alpha^\*$；另在 $\alpha\in\{0.5\alpha^\*,2\alpha^\*\}$ 上各跑一遍 **ρ=20% 单点**作为附录，证明结论不依赖 α 的取值。

### 5.5 manifest 字段与生成后断言

```jsonc
{
  "dataset": "SWaT", "rho": 0.10, "seed": 7,
  "n_timestamps_total": 495000, "n_contaminated": 49500,
  "rho_win": 0.118,                      // 窗口级曝光率，见 §5.6
  "segments": [{"t_start": 1234, "L": 500, "type": "level_shift",
                "channels": [3, 7, 11], "alpha": 1.2}],
  "nested_key": "s7_swat_v1",            // 同一 key 下的各 ρ 共享段序
  "alpha_star": 1.2, "source": "calib_SWaT.json",
  "clean_sha256": "...", "out_sha256": "...",
  "gen_script": "bench/gen_contam.py", "gen_sha256": "..."
}
```

**必须通过的断言**（生成脚本内，失败即退出）：

1. 精确性：`n_contaminated == round(rho * n_total)`（四舍五入误差记为 0）；
2. 嵌套：$\mathcal{A}_{1\%}\subseteq\mathcal{A}_{5\%}\subseteq\mathcal{A}_{10\%}\subseteq\mathcal{A}_{20\%}$（按 `t_start` 逐段比对）；
3. 非污染区**逐元素相等**于 clean 数组（只有被注入的位置允许变化）；
4. 执行器通道（依据 §5.3 的 continuous/discrete 划分表）在污染前后完全一致。

### 5.6 窗口级污染率 $\rho_{win}$ 必须报告

ρ 定义在时间戳级（§5.1），但**模型训练在窗口上**。一段长 L 的污染段会覆盖约 $L+W-1$ 个窗口，所以

$$\rho_{win}\approx\rho\cdot\left(1+\frac{W-1}{\bar{L}}\right)$$

短段会显著放大曝光：$\bar{L}=W$ 时 $\rho_{win}\approx 2\rho$；$\bar{L}\gg W$ 时 $\rho_{win}\to\rho$。

三个后果，都要在文中交代：

1. **不同数据集的 ρ_win 不可直接横比**（W 不同：SWaT 35、其余 70），叙述时以 ρ 为准、ρ_win 附列；
2. 段长分布是**隐藏的自由参数**——它通过 ρ_win 影响模型实际看到多少污染。既然 §5.3 固定了段长分布，ρ_win 就是 ρ 的单调函数，结论不受影响，但**数值必须披露**；
3. `ρ_win` 由生成脚本直接统计（数被污染窗口数 / 总窗口数），不靠公式估算。

---

## 6. 预处理与缩放控制

**现状**（已核对代码）：`get_data()` 内部调用 `preprocess()`，而 `preprocess = MinMaxScaler().fit_transform(df)`（`utils.py:75-96`）；`add_noise()` 在 `get_data()` **之后**（`main.py:168` → `173`）。即：**scaler 永远拟合在 clean 原始训练集上，污染发生在缩放之后。**

这带来一个必须显式选择的岔路：

| 方案 | 注入空间 | 改动 | §7「scaler 只拟合一次」 |
|---|---|---|---|
| **(A) 缩放后注入** | 与模型输入同一空间（MinMax 输出） | 无（沿用现有顺序） | **自动满足**（scaler 在注入前拟合完） |
| **(B) 原始空间注入** | 真实物理单位 | 需给 `preprocess` 加 `fit_data` 参数，用 clean 数组拟合、再 transform 污染数组 | 需实现 |

### 6.1 ⚠️ r1 的二选一理由站不住：两种方案对**扰动本身**是等价的

r1 选 (B) 的理由是"方案 (A) 下 MinMax 是逐通道仿射，等价于给每个通道一个不同的有效 α"。**这个理由不成立**——(B) 也是逐通道的：`ασ_j` 里的 `σ_j` 本身就是该通道自己的 σ。

设 clean 拟合的 scaler 在第 $j$ 通道是仿射 $v\mapsto s_j(v-m_j)$（MinMax 即 $s_j=1/(\max_j-\min_j)$，$m_j=\min_j$）。则

$$\text{(B) 结果}=s_j\big[(x_{t,j}+\alpha\sigma_j^{raw})-m_j\big]=x^{norm}_{t,j}+\alpha\underbrace{s_j\,\sigma_j^{raw}}_{=\,\sigma_j^{norm}}=\text{(A) 结果}$$

**已实测确认**（SWaT 量纲跨 5 个数量级的合成数据）：`max|A−B| = 2.2e-16`（纯浮点结合律误差），未注入通道**逐位相等**。MinMaxScaler 不裁剪，故连"越界"这一条都不产生差异。

**所以 (A)/(B) 的真正差别只有两条**，都不涉及扰动内容：

1. **α 的报告单位**：物理量 vs 模型输入空间；
2. **(B) 需要原始 scaler / 原始数据**，而 **(A) 不需要**。

### 6.2 ⚠️ 新发现：WADI 的 `.npy` 已经被归一化了，SWaT 的没有

| 数据集 | `<ds>_train.npy` 的取值范围 | 是否原始物理单位 |
|---|---|---|
| SWaT | `[0, 1201.0]`，单通道如 ch1 = 120.6~817.6 | ✅ 是 |
| WADI | `[0, 1]`，127 个通道**全部** `min==0`、仅 27 个 `max==1` | ❌ 已被 MinMax 归一化 |

这意味着：**取 (B) 就得为 WADI 去读 `data/WADI/WADI_14days.csv`（777 MB）重建物理量**——raw 数据确实在（`/data/sdc/yrlbkp/anomaly/data/WADI/`），但这是 pilot 阶段本可以避免的开销。而且 §5.3 的 `σ_j`、`μ_j` 在 (B) 下要来自 CSV，与训练数据（`.npy`）**不是同一次缩放**，反而是新的不一致来源。

### 6.3 r2 改为选 (A)

由 6.1 的等价性，(A) 与 (B) 给出**同一条污染曲线**，而 (A) 少一个 777 MB 依赖、少一次缩放不一致。因此：

- **注入在与模型输入同一空间进行**，即沿用现有顺序（`get_data` → `preprocess` → 注入）；
- **α 的报告口径**改为 *"该通道在模型输入空间中的 σ 的倍数"*——对两个数据集定义一致（因为都定义在模型实际看到的张量上），比"物理单位"更贴合"模型看到了多强的污染"这个问题；
- §5.3 的 `σ_j` 改为在**模型输入空间**上统计（每数据集一份），写入 `calib_<ds>.json`；
- §13-R14 记下这个口径变更。

> ⚠️ 仍需注意（与 (A)/(B) 之争无关）：**scaler 必须在注入前拟合于 clean**。(A) 天然满足（`preprocess` 在 `add_noise` 之前）。当前 `--noise` 路径也是这个顺序，它"物理无意义"的真正原因是 **±2/±5 是绝对常数、不是 σ 的倍数**（§15），而不是顺序问题。

**`utils.py` 的 `preprocess(df, fit_data=None)` 保留但不再必须**：它仍可用于未来的原始空间注入或外部对照（(B) 的复现），已实现、`fit_data=None` 时与改动前**逐位一致**（实测 `np.array_equal`，dtype 仍 float32）。`fit_data` 生效时，未注入的正常段与"clean 全量归一化"逐位一致，注入段被推出 $[0,1]$（实测注入段均值 3.49 vs 正常段 0.52）——而自拟合会把注入压回 $[0.000,1.000]$，异常直接不可见。

**测试集/验证集的处理不变**：仍然各自拟合（与论文主表口径一致，保证 ρ=0 可复现主表）。

---

## 7. Stage 3：训练与评测

### 7.1 命令模板

```bash
# 复用 script/addition/ablation.sh 的调度骨架（3 卡并行、已有结果跳过、可重入）
for data in SWaT WADI; do
  for bb in mamba nompa nomssa; do
    for rho in 00 01 05 10 20; do
      for s in 7 17 27; do
        CUDA_VISIBLE_DEVICES=$gpu python main.py \
          --dataset $data --backbone $bb \
          --contam-file artifacts/contam/${data}_rho${rho}_seed${s}.npy \
          --scaler-from  artifacts/contam/${data}_rho00_seed${s}.npy \
          --seed $s --name contam/${bb} --trainmode train-test
      done
    done
  done
done
# length-matched 对照：多跑 3×(data,bb,seed) 的 clean-L80
```

### 7.2 main.py 需要的接口改动

| 改动 | 内容 | 必要性 |
|---|---|---|
| `--contam-file` | 若给出，则跳过 `add_noise`，直接 `np.load` 污染数组替换 `train_data`。**是方案 (A)**：污染数组已与 `.npy` 同空间，注入在 `preprocess` 之前、由脚本自己完成（§6.3） | 核心 |
| `--contam-manifest` | 读 `gen_contam.py` 的 manifest，把 `rho` / `rho_win` / `n_contaminated` / `nested_key` / `alpha_star` 写进落盘 JSON 的 `contam` 字段 | 核心 |
| `--backbone nomssa` / `nompa` | 注册 §3.2 的两个删除变体 | ✅ 已完成 |
| `--result-dir` / `--result-name` | ✅ 已完成：结果 JSON 的落盘目录与文件名。默认值 = 原行为（`results/ablation/<bb>_<ds>_seed<s>.json`）；**污染实验必须传 `results/contam`**，否则会覆盖 PRD1 的中间块消融结果（§13-R16） | 核心 |
| `--noise` | **保留不动**（旧实验仍要能跑通），但污染实验不使用它 | 兼容 |

> `--scaler-from` **不再需要**：§6.3 选定方案 (A) 后，`preprocess` 仍在 `add_noise`/`--contam-file` 之前对 clean 训练集拟合，`fit_data` 参数只作为将来复现方案 (B) 的入口保留（`utils.py:75` 已实现）。

`config.noise`、`seed`、`dataset`、`backbone`、`rho` 全部写进落盘 JSON（现有 schema 已含前三项，加 `contam` 字段）。

### 7.3 既成事实：阈值选择天然不受污染 ✅

`main.py:175-182`：验证集是**从 test 切出来的最后 35%**，不是从训练集切。

$$\text{train} \to \text{加污染},\qquad \text{val}\subset\text{test} \to \text{永不接触污染}$$

所以 `bf_search` 选出的阈值**不受 ρ 影响**，F1 的变化里不含"阈值被污染带偏"这一项。这是本设计的一个优点，**必须在论文里写出来**（否则审稿人会怀疑阈值选择是混淆源）。

**同时必须做的一个校验**：ρ=0% 且 clean-L80 两条腿的 `val` 必须与主表完全一致（同一段数据），否则阈值不可比。

**模型选择口径也一致**：`main.py:227-250` 按 val F1 选 best epoch。因为 val 不接触污染，**三个变体、五个 ρ 档的 best-epoch 判据是同一个**，污染的效应只通过训练数据进入模型，不通过模型选择。这一点同样要在论文里写明（否则会被怀疑"污染改变了早停点，从而改变了模型复杂度"）。

### 7.4 ⚠️ 已知隐患：val 与 test 重叠

同一段代码还有一处**既存缺陷**：`test_data = test_data[:-n]` 之后紧接 `val_data = test_data[-n:]`，取的是**已被截断数组**的末尾，因此

$$\text{val} = [0.30L,\ 0.65L),\qquad \text{test} = [0,\ 0.65L)\ \Rightarrow\ \text{val}\subsetneq\text{test}$$

即验证集是测试集的子集，用于选阈值的样本也被计入测试指标 → F1 **系统性偏乐观**，且偏移量是否随 ρ 变化未知。

处置（三选一，**推荐第 3 个**）：
1. 不动，仅在附录披露（最省事，但 F1 结果会被质疑）；
2. 修成 `val_data = test_data[n:]` 之类的互斥切分（干净，但会让本项目**所有历史 F1 都不可比**）；
3. **主指标用 AUPRC（阈值无关，不受此缺陷影响），F1 降级为辅助并在此处明确披露**——本 PRD 的 ρ=0% 行仍报 F1，但只用于"与主表一致性核对"，不用于跨变体比较。

---

## 8. 指标

### 8.1 检测层

| 指标 | 角色 | 说明 |
|---|---|---|
| **AUPRC** | **主** | 阈值无关、对不平衡稳健、绕开 §7.4 的重叠缺陷与 `bf_search` 的 oracle 质疑 |
| F1 | 次 | 走 PRD3.1 的 non-oracle 阈值协议；仅用于与主表口径对齐，不做跨变体主结论 |

### 8.2 重建层（test 侧）

$$E_N(\rho)=\tfrac{1}{|\mathcal D_N|}\sum_{x\in\mathcal D_N}e(x),\quad E_A(\rho)=\tfrac{1}{|\mathcal D_A|}\sum_{x\in\mathcal D_A}e(x),\quad Gap=E_A-E_N$$

$Gap\downarrow$ 是最直接的信号（重构式检测依赖正常/异常之间足够大的重构差）。

### 8.3 重建层（train-injected 侧）—— r1 的循环论证修补

r1 把 $E_A$ 定义在 **test 集**上，然后用"$E_A\downarrow$"来论证"模型学会了重建异常"。**但 test 集里的真实异常从未进入训练**，污染不可能直接教模型重建它们——这个假设在 r1 的指标下**原理上无法验证**。

r2 增加一个量：

$$E_A^{inj}(\rho)=\tfrac{1}{|\mathcal A_\rho|}\sum_{t\in\mathcal A_\rho}e(x_t)\quad(\text{被注入的训练段本身，不做留出})$$

并明确两者的分工：

| 量 | 测什么 | 支撑的结论 |
|---|---|---|
| $E_A^{inj}$ | 模型对**它见过**的异常的拟合程度 | "污染让模型把异常当正常来学"（直接证据） |
| $E_A$ | 对**未见**真实异常的泛化 | "污染损害了对真实异常的判别"（间接、更接近实际危害） |
| $E_N$ | clean 正常模式是否被破坏 | "正常模式学习受损" |
| $Gap$ | 端到端判别余量 | 与 AUPRC 最相关的量 |

### 8.4 跨变体归一化

不同变体的表示空间不同，**绝对**重建误差不可跨变体比。故对每个变体按自身 clean 状态归一化：

$$R_N(\rho)=\frac{E_N(\rho)}{E_N(0)},\qquad R_A(\rho)=\frac{E_A(\rho)}{E_A(0)},\qquad R_N(0)=R_A(0)=1$$

**统计顺序必须写死**：先在**每个 seed 内部**归一化，再对 seed 求 mean±std。若先求均值再相除，误差棒会丢失（r1 未规定顺序）。

### 8.5 可选的分离度指标

$$Sep(\rho)=\frac{\mu_A-\mu_N}{\sqrt{(\sigma_A^2+\sigma_N^2)/2}}$$

与 `Gap` 冗余。**主文只报 `Gap`**；`Sep` 作为附录的稳健性校验（若两者结论一致，说明结论不依赖具体度量选择）。

---

## 9. 分析与判据

### 9.1 图 (a) 必须画归一化退化，不能画绝对 AUPRC

三个变体在 ρ=0 时 AUPRC **本来就不同**（那是 PRD1 消融的结论）。直接画绝对 AUPRC 会把"起点低"误读成"退化快"。故：

$$\Delta\mathrm{AUPRC}(\rho)=\mathrm{AUPRC}(0)-\mathrm{AUPRC}(\rho)\quad\text{或}\quad \frac{\Delta\mathrm{AUPRC}}{\mathrm{AUPRC}(0)}$$

**主图用归一化退化；绝对曲线放附录。**（r1 的 §11.4 已经为重建误差写对了这条原则，却在 §13(a) 对 AUPRC 违反了它——r2 统一。）

### 9.2 显著性检验

同一 seed 下三个变体用**同一份**污染数据，所以**可配对**：

- 每 seed 得到 $d_s=\Delta\mathrm{AUPRC}^{\text{PESR}}_s-\Delta\mathrm{AUPRC}^{\text{PESR}-\text{MPA}}_s$；
- 3 seeds 上做单侧配对 t 检验（或 Wilcoxon signed-rank）；
- 报告 $d$ 的 mean±std 与 p 值。

仅在 mean±std 上比较、不做检验，会招致"2% 的差距在噪声内"的质疑。

### 9.3 方差来源分解（r1 的 seed 设计有混淆）

r1 的 seed 同时扰动**污染位置**与**模型初始化**，所以 $\sigma$ 里混了两个来源，无法判断退化曲线的抖动是"数据噪声"还是"训练噪声"。r2 用**嵌套 seed**：

$$\text{数据 seed}\ s_d\in\{7,17,27\}\ \times\ \text{初始化 seed}\ s_i\in\{1,2,3\}$$

- **主曲线**：每个 $s_d$ 固定对应的 $s_i$（3 条，共 3 次训练的代价与 r1 相同）；
- **方差分解**（只做 PESR × ρ=0% 和 ρ=20% 两档）：在 $s_d{=}7$ 下扫 $s_i{=}1,2,3$，报告组内 σ（初始化引起）与组间 σ（污染位置引起）。

这样"曲线抖动主要来自初始化而非污染强度"这类风险就能被直接排除，代价只有 4 次额外训练。

### 9.4 判据（不要求严格单调）

允许 $P(5\%)<P(10\%)$ 这类小波动。判据按优先级：

1. 总体趋势（ρ 高 → 性能低）；
2. mean±std 与 §9.2 的检验；
3. ρ=20% 相对 clean 的**归一化**退化幅度；
4. **不同变体的退化斜率**（这是 Q3 的核心，比绝对水平更重要）。

### 9.5 核心量化量

$$\Delta P_{20}=\frac{P(0)-P(20)}{P(0)}$$

对每个变体报一个数，作为"20% 污染下的相对性能损失"。Q3 的判据即 $\Delta P_{20}^{\text{PESR}}<\Delta P_{20}^{\text{PESR}-\text{MPA/MSSA}}$。

---

## 10. 图表规格

正文放一个 3-panel figure（排版细节见 `docs/PRD4_FigX_FigY_绘图规格.md`）。三条曲线 = `PESR − MPA` / `PESR − MSSA` / `PESR`。

| Panel | 横轴 | 纵轴 | 目的 |
|---|---|---|---|
| **(a) 检测退化** | ρ | **归一化** $\Delta\mathrm{AUPRC}$ | 谁的退化斜率更小 |
| **(b) 重建行为** | ρ | $R_N,\ R_A$ | 正常侧恶化？异常侧变易被重建？ |
| **(c) 判别余量** | ρ | $Gap$（归一化后） | 正常/异常的分数分离是否被压缩 |

**每个 panel 都必须带 error bar**（3 个数据 seed）。只画均值线不画误差棒，会被判"看不出显著性"。

**(a) 的四个关键数字**（正文用表格给出，与图互补）：每变体在 ρ=0% 与 ρ=20% 的绝对 AUPRC、$\Delta P_{20}$、以及 §5.2 的 `clean-L80` 对照值。

---

## 11. 假设与证伪条件

| # | 假设 | 若成立 → 支撑 | **什么结果会推翻它** |
|---|---|---|---|
| **H1** | $\rho\uparrow\Rightarrow \mathrm{AUPRC}\downarrow$ | R1：污染确实损害检测 | ρ=20% 时 AUPRC 与 ρ=0% 的差 < 2×std；或 `clean-L80` 的退化 ≥ 污染退化（说明是数据量效应） |
| **H2** | $\rho\uparrow\Rightarrow Gap\downarrow$（或 $Sep\downarrow$） | R1：判别余量被压缩 | $Gap$ 无单调趋势；或 $E_A$ 反而上升（污染让模型对异常更敏感，与预期机制相反） |
| **H3** | $\Delta P_{20}^{\text{PESR}}<\Delta P_{20}^{\text{PESR}-\text{MPA/MSSA}}$ | R2：MPA/MSSA 提升污染鲁棒性 | 不等式反向；或配对检验 p>0.05 |
| **H4** | PESR 的 $Sep$ 退化更小 | R2：分离更稳定 | 同 H3 |

**H1 是 H3 的前提**：若 H1 被推翻（污染根本不伤性能），则整个 motivation 叙事需要重写——**这必须在实验设计阶段就作为"可能结局"接受，而不是事后补救**。r1 通篇没有给出任何证伪条件，等于预设了结论。

**可以支持的 claim**（H3+H4 同时成立时）：

> As the training contamination ratio increases, PESR exhibits a smaller degradation in detection performance and maintains a more stable normal–anomaly separation than its ablated variants, indicating that multi-view and multi-scale representations improve the robustness of reconstruction learning against anomalous contamination.

**不能支持的 claim**：

> ~~PESR removes/filters anomalies.~~ 实验只能说明表示更稳定，不能说明模块在"剔除"异常。
> ~~PESR eliminates the influence of anomalous observations.~~ 只能说明退化更小。

---

## 12. 计算预算

**按阶段 A（WADI pilot）→ A5 实测外推 → 阶段 C 决策的方式来管，不预先承诺全量。**

### 12.1 阶段 A：WADI pilot（现在就做）

| 项 | runs | 估算 |
|---|---|---|
| A0 接线 + 生成变体 + 验收（CPU） | — | ✅ 已完成，0 GPU·h |
| A2 冒烟测试（ρ=0% × 3 变体 × 1 seed） | 3 | ~2 GPU·h |
| A3/A4 生成 + 校验（CPU，秒级~分钟级） | — | ~0 |
| 阶段 A 合计 | 3 | **~2 GPU·h（8 卡上十几分钟）** |

**A2 同时产出 §12.2 需要的实测数字**：每 epoch 真实耗时（`main.py:230` 的 10 epoch 循环里加一行计时即可）。**在 A5 之前不要启动阶段 B/C。**

### 12.2 阶段 B/C 的估算（**估算值，A5 用实测替换**）

估算方法：`docs/推理成本统计结果_v2.md` 的实测单窗口推理延时 × 训练窗口数 × 3（前向+反向≈3×前向）× 10 epoch。

| 数据集 | 训练窗口 | 单窗口 ms | 估算 /run | runs（3 变体） | 小计 |
|---|---|---|---|---|---|
| WADI | ~20k | ~3.8（b=32 实测 3.769；b=24 未测） | ~0.63 h | 58 | **~37 GPU·h** |
| SWaT（全量） | ~495k | ~1.6（b=64 实测 1.640） | ~6.8 h | 58 | **~395 GPU·h** |

**WADI 全量是安全的（37 GPU·h ≈ 8 卡 5 小时），SWaT 是大头（91%）。** 阶段 C 的规模在 A5 决定，候选：

| 方案 | SWaT 小计 | 代价 |
|---|---|---|
| SWaT 训练集下采样到 100k 窗口 | ~80 GPU·h | ρ=0% 不再复现论文主表；需一次性校验下采样不改变 ρ=0 性能 |
| SWaT 全量 | ~395 GPU·h | 占 8 卡近三天 |
| ρ 网格减为 {0, 10, 20}% | −40% | 曲线只有 3 点，趋势说服力下降 |
| seed 减为 2 | −33% | 配对检验乏力（n=2 做不了检验） |

> ⚠️ `--max_train_size` **当前不可用**：`utils.py:37` 把它乘上一个硬编码常数 `58317`（MSL 的训练长度），对其他数据集语义错误。若选下采样方案，需先修这行。

---

## 13. 风险与对策

| # | 风险 | 对策 |
|---|---|---|
| **R0** | ~~MPA 的消融根本没有接线~~ | **已闭环**（A0）：`make_ablation_models.py` 扩为两类消融、生成 4 个变体，`BACKBONE_REGISTRY` 已注册；`smoke_variants.py` **12/12 通过**。剩下的是 A2 的"能否训出同量级 F1" |
| R1 | **H1 不成立**：污染不伤性能（可能因为合成异常太"假"，模型轻易忽略） | §5.4 的 α 标定保证异常可检出；若仍无退化，改为报告"PESR 对污染不敏感"这一正面结论，并检查 α 敏感性（0.5α*/2α*） |
| R2 | **数据量混淆**（§5.2） | `clean-L80` 对照；ρ=20% 档必报分解 |
| R3 | **val∩test 重叠**（§7.4） | AUPRC 为主指标；F1 明确披露 |
| R4 | **SWaT 训练集并非严格 clean** | 先测：clean-trained 模型在 SWaT train 上的平均分数分布。若 train 侧已有高分点，说明训练集含真实异常，需在文中改为"additional synthetic contamination on top of a not-provably-clean training set" |
| R5 | **`simpip` 是历史遗留路径**（§3.2），能否训出可用性能未知 | 由 A2 的 WADI ρ=0% 冒烟测试判定；不达标则退回两变体，**不要**拿坏掉的路径当对照 |
| R6 | **`loss_c` 的归属**（§2.1） | 正文显式声明 `PESR − MPA` 同时去掉一致性正则；不试图单独消融它 |
| R7 | **ρ_win ≠ ρ** 导致跨数据集误比（§5.6） | 叙述以 ρ 为准、ρ_win 附列；段网格用 A5 的检查表冻结 |
| R8 | **段长/通道数分布是隐藏自由参数** | A5 冻结并写进 manifest；附录报告在 2 组不同段长分布下 ρ=20% 的结果 |
| R9 | **算力失控** | 分阶段：先 pilot（2 GPU·h）→ 实测外推 → 再决定 SWaT 规模（§12） |
| R10 | **WADI 的 ρ=1% 档无法用合理段长表示**（§5.3） | 建议 A5 直接删掉该档（WADI 用 {0,5,10,20}%），不要用短段硬凑 |
| R11 | **WADI 训练集仅 20k**，段数下限 5 时各档段数都是 5，实际只有段长在变 | 三个 seed 的段位置只有 5 个自由选择 → 方差大。若 §9.2 的配对检验不显著，需增加 seed 或加长段 |
| R12 | **与 PRD1 的 `w/o MSSA` 撞名**：PRD1 是替换型（`attmamba`→Transformer/LSTM），本 PRD 是删除型（整块拿掉）。同一篇论文里两个不同模型用同一个名字，表格会出现两个不一致的数 | **已定名**（§3.2）：PRD1 → `Mamba→Transformer` / `Mamba→LSTM`；本 PRD → `PESR − MPA` / `PESR − MSSA`。两张表同时出现时在表注点明二者不同 |
| R13 | 两个消融容量不对称（−76.5% vs −1.2%，§3.3）——但本实验只说明"模块各自有用"，不排名 | 只需一句限定（§3.3）：结论只写"各自相对自身 ρ=0% 的退化"，不写跨变体排名。**不做**容量对齐、不补 FLOPs 表 |
| R14 | 注入空间的口径（§6.3 由 (B) 改为 (A)）：α 报"模型输入空间 σ 的倍数"而非物理单位 | 文中明确写出口径即可；物理单位版本可另做附录（(B) 已可由 `preprocess(fit_data=...)` 复现）。**不得**混用两个口径的数字 |
| R15 | **WADI 的 `dead` 通道占 34/127**，其中 4 个是原始 CSV 整列 NaN（缺失传感器）而非恒定执行器 | 正文区分两种成因；段内抽通道只在 `injectable_idx`（66 个）里抽，否则 `ασ=0` 的空操作会使实际 ρ 低于标称值（§5.3 注意 3） |
| R16 | **污染 run 会覆盖 PRD1 的消融结果**：`main.py` 原先硬编码 `results/ablation/<bb>_<ds>_seed<s>.json`，而 ρ=0% 的污染 run 用的 backbone 名与 PRD1 完全相同（`mamba`），文件名撞车 | **已发生并已处置**（A2 首次跑就撞上了）：出全量前备份 `results/ablation/*.json`；`main.py` 增加 `--result-dir` / `--result-name`，污染 run 一律写 `results/contam/`。**所有污染调度脚本都必须传这两个参数** |

---

## 14. 交付物清单

| 类型 | 路径 | 谁产出 |
|---|---|---|
| 连续/离散通道划分 | `artifacts/contam/channel_split_<ds>.json`（含 `injectable_idx`） | ✅ `bench/channel_split.py` |
| α 标定 | `artifacts/contam/calib_<ds>.json` | `bench/gen_contam.py --calib` |
| 污染数据 + manifest | `artifacts/contam/<ds>_rho<RR>_seed<S>.{npy,json}` | `bench/gen_contam.py` |
| 生成器校验报告 | `artifacts/contam/verify_<ds>.md`（含 §5.5 四项断言结果） | `bench/verify_contam.py` |
| 单次训练结果 | `results/contam/<bb>_<ds>_rho<RR>_seed<S>.json` | `main.py`（现有 schema + `timing` 字段） |
| 聚合表 | `results/contam/summary.{md,csv}` | `bench/summarize_contam.py` |
| 主图 | `figs/contam_<ds>.pdf`（3 panel + error bar） | `bench/plot_contam.py` |
| A2 冒烟测试脚本 | `script/contam/A2_smoke_WADI.sh` | ✅ 已写好，正在跑 |
| 调度脚本 | `script/contam/<ds>.sh`（仿 `script/addition/ablation.sh`） | 手写 |

**代码改动汇总**

| 文件 | 改动 | 阶段 |
|---|---|---|
| `script/addition/make_ablation_models.py` | ✅ **已完成**：扩为两类消融（替换型 op `replace` + 删除型 op `drop`），生成 4 个变体，每处补丁必须命中一次 | A0 ✅ |
| `model/Model_nomssa.py` / `model/Model_nompa.py`（**生成物，禁止手改**） | ✅ 已生成（diff 2 行 / 27 行） | A0 ✅ |
| `script/addition/smoke_variants.py` | ✅ 新增：A0 验收脚本（构造 + 前向 + 参数量断言） | A0 ✅ |
| `main.py:20` | ✅ `BACKBONE_REGISTRY` 加 `'nomssa'` / `'nompa'` | A0 ✅ |
| `bench/channel_split.py` | ✅ **已完成**：连续/离散通道划分表（§5.3），产出 `artifacts/contam/channel_split_<ds>.json`；`gen_contam.py` 直接 import 它拿 `injectable_idx` | A0 ✅ |
| `utils.py:75` | ✅ **已完成**：拆出 `_clean()`，`preprocess(df, fit_data=None)` — `None` 时逐位不变（§6.3：改为方案 (A) 后不再必须，保留供 (B) 复现/附录） | A0 ✅ |
| `main.py` | ✅ 已完成：epoch 循环加计时，落盘 JSON 增 `timing` 字段（`epoch_secs` / `mean_epoch_sec` / `n_windows`），供 §12.2 用实测替换估算 | A0 ✅ |
| `main.py` | 加 `--contam-file` / `--contam-manifest`（与 A3 的 `.npy`+manifest 格式契约一起定，故与 `bench/gen_contam.py` 同批做）；落盘 JSON 加 `contam` 字段 | A3 |
| `bench/gen_contam.py` | 新增：§5 的全部生成逻辑 + §5.5 四项断言 | A3 |
| `bench/verify_contam.py` | 新增：独立复核断言并出 `verify_<ds>.md` | A4 |
| `bench/summarize_contam.py` / `plot_contam.py` | 新增：聚合表 + 3-panel 图 | B2 |
| `script/contam/<ds>.sh` | 新增：仿 `script/addition/ablation.sh` 的调度骨架 | B1 |

**阶段 A（WADI pilot）的验收清单**——全绿才算"跑通全流程"：

1. ~~A0 接线后 `--backbone {nomssa,nompa}` 都能起训~~ → ✅ **已通过**：`smoke_variants.py` 12/12（4 数据集 × 3 变体）前向 OK；
2. `nomssa` 的落盘 JSON 里 `n_midstack == 0`（证明中间块真的被删了），且总参数 ≈ **73.05 M**（WADI，C=127）；
3. `nompa` 的总参数 ≈ **218.66 M**（**不是** 221.05 M——后者说明死参数没清）；
4. A2 三个变体在 WADI ρ=0% 上都能训出与 `mamba` 同量级的 F1（`nompa` 是唯一有实质不确定性的：走的 `simpip` 是历史遗留路径）；
5. A3 生成的各档 `.npy` 全部通过 §5.5 的四项断言；
6. A4 的 `verify_WADI.md` 中 ρ_win 通过 §5.3 的冻结检查表；
7. A5 产出两个数：**实测每 epoch 耗时**（替换 §12.2 的估算）与**冻结后的段几何**；
8. `results/contam/` 与 `figs/contam_WADI.pdf` 能端到端生成出来。

---

## 15. 与现有 robustness 实验的关系

`script/robust/{SWaT,WADI,msl,SMAP}.sh` 是现役实验，用 `--noise 0.5`（**单点 ρ=50%**），`add_noise`（`main.py:88-112`）做的是**随机独立时间戳替换**，噪声为 uniform(±2) / spike(±5) / random-walk 三选一。

**它不能承担 R1/R2**，原因有二（r1 只说了第一条）：

1. **形态不符**：随机独立替换不具时间结构，测的是"噪声鲁棒性"，不是"异常污染下的正常模式学习"；
2. **强度无口径**：±2 / ±5 是**绝对常数**，不是该通道 σ 的倍数——同一个 ±5 对量纲跨几个数量级的通道意味着完全不同的强度，跨通道/跨数据集都不可比。**注意**：这与注入先后无关（§6.1 已证明 (A)/(B) 等价），§6.3 改选 (A) 之后这一点仍未解决，只是本 PRD 用 `ασ_j` 取代了绝对常数。

**处置**：保留 `script/robust/*.sh` 作为**附录的噪声鲁棒性实验**（它是独立的、可报的结果），`--noise` 代码路径不动。本 PRD 的污染实验是**新增**，不替换它。论文中两者分开表述：

| 实验 | 测什么 | 对应审稿点 |
|---|---|---|
| `script/robust/*`（ρ=0.5 随机替换） | 输入噪声鲁棒性 | 附录 |
| 本 PRD（ρ 曲线，结构化污染段） | 训练期异常污染的动机验证 | R1 / R2 主文 |

---

## 16. 最终配置一览

| 项 | 设定 |
|---|---|
| **执行顺序** | **先 WADI pilot（2 GPU·h）→ A5 冻结参数与实测预算 → 再决定 SWaT 规模**（§4/§12）|
| 数据集 | WADI（先）、SWaT（后，规模待 A5 定） |
| ρ | WADI: 0/5/10/20%（1% 档 A5 建议删）；SWaT: 0/1/5/10/20% |
| runs | 3 变体 × ρ 档数 × 3 seed + 9 (L80) + 4 (方差分解)；WADI 现为 **49**，SWaT 为 **58** |
| 段几何 | **pilot 期可调**（初值 $n_{seg}=\max(5,\mathrm{round}(\rho N/300))$、长度 {0.33,1,3}× 均值、通道数 {1,2,3}），A5 按 §5.3 检查表冻结 |
| 注入层级 | timestamp level，**在模型输入空间**注入（§6.3 方案 (A)；scaler 拟合于 clean，天然满足） |
| 可注入通道 | 仅 `channel_split.py` 判为 `continuous` 的：WADI 66、SWaT 25（§5.3） |
| 注入形态 | 连续异常段，段内从 `injectable_idx` 抽连续通道；nested 集合 |
| 异常类型 | level shift / amplitude scaling / short-duration perturbation |
| 强度 α | $\alpha^\*$ 由 clean 模型 ≥90% 检出来标定；敏感性别测 0.5α*/2α* |
| 变体 | `mamba`（PESR）/ `nompa`（PESR − MPA）/ `nomssa`（PESR − MSSA）|
| 消融类型 | **两者都是"删除"**，不是替换（替换型是 PRD1 的另一组实验，命名见 §3.2）|
| 容量（SMAP） | 193.52 / 191.13 / 45.53 M（不对称，但只需一句限定，§3.3；不报 FLOPs）|
| `PESR − MPA` 的嵌入 | **保持论文式 `DataEmbedding` 不变**（只移 MPA 本身，不改其余组件；§3.2）|
| 一致性正则 | `loss_c` 随 MPA 一并移除，正文显式披露 |
| seed | 数据 seed {7,17,27} × 初始化 seed {1,2,3}（嵌套，§9.3） |
| 主指标 | AUPRC（阈值无关） |
| 次指标 | F1（走 PRD3.1 非 oracle 协议；披露 val∩test 重叠） |
| 重建指标 | $E_N$、$E_A$、$E_A^{inj}$、$Gap$、归一化 $R_N/R_A$ |
| 统计 | 每 seed 内归一化 → mean±std → 配对检验 |
| 预算 | WADI pilot 2 GPU·h → WADI 全量 37 GPU·h → SWaT 规模待 A5（§12）|
