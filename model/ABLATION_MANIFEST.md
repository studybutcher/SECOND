# 消融变体生成清单

源文件：`model/Model.py`（384 行）

由 `script/addition/make_ablation_models.py` 生成，禁止手工编辑变体文件。

| 变体 | 类型 | 改动 | 注册名 |
|---|---|---|---|
| `Model_trans.py` | 替换 | 中间块 mixer → Transformer | `transformer` |
| `Model_lstm.py` | 替换 | 中间块 mixer → 双向 LSTM | `lstm` |
| `Model_nomssa.py` | 删除 | 删掉 8 层中间块 | `nomssa` |
| `Model_nompa.py` | 删除 | 删掉 MPA 多视图分块 + `loss_c`，嵌入→单层线性 | `nompa` |

## `model/Model_trans.py`

替换为 `TransformerResidualBlock`（PRD1：测 SSM mixer 是否不可替代）。

diff 共 5 行：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import TransformerResidualBlock
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([TransformerResidualBlock(args) for _ in range(args.n_layer)])
```

## `model/Model_lstm.py`

替换为 `LSTMMixerBlock`（PRD1：测 SSM mixer 是否不可替代）。

diff 共 5 行：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import LSTMMixerBlock
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([LSTMMixerBlock(args) for _ in range(args.n_layer)])
```

## `model/Model_nomssa.py`

整段删除 8 层中间块（PRD4text_r2：测该模块是否需要存在）。`pipline` 里是 `for layer in self.attmamba`，空 ModuleList 使循环成为 no-op，`x_combined` 保持 `[b, C, 512]`，形状不变。

diff 共 2 行：

```diff
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([])   # 消融：删除中间块（PRD4text_r2 §3.2）
```

## `model/Model_nompa.py`

删除 MPA（多视图分块 + `loss_c`）。两处改动：清掉 `embedding_inpatch` / `embedding_patchwise` / `projection1` / `projection2` 四个不再被调用的模块；`pipline` → `simpip`（`simpembdding` 保持论文式 `DataEmbedding` 不变）。

diff 共 27 行：

```diff
-        self.embedding_inpatch = nn.ModuleList()
-        self.embedding_patchwise = nn.ModuleList()
-        self.attentiona = nn.ModuleList()
-        self.attentionb = nn.ModuleList()
-        self.projection1 = nn.ModuleList()
-        self.projection2 = nn.ModuleList()
-        for i, patchsize in enumerate(self.patch_size):
-            patchnum=window_length //patchsize
-            self.embedding_inpatch.append(DataEmbedding(patchsize, self.latent_size, dropout))#?没设置dropout
-            self.embedding_patchwise.append(DataEmbedding(patchnum, self.latent_size, dropout))
-            # if self.input_size == 25:
-            #     head = 5
-            # elif self.input_size == 51: 
-            #     head = 3
-            # elif self.input_size == 127: 
-            #     head = 1
-            # elif self.input_size == 55: 
-            #     head = 5
-            # self.attentiona.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
-            # self.attentionb.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
-            self.projection1.append(nn.Linear(patchnum*self.latent_size,self.latent_size))#?这个可能每个实验都要改？怎么用其他变量转化
-            self.projection2.append(nn.Linear(patchsize*self.latent_size,self.latent_size))
+        # 消融：此处删除 22 行 —— 清掉 MPA 死参数（embedding_inpatch/patchwise + projection1/2）（PRD4text_r2 §3.2）
-        x_a,pred_result,loss_c = self.pipline(x)
+        x_a,pred_result,loss_c = self.simpip(x)   # 消融：无 MPA 管线
-            x_a,pred_result,_ = self.pipline(x,'test')
+            x_a,pred_result,_ = self.simpip(x,'test')   # 消融：无 MPA 管线
```

## `model/Model_nompamssa.py`

同时删除 MPA 与 MSSA（前两者的并集）。剩余的路径：`simpembdding` → `project4` → `attention_sim` → `attention_ALL` → `projection3` → `predict`。

diff 共 29 行：

```diff
-        self.embedding_inpatch = nn.ModuleList()
-        self.embedding_patchwise = nn.ModuleList()
-        self.attentiona = nn.ModuleList()
-        self.attentionb = nn.ModuleList()
-        self.projection1 = nn.ModuleList()
-        self.projection2 = nn.ModuleList()
-        for i, patchsize in enumerate(self.patch_size):
-            patchnum=window_length //patchsize
-            self.embedding_inpatch.append(DataEmbedding(patchsize, self.latent_size, dropout))#?没设置dropout
-            self.embedding_patchwise.append(DataEmbedding(patchnum, self.latent_size, dropout))
-            # if self.input_size == 25:
-            #     head = 5
-            # elif self.input_size == 51: 
-            #     head = 3
-            # elif self.input_size == 127: 
-            #     head = 1
-            # elif self.input_size == 55: 
-            #     head = 5
-            # self.attentiona.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
-            # self.attentionb.append(attentionstacking(selfattentionblock,head=4,indim=self.latent_size,feeddim=512))
-            self.projection1.append(nn.Linear(patchnum*self.latent_size,self.latent_size))#?这个可能每个实验都要改？怎么用其他变量转化
-            self.projection2.append(nn.Linear(patchsize*self.latent_size,self.latent_size))
+        # 消融：此处删除 22 行 —— 清掉 MPA 死参数（embedding_inpatch/patchwise + projection1/2）（PRD4text_r2 §3.2）
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([])   # 消融：删除中间块
-        x_a,pred_result,loss_c = self.pipline(x)
+        x_a,pred_result,loss_c = self.simpip(x)   # 消融：无 MPA 管线
-            x_a,pred_result,_ = self.pipline(x,'test')
+            x_a,pred_result,_ = self.simpip(x,'test')   # 消融：无 MPA 管线
```

