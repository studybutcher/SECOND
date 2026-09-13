# 消融变体生成清单

源文件：`model/Model.py`（384 行）

由 `script/addition/make_ablation_models.py` 生成，禁止手工编辑变体文件。

## `model/Model_trans.py`

替换为 `TransformerResidualBlock`，diff 共 5 行：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import TransformerResidualBlock
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([TransformerResidualBlock(args) for _ in range(args.n_layer)])
```

## `model/Model_lstm.py`

替换为 `LSTMMixerBlock`，diff 共 5 行：

```diff
-from .mamba import ModelArgs, ResidualBlock
+from .mamba import ModelArgs
+from .backbone import LSTMMixerBlock
-        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
+        self.attmamba = nn.ModuleList([LSTMMixerBlock(args) for _ in range(args.n_layer)])
```

