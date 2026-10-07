#!/usr/bin/env python
"""从 model/Model.py 生成消融变体

两类消融，都是"复制 + 精确改动"，保证与基线可审计：

  替换型（PRD1：测 SSM mixer 是否不可替代）
      model/Model_trans.py   中间块 mixer -> Transformer
      model/Model_lstm.py    中间块 mixer -> 双向 LSTM

  删除型（PRD4text_r2：测模块本身是否需要存在）
      model/Model_nomssa.py  删掉 8 层中间块
      model/Model_nompa.py   删掉 MPA 多视图分块（+ loss_c），
                             嵌入退化为单层线性（不用论文的 DataEmbedding）

禁止手工编辑变体文件；改基线后重跑本脚本。

用法：
    python script/addition/make_ablation_models.py
"""
import difflib
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, 'model', 'Model.py')

# ---------------------------------------------------------------- 替换型（PRD1）
# (kind, 匹配串, 替换串, 说明)；`{cls}` 由 VARIANTS 填入
SUBSTITUTION_OPS = [
    ('replace', 'from .mamba import ModelArgs, ResidualBlock',
     'from .mamba import ModelArgs\nfrom .backbone import {cls}',
     'import：ResidualBlock → 变体 residual block'),
    ('replace',
     '        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])',
     '        self.attmamba = nn.ModuleList([{cls}(args) for _ in range(args.n_layer)])',
     '中间块：替换 mixer'),
]

SUBSTITUTION_VARIANTS = {
    'trans': 'TransformerResidualBlock',
    'lstm': 'LSTMMixerBlock',
}

# ---------------------------------------------------------------- 删除型（PRD4text_r2）
# kind='drop' 表示按"首行锚点/末行锚点"整段删除（含两端），锚点须整行精确命中，
# 这样不必去匹配区间内部的注释与空白。
DELETION_VARIANTS = {
    # 删掉 8 层中间块：pipline 里的调用是 `for layer in self.attmamba: ...`，
    # 空 ModuleList 让循环变成 no-op，x_combined 仍是 [b, C, 512]，形状不变。
    'nomssa': [
        ('replace',
         '        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])',
         '        self.attmamba = nn.ModuleList([])   # 消融：删除中间块（PRD4text_r2 §3.2）',
         '中间块：整段删除'),
    ],
    # 删掉 MPA：切到无分块的 simpip 管线，并清掉它不再用到的四个模块。
    # 注意：simpip 自带的 `simpembdding` 仍是论文式的 DataEmbedding，**保持不变**——
    # 消融只移掉被考察的模块，不改其余组件的设置。若日后 simplex 路径效果不佳，
    # 再考虑换成单层 `nn.Linear(window_length, indim)`（见 PRD4text_r2 §3.2）。
    'nompa': [
        ('drop',
         '        self.embedding_inpatch = nn.ModuleList()',
         '            self.projection2.append(nn.Linear(patchsize*self.latent_size,self.latent_size))',
         '清掉 MPA 死参数（embedding_inpatch/patchwise + projection1/2）'),
        ('replace',
         '        x_a,pred_result,loss_c = self.pipline(x)',
         '        x_a,pred_result,loss_c = self.simpip(x)   # 消融：无 MPA 管线',
         'forward：pipline → simpip'),
        ('replace',
         "            x_a,pred_result,_ = self.pipline(x,'test')",
         "            x_a,pred_result,_ = self.simpip(x,'test')   # 消融：无 MPA 管线",
         'reconstructed_probability：pipline → simpip'),
    ],
    # 同时删掉两个模块 = MPA 的改动 ∪ MSSA 的改动。两者互不重叠：
    #   MPA  → 切 simpip 管线 + 清掉 4 个模块（都在 __init__ 与两处调用点）
    #   MSSA → 把 attmamba 换成空 ModuleList
    # 合起来后模型退化为：simpembdding → project4 → attention_sim → attention_ALL
    #                     → projection3 → predict（只有 CRR 头和两个 attention 保留）
    'nompamssa': [
        ('drop',
         '        self.embedding_inpatch = nn.ModuleList()',
         '            self.projection2.append(nn.Linear(patchsize*self.latent_size,self.latent_size))',
         '清掉 MPA 死参数（embedding_inpatch/patchwise + projection1/2）'),
        ('replace',
         '        x_a,pred_result,loss_c = self.pipline(x)',
         '        x_a,pred_result,loss_c = self.simpip(x)   # 消融：无 MPA 管线',
         'forward：pipline → simpip'),
        ('replace',
         "            x_a,pred_result,_ = self.pipline(x,'test')",
         "            x_a,pred_result,_ = self.simpip(x,'test')   # 消融：无 MPA 管线",
         'reconstructed_probability：pipline → simpip'),
        ('replace',
         '        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])',
         '        self.attmamba = nn.ModuleList([])   # 消融：删除中间块',
         '中间块：整段删除'),
    ],
}


def apply_ops(text, ops, cls=None):
    """按顺序执行补丁。每个 op 必须精确命中，否则报错退出（不给静默失败的机会）。"""
    out = text
    for kind, a, b, desc in ops:
        if kind == 'replace':
            old, new = a, (b.format(cls=cls) if cls else b)
            n = out.count(old)
            if n != 1:
                raise SystemExit(f'[FAIL] 待替换串命中 {n} 次（应为 1 次）：{desc}\n  {old!r}')
            out = out.replace(old, new)
        elif kind == 'drop':
            lines = out.split('\n')
            hits = [i for i, l in enumerate(lines) if l == a]
            if len(hits) != 1:
                raise SystemExit(f'[FAIL] 首行锚点命中 {len(hits)} 次（应为 1 次）：{desc}\n  {a!r}')
            i = hits[0]
            tail = [j for j, l in enumerate(lines) if l == b and j >= i]
            if len(tail) != 1:
                raise SystemExit(f'[FAIL] 末行锚点命中 {len(tail)} 次（应为 1 次）：{desc}\n  {b!r}')
            j = tail[0]
            lines[i:j + 1] = [f'        # 消融：此处删除 {j - i + 1} 行 —— {desc}（PRD4text_r2 §3.2）']
            out = '\n'.join(lines)
        else:
            raise SystemExit(f'[FAIL] 未知 op：{kind}')
    return out


def diff_lines(a, b):
    return [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='')
            if l[:1] in '+-' and not l.startswith(('+++', '---'))]


def emit(src, name, ops, cls, manifest, note):
    new_text = apply_ops(src, ops, cls)
    with open(os.path.join(REPO_ROOT, 'model', f'Model_{name}.py'), 'w') as f:
        f.write(new_text)
    d = diff_lines(src, new_text)
    head = f'Model_{name}.py' + (f'   类={cls}' if cls else '')
    print(f'[OK] model/{head}   diff={len(d)} 行')
    for line in d:
        print('       ' + line)
    manifest += [f'## `model/Model_{name}.py`', '', note, '',
                 f'diff 共 {len(d)} 行：', '', '```diff'] + d + ['```', '']


def main():
    with open(SRC) as f:
        src = f.read()

    manifest = ['# 消融变体生成清单', '',
                f'源文件：`model/Model.py`（{len(src.splitlines())} 行）', '',
                '由 `script/addition/make_ablation_models.py` 生成，禁止手工编辑变体文件。', '',
                '| 变体 | 类型 | 改动 | 注册名 |', '|---|---|---|---|',
                '| `Model_trans.py` | 替换 | 中间块 mixer → Transformer | `transformer` |',
                '| `Model_lstm.py` | 替换 | 中间块 mixer → 双向 LSTM | `lstm` |',
                '| `Model_nomssa.py` | 删除 | 删掉 8 层中间块 | `nomssa` |',
                '| `Model_nompa.py` | 删除 | 删掉 MPA 多视图分块 + `loss_c`，嵌入→单层线性 | `nompa` |',
                '']

    for name, cls in SUBSTITUTION_VARIANTS.items():
        emit(src, name, SUBSTITUTION_OPS, cls, manifest,
             f'替换为 `{cls}`（PRD1：测 SSM mixer 是否不可替代）。')

    for name, ops in DELETION_VARIANTS.items():
        note = {
            'nomssa': '整段删除 8 层中间块（PRD4text_r2：测该模块是否需要存在）。'
                      '`pipline` 里是 `for layer in self.attmamba`，空 ModuleList 使循环成为 no-op，'
                      '`x_combined` 保持 `[b, C, 512]`，形状不变。',
            'nompa': '删除 MPA（多视图分块 + `loss_c`）。两处改动：清掉 `embedding_inpatch` / '
                     '`embedding_patchwise` / `projection1` / `projection2` 四个不再被调用的模块；'
                     '`pipline` → `simpip`（`simpembdding` 保持论文式 `DataEmbedding` 不变）。',
            'nompamssa': '同时删除 MPA 与 MSSA（前两者的并集）。剩余的路径：'
                         '`simpembdding` → `project4` → `attention_sim` → `attention_ALL` '
                         '→ `projection3` → `predict`。',
        }[name]
        emit(src, name, ops, None, manifest, note)

    with open(os.path.join(REPO_ROOT, 'model', 'ABLATION_MANIFEST.md'), 'w') as f:
        f.write('\n'.join(manifest) + '\n')
    print('[OK] model/ABLATION_MANIFEST.md')


if __name__ == '__main__':
    main()
