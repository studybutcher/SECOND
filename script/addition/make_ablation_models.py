#!/usr/bin/env python
"""从 model/Model.py 生成消融变体 model/Model_{trans,lstm}.py

消融只替换中间块（`self.attmamba` 那 8 层）的 mixer，其余一字不改。
生成方式：复制 + 精确替换 2 处（共 5 行 diff），保证与基线可审计。

用法：
    python script/addition/make_ablation_models.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, 'model', 'Model.py')

# (旧串, 新串, 说明)  —— 每处必须恰好命中一次
PATCHES = [
    ('from .mamba import ModelArgs, ResidualBlock',
     'from .mamba import ModelArgs\nfrom .backbone import {cls}',
     'import：ResidualBlock → 变体 residual block'),
    ('        self.attmamba = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])',
     '        self.attmamba = nn.ModuleList([{cls}(args) for _ in range(args.n_layer)])',
     '中间块：替换 mixer'),
]

VARIANTS = {
    'trans': 'TransformerResidualBlock',
    'lstm': 'LSTMMixerBlock',
}


def make(src_text, cls):
    out = src_text
    for old, new_tpl, desc in PATCHES:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f'[FAIL] 待替换串命中 {n} 次（应为 1 次）：{desc}\n  {old!r}')
        out = out.replace(old, new_tpl.format(cls=cls))
    return out


def diff_lines(a, b):
    import difflib
    d = [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='')
         if l[:1] in '+-' and not l.startswith(('+++', '---'))]
    return d


def main():
    with open(SRC) as f:
        src = f.read()

    manifest = ['# 消融变体生成清单', '',
                f'源文件：`model/Model.py`（{len(src.splitlines())} 行）',
                '',
                '由 `script/addition/make_ablation_models.py` 生成，禁止手工编辑变体文件。',
                '']

    for suffix, cls in VARIANTS.items():
        new_text = make(src, cls)
        out_path = os.path.join(REPO_ROOT, 'model', f'Model_{suffix}.py')
        with open(out_path, 'w') as f:
            f.write(new_text)
        d = diff_lines(src, new_text)
        print(f'[OK] model/Model_{suffix}.py   类={cls}   diff={len(d)} 行')
        for line in d:
            print('       ' + line)
        manifest += [f'## `model/Model_{suffix}.py`', '',
                     f'替换为 `{cls}`，diff 共 {len(d)} 行：', '', '```diff'] + d + ['```', '']

    with open(os.path.join(REPO_ROOT, 'model', 'ABLATION_MANIFEST.md'), 'w') as f:
        f.write('\n'.join(manifest) + '\n')
    print('[OK] model/ABLATION_MANIFEST.md')


if __name__ == '__main__':
    main()
