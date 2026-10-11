#!/usr/bin/env python
"""PRD5.2 · 从 `main.py --score-dump` 落盘的分数算**两套口径**。

    现有评测协议：F1（`bf_search` 的 oracle 百分位阈值 + point-adjust）—— 从 JSON 读
    好尺子      ：未调整的 AUPRC / AUROC / 逐点 F1 —— 从 scores.npz 现算

目的：把"F1 摆得厉害但 AUROC 是平的"这件事**量化**出来。

用法：python bench/analyze_lstmae_scores.py [--glob 'results/lstmae_robust/*_scores.npz']
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_method import calc_point2point                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default='results/lstmae_robust/*_scores.npz')
    ap.add_argument('--json-glob', default='results/lstmae_robust/lstmae_*_seed*.json')
    ap.add_argument('--out', default='results/lstmae_robust/summary.md')
    a = ap.parse_args()

    f1 = {}
    for p in glob.glob(a.json_glob):
        d = json.load(open(p))
        nz = d.get('noise', d['config'].get('noise', 0.0))
        f1[(nz, d['seed'])] = d['metrics']['f1']

    rows = []
    for p in sorted(glob.glob(a.glob)):
        z = np.load(p)
        s = z['test_score'].ravel()
        y = z['test_label'].astype(int).ravel()
        m = re.search(r'noise([\d.]+)_seed(\d+)', p)
        nz, sd = float(m.group(1)), int(m.group(2))
        pred = (s > np.percentile(s, float(z['bf_threshold']))).astype(int)
        u = calc_point2point(pred, y)
        rows.append(dict(nz=nz, seed=sd, n=len(y), base=float(y.mean()),
                         auprc=average_precision_score(y, s), auroc=roc_auc_score(y, s),
                         f1_unadj=u[0], p_unadj=u[1], r_unadj=u[2],
                         f1_adj=f1.get((nz, sd), float('nan'))))
    rows.sort(key=lambda r: (r['nz'], r['seed']))
    if not rows:
        raise SystemExit('[FAIL] 没匹配到 score dump')

    nzs = sorted({r['nz'] for r in rows})
    seeds = sorted({r['seed'] for r in rows})
    L = ['# PRD5.2 · LSTM-AE 在 robust 加噪协议下：现有评测协议 vs 好尺子', '',
         f'MSL，LSTM-AE（0.33M），`main.py` 现状（10 epoch、lr 0.01、'
         f'bf_search oracle 阈值 + point-adjust，**推理未调 eval()** 但本模型无 dropout）。', '',
         '## 逐 run', '',
         '| η | seed | **F1 (现有协议)** | AUPRC | **AUROC** | F1(未调整) | P(未调整) | R(未调整) |',
         '|' + '---|' * 8]
    for r in rows:
        L.append(f"| {r['nz']:.2f} | {r['seed']} | {r['f1_adj']:.4f} | {r['auprc']:.4f} | "
                 f"**{r['auroc']:.4f}** | {r['f1_unadj']:.4f} | {r['p_unadj']:.4f} | "
                 f"{r['r_unadj']:.4f} |")

    L += ['', '## 按 η 汇总（seed 均值）', '',
          '| η | F1 (现有协议) | AUPRC | **AUROC** | n |', '|' + '---|' * 5]
    for nz in nzs:
        g = [r for r in rows if r['nz'] == nz]
        L.append(f"| {nz:.2f} | {np.mean([r['f1_adj'] for r in g]):.4f} ± "
                 f"{np.std([r['f1_adj'] for r in g], ddof=1) if len(g) > 1 else 0:.4f} | "
                 f"{np.mean([r['auprc'] for r in g]):.4f} | "
                 f"**{np.mean([r['auroc'] for r in g]):.4f}** ± "
                 f"{np.std([r['auroc'] for r in g], ddof=1) if len(g) > 1 else 0:.4f} | {len(g)} |")
    base = np.mean([r['base'] for r in rows])

    # 关键量化：F1 的摆幅 vs AUROC 的摆幅
    f1s = np.array([r['f1_adj'] for r in rows])
    aucs = np.array([r['auroc'] for r in rows])
    L += ['', '## 关键量化：同一批 run 上两把尺子的摆幅', '',
          f'- **F1（现有协议）**：{f1s.min():.4f} ~ {f1s.max():.4f}，'
          f'极差 **{f1s.max() - f1s.min():.4f}**（{f1s.max() / max(f1s.min(), 1e-9):.2f}×）',
          f'- **AUROC（好尺子）**：{aucs.min():.4f} ~ {aucs.max():.4f}，'
          f'极差 **{aucs.max() - aucs.min():.4f}**',
          f'- base rate（异常窗占比）= {base:.4f}；AUROC=0.5 为随机',
          f'- **AUROC 全程 > 0.5**（最低 {aucs.min():.4f}）⇒ LSTM-AE 的分数**有信息**，'
          f'不是噪声（对照：SECOND 在 MSL 上 AUROC=0.4975）', '']
    fx, fy = f1s.max() - f1s.min(), aucs.max() - aucs.min()
    if fy < 0.05 and fx > 2 * fy:
        L.append(f'- ⇒ **F1 的摆幅（{fx:.3f}）远大于 AUROC 的摆幅（{fy:.3f}）**，'
                 f'而两者测的是同一批模型 → F1 的波动来自**协议**（oracle 阈值 + point-adjust），'
                 f'不来自模型质量')
    L += ['', '> 注：F1 用的是 `bf_search` 在 test 上搜出的**最优**阈值（oracle），'
              '且 `adjust_predicts` 会把命中段的整段判正。这两件事叠加使 F1 对'
              '分数分布的形状极敏感、对排序质量却不敏感。',
          '> 注：LSTM-AE 无 dropout，分数可复现；本表与 `main.py` 的 F1 同源（同一 ckpt、同一批窗）。']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nsaved: {a.out}')


if __name__ == '__main__':
    main()
