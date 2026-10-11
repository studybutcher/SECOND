#!/usr/bin/env python
"""表示层定量分析：逐实体 normal-reference kNN + 无阈值指标（PRD5 §6 / §7 / §8 / §11）。

口径（全部预注册，不接受事后挑选）：
  * reference bank  = **逐实体**，训练窗等间隔抽 min(N, 400) 条（确定性，无 RNG）
  * 标准化          = 用该实体 bank 的逐维 mean/std；丢掉 bank 上方差≈0 的维度
  * kNN 分数        = 到 bank 的 k 近邻的平均 L2 距离（越大越异常）
  * 主指标          = repr-AUPRC / repr-AUROC（排序类，跨变体可比）
  * 辅助            = D_N, D_A, RSR=D_A/D_N, Sep_score
  * 聚合            = 实体级 macro mean（主）/ pooled（次）；σ=0 实体单独标记

用法：
    python bench/analyze_repr.py --glob 'results/repr/*_MSL_seed42_test.npz' \
        --k 10 --out results/repr/summary_seed42
"""
import argparse
import glob as globmod
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, pairwise_distances

EPS = 1e-12


def even_spaced_bank(n, target):
    """等间隔取 min(n, target) 条（确定性）。"""
    m = min(n, target)
    if m >= n:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, m)).astype(np.int64))


def knn_scores(zq, zb, k):
    """到 zb 的 k 近邻平均距离。"""
    d = pairwise_distances(zq, zb, metric='euclidean')
    kk = min(k, d.shape[1])
    nn = np.sort(d, axis=1)[:, :kk]
    return nn.mean(axis=1), d


def safe_metrics(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return average_precision_score(y, s), roc_auc_score(y, s)


def std_with_bank(ztr, zte):
    mu = ztr.mean(0)
    sd = ztr.std(0)
    keep = sd > max(1e-6, 1e-6 * sd.max())
    if keep.sum() < 2:
        return None, None, int(keep.sum())
    return (zte[:, keep] - mu[keep]) / sd[keep], (ztr[:, keep] - mu[keep]) / sd[keep], int(keep.sum())


def analyze_one(test_npz, k, label_key, bank_target, use_all_bank=False):
    base = test_npz.replace('_test.npz', '')
    train_npz = base + '_train.npz'
    if not os.path.exists(train_npz):
        raise SystemExit(f'[FAIL] 找不到配对的训练 npz: {train_npz}')
    te, tr = np.load(test_npz), np.load(train_npz)
    meta = json.load(open(base + '_test.json'))
    y_all, eid, bt = te[label_key].astype(int), te['entity_id'], tr['entity_id']
    rows = []
    for e in np.unique(eid):
        mte, mtr = eid == e, bt == e
        if mte.sum() < 50 or mtr.sum() < 50:
            continue
        y = y_all[mte]
        rec = dict(entity=meta['entity_names'][e], n_test=int(mte.sum()), n_bank=0,
                   rate=float(y.mean()), dim=None, nn_time_gap=None)
        for stage in ('zp', 'vt', 'mu'):
            ztr_raw, zte_raw = tr[stage][mtr], te[stage][mte]
            idx = np.arange(ztr_raw.shape[0]) if use_all_bank else even_spaced_bank(
                ztr_raw.shape[0], bank_target)
            zte_s, ztr_s, dim = std_with_bank(ztr_raw[idx], zte_raw)
            if zte_s is None:
                rec[f'{stage}_auprc'] = rec[f'{stage}_auroc'] = np.nan
                rec[f'{stage}_dn'] = rec[f'{stage}_da'] = np.nan
                rec[f'{stage}_sep'] = np.nan
                continue
            s, d = knn_scores(zte_s, ztr_s, k)
            ap, auc = safe_metrics(y, s)
            dn, da = (s[y == 0].mean() if (y == 0).any() else np.nan,
                      s[y == 1].mean() if (y == 1).any() else np.nan)
            sep = ((da - dn) / np.sqrt((s[y == 1].std() ** 2 + s[y == 0].std() ** 2) / 2 + EPS)
                   if (y == 0).any() and (y == 1).any() else np.nan)
            rec.update({f'{stage}_auprc': ap, f'{stage}_auroc': auc,
                        f'{stage}_dn': dn, f'{stage}_da': da, f'{stage}_sep': sep})
            if stage == 'vt':
                rec['dim'] = dim
                rec['n_bank'] = int(ztr_s.shape[0])
                # Gate 1：最近邻在时间上离查询窗多远（≤1 说明 bank 近似查询窗的拷贝）
                if not use_all_bank:
                    j = np.argmin(d, axis=1)
                    gap = np.abs(te['win_start'][mte] - tr['win_start'][mtr][idx][j])
                    rec['nn_time_gap'] = float(np.median(gap))
        # 原生分数（同一批窗口，未 point-adjust）
        s_native = te['score'][mte]
        ap0, auc0 = safe_metrics(y, s_native)
        rec.update({'score_auprc': ap0, 'score_auroc': auc0,
                    'score_dn': s_native[y == 0].mean() if (y == 0).any() else np.nan,
                    'score_da': s_native[y == 1].mean() if (y == 1).any() else np.nan})
        rows.append(rec)
    return rows, meta


def macro(rows, key):
    v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, len(v)) if len(v) else (np.nan, np.nan, 0)


def pooled(te, tr, k, label_key, bank_target):
    """pooled 口径：各实体用**自己**的 bank 标准化后拼起来（次要结果）。"""
    y_all, eid, bt = te[label_key].astype(int), te['entity_id'], tr['entity_id']
    S, Y = {s: [] for s in ('zp', 'vt', 'mu', 'score')}, []
    for e in np.unique(eid):
        mte, mtr = eid == e, bt == e
        if mte.sum() < 50 or mtr.sum() < 50:
            continue
        for stage in ('zp', 'vt', 'mu'):
            zte_s, ztr_s, _ = std_with_bank(tr[stage][mtr], te[stage][mte])
            if zte_s is None:
                continue
            s, _ = knn_scores(zte_s, ztr_s[even_spaced_bank(len(ztr_s), bank_target)], k)
            S[stage].append((e, s))
        S['score'].append((e, te['score'][mte]))
        Y.append((e, y_all[mte]))
    out = {}
    for key, lst in S.items():
        s = np.concatenate([x[1] for x in lst])
        y = np.concatenate([x[1] for x in Y])
        out[key + '_auprc'], out[key + '_auroc'] = safe_metrics(y, s)
    return out


def fmt(v, nd=4):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', nargs='+', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--bank-target', type=int, default=400)
    ap.add_argument('--bank-all', action='store_true', help='用全量训练窗作 bank（敏感性）')
    ap.add_argument('--label', default='label_any', choices=['label_any', 'label_last'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    files = sorted({f for g in a.glob for f in globmod.glob(g)})
    files = [f for f in files if f.endswith('_test.npz')]
    if not files:
        raise SystemExit('[FAIL] 没匹配到 *_test.npz')

    DEGENERATE = {'C-2', 'D-14', 'M-6', 'S-2', 'T-5'}   # MSL：col0 恒值，重构类方法原理上无法检出
    all_rows, summary = {}, []
    for f in files:
        name = os.path.basename(f).replace('_test.npz', '')
        rows, meta = analyze_one(f, a.k, a.label, a.bank_target, a.bank_all)
        all_rows[name] = rows
        deg = [r for r in rows if r['entity'] in DEGENERATE]
        keep = [r for r in rows if r['entity'] not in DEGENERATE]
        pooled_res = pooled(np.load(f), np.load(f.replace('_test.npz', '_train.npz')),
                            a.k, a.label, a.bank_target)
        summary.append(dict(name=name, backbone=meta['backbone'], n_ent=len(rows), meta=meta,
                            rows=rows, keep=keep, deg=deg, pooled=pooled_res))
        print(f'  {name}: {len(rows)} 实体，其中 σ=0 实体 {len(deg)} 个')

    # ---------------- 主表 ----------------
    lines = [f'# 表示层定量分析 —— 单 seed 汇总（k={a.k}, bank={a.bank_target}, {a.label}）',
             '',
             f'来源文件：{len(files)} 个 test npz。**主数是实体级 macro mean**（跨实体离散度），'
             f'pooled 与"不含 σ=0 实体"另列。', '']
    hdr = ('| 模型 | repr-AUPRC(V_t) | repr-AUROC(V_t) | repr-AUPRC(μ_t) | repr-AUPRC(Z_p) | '
           'D_N | D_A | Sep | **score-AUPRC(原生)** | 实体数 | base rate |')
    lines += [hdr, '|' + '---|' * 11]
    for s in summary:
        kp = s['keep'] or s['rows']
        m = {key: macro(kp, key) for key in ('vt_auprc', 'vt_auroc', 'mu_auprc', 'zp_auprc',
                                             'vt_dn', 'vt_da', 'vt_sep', 'score_auprc')}
        base = macro(kp, 'rate')[0]
        lines.append(
            f"| {s['backbone']} | {fmt(m['vt_auprc'][0])} ± {fmt(m['vt_auprc'][1],3)} "
            f"| {fmt(m['vt_auroc'][0])} | {fmt(m['mu_auprc'][0])} | {fmt(m['zp_auprc'][0])} "
            f"| {fmt(m['vt_dn'][0],3)} | {fmt(m['vt_da'][0],3)} | {fmt(m['vt_sep'][0],3)} "
            f"| **{fmt(m['score_auprc'][0])}** | {m['vt_auprc'][2]} | {fmt(base,4)} |")
    lines += ['', '> 注：`Sep` 与 `D_N/D_A` 只在同一样本内可比；主结论看 AUPRC/AUROC（排序类）。'
                  '`score-AUPRC` 是同一批窗口上**原生重构分数**的无阈值 AUPRC（未 point-adjust）。', '']

    # ---------------- pooled + 退化实体 ----------------
    lines += ['## pooled 口径（次要）', '', '| 模型 | V_t AUPRC | V_t AUROC | Z_p AUPRC | μ_t AUPRC | score AUPRC |',
              '|---|---|---|---|---|---|']
    for s in summary:
        p = s['pooled']
        lines.append(f"| {s['backbone']} | {fmt(p.get('vt_auprc'))} | {fmt(p.get('vt_auroc'))} "
                     f"| {fmt(p.get('zp_auprc'))} | {fmt(p.get('mu_auprc'))} | {fmt(p.get('score_auprc'))} |")
    lines += ['', '`σ=0 实体`（MSL: C-2, D-14, M-6, S-2, T-5，信号恒值，重构类方法原理上无法检出）'
                  '单独列，不进上面的 macro mean：', '',
              '| 模型 | 实体 | 窗数 | 异常率 | V_t AUPRC | score AUPRC |', '|---|---|---|---|---|---|']
    for s in summary:
        for r in s['deg']:
            lines.append(f"| {s['backbone']} | {r['entity']} | {r['n_test']} | {r['rate']:.4f} "
                         f"| {fmt(r['vt_auprc'])} | {fmt(r['score_auprc'])} |")
    lines += ['']

    # ---------------- 逐实体 ----------------
    lines += ['## 逐实体明细（V_t）', '',
              '| 实体 | 窗数 | 异常率 | bank | dim | NN时间间隔 | ' +
              ' | '.join(s['backbone'] for s in summary) + ' |',
              '|' + '---|' * (7 + len(summary))]
    ent_order = [r['entity'] for r in summary[0]['keep'] + summary[0]['deg']]
    for ent in ent_order:
        first = next((r for r in summary[0]['keep'] + summary[0]['deg'] if r['entity'] == ent), None)
        if first is None:
            continue
        cells = [f"{fmt(r['vt_auprc'])}" if r else '—'
                 for r in (next((x for x in (s['keep'] + s['deg']) if x['entity'] == ent), None)
                           for s in summary)]
        lines.append(f"| {ent} | {first['n_test']} | {first['rate']:.4f} | {first['n_bank']} "
                     f"| {first['dim']} | {fmt(first['nn_time_gap'],1)} | " + ' | '.join(cells) + ' |')
    lines += ['']

    # ---------------- Gate ----------------
    lines += ['## 有效性 Gate（PRD5 §11）', '']
    for s in summary:
        m0 = s['meta']
        lines += [f"### {s['backbone']}", '',
                  f"- **Gate 0** 重复前向 `max|ΔV_t|` = "
                  f"`{m0.get('repeat_check_max_abs_diff')}` → "
                  f"{'一致 ✅' if m0.get('repeat_check_ok') else '不一致 ❌'}；"
                  f"eval 模式 `{m0['eval_mode']}`；前向路径 `{m0['forward_path']}`",
                  f"- **Gate 1** bank 大小 {macro(s['keep'], 'n_bank')[0]:.0f}（min "
                  f"{min(r['n_bank'] for r in s['keep'])}）、k={a.k}；"
                  f"最近邻时间间隔中位 {fmt(macro(s['keep'], 'nn_time_gap')[0],1)} 窗"
                  f"（≤1 说明 bank 近似查询窗的拷贝）",
                  f"- **Gate 2** prevalence(any)={m0['prevalence_any']:.4f} / "
                  f"(last)={m0['prevalence_last']:.4f}；异常方向 D_A>D_N 的实体占比 "
                  f"{np.mean([r['vt_da'] > r['vt_dn'] for r in s['keep']]):.2f}", '']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    cols = ['entity', 'n_test', 'n_bank', 'rate', 'dim', 'nn_time_gap']
    for st in ('zp', 'vt', 'mu'):
        cols += [f'{st}_auprc', f'{st}_auroc', f'{st}_dn', f'{st}_da', f'{st}_sep']
    cols += ['score_auprc', 'score_auroc']
    with open(a.out + '.csv', 'w') as f:
        f.write('variant,' + ','.join(cols) + '\n')
        for name, rows in all_rows.items():
            for r in rows:
                f.write(name + ',' + ','.join(
                    '' if r.get(c) is None or (isinstance(r.get(c), float) and np.isnan(r.get(c)))
                    else (f'{r[c]:.6g}' if isinstance(r.get(c), float) else str(r[c]))
                    for c in cols) + '\n')
    with open(a.out + '.json', 'w') as f:
        json.dump({name: rows for name, rows in all_rows.items()}, f, indent=2, default=str)
    print('saved:', a.out + '.{md,csv,json}')


if __name__ == '__main__':
    main()
