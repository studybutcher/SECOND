"""把 results/bench_raw/*.json 汇总成表格

产出：
    results/bench_raw/summary.md   —— 可直接贴论文/文档
    results/bench_raw/summary.csv  —— 便于二次处理（注意 .gitignore 含 *.csv）

用法：
    python bench/summarize_cost.py
"""
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, 'results', 'bench_raw')
DATASETS = ['SMAP', 'MSL', 'SWaT', 'WADI']
MODELS = ['SECOND', 'MUTANT']


def load_records():
    recs = {}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, '*.json'))):
        with open(path) as f:
            r = json.load(f)
        if 'model' not in r or 'cost' not in r:
            continue                       # 跳过 summary 之类的文件
        recs[(r['model'], r['dataset'])] = r
    return recs


def params_at_opbatch(rec):
    """MUTANT 的 att_encoder.h_n/s_n 是 (1, batch, hidden) 的 Parameter（Model.py:82-83），
    参数量随 batch 变。部署 batch 是 config.batch_size，故按它折算：
        n_params(b) = n_params(b=1) - 2*hidden*1 + 2*hidden*b
    SECOND 无 batch 耦合，返回值与 n_params 相同。"""
    c, cfg = rec['cost'], rec['config']
    h, opb = cfg.get('hidden_size'), cfg.get('batch_size')
    if h is None or opb is None:
        return c['n_params']
    return c['n_params'] - 2 * h + 2 * h * opb


def fmt(x, n=3):
    if x is None:
        return '-'
    if isinstance(x, float):
        if x != 0 and abs(x) < 1e-3:
            return f'{x:.3e}'
        return f'{x:.{n}f}'
    return str(x)


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |',
           '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def main():
    recs = load_records()
    if not recs:
        print('没有找到原始记录，请先跑 collect_cost_second.py / collect_cost_mutant.py')
        return
    md = ['# 推理成本原始数据汇总', '',
          f'共 {len(recs)} 条记录，来自 `results/bench_raw/*.json`。', '']

    # ---------- 一致性检查：跨库 batch 采样点必须逐点相同 ----------
    warns, missing = [], []
    for d in DATASETS:
        a, b = recs.get(('SECOND', d)), recs.get(('MUTANT', d))
        if not a or not b:
            lack = ' + '.join(n for n, r in (('SECOND', a), ('MUTANT', b)) if not r)
            missing.append(f'{d}（缺 {lack} 记录）')
            continue
        for variant in ['as_shipped', 'core_only']:
            sa = {k for k, v in a['latency'].get(variant, {}).items() if 'mean' in v}
            sb = {k for k, v in b['latency'].get(variant, {}).items() if 'mean' in v}
            if sa != sb:
                key = lambda s: sorted(int(x) for x in s)
                warns.append(f'{d} / {variant}: batch 采样点不一致 — '
                             f'SECOND={key(sa)} MUTANT={key(sb)}')
        ma = {k for k, v in a.get('memory', {}).items() if 'inference_peak_MB' in v}
        mb_ = {k for k, v in b.get('memory', {}).items() if 'inference_peak_MB' in v}
        if ma != mb_:
            key = lambda s: sorted(int(x) for x in s)
            warns.append(f'{d} / memory: batch 采样点不一致 — '
                         f'SECOND={key(ma)} MUTANT={key(mb_)}')
    if warns:
        md += ['## ⚠️ 一致性警告（跨库 batch 采样点不一致，曲线不可直接对比）', ''] + \
              [f'- {w}' for w in warns] + ['']
    elif missing:
        md += ['## ⚠️ 一致性未能校验（两库记录未配对，不得据此下结论）', ''] + \
              [f'- 缺少: {m}' for m in missing] + ['']
    else:
        md += ['> ✅ 跨库 batch 采样点逐点一致。', '']

    # ---------- 表 1：计算成本 ----------
    md += ['## 表 1  计算成本（batch=1 单窗口前向，FP32）', '']
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = recs.get((m, d))
            if not r:
                continue
            c = r['cost']
            cfg = r['config']
            p_op = params_at_opbatch(r)
            rows.append([m, d, f"{cfg['input_dim']}/{cfg['window_length']}",
                         fmt(p_op / 1e6, 4), fmt(c['weight_mem_MB'], 2),
                         fmt(c.get('flops_per_window_G'), 6),
                         fmt(c.get('flops_per_timestep_G'), 6),
                         fmt(c.get('midstack_flops_share'), 4)])
    md += [table(['模型', '数据集', 'C/W', 'params(M)<sup>†</sup>', '权重(MB)',
                  'GFLOPs/窗口', 'GFLOPs/时间步', '中间块占比'], rows),
           '',
           '> <sup>†</sup> params 按各自部署 batch 折算。MUTANT 的 `att_encoder.h_n/s_n` 是'
           ' `(1,batch,hidden)` 的 Parameter（`Model.py:82-83`），参数量随 batch 线性增长，'
           '故按训练/部署 batch（120）计；SECOND 的 `self.batch_size` 在 forward 中未被使用'
           '（用的是运行时 `len(x)`），参数量与 batch 无关。',
           '']

    # ---------- 表 2：延时 ----------
    md += ['## 表 2  推理延时（ms/窗口，端到端推理路径）', '']
    for variant in ['as_shipped', 'core_only']:
        md += [f'### {variant}', '']
        batches = sorted({int(b) for m in MODELS for d in DATASETS
                          if (m, d) in recs and variant in recs[(m, d)]['latency']
                          for b in recs[(m, d)]['latency'][variant]})
        rows = []
        for m in MODELS:
            for d in DATASETS:
                r = recs.get((m, d))
                if not r or variant not in r['latency']:
                    continue
                lat = r['latency'][variant]
                row = [m, d]
                for b in batches:
                    s = lat.get(str(b)) or {}
                    row.append(fmt(s.get('ms_per_window'), 3))
                rows.append(row)
        md += [table(['模型', '数据集'] + [f'b={b}' for b in batches], rows), '']

    # ---------- 表 3：显存 ----------
    md += ['## 表 3  峰值显存（MB）', '']
    batches = sorted({int(b) for m in MODELS for d in DATASETS
                      if (m, d) in recs for b in recs[(m, d)].get('memory', {})})
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = recs.get((m, d))
            if not r:
                continue
            row = [m, d]
            for b in batches:
                s = r.get('memory', {}).get(str(b)) or {}
                row.append(fmt(s.get('inference_peak_MB'), 1))
            rows.append(row)
    md += [table(['模型', '数据集'] + [f'b={b}' for b in batches], rows), '']

    # ---------- 表 4：跨库比值 ----------
    md += ['## 表 4  跨库比值（SECOND / MUTANT）', '',
           '> 延时比 < 1 表示 SECOND 更快（原文标注 **粗体**）。', '']
    rows = []
    for d in DATASETS:
        a, b = recs.get(('SECOND', d)), recs.get(('MUTANT', d))
        if not a or not b:
            continue

        def ratio(x, y):
            if not y or x is None:
                return '-'
            v = x / y
            s = f'{v:.2f}x'
            return f'**{s}**' if v < 1 else s

        def lat(rec, variant, B):
            return rec['latency'][variant].get(str(B), {}).get('ms_per_window')

        rows.append([d,
                     ratio(params_at_opbatch(a), params_at_opbatch(b)),
                     ratio(a['cost'].get('flops_per_window', 0), b['cost'].get('flops_per_window', 0)),
                     ratio(a['cost']['weight_mem_MB'], b['cost']['weight_mem_MB']),
                     ratio(lat(a, 'as_shipped', 1), lat(b, 'as_shipped', 1)),
                     ratio(lat(a, 'as_shipped', 120), lat(b, 'as_shipped', 120)),
                     ratio(lat(a, 'core_only', 120), lat(b, 'core_only', 120))])
    md += [table(['数据集', '参数量比', 'FLOPs 比', '权重显存比',
                  '延时比 b=1', '延时比 b=120<br>(as_shipped)', '延时比 b=120<br>(core_only)'], rows), '']

    # ---------- 表 5：计数器覆盖 ----------
    md += ['## 表 5  计数器覆盖声明', '']
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = recs.get((m, d))
            if not r:
                continue
            rows.append([m, d, ', '.join(r['cost'].get('counter_ops_seen', [])),
                         ', '.join(r['cost'].get('counter_ops_missing', []))])
    md += [table(['模型', '数据集', '已计入', '未计入'], rows), '']

    # ---------- 表 6：测量元数据 ----------
    md += ['## 表 6  测量元数据', '']
    rows = []
    for m in MODELS:
        for d in DATASETS:
            r = recs.get((m, d))
            if not r:
                continue
            p, e = r['protocol'], r['env']
            ck = r.get('ckpt', {})
            rows.append([m, d, e['gpu'].replace('NVIDIA ', ''), e['torch'],
                         p.get('torch_num_threads'), p.get('omp_num_threads'),
                         ck.get('weights_loaded'), (ck.get('sha256') or '-')[:12]])
    md += [table(['模型', '数据集', 'GPU', 'torch', 'torch线程', 'OMP',
                  '载入权重', 'ckpt sha256'], rows), '']

    out_md = os.path.join(RAW_DIR, 'summary.md')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('written:', out_md)

    # csv
    lines = ['model,dataset,C,W,n_params_M,weight_mem_MB,flops_per_window_G,'
             'flops_per_timestep_G,midstack_share,ms_per_window_b1_as_shipped,'
             'ms_per_window_opb_as_shipped,peak_mem_MB_b1']
    for m in MODELS:
        for d in DATASETS:
            r = recs.get((m, d))
            if not r:
                continue
            cfg = r['config']
            opb = str(cfg.get('batch_size'))
            lines.append(','.join(str(x) for x in [
                m, d, cfg['input_dim'], cfg['window_length'],
                r['cost']['n_params_M'], round(r['cost']['weight_mem_MB'], 3),
                r['cost'].get('flops_per_window_G'), r['cost'].get('flops_per_timestep_G'),
                round(r['cost'].get('midstack_flops_share', 0), 4),
                round(r['latency']['as_shipped'].get('1', {}).get('ms_per_window', 0), 4),
                round(r['latency']['as_shipped'].get(opb, {}).get('ms_per_window', 0), 4),
                round(r.get('memory', {}).get('1', {}).get('inference_peak_MB', 0), 1),
            ]))
    out_csv = os.path.join(RAW_DIR, 'summary.csv')
    with open(out_csv, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('written:', out_csv)


if __name__ == '__main__':
    main()
