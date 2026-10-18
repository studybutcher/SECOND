#!/usr/bin/env python3
"""PRD6:汇总 results/robust_ms/ 下主模型高斯噪声多 seed 结果。

产出:
  results/robust_ms/summary.csv  逐 η 的 mean/sd/n
  results/robust_ms/summary.md   人读表
  results/robust_ms/table_ms.tex 论文表(MSL 列,其余留 —)

用法:
  python script/robust/summarize_robust_ms.py [--data MSL]
"""
import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

EPS_ORDER = ["0", "0.05", "0.1", "0.15", "0.2", "0.5"]
LABEL = {"0": "-", "0.05": "5\\%", "0.1": "10\\%",
         "0.15": "15\\%", "0.2": "20\\%", "0.5": "50\\%"}


def fmt_key(x):
    """把 0/0.05/... 统一成紧凑字符串,兼容 json 里是 0.0 的情况。"""
    try:
        return "%g" % float(x)
    except (TypeError, ValueError):
        return str(x)


def msd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], None, 1
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="MSL")
    ap.add_argument("--dir", default="results/robust_ms")
    args = ap.parse_args()

    rows = defaultdict(list)
    pat = os.path.join(args.dir, f"pesr_{args.data}_gauss_noise*_seed*.json")
    files = sorted(glob.glob(pat))
    for p in files:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        cfg = d.get("config") or {}
        nz = fmt_key(cfg.get("noise", 0))
        seed = d.get("seed")
        m = d.get("metrics") or {}
        if "f1" not in m:
            print(f"[skip] {p}: no f1")
            continue
        rows[nz].append({"seed": seed, "precision": m.get("precision"),
                         "recall": m.get("recall"), "f1": m.get("f1")})

    order = [e for e in EPS_ORDER if e in rows] + \
            sorted(k for k in rows if k not in EPS_ORDER)

    # ---- csv ----
    out_csv = os.path.join(args.dir, f"summary_{args.data}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["noise", "n", "seeds", "precision_mean", "precision_sd",
                    "recall_mean", "recall_sd", "f1_mean", "f1_sd"])
        for nz in order:
            rs = rows[nz]
            p_m, p_s, n = msd([r["precision"] for r in rs])
            r_m, r_s, _ = msd([r["recall"] for r in rs])
            f_m, f_s, _ = msd([r["f1"] for r in rs])
            w.writerow([nz, n, ",".join(sorted(str(r["seed"]) for r in rs)),
                        p_m, p_s, r_m, r_s, f_m, f_s])

    # ---- markdown ----
    md = [f"# PRD6 · {args.data} · 主模型 PESR × 高斯噪声 × 多 seed\n",
          f"> 源:`{args.dir}/pesr_{args.data}_gauss_noise*_seed*.json`  "
          f"(共 {len(files)} 个)\n",
          "| $\\varepsilon$% | Precision | Recall | **F1** | n | seeds |",
          "|---|---|---|---|---|---|"]
    for nz in order:
        rs = rows[nz]
        p_m, p_s, n = msd([r["precision"] for r in rs])
        r_m, r_s, _ = msd([r["recall"] for r in rs])
        f_m, f_s, _ = msd([r["f1"] for r in rs])

        def cell(m, s):
            if m is None:
                return "—"
            return f"{m:.4f}" if s is None else f"{m:.4f} ± {s:.4f}"

        md.append(f"| {LABEL.get(nz, nz)} | {cell(p_m, p_s)} | "
                  f"{cell(r_m, r_s)} | {cell(f_m, f_s)} | {n} | "
                  f"{','.join(sorted(str(r['seed']) for r in rs))} |")
    md.append("")
    md.append("> sd 是 **噪声实现 + 训练随机性** 的合并方差"
              "(set_seed 在 add_noise 之前),见 PRD6 §1.3。")
    out_md = os.path.join(args.dir, f"summary_{args.data}.md")
    open(out_md, "w").write("\n".join(md) + "\n")

    # ---- LaTeX(MSL 列填充) ----
    def tex_cell(vals):
        m, s, n = msd(vals)
        if m is None:
            return "—"
        return (f"${m:.4f}$" if s is None
                else f"${m:.4f} \\pm {s:.4f}$")

    tex = [
        "\\begin{table*}[!h]",
        "\\caption{Model Evaluation with Replace Noise on Four Datasets "
        "(mean\\,$\\pm$\\,std over 3 seeds; MSL only in this round)}",
        "\\label{robust}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lllllllllllll}",
        "\\toprule",
        "\\multicolumn{1}{c}{\\multirow{2}{*}{$\\varepsilon$\\%}} & "
        "\\multicolumn{3}{c}{MSL}",
        "& \\multicolumn{3}{c}{SMAP} & \\multicolumn{3}{c}{SWaT} "
        "& \\multicolumn{3}{c}{WADI} \\\\",
        "\\cmidrule(r){2-4} \\cmidrule(r){5-7} \\cmidrule(r){8-10} "
        "\\cmidrule(r){11-13}",
        "\\multicolumn{1}{c}{} & Precision & Recall & F1-score",
        "& Precision & Recall & F1-score & Precision & Recall & F1-score",
        "& Precision & Recall & F1-score \\\\",
        "\\midrule",
    ]
    for nz in order:
        rs = rows[nz]
        cells = [tex_cell([r[k] for r in rs]) for k in
                 ("precision", "recall", "f1")]
        tex.append(f"{LABEL.get(nz, nz)} & " + " & ".join(cells) +
                   " & — & — & — & — & — & — & — & — & — \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    out_tex = os.path.join(args.dir, f"table_{args.data.lower()}.tex")
    open(out_tex, "w").write("\n".join(tex) + "\n")

    print("\n".join(md))
    print(f"\n-> {out_csv}\n-> {out_md}\n-> {out_tex}")


if __name__ == "__main__":
    main()
