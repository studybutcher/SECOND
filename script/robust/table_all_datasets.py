#!/usr/bin/env python3
"""PRD6:把四个数据集的高斯噪声多 seed 结果合成一张论文表。

与 `summarize_robust_ms.py` 的区别:那个是**单数据集**汇总(md/csv);
这个只产出**跨数据集合并**的 LaTeX 表(论文 robust 表格式)。
缺失的 (数据集, η, seed) 组合显示 `—`,并在表下用注释计数。

用法:
  python script/robust/table_all_datasets.py
  python script/robust/table_all_datasets.py --datasets MSL SMAP
"""
import argparse
import glob
import json
import os
import re
import statistics

EPS = ["0", "0.05", "0.1", "0.15", "0.2", "0.5"]
LABEL = {"0": "-", "0.05": "5\\%", "0.1": "10\\%",
         "0.15": "15\\%", "0.2": "20\\%", "0.5": "50\\%"}


def key(x):
    try:
        return "%g" % float(x)
    except (TypeError, ValueError):
        return str(x)


def collect(ds, res_dir):
    out = {}
    for p in glob.glob(os.path.join(res_dir,
                                    f"pesr_{ds}_gauss_noise*_seed*.json")):
        d = json.load(open(p))
        m = re.search(r"noise([\d.]+)_seed(\d+)", p)
        if not m:
            continue
        nz, seed = key(m.group(1)), int(m.group(2))
        met = d.get("metrics") or {}
        if "f1" not in met:
            continue
        out.setdefault(nz, {})[seed] = met
    return out


def cell(mets, k):
    vals = [m[k] for m in mets.values() if m.get(k) is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"${vals[0]:.4f}$"
    return f"${statistics.mean(vals):.4f} \\pm {statistics.stdev(vals):.4f}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["MSL", "SMAP", "SWaT", "WADI"])
    ap.add_argument("--dir", default="results/robust_ms")
    ap.add_argument("--out", default="results/robust_ms/table_all.tex")
    a = ap.parse_args()

    data = {ds: collect(ds, a.dir) for ds in a.datasets}
    ncol = 1 + 3 * len(a.datasets)
    colspec = "l" + "l" * (ncol - 1)

    L = ["\\begin{table*}[!h]",
         "\\caption{Model Evaluation with Replace Noise on Four Datasets "
         "(PESR, mean\\,$\\pm$\\,std over 3 seeds; Gaussian $N(0,1)$ "
         "whole-row replacement)}",
         "\\label{robust}",
         "\\resizebox{\\textwidth}{!}{",
         f"\\begin{{tabular}}{{{colspec}}}",
         "\\toprule",
         "\\multicolumn{1}{c}{\\multirow{2}{*}{$\\varepsilon$\\%}} & " +
         " & ".join(f"\\multicolumn{{3}}{{c}}{{{ds}}}"
                    for ds in a.datasets) + " \\\\"]
    cmids = " ".join(f"\\cmidrule(r){{{2 + 3 * i}-{4 + 3 * i}}}"
                     for i in range(len(a.datasets)))
    L.append(cmids)
    L.append("\\multicolumn{1}{c}{} & " +
             " & ".join(["Precision & Recall & F1-score"] *
                        len(a.datasets)) + " \\\\")
    L.append("\\midrule")

    missing = []
    for nz in EPS:
        cells = []
        for ds in a.datasets:
            mets = data[ds].get(nz, {})
            cells.append(cell(mets, "precision"))
            cells.append(cell(mets, "recall"))
            cells.append(cell(mets, "f1"))
            if len(mets) < 3:
                missing.append(f"{ds} η={nz}: {len(mets)}/3")
        L.append(f"{LABEL[nz]} & " + " & ".join(cells) + " \\\\")

    L += ["\\bottomrule", "\\end{tabular}}"]
    if missing:
        L.append("\\par\\smallskip\\footnotesize Missing seeds: " +
                 "; ".join(missing))
    L.append("\\end{table*}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
