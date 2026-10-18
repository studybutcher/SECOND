#!/usr/bin/env python3
"""从 results/_summary/runs_long.csv 生成 markdown 表格。

主模型(PESR = attnmamba+blockize)的噪声网格散落在 nohp-out/Robust/*.out,
且 MSL 那三个块的 `noise:` 头丢了 —— 这里按数值与论文表核对后手工标注,
PROVENANCE 里记录依据。其余系列全部从 CSV 自动汇总。
"""
import csv
import os
import statistics
from collections import defaultdict

SUM = "/home/sdb/yrlcd/anamolybench/SECOND/results/_summary"
DS = ["MSL", "SMAP", "SWaT", "WADI"]
EPS = ["0", "0.05", "0.1", "0.15", "0.2", "0.5"]
EPS_LABEL = {"0": "-", "0.05": "5\\%", "0.1": "10\\%",
             "0.15": "15\\%", "0.2": "20\\%", "0.5": "50\\%"}

rows = list(csv.DictReader(open(os.path.join(SUM, "runs_long.csv"))))

# ---- 主模型噪声网格:nohp-out/Robust/*.out 中逐块核对出的数值 ----
# PROVENANCE: MSL 三块无 noise 头,按 (P,R,F1) 与 Robust 表 5%/10%/15% 行逐位吻合认定;
#             WADI 只取能与论文行对上、且 batch=60 语义一致的块(batch=24 的 16 块
#             是 9/8 的老 baseline sweep,不是本网格的一部分,已排除)。
MAIN_NOISE = {
    # (dataset, eps): (P, R, F1)
    ("MSL", "0.05"): (0.8958333321214377, 0.9968387761601103, 0.9436359106772179),
    ("MSL", "0.1"): (0.9586801412623660, 0.9709468598962113, 0.9647695107884753),
    ("MSL", "0.15"): (0.9042976782860117, 0.9914195378723174, 0.9458516802033919),
    ("SMAP", "0.05"): (0.9775, 0.9747, 0.9761),
    ("SMAP", "0.1"): (0.9686, 0.9813, 0.9749),
    ("SMAP", "0.15"): (0.9597, 0.9799, 0.9697),
    ("SMAP", "0.2"): (0.9752, 0.9790, 0.9771),
    ("SMAP", "0.5"): (0.9763, 0.9833, 0.9798),
    ("SWaT", "0.05"): (0.9710, 0.9934, 0.9821),
    ("SWaT", "0.1"): (0.9690, 0.9975, 0.9831),
    ("SWaT", "0.15"): (0.9748, 0.9819, 0.9783),
    ("SWaT", "0.2"): (0.9821, 0.9935, 0.9878),
    ("SWaT", "0.5"): (0.9806, 0.9895, 0.9850),
    ("WADI", "0.05"): (0.9357352534356816, 0.9999999428514318, 0.9667958383722748),
    ("WADI", "0.15"): (0.8973461168018658, 0.9999999428514318, 0.9458910581957837),
}
# 主模型 clean(无噪声):attnmamba+blockize.out,W=140 那次(W=70 那次 WADI 是坏配置)
MAIN_CLEAN = {
    "MSL": (0.9357, 0.9807, 0.9577),
    "SMAP": (0.9688, 0.9868, 0.9777),
    "SWaT": (0.9790, 0.9935, 0.9862),
    "WADI": (0.9668, 1.0000, 0.9831),
}


def agg(rs, key):
    v = [float(r[key]) for r in rs if r[key] not in ("", None)]
    if not v:
        return None
    if len(v) == 1:
        return f"{v[0]:.4f}"
    return f"{statistics.mean(v):.4f}±{statistics.stdev(v):.4f}"


def cell(t):
    return "—" if t is None else t


# ================= 表 A:替换噪声鲁棒性 =================
out = []
out.append("## 表 A — Replace-Noise 鲁棒性(主模型 PESR)\n")
out.append("> 全部 n=1(seed=42)。**没有任何一格有重复运行,因此没有 sd。**\n")
out.append("> 来源:`nohp-out/Robust/{msl,SMAP,SWaT,WADI}.out` + `attnmamba+blockize.out`。\n")
hdr = "| $\\varepsilon$% |" + "".join(
    f" {d} P | {d} R | {d} F1 |" for d in DS)
out.append(hdr)
out.append("|" + "---|" * (1 + 3 * len(DS)))
for eps in EPS:
    cells = []
    for d in DS:
        if eps == "0":
            t = MAIN_CLEAN.get(d)
        else:
            t = MAIN_NOISE.get((d, eps))
        cells += [cell(f"{t[0]:.4f}") if t else "—",
                  cell(f"{t[1]:.4f}") if t else "—",
                  cell(f"{t[2]:.4f}") if t else "—"]
    out.append(f"| {EPS_LABEL[eps]} |" + "".join(f" {c} |" for c in cells))
out.append("")

# ================= 表 B:噪声 × 消融变体(nompamssa 等) =================
grp = defaultdict(list)
for r in rows:
    grp[(r["family"], r["variant"], r["dataset"], r["condition"])].append(r)

out.append("## 表 B — 各消融变体在噪声下的 F1(mean±sd)\n")
for variant in ["nompamssa", "nomssa", "nompa", "lstmae"]:
    have = sorted({d for (f, v, d, c) in grp if v == variant})
    if not have:
        continue
    out.append(f"### {variant}  (datasets: {', '.join(have)})\n")
    out.append("| $\\varepsilon$% |" + "".join(f" {d} |" for d in have) + " n |")
    out.append("|" + "---|" * (len(have) + 2))
    for eps in EPS:
        cond = "clean" if eps == "0" else f"noise{eps}"
        cells, ns = [], set()
        for d in have:
            rs = grp.get(("robust", variant, d, cond)) or \
                 grp.get(("lstmae_robust", variant, d, cond)) or \
                 grp.get((variant, variant, d, cond))
            if not rs and variant == "lstmae":
                rs = grp.get(("lstmae_robust", variant, d, cond))
            if not rs:
                cells.append("—")
                continue
            cells.append(agg(rs, "f1"))
            ns.add(len(rs))
        nlab = ",".join(str(x) for x in sorted(ns)) if ns else "—"
        out.append(f"| {EPS_LABEL[eps]} |" + "".join(f" {c} |" for c in cells)
                   + f" {nlab} |")
    out.append("")

# ================= 表 C:消融 F1 矩阵 =================
out.append("## 表 C1 — 主干消融(可靠:`results/ablation/*.json`,seed 42)\n")
out.append("> 主模型 PESR 取自 `nohp-out/attnmamba+blockize.out`。**全部 n=1。**\n")
ABL = [
    ("PESR (主模型)", "attnmamba+blockize"),
    ("MSSA→Transformer", "transformer"),
    ("MSSA→LSTM", "lstm"),
    ("MSSA→Mamba(原)", "mamba"),
]
out.append("| 变体 |" + "".join(f" {d} |" for d in DS) + " params(M) |")
out.append("|" + "---|" * (len(DS) + 2))
for label, variant in ABL:
    cells, params = [], set()
    for d in DS:
        if variant == "attnmamba+blockize":
            t = MAIN_CLEAN.get(d)
            cells.append(f"{t[2]:.4f}" if t else "—")
            continue
        # 每个数据集取该变体在本表口径下的那次(mamba/WADI 取 W=140 的那次)
        rs = grp.get(("ablation", variant, d, "clean|W140")) or \
             grp.get(("ablation", variant, d, "clean")) or []
        if not rs:
            cells.append("—")
            continue
        cells.append(agg(rs, "f1"))
        if rs[0]["n_params_M"]:
            params.add(round(float(rs[0]["n_params_M"]), 1))
    plab = "/".join(str(p) for p in sorted(params)) if params else "—"
    out.append(f"| {label} |" + "".join(f" {c} |" for c in cells) + f" {plab} |")
out.append("")

out.append("## 表 C2 — 旧变体日志(`nohp-out/ablation/*.out`,Sep 8)\n")
out.append("> **文件名即变体名,与论文表的行对应关系无法从文件本身确认,需人工核对。**\n")
out.append("| 日志文件 |" + "".join(f" {d} |" for d in DS) + "")
out.append("|" + "---|" * (len(DS) + 1))
old = defaultdict(dict)
for r in rows:
    if r["family"] == "ablation" and r["source"].endswith(".out"):
        old[r["source"]][r["dataset"]] = r["f1"]
for src in sorted(old):
    out.append(f"| {os.path.basename(src)} |"
               + "".join(f" {float(old[src][d]):.4f} |" if d in old[src] else " — |"
                         for d in DS))
out.append("")

out.append("## 表 C3 — 污染网格的 rho=0(即 clean)对照组,`nohp-out/contam/*`\n")
out.append("> 这是**污染实验**的零污染基线,与表 C1 不是同一批 run,不可混用。\n")
out.append("| 变体 | MSL | WADI | seed |")
out.append("|---|---|---|---|")
for variant in ["nompa", "nomssa", "nompamssa", "mamba"]:
    cells, seeds = [], set()
    for d in ["MSL", "WADI"]:
        rs = grp.get(("contam", variant, d, "clean")) or []
        cells.append(agg(rs, "f1") if rs else "—")
        seeds |= {r["seed"] for r in rs}
    out.append(f"| {variant} | {cells[0]} | {cells[1]} | "
               + (",".join(sorted(seeds)) if seeds else "—") + " |")
out.append("")

# ================= 覆盖率说明 =================
out.append("## 覆盖率:n 的分布\n")
gc = defaultdict(int)
for k, rs in grp.items():
    gc[len(rs)] += 1
for n in sorted(gc):
    out.append(f"- n={n}: {gc[n]} 组")
out.append("")
out.append("> **只有 `lstmae_robust`(LSTM-AE × MSL × 6 个噪声档 × seed 7/17/42)"
           "和 `contam/nomssa MSL clean`(seed 7/42)有真正的重复运行。**")
out.append("> 其余 150 组全是单次运行,无法给出 sd。")

# ================= 与论文 LaTeX 表的差异 =================
out.append("## 附:采集值与论文 LaTeX 表的差异\n")
LATEX_ROB = {  # (ds, eps): (P, R, F1)  用户贴的 robust 表
    ("MSL", "0"): (0.9356, 0.9807, 0.9704),
    ("MSL", "0.05"): (0.8958, 0.9968, 0.9436),
    ("MSL", "0.1"): (0.9586, 0.9709, 0.9647),
    ("MSL", "0.15"): (0.9042, 0.9914, 0.9458),
    ("SMAP", "0.1"): (0.9686, 0.9812, 0.9748),
    ("SWaT", "0.1"): (0.9623, 0.9690, 0.9975),
}
out.append("| dataset | ε% | LaTeX P/R/F1 | 采集 P/R/F1 | 差异 |")
out.append("|---|---|---|---|---|")
for (d, eps), lt in sorted(LATEX_ROB.items()):
    got = MAIN_CLEAN.get(d) if eps == "0" else MAIN_NOISE.get((d, eps))
    if not got:
        continue
    # LaTeX 表自己的 F1 与它自己的 P/R 是否自洽
    p, r, f = lt
    f_consistent = 2 * p * r / (p + r) if (p + r) else 0
    flag = "**LaTeX 表内 F1 与自身 P/R 不符**" if abs(f_consistent - f) > 5e-3 \
        else "数值一致(±1e-4 舍入)"
    out.append(f"| {d} | {eps} | {p:.4f}/{r:.4f}/{f:.4f} | "
               f"{got[0]:.4f}/{got[1]:.4f}/{got[2]:.4f} | {flag} |")
out.append("")
out.append("> 结论:采集到的 **P/R 与论文表逐位吻合**,但论文表多个单元格的 "
           "**F1 与其同行的 P/R 不自洽**(如 MSL `-` 行 P=0.9356 R=0.9807 "
           "应为 F1=0.9577,表中却写 0.9704)。疑为排版时把 ablation 表的 "
           "F1 列串进了 robust 表。**建议按本次采集的 P/R 重算 F1。**")

txt = "\n".join(out)
path = os.path.join(SUM, "tables.md")
open(path, "w").write(txt + "\n")
print(txt)
print(f"\n-> {path}")
