#!/usr/bin/env python3
"""统一采集 results/*.json 与 nohp-out/**/*.{out,log} 中的检测指标,按
(family, variant, dataset, condition) 分组做 mean±sd。

输出:
  - runs_long.csv   每一条独立 run 一行(原始记录)
  - groups.csv      分组后的 mean/sd/n
"""
import json
import re
import csv
import os
import glob

ROOT = "/home/sdb/yrlcd/anamolybench/SECOND"
RESULTS = os.path.join(ROOT, "results")
NOHP = os.path.join(ROOT, "nohp-out")

HEADER_RE = re.compile(r"^(\w[\w .\-]*?):\s*(.*)$")
# 无冒号的头行,如 `window_length 35` / `latent_size 256`
HEADER_NC_RE = re.compile(r"^(window_length|latent_size|batch)\s+(\S+)\s*$", re.M)
METRIC_RE = re.compile(
    r"precision:\s*([\d.]+)\s+recall:\s*([\d.]+)\s+f_score\s+([\d.]+)"
)
CONTAM_RE = re.compile(r"contam:\s*rho=([\d.]+)")
NOISE_HDR_RE = re.compile(r"^noise:\s*([\d.]+)\s*$", re.M)
NOISE_ALT_RE = re.compile(r"^noise\s+([\d.]+)\s*$", re.M)

runs = []


def add(**kw):
    kw.setdefault("source", "")
    kw.setdefault("prio", 1)
    runs.append(kw)


def norm_seed(s):
    """统一 seed 表示,避免 int 42 与 str '42' 被当成两个不同 run。"""
    if s is None:
        return "?"
    try:
        return str(int(float(s)))
    except (TypeError, ValueError):
        return str(s)


def norm_cond(cond, wl=None):
    """统一 condition 标签。

    各数据集的常规 window 是 MSL/SMAP/WADI=70、SWaT=35;只有 W=140 的
    "加长窗"实验才需要区分,否则同一个 run 的 json 记录(带 window_length)
    与 .out 记录(头行丢了)会因后缀不一致而拆成两条。
    """
    c = cond
    if wl is not None:
        try:
            if int(float(wl)) == 140:
                c = f"{c}|W140"
        except (TypeError, ValueError):
            pass
    return c


# ---------- 1. results/*.json ----------
for path in sorted(glob.glob(os.path.join(RESULTS, "**", "*.json"), recursive=True)):
    rel = os.path.relpath(path, RESULTS)
    family = rel.split(os.sep)[0]
    if family in ("repr", "bench_raw", "threshold", "evalcheck"):
        continue  # 表示层/成本/阈值,不是 P-R-F1 检测结果
    try:
        d = json.load(open(path))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    m = d.get("metrics") or {}
    if not m or "f1" not in m:
        continue
    cfg = d.get("config") or {}
    noise = cfg.get("noise")
    wl = cfg.get("window_length")
    contam = d.get("contam") or {}
    contam_test = d.get("contam_test") or {}
    rho = contam.get("rho", contam_test.get("rho"))
    side = "test" if contam_test and not contam else "train"
    cond_raw = (
        f"noise{noise}" if (noise or 0) > 0
        else (f"{side}-rho{rho}" if rho else "clean")
    )
    add(
        family=family,
        variant=d.get("backbone") or "SECOND",
        dataset=d.get("dataset"),
        condition_kind="noise" if (noise or 0) > 0 else ("contam" if rho else "clean"),
        condition=norm_cond(cond_raw, wl),
        seed=norm_seed(d.get("seed")),
        precision=m.get("precision"),
        recall=m.get("recall"),
        f1=m.get("f1"),
        n_params_M=(d.get("params") or {}).get("n_params_M"),
        source=rel,
        prio=0,
    )

# ---------- 2. nohp-out/**/*.{out,log} ----------
for path in sorted(
    glob.glob(os.path.join(NOHP, "**", "*.out"), recursive=True)
    + glob.glob(os.path.join(NOHP, "**", "*.log"), recursive=True)
):
    rel = os.path.relpath(path, NOHP)
    if "launcher" in rel or "progress" in rel or "cache_" in rel:
        continue
    try:
        txt = open(path, errors="replace").read()
    except Exception:
        continue
    if not METRIC_RE.search(txt):
        continue
    # 每个 run 以 ====... 分隔
    for block in txt.split("=" * 90):
        mm = METRIC_RE.search(block)
        if not mm:
            continue
        hdr = {}
        for line in block.splitlines():
            hm = HEADER_RE.match(line.strip())
            if hm and hm.group(1) not in hdr:
                hdr[hm.group(1)] = hm.group(2).strip()
        for k, v in HEADER_NC_RE.findall(block):
            hdr.setdefault(k, v)
        name = hdr.get("name", "")
        fam = name.split("/")[0] if "/" in name else rel.split(os.sep)[0]
        variant = hdr.get("backbone") or name.split("/")[-1] or "SECOND"
        noise = hdr.get("noise")
        if noise is None:
            n2 = re.search(r"^noise\s+([\d.]+)\s*$", block, re.M)
            noise = n2.group(1) if n2 else None
        cm = CONTAM_RE.search(block)
        rho = cm.group(1) if cm else None
        wl = hdr.get("window_length", "")
        cond = (
            f"noise{noise}" if noise and float(noise) > 0
            else (f"train-rho{rho}" if rho and float(rho) > 0 else "clean")
        )
        add(
            family=fam or rel.split(os.sep)[0],
            variant=variant,
            dataset=hdr.get("dataset", "?"),
            condition_kind="noise" if (noise and float(noise) > 0) else (
                "contam" if rho and float(rho) > 0 else "clean"),
            condition=norm_cond(cond, wl),
            seed=norm_seed(hdr.get("seed")),
            precision=float(mm.group(1)),
            recall=float(mm.group(2)),
            f1=float(mm.group(3)),
            n_params_M=None,
            source=rel,
            prio=1,
        )

# ---------- 去重 ----------
# 同一个 (family, variant, dataset, condition, seed, window) 视为同一次 run;
# json 记录(prio=0,信息更全)优先于 .out/.log 记录(prio=1)。
best = {}
for r in runs:
    key = (r["family"], r["variant"], r["dataset"], r["condition"], r["seed"])
    if key not in best or r["prio"] < best[key]["prio"]:
        best[key] = r
uniq = sorted(best.values(),
              key=lambda r: (r["family"], r["variant"], r["dataset"], r["condition"], r["seed"]))

outdir = os.path.join(ROOT, "results", "_summary")
os.makedirs(outdir, exist_ok=True)
fields = ["family", "variant", "dataset", "condition_kind", "condition",
          "seed", "precision", "recall", "f1", "n_params_M", "source", "prio"]
with open(os.path.join(outdir, "runs_long.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(uniq)

print(f"collected {len(uniq)} unique runs "
      f"(from {len(runs)} raw records) -> results/_summary/runs_long.csv\n")

# ---------- 分组 ----------
from collections import defaultdict
import statistics

groups = defaultdict(list)
for r in uniq:
    groups[(r["family"], r["variant"], r["dataset"],
            r["condition_kind"], r["condition"])].append(r)


def fmt(vals):
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}"


rows = []
for k, rs in sorted(groups.items()):
    rows.append({
        "family": k[0], "variant": k[1], "dataset": k[2],
        "condition_kind": k[3], "condition": k[4],
        "n": len(rs),
        "seeds": ",".join(sorted(str(r["seed"]) for r in rs)),
        "precision": fmt([r["precision"] for r in rs if r["precision"] is not None]),
        "recall": fmt([r["recall"] for r in rs if r["recall"] is not None]),
        "f1": fmt([r["f1"] for r in rs if r["f1"] is not None]),
    })

with open(os.path.join(outdir, "groups.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "variant", "dataset",
                                      "condition_kind", "condition", "n", "seeds",
                                      "precision", "recall", "f1"])
    w.writeheader()
    w.writerows(rows)

print(f"{len(rows)} groups -> results/_summary/groups.csv\n")

# ---------- 终端打印 ----------
for fam in sorted(set(r["family"] for r in rows)):
    sub = [r for r in rows if r["family"] == fam]
    nmax = max(r["n"] for r in sub)
    print(f"\n########## {fam}  (最多 n={nmax}) ##########")
    print(f"{'variant':<14}{'dataset':<6}{'cond':<14}{'n':>3} {'seeds':<16}"
          f"{'precision':<20}{'recall':<20}{'F1':<20}")
    for r in sub:
        print(f"{r['variant']:<14}{r['dataset']:<6}{r['condition']:<14}"
              f"{r['n']:>3} {r['seeds']:<16}"
              f"{r['precision']:<20}{r['recall']:<20}{r['f1']:<20}")
