#!/usr/bin/env python
# coding=utf-8
"""PRD6.2：跑现有 ckpt 的 POT ③ 标定（不重训），支持按 shard 分卡。

用法：
  CUDA_VISIBLE_DEVICES=7 python script/robust/pot_cal_no_retrain.py \
      --model lstmae --shard 0 --nshards 3

每个 run 起一个独立子进程（避免 GPU 显存累积），日志写
nohp-out/pot_cal/{model}_{eta}_seed{seed}.log。
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ETA_TAG = {0.0: '0', 0.05: '0.05', 0.10: '0.10',
           0.15: '0.15', 0.2: '0.2', 0.5: '0.5'}
NOSES = [0.0, 0.05, 0.10, 0.15, 0.2, 0.5]
SEEDS = [42, 7, 17]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['lstmae', 'pesr'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--noises', default='0,0.05,0.1,0.15,0.2,0.5',
                    help='逗号分隔的噪声水平，如 "0" 只跑干净数据')
    ap.add_argument('--seeds', default='42,7,17')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--log-dir', default='nohp-out/pot_cal')
    ap.add_argument('--skip-existing', action='store_true',
                    help='已有完整 JSON（含 bf_free 与 scale=1.0）则跳过')
    args = ap.parse_args()

    out_dir = args.out_dir or (
        'results/threshold_lstmae_potcal' if args.model == 'lstmae'
        else 'results/threshold_pesr_potcal')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, args.log_dir), exist_ok=True)

    noises = [float(x) for x in args.noises.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    for n in noises:
        if n not in ETA_TAG:
            raise SystemExit(f'[FAIL] 未知噪声水平: {n}')
    runs = [(n, s) for n in noises for s in seeds]
    mine = [r for i, r in enumerate(runs) if i % args.nshards == args.shard]
    print(f'[{args.model}] shard {args.shard}/{args.nshards}: {len(mine)} runs',
          flush=True)

    def is_complete(eta, seed):
        p = os.path.join(ROOT, out_dir,
                         f'{args.model}_MSL_noise{ETA_TAG[eta]}_seed{seed}.json')
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                j = json.load(f)
            return ('bf_free' in j and
                    any(abs(r['scale'] - 1.0) < 1e-12 for r in j['scales']))
        except Exception:
            return False

    fail = 0
    for eta, seed in mine:
        tag = ETA_TAG[eta]
        if args.skip_existing and is_complete(eta, seed):
            print(f'[skip] {args.model} eta={eta} seed={seed}', flush=True)
            continue
        log = os.path.join(ROOT, args.log_dir,
                           f'{args.model}_noise{tag}_seed{seed}.log')
        cmd = [sys.executable, os.path.join(ROOT, 'bench/pot_scale_calibrate.py'),
               '--model', args.model, '--noise', str(eta), '--seed', str(seed),
               '--out-dir', out_dir]
        print(f'[run] {" ".join(cmd)}', flush=True)
        with open(log, 'w') as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        if p.returncode != 0:
            fail += 1
            print(f'[FAIL] {args.model} eta={eta} seed={seed} rc={p.returncode} '
                  f'log={log}', flush=True)
        else:
            print(f'[ok] {args.model} eta={eta} seed={seed}', flush=True)
    print(f'[{args.model}] shard {args.shard} done: fail={fail}', flush=True)
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
