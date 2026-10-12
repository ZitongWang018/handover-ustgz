#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把**离线跑出来**的 SwanLab 日志补传到云端。

为什么需要这个：

    三期 ACD 集群**完全没有外网**（github / pypi 都不可达），所以那边训练时
    只能 `EZ_SWANLAB_MODE=local`，指标写进 `runs/<name>/swanlog/` 就完了。
    等作业跑完（或跑到某个节点），在本机 / 二期这种有网的地方跑一下这个脚本，
    SwanLab 就会把本地日志补传上去，最终看到的曲线跟实时上传的一模一样。

典型用法（本机）：

    # 1) 把三期的 swanlog 拉回来
    HPC_HOST=hpc3login.hpc.hkust-gz.edu.cn python hpc_tools/hpc.py \\
        get ztwang/selfplay/runs/<run>/swanlog  ./swanlog_<run>

    # 2) 上传到 SwanLab
    python selfplay/scripts/ez_sync_swanlab.py ./swanlog_<run>

    # 3) 想让它每 10 分钟自动补传一次（"训练一半也能看到曲线"）：
    python selfplay/scripts/ez_sync_swanlab.py ./swanlog_<run> --watch 600

`--watch` 模式下脚本会反复调 `swanlab sync`，SwanLab 自己会做增量，
重复跑同一目录不会上去重产生重复实验。

⚠️ 前提：这台机器已经登录过 SwanLab
    `python -c "import swanlab; swanlab.login(api_key='<你的 key>')"`
    （凭据存在 ~/.swanlab/.netrc，权限 0600）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_sync(logdir: Path, dry: bool = False) -> bool:
    cmd = [sys.executable, "-m", "swanlab", "sync", str(logdir)]
    if dry:
        print(f"[dry-run] {' '.join(cmd)}")
        return True
    print(f"[sync] {logdir}", flush=True)
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("  ✗ 找不到 swanlab：pip install swanlab", flush=True)
        return False
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-6:])
    if r.returncode == 0:
        print(f"  ✓ 上传完成\n{tail}", flush=True)
        return True
    print(f"  ✗ 失败(exit {r.returncode})\n{tail}", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="把离线 SwanLab 日志补传到云端")
    ap.add_argument("logdir", nargs="+", type=Path,
                    help="一个或多个存放离线日志的目录（run_dir/swanlog）")
    ap.add_argument("--watch", type=int, default=0, metavar="SEC",
                    help="每 SEC 秒重试一次；0 = 只跑一次（默认）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    dirs = [d for d in args.logdir if d.exists()]
    missing = [str(d) for d in args.logdir if not d.exists()]
    for m in missing:
        print(f"[warn] 目录不存在，跳过：{m}")
    if not dirs:
        print("没有任何有效目录，退出")
        return 1

    while True:
        for d in dirs:
            _run_sync(d, dry=args.dry_run)
        if args.watch <= 0:
            break
        print(f"[watch] {args.watch}s 后重试（Ctrl-C 退出）", flush=True)
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n[watch] 收到中断，退出")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
