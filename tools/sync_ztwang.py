#!/usr/bin/env python
"""Mirror local ztwang/ -> remote ~/ztwang/ over SFTP.

Run from anywhere:
    <python> hpc_tools/sync_ztwang.py                 # -> 二期 (hpc2)
    <python> hpc_tools/sync_ztwang.py --host hpc3     # -> 三期 (hpc3)
    <python> hpc_tools/sync_ztwang.py --list          # dry run, list only

三期无法访问外网（不能 git clone/pull），只能这样推文件。
"""
import os
import pathlib
import posixpath
import re
import socket
import subprocess
import sys

import paramiko

LOCAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "ztwang"
# 兼容两种存放位置：<root>/ztwang/tools/ 与 <root>/hpc_tools/。
# 不做这层判断的话，从 <root>/ztwang/tools/ 跑会算成 <root>/ztwang/ztwang（不存在）
# -> "0 files, 0 uploaded"，静默什么都没同步。
_CANDS = [pathlib.Path(__file__).resolve().parent.parent / "ztwang",
          pathlib.Path(__file__).resolve().parent.parent]
LOCAL_ROOT = next((c for c in _CANDS if (c / "docs").is_dir() and (c / "selfplay").is_dir()),
                  _CANDS[0])
USER = "xyu608"
KEY = pathlib.Path.home() / ".ssh" / "hpc2gz_ed25519"
SKIP_DIRS = {".git", "__pycache__", "logs"}

HOSTS = {
    "hpc2": ("hpc2login.hpc.hkust-gz.edu.cn", "/hpc2hdd/home/xyu608/ztwang"),
    "hpc3": ("hpc3login.hpc.hkust-gz.edu.cn", "/data/user/xyu608/ztwang"),
}

# 校园网内网 IP：2026-10-12 实测。本机 DNS 偶尔解析不了集群域名（ssh 报
# "Could not resolve hostname"），但 Windows 的 nslookup 能拿到地址、网络本身是通的。
_FALLBACK_IP = {"hpc2": "10.120.18.63", "hpc3": "10.120.81.30"}


def resolve(host: str, name: str) -> str:
    """解析集群域名；失败就回退到 nslookup / 内置 IP，避免 DNS 抖动卡住同步。"""
    try:
        socket.getaddrinfo(host, 22)
        return host
    except socket.gaierror:
        pass
    for cand in (os.environ.get(f"{name.upper()}_IP"), _nslookup(host), _FALLBACK_IP.get(name)):
        if cand:
            print(f"[dns] {host} 解析失败 -> 回退 {cand}")
            return cand
    sys.exit(f"[dns] 无法解析 {host}；请设 {name.upper()}_IP=<ip> 重试")


def _nslookup(host: str):
    """Windows/Linux 通用：从 nslookup 输出里抠出 Address（跳过 DNS 服务器自身那行）。"""
    try:
        out = subprocess.run(["nslookup", host], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    addrs = re.findall(r"(?:Address|Addresses)\s*:\s*([0-9.]+)", out)
    addrs = [a for a in addrs if not a.startswith("127.")]
    # 第一个 Address 常是 DNS 服务器，取最后一个更接近目标
    return addrs[-1] if addrs else None


def connect(host):
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(host, username=USER, key_filename=str(KEY),
                allow_agent=False, look_for_keys=False, timeout=30,
                banner_timeout=60, auth_timeout=60)
    return cli


def ensure_dir(sftp, path, made):
    """mkdir -p over SFTP."""
    if path in made:
        return
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = cur + "/" + part
        if cur in made:
            continue
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass  # 并发/权限已存在
        made.add(cur)


def main():
    argv = sys.argv[1:]
    dry = "--list" in argv
    name = "hpc2"
    if "--host" in argv:
        name = argv[argv.index("--host") + 1]
    if name not in HOSTS:
        sys.exit(f"unknown host {name!r}; choose from {list(HOSTS)}")
    host, remote_root = HOSTS[name]
    host = resolve(host, name)

    files = []
    for dirpath, dirnames, filenames in os.walk(LOCAL_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            files.append(pathlib.Path(dirpath) / fn)
    files.sort()

    print(f"[{name}] {host} -> {remote_root}")
    cli = connect(host)
    sftp = cli.open_sftp()
    made = set()
    n = 0
    ensure_dir(sftp, remote_root, made)
    ensure_dir(sftp, posixpath.join(remote_root, "logs"), made)
    for p in files:
        rel = p.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(remote_root, rel)
        rel_dir = posixpath.dirname(rel)
        if dry:
            print("would upload", rel)
            continue
        if rel_dir and rel_dir != ".":
            ensure_dir(sftp, posixpath.join(remote_root, rel_dir), made)
        sftp.put(str(p), remote)
        print(f"  {rel}  ({p.stat().st_size} B)")
        n += 1
    sftp.close()
    cli.close()
    print(f"\n{len(files)} files, {n} uploaded -> {remote_root}")


if __name__ == "__main__":
    main()
