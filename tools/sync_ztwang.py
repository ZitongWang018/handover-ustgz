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
import sys

import paramiko

LOCAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "ztwang"
USER = "xyu608"
KEY = pathlib.Path.home() / ".ssh" / "hpc2gz_ed25519"
SKIP_DIRS = {".git", "__pycache__", "logs"}

HOSTS = {
    "hpc2": ("hpc2login.hpc.hkust-gz.edu.cn", "/hpc2hdd/home/xyu608/ztwang"),
    "hpc3": ("hpc3login.hpc.hkust-gz.edu.cn", "/data/user/xyu608/ztwang"),
}


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
