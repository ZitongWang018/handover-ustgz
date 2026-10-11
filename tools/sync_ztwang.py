#!/usr/bin/env python
"""Mirror local ztwang/ -> remote ~/ztwang/ over SFTP.

Run from anywhere:
    <python> hpc_tools/sync_ztwang.py            # upload changed files
    <python> hpc_tools/sync_ztwang.py --list     # dry run, list only
"""
import os
import pathlib
import posixpath
import sys

import paramiko

LOCAL_ROOT = pathlib.Path(__file__).resolve().parent.parent / "ztwang"
REMOTE_ROOT = "/hpc2hdd/home/xyu608/ztwang"
HOST = "hpc2login.hpc.hkust-gz.edu.cn"
USER = "xyu608"
KEY = pathlib.Path.home() / ".ssh" / "hpc2gz_ed25519"

SKIP_DIRS = {".git", "__pycache__", "logs"}


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, key_filename=str(KEY),
                allow_agent=False, look_for_keys=False, timeout=30,
                banner_timeout=60, auth_timeout=60)
    return cli


def main():
    dry = "--list" in sys.argv
    files = []
    for dirpath, dirnames, filenames in os.walk(LOCAL_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            files.append(p)
    files.sort()

    cli = connect()
    sftp = cli.open_sftp()
    made = set()
    for p in files:
        rel = p.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        rdir = posixpath.dirname(remote)
        if dry:
            print("would upload", rel)
            continue
        if rdir not in made:
            try:
                sftp.stat(rdir)
            except IOError:
                sftp.mkdir(rdir)
            made.add(rdir)
        sftp.put(str(p), remote)
        print(f"  {rel}  ({p.stat().st_size} B)")
    sftp.close()
    cli.close()
    print(f"\n{len(files)} files processed -> {REMOTE_ROOT}")


if __name__ == "__main__":
    main()
