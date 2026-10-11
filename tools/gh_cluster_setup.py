#!/usr/bin/env python
"""One-shot: wire the cluster's ~/ztwang to the GitHub repo and sync it.

Token is streamed over SFTP into a 0600 temp file on the cluster, used via
GIT_ASKPASS (never in argv/env/ps), then deleted at the end.

Usage:
  set GITHUB_TOKEN=<pat>
  <python> hpc_tools/gh_cluster_setup.py            # sync now, delete token after
  <python> hpc_tools/gh_cluster_setup.py --persist  # keep token for future pull/push
"""
import os
import pathlib
import stat
import sys

import paramiko

HOST = "hpc2login.hpc.hkust-gz.edu.cn"
USER = "xyu608"
KEY = str(pathlib.Path.home() / ".ssh" / "hpc2gz_ed25519")
OWNER = "ZitongWang018"
REPO = "handover-ustgz"
REMOTE = f"https://github.com/{OWNER}/{REPO}.git"

TOKEN_FILE = f"/hpc2hdd/home/{USER}/.gh_token"
ASKPASS = f"/hpc2hdd/home/{USER}/.gh_askpass.sh"


def main():
    persist = "--persist" in sys.argv
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN 未设置")

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, username=USER, key_filename=KEY, allow_agent=False,
                look_for_keys=False, timeout=30, banner_timeout=60)

    def run(cmd, timeout=300):
        _, out, err = cli.exec_command(cmd, timeout=timeout)
        o = out.read().decode("utf-8", "replace")
        e = err.read().decode("utf-8", "replace")
        rc = out.channel.recv_exit_status()
        return rc, o, e

    sftp = cli.open_sftp()
    try:
        print("[1/4] 上传 token (0600) ...")
        with sftp.file(TOKEN_FILE, "w") as fh:
            fh.write(token)
        sftp.chmod(TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)

        print("[2/4] 配置 GIT_ASKPASS ...")
        askpass_body = (
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) echo 'x-access-token' ;;\n"
            f"  *Password*) cat {TOKEN_FILE} ;;\n"
            "esac\n"
        )
        with sftp.file(ASKPASS, "w") as fh:
            fh.write(askpass_body)
        sftp.chmod(ASKPASS, stat.S_IRWXU)

        print("[3/4] 备份 + 设置 remote + 同步 ...")
        script = f"""
set -u
cd ~ || exit 1
BK=~/ztwang_backup_$(date +%Y%m%d_%H%M%S).tar.gz
tar czf "$BK" ztwang 2>/dev/null && echo "备份 -> $BK ($(du -h "$BK" | cut -f1))"

cd ~/ztwang || exit 1
git remote remove origin 2>/dev/null
git remote add origin {REMOTE}
export GIT_ASKPASS={ASKPASS}
export GIT_TERMINAL_PROMPT=0
echo "--- fetch ---"
git fetch origin 2>&1 | tail -5
echo "--- reset 到 origin/main ---"
git reset --hard origin/main 2>&1 | tail -3
git branch --set-upstream-to=origin/main main >/dev/null 2>&1 || git branch -M main
echo "--- 结果 ---"
git log --oneline -3
echo "tracked: $(git ls-files | wc -l) files"
echo "--- status (应为空) ---"
git status --short | head -5
echo "--- remote ---"
git remote -v
"""
        rc, out, err = run(script, timeout=300)
        print(out)
        if err.strip():
            print("stderr:", err[-800:])

        print("[4/4] 清理 ...")
        if persist:
            run(f"git config --global core.askpass {ASKPASS}")
            print("      已保留 token（后续 git pull/push 免认证）；文件模式 0600")
        else:
            run(f"rm -f {TOKEN_FILE} {ASKPASS}")
            print("      token 与 askpass 已从集群删除")
    finally:
        sftp.close()
        cli.close()


if __name__ == "__main__":
    main()
