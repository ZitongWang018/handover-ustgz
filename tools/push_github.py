#!/usr/bin/env python
"""Create a private GitHub repo and push the local ztwang/ workspace.

Usage (token passed via env so it never lands in a file or shell history file):
    set GITHUB_TOKEN=ghp_xxx
    <python> hpc_tools/push_github.py            # create repo + push
    <python> hpc_tools/push_github.py --dry-run  # only check auth + show plan

Requires: the local git repo at <workspace>/ztwang with a commit already made.
"""
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

OWNER = "ZitongWang018"
REPO = "handover-ustgz"
DESC = "HKUST-GZ HPC 二期工作区：集群规范、Slurm 作业模板、selfplay 工程"
LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent / "ztwang"
API = "https://api.github.com"


def api(token, method, path, payload=None):
    req = urllib.request.Request(
        API + path,
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hpc-workspace-setup",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return e.code, {"_error": body}


def git(*args, **kw):
    return subprocess.run(["git", "-C", str(LOCAL_REPO), *args],
                          capture_output=True, text=True, **kw)


def main():
    dry = "--dry-run" in sys.argv
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit("GITHUB_TOKEN 未设置。请先: set GITHUB_TOKEN=ghp_xxx")

    print(f"[1/4] 校验 token ...")
    code, me = api(token, "GET", "/user")
    if code != 200:
        sys.exit(f"token 无效 (HTTP {code}): {me}")
    login = me.get("login")
    print(f"      已认证为 {login}")
    scopes = me.get("_scopes")
    if login.lower() != OWNER.lower():
        print(f"      警告: token 属于 {login}，目标 owner 是 {OWNER}")

    print(f"[2/4] 检查仓库 {OWNER}/{REPO} ...")
    code, info = api(token, "GET", f"/repos/{OWNER}/{REPO}")
    if code == 200:
        print("      仓库已存在，跳过创建")
    elif code == 404:
        if dry:
            print("      仓库不存在（dry-run，不创建）")
            info = {"clone_url": f"https://github.com/{OWNER}/{REPO}.git"}
        else:
            print("      创建私有仓库 ...")
            code, info = api(token, "POST", "/user/repos", {
                "name": REPO,
                "description": DESC,
                "private": True,
                "has_issues": True,
                "has_wiki": False,
                "auto_init": False,
            })
            if code not in (200, 201):
                sys.exit(f"创建失败 (HTTP {code}): {info}")
            print(f"      已创建 https://github.com/{OWNER}/{REPO}")
    else:
        sys.exit(f"查询仓库失败 (HTTP {code}): {info}")

    print("[3/4] 配置 remote ...")
    if dry:
        print("      (dry-run 跳过)")
    else:
        git("remote", "remove", "origin")
        r = git("remote", "add", "origin",
                f"https://{OWNER}:{token}@github.com/{OWNER}/{REPO}.git")
        if r.returncode != 0:
            sys.exit(f"配置 remote 失败: {r.stderr}")
        print("      origin 已设置（token 仅存于本地 .git/config）")

    print("[4/4] 推送 ...")
    if dry:
        r = git("log", "--oneline", "-1")
        print("      待推送 commit:", r.stdout.strip())
        print("      dry-run 结束")
    else:
        r = git("push", "-u", "origin", "HEAD:main")
        print(r.stdout or r.stderr)
        if r.returncode != 0:
            sys.exit(f"推送失败: {r.stderr}")
        # 把 token 从 remote URL 里摘掉，避免明文留在 .git/config
        git("remote", "set-url", "origin",
            f"https://github.com/{OWNER}/{REPO}.git")
        print(f"      完成 -> https://github.com/{OWNER}/{REPO}")
        print("      已从 .git/config 移除 token（后续 push 需重新认证）")


if __name__ == "__main__":
    main()
