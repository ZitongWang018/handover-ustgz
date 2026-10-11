#!/usr/bin/env python
"""Push local ztwang/ to GitHub entirely via the REST API (no `git push`).

Useful when the sandbox blocks git's network access but plain HTTPS works.

    set GITHUB_TOKEN=<pat>
    <python> hpc_tools/push_via_api.py            # commit + update main
    <python> hpc_tools/push_via_api.py --dry-run  # show what would change
"""
import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

OWNER = "ZitongWang018"
REPO = "handover-ustgz"
BRANCH = "main"
API = "https://api.github.com"
LOCAL_REPO = pathlib.Path(__file__).resolve().parent.parent / "ztwang"
SKIP_DIRS = {".git", "__pycache__", "logs"}
EXEC_SUFFIX = (".sh", ".sbatch")

TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()


def api(method, path, payload=None):
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ztwang-pusher",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            return r.status, (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"_error": e.read().decode()}


def collect():
    out = {}
    for dirpath, dirnames, filenames in os.walk(LOCAL_REPO):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            rel = p.relative_to(LOCAL_REPO).as_posix()
            out[rel] = p
    return out


def main():
    dry = "--dry-run" in sys.argv
    if not TOKEN:
        sys.exit("GITHUB_TOKEN 未设置")

    files = collect()
    print(f"[1/5] 本地文件 {len(files)} 个")

    code, ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    if code != 200:
        sys.exit(f"读取 ref 失败 ({code}): {ref}")
    head_sha = ref["object"]["sha"]
    code, commit = api("GET", f"/repos/{OWNER}/{REPO}/git/commits/{head_sha}")
    base_tree = commit["tree"]["sha"]
    print(f"[2/5] 远端 main = {head_sha[:8]}  tree={base_tree[:8]}")

    # 远端已有文件清单，用于对比
    code, tree0 = api("GET", f"/repos/{OWNER}/{REPO}/git/trees/{base_tree}?recursive=1")
    remote_files = {x["path"]: x["sha"] for x in tree0.get("tree", []) if x["type"] == "blob"}

    entries = []
    changed = []
    for rel, p in sorted(files.items()):
        raw = p.read_bytes()
        if dry:
            changed.append(rel)
            continue
        blob = {
            "content": base64.b64encode(raw).decode(),
            "encoding": "base64",
        }
        code, res = api("POST", f"/repos/{OWNER}/{REPO}/git/blobs", blob)
        if code not in (200, 201):
            sys.exit(f"上传 blob 失败 {rel} ({code}): {res}")
        mode = "100755" if rel.endswith(EXEC_SUFFIX) else "100644"
        entries.append({"path": rel, "mode": mode, "type": "blob", "sha": res["sha"]})
        changed.append(rel)
    print(f"[3/5] 已上传 {len(entries)} 个 blob")

    if dry:
        new = [r for r in changed if r not in remote_files]
        print("  dry-run：将提交", len(changed), "个文件；远端新增", len(new), "个")
        for r in new:
            print("   +", r)
        return

    missing = [r for r in remote_files if r not in files]
    payload = {"base_tree": base_tree, "tree": entries}
    code, new_tree = api("POST", f"/repos/{OWNER}/{REPO}/git/trees", payload)
    if code not in (200, 201):
        sys.exit(f"创建 tree 失败 ({code}): {new_tree}")
    print(f"[4/5] 新 tree = {new_tree['sha'][:8]}")

    msg = os.environ.get("GIT_COMMIT_MSG") or (
        "feat: 接入三期 ACD (H100) 集群工作区与两集群对比规范\n\n"
        "- 新增 docs/HPC3_GUIDE.md：三期专项指南（H100/无外网/队列/配额）\n"
        "- CLUSTER_GUIDE 置顶两条铁律：只在 ztwang 内写文件、训练断点只留一个\n"
        "- 补齐两集群对照（CLUSTER_GUIDE / ENVIRONMENT / CLUSTER_SNAPSHOT）\n"
        "- check_cluster.sh 自动识别二期/三期；sync_ztwang.py 支持 --host\n"
        "- 新增 tools/sbatch/gpu_h100x8.sbatch  三期 H100 作业模板"
    )
    code, new_commit = api("POST", f"/repos/{OWNER}/{REPO}/git/commits",
                           {"message": msg, "tree": new_tree["sha"], "parents": [head_sha]})
    if code not in (200, 201):
        sys.exit(f"创建 commit 失败 ({code}): {new_commit}")
    print(f"      commit = {new_commit['sha'][:8]}")

    code, upd = api("PATCH", f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
                    {"sha": new_commit["sha"], "force": False})
    if code not in (200, 201):
        sys.exit(f"更新 ref 失败 ({code}): {upd}")
    print(f"[5/5] main -> {upd['object']['sha'][:8]}   OK")
    if missing:
        print("  注意：以下远端文件本地已无（本次未删除）:", missing)


if __name__ == "__main__":
    main()
