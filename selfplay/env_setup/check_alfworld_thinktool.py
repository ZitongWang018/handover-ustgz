#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_alfworld_thinktool.py -- ALFWorld 文本环境在 thinktool(py3.10) 上的端到端自检。

设计目标
--------
1. 不修改 `~/miniconda3/envs/thinktool`：只把 `third_party/py310deps` 插到 sys.path 最前面。
2. 不修改 `~/thinktool/`：ALFWorld 代码全部来自 `py310deps/alfworld`。
3. 覆盖 glibc 兼容层 -> import -> 建环境 -> reset -> step 的全链路。

用法
----
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate thinktool
    python ~/ztwang/selfplay/env_setup/check_alfworld_thinktool.py                 # N=2, 5 steps
    python ~/ztwang/selfplay/env_setup/check_alfworld_thinktool.py --batch-size 1
    python ~/ztwang/selfplay/env_setup/check_alfworld_thinktool.py --json /tmp/r.json

退出码 0 = 全链路通过。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.expanduser(
    os.environ.get("SELFPLAY_REPO", "~/ztwang/selfplay")
)
# DEPS 默认二期用的 py310deps；三期为 thinktool 之外的 conda env 服务时，
# 用 SELFPLAY_DEPS 指到 third_party/py311deps 或 py312deps 即可复用同一脚本。
DEPS = os.environ.get("SELFPLAY_DEPS") or os.path.join(REPO, "third_party", "py310deps")
SRC = os.path.join(REPO, "src")
DEFAULT_CONFIG = os.path.join(REPO, "configs", "base_config.yaml")

RESULTS: dict = {"checks": [], "timing": {}, "steps": [], "ok": False}


def log(msg: str) -> None:
    print(msg, flush=True)


def check(name: str, fn):
    """跑一个检查项，记录成功/失败。返回 (ok, value)。"""
    t0 = time.time()
    try:
        val = fn()
        dt = time.time() - t0
        RESULTS["checks"].append({"name": name, "ok": True, "secs": round(dt, 4),
                                  "value": _short(val)})
        log(f"[ OK ] {name}  ({dt:.3f}s)  -> {_short(val)}")
        return True, val
    except Exception as exc:  # noqa: BLE001
        dt = time.time() - t0
        RESULTS["checks"].append({"name": name, "ok": False, "secs": round(dt, 4),
                                  "error": f"{type(exc).__name__}: {exc}"})
        log(f"[FAIL] {name}  ({dt:.3f}s)")
        traceback.print_exc()
        return False, None


def _short(v, n: int = 120):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def bootstrap() -> None:
    """把 py310deps / src 放到 sys.path 最前面，然后装 glibc 兼容层。

    注意顺序：py310deps 必须 **先于** thinktool 的 site-packages，
    否则会 import 到错误的 numpy / 缺 textworld。
    """
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    if DEPS not in sys.path:
        sys.path.insert(0, DEPS)
    os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    # 关键：install() 必须在 import textworld 之前调用
    from alfworld_compat import install  # noqa: PLC0415
    install(verbose=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=2, help="init_env(batch_size=N)")
    ap.add_argument("--steps", type=int, default=5, help="reset() 之后额外 step() 的次数")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--split", default="eval_out_of_distribution",
                    choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    ap.add_argument("--max-games", type=int, default=0, help="把 game_files 截断到前 N 个(0=不截断)")
    ap.add_argument("--json", default="", help="把结果 JSON 写到该路径")
    args = ap.parse_args()

    log("=" * 78)
    log("ALFWorld on thinktool -- end-to-end self check")
    log(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    log(f"deps dir    : {DEPS}")
    log(f"config      : {args.config}")
    log(f"split       : {args.split}")
    log(f"batch_size  : {args.batch_size}")
    log("=" * 78)
    RESULTS["python"] = sys.version
    RESULTS["executable"] = sys.executable
    RESULTS["args"] = vars(args)

    # ---------- 0. bootstrap: glibc 兼容层 ----------
    ok, _ = check("alfworld_compat.install()", bootstrap)
    if not ok:
        return _finish(args)

    # ---------- 1. 依赖 import ----------
    import numpy  # noqa: PLC0415
    check("import numpy", lambda: numpy.__version__)
    check("import torch", lambda: __import__("torch").__version__)
    check("import textworld", lambda: __import__("textworld").__version__)
    check("import gym", lambda: __import__("gym").__version__)
    check("import alfworld", lambda: __import__("alfworld").__file__)
    ok, AlfredTWEnv = check(
        "import alfworld.agents.environment.alfred_tw_env",
        lambda: __import__("alfworld.agents.environment.alfred_tw_env",
                           fromlist=["AlfredTWEnv"]).AlfredTWEnv,
    )
    if not ok:
        return _finish(args)
    check("fast_downward.load_lib()", lambda: __import__("alfworld_compat").self_check())

    # ---------- 2. 读 config ----------
    import yaml  # noqa: PLC0415
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    tm = config["general"]["training_method"]
    log(f"config.general.training_method = {tm!r}   env.expert_type = "
        f"{config['env']['expert_type']!r}")
    assert tm in ("dagger", "dqn"), f"training_method 必须是 dagger/dqn，实际 {tm!r}"
    assert config["env"]["expert_type"] in ("handcoded", "downward")
    assert os.environ.get("ALFWORLD_DATA"), "ALFWORLD_DATA 未设置"

    # ---------- 3. 构造 AlfredTWEnv ----------
    t0 = time.time()
    env = AlfredTWEnv(config, train_eval=args.split)
    t_collect = time.time() - t0
    n_games_all = len(env.game_files)
    log(f"[time] AlfredTWEnv(config) 含 game_files 扫描: {t_collect:.3f}s, "
        f"共 {n_games_all} 个 game")
    RESULTS["timing"]["construct_env_and_scan"] = round(t_collect, 3)
    RESULTS["n_games_total"] = n_games_all
    if args.max_games > 0:
        env.game_files = env.game_files[: args.max_games]
        env.num_games = len(env.game_files)
    assert len(env.game_files) >= args.batch_size, (
        f"game_files({len(env.game_files)}) < batch_size({args.batch_size})")

    # ---------- 4. init_env + reset ----------
    t0 = time.time()
    tw = env.init_env(batch_size=args.batch_size)
    t_init = time.time() - t0
    log(f"[time] init_env(batch_size={args.batch_size}): {t_init:.3f}s")
    RESULTS["timing"]["init_env"] = round(t_init, 3)

    t0 = time.time()
    obs, infos = tw.reset()
    t_reset = time.time() - t0
    log(f"[time] reset(): {t_reset:.3f}s")
    RESULTS["timing"]["reset"] = round(t_reset, 3)

    # ---------- 4b. 打印第一次观测 ----------
    log("-" * 78)
    log(f"reset -> obs[0] ({len(obs[0])} chars):")
    log(obs[0])
    log(f"infos keys: {sorted(infos.keys())}")
    n_adm = len(infos["admissible_commands"][0])
    log(f"admissible_commands[0] 长度 = {n_adm}")
    log(f"admissible_commands[0][:5] = {infos['admissible_commands'][0][:5]}")
    log(f"won[0] = {infos['won'][0]}")
    log(f"extra.gamefile[0] = {infos['extra.gamefile'][0]}")
    log("-" * 78)

    RESULTS["reset_obs0"] = obs[0]
    RESULTS["reset_n_admissible"] = n_adm
    RESULTS["reset_won"] = [bool(x) for x in infos["won"]]
    RESULTS["reset_gamefiles"] = list(infos["extra.gamefile"])

    # ---------- 5. 连续 step ----------
    # 注意：textworld.gym batch env 的 step() 要求 **每个子环境一个 action**，
    # 即 len(actions) == batch_size。这里取每个子环境 admissible_commands 的第一条。
    nb = args.batch_size
    cur_obs, cur_infos = obs, infos
    for i in range(args.steps):
        actions = [cur_infos["admissible_commands"][b][0] for b in range(nb)]
        t0 = time.time()
        cur_obs, _rew, done, cur_infos = tw.step(actions)
        dt = time.time() - t0
        rec = {
            "i": i,
            "actions": actions,
            "secs": round(dt, 4),
            "won": [bool(x) for x in cur_infos["won"]],
            "done": [bool(x) for x in done],
            "n_admissible": [len(cur_infos["admissible_commands"][b]) for b in range(nb)],
        }
        RESULTS["steps"].append(rec)
        log(f"[step {i}] {dt:.4f}s  actions={actions}  "
            f"won={rec['won']} done={rec['done']} n_adm={rec['n_admissible']}")
        if all(rec["done"]):
            log(f"  (所有 episode 在第 {i} 步结束)")
            break

    step_secs = [s["secs"] for s in RESULTS["steps"]]
    if step_secs:
        RESULTS["timing"]["step_avg"] = round(sum(step_secs) / len(step_secs), 4)
        RESULTS["timing"]["step_min"] = min(step_secs)
        RESULTS["timing"]["step_max"] = max(step_secs)
        log(f"[time] step 统计: n={len(step_secs)} avg={RESULTS['timing']['step_avg']:.4f}s "
            f"min={min(step_secs):.4f}s max={max(step_secs):.4f}s")

    # ---------- 6. 收尾 ----------
    try:
        tw.close()
    except Exception:  # noqa: BLE001
        pass

    RESULTS["ok"] = all(c["ok"] for c in RESULTS["checks"]) and bool(RESULTS["steps"])
    log("=" * 78)
    log("RESULT: " + ("PASS" if RESULTS["ok"] else "FAIL"))
    log("=" * 78)
    return _finish(args)


def _finish(args) -> int:
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(RESULTS, fh, indent=2, ensure_ascii=False)
        log(f"[json] 写入 {args.json}")
    return 0 if RESULTS.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
