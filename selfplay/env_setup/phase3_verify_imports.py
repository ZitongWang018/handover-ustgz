#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""phase3_verify_imports.py -- 三期 ACD 的 **纯 import 级** 自检（不构造环境、不 step、不计算）。

覆盖 team-lead 指定的 5 条：
  1. python -V + PYTHONPATH 是否生效 (sys.path[:3])
  2. import torch / transformers / textworld / gym + 版本
  3. fast_downward.interface.load_lib() -> OK   (glibc 2.28 唯一硬骨头)
  4. glibc232 兼容层是否真被用上：打印 DOWNWARD_LIB_PATH + ldd 结果
  5. LD_LIBRARY_PATH 是否含 <env>/lib (GLIBCXX_3.4.29 靠 conda libstdc++)

用法（三期登录节点）：
    PYTHONPATH=/data/user/xyu608/ztwang/selfplay/src:/data/user/xyu608/ztwang/selfplay/third_party/py311deps \
    LD_LIBRARY_PATH=/share/anaconda3/envs/vllm017/lib \
    /share/anaconda3/envs/vllm017/bin/python phase3_verify_imports.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.environ.get("SELFPLAY_REPO", "/data/user/xyu608/ztwang/selfplay")
SRC = os.path.join(ROOT, "src")
DEPS = os.environ.get("SELFPLAY_DEPS", os.path.join(ROOT, "third_party", "py311deps"))

FAILS: list[str] = []


def sec(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def ok(name: str, val=None) -> None:
    extra = "" if val is None else f"  -> {val}"
    print(f"[ OK ] {name}{extra}")


def bad(name: str, err: str) -> None:
    FAILS.append(f"{name}: {err}")
    print(f"[FAIL] {name}  -> {err}")


# 确保 src / deps 在最前（与二期同一套注入逻辑）
for p in (SRC, DEPS):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------- 1. 解释器 / PYTHONPATH ----------
sec("1) 解释器与 PYTHONPATH")
print("python      :", sys.version.split()[0], f"({sys.executable})")
print("PYTHONPATH  :", os.environ.get("PYTHONPATH", "<unset>"))
print("sys.path[:3]:")
for p in sys.path[:3]:
    print("   ", p)
if sys.path[0] == SRC or SRC in sys.path[:3]:
    ok("src 在 sys.path 前部", SRC)
else:
    bad("src 未生效", f"{SRC} 不在 sys.path[:3]")
if DEPS in sys.path[:3]:
    ok("deps 在 sys.path 前部", DEPS)
else:
    bad("deps 未生效", f"{DEPS} 不在 sys.path[:3]")

# ---------- 兼容层（必须早于 import textworld） ----------
sec("0) alfworld_compat.install()  [必须早于 import textworld]")
DOWNWARD_LIB_PATH = None
try:
    from alfworld_compat import install  # noqa: PLC0415
    compat_dir = install(verbose=True)
    import fast_downward.interface as _fi  # noqa: PLC0415
    DOWNWARD_LIB_PATH = _fi.DOWNWARD_LIB_PATH
    ok("install() 完成", f"compat_dir={compat_dir}")
except Exception as exc:  # noqa: BLE001
    bad("alfworld_compat.install()", f"{type(exc).__name__}: {exc}")

# ---------- 2. 四个核心 import ----------
sec("2) import torch / transformers / textworld / gym")
for name in ("torch", "transformers", "textworld", "gym"):
    try:
        m = __import__(name)
        ok(f"import {name}", getattr(m, "__version__", "<no __version__>"))
    except Exception as exc:  # noqa: BLE001
        bad(f"import {name}", f"{type(exc).__name__}: {exc}")

# 额外：ALFWorld 本体 + 环境类（仍为 import 级，不构造环境）
try:
    import alfworld  # noqa: PLC0415
    ok("import alfworld", os.path.dirname(alfworld.__file__))
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # noqa: PLC0415
    ok("import AlfredTWEnv (未构造环境)", AlfredTWEnv.__name__)
except Exception as exc:  # noqa: BLE001
    bad("import alfworld / AlfredTWEnv", f"{type(exc).__name__}: {exc}")

# textworld.gym 注册入口也点一下（import 级）
try:
    import textworld.gym  # noqa: F401,PLC0415
    ok("import textworld.gym")
except Exception as exc:  # noqa: BLE001
    bad("import textworld.gym", f"{type(exc).__name__}: {exc}")

# ---------- 3. fast_downward.load_lib() ----------
sec("3) fast_downward.interface.load_lib()  [glibc 2.28 硬骨头]")
try:
    from fast_downward import interface as fdi  # noqa: PLC0415
    lib = fdi.load_lib()
    fdi.close_lib(lib)
    ok("load_lib() -> OK")
except Exception as exc:  # noqa: BLE001
    bad("fast_downward.interface.load_lib()", f"{type(exc).__name__}: {exc}")

# ---------- 4. glibc232 兼容层是否真被用上 ----------
sec("4) glibc232 兼容层核验")
print("DOWNWARD_LIB_PATH :", DOWNWARD_LIB_PATH)
print("ALFWORLD_GLIBC_COMPAT_DIR :", os.environ.get("ALFWORLD_GLIBC_COMPAT_DIR", "<unset>"))
if DOWNWARD_LIB_PATH:
    if "glibc232" in DOWNWARD_LIB_PATH:
        ok("DOWNWARD_LIB_PATH 指向 glibc232 降级副本")
    else:
        bad("DOWNWARD_LIB_PATH 未指向 glibc232", DOWNWARD_LIB_PATH)
    try:
        out = subprocess.run(["ldd", DOWNWARD_LIB_PATH], capture_output=True, text=True)
        print("--- ldd", DOWNWARD_LIB_PATH, "---")
        print(out.stdout.rstrip() or out.stderr.rstrip())
        if "not found" in (out.stdout + out.stderr):
            bad("ldd 存在未解析依赖", "见上方 'not found'")
        else:
            ok("ldd 全部可解析")
    except FileNotFoundError:
        print("(ldd 不可用，跳过)")
else:
    bad("DOWNWARD_LIB_PATH", "未设置（install() 未成功）")

# ---------- 5. LD_LIBRARY_PATH / libstdc++ ----------
sec("5) LD_LIBRARY_PATH 与 GLIBCXX_3.4.29")
ldlp = os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH :", ldlp or "<unset>")
if "/share/anaconda3/envs/" in ldlp and "/lib" in ldlp:
    ok("LD_LIBRARY_PATH 含 conda env 的 lib 目录")
else:
    bad("LD_LIBRARY_PATH", "不含 conda env 的 lib 目录（GLIBCXX_3.4.29 可能缺失）")

# 找出 venv/conda lib 下的 libstdc++ 并检查符号
cands = []
for d in ldlp.split(":"):
    if d:
        cands.append(os.path.join(d, "libstdc++.so.6"))
for c in cands:
    if os.path.exists(c):
        try:
            r = subprocess.run(["strings", c], capture_output=True, text=True)
            n = r.stdout.count("GLIBCXX_3.4.29")
            print(f"libstdc++ {c}: GLIBCXX_3.4.29 x{n}")
            if n > 0:
                ok("libstdc++ 提供 GLIBCXX_3.4.29")
            else:
                bad("libstdc++ 缺 GLIBCXX_3.4.29", c)
        except FileNotFoundError:
            print("(strings 不可用，跳过)")

# ---------- 汇总 ----------
sec("SUMMARY")
if FAILS:
    print(f"FAILED {len(FAILS)} 项：")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("ALL 5 CHECKS PASSED")
sys.exit(0)
