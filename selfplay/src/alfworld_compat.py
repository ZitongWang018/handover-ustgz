"""ALFWorld / TextWorld 在本集群上的兼容层。

**必须在 `import textworld` 之前调用 `install()`。**

背景：TextWorld 的 PDDL 规划器 `fast_downward/libdownward.so` 需要 glibc 2.32，
而二期是 Ubuntu 20.04（glibc 2.31）、三期是 Rocky 8（glibc 2.28），
所以 `textworld.envs.pddl.Env.__init__` 里那句 `fast_downward.load_lib()` 必崩。

`env_setup/patch_libdownward.py` 在自己的目录里造了一份降级副本 + 一个符号垫片；
本模块负责在运行时把 `fast_downward` 指过去。**不修改任何外部环境。**

用法：

    from alfworld_compat import install
    install()                      # 必须在 import textworld / alfworld 之前
    import textworld, alfworld     # noqa: E402
"""
from __future__ import annotations

import ctypes
import os
import sys

_REL_DIR = os.path.join("selfplay", "third_party", "glibc232")
_INSTALLED = False


def _candidates() -> list[str]:
    """patched 目录的候选位置：二期/三期家目录 + 环境变量覆盖。"""
    out = []
    env = os.environ.get("ALFWORLD_GLIBC_COMPAT_DIR")
    if env:
        out.append(env)
    # ~/ztwang/selfplay/third_party/glibc232
    out.append(os.path.join(os.path.expanduser("~"), "ztwang", _REL_DIR))
    # 本仓库相对位置（仓库布局：<repo>/selfplay/src/alfworld_compat.py -> <repo>/selfplay/third_party/...）
    here = os.path.dirname(os.path.abspath(__file__))
    out.append(os.path.abspath(os.path.join(here, "..", "third_party", "glibc232")))
    return out


def find_compat_dir() -> str | None:
    for d in _candidates():
        if d and os.path.isfile(os.path.join(d, "libdownward.so")) and os.path.isfile(
            os.path.join(d, "libglibc_single_threaded_shim.so")
        ):
            return d
    return None


def install(verbose: bool = True) -> str:
    """预加载符号垫片并把 fast_downward 指向降级后的 libdownward.so。

    返回实际使用的 compat 目录。可重复调用（幂等）。
    """
    global _INSTALLED
    if _INSTALLED:
        return os.environ.get("ALFWORLD_GLIBC_COMPAT_DIR", "")

    d = find_compat_dir()
    if d is None:
        raise RuntimeError(
            "找不到 patched libdownward.so。请先在集群上执行：\n"
            "    conda activate agentenv-alfworld\n"
            "    python ~/ztwang/selfplay/env_setup/patch_libdownward.py "
            "--out-dir ~/ztwang/selfplay/third_party/glibc232\n"
            f"（已尝试：{_candidates()}）"
        )

    # 1) 用 RTLD_GLOBAL 预加载垫片 -> 后面 dlopen(libdownward.so) 时能解析到
    #    __libc_single_threaded（glibc<2.32 没有这个变量）。无需 LD_PRELOAD。
    shim = os.path.join(d, "libglibc_single_threaded_shim.so")
    ctypes.CDLL(shim, mode=os.RTLD_GLOBAL)

    # 2) 把 fast_downward 的库路径指到降级副本。
    #    load_lib() 每次调用都会读这个模块级全局变量，所以改在这里一定生效。
    import fast_downward.interface as _fi  # noqa: PLC0415

    _fi.DOWNWARD_LIB_PATH = os.path.join(d, "libdownward.so")

    os.environ["ALFWORLD_GLIBC_COMPAT_DIR"] = d
    _INSTALLED = True
    if verbose:
        print(f"[alfworld_compat] libdownward.so -> {_fi.DOWNWARD_LIB_PATH}")
        print(f"[alfworld_compat] shim            -> {shim}")
    return d


def self_check() -> bool:
    """真正 load 一次规划器，确认兼容层可用（不构造环境）。"""
    install(verbose=False)
    import fast_downward  # noqa: PLC0415

    lib = fast_downward.load_lib()
    fast_downward.close_lib(lib)
    return True


if __name__ == "__main__":
    print("compat dir:", find_compat_dir())
    print("load_lib():", "OK" if self_check() else "FAIL")
    sys.exit(0)
