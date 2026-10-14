"""让 `third_party/py310deps` 里的包可用，**但不遮蔽 site-packages 里的 numpy/torch**。

为什么需要（2026-10-15 实测）：
    `third_party/py310deps/` 里含一个 **numpy 源码目录**。如果按常规把它放进
    `PYTHONPATH`，由于 PYTHONPATH 永远排在 site-packages **之前**，numpy 会从这个
    源码目录被导入，直接抛：
        ImportError: Error importing numpy: you should not try to import numpy
                     from its source directory
    （job 13049947 / 13049948 就是这样秒失败的。）

    sitecustomize 由 `site.py` 在 **site-packages 已经就位之后** 导入，所以这里用
    `sys.path.append`（追加到**末尾**）就得到正确的优先级：
        site-packages 的 numpy/torch 优先 ✅，py310deps 的 textworld/alfworld 仍可用 ✅
"""
import os
import sys

_root = os.environ.get("SKR_PY310DEPS") or os.path.join(
    os.path.expanduser("~"), "ztwang", "selfplay", "third_party", "py310deps")
if os.path.isdir(_root) and _root not in sys.path:
    sys.path.append(_root)      # 必须 append，不能 insert(0, ...)
