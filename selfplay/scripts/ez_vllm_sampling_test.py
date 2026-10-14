#!/usr/bin/env python
"""P1 验收门 #1b（纯 CPU）：HF 的"无害写法"必须被翻译成 vLLM 的合法写法。

为什么单独给这个写测试
---------------------------------------------------------------------------
`rollout._generate_batch` 是"**只在参数 > 0 时才生效**"：

    if cfg.top_k and cfg.top_k > 0: gen_kwargs["top_k"] = cfg.top_k

所以 `top_k=0` 在 HF 路径里等于"不启用"，**完全无害**，写的人也不会觉得有问题。
但 vLLM 把 0 当非法值直接抛：

    ValueError: top_k must be -1 (disable), or at least 1, got 0.

这个坑真实发生过一次（job 13026569）：`ez_train.mid_eval_solver` 里的 `top_k=0`
让 rank0 在**评测**时崩掉 ⇒ 其余 7 个 rank 卡在 NCCL 集合通信上等超时 ⇒
**outer 0 已跑完的 50 步 + 1.4 小时全部报废**。

所以这里把所有"HF 无害 / vLLM 非法"的组合钉死成测试。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ez_vllm_policy import normalize_sampling  # noqa: E402
from rollout import RolloutConfig  # noqa: E402

# (说明, cfg 覆盖项, 期望的 vLLM 取值)
CASES = [
    ("默认（top_k=-1 禁用）", {}, {"top_k": -1, "top_p": 0.95, "min_p": 0.0}),
    ("top_k=0 ← 就是崩过的那一种", {"top_k": 0}, {"top_k": -1}),
    ("top_k=-5", {"top_k": -5}, {"top_k": -1}),
    ("top_k=50（正常启用）", {"top_k": 50}, {"top_k": 50}),
    ("top_k=1（边界，必须保留）", {"top_k": 1}, {"top_k": 1}),
    ("top_p=0（HF 会关掉，vLLM 不接受 0）", {"top_p": 0.0}, {"top_p": 1e-6}),
    ("top_p=1.0（HF 不启用）", {"top_p": 1.0}, {"top_p": 1.0}),
    ("top_p>1（非法）", {"top_p": 1.5}, {"top_p": 1.0}),
    ("min_p=0（HF 不启用）", {"min_p": 0.0}, {"min_p": 0.0}),
    ("min_p<0（非法）", {"min_p": -0.3}, {"min_p": 0.0}),
    ("min_p=1.5（非法）", {"min_p": 1.5}, {"min_p": 1.0}),
    ("temperature=0（HF 走 greedy）", {"temperature": 0.0}, {"temperature": 1e-5}),
    ("repetition_penalty=0（非法）", {"repetition_penalty": 0.0},
     {"repetition_penalty": 1e-5}),
]


def main() -> int:
    bad = 0
    for name, over, want in CASES:
        cfg = RolloutConfig(**over)
        got = normalize_sampling(cfg)
        ok = all(abs(got[k] - v) < 1e-12 for k, v in want.items())
        bad += 0 if ok else 1
        print(f"{'  ok ' if ok else 'FAIL '} {name:34s} "
              f"-> {({k: got[k] for k in want})}"
              + ("" if ok else f"   期望 {want}"))
    # 兜底断言不能自己误伤：把每个输出再喂一遍必须仍然合法
    for name, over, _ in CASES:
        n = normalize_sampling(RolloutConfig(**over))
        assert n["top_k"] == -1 or n["top_k"] >= 1, name
        assert 0.0 < n["temperature"], name
        assert 0.0 < n["repetition_penalty"], name
        assert 0.0 < n["top_p"] <= 1.0, name
        assert 0.0 <= n["min_p"] <= 1.0, name
    print(f"  ok  {len(CASES)} 组互不越界")
    print(f"cases={len(CASES)} failures={bad}")
    print("P1-GATE-1b " + ("PASS：采样参数归一化正确" if bad == 0 else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
