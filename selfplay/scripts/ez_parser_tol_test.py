#!/usr/bin/env python
"""容错解析器 `parse_tool_call` 的回归单测（纯 CPU，无需 GPU）。

为什么值得一个独立测试文件
---------------------------------------------------------------------------
实测发现（2026-10-14）：基座模型在评测里 **53.1% 的回合被判 malformed**，
这些输出被 `_rollout_batch` / `run_ez_solver_rollouts` **静默换成 `look`**。
把 7550 条原始文本落盘后分类，失败模式极其集中：

| 失败模式 | 占比 |
|---|---|
| 扁平写法：缺 `arguments` 包裹层 | 62.0% |
| 内层用 `name`/`action` 当动作类型 | 14.6% |
| JSON 非法 / 截断 | 11.8% |
| 内层没有动作字段 | 6.1% |
| 非 JSON（散文 / 空 `<tool_call>`） | 3.0% |
| **只差最后一个 `}`** | 2.5% |

也就是说 **79.1% 是"语义完全正确、JSON 包裹层写歪"** —— 模型知道该做什么，
是我们的解析器把回答扔了。这个测试把每一条修好的写法都钉死，
以后再有人"优化" `parse_tool_call` 时不会把这些救回来的路径再弄丢。

用法：python scripts/ez_parser_tol_test.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from ez_proposer import parse_tool_call  # noqa: E402

# (说明, 原始输出, 期望 kind, 期望 action) —— 期望 action 为 None 表示不校验
CASES = [
    # ===== 必须救回来的（这就是 79.1% 的那些）=====
    ("扁平写法:缺 arguments 包裹层",
     '{"name": "go_to", "receptacle_name": "cabinet 2"}', "action", "go to cabinet 2"),
    ("内层用 name 当动作类型",
     '{"name":"alfworld_tool","arguments":{"name":"go_to","receptacle_name":"cabinet 1"}}',
     "action", "go to cabinet 1"),
    ("内层用 action 当动作类型",
     '{"name":"alfworld_tool","arguments":{"action":"take","object_name":"apple 1",'
     '"receptacle_name":"fridge 1"}}', "action", None),
    ("只差最后一个右括号",
     '{"name":"alfworld_tool","arguments":{"action_type":"go_to",'
     '"receptacle_name":"cabinet 1"}', "action", "go to cabinet 1"),
    ("动作类型写成 go to（空格别名）",
     '{"name":"alfworld_tool","arguments":{"action_type":"go to",'
     '"receptacle_name":"desk 1"}}', "action", "go to desk 1"),
    ("动作类型写成 goto（无下划线别名）",
     '{"action_type":"goto","receptacle_name":"table 1"}', "action", "go to table 1"),
    ("带 <think> 前缀 + 裸 JSON",
     '<think></think> I should look around.\n'
     '{"name":"alfworld_tool","arguments":{"action_type":"look"}}', "action", "look"),
    ("标准写法（本来就该过）",
     '{"name":"alfworld_tool","arguments":{"action_type":"go_to",'
     '"receptacle_name":"desk 1"}}', "action", "go to desk 1"),
    ("扁平 + 别名 + 缺右括号（组合）",
     '{"name": "go to", "receptacle_name": "shelf 1"', "action", "go to shelf 1"),
    # ===== 控制类 =====
    ("finish 控制调用",
     '{"name":"alfworld_tool","arguments":{"action_type":"finish",'
     '"task_description":"Put a hot apple in the fridge."}}', "finish", None),
    # ===== 必须仍然判 malformed（不能误伤）=====
    ("纯散文 + 空 tool_call", "1. Look around.\n2. Take the potato.\n<tool_call></tool_call>",
     "malformed", None),
    ("完全没有动作", "I am thinking about the room and its contents.", "malformed", None),
    ("动作类型不存在", '{"action_type":"teleport","receptacle_name":"moon 1"}',
     "malformed", None),
    ("模板参数缺失（go_to 没有 receptacle_name）",
     '{"action_type":"go_to"}', "malformed", None),
]


def run(tag: str) -> int:
    bad = 0
    print(f"== {tag} ==")
    for name, text, want_kind, want_action in CASES:
        p = parse_tool_call(text)
        ok = (p.kind == want_kind) and (want_action is None or p.action == want_action)
        bad += 0 if ok else 1
        got = getattr(p, "action", "")
        print(f"{'  ok ' if ok else 'FAIL '} {name:34s} kind={p.kind:9s} "
              f"action={got!r}" + ("" if ok else f"   期望 kind={want_kind} action={want_action!r}"))
    return bad


def main() -> int:
    os.environ.pop("EZ_PARSER_STRICT", None)
    bad = run("容错模式（默认）")
    # 严格模式：只保留"本来就该过"的那几条 + 控制类
    os.environ["EZ_PARSER_STRICT"] = "1"
    strict_expected_ok = {"标准写法（本来就该过）", "finish 控制调用"}
    sbad = 0
    print("== 严格模式 EZ_PARSER_STRICT=1（对照，证明开关有效）==")
    for name, text, want_kind, _ in CASES:
        p = parse_tool_call(text)
        if name in strict_expected_ok:
            ok = p.kind == want_kind
        else:
            ok = True   # 严格模式下这些救不回来是预期的，不判错
        sbad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'ERR'} {name:34s} kind={p.kind}")
    os.environ.pop("EZ_PARSER_STRICT", None)
    print(f"\ncases={len(CASES)} failures={bad} strict_errors={sbad}")
    print("PARSER-TOL PASS：容错路径全部救回、且没有误伤" if bad == 0
          else "PARSER-TOL FAIL")
    return 1 if (bad or sbad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
