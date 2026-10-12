"""评测用的策略实现（都满足 `run_rollouts` 需要的 `generate(messages_list, cfg)` 接口）。

* `ChatAPIPolicy`  —— 走 OpenAI 兼容 HTTP API（默认 DeepSeek-V4-Pro），当作 Qwen3-4B 的参照基线。
* `RandomPolicy`   —— 从 admissible 里均匀随机挑，用来给成功率钉一个下限。
* `EchoPolicy`     —— 只回 `look`，用于最朴素的连通性自检。
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_client import ChatClient
from prompts import build_system_prompt

_ADM_BLOCK_RE = re.compile(r"Admissible Commands:\n((?:- .*\n?)+)")


def parse_admissible_from_messages(messages: Sequence[Dict[str, str]]) -> List[str]:
    """从最近一条 user 消息里把 `Admissible Commands` 列表抠出来。"""
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        m = _ADM_BLOCK_RE.search(msg["content"])
        if m:
            return [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")]
    return []


class ChatAPIPolicy:
    """把 HTTP chat API 包成策略。并发发请求，保持环境步进的同步语义。"""

    def __init__(
        self,
        client: Optional[ChatClient] = None,
        model: str = "deepseek-v4-pro",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        effort: Optional[str] = "max",
        concurrency: int = 8,
        system_prompt: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.client = client or ChatClient(model=model)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self.concurrency = max(1, concurrency)
        self.system_prompt = system_prompt or build_system_prompt()
        self.extra_body = dict(extra_body or {})
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}

    def _one(self, messages: Sequence[Dict[str, str]]) -> str:
        msgs = list(messages)
        if msgs and msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs
        res = self.client.chat(
            msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model=self.model,
            effort=self.effort,
            extra_body=self.extra_body or None,
        )
        self.usage["prompt_tokens"] += res.prompt_tokens
        self.usage["completion_tokens"] += res.completion_tokens
        self.usage["calls"] += 1
        return res.text

    def generate(self, messages_list: Sequence[Sequence[Dict[str, str]]], cfg=None):
        if self.concurrency == 1 or len(messages_list) == 1:
            texts = [self._one(m) for m in messages_list]
        else:
            with ThreadPoolExecutor(max_workers=min(self.concurrency, len(messages_list))) as ex:
                texts = list(ex.map(self._one, messages_list))
        return texts, [[] for _ in texts], [0] * len(texts)


class RandomPolicy:
    def __init__(self, seed: int = 0, include_look_bias: float = 0.3):
        self.rng = random.Random(seed)
        self.include_look_bias = include_look_bias

    def generate(self, messages_list: Sequence[Sequence[Dict[str, str]]], cfg=None):
        texts = []
        for msgs in messages_list:
            adm = parse_admissible_from_messages(msgs)
            if not adm:
                texts.append("Thought: nothing to do.\nAction: look")
                continue
            if self.rng.random() < self.include_look_bias and "look" in adm:
                action = "look"
            else:
                action = self.rng.choice(adm)
            texts.append(f"Thought: pick a random admissible command.\nAction: {action}")
        return texts, [[] for _ in texts], [0] * len(texts)


class EchoPolicy:
    """永远输出 `look`，用来验证整条链路（环境/解析/奖励）是否通。"""

    def generate(self, messages_list, cfg=None):
        texts = ["Thought: I will look around.\nAction: look" for _ in messages_list]
        return texts, [[] for _ in texts], [0] * len(texts)


class ScriptedOraclePolicy:
    """按 admissible 里的确定性启发式挑动作（不调模型），用来量"环境本身可解性"的上限。

    规则很简单但有效：优先 take / heat / cool / clean / use / put，其次 go to，最后 look。
    这不是 ALFWorld 的官方 expert，只是给评测一个"确定性非学习"参考点。
    """

    _PRIORITY = ["put", "take", "heat", "cool", "clean", "use", "open", "go to", "look", "inventory"]

    def generate(self, messages_list, cfg=None):
        texts = []
        for msgs in messages_list:
            adm = parse_admissible_from_messages(msgs)
            action = "look"
            for kw in self._PRIORITY:
                cand = [a for a in adm if a.lower().startswith(kw)]
                if cand:
                    action = cand[0]
                    break
            texts.append(f"Thought: follow the scripted priority order.\nAction: {action}")
        return texts, [[] for _ in texts], [0] * len(texts)
