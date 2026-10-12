"""OpenAI 兼容的 LLM HTTP 客户端（默认走 DeepSeek）。

用途
----
1. **评测基线**：让 DeepSeek 直接玩 ALFWorld，作为 Qwen3-4B 的参照（`eval_alfworld.py --agent deepseek`）。
2. **可选裁判**：给轨迹打质量分 / 生成 hint（走 `judge()`）。

只用标准库（urllib），不引入额外依赖。
API key 读取顺序：`DEEPSEEK_API_KEY` 环境变量 -> `configs/secrets.env`。

⚠️ 三期 ACD 集群**完全无外网**，这个客户端只能在二期或本机跑。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://api.deepseek.com"
# 实测（2026-10-12）该端点的模型列表为 ['deepseek-flash', 'deepseek-v4-pro']。
# 用户要求"用最强的模型" -> deepseek-v4-pro。
DEFAULT_MODEL = "deepseek-v4-pro"
# deepseek-v4-pro 支持顶层 effort 参数（low / high / max），默认 high。
# 注意：effort 越高 reasoning token 越多，max_tokens 给小了会返回空 content。
VALID_EFFORTS = ("low", "high", "max")


def load_env_file(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def resolve_api_key(explicit: Optional[str] = None, secrets_path: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("DEEPSEEK_API_KEY"):
        return os.environ["DEEPSEEK_API_KEY"]
    if secrets_path is None:
        secrets_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "secrets.env")
    env = load_env_file(os.path.abspath(secrets_path))
    if env.get("DEEPSEEK_API_KEY"):
        return env["DEEPSEEK_API_KEY"]
    raise RuntimeError(
        "找不到 DeepSeek API key：请设 DEEPSEEK_API_KEY 环境变量，"
        f"或写进 {os.path.abspath(secrets_path)}（该文件已在 .gitignore 里）"
    )


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """最小可用的 OpenAI 兼容 chat 客户端（带重试）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        max_retries: int = 4,
        secrets_path: Optional[str] = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key, secrets_path)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ http

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:  # noqa: PERF203
                body = e.read().decode("utf-8", "replace")
                last_err = RuntimeError(f"HTTP {e.code}: {body[:400]}")
                if e.code in (400, 401, 403, 404):
                    raise last_err
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"请求 {url} 失败：{last_err}")

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.api_key}"}, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    # ------------------------------------------------------------------ api

    def list_models(self) -> List[str]:
        try:
            data = self._get("/models")
            return [m.get("id", "") for m in data.get("data", [])]
        except Exception as e:  # noqa: BLE001
            return [f"<list_models failed: {e}>"]

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
        model: Optional[str] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[dict] = None,
        effort: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if response_format:
            payload["response_format"] = response_format
        # DeepSeek V4 的推理强度开关（顶层参数，不是 extra_body 里的）。
        if effort is not None:
            eff = str(effort).lower()
            if eff not in VALID_EFFORTS:
                raise ValueError(f"effort 必须是 {VALID_EFFORTS} 之一，收到 {effort!r}")
            payload["effort"] = eff
        if extra_body:
            payload.update(extra_body)
        data = self._post("/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        usage = data.get("usage", {}) or {}
        return ChatResult(
            text=(msg.get("content") or "").strip(),
            model=data.get("model", payload["model"]),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            reasoning=msg.get("reasoning_content", "") or "",
            raw=data,
        )

    def judge(self, prompt: str, **kwargs: Any) -> str:
        """要求模型返回 JSON（用于打分/裁判）。"""
        res = self.chat(
            [{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **kwargs,
        )
        return res.text


if __name__ == "__main__":
    c = ChatClient()
    print("models:", c.list_models())
    r = c.chat(
        [{"role": "user", "content": "回答一个词：法国首都？"}],
        max_tokens=512,
        temperature=0,
        effort="max",
    )
    print("reply:", repr(r.text[:120]), "| model:", r.model,
          "| tokens:", r.prompt_tokens, r.completion_tokens)
    if not r.text:
        print("⚠️ content 为空，reasoning 长度 =", len(r.reasoning),
              "-> 需要调大 max_tokens")
