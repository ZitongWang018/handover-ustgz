"""Scoped tau2 routing: local Qwen agent plus DeepSeek user/NL judge."""

from __future__ import annotations

import contextlib
import os
import re
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit

MODEL = "deepseek/deepseek-flash"
DEEPSEEK_API_BASE = "https://api.deepseek.com"
_SECRET_KEYS = {"api_key", "authorization", "password"}
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")

USER_ARGS: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 2048,
    "num_retries": 0,
    "timeout": 600,
    "api_base": DEEPSEEK_API_BASE,
    "extra_body": {"thinking": {"type": "disabled"}},
}


def user_args() -> dict[str, Any]:
    return {**USER_ARGS, "extra_body": {"thinking": {"type": "disabled"}}}


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url or any(c.isspace() for c in base_url):
        raise ValueError("base_url must be non-empty without whitespace")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("base_url requires an http(s) scheme and host")
    if "?" in base_url or "#" in base_url or "@" in parts.netloc:
        raise ValueError("base_url must not contain query, fragment, or userinfo")
    try:
        parts.port
    except ValueError as exc:
        raise ValueError("base_url has an invalid port") from exc
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def redact(value: Any) -> Any:
    secret = os.environ.get("DEEPSEEK_API_KEY", "")

    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                k: "[REDACTED]" if str(k).lower() in _SECRET_KEYS else scrub(v)
                for k, v in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [scrub(v) for v in item]
        if isinstance(item, str):
            if secret:
                item = item.replace(secret, "[REDACTED]")
            return _SECRET_PATTERN.sub("[REDACTED]", item)
        return item

    return scrub(value)


@contextlib.contextmanager
def provider_context(base_url: str, served_model: str):
    endpoint = _normalize_base_url(base_url)
    if not isinstance(served_model, str) or not served_model.strip():
        raise ValueError("served_model must be non-empty")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("DEEPSEEK_API_KEY must be set and non-empty")

    import tau2.config as config
    import tau2.evaluator.evaluator_nl_assertions as nl_assertions
    import tau2.utils.llm_utils as llm_utils

    judge_args = user_args()
    judge_args["response_format"] = {"type": "json_object"}
    names = ("DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_NL_ASSERTIONS_ARGS")
    saved = [(m, n, getattr(m, n)) for m in (config, nl_assertions) for n in names]
    original_completion = llm_utils.completion
    counts: dict[str, int] = {}
    lock = threading.Lock()

    def guarded_completion(*args: Any, **kwargs: Any) -> Any:
        requested = kwargs["model"] if "model" in kwargs else (args[0] if args else None)
        call_kwargs = kwargs
        if requested == MODEL:
            extra = kwargs.get("extra_body")
            thinking = extra.get("thinking") if isinstance(extra, dict) else None
            if "api_key" in kwargs or kwargs.get("api_base") != DEEPSEEK_API_BASE:
                raise ValueError("DeepSeek calls require env auth and the exact api_base")
            if not isinstance(thinking, dict) or thinking.get("type") != "disabled":
                raise ValueError("DeepSeek calls must keep thinking disabled")
        elif requested == f"openai/{served_model}":
            if (
                kwargs.get("api_key", "dummy") != "dummy"
                or kwargs.get("api_base") != endpoint
            ):
                raise ValueError("local calls require a dummy key and normalized api_base")
            if kwargs.get("top_k", 20) != 20:
                raise ValueError("local calls require top_k=20")
            extra = kwargs.get("extra_body", {})
            if not isinstance(extra, dict):
                raise ValueError("local extra_body must be a mapping")
            call_kwargs = {k: v for k, v in kwargs.items() if k != "top_k"}
            call_kwargs["extra_body"] = {**extra, "top_k": 20}
        else:
            raise ValueError(f"unroutable tau2 model: {redact(requested)}")
        try:
            response = original_completion(*args, **call_kwargs)
        except Exception as exc:
            # Native tau2 persists exception messages in its checkpoints/logs.
            raise RuntimeError(redact(f"{type(exc).__name__}: {exc}")) from None
        returned = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
        key = f"{requested} -> {returned or 'unreported'}"
        with lock:
            counts[key] = counts.get(key, 0) + 1
        return response

    try:
        for module in (config, nl_assertions):
            module.DEFAULT_LLM_NL_ASSERTIONS = MODEL
            module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
        llm_utils.completion = guarded_completion
        yield counts
    finally:
        llm_utils.completion = original_completion
        for module, name, old in saved:
            setattr(module, name, old)
