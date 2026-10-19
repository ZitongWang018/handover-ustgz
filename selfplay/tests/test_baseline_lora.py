from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeOutput:
    token_ids = [ord(char) for char in "<action>look</action>ignored"]
    finish_reason = "stop"


class _FakeRequest:
    outputs = [_FakeOutput()]


class _FakeLLM:
    def __init__(self, **kwargs):
        self.engine_kwargs = kwargs
        self.generation_calls = []

    def generate(self, prompts, sampling_params, **kwargs):
        self.generation_calls.append((prompts, sampling_params, kwargs))
        return [_FakeRequest() for _ in prompts]


class _FakeLoRARequest:
    def __init__(self, name, version, path):
        self.name, self.version, self.path = name, version, path


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 999
    pad_token = "<pad>"
    eos_token = "<eos>"

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(token) for token in ids if token != self.pad_token_id)


@pytest.fixture
def fake_vllm(monkeypatch):
    package = ModuleType("vllm")
    package.__path__ = []
    package.LLM = _FakeLLM
    package.SamplingParams = _FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.delitem(sys.modules, "vllm.lora", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.lora.request", raising=False)
    return package


def _install_lora_request(monkeypatch):
    request_module = ModuleType("vllm.lora.request")
    request_module.LoRARequest = _FakeLoRARequest
    lora_module = ModuleType("vllm.lora")
    lora_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_module)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_module)


def _contract(kind, *, backend="vllm"):
    arm = {"kind": kind}
    if kind == "checkpoint":
        arm["solver_path"] = "/checkpoints/solver/adapter"
    return {
        "base_path": "/models/base",
        "arm_identity": arm,
        "protocol_signature": {
            "backend": backend,
            "gpu_memory_utilization": 0.37,
            "max_context_tokens": 32768,
            "batch_size": 7,
        },
    }


def test_factory_default_preserves_training_lora_and_false_omits_lora_budgets(fake_vllm):
    from ez_vllm_policy import build_vllm_engine

    default_engine = build_vllm_engine(
        "/models/base", gpu_memory_utilization=0.37, max_model_len=32768,
        max_num_seqs=7, max_num_batched_tokens=4096, seed=321)
    assert default_engine.engine_kwargs["enable_lora"] is True
    assert default_engine.engine_kwargs["max_lora_rank"] == 32
    assert default_engine.engine_kwargs["max_loras"] == 2
    assert default_engine.engine_kwargs["max_cpu_loras"] == 4

    base_engine = build_vllm_engine(
        "/models/base", gpu_memory_utilization=0.37, max_model_len=32768,
        max_num_seqs=7, max_num_batched_tokens=4096, seed=321,
        enable_lora=False)
    assert base_engine.engine_kwargs["enable_lora"] is False
    assert not ({"max_lora_rank", "max_loras", "max_cpu_loras"}
                & base_engine.engine_kwargs.keys())
    assert base_engine.engine_kwargs["gpu_memory_utilization"] == 0.37
    assert base_engine.engine_kwargs["max_model_len"] == 32768
    assert base_engine.engine_kwargs["max_num_seqs"] == 7
    assert base_engine.engine_kwargs["max_num_batched_tokens"] == 4096
    assert base_engine.engine_kwargs["seed"] == 321


def test_base_only_loads_non_lora_engine_and_generates_action_tag(fake_vllm):
    import eval_runner
    from rollout import RolloutConfig

    policy = eval_runner._load_policy(_contract("base-only"), _Tokenizer())
    assert policy.llm.engine_kwargs["enable_lora"] is False
    assert not ({"max_lora_rank", "max_loras", "max_cpu_loras"}
                & policy.llm.engine_kwargs.keys())

    text, token_ids, stop_lengths = policy.generate_from_token_ids(
        [[42]], RolloutConfig(stop_mode="action_tag", max_context_tokens=32768,
                              max_new_tokens=64))
    assert text == ["<action>look</action>"]
    assert token_ids == [[ord(char) for char in "<action>look</action>"]]
    assert stop_lengths == [len(token_ids[0])]
    prompts, sampling, generate_kwargs = policy.llm.generation_calls[0]
    assert prompts == [{"prompt_token_ids": [42]}]
    assert sampling.kwargs["stop"] == ["</action>"]
    assert sampling.kwargs["include_stop_str_in_output"] is True
    assert generate_kwargs["lora_request"] is None
    assert generate_kwargs["use_tqdm"] is False


def test_checkpoint_loads_lora_engine_and_mounts_solver_adapter(fake_vllm, monkeypatch, tmp_path):
    import eval_runner

    _install_lora_request(monkeypatch)
    (tmp_path / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    contract = _contract("checkpoint")
    contract["base_path"] = str(tmp_path)
    policy = eval_runner._load_policy(contract, _Tokenizer())
    assert policy.llm.engine_kwargs["enable_lora"] is True
    assert policy.llm.engine_kwargs["max_lora_rank"] == 32
    assert policy._lora_req.name == "solver"
    assert policy._lora_req.version == 1
    assert policy._lora_req.path == "/checkpoints/solver/adapter"


def test_base_succeeds_but_checkpoint_lora_failure_does_not_fallback(fake_vllm, tmp_path):
    import eval_runner

    attempts = []

    def moe_lora_blocking_llm(**kwargs):
        attempts.append(kwargs["enable_lora"])
        if kwargs["enable_lora"]:
            raise ValueError("unsupported MoE+LoRA engine setup")
        return _FakeLLM(**kwargs)

    fake_vllm.LLM = moe_lora_blocking_llm
    base_policy = eval_runner._load_policy(_contract("base-only"), _Tokenizer())
    assert base_policy.llm.engine_kwargs["enable_lora"] is False
    (tmp_path / "config.json").write_text('{"model_type": "qwen2_moe"}', encoding="utf-8")
    contract = _contract("checkpoint")
    contract["base_path"] = str(tmp_path)
    with pytest.raises(ValueError, match="unsupported MoE.*LoRA engine setup"):
        eval_runner._load_policy(contract, _Tokenizer())
    assert attempts == [False, True]


def test_unreadable_architecture_fails_before_engine(fake_vllm, tmp_path):
    import eval_runner
    attempts = []
    fake_vllm.LLM = lambda **kwargs: attempts.append(kwargs)
    contract = _contract("checkpoint")
    contract["base_path"] = str(tmp_path)
    with pytest.raises(ValueError, match="cannot inspect local model config"):
        eval_runner._load_policy(contract, _Tokenizer())
    assert attempts == []


def test_unknown_arm_kind_fails_closed():
    import eval_runner

    with pytest.raises(ValueError, match="unsupported evaluation arm kind"):
        eval_runner._load_policy(_contract("mystery"), _Tokenizer())
