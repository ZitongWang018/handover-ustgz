"""Exercise the actual SPADE config/aggregation with offline tau2 transports."""
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

import tau2_eval as backend
import tau2_provider as provider
import spade.core.eval.tau2_evaluator as te

BASE = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_tau2(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-unit-test-credential")
    modules = {}
    for name in ("tau2", "tau2.config", "tau2.evaluator", "tau2.evaluator.evaluator_nl_assertions",
                 "tau2.utils", "tau2.utils.llm_utils"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__file__ = str(BASE / "tau2-bench/src" / name.replace(".", "/") / "__init__.py")
        modules[name] = mod
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, mod)
    for name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        modules[name].DEFAULT_LLM_NL_ASSERTIONS = "original-judge"
        modules[name].DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0}
    transport_calls = []
    def completion(*args, **kwargs):
        transport_calls.append(kwargs)
        return types.SimpleNamespace(model="reported-" + kwargs["model"])
    modules["tau2.utils.llm_utils"].completion = completion
    monkeypatch.setattr(te, "_TAU2_AVAILABLE", True)
    monkeypatch.setattr(te, "EvaluationType", types.SimpleNamespace(ALL_WITH_NL_ASSERTIONS="all-with-nl"))
    monkeypatch.setattr(te, "get_tasks", lambda **kw: [types.SimpleNamespace(id=f"{kw['task_set_name']}-{i}")
                                                     for i in range(1 if kw.get("num_tasks") else 2)])
    run_calls = []
    state = {"damage": None, "modules": modules, "transport_calls": transport_calls,
             "run_calls": run_calls, "original_completion": completion}
    def run_tasks(**kwargs):
        run_calls.append(kwargs)
        llm = modules["tau2.utils.llm_utils"]
        llm.completion(model=kwargs["llm_agent"], **kwargs["llm_args_agent"])
        llm.completion(model=kwargs["llm_user"], **kwargs["llm_args_user"])
        judge = modules["tau2.evaluator.evaluator_nl_assertions"]
        llm.completion(model=judge.DEFAULT_LLM_NL_ASSERTIONS, **judge.DEFAULT_LLM_NL_ASSERTIONS_ARGS)
        sims = [types.SimpleNamespace(id=f"{task.id}-{trial}", task_id=task.id, trial=trial,
                termination_reason="agent_stop", ticks=None,
                reward_info=types.SimpleNamespace(reward=1.0 if trial % 2 == 0 else 0.0),
                messages=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "done"}])
                for task in kwargs["tasks"] for trial in range(kwargs["num_trials"])]
        if state["damage"] == "missing":
            sims.pop()
        elif state["damage"] == "duplicate":
            sims[-1] = sims[0]
        elif state["damage"] == "error":
            sims[0].termination_reason = "infrastructure_error"
        elif state["damage"] == "empty_trace":
            sims[0].messages = []
        elif state["damage"] == "invalid_reward":
            sims[0].reward_info.reward = float("nan")
        Path(kwargs["save_to"]).write_text(json.dumps({
            "llm_args_user": kwargs["llm_args_user"],
            "simulations": [{"task_id": s.task_id, "trial": s.trial, "messages": s.messages} for s in sims]}),
            encoding="utf-8")
        return types.SimpleNamespace(simulations=sims)
    monkeypatch.setattr(te, "run_tasks", run_tasks)
    return state


def test_config_changes_only_the_user_model_from_official_recipe():
    original = yaml.safe_load((BASE / "upstream/eval_offline/configs/_tool_use_tau2.yaml").read_text(encoding="utf-8"))
    local = yaml.safe_load((BASE / "tau2_config.yaml").read_text(encoding="utf-8"))
    original["tau2_eval"]["defaults"]["user_llm"] = provider.MODEL
    assert local == original


@pytest.mark.parametrize("smoke", [False, True])
def test_success_uses_real_spade_aggregator_and_all_three_roles(tmp_path, fake_tau2, smoke):
    result = backend.evaluate("http://127.0.0.1:30000/v1", "qwen", tmp_path, BASE / "tau2_config.yaml", smoke=smoke)
    assert result["complete"] is True
    assert result["scope"] == ("smoke" if smoke else "formal")
    assert result["aggregate"]["total_simulations"] == (3 if smoke else 24)
    assert result["aggregate"]["overall_pass_at_1"] == (1.0 if smoke else 0.5)
    for call in fake_tau2["run_calls"]:
        assert call["llm_agent"] == "openai/qwen"
        assert call["llm_args_agent"]["api_base"] == "http://127.0.0.1:30000/v1"
        assert call["llm_args_agent"]["extra_body"]["top_k"] == 20
        assert call["llm_user"] == provider.MODEL
        assert "api_key" not in call["llm_args_user"]
        assert call["seed"] == 300 and call["max_retries"] == 0
        assert call["num_trials"] == (1 if smoke else 4)
        assert call["evaluation_type"] == "all-with-nl"
    assert fake_tau2["modules"]["tau2.config"].DEFAULT_LLM_NL_ASSERTIONS == "original-judge"
    assert fake_tau2["modules"]["tau2.utils.llm_utils"].completion is fake_tau2["original_completion"]
    assert len(result["provider_observed_counts"]) == 2
    assert all("private-unit-test-credential" not in p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    with pytest.raises(FileExistsError):
        backend.evaluate("http://127.0.0.1:30000", "qwen", tmp_path, BASE / "tau2_config.yaml")


@pytest.mark.parametrize("damage", ["missing", "duplicate", "error", "empty_trace", "invalid_reward"])
def test_invalid_or_partial_run_is_not_a_valid_score(tmp_path, fake_tau2, damage):
    fake_tau2["damage"] = damage
    with pytest.raises(RuntimeError):
        backend.evaluate("http://127.0.0.1:30000", "qwen", tmp_path, BASE / "tau2_config.yaml")
    result = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert result["complete"] is False and result["error"]
    assert len(fake_tau2["run_calls"]) == 1
    assert (tmp_path / "retail/raw.json").is_file()
    assert not (tmp_path / "telecom").exists()


def test_provider_blocks_gpt_and_secret_overrides_and_restores_state(fake_tau2):
    utils = fake_tau2["modules"]["tau2.utils.llm_utils"]
    with provider.provider_context("http://127.0.0.1:30000", "qwen"):
        with pytest.raises(ValueError, match="unroutable"):
            utils.completion(model="openai/gpt-4.1")
        with pytest.raises(ValueError, match="env auth"):
            utils.completion(model=provider.MODEL, api_key="secret", **provider.user_args())
    assert fake_tau2["transport_calls"] == []
    assert utils.completion is fake_tau2["original_completion"]


def test_missing_key_is_fatal_not_skipped(tmp_path, fake_tau2, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        backend.evaluate("http://127.0.0.1:30000", "qwen", tmp_path, BASE / "tau2_config.yaml")
    assert not fake_tau2["run_calls"]
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["complete"] is False


def test_provider_scrubs_transport_errors_before_native_checkpoint(fake_tau2):
    utils = fake_tau2["modules"]["tau2.utils.llm_utils"]
    def failing(**kwargs):
        raise ValueError("transport leaked private-unit-test-credential")
    utils.completion = failing
    with provider.provider_context("http://127.0.0.1:30000", "qwen"):
        with pytest.raises(RuntimeError, match="REDACTED") as caught:
            utils.completion(model=provider.MODEL, **provider.user_args())
        assert "private-unit-test-credential" not in str(caught.value)
    assert utils.completion is failing


def test_official_aggregate_error_cannot_be_marked_complete(tmp_path, fake_tau2, monkeypatch):
    original = te.Tau2Evaluator._aggregate_spec_result
    def damaged(self, *args):
        result = original(self, *args)
        result.errors = 1
        return result
    monkeypatch.setattr(te.Tau2Evaluator, "_aggregate_spec_result", damaged)
    with pytest.raises(RuntimeError, match="official aggregate"):
        backend.evaluate("http://127.0.0.1:30000", "qwen", tmp_path, BASE / "tau2_config.yaml")
    assert not json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["complete"]
