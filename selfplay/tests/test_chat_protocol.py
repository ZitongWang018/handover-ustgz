"""Chat encoding must be identical before training and during evaluation."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def local_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    core = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "<|im_start|>": 2,
        "<|im_end|>": 3, "user": 4, "assistant": 5, "look": 6}, unk_token="[UNK]"))
    core.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]",
        pad_token="[PAD]", eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    tok.chat_template = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + "
        "m['content'] + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
    return tok


def test_actual_local_tokenizer_matches_policy_chat_ids_and_pins_template():
    from protocol import encode_solver_prompt, solver_prompt_identity
    from rollout import HFPolicy, RolloutConfig
    tok = local_tokenizer()
    messages = [{"role": "user", "content": "look"}]
    expected = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    actual = encode_solver_prompt(tok, "look", "chat_template")
    assert actual == expected == HFPolicy(None, tok, device="cpu").initial_prompt_ids(messages, RolloutConfig())
    rendered = tok.decode(actual, skip_special_tokens=False)
    assert rendered.count("<|im_start|>") == 2
    assert rendered.count("<|im_end|>") == 1
    assert "assistant" in rendered
    identity = solver_prompt_identity(tok, "chat_template")
    assert len(identity["chat_template_sha256"]) == 64
    other = copy.deepcopy(tok)
    other.chat_template += " "
    assert solver_prompt_identity(other, "chat_template") != identity


def test_chat_has_no_second_tokenization_or_silent_raw_fallback():
    from protocol import encode_solver_prompt
    class Spy:
        chat_template = "known template"
        def __call__(self, *args, **kwargs):
            raise AssertionError("chat path must not retokenize")
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "body"}]
            assert tokenize is True and add_generation_prompt is True
            return [10, 11, 12]
    assert encode_solver_prompt(Spy(), "body", "chat_template") == [10, 11, 12]
    missing = Spy()
    missing.chat_template = None
    with pytest.raises((ValueError, TypeError)):
        encode_solver_prompt(missing, "body", "chat_template")
    with pytest.raises(ValueError):
        encode_solver_prompt(Spy(), "body", "automatic")


def test_training_rejects_template_drift_before_loading_model_weights(tmp_path, monkeypatch):
    import grpo
    from protocol import solver_prompt_identity
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    tok = local_tokenizer()
    expected = solver_prompt_identity(tok, "chat_template")
    tok.chat_template += " drift"
    monkeypatch.setattr(grpo.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject template before model loading")
    monkeypatch.setattr(grpo.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(grpo.torch.cuda, "is_available", lambda: False)
    cfg = {"model": {"path": str(tmp_path), "dtype": "bfloat16", "attn_implementation": "sdpa"},
        "solver": {"prompt_format": "chat_template"},
        "runtime": {"expected_solver_prompt_identity": expected}}
    with pytest.raises(ValueError, match="template"):
        grpo.build_model_and_tokenizer(cfg, 0, 1)
    cfg["runtime"]["expected_tokenizer_class"] = "OtherTokenizer"
    with pytest.raises(ValueError, match="tokenizer class"):
        grpo.build_model_and_tokenizer(cfg, 0, 1)


def test_hf_evaluation_rejects_multi_gpu_visibility_before_loading_weights(monkeypatch):
    import torch
    import eval_runner
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    with pytest.raises(ValueError, match="one visible GPU"):
        eval_runner._load_policy({"base_path": "must-not-be-loaded",
            "protocol_signature": {"backend": "hf"}, "arm_identity": {"kind": "base-only"}}, None)


def test_checkpoint_chat_identity_and_encoding_sources_are_bound(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import checkpoint
    import eval_runner
    from protocol import solver_prompt_identity

    tok = local_tokenizer()
    identity = solver_prompt_identity(tok, "chat_template")
    solver_config = {"prompt_format": "chat_template"}
    cp = tmp_path / "ckpt"
    leaf = cp / "solver"
    leaf.mkdir(parents=True)
    (leaf / "adapter_config.json").write_text("{}", encoding="utf-8")
    manifest = {"resolved_config": {"solver": solver_config, "runtime": {
        "expected_solver_prompt_identity": identity,
        "expected_tokenizer_class": type(tok).__qualname__}},
        "baseline_binding": {"contract": {"protocol_signature": {
            **identity, "tokenizer_class": type(tok).__qualname__, "software": eval_runner._versions()}}},
        "source_files": {"selfplay/src/" + name: {"sha256": eval_runner.sha256_file(eval_runner.SRC / name)}
            for name in ("protocol.py", "rollout.py", "ez_solver.py")}}
    active = {"manifest": manifest}
    monkeypatch.setattr(checkpoint, "validate_checkpoint_v3", lambda *a, **k: {
        "manifest": active["manifest"], "solver_path": str(leaf)})
    def arm():
        return eval_runner._arm(SimpleNamespace(base_only=False, checkpoint=str(cp)), "base", solver_config,
            prompt_identity=identity, tokenizer_class=type(tok).__qualname__)
    assert arm()["kind"] == "checkpoint"
    for damage in ("source", "template", "software"):
        changed = copy.deepcopy(manifest)
        if damage == "source":
            changed["source_files"]["selfplay/src/ez_solver.py"]["sha256"] = "0" * 64
        elif damage == "template":
            changed["resolved_config"]["runtime"]["expected_solver_prompt_identity"]["chat_template_sha256"] = "0" * 64
            changed["baseline_binding"]["contract"]["protocol_signature"]["chat_template_sha256"] = "0" * 64
        else:
            changed["baseline_binding"]["contract"]["protocol_signature"]["software"]["tokenizers"] = "different"
        active["manifest"] = changed
        with pytest.raises(ValueError, match="identity|tokenization|software"):
            arm()


def test_real_hf_moe_checkpoint_evaluation_loads_solver_not_base(tmp_path, monkeypatch):
    import torch
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from ez_train import build_dual_lora
    from ez_vllm_policy import export_lora_adapter
    from eval_runner import _load_policy
    from peft import PeftModel

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(29)
    config = Qwen3MoeConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=2, num_experts_per_tok=1,
        max_position_embeddings=64)
    base = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    model = build_dual_lora(base, {"model": {
        "lora_r": 2, "lora_alpha": 2, "lora_targets": "all-linear"}})
    model.set_adapter("solver")
    model.eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B.solver" in name:
                parameter.fill_(0.15)
    ids = torch.tensor([[1, 4, 3]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits
    assert not torch.equal(expected, base_logits)
    solver_dir = export_lora_adapter(model, "solver", str(tmp_path / "adapter"))
    contract = {"base_path": str(base_dir), "protocol_signature": {"backend": "hf"},
        "arm_identity": {"kind": "checkpoint", "solver_path": solver_dir}}
    policy = _load_policy(contract, local_tokenizer())
    assert isinstance(policy.model, PeftModel)
    assert policy.model.training is False
    with torch.no_grad():
        actual = policy.model(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
