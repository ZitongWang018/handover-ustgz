"""Reviewer checks against public AdamW and an actual tiny Qwen3 MoE."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cpu_adamw import CPUStateAdamW


def test_adamw_matches_reference_with_inactive_parameters_and_resume():
    torch.manual_seed(17)
    actual = [torch.nn.Parameter(torch.randn(3, 4)) for _ in range(2)]
    expected = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    options = dict(lr=0.003, betas=(0.8, 0.97), eps=1e-7, weight_decay=0.04)
    optimizer = CPUStateAdamW(actual, **options)
    reference = torch.optim.AdamW(expected, foreach=False, **options)
    for index in (0, 1, 0, 1, 0):
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        inactive = actual[1-index].detach().clone()
        inactive_state = copy.deepcopy(optimizer.state.get(actual[1-index], {}))
        gradient = torch.randn_like(actual[index])
        actual[index].grad = gradient.clone()
        expected[index].grad = gradient.clone()
        optimizer.step()
        reference.step()
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        assert torch.equal(actual[1-index], inactive)
        for key, value in inactive_state.items():
            assert torch.equal(torch.as_tensor(optimizer.state[actual[1-index]][key]),
                               torch.as_tensor(value))
        assert all(v.device.type == "cpu" for state in optimizer.state.values()
                   for v in state.values() if isinstance(v, torch.Tensor))
    saved = copy.deepcopy(optimizer.state_dict())
    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    restored = CPUStateAdamW(restored_params, lr=1.0)
    restored.load_state_dict(saved)
    for p, q in zip(actual, restored_params):
        p.grad = torch.full_like(p, 0.25)
        q.grad = p.grad.clone()
    optimizer.step()
    restored.step()
    for p, q in zip(actual, restored_params):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    broken = copy.deepcopy(saved)
    first = next(iter(broken["state"].values()))
    first["exp_avg"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        restored.load_state_dict(broken)


def test_tiny_moe_all_linear_adapters_update_with_frozen_base_and_cpu_states():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from ez_train import build_dual_lora
    from ez_dist import select_adapter
    from training_memory import activation_offload_context

    torch.manual_seed(3)
    cfg = Qwen3MoeConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=4, num_experts=2, num_experts_per_tok=1,
        max_position_embeddings=64)
    model = build_dual_lora(Qwen3MoeForCausalLM(cfg), {
        "model": {"lora_r": 2, "lora_alpha": 2, "lora_targets": "all-linear"}})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = CPUStateAdamW(params, lr=1e-3)
    base = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" not in n}
    ids = torch.tensor([[1, 3, 4, 2]])
    for adapter in ("proposer", "solver"):
        select_adapter(model, adapter)
        before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
        with activation_offload_context(True):
            loss = model(input_ids=ids, labels=ids).loss
            assert torch.isfinite(loss)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert any(not torch.equal(p, before[n]) for n, p in model.named_parameters()
                   if "lora_" in n and adapter in n)
        assert all(torch.equal(p, before[n]) for n, p in model.named_parameters()
                   if "lora_" in n and adapter not in n)
        assert all(torch.equal(p, base[n]) for n, p in model.named_parameters() if n in base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA residency check needs GPU")
def test_cuda_parameters_keep_all_moment_states_on_cpu_after_restore():
    param = torch.nn.Parameter(torch.ones(5, device="cuda"))
    optimizer = CPUStateAdamW([param])
    param.grad = torch.ones_like(param)
    optimizer.step()
    restored = CPUStateAdamW([param])
    restored.load_state_dict(optimizer.state_dict())
    assert all(v.device.type == "cpu" for state in restored.state.values()
               for v in state.values() if isinstance(v, torch.Tensor))
