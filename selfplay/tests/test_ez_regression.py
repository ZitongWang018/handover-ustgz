"""Focused regressions for EZ's global grouping, typed deltas, and train math.

Run with the repository's CPU test environment:
    python -m pytest selfplay/tests/test_ez_regression.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ez_dist import choose_pool_indices, global_group_advantages, local_group_size, shard_group
from ez_proposer import parse_tool_call
from ez_state import StateSnapshot, state_delta, delta_covers, snapshot_from_infos
from ez_verify import ProposedTask, validity_filter


def _fact(pred, *args):
    return repr({"predicate": pred, "arguments": [
        {"name": name, "type": typ} for name, typ in args]})


def _snap(raw):
    from ez_state import normalize_fact
    facts = []
    for item in raw:
        import ast
        pred = ast.literal_eval(item)
        args = [a["name"] for a in pred["arguments"]]
        facts.append(f"{pred['predicate']}({', '.join(args)})")
    return StateSnapshot(facts=tuple(normalize_fact(x) for x in facts),
                         raw_facts=tuple(raw), mode="facts")


def test_global_group_advantage_matches_unsharded_group():
    rewards = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.4, 0.8,
               0.1, 0.2, 0.3, 0.7, 0.9, 0.6, 0.2, 0.8]
    full = global_group_advantages(rewards, ddof=1)
    assert local_group_size(16, 8) == 2
    shards = [shard_group(rewards, rank, 8) for rank in range(8)]
    reconstructed = [shards[rank][local] for local in range(2) for rank in range(8)]
    assert reconstructed == rewards
    assert global_group_advantages(reconstructed, ddof=1) == pytest.approx(full)
    with pytest.raises(ValueError):
        local_group_size(15, 8)


def test_pool_sampler_rotates_and_avoids_same_step_duplicates():
    pool, count = 11, 4
    batches = [choose_pool_indices(pool, count, step, seed=13) for step in range(6)]
    assert all(len(x) == len(set(x)) == count for x in batches)
    assert set(i for batch in batches for i in batch) == set(range(pool))
    with pytest.raises(ValueError):
        choose_pool_indices(3, 5, 0, 1)


def test_delta_counter_navigation_and_consistent_entity_renaming():
    s0 = _snap([
        _fact("in", ("apple 1", "o"), ("counter 1", "r")),
        _fact("in", ("apple 2", "o"), ("counter 1", "r")),
        _fact("at", ("player 0", "p"), ("kitchen 1", "r")),
    ])
    ref = _snap([
        _fact("in", ("apple 2", "o"), ("counter 1", "r")),
        _fact("heated", ("apple 1", "o")),
        _fact("in", ("apple 1", "o"), ("fridge 1", "r")),
        _fact("at", ("player 0", "p"), ("livingroom 1", "r")),
    ])
    renamed = _snap([
        _fact("in", ("apple 1", "o"), ("counter 1", "r")),
        _fact("heated", ("apple 2", "o")),
        _fact("in", ("apple 2", "o"), ("fridge 1", "r")),
        _fact("at", ("player 0", "p"), ("bedroom 1", "r")),
    ])
    split = _snap([
        _fact("heated", ("apple 1", "o")),
        _fact("in", ("apple 2", "o"), ("fridge 1", "r")),
        _fact("heated", ("potato 1", "o")),
        _fact("in", ("potato 1", "o"), ("fridge 1", "r")),
    ])
    ref_delta = state_delta(s0, ref)
    assert not any("player" in x for x in ref_delta.added + ref_delta.removed)
    assert delta_covers(ref_delta, state_delta(s0, renamed))
    assert not delta_covers(ref_delta, state_delta(s0, split))


def test_task_scoped_delta_ignores_unmentioned_source_and_container_side_effects():
    s0 = _snap([_fact("inreceptacle", ("apple 1", "o"), ("countertop 1", "r")),
                _fact("atlocation", ("agent", "p"), ("kitchen 1", "r"))])
    ref = _snap([_fact("inreceptacle", ("apple 1", "o"), ("fridge 1", "r")),
                 _fact("ishot", ("apple 1", "o")),
                 _fact("opened", ("cabinet 1", "r")),
                 _fact("atlocation", ("agent", "p"), ("livingroom 1", "r"))])
    candidate = _snap([_fact("inreceptacle", ("apple 2", "o"), ("fridge 1", "r")),
                       _fact("ishot", ("apple 2", "o")),
                       _fact("atlocation", ("agent", "p"), ("bedroom 1", "r"))])
    ref_delta = state_delta(s0, ref, "Put the apple in the fridge")
    assert delta_covers(ref_delta, state_delta(s0, candidate, "Put the apple in the fridge"))

    closed = _snap([_fact("not_opened", ("cabinet 1", "r"))])
    opened = _snap([_fact("opened", ("cabinet 1", "r"))])
    open_delta = state_delta(closed, opened, "Open the cabinet")
    assert open_delta.added
    assert not delta_covers(open_delta, state_delta(closed, closed, "Open the cabinet"))
    opened_fridge = _snap([_fact("not_opened", ("cabinet 1", "r")),
                           _fact("opened", ("fridge 1", "r"))])
    assert state_delta(closed, opened_fridge,
                       "Open the cabinet and put the apple in the fridge").is_empty()


def test_task_scope_accepts_real_textworld_typed_propositions():
    from types import SimpleNamespace

    def prop(pred, *args):
        return SimpleNamespace(name=pred, arguments=[
            SimpleNamespace(name=name, type=SimpleNamespace(name=typ)) for name, typ in args])

    before = snapshot_from_infos({"facts": [
        prop("inreceptacle", ("apple 1", "object"), ("countertop 1", "receptacle")),
        prop("atlocation", ("agent", "agent"), ("kitchen 1", "location")),
        prop("not_opened", ("cabinet 1", "receptacle")),
    ]})
    after = snapshot_from_infos({"facts": [
        prop("inreceptacle", ("apple 2", "object"), ("fridge 1", "receptacle")),
        prop("atlocation", ("agent", "agent"), ("bedroom 1", "location")),
        prop("ishot", ("apple 2", "object")),
        prop("opened", ("cabinet 1", "receptacle")),
    ]})
    delta = state_delta(before, after, "Put the apple in the fridge")
    assert delta.added and any("fridge" in fact for fact in delta.added)
    assert any("ishot" in fact for fact in delta.added)
    assert not any("opened" in fact or "not_opened" in fact
                   for fact in delta.added + delta.removed)
    assert not any("atlocation" in fact for fact in delta.added + delta.removed)


def test_validity_filter_handles_quantifiers_adjectives_and_unsupported_objects():
    snap = StateSnapshot(facts=("in(apple, fridge)", "at(player, kitchen)"),
                         raw_facts=(), mode="facts")
    delta = __import__("ez_state").StateDelta(added=("heated(apple)",))

    def task(description):
        return ProposedTask(description, snap, snap, delta,
                            proposer_actions=["put apple 1 in fridge 1"])

    assert validity_filter(task("Put two apples in a fridge."))[0]
    assert validity_filter(task("Put all apples in the fridge."))[0]
    assert validity_filter(task("Put a hot apple in the fridge."))[0]
    ok, reason = validity_filter(task("Put the melon in the fridge."))
    assert not ok and reason.startswith("unsupported_objects")


def test_declare_is_metadata_not_subtask_and_aliases_parse():
    declared = parse_tool_call('<tool_call>{"arguments":{"action_type":"declare_task",'
                               '"task_description":"heat apple"}}</tool_call>')
    assert declared.kind == "declare"
    sub = parse_tool_call('<tool_call>{"arguments":{"action_type":"subtask",'
                          '"task_description":"heat apple"}}</tool_call>')
    assert sub.kind == "sub_finish"
    put_alias = parse_tool_call('<tool_call>{"arguments":{"action_type":"put",'
                                '"object_name":"apple 1","receptacle_name":"fridge 1"}}</tool_call>')
    move_alias = parse_tool_call('<tool_call>{"arguments":{"action_type":"move",'
                                 '"object_name":"apple 1","receptacle_name":"fridge 1"}}</tool_call>')
    assert put_alias.kind == move_alias.kind == "action"
    assert put_alias.action == move_alias.action == "put apple 1 in/on fridge 1"


def test_selected_logits_and_microbatch_kl_gradient_equivalence():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from train_grpo import BatchTensors, grpo_loss, per_token_logps

    class SelectLM(torch.nn.Module):
        def __init__(self, selected=True):
            super().__init__()
            self.emb = torch.nn.Embedding(19, 7)
            self.head = torch.nn.Linear(7, 19, bias=False)
            self.selected = selected

        def forward(self, input_ids, attention_mask=None, use_cache=False,
                    logits_to_keep=None):
            h = self.emb(input_ids)
            if self.selected and logits_to_keep is not None:
                h = h[:, logits_to_keep, :]
            return SimpleNamespace(logits=self.head(h))

    torch.manual_seed(7)
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    mask = torch.tensor([[0, 1, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32)
    a, b = SelectLM(True), SelectLM(False)
    b.load_state_dict(a.state_dict())
    sparse = per_token_logps(a, ids, attn, chunk=2, loss_mask=mask)
    dense = per_token_logps(b, ids, attn, chunk=2)
    assert torch.allclose(sparse[mask[:, 1:].bool()], dense[mask[:, 1:].bool()], atol=1e-6)
    assert torch.count_nonzero(sparse[~mask[:, 1:].bool()]) == 0

    # One full batch (mean over valid sequences) versus microbatch-size one with
    # sequence sums scaled by the same global valid-sequence denominator.
    old_model = SelectLM(False)
    old_model.load_state_dict(a.state_dict())
    ref_model = SelectLM(False)
    ref_model.load_state_dict(a.state_dict())
    with torch.no_grad():
        ref_model.head.weight[1].add_(0.08)
        ref_model.head.weight[4].sub_(0.04)
    with torch.no_grad():
        old = per_token_logps(old_model, ids, attn)
        ref = per_token_logps(ref_model, ids, attn)
    assert torch.any(torch.abs(ref[mask[:, 1:].bool()] - old[mask[:, 1:].bool()]) > 1e-5)
    advantages = torch.tensor([0.7, -0.4])
    batch = BatchTensors(ids, attn, mask, advantages, n_active=int(mask[:, 1:].sum()))
    full_loss, _ = grpo_loss(a, batch, old, .3, ref_logps=ref, beta=.001)
    full_loss.backward()
    expected = [p.grad.detach().clone() for p in a.parameters()]
    a.zero_grad(set_to_none=True)
    total_valid = int((mask[:, 1:].sum(dim=1) > 0).sum())
    for i in range(2):
        mb = BatchTensors(ids[i:i+1], attn[i:i+1], mask[i:i+1], advantages[i:i+1],
                          int(mask[i:i+1, 1:].sum()))
        loss, _ = grpo_loss(a, mb, old[i:i+1], .3, ref_logps=ref[i:i+1],
                            beta=.001, sequence_reduction="sum")
        (loss / total_valid).backward()
    for p, grad in zip(a.parameters(), expected):
        assert torch.allclose(p.grad, grad, atol=2e-6, rtol=2e-5)
    beta_grad = [p.grad.detach().clone() for p in a.parameters()]
    a.zero_grad(set_to_none=True)
    beta0, _ = grpo_loss(a, batch, old, .3, ref_logps=ref, beta=0.0)
    beta0.backward()
    assert any(not torch.allclose(p.grad, g, atol=1e-7, rtol=1e-6)
               for p, g in zip(a.parameters(), beta_grad))


def test_qwen3_token_native_prompt_prefix_roundtrip():
    """Optional real-template test; set EZ_QWEN3_TOKENIZER to the local tokenizer."""
    tokenizer_path = os.environ.get("EZ_QWEN3_TOKENIZER")
    if not tokenizer_path:
        pytest.skip("EZ_QWEN3_TOKENIZER not set")
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer
    from rollout import HFPolicy, RolloutConfig, build_token_sequence

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
            self.captured = []
            self.responses = [tok('{"name":"look"}', add_special_tokens=False)["input_ids"],
                              tok('{"name":"inventory"}', add_special_tokens=False)["input_ids"]]

        def generate(self, input_ids, **kwargs):
            self.captured.append(input_ids.detach().cpu().tolist()[0])
            response = torch.tensor([self.responses[len(self.captured) - 1]],
                                    dtype=torch.long, device=input_ids.device)
            return torch.cat((input_ids, response), dim=1)

    model = Dummy()
    policy = HFPolicy(model, tok, device="cpu", enable_thinking=False)
    cfg = RolloutConfig(max_new_tokens=32, max_context_tokens=256)
    initial = [{"role": "system", "content": "system"},
               {"role": "user", "content": "task"}]
    ids = policy.initial_prompt_ids(initial, cfg)
    sampled_prefixes = [list(ids)]
    _, gen1, _ = policy.generate_from_token_ids([ids], cfg)
    assert gen1[0] == model.responses[0]
    feedback = "Observation: kitchen."
    ids = (ids + gen1[0] + policy.assistant_end_ids() + policy.append_feedback_ids(feedback)
           + policy.next_generation_prefix_ids(feedback))
    sampled_prefixes.append(list(ids))
    _, gen2, _ = policy.generate_from_token_ids([ids], cfg)
    assert gen2[0] == model.responses[1]
    train_ids = ids + gen2[0]
    assert model.captured[0] == sampled_prefixes[0]
    assert model.captured[1] == train_ids[:len(sampled_prefixes[1])]
    assert feedback in tok.decode(train_ids, skip_special_tokens=False)
    messages = initial + [{"role": "assistant", "content": tok.decode(gen1[0])},
                          {"role": "user", "content": feedback},
                          {"role": "assistant", "content": tok.decode(gen2[0])}]
    with pytest.raises(RuntimeError, match="not prefix-stable"):
        build_token_sequence(messages, [gen1[0], gen2[0]], policy)

def _two_rank_lora_worker(rank, world, init_method):
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from ez_dist import (broadcast_adapter_parameters, check_adapter_sync,
                         select_adapter, sync_active_adapter_gradients)

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    try:
        torch.manual_seed(991)
        config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, max_position_embeddings=32)
        base = Qwen3ForCausalLM(config)
        torch.manual_seed(500 + rank)  # intentionally different LoRA initialization
        lora = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                          target_modules=["q_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base, lora, adapter_name="proposer")
        model.add_adapter("solver", lora)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
                param.requires_grad_(True)
        broadcast_adapter_parameters(model, world, dist)
        opt_params = [p for n, p in model.named_parameters() if "lora_" in n]
        optimizer = torch.optim.AdamW(opt_params, lr=1e-2)

        proposer_weights = None
        for phase_idx, adapter in enumerate(("proposer", "solver", "proposer")):
            if adapter == "solver":
                proposer_weights = {n: p.detach().clone() for n, p in model.named_parameters()
                                    if "lora_" in n and "proposer" in n}
            select_adapter(model, adapter)
            optimizer.zero_grad(set_to_none=True)
            # Different local examples require real cross-rank gradient averaging.
            ids = torch.tensor([[1 + rank, 2, 3 + rank, 4]])
            loss = model(input_ids=ids, use_cache=False).logits.float().square().mean()
            loss.backward()
            if phase_idx == 0 and rank == 0:
                missing = next(p for n, p in model.named_parameters()
                               if "lora_" in n and "proposer" in n)
                missing.grad = None  # fixed collective still receives zero on this rank
            sync_active_adapter_gradients(model, adapter, world, bucket_mb=1, dist_module=dist)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert check_adapter_sync(model, adapter, world, dist) == 0.0
            if adapter == "solver":
                for name, param in model.named_parameters():
                    if name in proposer_weights:
                        assert torch.equal(param, proposer_weights[name])
            state = []
            for param, values in optimizer.state.items():
                state.append((tuple(param.shape), {k: (v.detach().cpu().tolist()
                                                       if torch.is_tensor(v) else v)
                                                   for k, v in values.items()}))
            gathered = [None] * world
            dist.all_gather_object(gathered, state)
            assert gathered[0] == gathered[1]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_peft_qwen3_adapter_sync(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.multiprocessing import spawn
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    spawn(_two_rank_lora_worker, args=(2, f"tcp://127.0.0.1:{port}"), nprocs=2,
          join=True)


def test_dual_adapter_checkpoint_save_reload_and_resume_state(tmp_path):
    torch = pytest.importorskip("torch")
    import copy
    import random
    import numpy as np
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from ez_train import (build_dual_lora, ez_save, load_ez_resume,
                          resolve_resume_directory)
    from ez_dist import select_adapter

    cfg = {"model": {"path": "base-model", "dtype": "float32", "lora": True,
                      "lora_r": 2, "lora_alpha": 4, "lora_targets": ["q_proj"]},
           "train": {"group_size": 4, "tasks_per_step": 1, "loss_norm": "seq",
                     "micro_batch_size": 1, "kl_beta": 0.001,
                     "proposer_steps": 2, "solver_steps": 2},
           "env": {"split": "train", "config_path": "missing-test-env.yaml"},
           "proposer": {"max_turns": 3}, "solver": {"max_turns": 3},
           "run": {"seed": 12, "name": "test", "output_dir": "irrelevant"}}
    qcfg = Qwen3Config(vocab_size=48, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=1, max_position_embeddings=32)
    torch.manual_seed(91)
    base = Qwen3ForCausalLM(qcfg)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    model = build_dual_lora(base, cfg)
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if "lora_" in n], lr=1e-3)
    fixed_ids = torch.tensor([[1, 2, 3, 4]])
    for adapter in ("proposer", "solver"):
        select_adapter(model, adapter)
        optimizer.zero_grad(set_to_none=True)
        model(input_ids=fixed_ids, use_cache=False).logits.float().square().mean().backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}

    class TinyTokenizer:
        def save_pretrained(self, folder):
            Path(folder, "tokenizer.marker").write_text("tiny")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    ez_save(model, TinyTokenizer(), optimizer, run_dir, "phase-end", cfg,
            {"next_outer": 1, "next_phase": "proposer", "next_step": 0}, [], 0, 1,
            uid_counter=7, dry_run=False)
    ckpt = resolve_resume_directory(run_dir / "ckpt", rank=0)
    state, pool, rng_path = load_ez_resume(ckpt, cfg, world=1, rank=0)
    assert state["next_outer"] == 1 and not pool
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    assert rng["uid_counter"] == 7
    expected_random, expected_numpy, expected_torch = random.random(), np.random.rand(), torch.rand(3)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)

    torch.manual_seed(92)
    base2 = Qwen3ForCausalLM(qcfg)
    base2.load_state_dict(base_state)
    restored = build_dual_lora(base2, cfg, resume_checkpoint=ckpt)
    loaded = {n: p.detach().clone() for n, p in restored.named_parameters() if "lora_" in n}
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(saved[k], loaded[k]) for k in saved)
    restored_optimizer = torch.optim.AdamW(
        [p for n, p in restored.named_parameters() if "lora_" in n], lr=1e-3)
    restored_optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu",
                                                  weights_only=False))
    assert len(restored_optimizer.state) == len(optimizer.state) > 0
    for adapter in ("proposer", "solver"):
        for current_model, current_optimizer in ((model, optimizer),
                                                  (restored, restored_optimizer)):
            select_adapter(current_model, adapter)
            current_optimizer.zero_grad(set_to_none=True)
            current_model(input_ids=fixed_ids, use_cache=False).logits.float().square().mean().backward()
            current_optimizer.step()
        left = {n: p.detach() for n, p in model.named_parameters() if "lora_" in n}
        right = {n: p.detach() for n, p in restored.named_parameters() if "lora_" in n}
        assert all(torch.allclose(left[k], right[k], atol=0, rtol=0) for k in left)

    with pytest.raises(ValueError, match="config/world mismatch"):
        load_ez_resume(ckpt, cfg, world=2, rank=0)
    changed = copy.deepcopy(cfg)
    changed["model"]["lora_alpha"] = 5
    with pytest.raises(ValueError, match="config/world mismatch"):
        load_ez_resume(ckpt, changed, world=1, rank=0)

    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    ez_save(model, TinyTokenizer(), optimizer, smoke_dir, "smoke", cfg,
            {"next_outer": 0, "next_phase": "proposer", "next_step": 0}, [], 0, 1,
            dry_run=True)
    with pytest.raises(ValueError, match="smoke/dry-run"):
        load_ez_resume(smoke_dir / "ckpt", cfg, world=1, rank=0)

    # Recovery resolves a complete previous checkpoint when the primary directory
    # was interrupted during promotion and contains only a partial file.
    (run_dir / "ckpt").rename(run_dir / "ckpt.previous")
    (run_dir / "ckpt").mkdir()
    (run_dir / "ckpt" / "partial").write_text("interrupted")
    assert resolve_resume_directory(run_dir / "ckpt", rank=0) == run_dir / "ckpt.previous"


def test_ignore_benchmark_done_preserves_won_and_outer_budget():
    from alfworld_env import _IgnoreDoneCoreWrapper

    class State(dict):
        pass

    class Core:
        def __init__(self):
            self.i = 0
        def reset(self):
            return State(won=False, lost=False, done=False), 0, False
        def step(self, action):
            self.i += 1
            return State(won=True, lost=False, done=True), 1, True

    class OuterBudget:
        def __init__(self, wrapped, limit):
            self.env, self.limit, self.steps = wrapped, limit, 0
        def reset(self):
            self.steps = 0
            return self.env.reset()
        def step(self, action):
            self.steps += 1
            state, reward, done = self.env.step(action)
            return state, reward, bool(done or self.steps >= self.limit)

    env = OuterBudget(_IgnoreDoneCoreWrapper(Core()), limit=2)
    env.reset()
    state, _, done = env.step("look")
    assert state["won"] is True and state["done"] is False and done is False
    state, _, done = env.step("look")
    assert state["won"] is True and done is True


def test_solver_estimation_keeps_native_prefix_and_done_control_rollout():
    from types import SimpleNamespace
    from ez_solver import run_ez_solver_rollouts
    from ez_state import StateDelta
    from rollout import RolloutConfig

    initial = StateSnapshot(facts=("atlocation(agent, kitchen)",), mode="facts")
    task = ProposedTask("Look around", initial, initial,
                        StateDelta(added=("heated(apple)",)), game_file="same-game")

    class Policy:
        def __init__(self):
            self.prompts = []
        def initial_prompt_ids(self, messages, cfg):
            return [1, 2, 3]
        def generate_from_token_ids(self, prompts, cfg):
            self.prompts.extend([list(x) for x in prompts])
            return ["not a tool call" for _ in prompts], [[8, 9] for _ in prompts], [2] * len(prompts)
        def generate(self, *args, **kwargs):
            raise AssertionError("solve-rate estimation must use the same token-native prefix as training")
        def assistant_end_ids(self): return [4]
        def append_feedback_ids(self, text): return [5]
        def next_generation_prefix_ids(self, text): return [6]

    class Env:
        batch_size = 1
        max_episode_steps = 1
        last_reset_facts = [["atlocation(agent, kitchen)"]]
        def reset(self, games):
            view = SimpleNamespace(env_desc="room", raw="room", task_desc="")
            return ["room"], [view], [["look"]], list(games), [None]
        def step(self, actions):
            return SimpleNamespace(obs=["done room"], facts=[["atlocation(agent, kitchen)"]],
                                   dones=[True], won=[True], admissible=[["look"]])

    policy = Policy()
    result = run_ez_solver_rollouts(Env(), policy, [task], RolloutConfig(max_steps=1),
                                    tokenize=False)
    assert policy.prompts == [[1, 2, 3]]
    assert result[0].traj.input_ids == []
    assert not result[0].traj.stats.truncated  # terminal malformed/control result ends the rollout


def test_eval_loads_only_solver_adapter_and_uses_official_done():
    torch = pytest.importorskip("torch")
    import types
    from eval_ez import _load_policy, _rollout_batch

    class Model(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.active_adapter = None
        def set_adapter(self, name):
            self.active_adapter = name

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<pad>"
        eos_token = "<eos>"

    calls = []
    model = Model()
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: model)
    peft = types.ModuleType("peft")
    class FakePeft:
        @staticmethod
        def from_pretrained(base, path, **kwargs):
            calls.append((str(path), kwargs.get("adapter_name")))
            return base
    peft.PeftModel = FakePeft
    import sys
    old_transformers, old_peft = sys.modules.get("transformers"), sys.modules.get("peft")
    sys.modules["transformers"], sys.modules["peft"] = transformers, peft
    try:
        args = types.SimpleNamespace(base="base", dtype="bfloat16", attn="eager",
                                     base_only=False, ckpt="/run/ckpt")
        policy = _load_policy(args)
        assert policy.model.active_adapter == "solver"
        assert calls == [(str(Path("/run/ckpt") / "solver"), "solver")]
        calls.clear()
        args.base_only = True
        _load_policy(args)
        assert calls == []
    finally:
        if old_transformers is None: sys.modules.pop("transformers", None)
        else: sys.modules["transformers"] = old_transformers
        if old_peft is None: sys.modules.pop("peft", None)
        else: sys.modules["peft"] = old_peft

    class FakePolicy:
        def initial_prompt_ids(self, messages, cfg): return [1, 2]
        def generate_from_token_ids(self, prompts, cfg):
            return ['<tool_call>{"arguments":{"action_type":"look"}}</tool_call>'], [[3]], [1]
        def assistant_end_ids(self): return [4]
        def append_feedback_ids(self, content): return [5]
        def next_generation_prefix_ids(self, content): return [6]
    class FakeEnv:
        def reset(self, games):
            view = types.SimpleNamespace(env_desc="room", raw="room", task_desc="look")
            return ["room"], [view], [[]], list(games), [None]
        def step(self, actions):
            return types.SimpleNamespace(won=[True], dones=[True], obs=["official goal reached"])
    out = _rollout_batch(FakeEnv(), FakePolicy(), ["game"],
                         [{"task_id": "id", "task_type": "type", "sample": 0}],
                         types.SimpleNamespace(), max_steps=1)
    assert out[0]["won"] is True and out[0]["max_steps_reached"] is False
