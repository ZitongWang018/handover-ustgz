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
    from grpo import BatchTensors, grpo_loss, per_token_logps

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
