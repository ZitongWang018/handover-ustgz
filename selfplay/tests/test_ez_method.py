"""Method-level proposer regressions using the real rollout loop and small fakes."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ez_proposer import ProposerConfig, parse_tool_call, run_proposer_exploration
from ez_solver import run_ez_solver_rollouts
from ez_state import snapshot_from_infos, state_delta
from ez_verify import ProposedTask, SubtaskRef, proposer_reward, validity_filter
from rollout import RolloutConfig


def _call(action_type, **kwargs):
    import json
    args = {"action_type": action_type, **kwargs}
    return '<tool_call>' + json.dumps({"name": "alfworld_tool", "arguments": args}) + '</tool_call>'


class FakePolicy:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def initial_prompt_ids(self, _messages, _cfg):
        return [1]

    def assistant_end_ids(self):
        return [2]

    def append_feedback_ids(self, _content):
        return [3]

    def next_generation_prefix_ids(self, _content):
        return [4]

    def generate_from_token_ids(self, prompts, _cfg):
        assert len(prompts) == 1
        output = self.outputs.pop(0)
        return [output], [[5]], None


class FakeEnv:
    batch_size = 1
    max_episode_steps = 30

    def __init__(self):
        self.last_reset_facts = [["inreceptacle(apple 1, countertop 1)"]]
        self.facts = list(self.last_reset_facts[0])
        self.steps = 0

    def reset(self, games):
        assert games == ["game.z8"]
        view = SimpleNamespace(env_desc="ALFWorld scene", raw="scene")
        return (["scene"], [view], [[]], ["game.z8"], {})

    def step(self, actions):
        self.steps += 1
        action = actions[0]
        if "fridge" in action:
            self.facts = ["inreceptacle(apple 1, fridge 1)"]
        else:
            self.facts = ["opened(cabinet 1)", *self.facts]
        return SimpleNamespace(obs=[f"did {action}"], facts=[list(self.facts)],
                               dones=[False], admissible=[[]])


def _roll(outputs, *, sync=True, monkeypatch=None):
    env = FakeEnv()
    cfg = ProposerConfig(group_size=1, max_turns=len(outputs), finish_nudge=False)
    groups = run_proposer_exploration(
        env, FakePolicy(outputs), ["game.z8"], cfg,
        synchronize_initial_state=sync)
    return groups[0][0], env


def test_parser_accepts_complete_bare_json_but_rejects_wrong_schema_and_null_targets():
    bare = '{"name":"alfworld_tool","arguments":{"action_type":"move",' \
           '"object_name":"apple 1","receptacle_name":"fridge 1"}}'
    assert parse_tool_call(bare).action == "put apple 1 in/on fridge 1"
    assert parse_tool_call(bare[:-1]).kind == "malformed"
    assert parse_tool_call('{"arguments":{"action_type":"finish",'
                           '"task_description":"x"}}').kind == "malformed"
    null_target = ('{"name":"alfworld_tool","arguments":{"action_type":"open",'
                   '"receptacle_name":null}}')
    assert parse_tool_call(null_target).kind == "malformed"


def test_action_subtask_action_finish_without_declare_is_accepted():
    rollout, env = _roll([
        _call("open", receptacle_name="cabinet 1"),
        _call("sub_finish", task_description="Open the cabinet."),
        _call("move", object_name="apple 1", receptacle_name="fridge 1"),
        _call("finish", task_description="Put the apple in the fridge."),
    ])
    assert rollout.accepted
    assert len(rollout.task.subtasks) == 1
    assert rollout.task.meta["declared_intent"] == ""
    assert rollout.task.meta["phase_at_finish"] == "task_execution"
    assert env.steps == 2


def test_explore_then_declare_then_execute_records_phase_and_intent():
    rollout, _ = _roll([
        _call("open", receptacle_name="cabinet 1"),
        _call("declare_task", task_description="Put the apple in the fridge."),
        _call("move", object_name="apple 1", receptacle_name="fridge 1"),
        _call("finish", task_description="Put the apple in the fridge."),
    ])
    assert rollout.accepted
    assert rollout.task.meta["declared_intent"] == "Put the apple in the fridge."
    assert rollout.task.meta["phase_at_finish"] == "task_execution"


@pytest.mark.parametrize("finish", [
    '<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"finish",',
    _call("finish"),
])
def test_truncated_or_missing_description_finish_cannot_be_accepted(finish):
    parsed = parse_tool_call(finish)
    assert parsed.kind in ("malformed", "finish")
    rollout, _ = _roll([
        _call("move", object_name="apple 1", receptacle_name="fridge 1"), finish,
    ])
    assert not rollout.accepted


def test_empty_declare_and_subtask_description_are_not_recorded():
    rollout, _ = _roll([
        _call("declare"),
        _call("move", object_name="apple 1", receptacle_name="fridge 1"),
        _call("sub_finish"),
        _call("finish", task_description="Put the apple in the fridge."),
    ])
    assert rollout.accepted
    assert rollout.task.subtasks == []
    assert rollout.task.meta["declared_intent"] == ""


def test_synchronize_false_skips_distributed_collective(monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_gather_object",
                        lambda *_a, **_k: pytest.fail("unexpected cross-rank collective"))
    rollout, _ = _roll([_call("move", object_name="apple 1", receptacle_name="fridge 1"),
                        _call("finish", task_description="Put the apple in the fridge.")],
                       sync=False)
    assert rollout.accepted


class FakeSolverPolicy:
    tokenizer = staticmethod(lambda _text, add_special_tokens=True: {"input_ids": [11, 12]})

    def __init__(self, first_actions):
        self.first_actions = list(first_actions)
        self.calls = 0

    def generate_from_token_ids(self, prompts, _cfg):
        self.calls += 1
        if self.calls == 1:
            actions = self.first_actions
        else:
            actions = ["undo" if action == "checkpoint_then_undo" else "look"
                       for action in self.first_actions]
        texts = [f"<action>{action}</action>" for action in actions]
        return texts, [[1] for _ in prompts], None


class FakeSolverEnv:
    max_episode_steps = 2

    def __init__(self, batch_size, s0, checkpoint1, final):
        self.batch_size = batch_size
        self.s0 = list(s0)
        self.last_reset_facts = [list(s0) for _ in range(batch_size)]
        self.states = [list(s0) for _ in range(batch_size)]
        self.checkpoint1 = list(checkpoint1)
        self.final = list(final)
        self.turn = 0

    def reset(self, games):
        self.turn = 0
        self.states = [list(facts) for facts in self.last_reset_facts]
        views = [SimpleNamespace(raw="Your task is: original goal", env_desc="")
                 for _ in games]
        return (["scene"] * len(games), views, [[] for _ in games], list(games), {})

    def step(self, actions):
        if self.turn == 0:
            for slot, action in enumerate(actions):
                if action == "complete":
                    self.states[slot] = list(self.final)
                elif action in ("checkpoint1", "checkpoint_then_undo"):
                    self.states[slot] = list(self.checkpoint1)
        elif self.turn == 1:
            for slot, action in enumerate(actions):
                if action == "undo":
                    self.states[slot] = list(self.s0)
        self.turn += 1
        return SimpleNamespace(obs=["step"] * self.batch_size,
                               facts=[list(s) for s in self.states],
                               dones=[False] * self.batch_size,
                               admissible=[[] for _ in range(self.batch_size)])


def test_real_solver_loop_preserves_intermediate_rewards_and_binary_proposer_difficulty():
    s0_facts = ["inreceptacle(apple 1, countertop 1)", "not_opened(cabinet 1)"]
    checkpoint1_facts = ["inreceptacle(apple 1, countertop 1)", "opened(cabinet 1)"]
    checkpoint2_facts = [*checkpoint1_facts, "holding(apple 1)"]
    final_facts = ["inreceptacle(apple 1, fridge 1)", "opened(cabinet 1)"]
    s0 = snapshot_from_infos({"facts": s0_facts})
    checkpoint1 = snapshot_from_infos({"facts": checkpoint1_facts})
    checkpoint2 = snapshot_from_infos({"facts": checkpoint2_facts})
    final = snapshot_from_infos({"facts": final_facts})
    goal = "Open the cabinet and put the apple in the fridge."
    task = ProposedTask(
        task_desc=goal, s0=s0, ref_final=final,
        ref_final_delta=state_delta(s0, final, goal),
        subtasks=[
            SubtaskRef("Open the cabinet.", checkpoint1,
                       state_delta(s0, checkpoint1, "Open the cabinet")),
            SubtaskRef("Hold the apple.", checkpoint2,
                       state_delta(s0, checkpoint2, "Hold the apple")),
        ], game_file="generated.z8",
        proposer_actions=["open cabinet 1", "put apple 1 in fridge 1"],
        proposer_action_types=["open", "put"],
    )
    assert validity_filter(task)[0]

    gp = 8
    first_actions = ["complete", "complete", "checkpoint_then_undo", "checkpoint1",
                     "checkpoint1", "fail", "fail", "fail"]
    env = FakeSolverEnv(gp, s0_facts, checkpoint1_facts, final_facts)
    cfg = RolloutConfig(group_size=1, max_steps=2, max_new_tokens=96,
                        max_context_tokens=32768, keep_recent_turns=1,
                        stop_mode="action_tag")
    outputs = run_ez_solver_rollouts(env, FakeSolverPolicy(first_actions),
                                     [task] * gp, cfg, tokenize=False)
    assert [out.traj.reward for out in outputs] == pytest.approx(
        [1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0])
    assert [out.verify.final_match for out in outputs] == [
        True, True, False, False, False, False, False, False]
    # Slot 2 reaches checkpoint1 and then returns to s0; its 0.5 reward proves
    # subtask verification searches the rollout history instead of only sT.
    assert outputs[2].snapshots[-1].facts == outputs[2].snapshots[0].facts
    assert outputs[2].verify.subtask_hits == [True, False]

    binary_solve_rate = sum(out.verify.final_match for out in outputs) / gp
    reward, parts = proposer_reward(task, binary_solve_rate)
    assert binary_solve_rate == pytest.approx(0.25)
    assert parts["r_difficulty"] == pytest.approx(0.5)
    assert parts["r_difficulty"] != pytest.approx(
        1 - abs(2 * sum(out.traj.reward for out in outputs) / gp - 1))


def test_invalid_proposal_keeps_zero_reward_and_is_excluded_by_acceptance_gate():
    rollout, _ = _roll([
        _call("move", object_name="apple 1", receptacle_name="fridge 1"),
        _call("finish"),
    ])
    assert not rollout.accepted
    assert rollout.reward == 0.0
    assert rollout.task is not None
    assert not validity_filter(rollout.task)[0]
