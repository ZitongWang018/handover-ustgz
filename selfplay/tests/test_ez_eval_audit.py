"""CPU regressions for the executable EZ evaluation protocol and reporting contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from ez_eval_protocol import aggregate_complete_evaluation, align_records_to_slots
from ez_proposer import parse_tool_call
from ez_prompts import resolve_action


def _tool(action_type, **kwargs):
    payload = {"name": "alfworld_tool",
               "arguments": {"action_type": action_type, **kwargs}}
    return '<tool_call>' + json.dumps(payload) + "</tool_call>"


@pytest.mark.parametrize(("kind", "args", "command"), [
    ("go_to", {"receptacle_name": "table 1"}, "go to table 1"),
    ("take", {"object_name": "apple 2", "receptacle_name": "counter 1"},
     "take apple 2 from counter 1"),
    ("move", {"object_name": "mug 1", "receptacle_name": "counter 1"},
     "put mug 1 in/on counter 1"),
    ("put", {"object_name": "mug 1", "receptacle_name": "counter 1"},
     "put mug 1 in/on counter 1"),
    ("open", {"receptacle_name": "fridge 1"}, "open fridge 1"),
    ("close", {"receptacle_name": "fridge 1"}, "close fridge 1"),
    ("look", {}, "look"), ("inventory", {}, "inventory"),
    ("examine", {"object_name": "apple 1"}, "examine apple 1"),
    ("use", {"object_name": "light 1"}, "use light 1"),
    ("heat", {"object_name": "potato 2", "receptacle_name": "microwave 1"},
     "heat potato 2 with microwave 1"),
    ("clean", {"object_name": "mug 1", "receptacle_name": "sinkbasin 1"},
     "clean mug 1 with sinkbasin 1"),
    ("cool", {"object_name": "apple 1", "receptacle_name": "fridge 1"},
     "cool apple 1 with fridge 1"),
    ("slice", {"object_name": "apple 1", "receptacle_name": "knife 1"},
     "slice apple 1 with knife 1"),
])
@pytest.mark.parametrize("strict", [False, True])
def test_documented_schema_actions_have_canonical_native_commands(kind, args, command,
                                                                   strict, monkeypatch):
    monkeypatch.setenv("EZ_PARSER_STRICT", "1" if strict else "0")
    parsed = parse_tool_call(_tool(kind, **args))
    assert parsed.kind == "action"
    assert parsed.action == command


def test_move_schema_is_never_reinterpreted_as_go_to(monkeypatch):
    monkeypatch.setenv("EZ_PARSER_STRICT", "0")
    parsed = parse_tool_call(_tool("move", object_name="apple 1", receptacle_name="fridge 1"))
    assert parsed.action == "put apple 1 in/on fridge 1"


@pytest.mark.parametrize("flag", ["EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA"])
def test_optional_fewshot_tool_call_uses_valid_documented_schema(monkeypatch, flag):
    import re
    import ez_prompts
    monkeypatch.setenv("EZ_SOLVER_FEWSHOT", "0")
    monkeypatch.setenv("EZ_SOLVER_FEWSHOT_SCHEMA", "0")
    monkeypatch.setenv(flag, "1")
    example = ez_prompts._solver_fewshot()
    tool = re.search(r"<tool_call>(.*?)</tool_call>", example, flags=re.S)
    assert tool
    parsed = parse_tool_call("<tool_call>" + tool.group(1) + "</tool_call>")
    assert parsed.kind == "action"
    assert parsed.action == "go to fridge 1"


def test_public_style_resolver_modes_and_strict_disable(monkeypatch):
    actions = ["look", "go to fridge 1", "take apple 1 from fridge 1", "use apple 1"]
    monkeypatch.setenv("EZ_SOLVER_STRICT_RESOLVE", "0")
    assert resolve_action("look", actions) == ("look", "exact")
    assert resolve_action("Action: look", actions) == ("look", "cleaned")
    assert resolve_action("2", actions) == ("take apple 1 from fridge 1", "index")
    assert resolve_action("2. take apple 1 from fridge 1", actions) == (
        "take apple 1 from fridge 1", "cleaned")
    assert resolve_action("I should use apple 1 now", actions) == ("use apple 1", "substring")
    assert resolve_action("go to fridge1", actions) == ("go to fridge 1", "fuzzy")
    assert resolve_action("unrecognized words", actions) == ("look", "fallback")
    monkeypatch.setenv("EZ_SOLVER_STRICT_RESOLVE", "1")
    assert resolve_action("look", actions) == ("", "none")


def test_alignment_is_fifo_for_shuffled_and_repeated_gamefiles():
    games = ["/split/game-a", "/split/game-a", "/split/game-b", "/split/game-a"]
    records = [{"task_id": "a", "task_type": "x", "sample": 0},
               {"task_id": "a", "task_type": "x", "sample": 1},
               {"task_id": "b", "task_type": "y", "sample": 0},
               {"task_id": "a", "task_type": "x", "sample": 2}]
    aligned = align_records_to_slots(games, records,
                                     ["/split/game-a", "/split/game-b",
                                      "/split/game-a", "/split/game-a"])
    assert [r["sample"] for r in aligned] == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="unexpected or excess"):
        align_records_to_slots(games, records, ["/split/game-z"] * 4)
    with pytest.raises(ValueError):
        align_records_to_slots(games, records, games[:-1])


def _complete_rows():
    rows = []
    expected = {"t0": "type-a", "t1": "type-a", "t2": "type-b"}
    vals = {"t0": [1, 0], "t1": [0, 0], "t2": [1, 1]}
    for task_id, typ in expected.items():
        for sample, won in enumerate(vals[task_id]):
            rows.append({"task_id": task_id, "task_type": typ,
                         "sample": sample, "won": bool(won)})
    return rows, expected


def test_complete_evaluation_statistics_are_task_sample_strict_and_named():
    rows, expected = _complete_rows()
    out = aggregate_complete_evaluation(rows, expected, 2)
    assert out["episode_weighted_sr"] == pytest.approx(3 / 6)
    assert out["sample_suite_success_rates"] == pytest.approx([2 / 3, 1 / 3])
    assert out["sample_suite_sr_population_std"] == pytest.approx(1 / 6)
    assert out["sample_suite_sr_sample_std"] == pytest.approx((1 / 18) ** 0.5)
    assert out["task_type"]["type-a"]["task_macro_sr"] == pytest.approx(0.25)
    assert out["task_type"]["type-b"]["episode_weighted_sr"] == 1
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError):
            aggregate_complete_evaluation(bad, expected, 2)
    bad_sample = [dict(r) for r in rows]
    bad_sample[0]["sample"] = 1.5
    with pytest.raises(ValueError, match="integer sample"):
        aggregate_complete_evaluation(bad_sample, expected, 2)
    bad_won = [dict(r) for r in rows]
    bad_won[0]["won"] = "False"
    with pytest.raises(ValueError, match="won"):
        aggregate_complete_evaluation(bad_won, expected, 2)


def test_malformed_eval_output_keeps_recovered_action_and_real_observation(monkeypatch):
    import eval_ez

    monkeypatch.setenv("EZ_SOLVER_STRICT_RESOLVE", "0")
    monkeypatch.setenv("EZ_SOLVER_ADMISSIBLE", "0")

    class Policy:
        def __init__(self, text):
            self.text = text
            self.feedback = []
        def initial_prompt_ids(self, messages, cfg): return [1]
        def generate_from_token_ids(self, prompts, cfg):
            return [self.text for _ in prompts], [[9, 10] for _ in prompts], [2] * len(prompts)
        def assistant_end_ids(self): return [2]
        def append_feedback_ids(self, text):
            self.feedback.append(text)
            return [3]
        def next_generation_prefix_ids(self, text): return [4]

    class Env:
        def reset(self, games):
            views = [SimpleNamespace(env_desc="room", raw="room", task_desc="task")]
            return ["room"], views, [["look", "go to fridge 1"]], list(games), [None]
        def step(self, commands):
            self.command = commands[0]
            return SimpleNamespace(won=[False], dones=[False], obs=["fridge is visible"],
                                   admissible=[["look", "go to fridge 1"]])

    env = Env()
    policy = Policy("go to fridge 1")
    rows = eval_ez._rollout_batch(
        env, policy, ["game"], [{"task_id": "id", "task_type": "type", "sample": 0}],
        SimpleNamespace(), max_steps=1, style="ours")
    assert env.command == "go to fridge 1"
    assert policy.feedback[0].startswith("Your response format was invalid")
    assert "You did: go to fridge 1" in policy.feedback[0]
    assert "fridge is visible" in policy.feedback[0]
    assert rows[0]["trace"][0]["raw_output"] == "go to fridge 1"
    assert rows[0]["trace"][0]["executed_command"] == "go to fridge 1"
    assert rows[0]["trace"][0]["resolver_mode"] == "exact"
    assert rows[0]["max_steps_reached"] is True
    assert rows[0]["budget_boundary_not_won"] is True


def test_rich_json_is_diagnostic_only_and_menu_resolver_selects_action(monkeypatch):
    import eval_ez

    monkeypatch.setenv("EZ_SOLVER_STRICT_RESOLVE", "1")
    class Policy:
        def initial_prompt_ids(self, messages, cfg): return [1]
        def generate_from_token_ids(self, prompts, cfg):
            text = '<tool_call>{"arguments":{"action_type":"go_to","receptacle_name":"fridge 1"}}</tool_call>'
            return [text], [[7, 8]], [2]
    class Env:
        def reset(self, games):
            view = SimpleNamespace(env_desc="room", raw="room", task_desc="task")
            return ["room"], [view], [["look", "go to fridge 1"]], list(games), [None]
        def step(self, commands):
            self.command = commands[0]
            return SimpleNamespace(won=[False], dones=[True], obs=["done"],
                                   admissible=[["look", "go to fridge 1"]])

    env = Env()
    rows = eval_ez._rollout_batch(
        env, Policy(), ["game"], [{"task_id": "id", "task_type": "type", "sample": 0}],
        SimpleNamespace(), max_steps=1, style="rich")
    assert env.command == "look"  # no menu action appears in the JSON response
    assert rows[0]["trace"][0]["json_parse_kind"] == "action"  # diagnostic only
    assert rows[0]["trace"][0]["parse_kind"] == "menu_fallback"
    assert rows[0]["trace"][0]["resolver_mode"] == "fallback"
    assert rows[0]["kinds"]["json_diagnostic_action_ignored"] == 1
    assert rows[0]["max_steps_reached"] is True
    assert rows[0]["budget_boundary_not_won"] is True


def test_rich_next_turn_uses_raw_observation_without_json_feedback(monkeypatch):
    import eval_ez
    monkeypatch.setenv("EZ_SOLVER_STRICT_RESOLVE", "1")
    class Policy:
        prompts = []
        call = 0
        def initial_prompt_ids(self, messages, cfg):
            self.prompts.append(messages[-1]["content"])
            return [1]
        def generate_from_token_ids(self, prompts, cfg):
            self.call += 1
            return ["look"], [[self.call + 5]], [1]
    class Env:
        step_no = 0
        def reset(self, games):
            view = SimpleNamespace(env_desc="room", raw="initial raw observation",
                                  task_desc="task")
            return ["initial raw observation"], [view], [["look"]], list(games), [None]
        def step(self, commands):
            self.step_no += 1
            return SimpleNamespace(won=[False], dones=[self.step_no == 2],
                                   obs=[f"raw observation {self.step_no}"],
                                   admissible=[["look"]])
    policy, env = Policy(), Env()
    eval_ez._rollout_batch(
        env, policy, ["game"], [{"task_id": "id", "task_type": "type", "sample": 0}],
        SimpleNamespace(), max_steps=2, style="rich")
    assert "Current observation:\nraw observation 1" in policy.prompts[1]
    assert "Your response format" not in policy.prompts[1]
    assert "valid tool call" not in policy.prompts[1]


@pytest.mark.parametrize(
    "text,bare_fallback,feedback_prefix",
    [
        ("unparseable answer", "0",
         "Your response was not a valid tool call; the fallback command was executed."),
        ("look", "1",
         "Your response format was invalid; the recovered command was executed."),
        ('<tool_call>{"name":"alfworld_tool","arguments":{"action_type":"declare","task_description":"look around"}}</tool_call>',
         "0", "Declarations are not environment actions; the fallback command was executed."),
    ],
)
def test_solver_nonterminal_malformed_recovery_and_control_keep_real_feedback(
        monkeypatch, text, bare_fallback, feedback_prefix):
    from ez_solver import run_ez_solver_rollouts
    from ez_state import StateDelta, StateSnapshot
    from ez_verify import ProposedTask
    from rollout import RolloutConfig

    monkeypatch.setenv("EZ_SOLVER_BARE_FALLBACK", bare_fallback)
    initial = StateSnapshot(facts=("atlocation(agent, kitchen)",), mode="facts")
    task = ProposedTask("Look around", initial, initial, StateDelta(),
                        game_file="game", meta={"uid": "uid"})

    class Policy:
        feedback = []
        def initial_prompt_ids(self, messages, cfg): return [1, 2]
        def generate_from_token_ids(self, prompts, cfg):
            return [text], [[8, 9]], [2]
        def assistant_end_ids(self): return [4]
        def append_feedback_ids(self, content):
            self.feedback.append(content)
            return [5]
        def next_generation_prefix_ids(self, content): return [6]

    class Env:
        batch_size = 1
        max_episode_steps = 1
        last_reset_facts = [["atlocation(agent, kitchen)"]]
        def reset(self, games):
            view = SimpleNamespace(env_desc="room", raw="room", task_desc="look around")
            return ["room"], [view], [["look"]], list(games), [None]
        def step(self, actions):
            self.actions = list(actions)
            return SimpleNamespace(obs=["the actual room observation"],
                                   facts=[["atlocation(agent, kitchen)"]],
                                   dones=[False], won=[False], admissible=[["look"]])

    env, policy = Env(), Policy()
    trajectories = run_ez_solver_rollouts(
        env, policy, [task], RolloutConfig(max_steps=1), tokenize=False)
    assert env.actions == ["look"]
    assert trajectories[0].traj.stats.truncated
    assert any(m["role"] == "assistant" and text in m["content"]
               for m in trajectories[0].traj.messages)
    assert policy.feedback[0].startswith(feedback_prefix)
    assert "You did: look" in policy.feedback[0]
    assert "the actual room observation" in policy.feedback[0]


def test_hf_eos_greedy_mode_has_no_custom_stop_or_sampling_inheritance():
    torch = pytest.importorskip("torch")
    from rollout import HFPolicy, RolloutConfig

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 2
        pad_token = "<pad>"
        eos_token = "<eos>"
        def pad(self, batch, padding=True, return_tensors=None):
            ids = torch.tensor(batch["input_ids"], dtype=torch.long)
            class Batch(dict):
                def to(self, device): return self
            return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))
        def decode(self, ids, skip_special_tokens=True): return "generated"

    model = torch.nn.Linear(1, 1)
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        prefix = kwargs["input_ids"]
        suffix = torch.tensor([[7, 2, 0]], dtype=torch.long)
        return torch.cat([prefix, suffix], dim=1)
    model.generate = generate
    policy = HFPolicy(model, Tokenizer(), device="cpu")
    cfg = RolloutConfig(max_new_tokens=3, temperature=0, top_p=0.95, top_k=20,
                        stop_mode="eos", max_context_tokens=32)
    texts, ids, _ = policy._generate_batch([], cfg, prompt_ids_list=[[1, 3]])
    assert texts == ["generated"] and ids == [[7]]
    assert captured["do_sample"] is False
    assert "stopping_criteria" not in captured
    assert all(captured[k] is None for k in ("temperature", "top_p", "top_k", "min_p"))
    with pytest.raises(ValueError, match="unsupported stop_mode"):
        policy._generate_batch([], RolloutConfig(stop_mode="mystery"),
                               prompt_ids_list=[[1]])


def test_vllm_engine_uses_manifest_seed(monkeypatch):
    import types
    import eval_ez

    captured = {}
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2
        pad_token = "<pad>"
        eos_token = "<eos>"
        name_or_path = "tok"
        chat_template = "template"
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    vllm = types.ModuleType("ez_vllm_policy")
    vllm.VLLMPolicy = lambda engine, tokenizer, log_fn=None: ("policy", engine)
    def build(base, **kwargs):
        captured.update(kwargs)
        return "engine"
    vllm.build_vllm_engine = build
    vllm.export_lora_adapter = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ez_vllm_policy", vllm)
    args = SimpleNamespace(base="base", backend="vllm", base_only=True, seed=314,
                           max_context_tokens=40960)
    assert eval_ez._load_policy(args) == ("policy", "engine")
    assert captured["seed"] == 314


@pytest.mark.parametrize(
    "argv,expected_profile,expected_style,expected_steps,expected_temp,expected_samples",
    [
        (["--protocol", "public-rich"], "public-rich", "rich", 50, 0.0, 1),
        (["--protocol", "paper-approx", "--max-steps", "11",
          "--temperature", "0.5", "--samples", "2"],
         "custom", "ours", 11, 0.5, 2),
    ],
)
def test_eval_cli_profiles_resolve_to_recorded_effective_protocol(
        monkeypatch, tmp_path, argv, expected_profile, expected_style,
        expected_steps, expected_temp, expected_samples):
    import types
    import eval_ez

    monkeypatch.delenv("EZ_SOLVER_STYLE", raising=False)
    monkeypatch.delenv("EZ_SOLVER_FEWSHOT", raising=False)
    monkeypatch.delenv("EZ_SOLVER_FEWSHOT_SCHEMA", raising=False)
    monkeypatch.delenv("EZ_SOLVER_ADMISSIBLE", raising=False)
    monkeypatch.delenv("EZ_PARSER_STRICT", raising=False)
    monkeypatch.delenv("EZ_SOLVER_STRICT_RESOLVE", raising=False)
    out = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["eval_ez.py", *argv, "--base", "base",
                                     "--base-only", "--backend", "hf", "--out", str(out)])
    monkeypatch.setattr(eval_ez, "load_alfworld_config", lambda path: {})
    monkeypatch.setattr(eval_ez, "collect_game_files", lambda *args: ["/valid_unseen/task"])
    monkeypatch.setattr(eval_ez, "task_instance_id", lambda game: "task-0")
    monkeypatch.setattr(eval_ez, "task_type_of", lambda game: "type-a")
    monkeypatch.setitem(sys.modules, "alfworld_compat",
                        types.SimpleNamespace(install=lambda **kwargs: None))
    model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation="sdpa"))
    tokenizer = types.SimpleNamespace(name_or_path="fake-tokenizer", chat_template="fake-template")
    fake_policy = types.SimpleNamespace(model=model, tokenizer=tokenizer)
    monkeypatch.setattr(eval_ez, "_load_policy", lambda args: fake_policy)
    captured = {}
    class Env:
        def __init__(self, *args, **kwargs): captured["env"] = kwargs
        def close(self): pass
    monkeypatch.setattr(eval_ez, "ALFWorldBatchEnv", Env)
    def rollout(env, policy, games, records, cfg, max_steps, *, style=None):
        captured["cfg"] = cfg
        captured["style"] = style
        return [{"task_id": record["task_id"], "task_type": record["task_type"],
                 "game_file": games[i], "sample": record["sample"], "won": True,
                 "steps": 1, "actions": ["look"], "trace": [], "kinds": {},
                 "max_steps_reached": False}
                for i, record in enumerate(records)]
    monkeypatch.setattr(eval_ez, "_rollout_batch", rollout)
    assert eval_ez.main() == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    manifest = report["protocol_manifest"]
    assert manifest["protocol_effective"] == expected_profile
    assert manifest["prompt_style"] == expected_style
    assert report["summary"]["samples_per_task"] == expected_samples
    assert captured["cfg"].max_steps == expected_steps
    assert captured["cfg"].temperature == expected_temp
    assert captured["cfg"].stop_mode == ("eos" if expected_style == "rich" else "tool_call")
    assert captured["env"]["max_episode_steps"] == expected_steps
    if expected_profile == "public-rich":
        assert report["decode_effective"]["do_sample"] is False
        assert report["decode_effective"]["temperature"] is None
        assert manifest["resolver"]["menu_enabled"] is True
