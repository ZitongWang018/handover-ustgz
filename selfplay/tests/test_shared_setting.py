from types import SimpleNamespace
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


class CharTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        return {"input_ids": ([1] + ids if add_special_tokens else ids)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token != 1)


def test_skillrise_prompt_history_projection_and_raw_tokenization():
    from protocol import (build_skillrise_prompt, encode_raw_prompt,
                          format_skillrise_history, project_skillrise_action)

    history = [(f"action-{i}", f"observation-{i}") for i in range(1, 18)]
    formatted = format_skillrise_history(history)
    assert "Action 3: action-3\nObservation 3: observation-3" in formatted
    assert "Action 2:" not in formatted
    prompt = build_skillrise_prompt("room\nYour task is to: clean the mug", history,
                                    ["look", "help", "take mug"], 17)
    assert "Action 3: action-3" in prompt
    assert "Action 17: action-17\nObservation 17: observation-17" in prompt
    assert "'help'" not in prompt
    assert "<action> </action> tags" in prompt
    assert project_skillrise_action("Thought\n<action>TAKE MUG</action>")[0] == "take mug"
    assert project_skillrise_action("x" * 35) == ("x" * 30, False)
    tokenizer = CharTokenizer()
    assert encode_raw_prompt(tokenizer, "abc") == [1, ord("a"), ord("b"), ord("c")]


def test_goal_replacement_and_action_tag_stop_include_close():
    from protocol import action_tag_stop_len, replace_task_goal

    raw = "room details\nYour task is to: official goal\nother detail"
    changed = replace_task_goal(raw, "put apple in fridge")
    assert changed == "room details\nYour task is to: put apple in fridge\nother detail"
    assert "official goal" not in changed
    try:
        replace_task_goal("room details", "put apple in fridge")
    except ValueError:
        pass
    else:
        raise AssertionError("goal replacement must fail closed when no task line exists")

    tokenizer = CharTokenizer()
    generated = "<action>look</action>extra"
    ids = [ord(ch) for ch in generated]
    assert action_tag_stop_len(tokenizer, ids) == len("<action>look</action>")


def test_vllm_action_tag_uses_native_inclusive_stop(monkeypatch):
    import sys
    from types import ModuleType

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class Tokenizer(CharTokenizer):
        pad_token_id = 0
        eos_token_id = 999
        pad_token = "<pad>"
        eos_token = "<eos>"

    class Output:
        token_ids = [ord(ch) for ch in "<action>look</action>"]
        finish_reason = "stop"

    class Request:
        outputs = [Output()]

    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            self.params = sampling_params
            return [Request()]

    from ez_vllm_policy import VLLMPolicy
    from rollout import RolloutConfig

    llm = LLM()
    policy = VLLMPolicy(llm, Tokenizer())
    cfg = RolloutConfig(stop_mode="action_tag", max_context_tokens=32768)
    texts, ids, _ = policy.generate_from_token_ids([[42]], cfg)
    assert llm.params.kwargs["stop"] == ["</action>"]
    assert llm.params.kwargs["include_stop_str_in_output"] is True
    assert llm.params.kwargs["detokenize"] is True
    assert texts == ["<action>look</action>"]
    assert ids == [Output.token_ids]


def test_hf_action_tag_stop_records_complete_close_tag():
    torch = __import__("pytest").importorskip("torch")
    from rollout import _ActionTagStopCriteria

    tokenizer = CharTokenizer()
    criteria = _ActionTagStopCriteria(tokenizer, [0], max_new_tokens=100)
    ids = torch.tensor([[ord(ch) for ch in "<action>look</action>"]])
    stopped = criteria(ids, None)
    assert bool(stopped) if isinstance(stopped, bool) else bool(stopped[0])
    assert criteria.stop_len(0, 100) == len("<action>look</action>")


def test_ez_solver_records_independent_prompt_response_segments():
    from ez_solver import run_ez_solver_rollouts
    from ez_state import StateDelta, StateSnapshot
    from ez_verify import ProposedTask
    from protocol import MAX_PROMPT_CONTEXT
    from rollout import RolloutConfig

    initial = StateSnapshot(facts=("atlocation(agent, kitchen)",), mode="facts")
    task = ProposedTask("put apple in fridge", initial, initial,
                        StateDelta(added=("heated(apple)",)), game_file="game",
                        meta={"uid": "generated-1"})

    class Policy:
        tokenizer = CharTokenizer()

        def __init__(self):
            self.prompts = []

        def generate_from_token_ids(self, prompts, cfg):
            assert cfg.stop_mode == "action_tag"
            assert cfg.max_context_tokens == MAX_PROMPT_CONTEXT
            self.prompts.extend([list(prompt) for prompt in prompts])
            text = "private reasoning\n<action>look</action>"
            ids = [ord(ch) for ch in text]
            return [text] * len(prompts), [ids.copy() for _ in prompts], [len(ids)] * len(prompts)

    class Env:
        batch_size = 1
        max_episode_steps = 2
        last_reset_facts = [["atlocation(agent, kitchen)"]]

        def __init__(self):
            self.steps = 0

        def reset(self, games):
            view = SimpleNamespace(raw="kitchen\nYour task is to: official goal")
            return [view.raw], [view], [["look"]], list(games), [None]

        def step(self, actions):
            self.steps += 1
            return SimpleNamespace(obs=["kitchen unchanged"], facts=[["atlocation(agent, kitchen)"]],
                                   dones=[self.steps == 2], won=[False], admissible=[["look"]])

    policy = Policy()
    result = run_ez_solver_rollouts(
        Env(), policy, [task], RolloutConfig(max_steps=2, stop_mode="action_tag",
                                             max_context_tokens=MAX_PROMPT_CONTEXT), tokenize=True)
    traj = result[0].traj
    assert len(traj.segments) == 2
    prompt_texts = [policy.tokenizer.decode(prompt) for prompt, _ in traj.segments]
    response_ids = [response for _, response in traj.segments]
    assert all("put apple in fridge" in prompt for prompt in prompt_texts)
    assert all("official goal" not in prompt for prompt in prompt_texts)
    assert "private reasoning" not in prompt_texts[1]
    assert "Action 1: look\nObservation 1: kitchen unchanged" in prompt_texts[1]
    assert response_ids == [[ord(ch) for ch in "private reasoning\n<action>look</action>"]] * 2
    assert traj.input_ids == [] and traj.loss_mask == []
    assert traj.n_trainable_tokens == sum(map(len, response_ids))
    assert traj.has_trainable_tokens
