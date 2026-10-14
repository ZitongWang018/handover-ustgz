from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLRISE = ROOT / "skillrise"
sys.path.insert(0, str(SKILLRISE))
sys.path.insert(0, str(SKILLRISE / "upstream"))

from eval_protocol import (  # noqa: E402
    ATTEMPTS, PROFILE, SHARD_COUNT, SLOT_COUNT, SLOT_SEED_BASE, TASK_TYPES,
    ProtocolError, aggregate_rows, decode_vllm_token_ids, make_selection_manifest,
    sha256_bytes, sha256_json, game_id, shard_slots, tokenize_upstream_prompt, validate_selection_manifest,
    validate_shard_grid, verify_preflight,
)


def _manifest():
    pool = [f"/tmp/task-{i}.tw-pddl" for i in range(9)]
    return make_selection_manifest(
        pool, lambda p: TASK_TYPES[int(Path(p).stem.split("-")[1]) % len(TASK_TYPES)],
        pool_order_sha256="a" * 64, data_config_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
    )


def _rows(manifest):
    out = []
    for s in manifest["slots"]:
        attempts = []
        for a in range(ATTEMPTS):
            # Exactly one task succeeds on attempt 0; retries are cumulative metrics only.
            attempts.append({"attempt": a, "won": s["slot"] == 0 and a == 0,
                             "skipped_after_earlier_win": s["slot"] == 0 and a > 0})
        out.append({"slot": s["slot"], "worker_seed": s["worker_seed"],
                    "game_file": s["game_file"], "game_id": s["game_id"],
                    "task_type": s["task_type"], "sample": 0, "attempts": attempts})
    return out


def _shards(manifest, rows=None):
    rows = rows or _rows(manifest)
    common = {
        "schema_version": 1, "profile": PROFILE, "run_id": "run-x", "style": "react",
        "base_model": "/model/qwen3_4b", "model_identity": {"status": "weights_unverified"},
        "decode": {"temperature": 0.7, "top_p": 1.0}, "samples": 1,
        "engine": {"tensor_parallel_size": 1}, "tokenizer": {"chat_template_sha256": "d" * 64},
        "software": {"vllm": "test"}, "upstream_sources": {"runner": "e" * 64},
        "vllm_attention_backend": "FLASH_ATTN",
        "prompt_provenance": {"paper_exact": False, "kind": "upstream prompt",
                              "source": "test snapshot"},
        "glibc_compat": {"mode": "native_fast_downward", "path": None},
        "world_size": SHARD_COUNT, "selection_manifest_sha256": sha256_json(manifest),
    }
    return [{**common, "rank": rank, "rows": rows[rank * 16:(rank + 1) * 16]}
            for rank in range(SHARD_COUNT)]


def test_vendor_snapshot_hashes_match_source_manifest():
    data = json.loads((SKILLRISE / "upstream/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert data["upstream_commit"] == "2b38da3ad7b64414305d39609484803b07d3b9b0"
    assert "skillrise_rollout_loop.py" in data["active_validation_path"]
    assert "unreferenced isolated config" in data["excluded_config"]
    for entry in data["files"]:
        p = SKILLRISE / "upstream" / entry["snapshot_path"]
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == entry["snapshot_sha256"] == entry["source_sha256"]
        assert entry["byte_identical"] is True


def test_actual_dispatcher_and_generation_agent_have_no_string_stops():
    dispatcher = (SKILLRISE / "upstream/agent_system/multi_turn_rollout/__init__.py").read_text(encoding="utf-8")
    agent_loop = (SKILLRISE / "upstream/agent_system/multi_turn_rollout/skillrise_rollout_loop.py").read_text(encoding="utf-8")
    spmd = (SKILLRISE / "upstream/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py").read_text(encoding="utf-8")
    assert "from .skillrise_rollout_loop import TrajectoryCollector" in dispatcher
    assert "self.enable_thinking = False" in agent_loop
    assert "enable_thinking=self.enable_thinking" in agent_loop
    section = spmd.split("def generate_sequences_agent", 1)[1].split("def ", 1)[0]
    assert "min_tokens" in section
    assert "'stop'" not in section and '"stop"' not in section
    grpo_shell = (SKILLRISE / "upstream/examples/skillrise_alfworld/grpo_alfworld_qwen3_4b.sh").read_text(encoding="utf-8")
    assert "actor_rollout_ref.rollout.max_num_seqs=512" in grpo_shell
    from eval_protocol import upstream_engine_kwargs
    assert upstream_engine_kwargs()["max_num_seqs"] == 512
    launcher = (SKILLRISE / "run_eval_8gpu.sh").read_text(encoding="utf-8")
    slurm = (SKILLRISE / "skillrise_eval_8gpu.sbatch").read_text(encoding="utf-8")
    assert 'SKR_PROJECT_ROOT' in launcher and 'ALFWORLD_DATA' in launcher and 'ZTWANG_MODEL_PATH' in launcher
    assert 'VLLM_WORKER_MULTIPROC_METHOD=spawn' in launcher and 'OPENBLAS_NUM_THREADS=1' in launcher
    assert 'GPU_MASK="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"' in launcher
    assert 'CUDA_VISIBLE_DEVICES="${GPU_DEVICES[$rank]}"' in launcher
    assert 'CUDA_VISIBLE_DEVICES="${GPU_DEVICES[0]}"' in launcher
    assert "--cpus-per-task=64" in slurm and "run_eval_8gpu.sh" in slurm


def test_upstream_prompt_history_and_projection_are_used_verbatim():
    from agent_system.environments.alfworld.prompt import get_alfworld_prompt
    from agent_system.environments.alfworld.memory import SimpleMemory
    from agent_system.environments.alfworld.projection import alfworld_projection

    raw = "Welcome\n\nKitchen raw text  \n\nYour task is to: put an apple in the fridge."
    memory = SimpleMemory()
    memory.reset(1)
    prompt = get_alfworld_prompt(
        phase="play", turn_idx=0, traj_idx=0, init_observation=raw,
        curr_traj="", past_traj={}, admissible_actions="'look',\n 'take apple 1'",
        reflection={}, reflection_type="reflection_only",
    )
    assert "Kitchen raw text  \n\nYour task is to: put an apple in the fridge." in prompt
    memory.store({"text_obs": ["Obs-A"], "action": ["act-A"], "reward": [0.], "dones": [False], "won": [False]})
    history, length = memory.fetch(history_length=15)
    assert history == ["Action 1: act-A\nObservation 1: Obs-A"] and length == [1]
    parsed, valid = alfworld_projection(["<action> Take Apple 1 </action>"], [["take apple 1"]], phase="play")
    assert parsed == ["take apple 1"] and valid == [1]
    empty, invalid = alfworld_projection([""], [[]], phase="play")
    assert empty == [""] and invalid == [0]


def test_chat_template_is_single_user_default_thinking_disabled_and_length_gated():
    class FakeTokenizer:
        def __init__(self): self.calls = []
        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>\n<|im_start|>assistant\n"
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return list(range(len(text)))

    from eval_protocol import tokenize_upstream_prompt
    tok = FakeTokenizer()
    rendered, ids = tokenize_upstream_prompt(tok, "hello", max_tokens=500)
    messages, kwargs = tok.calls[0]
    assert messages == [{"role": "user", "content": "hello"}]
    assert kwargs == {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}
    assert len(ids) == len(rendered)
    with pytest.raises(ProtocolError, match="no truncation"):
        tokenize_upstream_prompt(tok, "hello", max_tokens=2)


def test_generation_decodes_ids_not_vllm_text():
    class Tok:
        def decode(self, ids, **kwargs):
            assert kwargs == {"skip_special_tokens": True}
            return "ids=" + ",".join(map(str, ids))
    class Output:
        token_ids = [2, 3, 4]
        text = "WRONG TEXT FIELD"
    assert decode_vllm_token_ids(Output(), Tok()) == ("ids=2,3,4", [2, 3, 4])


def test_same_game_restart_uses_wrapped_textworld_batch_env(tmp_path):
    from types import SimpleNamespace
    from eval_alfworld_prompting import _restart_same_game

    class Batch:
        def reset(self):
            return ["restarted"], {"admissible_commands": [["look"]]}

    inner = SimpleNamespace(last_commands=["old"], batch_env=Batch(), obs=None)
    outer = SimpleNamespace(env=inner)
    obs, infos = _restart_same_game(outer)
    assert inner.last_commands == [None]
    assert inner.obs == ["restarted"]
    assert obs == ["restarted"] and infos["admissible_commands"] == [["look"]]


def test_generation_parity_uses_exact_sampling_args_and_real_projection(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import eval_alfworld_prompting as runner

    class Params:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=Params))

    class Tokenizer:
        def decode(self, ids, **kwargs): return "<action> take apple 1 </action>"

    class LLM:
        def __init__(self): self.params = []
        def generate(self, prompts, sampling, **kwargs):
            self.params.append(sampling.kwargs)
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2])]) for _ in prompts]

    def projection(texts, menus, phase):
        assert len(texts) == len(menus) == 1
        assert menus[0] in (["take apple 1"], ["look"])
        return ["take apple 1" if menus[0] == ["take apple 1"] else "look"], [1]

    llm = LLM()
    report = runner._generation_parity_check(
        llm, Tokenizer(), [{"prompt_token_ids": [10]}, {"prompt_token_ids": [11]}],
        projection, [["take apple 1"], ["look"]], Params(**runner.UPSTREAM_AGENT_SAMPLING),
        tmp_path / "parity.json",
    )
    assert report["equivalent"] is True
    assert llm.params[0] == llm.params[1]
    assert llm.params[0]["logprobs"] == 0 and llm.params[0]["seed"] == 20261015
    assert json.loads((tmp_path / "parity.json").read_text())["adapter_sampling_params"] == llm.params[1]


def test_mock_two_slot_runner_keeps_histories_separate_across_turns_and_retries(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import eval_alfworld_prompting as runner
    monkeypatch.syspath_prepend(str(ROOT / "selfplay" / "src"))
    from agent_system.environments.alfworld.prompt import ALFWORLD_PLAY_PROMPT, get_alfworld_prompt
    from agent_system.environments.alfworld.memory import SimpleMemory
    from agent_system.environments.alfworld.projection import alfworld_projection

    pool = [f"/tmp/task-{i}.tw-pddl" for i in range(9)]
    pool_ids = [game_id(p) for p in pool]
    manifest = make_selection_manifest(
        pool, lambda p: TASK_TYPES[int(Path(p).stem.split("-")[1]) % len(TASK_TYPES)],
        pool_order_sha256=sha256_bytes(json.dumps(pool_ids, separators=(",", ":")).encode()),
        data_config_sha256=runner._sha_file(runner.UPSTREAM_CONFIG),
        source_manifest_sha256=runner._sha_file(runner.SOURCE_MANIFEST),
    )
    rows = manifest["slots"][:2]
    prompts_seen = []
    original_prompt = runner._prompt

    def capture_prompt(*args, **kwargs):
        prompt = original_prompt(*args, **kwargs)
        prompts_seen.append(prompt)
        return prompt

    monkeypatch.setattr(runner, "_prompt", capture_prompt)

    class Tokenizer:
        name_or_path = "mock-tokenizer"
        chat_template = "mock-template"
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return f"rendered:{messages[0]['content']}"
        def encode(self, text, **kwargs): return [len(text)]
        def decode(self, ids, **kwargs): return "<action> look </action>"

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        SamplingParams=lambda **kwargs: SimpleNamespace(kwargs=kwargs)))

    class Model:
        def generate(self, prompts, sampling, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[7])]) for _ in prompts]

    monkeypatch.setattr(runner, "_engine", lambda _: (Model(), object()))
    monkeypatch.setattr(runner, "_upstream_imports", lambda: (
        ALFWORLD_PLAY_PROMPT, get_alfworld_prompt, SimpleMemory, alfworld_projection))

    class FakeTW:
        def __init__(self, slot):
            self.slot = slot
            self.last_commands = [None]
            self.obs = None
            self.steps = 0
            self.batch_env = self
            self.reset_count = 0
        def reset(self):
            self.steps = 0
            self.obs = [f"Welcome slot-{self.slot}\n\nKitchen\n\nYour task is: put apple in fridge"]
            return self.obs, {"extra.gamefile": [rows[self.slot]["game_file"]],
                              "admissible_commands": [["look"]]}
        def step(self, actions):
            assert actions == ["look"]
            self.steps += 1
            obs = [f"obs-slot-{self.slot}-attempt-{self.reset_count}-turn-{self.steps}"]
            won = self.slot == 0 and self.reset_count == 0 and self.steps >= 2
            done = self.steps >= 2
            infos = {"won": [won], "admissible_commands": [["look"]]}
            return obs, [0.0], [done], infos
        def close(self): pass

    class WrappedEnv:
        def __init__(self, slot):
            self.env = FakeTW(slot)
            self.slot = slot
        def seed(self, seed): assert seed == rows[self.slot]["worker_seed"]
        def reset(self):
            self.env.reset()
            self.env.reset_count = 0
            return self.env.obs, {"extra.gamefile": [rows[self.slot]["game_file"]],
                                  "admissible_commands": [["look"]]}
        def step(self, actions): return self.env.step(actions)
        def close(self): pass

    class FakeBase:
        def __init__(self): self.next_slot = 0
        def init_env(self, batch_size):
            assert batch_size == 1
            slot = self.next_slot
            self.next_slot += 1
            return WrappedEnv(slot)

    base_env = FakeBase()
    monkeypatch.setattr(runner, "_config_pool", lambda: ({}, pool, base_env))

    # Reset attempts go through the TextWorld wrapper's core, as in upstream Worker.restart.
    original_restart = runner._restart_same_game
    def restart(env):
        env.env.reset_count += 1
        return original_restart(env)
    monkeypatch.setattr(runner, "_restart_same_game", restart)

    args = SimpleNamespace(samples=1, rank=0, world_size=8, preflight=True,
                           base=str(tmp_path), style="react", out_dir=tmp_path,
                           run_id="mock-run", selection_manifest=tmp_path / "selection.json")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    result = runner.run_rank(args, manifest)
    assert result["generation_parity"]["equivalent"] is True
    assert len(result["rows"]) == 2
    assert all(len(row["attempts"]) == 3 for row in result["rows"])
    assert result["rows"][0]["attempts"][0]["won"] is True
    assert all(a["skipped_after_earlier_win"] and a["steps"] == 0
               for a in result["rows"][0]["attempts"][1:])
    assert all(not a["won"] for a in result["rows"][1]["attempts"])
    for slot in range(2):
        turn_one = [p for p in prompts_seen if f"Welcome slot-{slot}" in p and "Action 1:" in p]
        assert len(turn_one) == (1 if slot == 0 else 3)
        for prompt in turn_one:
            assert f"Observation 1: obs-slot-{slot}-attempt-" in prompt
            assert f"Observation 1: ['obs-slot-{slot}" not in prompt
            assert f"obs-slot-{1-slot}-attempt-" not in prompt


def test_selection_manifest_has_128_seeded_slots_and_preserves_duplicates():
    manifest = _manifest()
    validate_selection_manifest(manifest)
    assert len(manifest["slots"]) == SLOT_COUNT
    assert [r["slot"] for r in manifest["slots"]] == list(range(SLOT_COUNT))
    assert [r["worker_seed"] for r in manifest["slots"]] == list(range(SLOT_SEED_BASE, SLOT_SEED_BASE + SLOT_COUNT))
    assert len({r["game_file"] for r in manifest["slots"]}) < SLOT_COUNT
    for rank in range(SHARD_COUNT):
        shard = shard_slots(manifest, rank)
        assert [x["slot"] for x in shard] == list(range(rank * 16, (rank + 1) * 16))


def test_attempt_zero_weighted_pass1_is_distinct_from_six_type_macro():
    manifest = _manifest()
    metrics = aggregate_rows(_rows(manifest))
    p1 = metrics["pass_at_1"]
    assert p1["episode_weighted_sr"] == pytest.approx(1 / 128)
    if len(p1["per_type_sr"]) < 6:
        assert p1["six_type_macro_sr"] is None
        assert p1["missing_task_types"]
    else:
        assert p1["six_type_macro_sr"] != p1["episode_weighted_sr"]
    assert "cumulative Pass@2" in metrics["pass_at_2"]["label"]


def test_shard_merge_requires_complete_grid_and_correct_slot_identity():
    manifest = _manifest()
    shards = _shards(manifest)
    rows = validate_shard_grid(shards, manifest)
    assert len(rows) == 128
    mutated = _shards(manifest)
    mutated[0]["rows"][0]["task_type"] = TASK_TYPES[-1]
    with pytest.raises(ProtocolError, match="identity"):
        validate_shard_grid(mutated, manifest)
    with pytest.raises(ProtocolError, match="Expected 8"):
        validate_shard_grid(shards[:-1], manifest)
    mixed = _shards(manifest)
    mixed[1]["style"] = "zero_shot"
    with pytest.raises(ProtocolError, match="incompatible"):
        validate_shard_grid(mixed, manifest)
    provenance_mismatch = _shards(manifest)
    provenance_mismatch[1]["prompt_provenance"] = {
        "paper_exact": False, "kind": "user-approved derived approximation", "source": "other"}
    with pytest.raises(ProtocolError, match="incompatible"):
        validate_shard_grid(provenance_mismatch, manifest)
    compat_mismatch = _shards(manifest)
    compat_mismatch[1]["glibc_compat"] = {"mode": "patched", "path": "/different/shim"}
    with pytest.raises(ProtocolError, match="incompatible"):
        validate_shard_grid(compat_mismatch, manifest)


def test_selection_local_hashes_reject_pool_config_and_source_drift(monkeypatch):
    import eval_alfworld_prompting as runner

    pool = [f"/tmp/local-task-{i}.tw-pddl" for i in range(9)]
    ids = [game_id(p) for p in pool]
    manifest = make_selection_manifest(
        pool, lambda p: TASK_TYPES[int(Path(p).stem.split("-")[-1]) % len(TASK_TYPES)],
        pool_order_sha256=sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
        data_config_sha256=runner._sha_file(runner.UPSTREAM_CONFIG),
        source_manifest_sha256=runner._sha_file(runner.SOURCE_MANIFEST),
    )
    runner.validate_selection_against_local(manifest, pool)
    with pytest.raises(ProtocolError, match="pool_order_sha256"):
        runner.validate_selection_against_local(manifest, pool[1:] + pool[:1])

    original_sha_file = runner._sha_file
    def changed_config_sha(path):
        return "0" * 64 if Path(path) == runner.UPSTREAM_CONFIG else original_sha_file(path)
    monkeypatch.setattr(runner, "_sha_file", changed_config_sha)
    with pytest.raises(ProtocolError, match="data_config_sha256"):
        runner.validate_selection_against_local(manifest, pool)

    def changed_source_sha(path):
        return "0" * 64 if Path(path) == runner.SOURCE_MANIFEST else original_sha_file(path)
    monkeypatch.setattr(runner, "_sha_file", changed_source_sha)
    with pytest.raises(ProtocolError, match="source_manifest_sha256"):
        runner.validate_selection_against_local(manifest, pool)


def test_zero_shot_merge_preserves_provenance_and_summary_path(tmp_path, capsys):
    import eval_alfworld_prompting as runner
    from summarize import summarize

    manifest = _manifest()
    runner._write_json(tmp_path / "selection.json", manifest)
    shards = _shards(manifest)
    for shard in shards:
        shard["style"] = "zero_shot"
        shard["prompt_provenance"] = {
            "paper_exact": False,
            "kind": "user-approved derived approximation",
            "source": "upstream play prompt with two reasoning instructions removed",
        }
        shard["glibc_compat"] = {"mode": "native_fast_downward", "path": None}
        runner._write_json(tmp_path / f"rank-{shard['rank']:02d}.json", shard)

    merged = runner.merge_rank_outputs(tmp_path, tmp_path / "selection.json")
    merged_path = tmp_path / "merged.json"
    assert merged["style"] == "zero_shot"
    assert "user-approved derived prompt approximation" in merged["protocol_status"]
    assert merged["prompt_provenance"] == shards[0]["prompt_provenance"]
    assert merged["glibc_compat"] == shards[0]["glibc_compat"]
    summary = summarize(merged_path)
    assert summary == merged["metrics"]["pass_at_1"]
    assert "Six-type macro" in capsys.readouterr().out


def test_preflight_requires_full_prompt_generation_action_and_observation_trace():
    event = {"prompt": "p", "raw_output": "raw", "observation_before": "before", "observation": "obs",
             "executed_action": "look", "valid_action_tag": False,
             "projected_is_admissible": False}
    result = {"generation_parity": {"equivalent": True},
              "rows": [{"attempts": [{"trace": [event]}]}, {"attempts": [{"trace": [event]}]}]}
    report = verify_preflight(result)
    assert report["diagnosis"] == "low_tag_rate_review"
    del event["observation"]
    with pytest.raises(ProtocolError, match="evidence"):
        verify_preflight(result)
