from __future__ import annotations

import json
import os
import random
import socket
import sys
from pathlib import Path

import pytest
import numpy as np

SELFPLAY = Path(__file__).resolve().parents[1]
ROOT = SELFPLAY.parent
SRC = SELFPLAY / "src"
sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

import ez_train  # noqa: E402
from ez_train_v2_contract import (  # noqa: E402
    ContractError,
    contract_signature,
    seal_checkpoint,
    sha256_file,
    validate_checkpoint_v3,
    validate_sealed_integrity,
)


def _v2_gloo_fresh_resume_worker(rank, init_method, run_dir, manifest, result_dir):
    import torch.distributed as dist
    from ez_train_v2 import _dist_preflight
    from ez_train_v2_contract import seal_checkpoint

    world = 2
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    run_dir = Path(run_dir)
    resolved_manifest, selected, resolved_dir = _dist_preflight(
        manifest, run_dir, "gloo-run", None)
    assert resolved_dir == run_dir and selected is None
    dist.barrier()
    if rank == 0:
        ckpt = run_dir / "ckpt"
        ckpt.mkdir()
        (ckpt / "solver").mkdir()
        (ckpt / "proposer").mkdir()
        for adapter in ("solver", "proposer"):
            (ckpt / adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (ckpt / adapter / "adapter_model.safetensors").write_bytes(b"adapter-" + adapter.encode())
        (ckpt / "optimizer.pt").write_bytes(b"optimizer")
        (ckpt / "task_pool.jsonl").write_text("", encoding="utf-8")
        (ckpt / "training_manifest.json").write_text(
            json.dumps(resolved_manifest, sort_keys=True), encoding="utf-8")
        (ckpt / "rank_state").mkdir()
        for worker_rank in range(world):
            (ckpt / "rank_state" / f"rank{worker_rank}.pt").write_bytes(b"rng")
        state = {
            "checkpoint_schema_version": 3, "contract_sha256": manifest["contract_sha256"],
            "world_size": world, "dry_run": False,
            "next_position": {"outer": 0, "phase": "proposer", "step": 1},
            "next_outer": 0, "next_phase": "proposer", "next_step": 1,
            "run_id": "gloo-run",
        }
        (ckpt / "train_state.json").write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        seal_checkpoint(ckpt, manifest["contract_sha256"])
    dist.barrier()
    resumed_manifest, selected, resumed_dir = _dist_preflight(
        manifest, run_dir, "gloo-run", run_dir / "ckpt")
    assert selected == (run_dir / "ckpt").resolve() and resumed_dir == run_dir
    assert resumed_manifest["contract_sha256"] == manifest["contract_sha256"]
    fresh_rejected = False
    try:
        _dist_preflight(manifest, run_dir, "gloo-run", None)
    except ContractError:
        fresh_rejected = True
    assert fresh_rejected
    (Path(result_dir) / f"rank{rank}.json").write_text(
        json.dumps({"selected": str(selected), "rank": rank}), encoding="utf-8")
    dist.destroy_process_group()


class _Tokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer_config.json").write_text('{"model_max_length":64}\n', encoding="utf-8")


def _tiny_dual_adapter():
    torch.manual_seed(123)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
    ))
    lora = LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, lora, adapter_name="proposer")
    model.add_adapter("solver", lora)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    for adapter in ("proposer", "solver"):
        ez_train.select_adapter(model, adapter)
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _manifest(smoke=False):
    manifest = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_training_v2",
        "checkpoint_eligibility": "smoke_only" if smoke else "formal_training",
        "resolved_config": {
            "run": {"name": "explorer-zero-v2"},
            "train": {"outer_loops": 1, "proposer_steps": 2, "solver_steps": 2},
            "runtime": {"world_size": 1},
        },
        "identity": {"base_inventory_sha256": "a" * 64},
    }
    manifest["contract_sha256"] = contract_signature(manifest)
    return manifest


@pytest.mark.parametrize("active", ["proposer", "solver"])
def test_v3_peft_checkpoint_save_reload_and_resume(tmp_path, active):
    model, optimizer = _tiny_dual_adapter()
    ez_train.select_adapter(model, active)
    manifest = _manifest()
    run_dir = tmp_path / "explorer-zero-v2-tiny"
    run_dir.mkdir()
    (run_dir / "run_protocol.json").write_text(json.dumps({
        "run_id": "tiny", "contract_sha256": manifest["contract_sha256"]
    }), encoding="utf-8")
    ez_train.ez_save_v3(
        model, _Tokenizer(), optimizer, run_dir, "outer0-P0", {},
        {"next_outer": 0, "next_phase": "proposer", "next_step": 1},
        [], 0, 1, uid_counter=9, dry_run=False, protocol_manifest=manifest,
    )
    assert model.active_adapter == active
    assert {n for n, p in model.named_parameters() if "lora_" in n and p.requires_grad} == {
        n for n, p in model.named_parameters() if "lora_" in n and active in n
    }
    checkpoint = run_dir / "ckpt"
    info = validate_checkpoint_v3(checkpoint, expected_contract=manifest, purpose="resume")
    assert info["state"]["next_position"] == {"outer": 0, "phase": "proposer", "step": 1}
    assert info["solver_path"] == str((checkpoint / "solver").resolve())
    assert torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)["state"]
    state, pool, rng_path = ez_train.load_ez_resume(checkpoint, {}, 1, 0, protocol_manifest=manifest)
    assert state["run_id"] == "tiny" and pool == [] and rng_path.is_file()

    torch.manual_seed(123)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
    ))
    loaded = PeftModel.from_pretrained(base, checkpoint / "proposer",
                                       adapter_name="proposer", is_trainable=True)
    loaded.load_adapter(checkpoint / "solver", adapter_name="solver", is_trainable=True)
    for adapter in ("proposer", "solver"):
        expected = {n: p.detach().cpu() for n, p in model.named_parameters()
                    if "lora_" in n and adapter in n}
        actual = {n: p.detach().cpu() for n, p in loaded.named_parameters()
                  if "lora_" in n and adapter in n}
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)

    loaded_optimizer = torch.optim.AdamW(
        [p for n, p in loaded.named_parameters() if "lora_" in n], lr=1e-3)
    saved_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(saved_rng["python"])
    np.random.set_state(saved_rng["numpy"])
    torch.set_rng_state(saved_rng["torch_cpu"])
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    restored_uid = ez_train.restore_ez_optimizer_rng(
        loaded_optimizer, checkpoint / "optimizer.pt", rng_path)
    actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert restored_uid == 9
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)

    fixed_ids = torch.tensor([[1, 4, 5, 6, 2]])
    ez_train.select_adapter(model, "proposer")
    ez_train.select_adapter(loaded, "proposer")
    for target, target_optimizer in ((model, optimizer), (loaded, loaded_optimizer)):
        target_optimizer.zero_grad(set_to_none=True)
        target(input_ids=fixed_ids, labels=fixed_ids).loss.backward()
        target_optimizer.step()
    for name, expected_param in model.named_parameters():
        if "lora_" in name:
            actual_param = dict(loaded.named_parameters())[name]
            torch.testing.assert_close(expected_param, actual_param, rtol=0, atol=0)


def test_smoke_checkpoint_is_sealed_but_not_resume_or_eval_eligible(tmp_path):
    model, optimizer = _tiny_dual_adapter()
    manifest = _manifest(smoke=True)
    run_dir = tmp_path / "explorer-zero-v2-smoke"
    run_dir.mkdir()
    ez_train.ez_save_v3(
        model, _Tokenizer(), optimizer, run_dir, "outer0-S0", {},
        {"next_outer": 1, "next_phase": "proposer", "next_step": 0},
        [], 0, 1, uid_counter=0, dry_run=True, protocol_manifest=manifest,
    )
    checkpoint = run_dir / "ckpt"
    assert validate_sealed_integrity(checkpoint)["state"]["dry_run"] is True
    with pytest.raises(ContractError, match="dry-run"):
        validate_checkpoint_v3(checkpoint, purpose="resume")
    with pytest.raises(ContractError, match="dry-run"):
        validate_checkpoint_v3(checkpoint, purpose="evaluation")


@pytest.mark.parametrize("position", [
    {"outer": 0, "phase": "proposer", "step": 2},
    {"outer": 1, "phase": "solver", "step": 0},
])
def test_v3_position_rejects_completed_phase_step(tmp_path, position):
    from ez_train_v2_contract import _validate_position

    with pytest.raises(ContractError):
        _validate_position({"next_position": position,
                            "next_outer": position["outer"],
                            "next_phase": position["phase"],
                            "next_step": position["step"]}, _manifest())


def test_v3_position_aliases_must_match_verified_next_position(tmp_path):
    from ez_train_v2_contract import _validate_position

    state = {"next_position": {"outer": 0, "phase": "proposer", "step": 1},
             "next_outer": 0, "next_phase": "solver", "next_step": 0}
    with pytest.raises(ContractError, match="disagrees"):
        _validate_position(state, _manifest())


def test_resealed_checkpoint_rejects_conflicting_position_aliases(tmp_path):
    model, optimizer = _tiny_dual_adapter()
    manifest = _manifest()
    run_dir = tmp_path / "explorer-zero-v2-tiny"
    run_dir.mkdir()
    ez_train.ez_save_v3(
        model, _Tokenizer(), optimizer, run_dir, "outer0-P0", {},
        {"next_outer": 0, "next_phase": "proposer", "next_step": 1},
        [], 0, 1, uid_counter=0, dry_run=False, protocol_manifest=manifest,
    )
    checkpoint = run_dir / "ckpt"
    state_path = checkpoint / "train_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["next_phase"] = "solver"
    state["next_step"] = 0
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    seal_checkpoint(checkpoint, manifest["contract_sha256"])
    with pytest.raises(ContractError, match="disagrees"):
        validate_checkpoint_v3(checkpoint, purpose="resume")


def test_v2_actual_generation_config_constructors_consume_explicit_yaml():
    import yaml

    cfg = yaml.safe_load((SELFPLAY / "configs/ez_repro_v2.yaml").read_text(encoding="utf-8"))
    p_cfg, s_cfg, p_tok = ez_train.build_ez_rollout_configs(cfg, local_group_size=2)
    assert (p_cfg.group_size, p_cfg.max_turns, p_cfg.max_new_tokens) == (2, 30, 1024)
    assert (p_cfg.top_k, p_cfg.min_p, p_cfg.max_context_tokens,
            p_cfg.keep_recent_turns, p_cfg.seed) == (-1, 0.0, 40960, 40, None)
    assert (s_cfg.max_steps, s_cfg.max_new_tokens, s_cfg.top_p,
            s_cfg.top_k, s_cfg.min_p, s_cfg.repetition_penalty) == (30, 1024, 0.95, -1, 0.0, 1.0)
    assert s_cfg.keep_recent_turns == 12 and s_cfg.seed is None
    assert p_tok.max_new_tokens == p_cfg.max_new_tokens
    assert p_tok.max_context_tokens == p_cfg.max_context_tokens
    assert p_tok.keep_recent_turns == p_cfg.keep_recent_turns


def test_base_inventory_accepts_hf_style_file_symlinks_and_rejects_broken(tmp_path):
    from ez_train_v2_contract import base_file_inventory

    base = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    base.mkdir()
    blobs.mkdir()
    (base / "config.json").write_text('{"model_type":"llama"}\n', encoding="utf-8")
    shard = blobs / "weights.safetensors"
    shard.write_bytes(b"tiny weights")
    try:
        (base / "model.safetensors").symlink_to(shard)
    except (OSError, NotImplementedError):
        pytest.skip("file symlink creation is unavailable on this Windows account")
    inventory = base_file_inventory(base)
    assert inventory["model.safetensors"]["symlink"] is True
    assert inventory["model.safetensors"]["sha256"] == sha256_file(shard)
    (base / "broken.safetensors").symlink_to(blobs / "missing.safetensors")
    with pytest.raises(ContractError, match="broken or non-file symlink"):
        base_file_inventory(base)


def test_environment_file_identity_covers_logic_and_task_trajectory(tmp_path):
    from ez_train_v2_contract import environment_file_identity

    data = tmp_path / "data"
    game_dir = data / "json_2.1.1" / "train" / "pick_and_place" / "task1"
    logic_dir = data / "logic"
    game_dir.mkdir(parents=True)
    logic_dir.mkdir()
    config_path = tmp_path / "alfworld.yaml"
    config_path.write_text("config\n", encoding="utf-8")
    domain = logic_dir / "domain.pddl"
    grammar = logic_dir / "grammar.twl2"
    domain.write_text("domain-v1\n", encoding="utf-8")
    grammar.write_text("grammar-v1\n", encoding="utf-8")
    game = game_dir / "game.tw-pddl"
    game.write_text("game\n", encoding="utf-8")
    traj = game_dir / "traj_data.json"
    traj.write_text('{"task":"one"}\n', encoding="utf-8")
    identity = environment_file_identity(
        {"logic": {"domain": str(domain), "grammar": str(grammar)}},
        config_path, [game],
    )
    assert identity["logic_files"]["domain"]["sha256"] == sha256_file(domain)
    assert identity["logic_files"]["grammar"]["sha256"] == sha256_file(grammar)
    assert identity["task_trajectories"][0]["trajectory"]["sha256"] == sha256_file(traj)
    traj.unlink()
    identity_missing = environment_file_identity(
        {"logic": {"domain": str(domain), "grammar": str(grammar)}},
        config_path, [game],
    )
    assert identity_missing["task_trajectories"][0]["trajectory"] is None


def test_v2_rank_local_single_gpu_mask_and_duplicate_full_mask(tmp_path, monkeypatch):
    import ez_train_v2

    cfg = {"solver": {"style": "ours", "menu_enabled": False, "fewshot": False,
                       "parser_strict": False, "fallback_mode": "raw"}}
    base, data = tmp_path / "base", tmp_path / "data"
    base.mkdir()
    data.mkdir()
    for name in ez_train_v2.LEGACY_SEMANTIC_ENV + ez_train_v2.PROMPT_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("ZTWANG_MODEL_PATH", raising=False)
    monkeypatch.delenv("ALFWORLD_DATA", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a1b2")
    result = ez_train_v2._check_environment(cfg, base, data, "hf", 8)
    assert result["device_environment"]["device_mapping"] == {
        "launch_mode": "one_visible_gpu_per_rank", "rank": 3,
        "physical_device_mask": "GPU-a1b2",
    }
    monkeypatch.setenv("LOCAL_RANK", "3")
    with pytest.raises(ContractError, match="LOCAL_RANK=0"):
        ez_train_v2._check_environment(cfg, base, data, "hf", 8)
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,1,3,4,5,6,7")
    with pytest.raises(ContractError, match="repeated physical device"):
        ez_train_v2._check_environment(cfg, base, data, "hf", 8)


def test_two_rank_gloo_preflight_fresh_resume_and_no_overwrite(tmp_path):
    import torch.multiprocessing as mp

    from ez_train_v2_contract import contract_signature

    run_dir = tmp_path / "runs" / "explorer-zero-v2-gloo-run"
    result_dir = tmp_path / "worker-results"
    result_dir.mkdir()
    manifest = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_training_v2",
        "resolved_config": {
            "run": {"name": "explorer-zero-v2", "output_dir": str(tmp_path / "runs")},
            "train": {"outer_loops": 1, "proposer_steps": 2, "solver_steps": 2},
            "runtime": {"world_size": 2},
        },
        "identity": {"base_inventory_sha256": "b" * 64},
    }
    manifest["contract_sha256"] = contract_signature(manifest)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _v2_gloo_fresh_resume_worker,
        args=(f"tcp://127.0.0.1:{port}", str(run_dir), manifest, str(result_dir)),
        nprocs=2, join=True,
    )
    rows = [json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)]
    assert [row["rank"] for row in rows] == [0, 1]
    assert all(row["selected"] == str((run_dir / "ckpt").resolve()) for row in rows)


def test_legacy_entry_rejects_v3_config_before_model_loading(tmp_path, capsys):
    import pytest

    config = tmp_path / "v3.yaml"
    config.write_text("protocol_version: 3\nprotocol_name: explorer_zero_training_v2\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        ez_train.main(["--config", str(config)])
    assert exc.value.code == 2
    assert "strict ez_train_v2" in capsys.readouterr().err


def test_v2_vllm_engine_receives_rank_distinct_seed_and_explicit_limits():
    import yaml

    cfg = yaml.safe_load((SELFPLAY / "configs/ez_repro_v2.yaml").read_text(encoding="utf-8"))
    kwargs0 = ez_train.v2_vllm_engine_kwargs(cfg, 0)
    kwargs1 = ez_train.v2_vllm_engine_kwargs(cfg, 1)
    assert kwargs0["seed"] == 0 and kwargs1["seed"] == 1
    assert kwargs0["seed"] != kwargs1["seed"]
    assert kwargs0["max_model_len"] == cfg["solver"]["max_context_tokens"] == 40960
    assert kwargs0["max_lora_rank"] == cfg["model"]["lora_r"] == 32
    assert kwargs0["trust_remote_code"] is False


def test_grpo_loss_stats_are_exact_sequence_mean_pg_plus_beta_kl():
    from types import SimpleNamespace
    from train_grpo import BatchTensors, grpo_loss, per_token_logps

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(17, 8)
            self.head = torch.nn.Linear(8, 17, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False,
                    logits_to_keep=None):
            hidden = self.emb(input_ids)
            if logits_to_keep is not None:
                hidden = hidden[:, logits_to_keep, :]
            return SimpleNamespace(logits=self.head(hidden))

    torch.manual_seed(21)
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    loss_mask = torch.tensor([[0, 1, 1, 1, 0], [0, 1, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        old = per_token_logps(model, ids, attention)
        ref = old.clone()
        ref[0, 1] += 0.12
        ref[0, 2] -= 0.08
        ref[1, 0] += 0.2
    batch = BatchTensors(ids, attention, loss_mask,
                         torch.tensor([0.8, -0.35]), n_active=4)
    beta = 0.001
    mean_loss, stats = grpo_loss(model, batch, old, 0.3, ref_logps=ref,
                                 beta=beta, loss_norm="seq", sequence_reduction="mean")
    assert stats["valid_sequence_count"] == 2
    assert stats["kl_sequence_sum"] > 0
    expected_sum = stats["pg_sequence_sum"] + beta * stats["kl_sequence_sum"]
    assert stats["objective_sequence_sum"] == pytest.approx(expected_sum, abs=1e-7)
    assert float(mean_loss.detach()) == pytest.approx(expected_sum / 2, abs=1e-6)
    summed_loss, summed_stats = grpo_loss(
        model, batch, old, 0.3, ref_logps=ref, beta=beta,
        loss_norm="seq", sequence_reduction="sum")
    assert float(summed_loss.detach()) == pytest.approx(expected_sum, abs=1e-6)
    assert summed_stats["objective_sequence_sum"] == pytest.approx(expected_sum, abs=1e-7)


def test_v3_metrics_are_persisted_locally_when_swanlab_is_disabled(tmp_path, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "swanlab":
            raise AssertionError("SwanLab must not be imported when the resolved config disables it")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setenv("EZ_SWANLAB_MODE", "cloud")
    reporter = ez_train._V2MetricsReporter(
        {"telemetry": {"swanlab": False}}, tmp_path, rank=0)
    reporter.log({"train/objective_mean": 0.5, "train/pg_mean": 0.4,
                  "train/kl_mean": 0.1, "bad": float("nan")}, step=7)
    reporter.finish()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows == [{"step": 7, "metrics": {"train/objective_mean": 0.5,
                                              "train/pg_mean": 0.4,
                                              "train/kl_mean": 0.1}}]
