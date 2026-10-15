"""CPU regressions for the minimal shared-protocol trainer."""
from __future__ import annotations

import sys
import json
import os
import random
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
SELFPLAY = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SELFPLAY))

torch = pytest.importorskip("torch")

from ez_train import backward_microbatches
from grpo import per_token_logps
from reward import TrajectoryStats
from rollout import Trajectory
from checkpoint import contract_signature


def _gloo_preflight_worker(rank, init_method, run_dir, manifest, result_dir):
    import os
    import torch.distributed as dist
    from checkpoint import seal_checkpoint
    from train import ContractError, _dist_preflight

    world = 2
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = "0"
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    run_dir = Path(run_dir)
    resolved, selected, resolved_dir = _dist_preflight(manifest, run_dir, "gloo-run", None)
    assert resolved_dir == run_dir and selected is None
    dist.barrier()
    if rank == 0:
        checkpoint = run_dir / "ckpt"
        checkpoint.mkdir()
        for adapter in ("solver", "proposer"):
            folder = checkpoint / adapter
            folder.mkdir()
            (folder / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (folder / "adapter_model.safetensors").write_bytes(b"adapter-" + adapter.encode())
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "task_pool.jsonl").write_text("", encoding="utf-8")
        (checkpoint / "training_manifest.json").write_text(
            json.dumps(resolved, sort_keys=True), encoding="utf-8")
        rank_state = checkpoint / "rank_state"
        rank_state.mkdir()
        for worker_rank in range(world):
            (rank_state / f"rank{worker_rank}.pt").write_bytes(b"rng")
        state = {
            "checkpoint_schema_version": 3,
            "contract_sha256": manifest["contract_sha256"],
            "world_size": world,
            "dry_run": False,
            "next_position": {"outer": 0, "phase": "proposer", "step": 1},
            "next_outer": 0,
            "next_phase": "proposer",
            "next_step": 1,
            "run_id": "gloo-run",
        }
        (checkpoint / "train_state.json").write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8")
        seal_checkpoint(checkpoint, manifest["contract_sha256"])
    dist.barrier()
    resumed, selected, resolved_dir = _dist_preflight(
        manifest, run_dir, "gloo-run", run_dir / "ckpt")
    assert selected == (run_dir / "ckpt").resolve()
    assert resolved_dir == run_dir and resumed["contract_sha256"] == manifest["contract_sha256"]
    try:
        _dist_preflight(manifest, run_dir, "gloo-run", None)
    except ContractError:
        pass
    else:
        raise AssertionError("fresh run must refuse to overwrite its existing directory")
    (Path(result_dir) / f"rank{rank}.json").write_text(
        json.dumps({"selected": str(selected), "rank": rank}), encoding="utf-8")
    dist.destroy_process_group()


def test_two_rank_gloo_fresh_resume_and_no_overwrite(tmp_path):
    import torch.multiprocessing as mp

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    run_dir = tmp_path / "runs" / "explorer-zero-gloo-run"
    result_dir = tmp_path / "worker-results"
    result_dir.mkdir()
    manifest = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_skillrise",
        "resolved_config": {
            "run": {"name": "explorer-zero", "output_dir": str(tmp_path / "runs")},
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
        _gloo_preflight_worker,
        args=(f"tcp://127.0.0.1:{port}", str(run_dir), manifest, str(result_dir)),
        nprocs=2, join=True,
    )
    rows = [json.loads((result_dir / f"rank{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)]
    assert [row["rank"] for row in rows] == [0, 1]
    assert all(row["selected"] == str((run_dir / "ckpt").resolve()) for row in rows)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(23, 9)
        self.head = torch.nn.Linear(9, 23)
        self.seen = []

    def forward(self, input_ids, attention_mask=None, use_cache=False,
                logits_to_keep=None):
        self.seen.append(input_ids.detach().cpu().clone())
        hidden = self.embedding(input_ids)
        if logits_to_keep is not None:
            hidden = hidden[:, logits_to_keep, :]
        return SimpleNamespace(logits=self.head(hidden))


def _traj(*, segments=(), input_ids=(), loss_mask=(), advantage=0.6):
    return Trajectory(
        task_id="task", task_type="pick", game_file="game.json",
        messages=[], input_ids=list(input_ids), loss_mask=list(loss_mask),
        stats=TrajectoryStats(), advantage=advantage,
        n_tokens=sum(len(response) for _, response in segments),
        segments=[(list(prompt), list(response)) for prompt, response in segments],
    )


def test_solver_segment_gradient_matches_episode_token_weighted_reference():
    torch.manual_seed(91)
    segments = [([1, 2], [3, 4, 5, 6]), ([7, 8, 9], [10, 11])]
    traj = _traj(segments=segments, advantage=0.65)
    model = TinyLM()
    reference = TinyLM()
    reference.load_state_dict(model.state_dict())

    backward_microbatches(
        model, model, [traj], pad_id=0, device=torch.device("cpu"),
        adapter="solver", clip_eps=0.3, micro_batch_size=1, world=1,
        loss_norm="seq", beta=0.0,
    )

    episode_tokens = sum(len(response) for _, response in segments)
    reference_objective = 0.0
    for prompt, response in segments:
        ids = torch.tensor([prompt + response], dtype=torch.long)
        attention = torch.ones_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.float32)
        mask[:, len(prompt):] = 1.0
        logps = per_token_logps(reference, ids, attention, loss_mask=mask)
        token_logps = logps[mask[:, 1:].bool()]
        segment_objective = -traj.advantage * token_logps.mean()
        reference_objective = reference_objective + (
            segment_objective * len(response) / episode_tokens)
    reference_objective.backward()

    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
    expected_inputs = [torch.tensor([prompt + response]) for prompt, response in segments]
    assert len(model.seen) == 2 * len(segments)
    for expected in expected_inputs:
        assert sum(torch.equal(seen, expected) for seen in model.seen) == 2
    assert all(seen.shape[1] < sum(len(p) + len(r) for p, r in segments)
               for seen in model.seen)


def test_proposer_trajectory_with_empty_segments_keeps_token_native_gradient():
    torch.manual_seed(17)
    model = TinyLM()
    trajectory = _traj(
        segments=[], input_ids=[1, 2, 3, 4, 5],
        loss_mask=[0, 0, 1, 1, 1], advantage=0.8)
    assert trajectory.segments == []
    assert trajectory.has_trainable_tokens

    loss, stats, valid = backward_microbatches(
        model, model, [trajectory], pad_id=0, device=torch.device("cpu"),
        adapter="proposer", clip_eps=0.3, micro_batch_size=1, world=1,
        loss_norm="seq", beta=0.0,
    )
    assert valid == 1 and stats[0]["valid_sequence_count"] == 1.0
    assert loss != 0.0
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)
               for parameter in model.parameters())


def test_shared_solver_sampling_settings_are_enforced():
    import train

    config = train._load_config(train.SELFPLAY_ROOT / "config.yaml")
    config["solver"]["temperature"] = 1.0
    with pytest.raises(train.ContractError, match="sampling"):
        train._validate_solver_sampling(config["solver"])


def test_real_peft_checkpoint_resumes_with_identical_adapter_weights(tmp_path):
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from checkpoint import ContractError, contract_signature, validate_checkpoint_v3
    from ez_train import (build_dual_lora, ez_save_v3, load_ez_resume,
                          restore_ez_optimizer_rng, select_adapter)

    torch.manual_seed(123)
    base = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
    ))
    config = {
        "model": {"lora_r": 2, "lora_alpha": 2, "lora_dropout": 0.0,
                  "lora_targets": ["q_proj", "v_proj"]},
    }
    model = build_dual_lora(base, config)
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "lora_" in n], lr=1e-3)
    ids = torch.tensor([[1, 4, 5, 6, 2]])
    for adapter in ("proposer", "solver"):
        select_adapter(model, adapter)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    select_adapter(model, "solver")
    active_before_save = model.active_adapter
    trainable_before_save = {
        name for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    }

    class TinyTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer_config.json").write_text(
                '{"model_max_length":64}\n', encoding="utf-8")

    manifest = {
        "schema_version": 3,
        "protocol_name": "explorer_zero_skillrise",
        "checkpoint_eligibility": "formal_training",
        "resolved_config": {
            "run": {"name": "explorer-zero"},
            "train": {"outer_loops": 1, "proposer_steps": 2, "solver_steps": 2},
            "runtime": {"world_size": 1},
        },
        "identity": {"base_inventory_sha256": "a" * 64},
    }
    manifest["contract_sha256"] = contract_signature(manifest)
    run_dir = tmp_path / "explorer-zero-tiny"
    run_dir.mkdir()
    ez_save_v3(
        model, TinyTokenizer(), optimizer, run_dir, "outer0-P0", {},
        {"next_outer": 0, "next_phase": "proposer", "next_step": 1},
        [], 0, 1, uid_counter=4, dry_run=False, protocol_manifest=manifest,
    )
    checkpoint = run_dir / "ckpt"
    validated = validate_checkpoint_v3(
        checkpoint, expected_contract=manifest, purpose="resume")
    assert validated["solver_path"] == str((checkpoint / "solver").resolve())
    assert model.active_adapter == active_before_save
    assert {
        name for name, parameter in model.named_parameters()
        if "lora_" in name and parameter.requires_grad
    } == trainable_before_save
    state, pool, rng_path = load_ez_resume(
        checkpoint, {}, world=1, rank=0, protocol_manifest=manifest)
    assert state["run_id"] == "tiny" and pool == [] and rng_path.is_file()
    assert torch.load(checkpoint / "optimizer.pt", map_location="cpu",
                      weights_only=False)["state"]

    saved_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    random.setstate(saved_rng["python"])
    np.random.set_state(saved_rng["numpy"])
    torch.set_rng_state(saved_rng["torch_cpu"])
    expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
    random.seed(987)
    np.random.seed(654)
    torch.manual_seed(321)

    torch.manual_seed(123)
    restored_base = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
    ))
    restored = build_dual_lora(restored_base, config, resume_checkpoint=checkpoint)
    for adapter in ("proposer", "solver"):
        expected = {name: value.detach().cpu() for name, value in model.named_parameters()
                    if "lora_" in name and adapter in name}
        actual = {name: value.detach().cpu() for name, value in restored.named_parameters()
                  if "lora_" in name and adapter in name}
        assert expected.keys() == actual.keys()
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
    restored_optimizer = torch.optim.AdamW(
        [p for n, p in restored.named_parameters() if "lora_" in n], lr=1e-3)
    uid = restore_ez_optimizer_rng(
        restored_optimizer, checkpoint / "optimizer.pt", rng_path)
    assert uid == 4 and len(restored_optimizer.state) > 0
    actual_rng = (random.random(), float(np.random.random()), torch.rand(3))
    assert expected_rng[:2] == actual_rng[:2]
    torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)

    for target, target_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        select_adapter(target, "solver")
        target_optimizer.zero_grad(set_to_none=True)
        target(input_ids=ids, labels=ids).loss.backward()
        target_optimizer.step()
    expected_lora = {name: value.detach().cpu() for name, value in model.named_parameters()
                     if "lora_" in name}
    actual_lora = {name: value.detach().cpu() for name, value in restored.named_parameters()
                   if "lora_" in name}
    assert expected_lora.keys() == actual_lora.keys()
    for name in expected_lora:
        torch.testing.assert_close(expected_lora[name], actual_lora[name], rtol=0, atol=0)

    with (checkpoint / "optimizer.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ContractError, match="hash mismatch"):
        validate_checkpoint_v3(checkpoint, purpose="resume")
