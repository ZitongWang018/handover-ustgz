from __future__ import annotations

import json
import multiprocessing
import os
import socket
import sys
from pathlib import Path
from types import ModuleType
from typing import Literal, Optional, Type, Union

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import training_runtime
from training_runtime import build_training_vllm_engine, clean_training_environment


def _report_child_environment(conn):
    names = (
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME", "MASTER_ADDR", "MASTER_PORT",
        "TORCHELASTIC_RUN_ID", "VLLM_DP_RANK", "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP",
        "VLLM_HOST_IP", "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V1", "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME", "OMP_NUM_THREADS", "HF_HOME",
    )
    conn.send({name: os.environ.get(name) for name in names})
    conn.close()


class _FakeEngineArgs:
    def __init__(self, tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 data_parallel_size: int = 1,
                 distributed_executor_backend: Optional[Union[str, Literal["uni", "mp"], type]] = "uni"):
        pass


_FakeEngineArgs.__init__.__annotations__["distributed_executor_backend"] = Optional[
    Union[str, Literal["uni", "mp"], Type[object]]]


class _FakeLLM:
    spawn_child = True
    fail = False
    received_kwargs = None
    child_env = None

    def __init__(self, **kwargs):
        type(self).received_kwargs = dict(kwargs)
        if type(self).spawn_child:
            ctx = multiprocessing.get_context("spawn")
            parent, child = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_report_child_environment, args=(child,))
            proc.start()
            child.close()
            type(self).child_env = parent.recv()
            proc.join(timeout=20)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                raise RuntimeError("fake spawned child did not exit")
            if proc.exitcode != 0:
                raise RuntimeError(f"fake spawned child exited with {proc.exitcode}")
        if type(self).fail:
            raise ValueError("synthetic engine initialization failure")


@pytest.fixture
def fake_vllm(monkeypatch):
    package = ModuleType("vllm")
    package.__path__ = []
    package.__version__ = "test-vllm-1"
    package.__file__ = "fake-vllm-package.py"
    package.LLM = _FakeLLM
    envs = ModuleType("vllm.envs")
    envs.environment_variables = {
        "VLLM_USE_V1": object(),
        "VLLM_ENABLE_V1_MULTIPROCESSING": object(),
        "VLLM_WORKER_MULTIPROC_METHOD": object(),
    }
    engine = ModuleType("vllm.engine")
    engine.__path__ = []
    arg_utils = ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = _FakeEngineArgs
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    _FakeLLM.spawn_child = True
    _FakeLLM.fail = False
    _FakeLLM.received_kwargs = None
    _FakeLLM.child_env = None
    return _FakeLLM


def _training_env(monkeypatch):
    values = {
        "RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "8", "GROUP_RANK": "0", "ROLE_RANK": "3",
        "ROLE_NAME": "trainer", "MASTER_ADDR": "training-master",
        "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "train-run",
        "VLLM_DP_RANK": "3", "VLLM_DP_SIZE": "8",
        "VLLM_DP_MASTER_IP": "training-master", "VLLM_DP_MASTER_PORT": "29600",
        "VLLM_HOST_IP": "10.9.0.5", "VLLM_PORT": "30100",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork", "VLLM_USE_V1": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "CUDA_VISIBLE_DEVICES": "2",
        "NCCL_SOCKET_IFNAME": "ib0", "OMP_NUM_THREADS": "5", "HF_HOME": "cache-root",
        "CUSTOM_TRAINING_SETTING": "keep-me",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def test_clean_training_environment_removes_launch_metadata_and_preserves_runtime(monkeypatch):
    source = {
        "RANK": "1", "WORLD_SIZE": "8", "ROLE_WORKER": "x",
        "TORCHELASTIC_RESTART_COUNT": "2", "VLLM_DP_RANK": "4",
        "MASTER_ADDR": "host", "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_SOCKET_IFNAME": "ib0", "OMP_NUM_THREADS": "8", "HF_HOME": "cache",
        "VLLM_HOST_IP": "192.0.2.4", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    cleaned = clean_training_environment(source)
    assert not ({"RANK", "WORLD_SIZE", "ROLE_WORKER", "TORCHELASTIC_RESTART_COUNT",
                 "VLLM_DP_RANK", "MASTER_ADDR"} & cleaned.keys())
    assert {key: cleaned[key] for key in (
        "CUDA_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "OMP_NUM_THREADS", "HF_HOME",
        "VLLM_HOST_IP", "VLLM_ENABLE_V1_MULTIPROCESSING")} == {
            "CUDA_VISIBLE_DEVICES": "0", "NCCL_SOCKET_IFNAME": "ib0",
            "OMP_NUM_THREADS": "8", "HF_HOME": "cache",
            "VLLM_HOST_IP": "192.0.2.4", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    assert source["RANK"] == "1"  # input mapping is not mutated


def test_engine_spawn_isolated_env_kwargs_preserved_and_training_gloo_pg_survives(
        tmp_path, monkeypatch, fake_vllm, capsys):
    torch = pytest.importorskip("torch")
    dist = pytest.importorskip("torch.distributed")
    original = _training_env(monkeypatch)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=0, world_size=1)
    try:
        engine_kwargs = {
            "gpu_memory_utilization": 0.35, "max_model_len": 32768,
            "max_lora_rank": 32, "max_num_seqs": 64,
            "max_num_batched_tokens": 8192, "enable_prefix_caching": True,
            "enforce_eager": False, "seed": 903, "trust_remote_code": False,
        }
        engine = build_training_vllm_engine("/models/base", rank=3,
                                            diagnostics_dir=tmp_path / "diag",
                                            **engine_kwargs)
        assert isinstance(engine, _FakeLLM)
        assert {key: os.environ.get(key) for key in original} == original

        child = _FakeLLM.child_env
        assert child is not None
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                     "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
                     "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
                     "VLLM_DP_RANK", "VLLM_DP_SIZE", "VLLM_DP_MASTER_IP"):
            assert child[name] is None
        assert child["VLLM_HOST_IP"] == "127.0.0.1"
        assert child["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
        assert child["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"
        assert child["VLLM_USE_V1"] == "1"
        assert child["CUDA_VISIBLE_DEVICES"] == "2"
        assert child["NCCL_SOCKET_IFNAME"] == "ib0"
        assert child["OMP_NUM_THREADS"] == "5"
        assert child["HF_HOME"] == "cache-root"
        startup_lines = [line for line in capsys.readouterr().out.splitlines()
                         if line.startswith("TRAINING_VLLM_STARTUP ")]
        assert len(startup_lines) == 1
        startup = json.loads(startup_lines[0].removeprefix("TRAINING_VLLM_STARTUP "))
        assert startup["rank"] == 3
        assert startup["device_mask"] == "2"
        assert startup["vllm_module_file"] == "fake-vllm-package.py"
        assert startup["topology"]["distributed_executor_backend"] == "uni"
        assert startup["generation_budgets"]["max_model_len"] == 32768
        assert startup["generation_budgets"]["gpu_memory_utilization"] == 0.35
        assert startup["diagnostics_dir"] == str(tmp_path / "diag")
        assert "training-master" not in startup_lines[0]

        forwarded = _FakeLLM.received_kwargs
        for key, value in engine_kwargs.items():
            assert forwarded[key] == value
        assert {key: forwarded[key] for key in (
            "model", "tokenizer", "dtype", "enable_lora", "max_loras",
            "max_cpu_loras", "disable_log_stats", "tensor_parallel_size",
            "pipeline_parallel_size", "data_parallel_size",
            "distributed_executor_backend")} == {
                "model": "/models/base", "tokenizer": "/models/base",
                "dtype": "bfloat16", "enable_lora": True, "max_loras": 2,
                "max_cpu_loras": 4, "disable_log_stats": True,
                "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
                "data_parallel_size": 1, "distributed_executor_backend": "uni"}

        value = torch.tensor([7], dtype=torch.int64)
        dist.all_reduce(value)
        assert value.item() == 7
    finally:
        dist.destroy_process_group()

    diagnostics = list((tmp_path / "diag").glob("training_vllm_rank3_*.json"))
    assert len(diagnostics) == 1
    record = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert record["vllm_version"] == "test-vllm-1"
    assert record["vllm_module_version"] == "test-vllm-1"
    assert record["vllm_module_file"] == "fake-vllm-package.py"
    assert record["device_mask"] == "2"
    assert record["topology"]["tensor_parallel_size"] == 1
    assert record["status"] == "success"
    assert "MASTER_ADDR" in record["removed_env_var_names"]
    assert "training-master" not in json.dumps(record)
    assert record["exception_traceback"] is None


def test_failure_restores_environment_and_writes_complete_traceback(
        tmp_path, monkeypatch, fake_vllm):
    original = _training_env(monkeypatch)
    _FakeLLM.spawn_child = False
    _FakeLLM.fail = True
    with pytest.raises(RuntimeError, match="no alternate backend"):
        build_training_vllm_engine(
            "/models/base", rank=3, diagnostics_dir=tmp_path,
            seed=903, gpu_memory_utilization=0.35, max_model_len=32768)
    assert {key: os.environ.get(key) for key in original} == original
    diagnostic = next(tmp_path.glob("training_vllm_rank3_*.json"))
    record = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert "synthetic engine initialization failure" in record["exception_traceback"]
    assert "ValueError" in record["exception_traceback"]
    assert record["status"] == "failed"
    assert _FakeLLM.received_kwargs["seed"] == 903
    assert _FakeLLM.received_kwargs["gpu_memory_utilization"] == 0.35
    assert record["vllm_version"] == "test-vllm-1"


def test_multi_device_mask_is_rejected_with_diagnostic_and_environment_restored(
        tmp_path, monkeypatch, fake_vllm):
    original = _training_env(monkeypatch)
    original["CUDA_VISIBLE_DEVICES"] = "2,3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    _FakeLLM.spawn_child = False
    with pytest.raises(RuntimeError) as error:
        build_training_vllm_engine("/models/base", rank=3, diagnostics_dir=tmp_path)
    assert "run_train.sh" in str(error.value.__cause__)
    assert {key: os.environ.get(key) for key in original} == original
    assert _FakeLLM.received_kwargs is None
    diagnostic = next(tmp_path.glob("training_vllm_rank3_*.json"))
    record = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["device_mask"] == "2,3"
    assert "run_train.sh" in record["exception_traceback"]

\n