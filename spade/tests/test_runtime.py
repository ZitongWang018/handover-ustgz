import json
from pathlib import Path

import pytest

import runtime
import run_tau2


@pytest.fixture
def local_model(tmp_path):
    root = tmp_path / "ztwang"
    model = root / "models/qwen"
    model.mkdir(parents=True)
    config = {"model_type": "qwen3_moe", "num_hidden_layers": 48, "hidden_size": 2048,
              "num_attention_heads": 32, "num_key_value_heads": 4, "num_experts": 128,
              "num_experts_per_tok": 8, "vocab_size": 151936, "rope_theta": 10000000,
              "head_dim": 128, "moe_intermediate_size": 768}
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model / "tokenizer_config.json").write_text('{"chat_template":"local test template"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture never loaded into GPU")
    return root.resolve(), model.resolve()


def test_model_identity_and_missing_shard_are_checked(local_model):
    root, model = local_model
    assert runtime.check_model(model, root)["architecture"]["num_key_value_heads"] == 4
    index = model / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"layer":"missing.safetensors"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        runtime.check_model(model, root)


def test_ztwang_prefix_does_not_authorize_sibling(local_model):
    root, _ = local_model
    with pytest.raises(ValueError, match="outside"):
        runtime.bounded(root.parent / "ztwang-other/file", root)


@pytest.mark.parametrize("url", ["http://user:password@localhost", "http://localhost?key=example", "file:///tmp/a", "http://localhost/v1/v1"])
def test_endpoint_rejects_unsafe_or_duplicate_path(url):
    with pytest.raises(ValueError):
        runtime.endpoint_url(url)


def test_eight_card_server_uses_official_sglang_tool_parser(local_model):
    _, model = local_model
    command = run_tau2.server_command(model, "http://127.0.0.1:30000/v1", "qwen")
    assert command[1:3] == ["-m", "sglang.launch_server"]
    assert command[command.index("--tp") + 1] == "8"
    assert command[command.index("--dp") + 1] == "1"
    assert command[command.index("--tool-call-parser") + 1] == "qwen"
    assert "pkill" not in command and "ray" not in command


def test_dry_run_does_not_create_outputs_or_call_endpoint(local_model, monkeypatch, capsys):
    root, model = local_model
    source = root / "repo/spade"
    source.mkdir(parents=True)
    (source / "tau2_config.yaml").write_text("tau2_eval: {}", encoding="utf-8")
    monkeypatch.setattr(run_tau2, "BASE", source)
    monkeypatch.setattr(run_tau2, "verify_pins", lambda: dict(runtime.PINS))
    monkeypatch.setattr(run_tau2, "urlopen", lambda *a, **k: pytest.fail("dry run used network"))
    monkeypatch.setattr(run_tau2, "serve", lambda *a: pytest.fail("dry run launched server"))
    output = root / "runs/preview"
    assert run_tau2.main(["evaluate", "--root", str(root), "--model", str(model), "--output", str(output)]) == 0
    assert not output.exists()
    assert json.loads(capsys.readouterr().out)["execute"] is False


def test_cache_locations_stay_in_owned_run(local_model):
    root, _ = local_model
    output = root / "runs/test"
    output.mkdir(parents=True)
    env = runtime.cache_environment(output)
    for key, value in env.items():
        if key.endswith("DIR") or key in {"TEMP", "TMP", "TORCH_HOME", "HF_HOME", "HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE", "CUDA_CACHE_PATH", "HUGGINGFACE_HUB_CACHE"}:
            if key != "TAU2_DATA_DIR":
                assert Path(value).resolve().is_relative_to(output)
    assert "HOME" not in env and "CODEX_HOME" not in env
    assert env["HF_HUB_OFFLINE"] == "1"


def test_bundled_source_and_task_pool_pins_are_exact():
    assert runtime.verify_pins() == runtime.PINS
    for domain, count in runtime.DOMAINS.items():
        folder = runtime.BASE / "tau2-bench/data/tau2/domains" / domain
        split = json.loads((folder / "split_tasks.json").read_text(encoding="utf-8"))["base"]
        assert len(split) == len(set(split)) == count
    assert sum(runtime.DOMAINS.values()) * 4 == 1112
