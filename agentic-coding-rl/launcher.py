"""Build Miles commands without importing torch, Ray, or Miles itself."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from algorithms.dis_config import dis_settings

PROJECT = Path(__file__).resolve().parent
LOCK = json.loads((PROJECT / "UPSTREAM.lock.json").read_text(encoding="utf-8"))
ALGORITHMS = ("grpo", "grpo_dis", "gspo", "grpo_tis", "grpo_mis", "score_centering", "ppo_single")


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def workspace_path(config: dict, value: str | Path) -> Path:
    root = Path(config["workspace_root"]).expanduser().resolve()
    value = Path(value).expanduser()
    path = (value if value.is_absolute() else root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes workspace_root: {path}")
    return path


def check_workspace(config: dict) -> None:
    root = Path(config["workspace_root"]).expanduser().resolve()
    if root.name != "ztwang" or not root.is_dir() or "/path/to/" in str(root):
        raise ValueError("Set workspace_root to your confirmed, existing ztwang directory")
    if not PROJECT.is_relative_to(root):
        raise ValueError(f"Place this checkout inside the confirmed workspace: {root}")
    for value in config["paths"].values():
        workspace_path(config, value)


def paths(config: dict) -> dict[str, Path]:
    return {key: workspace_path(config, value) for key, value in config["paths"].items()}


def checkout(name: str) -> Path:
    return PROJECT / ".deps" / name


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def verify_sources() -> None:
    for name, pin in LOCK.items():
        source = checkout(name)
        if not (source / ".git").exists():
            raise ValueError(f"Missing {source}; run bootstrap first")
        if git_head(source) != pin["commit"]:
            raise ValueError(f"{name} HEAD differs from UPSTREAM.lock.json")
        dirty = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
        if dirty:
            raise ValueError(f"Tracked upstream files changed in {source}; preserve the diff and review it")


def model_args(model_type: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_type):
        raise ValueError("Invalid megatron_type")
    source = checkout("miles") / "miles/utils/external_utils/model_args_utils.py"
    spec = importlib.util.spec_from_file_location("swe_rl_model_args", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return shlex.split(module.load_model_args(model_type))


def build_train(config: dict, algo: str, mode: str, run_dir: Path, *, smoke=False, evaluate=False) -> list[str]:
    if algo not in ALGORITHMS:
        raise ValueError(f"Supported algorithms: {', '.join(ALGORITHMS)}; SAO is a future extension")
    if mode not in ("sync", "async"):
        raise ValueError("mode must be sync or async")
    p = paths(config)
    t, r, a, m = (config[k] for k in ("training", "runtime", "agent", "model"))
    total = int(t["total_trajectories_per_update"])
    group = 1 if algo == "ppo_single" else int(t["group_size"])
    if total <= 0 or group <= 0 or total % group:
        raise ValueError("total_trajectories_per_update must be positive and divisible by group_size")
    if algo != "ppo_single" and group < 2:
        raise ValueError("Group-relative baselines require group_size >= 2")
    actor_gpus = int(r["num_gpus"] if mode == "sync" else r["async_actor_gpus"])
    rollout_gpus = int(r["num_gpus"] if mode == "sync" else r["num_gpus"] - actor_gpus)
    parallel = int(r["tensor_parallel"]) * int(r["pipeline_parallel"]) * int(r["context_parallel"])
    if actor_gpus <= 0 or rollout_gpus <= 0 or actor_gpus % parallel:
        raise ValueError("Invalid GPU allocation or TP*PP*CP does not divide actor GPUs")
    if total % (actor_gpus // parallel):
        raise ValueError("Trajectory batch must be divisible by the actor data-parallel size")
    if rollout_gpus % int(r["rollout_gpus_per_engine"]):
        raise ValueError("rollout_gpus must be divisible by rollout_gpus_per_engine")
    if int(a["max_concurrent"]) < total:
        raise ValueError("agent.max_concurrent must cover total_trajectories_per_update")
    if a["client_timeout_seconds"] <= a["timeout_seconds"]:
        raise ValueError("client_timeout_seconds must exceed server timeout_seconds")
    if not 0 < t["temperature"] or not 0 < t["top_p"] <= 1:
        raise ValueError("Invalid training sampling settings")
    if "REPLACE_" in r["session_external_host"] or not r["session_external_host"]:
        raise ValueError("Set runtime.session_external_host to the trainer host reachable from task containers")
    if algo in ("grpo_dis", "score_centering") and (t["top_p"] != 1 or t["top_k"] != -1):
        raise ValueError(f"This {algo} recipe uses unfiltered sampling: top_p=1, top_k=-1")

    args = model_args(m["megatron_type"])

    def add(flag, *values):
        args.append("--" + flag)
        args.extend(str(v) for v in values)

    for flag, value in {
        "train-backend": "megatron", "hf-checkpoint": p["hf_checkpoint"],
        "ref-load": p["torch_dist_checkpoint"], "load": run_dir / "checkpoints",
        "save": run_dir / "checkpoints", "save-interval": 1 if smoke else t["save_interval"],
        "prompt-data": p["train_jsonl"], "input-key": "prompt", "metadata-key": "metadata",
        "num-rollout": t["critic_warmup_rollouts"] + 2 if smoke and algo == "ppo_single" else (2 if smoke else t["num_rollouts"]),
        "rollout-batch-size": total // group, "n-samples-per-prompt": group,
        "global-batch-size": total, "rollout-temperature": t["temperature"],
        "rollout-top-p": t["top_p"], "rollout-top-k": t["top_k"],
        "rollout-max-response-len": t["max_response_len"], "max-seq-len": t["max_seq_len"],
        "sglang-context-length": t["max_seq_len"] + t["max_response_len"],
        "seed": t["seed"], "rollout-seed": t["seed"],
        "tensor-model-parallel-size": r["tensor_parallel"],
        "pipeline-model-parallel-size": r["pipeline_parallel"],
        "context-parallel-size": r["context_parallel"],
        "expert-model-parallel-size": r["expert_parallel"], "expert-tensor-parallel-size": 1,
        "recompute-granularity": "full", "recompute-method": "uniform", "recompute-num-layers": 1,
        "max-tokens-per-gpu": r["max_tokens_per_gpu"],
        "optimizer": "adam", "lr": t["lr"], "lr-decay-style": "constant",
        "weight-decay": t["weight_decay"], "adam-beta1": 0.9, "adam-beta2": 0.98,
        "advantage-estimator": "ppo" if algo == "ppo_single" else ("gspo" if algo == "gspo" else "grpo"),
        "eps-clip": t["gspo_clip_low"] if algo == "gspo" else t["clip_low"],
        "eps-clip-high": t["gspo_clip_high"] if algo == "gspo" else t["clip_high"],
        "kl-coef": 0, "kl-loss-coef": t["kl_loss_coef"], "kl-loss-type": "low_var_kl",
        "entropy-coef": t["entropy_coef"],
        "actor-num-nodes": 1, "actor-num-gpus-per-node": actor_gpus,
        "rollout-num-gpus": rollout_gpus, "rollout-num-gpus-per-engine": r["rollout_gpus_per_engine"],
        "sglang-mem-fraction-static": r["sglang_mem_fraction"],
        "sglang-router-port": r["router_port"],
        "custom-generate-function-path": "miles.rollout.generate_hub.agentic_tool_call.generate",
        "custom-agent-function-path": "swe_agent_function.run",
        "custom-rm-path": "generate.reward_func", "rollout-function-path": "generate.RolloutFn",
        "dynamic-sampling-filter-path": "miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted",
        "tito-model": m["tito_model"], "session-server-port": r["session_port"],
        "session-server-workers": r["session_workers"], "session-server-ip": "0.0.0.0",
        "session-server-external-host": r["session_external_host"],
        "attention-dropout": 0, "hidden-dropout": 0, "attention-backend": "flash",
        "dump-details": run_dir / "traces",
    }.items():
        add(flag, value)
    for flag in ("bf16", "balance-data", "use-dynamic-batch-size", "use-session-server",
                 "accumulate-allreduce-grads-in-fp32", "attention-softmax-in-fp32", "use-kl-loss"):
        add(flag)
    if int(r["tensor_parallel"]) > 1:
        add("sequence-parallel")
    if m["tool_call_parser"]:
        add("sglang-tool-call-parser", m["tool_call_parser"])
    if m["reasoning_parser"]:
        add("sglang-reasoning-parser", m["reasoning_parser"])
    if r["optimizer_cpu_offload"]:
        for flag in ("optimizer-cpu-offload", "overlap-cpu-optimizer-d2h-h2d", "use-precision-aware-optimizer"):
            add(flag)
    if mode == "sync":
        add("colocate")
    else:
        add("fully-async")
        add("pause-generation-mode", "in_place")
        add("async-max-concurrent-samples", a["max_concurrent"])
        add("async-data-buffer-capacity-factor", t["buffer_capacity_factor"])
        add("max-weight-staleness", t["max_weight_staleness"])
        add("update-weight-transfer-mode", "broadcast")

    # Basic arms use the probabilities recorded during generation. Correction
    # arms use the upstream three-policy loss instead, avoiding double weighting.
    # Neither choice is the SAO DIS objective.
    if algo not in ("grpo_tis", "grpo_mis"):
        add("use-rollout-logprobs")
    if algo == "grpo_dis":
        lower, upper = dis_settings(config)
        add("loss-type", "custom_loss")
        add("custom-loss-function-path", "algorithms.grpo_dis.loss_function")
        add("dis-ratio-low", lower)
        add("dis-ratio-high", upper)
    if algo in ("grpo_tis", "grpo_mis"):
        # The upstream correction hook compares a detached current trainer score to
        # actual rollout probabilities, so it needs the trainer's forward pass.
        add("use-tis")
        add("custom-tis-function-path", "examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp")
        add("tis-level", "token")
        add("tis-mode", "truncate" if algo == "grpo_tis" else "mask")
        add("tis-lower-bound", 0 if algo == "grpo_tis" else t["mis_low"])
        add("tis-upper-bound", t["tis_clip"] if algo == "grpo_tis" else t["mis_high"])
    if algo == "score_centering":
        add("loss-type", "score_centering")
        add("rollout-top-logprobs-num", t["score_centering_top_k"])
        add("score-centering-is", t["score_centering_is"])
        add("score-centering-tis-clip", t["tis_clip"])
        add("score-centering-mis-low", t["mis_low"])
        add("score-centering-mis-high", t["mis_high"])
        add("disable-grpo-std-normalization")
        add("calculate-per-token-loss")
    if algo == "ppo_single":
        add("critic-lr", t["critic_lr"])
        add("num-critic-only-steps", t["critic_warmup_rollouts"])
        add("gamma", t["gamma"])
        add("lambd", t["gae_lambda"])
        add("normalize-advantages")
    if evaluate:
        add("eval-prompt-data", "swebench_verified", p["eval_jsonl"])
        add("eval-interval", t["eval_interval"])
        add("n-samples-per-eval-prompt", 1)
        add("eval-temperature", t["eval_temperature"])
        add("eval-top-p", 1)
        add("eval-top-k", -1)
        add("eval-max-response-len", t["max_response_len"])
    entry = checkout("miles") / ("train_async.py" if mode == "async" else "train.py")
    return [sys.executable, str(entry), *args]


def runtime_env(config: dict) -> dict[str, str]:
    p = paths(config)
    cache = p["cache_root"]
    env = {
        "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONPATH": os.pathsep.join([str(PROJECT), str(checkout("miles")), config["model"]["megatron_path"],
                                     str(checkout("miles") / "examples/swe-agent-harbor-docker")]),
        "HF_HOME": str(cache / "huggingface"), "XDG_CACHE_HOME": str(cache),
        "TORCH_HOME": str(cache / "torch"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "CUDA_CACHE_PATH": str(cache / "cuda"), "UV_CACHE_DIR": str(cache / "uv"),
        "WANDB_DIR": str(cache / "wandb"), "AGENT_SERVER_URL": config["agent"]["server_url"],
        "AGENT_MODEL_NAME": "model", "AGENT_TRIAL_TIMEOUT": str(config["agent"]["client_timeout_seconds"]),
        "SGLANG_RETURN_ORIGINAL_LOGPROB": "0",
    }
    # Credentials stay in the process environment, never in manifests or argv.
    for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NCCL_DEBUG", "HARBOR_ADMIN_SECRET"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def shell_command(argv: list[str]) -> str:
    return shlex.join(argv)
