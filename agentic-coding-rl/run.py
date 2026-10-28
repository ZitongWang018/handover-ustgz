#!/usr/bin/env python3
"""SWE RL preparation and single-node training. Python 3.12+, standard-library CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import urllib.request
import uuid

from launcher import (ALGORITHMS, LOCK, PROJECT, build_train, check_workspace, checkout,
                      git_head, model_args, paths, read_config, runtime_env, shell_command,
                      verify_sources, workspace_path)
from algorithms.dis_config import dis_settings


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute(argv, *, cwd=None, env=None):
    print("+ " + shell_command([str(s) for s in argv]), flush=True)
    subprocess.run(argv, cwd=cwd, env=env, check=True)


def bootstrap():
    for name, pin in LOCK.items():
        target = checkout(name)
        if target.exists():
            if not (target / ".git").exists() or git_head(target) != pin["commit"]:
                raise ValueError(f"Existing {target} differs from the pin; no files were replaced")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Do not change machine-wide Git TLS or authentication configuration.
        git = ["git"] + (["-c", "http.sslBackend=openssl"] if os.name == "nt" else [])
        execute([*git, "clone", "--depth", "1", "--branch", pin["ref"], pin["url"], str(target)])
        if git_head(target) != pin["commit"]:
            execute([*git, "-C", str(target), "fetch", "--depth", "1", "origin", pin["commit"]])
            execute(["git", "-C", str(target), "checkout", "--detach", pin["commit"]])
    verify_sources()
    print("Pinned upstream checkouts ready. No GPU packages, datasets, or models installed.")


def check_data(config, *, evaluate=False):
    p = paths(config)
    collections = {}
    for label in (["train", "eval"] if evaluate else ["train"]):
        source = p[label + "_jsonl"]
        ids = []
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            metadata = row["metadata"]
            iid = metadata["instance_id"]
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", iid):
                raise ValueError(f"Unsafe task ID: {iid}")
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"Missing prompt: {iid}")
            if metadata["agent_name"] != config["agent"]["name"]:
                raise ValueError(f"Agent changed after data preparation: {iid}; regenerate into a new dataset")
            if metadata.get("max_seq_len") != config["training"]["max_seq_len"]:
                raise ValueError(f"Context budget changed after data preparation: {iid}; regenerate into a new dataset")
            if {"patch", "test_patch", "solution"}.intersection(row) or {"patch", "test_patch", "solution"}.intersection(metadata):
                raise ValueError(f"Gold solution must not be in rollout metadata: {iid}")
            task = workspace_path(config, p["tasks_dir"] / iid)
            for required in ("task.toml", "instruction.md", "environment/Dockerfile", "tests/test.sh"):
                if not (task / required).is_file():
                    raise ValueError(f"Missing task artifact: {task / required}")
            ids.append(iid)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty dataset or duplicate IDs: {source}")
        collections[label] = set(ids)
    if evaluate and collections["train"] & collections["eval"]:
        raise ValueError("Train/eval instance IDs overlap")
    return {name: len(ids) for name, ids in collections.items()}


def doctor(config, *, gpu=False, evaluate=False):
    check_workspace(config)
    verify_sources()
    p = paths(config)
    for name in ("hf_checkpoint", "torch_dist_checkpoint"):
        if not p[name].is_dir():
            raise ValueError(f"Missing {name}: {p[name]}")
    hf = json.loads((p["hf_checkpoint"] / "config.json").read_text())
    expected = model_args(config["model"]["megatron_type"])
    mappings = {"--num-layers": "num_hidden_layers", "--hidden-size": "hidden_size",
                "--num-attention-heads": "num_attention_heads", "--rotary-base": "rope_theta"}
    for flag, field in mappings.items():
        if flag in expected and field in hf and float(expected[expected.index(flag) + 1]) != float(hf[field]):
            raise ValueError(f"HF {field} does not match megatron_type (check model variant, including RoPE)")
    counts = check_data(config, evaluate=evaluate)
    if gpu:
        import torch
        if torch.cuda.device_count() < config["runtime"]["num_gpus"]:
            raise ValueError("Insufficient visible GPUs")
        import ray
        import sglang
        print(f"torch={torch.__version__}; ray={ray.__version__}; sglang={sglang.__version__}")
        if not Path(config["model"]["megatron_path"]).is_dir():
            raise ValueError("Megatron path missing inside the training container")
        url = config["agent"]["server_url"].rstrip("/") + "/health"
        # Do not route loopback health checks through an HTTP proxy.
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=10) as result:
            print("Harbor health:", result.read().decode()[:600])
    print(f"Preflight OK: {counts}. This does not validate a GPU training step.")


def convert(config, dry_run):
    p = paths(config)
    command = [sys.executable, str(checkout("miles") / "tools/convert_hf_to_torch_dist.py"),
               *model_args(config["model"]["megatron_type"]), "--hf-checkpoint", str(p["hf_checkpoint"]),
               "--save", str(p["torch_dist_checkpoint"])]
    if dry_run:
        print(shell_command(command))
        return
    check_workspace(config)
    verify_sources()
    if not (p["hf_checkpoint"] / "config.json").is_file():
        raise ValueError("Provide the local HF checkpoint first")
    if p["torch_dist_checkpoint"].exists():
        raise ValueError("Conversion output already exists; choose a new destination or reuse it")
    env = {**os.environ, **runtime_env(config)}
    p["cache_root"].mkdir(parents=True, exist_ok=True)
    execute(command, cwd=checkout("miles"), env=env)


def prepare_data(config, kind, limit):
    """Run the pinned Harbor adapter and export prompts from the generated tasks."""
    check_workspace(config)
    verify_sources()
    p = paths(config)
    output = p["train_jsonl" if kind == "train" else "eval_jsonl"]
    if output.exists():
        raise ValueError(f"Output already exists: {output}; datasets are not overwritten")
    stage = output.parent / (output.stem + "_harbor_source")
    if stage.exists():
        raise ValueError(f"Previous adapter output exists: {stage}; inspect it and choose a new output path")
    env = {**os.environ, **runtime_env(config)}
    env["TMPDIR"] = str(p["cache_root"] / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    harbor = checkout("harbor")
    if kind == "train":
        adapter = harbor / "adapters/swegym/run_adapter.py"
        command = [sys.executable, str(adapter), "--dataset", config["data"]["swegym_subset"]]
    else:
        adapter = harbor / "adapters/swebench/src/swebench_adapter/main.py"
        command = [sys.executable, str(adapter)]
    limit = limit if limit is not None else config["data"]["train_limit" if kind == "train" else "eval_limit"]
    if limit < 0:
        raise ValueError("limit must be zero (all) or a positive integer")
    command += ["--task-dir", str(stage), "--timeout", str(config["agent"]["task_timeout_seconds"])]
    if limit:
        command += ["--limit", str(limit)]
    execute(command, cwd=adapter.parent, env=env)
    tasks = sorted(stage.glob("*/task.toml"))
    if not tasks or (limit and len(tasks) != limit):
        raise ValueError("Adapter produced zero tasks or fewer tasks than requested; inspect its output")
    rows = []
    artifacts = {}
    for marker in tasks:
        task = marker.parent
        raw = json.loads((task / "tests/config.json").read_text(encoding="utf-8"))
        iid = task.name
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", iid):
            raise ValueError(f"Unsafe task ID: {iid}")
        target = workspace_path(config, p["tasks_dir"] / iid)
        if target.exists():
            raise ValueError(f"Task collision: {target}")
        rows.append({"prompt": raw["problem_statement"], "metadata": {
            "instance_id": iid, "agent_name": config["agent"]["name"],
            "split": "train" if kind == "train" else "test",
            "max_seq_len": config["training"]["max_seq_len"],
        }})
        artifacts[iid] = {str(f.relative_to(task)): sha256(f) for f in sorted(task.rglob("*")) if f.is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    p["tasks_dir"].mkdir(parents=True, exist_ok=True)
    for marker in tasks:
        shutil.copytree(marker.parent, p["tasks_dir"] / marker.parent.name)
    # The original adapter output remains on disk for audit; only prompts go to Miles.
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output.with_suffix(".manifest.json"), {
        "created_at": now(), "kind": kind, "count": len(rows), "limit": limit,
        "dataset": ("SWE-Gym/SWE-Gym-Lite" if config["data"]["swegym_subset"] == "lite" else "SWE-Gym/SWE-Gym")
                   if kind == "train" else "princeton-nlp/SWE-bench_Verified",
        "hf_revision": "adapter default; actual task contents hashed below",
        "jsonl_sha256": sha256(output), "harbor_commit": LOCK["harbor"]["commit"],
        "task_files_sha256": artifacts,
    })
    print(f"Prepared {len(rows)} {kind} tasks: {output}")


def serve(config):
    check_workspace(config)
    verify_sources()
    p = paths(config)
    check_data(config)
    a = config["agent"]
    service = p["run_root"] / ("harbor-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    service.mkdir(parents=True)
    env = {**os.environ, **runtime_env(config),
           "HARBOR_TASKS_DIR": str(p["tasks_dir"]), "HARBOR_ENV_TYPE": "docker",
           "DOCKER_PRUNE_INTERVAL_SEC": "inf", "HARBOR_DELETE_CONTAINERS": "false",
           "AGENT_MAX_INPUT_TOKENS": str(config["training"]["max_seq_len"]),
           "AGENT_MAX_OUTPUT_TOKENS": str(config["training"]["max_response_len"])}
    command = [sys.executable, str(checkout("harbor") / "miles_agent_server.py"),
               "--host", a["bind_host"], "--port", str(a["port"]),
               "--max-concurrent", str(a["max_concurrent"]), "--agent-timeout", str(a["timeout_seconds"]),
               "--dashboard-port", "0", "--trials-dir", str(service / "trials"),
               "--dashboard-log-path", str(service / "requests.jsonl")]
    write_json(service / "manifest.json", {"created_at": now(), "harbor_commit": LOCK["harbor"]["commit"],
               "command": command, "tasks_dir": str(p["tasks_dir"]), "global_prune_disabled": True})
    execute(command, cwd=checkout("harbor"), env=env)


def check_ports(config):
    r = config["runtime"]
    ports = [r["ray_port"], r["ray_dashboard_port"], r["router_port"],
             *range(r["session_port"], r["session_port"] + r["session_workers"])]
    if len(ports) != len(set(ports)):
        raise ValueError("Configured Ray/router/session ports overlap")
    for port in ports:
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError as exc:
                raise ValueError(f"Port {port} occupied; no existing service was stopped") from exc


def train(config, options):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", options.run_id):
        raise ValueError("run-id must be 1..80 letters/digits/._- and start with a letter or digit")
    p = paths(config)
    run_dir = workspace_path(config, p["run_root"] / options.run_id)
    command = build_train(config, options.algo, options.mode, run_dir, smoke=options.smoke, evaluate=options.eval)
    if options.dry_run:
        print(shell_command(command))
        return
    if platform.system() != "Linux":
        raise ValueError("GPU training requires Linux; use --dry-run locally")
    doctor(config, gpu=True, evaluate=options.eval)
    check_ports(config)
    if run_dir.exists():
        raise ValueError("run-id already exists; use a new ID (automatic resume is intentionally disabled)")
    temp = p["cache_root"] / "ray" / options.run_id
    if len(str(temp)) > 65:
        raise ValueError("Ray temp path too long; shorten paths.cache_root (must remain in ztwang)")
    run_dir.mkdir(parents=True)
    manifest = {"started_at": now(), "state": "preparing", "algorithm": options.algo,
                "mode": options.mode, "smoke": options.smoke, "evaluation_enabled": options.eval,
                "upstreams": LOCK, "config": config, "command": command,
                "train_jsonl_sha256": sha256(p["train_jsonl"]),
                "wrapper_source_sha256": {name: sha256(PROJECT / name) for name in (
                    "launcher.py", "run.py", *[str(f.relative_to(PROJECT)) for f in sorted((PROJECT / "algorithms").glob("*.py"))])},
                "container_image_id": os.environ.get("SWE_RL_IMAGE_ID", "not recorded; direct launch"),
                "parent_commit": git_head(PROJECT),
                "evaluation_jsonl_sha256": sha256(p["eval_jsonl"]) if options.eval else None}
    if options.algo == "grpo_dis":
        lower, upper = dis_settings(config)
        manifest["algorithm_details"] = {"objective": "direct_double_sided_is_v1", "ratio_low": lower,
            "ratio_high": upper, "strict_bounds": True, "normalizer": "original_action_tokens",
            "regularizers": "reference_KL_and_entropy_on_original_action_tokens"}
    write_json(run_dir / "manifest.json", manifest)
    env = {**os.environ, **runtime_env(config)}
    # Ray's UNIX socket names have a short length limit: keep the configured
    # cache_root short on the server (e.g. <ztwang>/.cache/swe-rl).
    temp.mkdir(parents=True)
    env["RAY_TMPDIR"] = str(temp)
    env["TMPDIR"] = str(p["cache_root"] / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    r = config["runtime"]
    address = f"http://127.0.0.1:{r['ray_dashboard_port']}"
    ray_address = f"127.0.0.1:{r['ray_port']}"
    env["RAY_ADDRESS"] = ray_address
    worker_env = runtime_env(config)
    worker_env.update({"RAY_ADDRESS": ray_address, "RAY_TMPDIR": str(temp), "TMPDIR": env["TMPDIR"]})
    job_id = "swe-rl-" + options.run_id
    manifest.update({"ray_address": ray_address, "ray_dashboard": address, "submission_id": job_id})
    write_json(run_dir / "manifest.json", manifest)
    head = [sys.executable, "-m", "ray.scripts.scripts", "start", "--head",
            "--node-ip-address", r["node_ip"], "--port", str(r["ray_port"]),
            "--dashboard-host", "127.0.0.1", "--dashboard-port", str(r["ray_dashboard_port"]),
            "--num-gpus", str(r["num_gpus"]), "--temp-dir", str(temp), "--disable-usage-stats"]
    execute(head, env=env)
    # Use the actual Miles train.py entry directly. Do not invoke upstream
    # command_utils, whose startup cleanup kills processes by global name.
    from ray.job_submission import JobSubmissionClient
    client = JobSubmissionClient(address)
    try:
        client.submit_job(entrypoint=shell_command(command), submission_id=job_id,
                          runtime_env={"env_vars": worker_env})
        manifest["state"] = "submitted"
        write_json(run_dir / "manifest.json", manifest)
        follow = [sys.executable, "-m", "ray.scripts.scripts", "job", "logs", "--address", address, "--follow", job_id]
        with (run_dir / "train.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(follow, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            process.wait()
        result = client.get_job_status(job_id)
        status = result.value
        manifest["state"] = status
        if status != "SUCCEEDED":
            raise RuntimeError(f"Ray job ended with {status}; see {run_dir / 'train.log'}")
    except BaseException:
        # Only stop the exact job created by this launcher; never global ray stop.
        client.stop_job(job_id)
        manifest["state"] = "failed_or_interrupted"
        raise
    finally:
        manifest["finished_at"] = now()
        write_json(run_dir / "manifest.json", manifest)
    print(f"Job exited successfully. Artifacts: {run_dir}. Inspect rewards, masks and checkpoints before claiming reproduction.")
    print("The dedicated Ray service stays up until this training container exits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap", help="Clone the two exact upstream commits")
    for name in ("doctor", "convert", "prepare-data", "serve", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default=str(PROJECT / "configs/local.json"))
        if name in ("train", "convert"):
            sub.add_argument("--dry-run", action="store_true")
        if name in ("train", "doctor"):
            sub.add_argument("--eval", action="store_true", help="Require and use the held-out Verified JSONL")
        if name == "doctor":
            sub.add_argument("--gpu", action="store_true")
        if name == "prepare-data":
            sub.add_argument("--kind", choices=("train", "eval"), default="train")
            sub.add_argument("--limit", type=int)
        if name == "train":
            sub.add_argument("--algo", choices=ALGORITHMS, default="grpo")
            sub.add_argument("--mode", choices=("sync", "async"), default="async")
            sub.add_argument("--run-id", required=True)
            sub.add_argument("--smoke", action="store_true", help="Two updates (PPO adds critic warmup)")
    args = parser.parse_args()
    if args.command == "bootstrap":
        bootstrap()
        return
    config = read_config(args.config)
    if args.command == "doctor":
        doctor(config, gpu=args.gpu, evaluate=args.eval)
    elif args.command == "convert":
        convert(config, args.dry_run)
    elif args.command == "prepare-data":
        prepare_data(config, args.kind, args.limit)
    elif args.command == "serve":
        serve(config)
    elif args.command == "train":
        train(config, args)


if __name__ == "__main__":
    if os.name == "nt":
        # Command previews may contain non-ASCII checkpoint/workspace paths.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ERROR: {error}") from error
