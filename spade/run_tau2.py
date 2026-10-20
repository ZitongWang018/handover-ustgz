#!/usr/bin/env python3
"""Preview by default. Run only tau2 evaluation or its allocated eight-GPU server."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
sys.dont_write_bytecode = True
import traceback
from urllib.parse import urlsplit
from urllib.request import urlopen

from runtime import (BASE, MODEL_NAME, bounded, cache_environment, check_model,
                     endpoint_url, inventory, sha256, task_manifest, verify_pins)


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class RedactedWriter:
    def __init__(self, stream, redact):
        self.stream, self.redact = stream, redact
    def write(self, value):
        self.stream.write(self.redact(value))
        return len(value)
    def flush(self):
        self.stream.flush()


def server_command(model, endpoint, name):
    url = urlsplit(endpoint_url(endpoint))
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("the owned model server must bind a loopback HTTP address")
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", str(model),
            "--host", "127.0.0.1", "--port", str(url.port or 30000), "--tp", "8", "--dp", "1",
            "--mem-fraction-static", "0.7", "--served-model-name", name,
            "--tool-call-parser", "qwen", "--trust-remote-code"]


def serve(command, output):
    import torch
    if torch.cuda.device_count() != 8:
        raise ValueError("serve requires exactly eight allocated visible GPUs")
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and
           k not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_ACCESS_TOKEN"}}
    with (output / "server.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name != "nt")
        write_new(output / "server_process.json", {"pid": proc.pid, "command": command})
        try:
            if proc.wait() != 0:
                raise RuntimeError("SGLang server exited with an error; see server.log")
        finally:
            if proc.poll() is None:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serve", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--served-model-name", default=MODEL_NAME)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve(strict=True)
    if root.name != "ztwang" or not root.is_dir():
        raise ValueError("--root must be the confirmed existing ztwang directory")
    bounded(BASE, root, exists=True)
    model = bounded(args.model, root, exists=True)
    output = bounded(args.output, root)
    if not model.is_dir():
        raise ValueError("--model must be a local HF directory")
    for source in (model, BASE):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("output must be independent of model and source directories")
    if output.exists():
        raise FileExistsError("output already exists; choose a fresh run directory")
    if not args.served_model_name or args.served_model_name.startswith(("openai/", "deepseek/")):
        raise ValueError("provide the exact unprefixed Qwen server name")
    if args.smoke and args.action != "evaluate":
        raise ValueError("--smoke is only valid for evaluation")
    endpoint = endpoint_url(args.endpoint)
    pins = verify_pins()
    identity = check_model(model, root)
    command = server_command(model, endpoint, args.served_model_name) if args.action == "serve" else None
    config = BASE / "tau2_config.yaml"
    plan = {"action": args.action, "model": str(model), "output": str(output), "pins": pins,
            "endpoint": endpoint, "served_model": args.served_model_name,
            "smoke": args.smoke, "model_identity": identity, "config_sha256": sha256(config),
            "command": command, "execute": args.execute}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.action == "evaluate" and (sys.version_info < (3, 12) or not os.environ.get("DEEPSEEK_API_KEY")):
        raise ValueError("evaluation requires Python >=3.12 and runtime DEEPSEEK_API_KEY")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(cache_environment(output))
    from tau2_provider import redact
    try:
        plan["model_files"] = inventory(model, root)
        plan["tasks"] = task_manifest(root)
        plan["software"] = {}
        for package in ("sglang", "torch", "tau2", "litellm", "openai", "PyYAML"):
            try:
                plan["software"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                plan["software"][package] = "not-installed"
        write_new(output / "manifest.json", redact(plan))
        if args.action == "serve":
            serve(command, output)
        else:
            with urlopen(endpoint + "/v1/models", timeout=15) as response:
                models = json.load(response)
            if args.served_model_name not in {item.get("id") for item in models.get("data", [])}:
                raise ValueError("the endpoint does not advertise the expected Qwen model name")
            write_new(output / "endpoint_models.json", redact(models))
            sys.path[:0] = [str(BASE / "upstream"), str(BASE / "tau2-bench/src")]
            os.chdir(output)
            # Do not search ancestor directories for .env files or load other credentials.
            import dotenv
            dotenv.load_dotenv = lambda *a, **kw: False
            from tau2_eval import evaluate
            with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
                with redirect_stdout(RedactedWriter(stdout, redact)), redirect_stderr(RedactedWriter(stderr, redact)):
                    result = evaluate(endpoint, args.served_model_name, output, config, smoke=args.smoke)
            if result.get("complete") is not True:
                raise RuntimeError("tau2 result is incomplete")
        write_new(output / "status.json", {"complete": True, "action": args.action})
        return 0
    except BaseException:
        write_new(output / "failure.json", {"complete": False, "error": redact(traceback.format_exc())})
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        from tau2_provider import redact
        print(redact(str(exc)), file=sys.stderr)
        raise SystemExit(1)
