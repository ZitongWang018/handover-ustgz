#!/usr/bin/env python3
"""Export a checkpoint's solver adapter as a plain merged Hugging Face model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))


import eval_runner
from transformers import AutoTokenizer

from merged_model import export_solver_model, validate_merged_model

def _requested_arm(base, checkpoint, config):
    config_path = Path(config).expanduser().resolve(strict=True)
    config_value = eval_runner._load_yaml(config_path)
    solver_config = eval_runner._solver_config(config_value)
    base_path, _, base_digest = eval_runner._base_identity(str(base))
    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True,
                                              local_files_only=True)
    from protocol import solver_prompt_identity
    prompt_identity = solver_prompt_identity(tokenizer,
        solver_config.get("prompt_format", "raw"))
    arm = eval_runner._arm(argparse.Namespace(base_only=False, checkpoint=str(checkpoint)),
        base_digest, solver_config, prompt_identity=prompt_identity,
        tokenizer_class=type(tokenizer).__qualname__)
    return base_digest, arm

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if Path(args.output).expanduser().exists():
            raise FileExistsError(f"refusing to overwrite merged model: {args.output}")
        base_digest, arm = _requested_arm(args.base, args.checkpoint, args.config)
        manifest = export_solver_model(args.base, arm["solver_path"], args.output,
                                       checkpoint_identity=arm)
        validate_merged_model(args.output, base_digest=base_digest, arm=arm)
        print(f"merged solver model written to {args.output} ({manifest['manifest_sha256']})")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
