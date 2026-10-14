#!/usr/bin/env python3
"""Generate (but never execute) matched ALFWorld evaluation commands."""
from __future__ import annotations

import argparse
import json
import shlex
import uuid
from pathlib import Path

TASKS = 134
SHARDS = 8
KNOWN_ENV = (
    "EZ_SOLVER_STYLE", "EZ_SOLVER_FEWSHOT", "EZ_SOLVER_FEWSHOT_SCHEMA",
    "EZ_SOLVER_ADMISSIBLE", "EZ_SOLVER_STRICT_RESOLVE", "EZ_PARSER_STRICT",
    "EZ_SOLVER_BARE_FALLBACK", "EZ_MAX_STEPS", "EZ_MAX_NEW", "EZ_MAX_NEW_TOKENS",
    "EZ_MAX_CONTEXT_TOKENS", "EZ_TEMPERATURE", "EZ_TOP_P", "EZ_TOP_K",
    "EZ_REPETITION_PENALTY", "EZ_ACTION_MENU", "EZ_SAMPLES", "EZ_SEED",
    "EZ_TAG", "EZ_BACKEND", "EZ_BATCH", "EZ_BASE_ONLY", "EZ_CKPT",
    "EZ_PROTOCOL", "EZ_SHARDS",
)
ARMS = {
    "paper_no_menu": {"protocol": "paper-approx", "menu": "off", "steps": 30,
                      "samples": 4, "temperature": 1.0, "top_p": 0.95,
                      "top_k": -1, "repetition_penalty": 1.05, "max_new": 96,
                      "context": 40960},
    "json_menu": {"protocol": "paper-approx", "menu": "on", "steps": 30,
                   "samples": 4, "temperature": 1.0, "top_p": 0.95,
                   "top_k": -1, "repetition_penalty": 1.05, "max_new": 96,
                   "context": 40960},
    "json_menu_50": {"protocol": "paper-approx", "menu": "on", "steps": 50,
                      "samples": 4, "temperature": 1.0, "top_p": 0.95,
                      "top_k": -1, "repetition_penalty": 1.05, "max_new": 96,
                      "context": 40960},
    "public_rich": {"protocol": "public-rich", "menu": "on", "steps": 50,
                     "samples": 1, "temperature": 0.0, "top_p": 0.95,
                     "top_k": -1, "repetition_penalty": 1.0, "max_new": 32,
                     "context": 40960},
}
PAIR_DESIGNS = [
    {"arms": ["paper_no_menu", "json_menu"], "isolated_factor": "action_menu"},
    {"arms": ["json_menu", "json_menu_50"], "isolated_factor": "max_steps"},
    {"arms": ["paper_no_menu", "public_rich"], "isolated_factor": None,
     "description": "multi-factor scaffold control; not a single-factor comparison"},
]


def _command(script: str, *, base: str, data: str, ckpt: str | None,
             backend: str, seed: int, arm_name: str, arm: dict,
             tag: str) -> tuple[list[str], dict]:
    values = {
        "ZTWANG_MODEL_PATH": base,
        "ALFWORLD_DATA": data,
        "EZ_BASE_ONLY": "0" if ckpt else "1",
        "EZ_CKPT": ckpt or "",
        "EZ_BACKEND": backend,
        "EZ_PROTOCOL": arm["protocol"],
        "EZ_ACTION_MENU": arm["menu"],
        "EZ_SAMPLES": str(arm["samples"]),
        "EZ_MAX_STEPS": str(arm["steps"]),
        "EZ_MAX_NEW_TOKENS": str(arm["max_new"]),
        "EZ_MAX_CONTEXT_TOKENS": str(arm["context"]),
        "EZ_TEMPERATURE": str(arm["temperature"]),
        "EZ_TOP_P": str(arm["top_p"]),
        "EZ_TOP_K": str(arm["top_k"]),
        "EZ_REPETITION_PENALTY": str(arm["repetition_penalty"]),
        "EZ_SEED": str(seed),
        "EZ_TAG": tag,
        "EZ_SHARDS": str(SHARDS),
        "EZ_BATCH": "2",
    }
    argv = ["env"]
    for name in KNOWN_ENV:
        argv.extend(["-u", name])
    argv.extend(f"{key}={value}" for key, value in values.items())
    argv.extend(["sbatch", "--export=ALL", script])
    return argv, values


def generate_plan(*, base: str, data: str, out_dir: str, ckpt: str | None = None,
                  arms: list[str] | None = None, backend: str = "vllm",
                  seed: int = 42, script: str = "scripts/ez_eval_full_8gpu.sbatch") -> dict:
    selected = list(ARMS) if arms is None else list(arms)
    if len(set(selected)) != len(selected) or any(name not in ARMS for name in selected):
        raise ValueError("arms must be unique known arm names")
    output = Path(out_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    plan_id = uuid.uuid4().hex[:12]
    jobs = []
    for arm_name in selected:
        arm = dict(ARMS[arm_name])
        models = [("base", None)]
        if ckpt:
            models.append(("solver", ckpt))
        for model_name, model_ckpt in models:
            tag = f"{plan_id}-{arm_name}-{model_name}-s{seed}"
            argv, values = _command(script, base=base, data=data, ckpt=model_ckpt,
                                    backend=backend, seed=seed, arm_name=arm_name,
                                    arm=arm, tag=tag)
            jobs.append({"arm": arm_name, "model": model_name, "tag": tag,
                         "ckpt": model_ckpt, "settings": arm,
                         "expected_protocol_effective": {
                             "paper_no_menu": "paper-approx",
                             "json_menu": "custom",
                             "json_menu_50": "custom",
                             "public_rich": "public-rich",
                         }[arm_name],
                         "environment": values, "argv": argv,
                         "shell": shlex.join(argv),
                         "expected_task_count": TASKS,
                         "expected_samples_per_task": arm["samples"],
                         "expected_episode_count": TASKS * arm["samples"],
                         "expected_shards": SHARDS})
    return {
        "schema_version": 1,
        "plan_id": plan_id,
        "execution_policy": "manual_only; generator does not invoke env, sbatch, SSH, or a model",
        "base_model_path": base,
        "checkpoint_path": ckpt,
        "alfworld_data_path": data,
        "backend": backend,
        "seed": seed,
        "expected_task_count": TASKS,
        "expected_shards": SHARDS,
        "arms": selected,
        "pair_designs": [pair for pair in PAIR_DESIGNS
                         if all(name in selected for name in pair["arms"])],
        "jobs": jobs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="explicit base model path")
    parser.add_argument("--alfworld-data", required=True, help="explicit ALFWorld data path")
    parser.add_argument("--out-dir", required=True, help="new unique plan output directory")
    parser.add_argument("--ckpt", help="optional solver checkpoint; emits paired base and adapter jobs")
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=None)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbatch-script", default="scripts/ez_eval_full_8gpu.sbatch")
    args = parser.parse_args()
    try:
        plan = generate_plan(base=args.base, data=args.alfworld_data,
                            out_dir=args.out_dir, ckpt=args.ckpt, arms=args.arms,
                            backend=args.backend, seed=args.seed, script=args.sbatch_script)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    output = Path(args.out_dir)
    (output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8", newline="\n")
    script = "#!/usr/bin/env bash\nset -euo pipefail\n\n# Review and submit each command manually. This file is never run by the generator.\n"
    script += "\n".join(job["shell"] for job in plan["jobs"]) + "\n"
    (output / "run.sh").write_text(script, encoding="utf-8", newline="\n")
    print("wrote manual-only plan files")


if __name__ == "__main__":
    main()
