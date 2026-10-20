"""Official tau2-bench evaluation routed through the DeepSeek provider patch."""
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

import tau2_provider as provider

_TASKS = (("retail", "base"), ("telecom", "base"), ("airline", "base"))
_DEFAULTS = {
    "max_concurrency": 48,
    "num_trials": 4,
    "max_steps": 200,
    "agent_temperature": 0.0,
    "agent_top_p": 0.8,
    "agent_top_k": 20,
    "agent_max_tokens": 16384,
    "user_temperature": 0.0,
    "user_max_tokens": 2048,
}


def _v1(url):
    value = str(url).rstrip("/")
    return value if value.endswith("/v1") else f"{value}/v1"


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _task_id(task):
    value = _get(task, "id")
    if value is None:
        raise RuntimeError("tau2 task is missing an id")
    return str(value)


def _load_specs(config_path, smoke, tt):
    import yaml

    expected = {"tau2_eval": {"defaults": {**_DEFAULTS, "user_llm": provider.MODEL},
                             "tasks": [{"domain": d, "split": s} for d, s in _TASKS]}}
    loaded = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if loaded != expected:
        raise RuntimeError(f"tau2 config is not the required official mirror: {config_path}")
    _, specs = tt.load_tau2_eval_config(str(config_path))
    return [replace(spec, truncate=1, num_trials=1) for spec in specs] if smoke else specs


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _write_summary(output_dir, summary):
    path = Path(output_dir) / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = provider.redact(summary)
    text = clean if isinstance(clean, str) else json.dumps(clean, indent=2, sort_keys=True, default=str)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text.rstrip() + "\n")


def _validate(native, tasks, num_trials, te):
    simulations = _get(native, "simulations")
    if simulations is None and hasattr(native, "model_dump"):
        simulations = (native.model_dump() or {}).get("simulations")
    simulations = list(simulations or [])
    expected = {(str(_task_id(task)), trial) for task in tasks for trial in range(num_trials)}
    seen = set()
    errors = getattr(te, "_ERROR_TERMINATION_REASONS", ())
    for simulation in simulations:
        trial = _get(simulation, "trial")
        if isinstance(trial, bool) or not isinstance(trial, int):
            raise RuntimeError(f"invalid simulation trial: {trial!r}")
        key = (str(_get(simulation, "task_id", "")), trial)
        if key in seen:
            raise RuntimeError(f"duplicate simulation result: {key}")
        if key not in expected:
            raise RuntimeError(f"unexpected/excess simulation result: {key}")
        seen.add(key)
        reason = _get(simulation, "termination_reason")
        reason = _get(reason, "value", reason)
        if reason in errors:
            raise RuntimeError(f"infrastructure termination for {key}: {reason}")
        if reason not in {"agent_stop", "user_stop", "max_steps", "context_window_exceeded"}:
            raise RuntimeError(f"unknown termination reason for {key}: {reason}")
        if _get(simulation, "reward_info") is None:
            raise RuntimeError(f"missing reward_info: {key}")
        reward = _get(_get(simulation, "reward_info"), "reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward) or not -1e-6 <= reward <= 1 + 1e-6:
            raise RuntimeError(f"invalid reward: {key}")
        if not _get(simulation, "messages"):
            raise RuntimeError(f"empty messages: {key}")
    missing = expected - seen
    if missing or len(simulations) != len(expected):
        raise RuntimeError(f"incomplete task_id x trial Cartesian results; missing={len(missing)}")


def evaluate(base_url, served_model, output_dir, config_path, *, smoke=False):
    """Run all official tau2 domains and write only a redacted results.json."""
    output_dir = Path(output_dir)
    if (output_dir / "results.json").exists() or (output_dir / "results.json").is_symlink():
        raise FileExistsError("refusing to overwrite existing evaluation results")
    endpoint = _v1(base_url)
    summary = {
        "scope": "smoke" if smoke else "formal",
        "complete": False,
        "protocol_deviations": [
            f"user simulator and NL-assertion judge use {provider.MODEL}",
            "infrastructure resamples disabled (max_retries=0)",
        ],
        "per_spec_results": [],
        "official_metrics": {},
        "provider_observed_counts": {},
        "raw_paths": [],
    }
    observed = {}
    try:
        from spade.core.eval import tau2_evaluator as te
        from spade.core.eval import tau2_tasks as tt

        if not getattr(te, "_TAU2_AVAILABLE", False):
            raise RuntimeError(f"tau2-bench is unavailable: {getattr(te, '_TAU2_IMPORT_ERROR', None)}")
        base = Path(__file__).resolve().parent
        origins = [(te, base / "upstream"), (tt, base / "upstream"),
                   (sys.modules.get("tau2"), base / "tau2-bench/src")]
        for module, expected in origins:
            filename = getattr(module, "__file__", None)
            if not filename or not Path(filename).resolve().is_relative_to(expected.resolve()):
                raise RuntimeError("evaluation imported a different SPADE/tau2 source than the pinned checkout")
        specs = _load_specs(config_path, smoke, tt)
        evaluator = te.Tau2Evaluator(sglang_base_url=endpoint,
                                    openrouter_api_key="unused-deepseek-route")
        spec_results = []
        started = time.monotonic()
        with provider.provider_context(endpoint, served_model) as observed:
            for spec in specs:
                domain_dir = output_dir / spec.domain
                raw_path = domain_dir / "raw.json"
                summary["raw_paths"].append(str(raw_path))
                if raw_path.exists():
                    raise RuntimeError(f"refusing to overwrite existing raw results: {raw_path}")
                domain_dir.mkdir(parents=True, exist_ok=True)
                tasks = list(te.get_tasks(task_set_name=spec.domain,
                                          task_split_name=spec.split,
                                          num_tasks=1 if smoke else None))
                task_ids = [_task_id(task) for task in tasks]
                if not task_ids or len(task_ids) != len(set(task_ids)):
                    raise RuntimeError(f"tasks for {spec.domain}/{spec.split} are empty or duplicate")
                t0 = time.monotonic()
                native = te.run_tasks(
                    domain=spec.domain, tasks=tasks, agent="llm_agent", user="user_simulator",
                    llm_agent=f"openai/{served_model}",
                    llm_args_agent={"api_base": endpoint, "api_key": "dummy",
                                    "temperature": spec.agent_temperature,
                                    "top_p": spec.agent_top_p,
                                    "max_tokens": spec.agent_max_tokens,
                                    "num_retries": 0, "timeout": 600,
                                    "extra_body": {"top_k": spec.agent_top_k,
                                                   "chat_template_kwargs": {"enable_thinking": False}}},
                    llm_user=provider.MODEL, llm_args_user=dict(provider.user_args()),
                    num_trials=spec.num_trials, max_steps=spec.max_steps,
                    max_concurrency=spec.max_concurrency, console_display=False,
                    evaluation_type=te.EvaluationType.ALL_WITH_NL_ASSERTIONS,
                    seed=300, max_retries=0, save_to=raw_path, save_dir=domain_dir)
                _validate(native, tasks, spec.num_trials, te)
                result = evaluator._aggregate_spec_result(spec, len(tasks), native, time.monotonic() - t0)
                if result.errors or result.num_simulations != len(tasks) * spec.num_trials:
                    raise RuntimeError("official aggregate reports errors or a changed denominator")
                spec_results.append(result)
                row = _plain(result)
                row["raw_path"] = str(raw_path)
                summary["per_spec_results"].append(row)
        total_simulations = sum(result.num_simulations for result in spec_results)
        total_passed = sum(result.num_passed for result in spec_results)
        aggregate = te.Tau2EvalResult(
            total_tasks=sum(result.num_tasks for result in spec_results),
            total_simulations=total_simulations, total_passed=total_passed,
            total_errors=sum(result.errors for result in spec_results),
            overall_pass_at_1=(total_passed / total_simulations if total_simulations else 0.0),
            per_spec_results=spec_results, elapsed_sec=time.monotonic() - started)
        summary["aggregate"] = _plain(aggregate)
        summary["official_metrics"] = dict(aggregate.to_metrics_dict())
        summary["provider_observed_counts"] = dict(observed or {})
        summary["complete"] = True
        _write_summary(output_dir, summary)
        return summary
    except Exception as exc:
        summary["complete"] = False
        summary["provider_observed_counts"] = dict(observed or {})
        summary["error"] = provider.redact(f"{type(exc).__name__}: {exc}")
        _write_summary(output_dir, summary)
        raise
