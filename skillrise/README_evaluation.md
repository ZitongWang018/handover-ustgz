# SkillRise ALFWorld evaluation

The new runner follows the active validation collector in the vendored source snapshot at commit `2b38da3ad7b64414305d39609484803b07d3b9b0`. It is a code-aligned evaluation, not a paper-exact reproduction: the paper's exact prompt and held-out task list are not published in the recovered repository. `react` uses the upstream play prompt. `zero_shot` is a user-approved custom approximation that removes the two reasoning instructions; it is not an author-published baseline.

The suite fixes 128 worker-seeded slots on `valid_seen`, preserves duplicate game instances as distinct slots, and runs one sample per slot. The active collector allows ten turns per attempt and three same-task attempts without reflection. Attempt zero is primary Pass@1; cumulative retry metrics are separately labeled. The official TextWorld episode budget remains 50. Reports include slot-weighted success and per-type success; the six-type macro is shown separately and is unavailable if any task type is absent.

## Run on a target server

Use an existing checkout, Python environment, ALFWorld dataset and local Qwen3-4B model directory. No install or download step is part of the launcher. First activate the target environment if needed, then set the paths:

```bash
export SKR_PROJECT_ROOT="/path/to/repo"
export SKR_PYTHON="/path/to/environment/bin/python"  # optional; defaults to python
export ALFWORLD_DATA="/path/to/existing/alfworld-data"
export ZTWANG_MODEL_PATH="/path/to/existing/qwen3_4b"
export EZ_SELECTION_MANIFEST="$SKR_PROJECT_ROOT/skillrise/selection_valid_seen.json"
```

Prepare the immutable slot list once on a machine with the same dataset and installed ALFWorld/TextWorld versions:

```bash
cd "$SKR_PROJECT_ROOT"
PYTHONPATH="$SKR_PROJECT_ROOT/selfplay/src:$SKR_PROJECT_ROOT/selfplay/third_party/py310deps" \
  "${SKR_PYTHON:-python}" skillrise/eval_alfworld_prompting.py \
  --prepare-selection "$EZ_SELECTION_MANIFEST"
```

Run one style per invocation. The script first performs a two-slot GPU preflight, including a seeded diagnostic comparing upstream and runner `SamplingParams`; it then starts eight independent one-GPU ranks, checks every exit code, and strictly merges all 128 slots. The preflight is a separate process so its diagnostic sampling does not advance formal-run RNG state.

```bash
EZ_STYLE=react bash "$SKR_PROJECT_ROOT/skillrise/run_eval_8gpu.sh"
EZ_STYLE=zero_shot bash "$SKR_PROJECT_ROOT/skillrise/run_eval_8gpu.sh"
```

`skillrise/skillrise_eval_8gpu.sbatch` is a template for the original HKUST cluster. Its account, partition and log paths are site-specific; other clusters should use the portable launcher or override those Slurm directives. It defaults the project path and Conda environment only for that template. `SKR_REQUIRE_GLIBC_COMPAT=1` can require the repository's optional compatibility shim; otherwise a host without that shim uses its native `fast_downward` installation.

An output directory is unique per run and must not already exist. Raw turn traces, parity diagnostics, per-rank JSON, and merged metrics are kept together. `python skillrise/summarize.py <run>/merged.json` prints weighted Pass@1 and the six-type macro separately.

## Validation status

CPU mocks cover source hashes, prompt construction, token-ID decoding, exact sampling arguments, slot identity, retry behavior, and per-slot history isolation. This checkout has not produced a target-server GPU preflight or a full evaluation result. Do not infer GPU parity, throughput, or score improvement until that server-side check completes.
