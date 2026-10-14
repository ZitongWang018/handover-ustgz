# Explorer-Zero repair and verification status

This document replaces the earlier implementation plans for the current code path. It records what the repository now implements, what remains unverified, and the protocol boundaries needed for a fair reproduction. It does not claim paper-exact behavior where the paper leaves details unspecified.

## Implemented in this branch

- **Global GRPO sharding:** `train.tasks_per_step` counts global task groups and defaults to 1; `group_size` defaults to 16. For eight ranks, each rank samples 2 trajectories from the same initial game. A group size that is smaller than or not divisible by `world_size` fails explicitly. Rewards are gathered before global group normalization and advantages are then sharded back to ranks. The task pool is gathered identically on every rank, with rank-qualified UIDs. Pool selection uses a seeded permutation and moving cursor so successive steps cover the pool without duplicate UIDs in one step.
- **Adapter synchronization:** EZ uses the raw PEFT model and fixed-order, bucketed LoRA gradient all-reduce after local accumulation. Missing local gradients contribute zeros; inactive-adapter gradients are rejected. Both adapters are broadcast at initialization. Parameter equality checks run at each phase's first update and every configured interval. The smoke log emits explicit `[adapter-sync] ... status=PASS` lines after real parameter comparisons.
- **Microbatch and loss:** local forward/backward microbatches default to 1, while one optimizer update covers all local trajectories for a task group. Advantages are computed outside microbatching. Sequence reductions use the global valid-sequence denominator; EZ rejects token reduction. Optional KL uses the disabled-adapter base reference without cloning the base model. Gradients are synchronized once per update, then clipped and checked for finite values.
- **Selected logits and generation:** the training log-probability path uses Qwen3 `logits_to_keep` for masked positions when supported and falls back to dense logits otherwise. EZ generation defaults to prompt chunks of 2, sorts by prompt length and restores original order, uses token-native accumulated buffers across turns, and fails loudly when the hard context budget is exceeded. Per-row stopping is retained. The mainline batching default remains unchanged; its legacy transcript builder now rejects templates that are not prefix-stable.
- **Action and control protocol:** generated assistant token IDs, including malformed and control responses, remain in the training history. `declare` records intent only; `sub_finish` records a state snapshot. Aliases include `declare_task`, `subtask`, and `finish_task`. The target environment observed for this implementation is ALFWorld 0.3.3; placement aliases map to the documented `put ... in/on ...` form. No fuzzy action correction is performed on the EZ raw path. The exact command acceptance should still be checked against the installed cluster package and logged admissible-command diagnostics.
- **Benchmark completion:** the optional EZ wrapper preserves TextWorld `won` facts while suppressing only benchmark-goal termination; an outer TextWorld `Limit` still enforces the episode budget. Official evaluation keeps benchmark completion behavior unchanged. Solver reset facts are checked against the proposed task's initial state.
- **State deltas:** snapshots preserve raw fact structures and normalized fact multiplicity. New task pool records use schema 2; legacy records cannot recover lost multiplicity or entity bindings and are not accepted for formal resume. Typed entity renaming is matched consistently across a delta, with a search budget that fails explicitly. Player/agent navigation is projected out, including `atlocation`. When a task description names known ALFWorld entities, spatial relationships require named endpoints; incidental source locations are not hidden requirements. Container open/close changes count only when the specific container is explicitly named as the open/close target. Unknown predicates involving named task entities remain conservative. This is a bounded task-relevance heuristic, not a recovered author specification.
- **Resume/checkpoint:** complete checkpoints contain both adapters, optimizer state, task pool, train position and per-rank RNG. Saves occur every five updates and at phase boundaries. Promotion uses a temporary directory and keeps the previous checkpoint until the new one is promoted. Resume validates world size and key training settings before loading. Incomplete legacy checkpoints are refused.
- **Evaluation:** `src/eval_ez.py` explicitly loads a base model and optionally `checkpoint/solver`; `--base-only` provides the matched base control. It evaluates official `eval_out_of_distribution` tasks with benchmark `won`, defaults to the full split and four samples per task, and reports sample-level success mean/std and task-type results. The sample standard deviation is not a training-seed standard deviation. Prompt protocol and decoding settings are shared between adapter and base control; admissible commands and expert plans are not exposed.

## Required verification before interpreting results

CPU regression coverage lives in `selfplay/tests/test_ez_regression.py` and the compatibility suite in `selfplay/scripts/ez_unit_test.py`. Run at minimum:

```bash
python -m pytest selfplay/tests/test_ez_regression.py -q
python selfplay/scripts/ez_unit_test.py
python selfplay/scripts/ez_check_smoke_selftest.py
```

The regression suite includes global G=16 sharding, rank rotation, typed TextWorld-style facts, object-binding counterexamples, quantity/noun filtering, generation-prefix capture, selected-logit and KL microbatch equivalence, two-rank CPU Gloo with real PEFT/Qwen3 adapters, inactive-adapter isolation, missing-gradient collectives, and done/limit behavior. GPU/NCCL behavior and full ALFWorld rollouts have not been run in this local code session.

An eight-GPU smoke must use an explicit base-model path and verify all ranks emit proposer and solver adapter-sync PASS records. If rewards are all equal or parameters do not move because group advantages are zero, the checker labels this **structure/communication passed, learning signal unverified**. It is not a learning-validation pass. The formal acceptance sequence is:

1. Confirm `ZTWANG_MODEL_PATH` points to the intended Qwen3-4B weights by checking the actual model card and path. The paper's model name alone does not establish that it is Qwen3-4B-Base, and this config does not replace the user's explicit path.
2. Run the eight-rank smoke with global `G=16`, `TPS=1`, and verify local `G=2` in logs.
3. Require facts-channel checks, every rank log, global group-size checks, and real proposer/solver adapter parameter synchronization checks.
4. Separately establish nonzero reward variation and nonzero solver/proposer learning signal before calling the run a learning validation.
5. Run adapter and `--base-only` evaluation with identical seed, decode options, full `valid_unseen`, and four samples per task. Report sample-level dispersion separately from independent training seeds.

## Cluster commands

On hpc2, after setting the explicit base path and checking the configured conda/data paths:

```bash
export ZTWANG_MODEL_PATH=/absolute/path/to/qwen3-4b-base
cd ~/ztwang/selfplay
sbatch -p i64m1tga800u8ka -t 02:00:00 --gres=gpu:8 scripts/ez_smoke_8gpu.sbatch
sbatch scripts/ez_train_a800x8.sbatch
```

The smoke uses `G=16`, global `TPS=1`, and `G_P=2` by default. The formal script targets the A800 eight-GPU partition, writes checkpoints and logs under `~/jhspoolers/ez_runs`, and keeps only the active checkpoint. On hpc3, use the cluster's ACD partition and storage rules from `docs/CLUSTER_GUIDE.md`; the checked-in formal sbatch template is hpc2-specific. Set `EZ_RUN_OUTPUT_DIR` explicitly when adapting it.

Evaluation example:

```bash
python src/eval_ez.py --base "$ZTWANG_MODEL_PATH" --ckpt /path/to/run/ckpt \
  --config configs/base_config.yaml --n 0 --samples 4 --seed 42 \
  --max-context-tokens 40960 --top-k 0 --out runs/eval_ez/solver.json
python src/eval_ez.py --base "$ZTWANG_MODEL_PATH" --base-only \
  --config configs/base_config.yaml --n 0 --samples 4 --seed 42 \
  --max-context-tokens 40960 --top-k 0 --out runs/eval_ez/base.json
```

## Remaining limitations and specification uncertainty

- No GPU/NCCL smoke, memory peak, throughput, or performance improvement is claimed here.
- Full cross-turn KV/prefix caching, vLLM, ScienceWorld, and AppWorld are not implemented.
- Observational subtasks represented only by action/return pairs are not yet verified; do not claim their paper handling is reproduced.
- The paper does not specify all ALFWorld predicate projection, object-binding, action-translation, sampling, and KL-reference details. Current implementations are explicit, tested approximations. In particular, task relevance is derived from known entity names in task text; generic or poorly grounded descriptions may keep extra facts or miss intent. Treat such tasks cautiously, inspect saved raw facts, and do not claim recovered author-exact semantics.
- The normal mainline path still uses transcript rebuilding and is only guarded by fail-fast prefix checks. EZ uses token-native histories; broader mainline model/template compatibility needs separate evaluation.
- Checkpoint resume is intended for the same world size and key configuration. It deliberately rejects mismatches instead of silently changing the trajectory.
