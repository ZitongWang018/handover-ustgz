# Earlier SkillRise ALFWorld audit notes: superseded findings

The first audit found mismatched episode counts, response length, parser behavior, and uncertain task ordering. Its then-current numbers are retained in old run artifacts, but they are not evidence for the code-aligned runner described in [DECISION_zero_shot_vs_react.md](DECISION_zero_shot_vs_react.md).

Corrections established by tracing the package dispatcher and active collector:

- The executed evaluation path is `skillrise_rollout_loop.TrajectoryCollector`, selected by `agent_system/multi_turn_rollout/__init__.py`; the generic rollout loop is not the active validation path.
- The active validation uses three same-task attempts, ten turns per attempt, no reflection, and skips later attempts after a win. Attempt zero is the primary Pass@1 outcome; cumulative retries are reported separately.
- The 128 validation slots come from 128 independently seeded workers choosing the first entry from a shuffled full game pool. Duplicate game IDs are valid distinct slots, so taking the first 128 games or deduplicating them changes the protocol.
- Sampling uses the active validation parameters from the GRPO shell and `generate_sequences_agent`; that method has no stop strings. The runner preserves `detokenize=False` semantics by decoding generated token IDs itself.
- The old “six-type macro equals paper Avg.” interpretation is withdrawn. The new report separates slot-weighted Pass@1 from the six-type macro and marks the latter unavailable if a task type is absent.
- `qwen3_4b_inst_2507` was selected in an earlier exploratory decision from a tiny run. That does not establish the paper's model identity and is not used by the current configured runner. The current configured path remains `qwen3_4b`; local weights are not independently verified by model-card ancestry metadata.

The new runner and strict merger have CPU mock coverage, including multi-slot retry history isolation. This is not a GPU result. Generation parity, actual TextWorld execution, throughput, and scores still require the target-server GPU check. No job has been submitted from this implementation workspace.
