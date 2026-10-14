# SkillRise ALFWorld evaluation profiles

This file records the executable comparison choices. It does not claim to recover the paper's unpublished baseline prompts or task list.

## ReAct profile

The vendored source snapshot is from `Within-yao/SkillRise` commit `2b38da3ad7b64414305d39609484803b07d3b9b0`. The active dispatcher in `agent_system/multi_turn_rollout/__init__.py` exports `skillrise_rollout_loop.TrajectoryCollector`; the separate generic `rollout_loop.py` is not the validation path. The runner calls the active source's ALFWorld prompt, `SimpleMemory`, and projection functions.

The evaluation uses `valid_seen`, 128 worker slots with seeds 1000 through 1127, and each worker's first game after shuffling the full unsorted pool. Repeated game IDs are retained as separate slots. Each attempt has at most ten turns; the active validation config performs three same-task attempts without reflection and skips a slot after it wins. Attempt zero is reported as Pass@1. Cumulative Pass@2/3 are additional code-path diagnostics.

The active prompt explicitly disables Qwen thinking mode. Each turn is a single user message, chat-templated and tokenized with special-token addition disabled. The upstream action projection receives the model's decoded token IDs; an empty or malformed projection is passed through unchanged. Prompts longer than 4096 tokens fail explicitly. The official TextWorld environment budget remains 50 steps.

## Zero-shot-derived profile

The upstream repository does not publish a zero-shot prompt for this comparison. With user authorization, this profile is provided as a **custom derived approximation**: it removes the two reasoning instructions from the upstream play prompt while preserving its remaining prompt, admissible-action scaffold, environment, and decoding settings. It is not presented as the authors' zero-shot baseline.

## Model identity and prior runs

The configured model path remains `qwen3_4b`; no weights were replaced. A model card's `base_model` field names an ancestor and does not prove the current directory's weight identity. The local identity report is metadata-only and labels the weights `weights_unverified` unless separately hashed and checked. The earlier decision to switch to `qwen3_4b_inst_2507` based on a tiny success-rate comparison is withdrawn; old result files remain historical artifacts and are not results for the code-aligned runner.

## Claims and measurements

The primary suite metric is slot-weighted Pass@1 over all 128 selected slots. Per-type success rates and the six-type macro are reported separately; a missing type makes the six-type macro unavailable. The 128 slots may include repeated game instances by design. No GPU evaluation, speedup, or score improvement has yet been measured for this runner.
