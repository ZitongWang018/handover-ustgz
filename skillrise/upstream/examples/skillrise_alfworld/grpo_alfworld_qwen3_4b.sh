#!/usr/bin/env bash
set -x
# Pure GRPO baseline (1 attempt, no meta). Full config for 8 GPUs/node
# (8 GPUs/node). group_size=24 so per-step rollout volume matches the SkillRise volume (16x8x3=384 task-plays).
ENGINE=${1:-vllm}
shift 2>/dev/null || true
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

pip3 install alfworld

project_name="verl_agent_alfworld_meta"
TIMESTAMP="$(date +%Y%m%d%H%M%S)"
experiment_name="grpo_qwen3-4b_${TIMESTAMP}"
export OUTPUT_DIR="${SKILLRISE_OUTPUT_ROOT:-$HOME/experiments/skillrise}/${project_name}/${experiment_name}"
mkdir -p "$OUTPUT_DIR"

train_data_size=16
val_data_size=128
group_size=24
mode="mean_norm"
num_cpus_per_env_worker=0.15

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$SKILLRISE_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.5 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    +algorithm.step_gamma=0.95 \
    +algorithm.traj_gamma=0.6 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    reward_model.reward_manager=episode \
    env.env_name="alfworld/AlfredTWEnv" \
    env.seed=0 \
    env.alfworld.eval_dataset=eval_in_distribution \
    env.max_steps=30 \
    env.max_turns=10 \
    env.rollout.n=$group_size \
    env.num_attempts=1 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    +env.do_reflection=False \
    +env.val_num_attempts=3 \
    +env.val_do_reflection=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.rollout_data_dir=$OUTPUT_DIR/rollouts \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.log_val_generations=10 \
    trainer.save_freq=5 \
    trainer.save_start_step=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.ray_wait_register_center_timeout=3600 \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/grpo_alfworld_qwen3_4b.log"
