#!/bin/bash

set -x

PROJECT_NAME="vagen_experiments"
EXPERIMENT_NAME="ppo_qwen25vl3b_rlcer_dual_role"

SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
BASEDIR=$(cd "${SCRIPTDIR}/../../.." && pwd)
# Ensure local verl package (BASEDIR/verl/verl) is importable even without `pip install -e ./verl`.
export PYTHONPATH="${BASEDIR}/verl:${BASEDIR}:${PYTHONPATH}"

EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_vision.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_vision.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# OpenAI-compatible endpoint for rubricator.policy mode.
# 说明：policy 模式下 rubricator 会通过该服务调用“同一个模型”。
RUBRICATOR_BASE_URL=${RUBRICATOR_BASE_URL:-"http://127.0.0.1:8000/v1"}
RUBRICATOR_MODEL=${RUBRICATOR_MODEL:-"Qwen/Qwen2.5-VL-3B-Instruct"}
RUBRICATOR_API_KEY=${RUBRICATOR_API_KEY:-"EMPTY"}

mkdir -p ${EXPERIMENT_DIR}

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=32 \
    data.max_prompt_length=1000 \
    data.max_response_length=4000 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${REF_MODEL_PATH} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    reward_model.reward_manager=batch \
    custom_reward_function.path=vagen/rlcer/reward_rlcer.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.enable_dual_role_update=true \
    custom_reward_function.reward_kwargs.alpha=0.2 \
    custom_reward_function.reward_kwargs.lambda_cot=1.0 \
    custom_reward_function.reward_kwargs.outcome_weight=1.0 \
    custom_reward_function.reward_kwargs.fallback_to_heuristic=true \
    custom_reward_function.reward_kwargs.rubricator.mode=policy \
    custom_reward_function.reward_kwargs.rubricator.base_url=${RUBRICATOR_BASE_URL} \
    custom_reward_function.reward_kwargs.rubricator.api_key=${RUBRICATOR_API_KEY} \
    custom_reward_function.reward_kwargs.rubricator.model=${RUBRICATOR_MODEL} \
    custom_reward_function.reward_kwargs.rubricator.max_rubrics=8 \
    custom_reward_function.reward_kwargs.verifier.mode=heuristic \
    trainer.total_training_steps=401 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >(tee ${BASEDIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >/dev/null)
