#!/bin/bash
#
# PLAN B: dual_role=true with FULLY ISOLATED dual update_actor (rubricator evolves
#          in its own clean forward, no shared batch-stat with reasoner).
#
# Keeps enable_dual_role_update=true AND dual_role_strategy=shared_rho so that:
#   (A) rubric generation still runs (current actor produces rubrics)
#   (B) grounded process reward (lambda_cot * cot_reward) still flows into the
#       reasoner's score — world-model CoT supervision preserved
#   (C) rubricator STILL gets PPO-style updates, but via an INDEPENDENT second
#       update_actor call (`rubricator_update_mode=isolate`), never concatenated
#       into the reasoner forward.  Both roles evolve; neither contaminates the
#       other's batch statistics / PPO ratio.
#
# This is the TRUE dual-role: unlike plan A (which dropped C entirely, ≈ false),
# plan B keeps the rubricator gradient but runs it cleanly.  Whether plan B can
# MATCH or EXCEED plan A / the false baseline (0.93) is the open research question
# this run answers.
#
# Bug fixes baked in (vs a naive dual-forward attempt):
#   - no rub_loss_weight double scaling: advantages are pre-scaled upstream and
#     the vanilla single-role path applies no extra rub_w
#   - ref_log_prob is computed for the rubricator batch when use_kl_loss=True
#     (rubricator was never run through the ref policy before)
#   - rubricator batch is padded to be divisible by DP world_size
#   - isolated_rubricator_batch state is reset every step (no stale carry-over)
#
# DELTA vs dual_role baseline:
#   custom_reward_function.reward_kwargs.rubricator_update_mode=isolate   (NEW)
# Everything else identical.
#
# SUCCESS CRITERIA (read from log):
#   - rlcer/train/rubricator_update_mode == 1.0 every step   (switch took effect)
#   - actor/ppo_kl ~ 0.0003 level                              (reasoner clean)
#   - actor_rub/ppo_kl, actor_rub/entropy present              (rubricator updates)
#   - actor/entropy back to healthy range
#   - aux/sokoban/traj_success/mean@1 → compare vs plan A (~0.93) and dual_role (0.54)
#
# KEY COMPARISON:
#   - If plan B ≥ plan A (~0.93)  -> rubricator evolution has POSITIVE value,
#                                    dual_role justified. Pursue plan B.
#   - If plan B ≈ plan A          -> rubricator evolution is neutral; plan A
#                                    (simpler, no extra forward) is preferred.
#   - If plan B < plan A          -> rubricator evolution mildly harmful even
#                                    when clean; use plan A, retire C.

set -x

PROJECT_NAME="jiayu_agent"
EXPERIMENT_NAME="ppo_qwen25vl3b_planb_isolate_svg_0701"

SCRIPTDIR=$(cd "$(dirname "$0")" && pwd)
BASEDIR=$(cd "${SCRIPTDIR}/../../.." && pwd)
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_svg_vision.yaml
DATASET_VAL=${SCRIPTDIR}/val_svg_vision.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=/personal/jiayu2026/models/vagen_svg
mkdir -p ${EXPERIMENT_DIR}

RAY_PORT=0 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=4,5 /opt/conda/envs/vagen/bin/python3 -m vagen.main_ppo \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=16 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
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
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=600 \
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
    custom_reward_function.reward_kwargs.dual_role_strategy=shared_rho \
    custom_reward_function.reward_kwargs.rubricator_update_mode=isolate \
    custom_reward_function.reward_kwargs.alpha=0.2 \
    custom_reward_function.reward_kwargs.lambda_cot=0.5 \
    custom_reward_function.reward_kwargs.rubricator_loss_weight=0.0005 \
    custom_reward_function.reward_kwargs.outcome_weight=1.0 \
    custom_reward_function.reward_kwargs.fallback_to_heuristic=true \
    custom_reward_function.reward_kwargs.policy_rubricator_max_new_tokens=4096 \
    custom_reward_function.reward_kwargs.normalize_rubric_weights=true \
    custom_reward_function.reward_kwargs.ema_rule_corr_filter=false \
    custom_reward_function.reward_kwargs.trivial_rubric_cooldown=false \
    custom_reward_function.reward_kwargs.alignment_rubricator_reward=true \
    custom_reward_function.reward_kwargs.rubricator.mode=policy \
    custom_reward_function.reward_kwargs.rubricator.max_rubrics=8 \
    custom_reward_function.reward_kwargs.verifier.mode=svg_grounded \
    custom_reward_function.reward_kwargs.scheduler.warmup_enabled=true \
    custom_reward_function.reward_kwargs.scheduler.warmup_steps=100 \
    custom_reward_function.reward_kwargs.scheduler.warmup_from_lambda_cot=0.0 \
    custom_reward_function.reward_kwargs.scheduler.warmup_from_rub_loss_weight=0.0 \
    custom_reward_function.reward_kwargs.scheduler.corr_gating_enabled=false \
    custom_reward_function.reward_kwargs.scheduler.plateau_enabled=false \
    trainer.total_epochs=400 \
    trainer.total_training_steps=1200 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >(tee ${BASEDIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log >/dev/null)
