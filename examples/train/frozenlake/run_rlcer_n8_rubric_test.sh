#!/usr/bin/env bash
#
# 1200-step FrozenLake RLCER run for accepted/rejected rubric statistics.
# Mirrors the corrected Sokoban n=8 long-run setup and writes a structured report.
#
# Examples:
#   bash examples/train/frozenlake/run_rlcer_n8_rubric_test.sh
#   TOTAL_STEPS=1200 CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 \
#     bash examples/train/frozenlake/run_rlcer_n8_rubric_test.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
TASK_NAME=${TASK_NAME:-FrozenLake}
TARGET_LABEL=${TARGET_LABEL:-FrozenLake trajectory success}
ANALYSIS_DIR=${ANALYSIS_DIR:-${BASE_DIR}/examples/ablation/rubric_analysis/frozenlake}
VERIFIER_MODE=${VERIFIER_MODE:-grounded}
ALIGNMENT_TARGET=${ALIGNMENT_TARGET:-traj_success}
CORR_SCHEMA_VERSION=${CORR_SCHEMA_VERSION:-frozenlake_rlcer_n8_rubric_long_v1}

TRAINING_SEED=${TRAINING_SEED:-44}
PROJECT_NAME=${PROJECT_NAME:-jiayu_agent}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-frozenlake_rlcer_n8_rubric_long_v1_seed_${TRAINING_SEED}}
EXPERIMENT_DIR=${EXPERIMENT_DIR:-${BASE_DIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}}
CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
ROLLOUT_DIR=${EXPERIMENT_DIR}/rollout_data
VALIDATION_DIR=${EXPERIMENT_DIR}/validation
ANALYSIS_OUTPUT_DIR=${EXPERIMENT_DIR}/rubric_analysis
LOG_FILE=${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log

DATASET_TRAIN=${DATASET_TRAIN:-${SCRIPT_DIR}/train_frozenlake_vision.yaml}
DATASET_VAL=${DATASET_VAL:-${SCRIPT_DIR}/val_frozenlake_vision.yaml}
AGENT_LOOP_CONFIG=${BASE_DIR}/vagen/configs/agent.yaml
REF_MODEL_PATH=${REF_MODEL_PATH:-/personal/jiayu2026/models/vagen_frozenlake}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/vagen/bin/python3}
GPU_LIST=${CUDA_VISIBLE_DEVICES:-4,5}
N_GPUS=${N_GPUS:-2}

TOTAL_STEPS=${TOTAL_STEPS:-1200}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
RUBRICATOR_SUBSAMPLE_SIZE=${RUBRICATOR_SUBSAMPLE_SIZE:-16}
RUBRICATOR_UPDATE_INTERVAL=${RUBRICATOR_UPDATE_INTERVAL:-4}
DUAL_ROLE_MIN_VALID_RATIO=${DUAL_ROLE_MIN_VALID_RATIO:-0.01}
SAVE_FREQ=${SAVE_FREQ:-800}
TEST_FREQ=${TEST_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
AUDIT_INTERVAL=${AUDIT_INTERVAL:-20}
AUDIT_MAX_GROUPS=${AUDIT_MAX_GROUPS:-16}
CORR_MIN_SAMPLES=${CORR_MIN_SAMPLES:-64}
WARMUP_STEPS=${WARMUP_STEPS:-200}
TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}

for required_path in \
    "${PYTHON_BIN}" \
    "${REF_MODEL_PATH}" \
    "${DATASET_TRAIN}" \
    "${DATASET_VAL}" \
    "${ANALYSIS_DIR}/analyze_rubrics.py"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 2
    fi
done

if (( TRAIN_BATCH_SIZE < PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_BATCH_SIZE must be >= PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi

mkdir -p "${EXPERIMENT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}"

echo "Starting ${TASK_NAME} RLCER n=8 1200-step rubric run"
echo "  experiment=${PROJECT_NAME}/${EXPERIMENT_NAME}"
echo "  steps=${TOTAL_STEPS}, train_batch=${TRAIN_BATCH_SIZE}, rollout.n=8"
echo "  audit_interval=${AUDIT_INTERVAL}, corr_min_samples=${CORR_MIN_SAMPLES}"
echo "  log=${LOG_FILE}"

RAY_PORT=0 \
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
"${PYTHON_BIN}" -m vagen.main_ppo \
    --config-path="${BASE_DIR}/vagen/configs" \
    --config-name=vagen_multiturn \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.seed="${TRAINING_SEED}" \
    +data.base_seed="${TRAINING_SEED}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=1000 \
    data.max_response_length=4000 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.use_fused_kernels=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.disable_log_stats=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    critic.optim.lr=1e-5 \
    critic.model.path="${REF_MODEL_PATH}" \
    critic.model.use_remove_padding=true \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    reward_model.reward_manager=batch \
    custom_reward_function.path=vagen/rlcer/reward_rlcer.py \
    custom_reward_function.name=compute_score \
    custom_reward_function.reward_kwargs.enable_dual_role_update=true \
    custom_reward_function.reward_kwargs.dual_role_strategy=shared_rho \
    custom_reward_function.reward_kwargs.rubricator_update_mode=protected \
    custom_reward_function.reward_kwargs.rubricator_grad_norm_ratio_cap=0.02 \
    custom_reward_function.reward_kwargs.anchor_kl.enabled=true \
    custom_reward_function.reward_kwargs.anchor_kl.interval=20 \
    custom_reward_function.reward_kwargs.anchor_kl.samples=8 \
    custom_reward_function.reward_kwargs.rubric_replay.enabled=true \
    custom_reward_function.reward_kwargs.rubric_replay.max_batches=4 \
    custom_reward_function.reward_kwargs.rubric_replay.min_valid_ratio=0.05 \
    custom_reward_function.reward_kwargs.rubric_replay.recovery_batch_size=8 \
    custom_reward_function.reward_kwargs.rubric_replay.recovery_advantage=1.0 \
    custom_reward_function.reward_kwargs.alpha=0.1 \
    custom_reward_function.reward_kwargs.lambda_cot=0.2 \
    custom_reward_function.reward_kwargs.rubricator_loss_weight=0.0005 \
    custom_reward_function.reward_kwargs.rubricator_subsample_size="${RUBRICATOR_SUBSAMPLE_SIZE}" \
    custom_reward_function.reward_kwargs.rubricator_update_interval="${RUBRICATOR_UPDATE_INTERVAL}" \
    custom_reward_function.reward_kwargs.dual_role_min_valid_ratio="${DUAL_ROLE_MIN_VALID_RATIO}" \
    custom_reward_function.reward_kwargs.outcome_weight=1.0 \
    custom_reward_function.reward_kwargs.fallback_to_heuristic=false \
    custom_reward_function.reward_kwargs.policy_rubricator_max_new_tokens=4096 \
    custom_reward_function.reward_kwargs.normalize_rubric_weights=true \
    custom_reward_function.reward_kwargs.ema_rule_corr_filter=true \
    custom_reward_function.reward_kwargs.corr_ema_beta=0.999 \
    custom_reward_function.reward_kwargs.corr_min_samples="${CORR_MIN_SAMPLES}" \
    custom_reward_function.reward_kwargs.corr_calibration_interval=1 \
    custom_reward_function.reward_kwargs.corr_schema_version="${CORR_SCHEMA_VERSION}" \
    custom_reward_function.reward_kwargs.insufficient_corr_policy=reject \
    custom_reward_function.reward_kwargs.alignment_target="${ALIGNMENT_TARGET}" \
    custom_reward_function.reward_kwargs.trivial_rubric_cooldown=false \
    custom_reward_function.reward_kwargs.trivial_rubric_scale=0.5 \
    custom_reward_function.reward_kwargs.alignment_rubricator_reward=true \
    custom_reward_function.reward_kwargs.rubric_validation_enabled=true \
    custom_reward_function.reward_kwargs.rubric_audit.enabled=true \
    custom_reward_function.reward_kwargs.rubric_audit.interval="${AUDIT_INTERVAL}" \
    custom_reward_function.reward_kwargs.rubric_audit.max_groups="${AUDIT_MAX_GROUPS}" \
    custom_reward_function.reward_kwargs.rubric_audit.include_raw_proposal=true \
    custom_reward_function.reward_kwargs.rubric_audit.max_raw_chars=6000 \
    custom_reward_function.reward_kwargs.rubricator.mode=policy \
    custom_reward_function.reward_kwargs.rubricator.max_rubrics=8 \
    custom_reward_function.reward_kwargs.verifier.mode="${VERIFIER_MODE}" \
    custom_reward_function.reward_kwargs.scheduler.warmup_enabled=true \
    custom_reward_function.reward_kwargs.scheduler.warmup_steps="${WARMUP_STEPS}" \
    custom_reward_function.reward_kwargs.scheduler.warmup_from_lambda_cot=0.0 \
    custom_reward_function.reward_kwargs.scheduler.warmup_from_rub_loss_weight=0.0 \
    custom_reward_function.reward_kwargs.scheduler.corr_gating_enabled=false \
    custom_reward_function.reward_kwargs.scheduler.plateau_enabled=false \
    custom_reward_function.reward_kwargs.scheduler.recovery_enabled=true \
    custom_reward_function.reward_kwargs.scheduler.recovery_enter_threshold=0.8 \
    custom_reward_function.reward_kwargs.scheduler.recovery_exit_threshold=0.95 \
    custom_reward_function.reward_kwargs.scheduler.recovery_enter_patience=5 \
    custom_reward_function.reward_kwargs.scheduler.recovery_exit_patience=20 \
    custom_reward_function.reward_kwargs.scheduler.recovery_warm_restart_steps=50 \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.resume_mode=auto \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.log_val_generations=32 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    2>&1 | tee "${LOG_FILE}"

echo "Training complete. Building ${TASK_NAME} rubric report."
if grep -q '"rubric_audit"' "${ROLLOUT_DIR}"/*.jsonl 2>/dev/null; then
    "${PYTHON_BIN}" "${ANALYSIS_DIR}/analyze_rubrics.py" \
        --rollout-dir "${ROLLOUT_DIR}" \
        --output-dir "${ANALYSIS_OUTPUT_DIR}" \
        --task-name "${TASK_NAME}" \
        --target-label "${TARGET_LABEL}" \
        --phase-size 200 \
        2>&1 | tee "${EXPERIMENT_DIR}/rubric_analysis.log"
else
    echo "No rubric_audit record yet; skipping report generation for this short run."
fi

echo "Key online metrics:"
grep -E "rubric_acceptance_rate|correlation_unavailable_rate|accepted_correlation|rejected_correlation" \
    "${LOG_FILE}" | tail -n 120 || true

echo "Reports:"
echo "  ${ANALYSIS_OUTPUT_DIR}/rubric_metrics_over_training.csv"
echo "  ${ANALYSIS_OUTPUT_DIR}/rubric_metrics_by_phase.csv"
echo "  ${ANALYSIS_OUTPUT_DIR}/rubric_audit_summary.json"
echo "  ${ANALYSIS_OUTPUT_DIR}/rubric_analysis_report.md"
