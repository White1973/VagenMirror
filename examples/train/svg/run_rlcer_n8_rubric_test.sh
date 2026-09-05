#!/usr/bin/env bash
# 1200-step SVG RLCER run with protected shared-actor dual-role updates.
#
# Examples:
#   bash examples/train/svg/run_rlcer_n8_rubric_test.sh
#   TRAINING_SEED=43 CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 \
#     bash examples/train/svg/run_rlcer_n8_rubric_test.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

export TASK_NAME=${TASK_NAME:-SVG}
export TARGET_LABEL=${TARGET_LABEL:-"SVG trajectory success"}
export ANALYSIS_DIR=${ANALYSIS_DIR:-${BASE_DIR}/examples/ablation/rubric_analysis/svg}
export VERIFIER_MODE=${VERIFIER_MODE:-svg_grounded}
export ALIGNMENT_TARGET=${ALIGNMENT_TARGET:-env_reward}
export CORR_SCHEMA_VERSION=${CORR_SCHEMA_VERSION:-svg_rlcer_n8_rubric_protected_v1}
export TRAINING_SEED=${TRAINING_SEED:-42}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-svg_rlcer_n8_rubric_protected_v1_seed_${TRAINING_SEED}}
export DATASET_TRAIN=${DATASET_TRAIN:-${SCRIPT_DIR}/train_svg_vision.yaml}
export DATASET_VAL=${DATASET_VAL:-${SCRIPT_DIR}/val_svg_vision.yaml}
export REF_MODEL_PATH=${REF_MODEL_PATH:-/personal/jiayu2026/models/vagen_svg}
export TOTAL_STEPS=${TOTAL_STEPS:-1200}
export PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/vagen/bin/python3}

exec "${BASE_DIR}/examples/train/frozenlake/run_rlcer_n8_rubric_test.sh"
