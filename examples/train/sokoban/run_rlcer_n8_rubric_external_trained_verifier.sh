#!/usr/bin/env bash
# Sokoban RLCER using the trained Qwen2.5-VL LoRA verifier over HTTP.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

export VERIFIER_MODE=external
export VERIFIER_BASE_URL="${VERIFIER_BASE_URL:-http://127.0.0.1:5007/v1}"
export VERIFIER_MODEL="${VERIFIER_MODEL:-qwen-verifier-lora}"
export VERIFIER_API_KEY="${VERIFIER_API_KEY:-EMPTY}"
export VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-120}"
export VERIFIER_SINGLE_CRITERION_REQUESTS="${VERIFIER_SINGLE_CRITERION_REQUESTS:-1}"
# One question group has n=8 trajectories. Concurrently judge those independent
# trajectories while retaining one-criterion requests that match SFT training.
export VERIFIER_GROUP_CONCURRENCY="${VERIFIER_GROUP_CONCURRENCY:-8}"
# Maximum in-flight external VLM requests for one n=8 question group.
export VERIFIER_REQUEST_CONCURRENCY="${VERIFIER_REQUEST_CONCURRENCY:-16}"
# The SFT target is a short JSON boolean list; 64 tokens leaves ample margin.
export VERIFIER_MAX_TOKENS="${VERIFIER_MAX_TOKENS:-64}"
export TRAINING_SEED="${TRAINING_SEED:-45}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sokoban_rlcer_n8_external_trained_verifier_seed_${TRAINING_SEED}}"
export CORR_SCHEMA_VERSION="${CORR_SCHEMA_VERSION:-sokoban_rlcer_n8_external_trained_verifier_v1}"

if [[ "${SKIP_VERIFIER_CHECK:-0}" != 1 ]]; then
  bash "${BASE_DIR}/examples/deploy/verifier/check_qwen25_vl_3b.sh"
fi
exec "${SCRIPT_DIR}/run_rlcer_n8_rubric_test.sh"
