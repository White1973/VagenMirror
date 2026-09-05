#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Verifier-Merged}"
export VERIFIER_SERVED_MODEL_NAME="${VERIFIER_SERVED_MODEL_NAME:-qwen-verifier-lora}"
export VERIFIER_PORT="${VERIFIER_PORT:-5007}"

if [[ ! -f "${VERIFIER_MODEL_PATH}/config.json" ]]; then
  echo "[ERROR] Merged verifier is missing: ${VERIFIER_MODEL_PATH}" >&2
  echo "Run: bash ${SCRIPT_DIR}/merge_qwen25vl_verifier_lora.sh" >&2
  exit 2
fi
exec "${SCRIPT_DIR}/serve_qwen25_vl_3b.sh"
