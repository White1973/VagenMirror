#!/usr/bin/env bash
set -euo pipefail

# Deploy Qwen2.5-VL-3B-Instruct as an OpenAI-compatible external verifier.
#
# Example:
#   CUDA_VISIBLE_DEVICES=7 bash examples/deploy/verifier/serve_qwen25_vl_3b.sh
#
# Optional overrides:
#   VERIFIER_PORT=5007
#   VERIFIER_HOST=0.0.0.0
#   VERIFIER_MODEL_PATH=/path/to/model
#   VERIFIER_SERVED_MODEL_NAME=qwen-verifier
#   VERIFIER_MEM_FRACTION=0.80
#   VERIFIER_CONTEXT_LENGTH=8192
#   VERIFIER_TP_SIZE=1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is not set."
    echo "Select GPU(s) not used by PPO, for example:"
    echo "  CUDA_VISIBLE_DEVICES=7 bash $0"
    exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vagen/bin/python3}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
VERIFIER_SERVED_MODEL_NAME="${VERIFIER_SERVED_MODEL_NAME:-qwen-verifier}"
VERIFIER_HOST="${VERIFIER_HOST:-0.0.0.0}"
VERIFIER_PORT="${VERIFIER_PORT:-5007}"
VERIFIER_TP_SIZE="${VERIFIER_TP_SIZE:-1}"
VERIFIER_MEM_FRACTION="${VERIFIER_MEM_FRACTION:-0.80}"
VERIFIER_CONTEXT_LENGTH="${VERIFIER_CONTEXT_LENGTH:-8192}"
VERIFIER_MAX_RUNNING_REQUESTS="${VERIFIER_MAX_RUNNING_REQUESTS:-32}"

# FlashInfer defaults to ~/.cache. Ray/container jobs may have a read-only home,
# so keep all runtime caches inside the writable workspace.
VERIFIER_CACHE_ROOT="${VERIFIER_CACHE_ROOT:-/personal/jiayu2026/.cache/qwen25_vl_verifier}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VERIFIER_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${VERIFIER_CACHE_ROOT}/huggingface}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${VERIFIER_CACHE_ROOT}/flashinfer}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${FLASHINFER_WORKSPACE_BASE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Python executable not found: ${PYTHON_BIN}"
    exit 2
fi
if [[ ! -d "${VERIFIER_MODEL_PATH}" ]]; then
    echo "[ERROR] Verifier model directory not found: ${VERIFIER_MODEL_PATH}"
    exit 2
fi
if (( VERIFIER_TP_SIZE < 1 )); then
    echo "[ERROR] VERIFIER_TP_SIZE must be at least 1."
    exit 2
fi

IFS=',' read -r -a verifier_gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#verifier_gpu_list[@]} < VERIFIER_TP_SIZE )); then
    echo "[ERROR] VERIFIER_TP_SIZE=${VERIFIER_TP_SIZE}, but only ${#verifier_gpu_list[@]} GPU(s) are visible."
    exit 2
fi

echo "[INFO] Starting Qwen2.5-VL external verifier"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] model_path=${VERIFIER_MODEL_PATH}"
echo "[INFO] served_model_name=${VERIFIER_SERVED_MODEL_NAME}"
echo "[INFO] endpoint=http://127.0.0.1:${VERIFIER_PORT}/v1"
echo
echo "[INFO] Use these settings in the training shell:"
echo "  export VERIFIER_BASE_URL=http://127.0.0.1:${VERIFIER_PORT}/v1"
echo "  export VERIFIER_MODEL=${VERIFIER_SERVED_MODEL_NAME}"
echo "  export VERIFIER_API_KEY=EMPTY"

exec "${PYTHON_BIN}" -m sglang.launch_server \
    --host "${VERIFIER_HOST}" \
    --port "${VERIFIER_PORT}" \
    --model-path "${VERIFIER_MODEL_PATH}" \
    --served-model-name "${VERIFIER_SERVED_MODEL_NAME}" \
    --tp-size "${VERIFIER_TP_SIZE}" \
    --mem-fraction-static "${VERIFIER_MEM_FRACTION}" \
    --context-length "${VERIFIER_CONTEXT_LENGTH}" \
    --max-running-requests "${VERIFIER_MAX_RUNNING_REQUESTS}" \
    --enable-multimodal \
    --trust-remote-code \
    --log-level warning
