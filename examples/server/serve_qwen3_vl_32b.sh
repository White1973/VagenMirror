#!/usr/bin/env bash
set -euo pipefail

# Deploy Qwen3-VL-32B-Instruct as the OpenAI-compatible external verifier.
#
# Single-GPU deployment (requires enough GPU memory):
#   CUDA_VISIBLE_DEVICES=7 \
#     bash examples/deploy/verifier/serve_qwen3_vl_32b.sh
#
# Optional multi-GPU tensor parallel deployment:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 VERIFIER_TP_SIZE=4 \
#     bash examples/deploy/verifier/serve_qwen3_vl_32b.sh

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES is not set."
    echo "Select GPUs not used by PPO, for example:"
    echo "  CUDA_VISIBLE_DEVICES=7 bash $0"
    exit 2
fi

VERIFY_ENV="${VERIFY_ENV:-/personal/jiayu2026/conda_envs/verify_qwen3vl}"
PYTHON_BIN="${PYTHON_BIN:-${VERIFY_ENV}/bin/python}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/personal/jiayu2026/models/Qwen3-VL-32B-Instruct}"
VERIFIER_SERVED_MODEL_NAME="${VERIFIER_SERVED_MODEL_NAME:-qwen-verifier}"
VERIFIER_HOST="${VERIFIER_HOST:-127.0.0.1}"
VERIFIER_PORT="${VERIFIER_PORT:-5007}"
VERIFIER_TP_SIZE="${VERIFIER_TP_SIZE:-1}"
VERIFIER_MEM_FRACTION="${VERIFIER_MEM_FRACTION:-0.85}"
VERIFIER_CONTEXT_LENGTH="${VERIFIER_CONTEXT_LENGTH:-8192}"
VERIFIER_MAX_RUNNING_REQUESTS="${VERIFIER_MAX_RUNNING_REQUESTS:-16}"

VERIFIER_CACHE_ROOT="${VERIFIER_CACHE_ROOT:-/personal/jiayu2026/.cache/qwen3vl_verifier}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VERIFIER_CACHE_ROOT}/xdg}"
export HF_HOME="${HF_HOME:-${VERIFIER_CACHE_ROOT}/huggingface}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${VERIFIER_CACHE_ROOT}/flashinfer}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${FLASHINFER_WORKSPACE_BASE}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[ERROR] Qwen3-VL verifier Python not found: ${PYTHON_BIN}"
    echo "Expected Conda environment: ${VERIFY_ENV}"
    exit 2
fi
if [[ ! -f "${VERIFIER_MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] Model config not found: ${VERIFIER_MODEL_PATH}/config.json"
    exit 2
fi
if (( VERIFIER_TP_SIZE < 1 )); then
    echo "[ERROR] VERIFIER_TP_SIZE must be at least 1."
    exit 2
fi

IFS=',' read -r -a verifier_gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#verifier_gpu_list[@]} != VERIFIER_TP_SIZE )); then
    echo "[ERROR] Visible GPU count (${#verifier_gpu_list[@]}) must equal VERIFIER_TP_SIZE (${VERIFIER_TP_SIZE})."
    echo "Example: CUDA_VISIBLE_DEVICES=4,5,6,7 VERIFIER_TP_SIZE=4 bash $0"
    exit 2
fi

# Fail before allocating GPU memory if the wrong checkpoint or environment is
# selected.
"${PYTHON_BIN}" - "${VERIFIER_MODEL_PATH}" <<'PY'
import sys
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    sys.argv[1],
    trust_remote_code=True,
    local_files_only=True,
)
if config.model_type != "qwen3_vl":
    raise SystemExit(
        f"Expected model_type='qwen3_vl', got {config.model_type!r}"
    )
print(
    f"[INFO] Transformers recognized {type(config).__name__} "
    f"(model_type={config.model_type})"
)
PY

if [[ "${VERIFIER_CONFIG_ONLY:-0}" == "1" ]]; then
    echo "[INFO] Configuration-only validation passed; GPU server launch skipped."
    exit 0
fi

echo "[INFO] Starting Qwen3-VL-32B external verifier"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] tensor_parallel_size=${VERIFIER_TP_SIZE}"
echo "[INFO] model_path=${VERIFIER_MODEL_PATH}"
echo "[INFO] served_model_name=${VERIFIER_SERVED_MODEL_NAME}"
echo "[INFO] endpoint=http://${VERIFIER_HOST}:${VERIFIER_PORT}/v1"
echo
echo "[INFO] Training-side settings:"
echo "  export VERIFIER_BASE_URL=http://127.0.0.1:${VERIFIER_PORT}/v1"
echo "  export VERIFIER_MODEL=${VERIFIER_SERVED_MODEL_NAME}"
echo "  export VERIFIER_API_KEY=EMPTY"

extra_args=()
if [[ -n "${VERIFIER_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "${VERIFIER_EXTRA_ARGS}"
fi

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
    --log-level warning \
    "${extra_args[@]}"
