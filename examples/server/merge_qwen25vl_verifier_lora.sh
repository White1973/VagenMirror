#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/vagen/bin/python}"
BASE_MODEL="${BASE_MODEL:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Instruct}"
ADAPTER="${ADAPTER:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Verifier-LoRA/final_adapter}"
MERGED_MODEL="${MERGED_MODEL:-/personal/jiayu2026/models/Qwen2.5-VL-3B-Verifier-Merged}"

exec "${PYTHON_BIN}" "${BASE_DIR}/tools/verifier_data/merge_qwen25vl_verifier_lora.py" \
  --base-model "${BASE_MODEL}" --adapter "${ADAPTER}" --output "${MERGED_MODEL}"
