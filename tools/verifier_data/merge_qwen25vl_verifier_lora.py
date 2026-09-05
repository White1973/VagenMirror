#!/usr/bin/env python3
"""Merge the trained verifier LoRA into a standalone Qwen2.5-VL checkpoint."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from train_qwen25vl_verifier import disable_incompatible_torchao_dispatch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output}")

    disable_incompatible_torchao_dispatch()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True, max_shard_size="4GB")
    processor = AutoProcessor.from_pretrained(args.adapter, trust_remote_code=True)
    processor.save_pretrained(args.output)
    print(f"Merged verifier saved to {args.output}")


if __name__ == "__main__":
    main()
