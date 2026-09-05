#!/usr/bin/env python3
"""LoRA SFT for a Qwen2.5-VL rubric verifier."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)


def disable_incompatible_torchao_dispatch() -> None:
    """Let ordinary BF16 LoRA work when an unrelated old torchao is present.

    PEFT 0.20 raises during every target-module probe when torchao is installed
    but older than 0.16, even when the model has no TorchAO quantized weights.
    This model is plain BF16, so skipping only that dispatcher is correct; PEFT
    then falls through to its standard torch.nn.Linear implementation.
    """
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return
    from packaging.version import Version
    if Version(version) < Version("0.16.0"):
        import peft.tuners.lora.torchao as peft_torchao
        peft_torchao.is_torchao_available = lambda: False
        warnings.warn(
            f"Ignoring incompatible torchao=={version} for unquantized BF16 LoRA; "
            "PEFT will use standard torch.nn.Linear adapters."
        )


def load_jsonl(path: Path) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if len(row.get("images", [])) != 1 or len(row.get("messages", [])) != 2:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable examples in {path}")
    return Dataset.from_list(rows)


def vl_messages(example: dict, include_assistant: bool) -> list[dict]:
    image = example["images"][0]
    user_text = example["messages"][0]["content"]
    if user_text.startswith("<image>\n"):
        user_text = user_text[len("<image>\n"):]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    }]
    if include_assistant:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": example["messages"][1]["content"]}],
        })
    return messages


class VerifierCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    def _encode(self, messages, generation_prompt: bool):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt
        )
        images, videos = process_vision_info(messages)
        return self.processor(
            text=[text], images=images, videos=videos, padding=False,
            truncation=True, max_length=self.max_length, return_tensors="pt",
        )

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        encoded = []
        prompt_lengths = []
        for example in examples:
            full_messages = vl_messages(example, include_assistant=True)
            prompt_messages = vl_messages(example, include_assistant=False)
            full = self._encode(full_messages, generation_prompt=False)
            prompt = self._encode(prompt_messages, generation_prompt=True)
            encoded.append({
                "input_ids": full["input_ids"][0],
                "attention_mask": full["attention_mask"][0],
                "pixel_values": full["pixel_values"],
                "image_grid_thw": full["image_grid_thw"],
            })
            prompt_lengths.append(prompt["input_ids"].shape[1])

        text_features = [
            {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
            for x in encoded
        ]
        batch = self.processor.tokenizer.pad(
            text_features, padding=True, return_tensors="pt"
        )
        # Qwen2.5-VL represents each image as a variable number of flattened
        # patches; concatenate these exactly as the processor does for a batch.
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in encoded], dim=0)
        batch["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in encoded], dim=0)
        labels = batch["input_ids"].clone()
        for i, prompt_len in enumerate(prompt_lengths):
            labels[i, : min(prompt_len, labels.shape[1])] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--validation-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--save-steps", type=int, default=250)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=45)
    p.add_argument("--max-train-samples", type=int)
    p.add_argument("--max-eval-samples", type=int)
    return p.parse_args()


def main():
    args = parse_args(); set_seed(args.seed)
    disable_incompatible_torchao_dispatch()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    for parameter in model.visual.parameters():
        parameter.requires_grad = False
    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        # A full-name regex is required: Qwen2.5-VL's vision MLP also uses
        # gate/up/down_proj names, and suffix matching would train visual LoRA.
        target_modules=(
            r"model\.language_model\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))"
        ),
    )
    model = get_peft_model(model, peft_config)
    if int(__import__("os").environ.get("LOCAL_RANK", "0")) == 0:
        model.print_trainable_parameters()

    train = load_jsonl(args.train_file)
    valid = load_jsonl(args.validation_file)
    if args.max_train_samples: train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_eval_samples: valid = valid.select(range(min(args.max_eval_samples, len(valid))))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, warmup_ratio=0.05, weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=args.logging_steps, report_to="none",
        remove_unused_columns=False, ddp_find_unused_parameters=False,
        dataloader_num_workers=2, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=valid,
        data_collator=VerifierCollator(processor, args.max_length),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    processor.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
