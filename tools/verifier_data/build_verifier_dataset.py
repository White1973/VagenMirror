#!/usr/bin/env python3
"""Build verifier candidate and SFT datasets from VAGEN rollout artifacts.

Two operations are intentionally separate:

* ``extract`` normalizes raw PPO rollouts into per-turn, image-backed records.
  These records are annotation candidates and contain no invented rubric labels.
* ``convert-judged`` converts teacher-judged rubric records into the exact
  multi-criterion JSON format consumed by ``LLMVerifier``.

Splits are assigned from the current-board image hash so identical boards cannot
cross train/validation/test boundaries.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ASSISTANT_SPLIT_RE = re.compile(
    r"(?:<\|im_start\|>assistant\s*)?(.*?)<\|im_end\|>", re.DOTALL
)
SECTION_RE = re.compile(
    r"<(observation|think|answer|prediction)>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)


def stable_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_hash(path: Path) -> str:
    return stable_hash_bytes(path.read_bytes())


def split_for_hash(value: str) -> str:
    bucket = int(value[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def numeric_jsonl_files(rollout_dir: Path) -> list[Path]:
    files = [p for p in rollout_dir.glob("*.jsonl") if p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def extract_turns(output: str) -> list[str]:
    turns = []
    for match in ASSISTANT_SPLIT_RE.finditer(output or ""):
        text = match.group(1).strip()
        if "<observation>" in text.lower() or "<answer>" in text.lower():
            turns.append(text)
    if not turns and (output or "").strip():
        turns.append((output or "").strip())
    return turns


def extract_sections(response: str) -> dict[str, str]:
    sections = {"observation": "", "think": "", "answer": "", "prediction": ""}
    for name, value in SECTION_RE.findall(response or ""):
        sections[name.lower()] = value.strip()
    return sections


def reservoir_trajectories(
    rollout_dir: Path, max_trajectories: int | None, seed: int
) -> list[tuple[Path, int, dict[str, Any]]]:
    rng = random.Random(seed)
    selected: list[tuple[Path, int, dict[str, Any]]] = []
    seen = 0
    for file_path in numeric_jsonl_files(rollout_dir):
        with file_path.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen += 1
                item = (file_path, row_idx, obj)
                if max_trajectories is None or len(selected) < max_trajectories:
                    selected.append(item)
                else:
                    j = rng.randrange(seen)
                    if j < max_trajectories:
                        selected[j] = item
    return sorted(selected, key=lambda x: (int(x[0].stem), x[1]))


def candidate_rows(
    task: str,
    experiment_dir: Path,
    max_trajectories: int | None,
    seed: int,
) -> Iterable[dict[str, Any]]:
    rollout_dir = experiment_dir / "rollout_data"
    for json_path, row_idx, obj in reservoir_trajectories(
        rollout_dir, max_trajectories=max_trajectories, seed=seed
    ):
        step = int(json_path.stem)
        image_dir = rollout_dir / f"image_{step}" / f"images_{row_idx}"
        images = sorted(
            image_dir.glob("*.png"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
        )
        turns = extract_turns(str(obj.get("output", "")))
        usable = min(len(images), len(turns))
        if usable == 0:
            continue
        trajectory_id = f"{task}:{step}:{row_idx}"
        history: list[str] = []
        for turn_idx in range(usable):
            img = images[turn_idx].resolve()
            response = turns[turn_idx]
            board_hash = image_hash(img)
            yield {
                "id": f"{trajectory_id}:{turn_idx}",
                "task": task,
                "split": split_for_hash(board_hash),
                "group_id": board_hash,
                "trajectory_id": trajectory_id,
                "step": step,
                "row_index": row_idx,
                "turn_index": turn_idx,
                "image": str(img),
                "trajectory_images": [str(p.resolve()) for p in images],
                "history": list(history),
                "response": response,
                "sections": extract_sections(response),
                "trajectory_success": float(obj.get("traj_success", 0.0)),
                "score": float(obj.get("score", 0.0)),
                "label_status": "unlabeled",
                "source": str(json_path.resolve()),
            }
            history.append(response)


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        return [
            {"role": str(x.get("role", "")), "content": str(x.get("content", ""))}
            for x in record["messages"]
        ]
    conversations = record.get("conversations", [])
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    return [
        {
            "role": role_map.get(str(x.get("from", "")), str(x.get("from", ""))),
            "content": str(x.get("value", "")),
        }
        for x in conversations
    ]


def parse_rubric(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if fence:
        raw = fence.group(1).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def verifier_prompt(task: str, response: str, criteria: list[str]) -> str:
    return (
        "Evaluate each criterion as strict True/False against the response reasoning.\n"
        "[+] criteria: True = quality IS present. [-] criteria: True = flaw IS present.\n"
        "Use the provided image to verify factual claims about object positions, "
        "actions, and spatial relationships.\n"
        "Return JSON only: {\"judgement\": [bool, ...]} with the same order.\n"
        f"Task: {task}\n\nResponse:\n{response}\n\n"
        f"Criteria:\n{json.dumps(criteria, ensure_ascii=False)}"
    )


def judged_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = normalize_messages(record)
            response = next(
                (m["content"] for m in reversed(messages) if m["role"] == "assistant"),
                "",
            )
            rubric = parse_rubric(str(record.get("rubric", "")))
            rules = (rubric or {}).get("rubrics", [])
            labels = record.get("judgement", [])
            images = [Path(p) for p in record.get("images", []) if Path(p).is_file()]
            if not response or not rules or not images or len(labels) != len(rules):
                continue
            task = "frozenlake" if "frozen" in " ".join(m["content"] for m in messages).lower() else "sokoban"
            annotated = []
            rule_ids = []
            for rule in rules:
                points = float(rule.get("weight", rule.get("points", 0.0)))
                annotated.append(("[+] " if points >= 0 else "[-] ") + str(rule.get("description", "")))
                rule_ids.append(str(rule.get("rule_id", "")))
            board_hash = image_hash(images[0])
            label_values = [bool(x) for x in labels]
            yield {
                "id": stable_hash_bytes(f"{path}:{line_idx}".encode())[:24],
                "task": task,
                "split": split_for_hash(board_hash),
                "group_id": board_hash,
                "images": [str(p.resolve()) for p in images],
                "messages": [
                    {"role": "user", "content": "<image>\n" + verifier_prompt(task, response, annotated)},
                    {
                        "role": "assistant",
                        "content": json.dumps({"judgement": label_values}),
                    },
                ],
                "response": response,
                "criteria": annotated,
                "rule_ids": rule_ids,
                "judgement": label_values,
                "label_source": "external_vlm_teacher",
                "source": str(path.resolve()),
                "source_line": line_idx,
            }


def summarize(paths: list[Path], output: Path) -> None:
    stats: dict[str, Any] = {"files": {}, "total": 0, "by_split": Counter(), "by_task": Counter(), "by_rule": Counter(), "labels": Counter()}
    groups: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                count += 1
                stats["total"] += 1
                stats["by_split"][row.get("split", "unknown")] += 1
                stats["by_task"][row.get("task", "unknown")] += 1
                groups[row.get("group_id", "")].add(row.get("split", "unknown"))
                for rid in row.get("rule_ids", []):
                    stats["by_rule"][rid] += 1
                for label in row.get("judgement", []):
                    stats["labels"][str(bool(label)).lower()] += 1
        stats["files"][str(path)] = count
    leaking = [g for g, splits in groups.items() if len(splits) > 1]
    stats["group_leakage_count"] = len(leaking)
    serializable = {
        **stats,
        "by_split": dict(stats["by_split"]),
        "by_task": dict(stats["by_task"]),
        "by_rule": dict(stats["by_rule"]),
        "labels": dict(stats["labels"]),
    }
    output.write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract")
    extract.add_argument("--task", choices=("sokoban", "frozenlake"), required=True)
    extract.add_argument("--experiment-dir", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--max-trajectories", type=int)
    extract.add_argument("--seed", type=int, default=20260828)

    convert = sub.add_parser("convert-judged")
    convert.add_argument("--input", type=Path, nargs="+", required=True)
    convert.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "extract":
        rows = candidate_rows(args.task, args.experiment_dir, args.max_trajectories, args.seed)
        count = write_jsonl(args.output, rows)
        print(json.dumps({"output": str(args.output), "records": count}))
        return

    all_rows = []
    for input_path in args.input:
        all_rows.extend(judged_rows(input_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for split in ("train", "validation", "test"):
        path = args.output_dir / f"{split}.jsonl"
        write_jsonl(path, (row for row in all_rows if row["split"] == split))
        output_paths.append(path)
    summarize(output_paths, args.output_dir / "summary.json")
    print((args.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    main()
