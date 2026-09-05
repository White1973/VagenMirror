#!/usr/bin/env python3
"""Build task-aligned, raster-grounded verifier SFT data from rollout candidates.

The labels in this file are deliberately limited to facts recoverable from the
current/next rendered board and response syntax.  No trajectory-success label
is repurposed as a rubric judgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image


TAG_RE = {name: re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.I | re.S)
          for name in ("observation", "think", "answer", "prediction")}
VERT = {"above": -1, "below": 1, "same row": 0}
HORIZ = {"left": -1, "right": 1, "same column": 0}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def tile_counts(im: Image.Image, size: int, r: int, c: int) -> Counter:
    return Counter(im.crop((c * size, r * size, (c + 1) * size, (r + 1) * size)).getdata())


@lru_cache(maxsize=50000)
def parse_board(path: str, task: str) -> dict:
    im = Image.open(path).convert("RGB")
    size = 64 if task == "frozenlake" else 16
    rows, cols = im.height // size, im.width // size
    state = {"player": None, "goals": [], "holes": [], "boxes": [], "targets": [], "walls": []}
    for r in range(rows):
        for c in range(cols):
            px = tile_counts(im, size, r, c)
            if task == "frozenlake":
                if px[(61, 202, 242)] > 100 or px[(86, 227, 247)] > 100:
                    state["holes"].append((r, c))
                if px[(207, 117, 43)] > 40 or px[(171, 81, 48)] > 40:
                    state["goals"].append((r, c))
                if px[(230, 69, 57)] > 20 or px[(99, 171, 63)] > 30:
                    state["player"] = (r, c)
            else:
                if px[(176, 61, 0)] > 100:
                    state["walls"].append((r, c))
                if px[(41, 202, 26)] > 30:
                    state["player"] = (r, c)
                if sum(n for (red, green, blue), n in px.items()
                       if red > 210 and green < 60 and blue < 60) > 20:
                    state["boxes"].append((r, c))
                if px[(215, 196, 0)] > 30 or px[(215, 114, 0)] > 50:
                    state["targets"].append((r, c))
    state["shape"] = (rows, cols)
    # Some FrozenLake player sprites contain enough brown pixels to trigger the
    # chest/goal palette.  When a second visible goal exists, the player tile is
    # the sprite false-positive rather than another goal.
    if task == "frozenlake" and state["player"] in state["goals"] and len(state["goals"]) > 1:
        state["goals"].remove(state["player"])
    return state


@lru_cache(maxsize=20000)
def map_id(first_image: str, task: str) -> str:
    s = parse_board(first_image, task)
    static = {"task": task, "shape": s["shape"], "walls": s["walls"]}
    if task == "frozenlake":
        static.update(goals=s["goals"], holes=s["holes"])
    else:
        # Initial rollout frames expose targets.  Include boxes too because
        # their initial placement is part of a Sokoban level definition.
        static.update(targets=s["targets"], boxes=s["boxes"])
    return sha(json.dumps(static, sort_keys=True))[:24]


def split_for_group(group: str) -> str:
    bucket = int(sha(group)[:8], 16) % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def relation(desc: str) -> tuple[int | None, int | None] | None:
    low = desc.lower()
    vr = next((v for k, v in VERT.items() if k in low), None)
    hr = next((v for k, v in HORIZ.items() if k in low), None)
    return None if vr is None and hr is None else (vr, hr)


def claimed_relation(text: str, aliases: list[str]) -> tuple[int | None, int | None] | None:
    low = text.lower()
    names = "|".join(re.escape(x) for x in aliases)
    patterns = [
        rf"(?:{names})\s+(?:is|are|will be)\s+(.*?)(?:of|relative to)\s+(?:the\s+)?player",
        rf"both\s+the\s+[^.]*?(?:{names})[^.]*?\s+are\s+(.*?)(?:of|relative to)\s+(?:the\s+)?player",
    ]
    for pat in patterns:
        m = re.search(pat, low)
        if m and relation(m.group(1)) is not None:
            return relation(m.group(1))
    return None


def matches(rel: tuple[int | None, int | None], player: tuple[int, int], obj: tuple[int, int]) -> bool:
    vr, hr = rel
    return (vr is None or vr == sign(obj[0] - player[0])) and (hr is None or hr == sign(obj[1] - player[1]))


def grounded_text(text: str, state: dict, task: str) -> bool:
    player = state.get("player")
    if player is None:
        return False
    if task == "frozenlake":
        specs = [(["goal", "g"], state["goals"], True), (["hole", "o"], state["holes"], False)]
    else:
        specs = [(["box", "crate", "x"], state["boxes"], True), (["target", "goal", "o"], state["targets"], True)]
    checked = 0
    for aliases, positions, required in specs:
        rel = claimed_relation(text, aliases)
        mentioned = any(re.search(rf"\b{re.escape(a)}\b", text.lower()) for a in aliases)
        if rel is None:
            if required or mentioned:
                return False
            continue
        checked += 1
        if not positions or not any(matches(rel, player, pos) for pos in positions):
            return False
    return checked > 0


def prompt(task: str, response: str, criteria: list[str]) -> str:
    return (
        "Evaluate each criterion as strict True/False against the response reasoning.\n"
        "[+] criteria: True = quality IS present. [-] criteria: True = flaw IS present.\n"
        "Use the provided image to verify factual claims about object positions, actions, and spatial relationships.\n"
        "Return JSON only: {\"judgement\": [bool, ...]} with the same order.\n"
        f"Task: {task}\n\nResponse:\n{response}\n\nCriteria:\n"
        + json.dumps(criteria, ensure_ascii=False)
    )


def label_row(row: dict) -> dict | None:
    task, response = row["task"], row["response"]
    tags = {k: TAG_RE[k].search(response) for k in TAG_RE}
    current = parse_board(row["image"], task)
    if current["player"] is None:
        return None
    complete = all(tags.values()) and sum(len(TAG_RE[k].findall(response)) for k in TAG_RE) == 4
    answer = tags["answer"].group(1) if tags["answer"] else ""
    actions = [x.strip().lower() for x in answer.split(",") if x.strip()]
    limit = 5 if task == "frozenlake" else 3
    action_ok = bool(actions) and len(actions) <= limit and all(x in {"left", "right", "up", "down"} for x in actions)
    obs_ok = bool(tags["observation"]) and grounded_text(tags["observation"].group(1), current, task)

    criteria = [
        "[+] FORMAT_COMPLIANCE: The response contains exactly one complete <observation>, <think>, <answer>, and <prediction> section.",
        f"[+] ACTION_SYNTAX: <answer> contains 1 to {limit} comma-separated actions and every action is one of Left, Down, Right, Up.",
        ("[+] FROZENLAKE_SPATIAL_GROUNDING: Every stated goal/hole position in <observation> agrees with the rendered FrozenLake board; a goal relation is explicitly stated."
         if task == "frozenlake" else
         "[+] SOKOBAN_SPATIAL_GROUNDING: The stated box and target positions in <observation> agree with the rendered Sokoban board."),
    ]
    labels = [bool(complete), bool(action_ok), bool(obs_ok)]
    first = row["trajectory_images"][0]
    group = f"{task}:{map_id(first, task)}"
    split = split_for_group(group)
    rid = ["FORMAT_COMPLIANCE", "ACTION_SYNTAX", "SPATIAL_GROUNDING"]
    return {
        "id": sha(row["id"] + ":grounded-v1")[:24], "task": task, "split": split,
        "group_id": group, "trajectory_id": row["trajectory_id"], "turn_index": row["turn_index"],
        "images": [row["image"]], "response": response, "criteria": criteria,
        "rule_ids": rid, "judgement": labels, "label_source": "deterministic_raster_grounded_v1",
        "messages": [
            {"role": "user", "content": "<image>\n" + prompt(task, response, criteria)},
            {"role": "assistant", "content": json.dumps({"judgement": labels})},
        ],
        "source": row["source"], "source_candidate_id": row["id"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-per-task", type=int, default=10000)
    args = ap.parse_args()
    rows, per_task = [], Counter()
    for path in args.input:
        with path.open() as f:
            for line in f:
                raw = json.loads(line)
                if per_task[raw["task"]] >= args.max_per_task:
                    continue
                try:
                    out = label_row(raw)
                except (OSError, ValueError):
                    out = None
                if out is not None:
                    rows.append(out); per_task[out["task"]] += 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": len(rows), "by_task": dict(per_task), "by_split": Counter(),
             "labels_by_rule": defaultdict(Counter), "groups_by_split": defaultdict(set)}
    handles = {s: (args.output_dir / f"{s}.jsonl").open("w") for s in ("train", "validation", "test")}
    try:
        for row in rows:
            handles[row["split"]].write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["by_split"][row["split"]] += 1
            stats["groups_by_split"][row["split"]].add(row["group_id"])
            for rule, value in zip(row["rule_ids"], row["judgement"]):
                stats["labels_by_rule"][rule][str(value).lower()] += 1
    finally:
        for h in handles.values(): h.close()
    groups = list(stats.pop("groups_by_split").values())
    leakage = sum(len(groups[i] & groups[j]) for i in range(3) for j in range(i + 1, 3))
    summary = {**stats, "by_split": dict(stats["by_split"]),
               "labels_by_rule": {k: dict(v) for k, v in stats["labels_by_rule"].items()},
               "map_group_leakage_count": leakage}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
