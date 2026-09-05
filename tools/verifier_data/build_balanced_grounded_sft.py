#!/usr/bin/env python3
"""Create balanced rubric-level verifier SFT pairs from grounded rollout boards."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("grounded", HERE / "build_grounded_rubric_dataset.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)


CRITERIA = {
    "FORMAT_COMPLIANCE": "[+] FORMAT_COMPLIANCE: The response contains exactly one complete <observation>, <think>, <answer>, and <prediction> section.",
    "ACTION_SYNTAX_FROZEN": "[+] ACTION_SYNTAX: <answer> contains 1 to 5 comma-separated actions and every action is one of Left, Down, Right, Up.",
    "ACTION_SYNTAX_SOKOBAN": "[+] ACTION_SYNTAX: <answer> contains 1 to 3 comma-separated actions and every action is one of Left, Down, Right, Up.",
    "SPATIAL_FROZEN": "[+] FROZENLAKE_SPATIAL_GROUNDING: Every stated goal/hole position in <observation> agrees with the rendered FrozenLake board; a goal relation is explicitly stated.",
    "SPATIAL_SOKOBAN": "[+] SOKOBAN_SPATIAL_GROUNDING: The stated box and target positions in <observation> agree with the rendered Sokoban board.",
}


def phrase(obj, player, pos):
    dr, dc = g.sign(pos[0] - player[0]), g.sign(pos[1] - player[1])
    v = {-1: "above", 0: "in the same row as", 1: "below"}[dr]
    h = {-1: "left", 0: "in the same column as", 1: "right"}[dc]
    return f"The {obj} is {v} and {h} of the player"


def canonical_observation(state, task, wrong=False):
    player = state["player"]
    if task == "frozenlake":
        items = [("goal", state["goals"][0])]
        if state["holes"]: items.append(("hole", state["holes"][0]))
    else:
        items = [("box", state["boxes"][0]), ("target", state["targets"][0])]
    parts = [phrase(name, player, pos) for name, pos in items]
    if wrong:
        # Corrupt the unique goal (FrozenLake) or first box (Sokoban), never a
        # non-unique hole that could accidentally match a different hole.
        if "above" in parts[0]: parts[0] = parts[0].replace("above", "below", 1)
        elif "below" in parts[0]: parts[0] = parts[0].replace("below", "above", 1)
        elif "same row" in parts[0]: parts[0] = parts[0].replace("in the same row as", "below", 1)
        elif "left" in parts[0]: parts[0] = parts[0].replace("left", "right", 1)
        else: parts[0] = parts[0].replace("right", "left", 1)
    return "; ".join(parts) + "."


def normalized_response(raw, observation, answer=None):
    vals = {k: (g.TAG_RE[k].search(raw).group(1) if g.TAG_RE[k].search(raw) else "") for k in g.TAG_RE}
    vals["observation"] = observation
    vals["think"] = vals["think"] or "I will choose the next move from the visible board."
    vals["answer"] = answer or vals["answer"] or "Left"
    vals["prediction"] = vals["prediction"] or "The player position will change."
    return "\n".join(f"<{k}>{vals[k]}</{k}>" for k in ("observation", "think", "answer", "prediction"))


def record(raw, group, split, response, rule, criterion, label, suffix):
    labels = [bool(label)]; criteria = [criterion]
    return {
        "id": g.sha(raw["id"] + suffix)[:24], "task": raw["task"], "split": split,
        "group_id": group, "trajectory_id": raw["trajectory_id"], "turn_index": raw["turn_index"],
        "images": [raw["image"]], "response": response, "criteria": criteria,
        "rule_ids": [rule], "judgement": labels,
        "label_source": "deterministic_raster_counterfactual_v1",
        "counterfactual_type": suffix.strip(":"),
        "messages": [
            {"role": "user", "content": "<image>\n" + g.prompt(raw["task"], response, criteria)},
            {"role": "assistant", "content": json.dumps({"judgement": labels})},
        ], "source": raw["source"], "source_candidate_id": raw["id"],
    }


def variants(raw):
    task = raw["task"]; state = g.parse_board(raw["image"], task)
    required = state["goals"] if task == "frozenlake" else state["boxes"] and state["targets"]
    if state["player"] is None or not required: return []
    group = f"{task}:{g.map_id(raw['trajectory_images'][0], task)}"; split = g.split_for_group(group)
    good_obs = canonical_observation(state, task, False); bad_obs = canonical_observation(state, task, True)
    good = normalized_response(raw["response"], good_obs, "Left")
    bad_space = normalized_response(raw["response"], bad_obs, "Left")
    bad_action = normalized_response(raw["response"], good_obs, "Jump")
    bad_format = good.replace("</prediction>", "", 1)
    spatial = CRITERIA["SPATIAL_FROZEN" if task == "frozenlake" else "SPATIAL_SOKOBAN"]
    action = CRITERIA["ACTION_SYNTAX_FROZEN" if task == "frozenlake" else "ACTION_SYNTAX_SOKOBAN"]
    return [
        record(raw, group, split, good, "SPATIAL_GROUNDING", spatial, True, ":spatial-positive"),
        record(raw, group, split, bad_space, "SPATIAL_GROUNDING", spatial, False, ":spatial-negative"),
        record(raw, group, split, good, "ACTION_SYNTAX", action, True, ":action-positive"),
        record(raw, group, split, bad_action, "ACTION_SYNTAX", action, False, ":action-negative"),
        record(raw, group, split, good, "FORMAT_COMPLIANCE", CRITERIA["FORMAT_COMPLIANCE"], True, ":format-positive"),
        record(raw, group, split, bad_format, "FORMAT_COMPLIANCE", CRITERIA["FORMAT_COMPLIANCE"], False, ":format-negative"),
    ]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,nargs="+",required=True)
    ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--boards-per-task",type=int,default=3000)
    a=ap.parse_args(); selected=Counter(); rows=[]
    for p in a.input:
        for line in p.open():
            raw=json.loads(line); task=raw["task"]
            if selected[task]>=a.boards_per_task: continue
            try: out=variants(raw)
            except (OSError,ValueError,IndexError): out=[]
            if out: rows.extend(out); selected[task]+=1
    a.output_dir.mkdir(parents=True,exist_ok=True); handles={s:(a.output_dir/f"{s}.jsonl").open("w") for s in ("train","validation","test")}
    stats={"records":len(rows),"source_boards":dict(selected),"by_task":Counter(),"by_split":Counter(),"labels_by_rule":defaultdict(Counter)}; gs=defaultdict(set)
    try:
        for x in rows:
            handles[x["split"]].write(json.dumps(x,ensure_ascii=False)+"\n"); stats["by_task"][x["task"]]+=1; stats["by_split"][x["split"]]+=1; gs[x["split"]].add(x["group_id"])
            stats["labels_by_rule"][x["rule_ids"][0]][str(x["judgement"][0]).lower()]+=1
    finally:
        for h in handles.values(): h.close()
    sets=list(gs.values()); leak=sum(len(sets[i]&sets[j]) for i in range(3) for j in range(i+1,3))
    summary={**stats,"by_task":dict(stats["by_task"]),"by_split":dict(stats["by_split"]),"labels_by_rule":{k:dict(v) for k,v in stats["labels_by_rule"].items()},"map_group_leakage_count":leak}
    (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
