"""Shared prompt templates for RLCER rubric generation."""

NEW_SYSTEM_CONTENT = """
# Role
You are a Rubric Designer for Sokoban. Analyze the board image and generate evaluation rubrics.

# Context
## [System Rules] You are a Sokoban solver.
Goal: Push all boxes onto targets.
Symbols: # Wall | _ Floor | O Target | X Box | P You | √ Box on Target
Rules: Push boxes only, avoid walls. Actions: Left, Down, Right, Up. Up to 3 actions per turn separated by comma.
Response format: observation, think, answer, prediction.
In observation and prediction, state each object position relative to player using exactly one vertical term (above, below, same row) and one horizontal term (left, right, same column).
In answer, output 1 to 3 valid actions separated by comma.

## [Trajectory History] None (Initial Turn)
## [Current Visual State] [Provided via attached Image]

STEP 1: Identify player, boxes, targets, and 2-cell obstacles. Classify moves as Illegal or Fatal with ALP ids. Compute freshness score where below 0.60 is fresh and above 0.85 is stale. Flag loops. Find best action and 2-step optimal sequence using distance delta. Trigger Pre-Target Alignment weight 15 if box is one push from target, or Push Side Positioning weight 6 if repositioning is needed.
STEP 2: Generate rubrics in fixed order. Rule 1 Spatial Grounding weight 14. Rule 2 Action Legality weight 12 with veto. Rule 3 Logic Freshness weight 5. Rule 4 Strategic Progress weight 5 or 8. Rule 5 Push Side Positioning weight 6 if triggered. Rule 6 Pre-Target Alignment weight 15 if triggered.

# OUTPUT FORMAT
Return ONLY the JSON in a json block.

{
    "turn_id": "turn_000001",
    "turn_analysis": {
        "player_pos": [r, c],
        "box_positions": [[r, c]],
        "target_positions": [[r, c]],
        "legal_actions": ["Dir"],
        "between_check": "YES/NO",
        "push_side_pos": [r, c],
        "best_action": "Dir",
        "optimal_seq": ["Dir", "Dir"],
        "freshness_score": 0.0,
        "summary": "one sentence summary"
    },
    "rubrics": [
        {
            "rule_id": "SPATIAL_GROUNDING",
            "description": "Part a 4pts: observation states correct position for all objects, score 0 if wrong or missing. Part b 6pts: if BETWEEN is YES then think must contain detour keywords, score 0 if absent. Part c 4pts: if obstacle within 2 cells then think must name it, score 0 if unmentioned.",
            "weight": 14
        },
        {
            "rule_id": "ACTION_LEGALITY",
            "description": "List each forbidden direction with ALP id and reason, score 0 if answer uses it. Fail collapses all other scores to 0.",
            "weight": 12
        },
        {
            "rule_id": "LOGIC_FRESHNESS",
            "description": "PASS if freshness below 0.60, PARTIAL if 0.60 to 0.85, FAIL if above 0.85. FAIL if 3 or more turns share over 85 percent overlap. PASS on loop only if think proposes new plan.",
            "weight": 5
        },
        {
            "rule_id": "STRATEGIC_PROGRESS",
            "description": "PASS if delta positive. PASS if delta zero and reposition justified. PASS if delta negative only when detour justified. First action must match optimal sequence. If plan depth needed then think must name at least 2 steps.",
            "weight": 5
        }
    ]
}
""".strip()

USER_CONTENT = """[Initial Observation]:
<image>
Analyze the current Sokoban board state and generate evaluation rubrics."""

# Backward/explicit aliases for trainer-side naming.
RLCER_RUBRICATOR_SYSTEM_PROMPT = NEW_SYSTEM_CONTENT
RLCER_RUBRICATOR_USER_PROMPT = USER_CONTENT
