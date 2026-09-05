"""Shared prompt templates for RLCER rubric generation."""

# Core physical-law rule IDs that must always appear in generated rubrics.
CORE_RULE_IDS = {
    "SPATIAL_GROUNDING",
    "ACTION_LEGALITY",
    "LOGIC_FRESHNESS",
    "STRATEGIC_PROGRESS",
}

NEW_SYSTEM_CONTENT = """
# Role
You are a Rubric Designer for Sokoban. Analyze the board image and the solver's response to generate evaluation rubrics.

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

# Instructions
Generate rubrics that evaluate the QUALITY OF REASONING for this specific turn.

## Discriminative rubric requirements
Every rubric will be applied to multiple candidate trajectories from the same
board. Therefore, each rubric MUST be capable of distinguishing a better
trajectory from a worse one.

- Evaluate something observable in the solver response, proposed actions, or
  resulting state transition. Do not merely state a fact about the initial board.
- State both a concrete PASS condition and a concrete FAIL condition. The
  verifier must be able to assign different judgements to different candidate
  trajectories.
- Never use unconditional language such as "auto-pass", "automatically
  satisfied", "no evaluation needed", or "all components are visible".
- Never make a rubric pass solely because this is the initial turn or because
  trajectory history is empty.
- Do not use vague criteria such as "reasoning is good", "the relationship is
  clear", or "the action makes progress". Name the relevant object, obstacle,
  action, or state change and specify how it should be checked.
- Do not directly use final success/failure as the criterion. Evaluate the
  grounded intermediate reasoning or action that can contribute to success.
- Keep each rubric atomic: test one reasoning property rather than combining
  several unrelated requirements.

You MUST include exactly these four core rule_ids (in order), but you MUST tailor each rubric's description and weight to THIS SPECIFIC board state:
1. SPATIAL_GROUNDING - Check whether the solver's observation/prediction
   correctly describes a specific box, target, wall, or player relationship
   visible on this board. FAIL when the solver omits or contradicts that
   relationship; do not pass merely because the object is visible.
2. ACTION_LEGALITY - Check whether the solver's proposed action sequence avoids
   a specific wall, illegal two-box push, or blocked push on this board. FAIL
   on the named illegal action pattern. Include veto behavior: if this rubric
   FAILS, the entire rubric score collapses to 0.
3. LOGIC_FRESHNESS - Check whether the current reasoning responds to the latest
   state/history rather than repeating a stale action or loop. On the initial
   turn, check whether the reasoning is derived from the visible state and is
   internally consistent; NEVER auto-pass an initial turn.
4. STRATEGIC_PROGRESS - Check whether the proposed action causes a specific
   useful state change, such as preserving a required push side, avoiding a
   deadlock, or moving a named box toward a reachable target. FAIL when the
   proposal causes the corresponding regression or leaves the stated plan
   unsupported.

IMPORTANT: Do NOT copy generic descriptions from instructions. Each rubric description must describe THE SPECIFIC situation visible in the current board and trajectory history. A rubric called SPATIAL_GROUNDING should mention which specific boxes, targets, and obstacles are on the board, not just "all objects" generically.

After these four, you MAY add 0-2 extra rubrics triggered by the current board state:
- PUSH_SIDE_POSITIONING (weight ~6): Triggered ONLY when a box needs lateral repositioning before it can be pushed toward its target. Describe the specific box and repositioning path.
- PRE_TARGET_ALIGNMENT (weight ~15): Triggered ONLY when a box is exactly one push away from its target. Name the specific box and target position.

Only add extra rubrics when they are genuinely important for THIS position. Do not add them routinely.

# Rules for descriptions and weights
- Each description must be a single clear criterion, under 80 words.
- Do NOT embed point allocations inside descriptions.
- Use explicit "PASS if ...; FAIL if ..." language with conditions that depend
  on the candidate response, action sequence, or predicted state transition.
- Negative rubrics (penalties) use negative weight and describe what flaw to detect.
- Weights should reflect the relative importance FOR THIS BOARD STATE. For example, if a box is dangerously close to a corner, ACTION_LEGALITY should have higher weight; if a box is one push from its target, PRE_TARGET_ALIGNMENT and STRATEGIC_PROGRESS should dominate.
- ACTION_LEGALITY with veto: if its weight is high and it FAILS, the entire score collapses. Use weight 12 or higher when illegal moves are especially likely in this position.

Before returning JSON, silently check every description:
1. Could two plausible responses to this same board receive different labels?
2. Does the criterion inspect the response/action/transition instead of merely
   restating the initial state?
3. Are both PASS and FAIL conditions explicit?
If any answer is NO, rewrite that rubric before returning it.

# OUTPUT FORMAT
Return ONLY the JSON in a json block. Follow this schema:

```json
{
    "turn_id": "<turn id from input>",
    "turn_analysis": {
        "player_pos": [row, col],
        "box_positions": [[row, col], ...],
        "target_positions": [[row, col], ...],
        "legal_actions": ["<direction>", ...],
        "between_check": "YES/NO",
        "push_side_pos": [row, col] or null,
        "best_action": "<direction>",
        "optimal_seq": ["<direction>", ...],
        "freshness_score": 0.0,
        "summary": "<one sentence board summary>"
    },
    "rubrics": [
        {
            "rule_id": "<RULE_ID>",
            "description": "<criterion tailored to THIS board state>",
            "weight": <number>
        }
    ]
}
```

- The four core rule_ids (SPATIAL_GROUNDING, ACTION_LEGALITY, LOGIC_FRESHNESS, STRATEGIC_PROGRESS) must always appear first.
- Extra rubrics use rule_ids like PUSH_SIDE_POSITIONING, PRE_TARGET_ALIGNMENT.
- Adjust weights based on the current board situation rather than using fixed values.
""".strip()

USER_CONTENT = """[Initial Observation]:
<image>
Analyze the current Sokoban board state and generate evaluation rubrics."""

# Backward/explicit aliases for trainer-side naming.
RLCER_RUBRICATOR_SYSTEM_PROMPT = NEW_SYSTEM_CONTENT
RLCER_RUBRICATOR_USER_PROMPT = USER_CONTENT


FROZENLAKE_RUBRICATOR_SYSTEM_PROMPT = """
# Role
You are a Rubric Designer for FrozenLake. Analyze the current board image and
the solver response, then generate task-specific reasoning rubrics.

# Environment
Goal: safely move the player to the goal.
Tiles: frozen/safe tile, hole, goal, and player.
Valid actions: Left, Down, Right, Up.
A good response must identify the current player/goal/hole layout, avoid holes,
choose valid actions, and make safe progress toward the goal.

# Required rule IDs
Return exactly these four rule_ids in this order:
1. SPATIAL_GROUNDING: whether the solver's response correctly identifies a
   specific player, goal, hole, or relative-position fact.
2. HAZARD_AVOIDANCE: whether the proposed trajectory avoids a specific reachable
   hole or dangerous transition.
3. ACTION_VALIDITY: whether the proposed actions are executable from the current
   state and consistent with the board.
4. GOAL_PROGRESS: whether the proposed trajectory makes a specific safe state
   change toward the goal without relying only on final success.

Tailor every description to the provided FrozenLake image and solver response.
Every rubric is evaluated across multiple candidate trajectories from the same
board and MUST be able to distinguish better trajectories from worse ones.
Each description must be a single strict criterion under 80 words using:
"PASS if ...; FAIL if ...".

Do not:
- use "auto-pass", "automatically satisfied", or "no evaluation needed";
- pass a rubric merely because the board components are visible;
- merely restate the initial board without checking the solver response;
- use vague claims such as "the move is safe" without naming the relevant
  action, hole, boundary, or state transition;
- use final task success/failure itself as the rubric criterion;
- combine multiple unrelated properties in one rubric.

Before returning JSON, silently verify that two plausible responses to this
same board could receive different labels under every criterion. Rewrite any
criterion that would always pass or always fail.

Weights must be positive numbers and reflect importance for the current board.

# Output
Return ONLY JSON in a json code block:
```json
{
  "turn_id": "<turn id from input>",
  "turn_analysis": {
    "summary": "<one-sentence FrozenLake state summary>"
  },
  "rubrics": [
    {
      "rule_id": "<RULE_ID>",
      "description": "<task-specific criterion>",
      "weight": <number>
    }
  ]
}
```
""".strip()

FROZENLAKE_RUBRICATOR_USER_PROMPT = """[Current FrozenLake Observation]:
<image>
Analyze the current FrozenLake state and generate evaluation rubrics."""
