import re
from typing import Dict, List


def parse_free_think(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse free_think format: ...think...<answer>...</answer>"""
    pattern = r"(.*?)<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None
    think_content = ""
    action_content = ""
    actions: List[str] = []

    if match:
        think_content = match.group(1).strip()
        action_content = match.group(2).strip()
        actions = [action_content] if action_content else []
        if len(actions) > max_actions:
            actions = actions[:max_actions]

    llm_response = f"{think_content}<answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": think_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_no_think(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse no_think format: <answer>...</answer>"""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None
    action_content = ""
    actions: List[str] = []

    if match:
        action_content = match.group(1).strip()
        actions = [action_content] if action_content else []

    llm_response = f"<answer>{action_content}</answer>"

    return {
        "llm_raw_response": response,
        "llm_response": llm_response,
        "think_content": "",
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_grounding(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse grounding format:
    <observation>...</observation><reasoning>...</reasoning>...<answer>...</answer>
    """
    pattern = (
        r"<observation>(.*?)</observation>\s*"
        r"<reasoning>(.*?)</reasoning>\s*"
        r".*?<answer>(.*?)</answer>"
    )
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None
    observation_content = ""
    reasoning_content = ""
    action_content = ""
    actions: List[str] = []

    if match:
        observation_content = match.group(1).strip()
        reasoning_content = match.group(2).strip()
        action_content = match.group(3).strip()
        actions = [action_content] if action_content else []

    return {
        "llm_raw_response": response,
        "observation_content": observation_content,
        "think_content": reasoning_content,
        "reasoning_content": reasoning_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_worldmodeling(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse worldmodeling format:
    <reasoning>...</reasoning><prediction>...</prediction>...<answer>...</answer>
    """
    pattern = (
        r"<reasoning>(.*?)</reasoning>\s*"
        r"<prediction>(.*?)</prediction>\s*"
        r".*?<answer>(.*?)</answer>"
    )
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None
    reasoning_content = ""
    prediction_content = ""
    action_content = ""
    actions: List[str] = []

    if match:
        reasoning_content = match.group(1).strip()
        prediction_content = match.group(2).strip()
        action_content = match.group(3).strip()
        actions = [action_content] if action_content else []

    return {
        "llm_raw_response": response,
        "think_content": reasoning_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_grounding_worldmodeling(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse grounding_worldmodeling format:
    <observation>...</observation><reasoning>...</reasoning><prediction>...</prediction>...<answer>...</answer>
    """
    pattern = (
        r"<observation>(.*?)</observation>\s*"
        r"<reasoning>(.*?)</reasoning>\s*"
        r"<prediction>(.*?)</prediction>\s*"
        r".*?<answer>(.*?)</answer>"
    )
    match = re.search(pattern, response, re.DOTALL)

    format_correct = match is not None
    observation_content = ""
    reasoning_content = ""
    prediction_content = ""
    action_content = ""
    actions: List[str] = []

    if match:
        observation_content = match.group(1).strip()
        reasoning_content = match.group(2).strip()
        prediction_content = match.group(3).strip()
        action_content = match.group(4).strip()
        actions = [action_content] if action_content else []

    return {
        "llm_raw_response": response,
        "observation_content": observation_content,
        "think_content": reasoning_content,
        "reasoning_content": reasoning_content,
        "prediction_content": prediction_content,
        "action_content": action_content,
        "actions": actions,
        "format_correct": format_correct,
    }


def parse_wm(response: str, action_sep: str = "~~", max_actions: int = 1) -> Dict:
    """Parse wm format (alias for grounding_worldmodeling)."""
    return parse_grounding_worldmodeling(response, action_sep, max_actions)


def parse_response(
    response: str,
    prompt_format: str = "free_think",
    action_sep: str = "~~",
    max_actions: int = 1,
) -> Dict:
    """Parse LLM response based on the specified prompt format."""
    parsers = {
        "free_think": parse_free_think,
        "no_think": parse_no_think,
        "grounding": parse_grounding,
        "worldmodeling": parse_worldmodeling,
        "grounding_worldmodeling": parse_grounding_worldmodeling,
        "wm": parse_wm,
        "default": parse_free_think,
    }
    parser = parsers.get(prompt_format, parse_free_think)
    return parser(response, action_sep, max_actions)
