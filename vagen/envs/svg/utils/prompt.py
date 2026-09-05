def system_prompt(**kwargs):
    format_type = kwargs.get("format", "default")

    base_prompt = """You are a precise SVG code generator.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Identify colors, dimensions, positions, and relationships between elements
3. Generate accurate SVG code that reproduces the image, you can use path for better shape

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +10.0"""

    if format_type in FORMAT_CONFIGS:
        example = FORMAT_CONFIGS[format_type].get("example", "")
        if example:
            return base_prompt + "\n" + "Example:\n" + example

    return base_prompt


def init_observation_template(**kwargs):
    observation = kwargs.get("observation", None)
    return f"""[Initial Observation]:
{observation}
Please carefully observe the image, and generate SVG code that reproduces it as accurately as possible.
Decide on your SVG code.
"""


def action_template(**kwargs):
    valid_action = kwargs.get("valid_action", None)
    observation = kwargs.get("observation", None)
    reward = kwargs.get("reward", None)
    done = kwargs.get("done", None)

    return f"""After your answer, the extracted valid SVG code is {valid_action}.
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Please revise your code to make it more precise and similar to the original image.
Decide on your revised SVG code.
"""


FORMAT_CONFIGS = {
    "free_think": {
        "format": "...think...<answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": """...think...I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
    "no_think": {
        "format": "<answer>...</answer>",
        "description": "You should provide only your answer.",
        "example": """<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
    "grounding": {
        "format": "...<observation>...</observation><reasoning>...</reasoning>...<answer>...</answer>",
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.",
        "additional_info": "The observation should be described in detail about what you see in the image.",
        "example": """<observation>I can see a red circle positioned at the top-left corner of the canvas, and a blue rectangle at the bottom-right. The circle has a radius of approximately 15 units and is centered at coordinates (25, 25). The rectangle is approximately 30 units wide by 20 units tall and positioned at coordinates (60, 60).</observation><reasoning>I need to create an SVG with a viewBox of 0 0 100 100 to properly position these elements. I'll add a circle element with the observed properties and a rectangle element with the observed properties.</reasoning>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
    "worldmodeling": {
        "format": "...<reasoning>...</reasoning><prediction>...</prediction>...<answer>...</answer>",
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.",
        "additional_info": "The prediction should describe what you expect to see after your actions are executed.",
        "example": """<reasoning>The image shows a red circle at the top-left and a blue rectangle at the bottom-right. I need to create an SVG that accurately reproduces these elements with their correct positions and dimensions.</reasoning><prediction>After implementing this SVG code, the result should closely match the original image. I expect a similarity score of at least 0.95, as the shapes and positions are relatively simple to reproduce.</prediction>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
    "grounding_worldmodeling": {
        "format": "...<observation>...</observation><reasoning>...</reasoning><prediction>...</prediction>...<answer>...</answer>",
        "description": "You should first give your thought process with your observation and reasoning, then predict next state, and finally the answer.",
        "additional_info": "Both the observation and prediction should describe what you see or expect to see in the environment.",
        "example": """<observation>I can see an image containing a red circle positioned at the top-left area of the canvas, approximately at coordinates (25, 25) with a radius of 15 units. There is also a blue rectangle at the bottom-right area, sized about 30x20 units and positioned at coordinates (60, 60).</observation><reasoning>Based on my observation, I need to create an SVG that precisely matches these elements. The circle appears to be slightly too far right, so I should adjust its x-coordinate to 20 instead of 25. The rectangle could benefit from being slightly wider.</reasoning><prediction>After implementing these adjustments, the resulting SVG should more closely match the original image. I expect the similarity score to improve to approximately 0.98.</prediction>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="35" height="20" fill="blue" />
</svg></answer>""",
    },
    "default": {
        "format": "...think...<answer>...</answer>",
        "description": "You should first give your thought process, and then your answer.",
        "example": """...think...I can see the image contains a red circle and a blue rectangle. The circle is positioned at the top-left, while the rectangle is at the bottom-right.
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
    "wm": {
        "format": "...<observation>...</observation><reasoning>...</reasoning><prediction>...</prediction>...<answer>...</answer>",
        "description": "You should first give your thought process with your observation and reasoning, then predict next state, and finally the answer.",
        "additional_info": "Both the observation and prediction should describe what you see or expect to see in the environment.",
        "example": """<observation>I can see an image containing a red circle and a blue rectangle.</observation><reasoning>I need to create SVG code reproducing these shapes.</reasoning><prediction>The generated SVG should closely match the original.</prediction>
<answer><svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="25" cy="25" r="15" fill="red" />
  <rect x="60" y="60" width="30" height="20" fill="blue" />
</svg></answer>""",
    },
}


def format_prompt(max_actions_per_step, action_sep, add_example=True, prompt_format="free_think"):
    """Generate format prompt based on the specified format."""
    config = FORMAT_CONFIGS.get(prompt_format, FORMAT_CONFIGS.get("default", FORMAT_CONFIGS["free_think"]))

    base = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by {action_sep}.
{config["description"]}"""

    if "additional_info" in config:
        base += f"\n{config['additional_info']}"

    base += f"""
Your response should be in the format of:
{config["format"]}"""

    if add_example:
        return base + "\n" + f"e.g. {config['example']}"

    return base
