import asyncio
import random
import re
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from vagen.envs.gym_image_env import GymImageEnv
from vagen.envs.svg.svg_utils import is_valid_svg, load_svg_dataset, process_and_rasterize_svg
from vagen.envs.svg.score import calculate_total_score
from vagen.envs.svg.grounded_features import build_svg_verifier_context
from .utils.prompt import (
    system_prompt,
    init_observation_template,
    action_template,
    format_prompt,
)
from .utils.utils import parse_response

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

# Process-level singletons for the (heavy) metric models. Each SVG env instance
# would otherwise lazy-load its own ~1.2GB copy of DINO + DreamSim; with async
# rollout dozens of envs live in the same Ray worker process, which OOMs the GPU
# (which is already ~60% full from the sglang rollout cache). Sharing one copy
# per (model_size, device) keeps GPU memory O(1) regardless of concurrency.
# GPU forwards from concurrent envs serialize naturally on the same stream, which
# is acceptable since these are small models.
_DINO_SINGLETON: Dict[tuple, Any] = {}
_DREAMSIM_SINGLETON: Dict[tuple, Any] = {}
_MODEL_LOCK = threading.Lock()


@dataclass
class SvgEnvConfig:
    dataset_name: str = "starvector/svg-icons-simple"
    data_dir: str = "data"
    seed: int = 42
    split: str = "train"
    action_sep: str = "~~"
    render_mode: str = "vision"
    max_actions_per_step: int = 1
    image_placeholder: str = "<image>"
    use_example_in_sys_prompt: bool = True
    prompt_format: str = "free_think"
    format_reward: float = 0.5
    format_penalty: float = 0.0
    model_size: str = "small"
    dino_weight: Optional[float] = None
    structural_weight: Optional[float] = None
    dreamsim_weight: Optional[float] = None
    device: Dict[str, Any] = field(default_factory=lambda: {"dino": 0, "dreamsim": 0})
    resolution: int = 256
    # A turn is considered "successful" when its total_score (similarity-based,
    # before format_reward) reaches this fraction of the per-turn max possible
    # similarity score. SVG is a continuous quality task; without this, traj_success
    # stays False forever (no 0/1 success logic existed), making the wandb
    # aux/<env>/traj_success metric permanently 0 and useless for monitoring.
    # Default 0.5 == dreamsim similarity >= ~0.5 (the dominant component).
    success_score_threshold: float = 0.5

    def __post_init__(self):
        import torch
        processed = {}
        for key, value in self.device.items():
            if isinstance(value, (int, float)):
                target = f"cuda:{int(value)}"
                if not torch.cuda.is_available() or int(value) >= torch.cuda.device_count():
                    target = "cpu"
                processed[key] = target
            else:
                processed[key] = value
        self.device = processed

    def get_score_config(self) -> Dict:
        config = {"model_size": self.model_size, "device": self.device}
        if self.dino_weight is not None:
            config["dino_weight"] = self.dino_weight
        if self.structural_weight is not None:
            config["structural_weight"] = self.structural_weight
        if self.dreamsim_weight is not None:
            config["dreamsim_weight"] = self.dreamsim_weight
        return config


class SVG(GymImageEnv):
    """
    SVG generation environment implementing the GymImageEnv async interface.

    The agent is shown a target image and must generate SVG code that
    reproduces it. Scoring uses DINOv2, DreamSim, and structural accuracy.
    """

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        self.config = SvgEnvConfig(**env_config)

        self.dataset = load_svg_dataset(
            data_dir=self.config.data_dir,
            dataset_name=self.config.dataset_name,
            split=self.config.split,
        )

        self.total_reward: float = 0.0
        self.reward: float = 0.0
        self.valid_actions: List[str] = []
        self.current_sample = None
        self.img_id: Optional[str] = None
        self.gt_svg_code: Optional[str] = None
        self.gt_image: Optional[Image.Image] = None
        self.gen_svg_code: Optional[str] = None
        self.gen_image: Optional[Image.Image] = None

        self._dino_model = None
        self._dreamsim_model = None

        self.rng = random.Random()
        if self.config.seed is not None:
            self.rng.seed(self.config.seed)

    async def close(self) -> None:
        self._dino_model = None
        self._dreamsim_model = None

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self.rng.seed(seed)

        dataset_length = len(self.dataset)
        index = self.rng.randint(0, dataset_length - 1)
        self.current_sample = self.dataset[index]

        self.gt_svg_code = self.current_sample.get("Svg", self.current_sample.get("svg", ""))
        self.img_id = self.current_sample.get(
            "Filename", self.current_sample.get("filename", f"image_{index}")
        )

        if not self.gt_svg_code:
            raise ValueError(f"Ground truth SVG code not found in sample at index {index}")

        _, self.gt_image = await asyncio.to_thread(
            process_and_rasterize_svg, self.gt_svg_code, self.config.resolution
        )

        self.total_reward = 0.0
        self.reward = 0.0
        self.gen_svg_code = ""
        self.gen_image = None
        self.valid_actions = []

        obs = self._render_obs(init_obs=True)
        return obs, {}

    async def system_prompt(self) -> Dict[str, Any]:
        return {"obs_str": self._get_system_prompt()}

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        parsed = parse_response(
            response=action_str,
            prompt_format=self.config.prompt_format,
            action_sep=self.config.action_sep,
            max_actions=self.config.max_actions_per_step,
        )

        actions = parsed.get("actions", [])
        if not actions:
            svg_code = self._extract_svg_code(action_str)
            if svg_code and is_valid_svg(svg_code):
                actions = [svg_code]
        else:
            svg_code = self._extract_svg_code(actions[0])
            if svg_code and is_valid_svg(svg_code):
                actions = [svg_code]
            else:
                actions = []

        metrics = {
            "turn_metrics": {
                "action_is_valid": len(actions) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            },
        }

        self.reward = 0.0
        self.valid_actions = []
        done = False
        info: Dict[str, Any] = {}
        info.update(parsed)

        if not actions:
            self.reward += self.config.format_penalty
            done = False
            info["dino_score"] = 0.0
            info["dreamsim_score"] = 0.0
            info["success"] = metrics["traj_metrics"]["success"]
            info["metrics"] = metrics
            info["verifier_context"] = build_svg_verifier_context(
                self.gt_image, None, self.gt_svg_code, None
            )
            self.total_reward += self.reward
            self.gen_svg_code = None
            obs = self._render_obs(init_obs=False)
            return obs, self.reward, done, info

        if parsed.get("format_correct", True):
            self.reward += self.config.format_reward

        self.gen_svg_code = actions[0]
        self.valid_actions = actions

        try:
            _, gen_image = await asyncio.to_thread(
                process_and_rasterize_svg, self.gen_svg_code, self.config.resolution
            )
            self.gen_image = gen_image

            score_config = self.config.get_score_config()

            dino = None
            dreamsim = None
            try:
                dino = self._get_dino_model()
            except Exception as e:
                logger.warning("Failed to load DINO model: %s", e)
            try:
                dreamsim = self._get_dreamsim_model()
            except Exception as e:
                logger.warning("Failed to load DreamSim model: %s", e)

            scores = await asyncio.to_thread(
                calculate_total_score,
                gt_im=self.gt_image,
                gen_im=gen_image,
                gt_code=self.gt_svg_code,
                gen_code=self.gen_svg_code,
                score_config=score_config,
                dino_model=dino,
                dreamsim_model=dreamsim,
            )

            self.reward += scores["total_score"]
            info["scores"] = scores
            info["dino_score"] = scores["dino_score"]
            info["dreamsim_score"] = scores["dreamsim_score"]
            metrics["turn_metrics"]["action_is_effective"] = scores["total_score"] > 0

            # Derive a 0/1 success flag from similarity quality. total_score
            # already = dino*dino_weight + structural*structural_weight +
            # dreamsim*dreamsim_weight (no format_reward here — that's added
            # separately above). Compute the max-achievable similarity score
            # with the configured weights and mark success if total_score
            # reaches `success_score_threshold` of it.
            sim_max = (
                max(0.0, self.config.dino_weight or 0.0)
                + max(0.0, self.config.structural_weight or 0.0)
                + max(0.0, self.config.dreamsim_weight or 0.0)
            )
            if sim_max > 0 and (scores["total_score"] / sim_max) >= self.config.success_score_threshold:
                metrics["traj_metrics"]["success"] = True

        except Exception as e:
            logger.warning("SVG scoring failed: %s", e)
            self.valid_actions = []
            metrics["turn_metrics"]["action_is_valid"] = False
            info["dino_score"] = 0.0
            info["dreamsim_score"] = 0.0

        info["metrics"] = metrics
        info["success"] = metrics["traj_metrics"]["success"]
        info["verifier_context"] = build_svg_verifier_context(
            self.gt_image, self.gen_image, self.gt_svg_code, self.gen_svg_code
        )
        self.total_reward += self.reward

        obs = self._render_obs(init_obs=False)
        return obs, self.reward, done, info

    def _get_system_prompt(self) -> str:
        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=self.config.use_example_in_sys_prompt,
            prompt_format=self.config.prompt_format,
        )
        return system_prompt(format=self.config.prompt_format) + "\n" + format_prompt_str

    def _render_obs(self, init_obs: bool = False) -> Dict[str, Any]:
        if init_obs:
            img = self.gt_image
        elif self.gen_svg_code:
            img = self.gen_image
        else:
            img = Image.new("RGB", (self.config.resolution, self.config.resolution), color="white")

        img_placeholder = self.config.image_placeholder
        multi_modal_input = {img_placeholder: [img]}

        format_prompt_str = format_prompt(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False,
            prompt_format=self.config.prompt_format,
        )

        if init_obs:
            obs_str = init_observation_template(observation=img_placeholder) + "\n" + format_prompt_str
        else:
            valid_action_str = self.valid_actions[0] if self.valid_actions else ""
            obs_str = action_template(
                valid_action=valid_action_str,
                observation=img_placeholder,
                reward=self.reward,
                done=False,
            ) + "\n" + format_prompt_str

        return {"obs_str": obs_str, "multi_modal_input": multi_modal_input}

    @staticmethod
    def _extract_svg_code(text: str) -> str:
        svg_match = re.search(r"<svg.*?</svg>", text, re.DOTALL)
        if svg_match:
            return svg_match.group(0)
        if "<svg" in text and "</svg>" in text:
            start_idx = text.find("<svg")
            end_idx = text.rfind("</svg>") + 6
            if start_idx < end_idx:
                return text[start_idx:end_idx]
        return ""

    def _get_dino_model(self):
        if self._dino_model is not None:
            return self._dino_model
        from vagen.envs.svg.dino import DINOScoreCalculator
        dino_device = self.config.device.get("dino", "cuda:0")
        key = (str(self.config.model_size), str(dino_device))
        with _MODEL_LOCK:
            m = _DINO_SINGLETON.get(key)
            if m is None:
                m = DINOScoreCalculator(
                    model_size=self.config.model_size, device=dino_device
                )
                _DINO_SINGLETON[key] = m
        self._dino_model = m
        return self._dino_model

    def _get_dreamsim_model(self):
        if self._dreamsim_model is not None:
            return self._dreamsim_model
        from vagen.envs.svg.dreamsim import DreamSimScoreCalculator
        dreamsim_device = self.config.device.get("dreamsim", "cuda:0")
        key = (str(dreamsim_device),)
        with _MODEL_LOCK:
            m = _DREAMSIM_SINGLETON.get(key)
            if m is None:
                m = DreamSimScoreCalculator(device=dreamsim_device)
                _DREAMSIM_SINGLETON[key] = m
        self._dreamsim_model = m
        return self._dreamsim_model
