# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto


def _json_safe_default(obj):
    """JSON default handler: coerce numpy scalars/arrays to native Python.

    Mirrors numpy generic types so reward_extra_infos / generation dumps that
    carry np.ndarray / np.integer / np.floating / np.bool_ fields (e.g. some
    env reward-info dicts) serialize without raising
    `Object of type ndarray is not JSON serializable` during validation dumps.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, masked_whiten
from vagen.utils.image_dump_actor import ImageDumpActor
from vagen.utils.upload_hugging_face import HFUploadManager
from vagen.utils.image_validation_logger import ValidationGenerationsLogger
from vagen.utils.concat_val_multi_turn import concat_val_multi_turn
from vagen.utils.image_token_utils import replace_image_tokens_for_logging
from vagen.rlcer.prompt_templates import (
    FROZENLAKE_RUBRICATOR_SYSTEM_PROMPT,
    FROZENLAKE_RUBRICATOR_USER_PROMPT,
    RLCER_RUBRICATOR_SYSTEM_PROMPT,
    RLCER_RUBRICATOR_USER_PROMPT,
)
from vagen.rlcer.scheduler import RubricatorScheduler
import vagen.custom_advantage
from vagen.custom_metric.metric import METRIC_REGISTRY
from vagen.custom_filter.filter import FILTER_REGISTRY


def _strip_multimodal_placeholders(text: str) -> str:
    """Remove common multimodal placeholders from decoded chat text.

    When building auxiliary rubricator prompts from decoded token ids, residual
    vision placeholders (e.g. Qwen-VL image tokens) can be preserved in text.
    Sending such text without matching `image_data` may trigger SGLang/processor
    index errors. This helper strips those placeholders for text-only prompts.
    """
    if not text:
        return ""

    s = str(text)
    # Common multimodal placeholders across templates/backends.
    for token in (
        "<image>",
        "[image]",
        "<img>",
        "<|image_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|video_pad|>",
        "<|audio_pad|>",
    ):
        s = s.replace(token, " ")

    # Defensive collapse for any remaining qwen-vl style control tokens.
    s = re.sub(r"<\|[^>]*?(image|vision|video|audio)[^>]*\|>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _flatten_chat_content(content) -> str:
    """Flatten structured chat content into text (preserving <image>/<video> markers)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            chunks.append(str(item))
            continue
        t = item.get("type")
        if t == "text":
            chunks.append(str(item.get("text", "")))
        elif t == "image":
            chunks.append("<image>")
        elif t == "video":
            chunks.append("<video>")
    return "".join(chunks)


def _record_rlcer_reward_diagnostics(metrics: dict, reward_extra_infos_dict: dict) -> None:
    """Expose numeric RLCER reward diagnostics in the per-step trainer log."""
    key_map = {
        "valid_ratio": "valid_ratio",
        "rubric_accepted_rate": "rubrics_accepted_rate",
        "rubric_acceptance_rate_overall": "rubric_acceptance_rate_overall",
        "rubric_acceptance_rate_given_computable": "rubric_acceptance_rate_given_computable_sample_mean",
        "correlation_unavailable_rate": "correlation_unavailable_rate_sample_mean",
        "corr_mean": "accepted_corr_mean",
        "corr_mean_all": "all_corr_mean",
        "corr_computable_ratio": "corr_computable_ratio",
        "group_target_computable": "group_target_computable_ratio",
        "group_corr_used": "group_corr_used_ratio",
        "tracker_corr_used": "tracker_corr_used_ratio",
        "tracker_corr": "tracker_corr_mean",
        "tracker_effective_n": "tracker_effective_n",
        "tracker_observation_count": "tracker_observation_count",
        "trivial_scale": "trivial_scale_mean",
        "proposal_format_ok": "proposal_format_ok_rate",
        "proposal_nonempty": "proposal_nonempty_rate",
        "reasoner_cot_format_ok": "reasoner_cot_format_ok_rate",
        "reasoner_answer_format_ok": "reasoner_answer_format_ok_rate",
        "rubric_total_count": "rubric_count_mean",
    }
    for source_key, metric_name in key_map.items():
        values = reward_extra_infos_dict.get(source_key)
        if values is None or len(values) == 0:
            continue
        try:
            numeric = np.asarray(values, dtype=np.float64)
            finite = numeric[np.isfinite(numeric)]
            if finite.size:
                metrics[f"rlcer/train/{metric_name}"] = float(finite.mean())
        except (TypeError, ValueError):
            continue

    # Count-weighted coverage metrics separate statistical unavailability from
    # genuine correlation rejection.
    try:
        total = np.asarray(
            reward_extra_infos_dict.get("rubric_total_count", []),
            dtype=np.float64,
        )
        accepted = np.asarray(
            reward_extra_infos_dict.get("accepted_correlation_count", []),
            dtype=np.float64,
        )
        rejected = np.asarray(
            reward_extra_infos_dict.get("rejected_correlation_count", []),
            dtype=np.float64,
        )
        unavailable = np.asarray(
            reward_extra_infos_dict.get("correlation_unavailable_count", []),
            dtype=np.float64,
        )
        if total.size and accepted.size == total.size:
            total_count = float(np.nansum(total))
            computable_count = float(np.nansum(accepted) + np.nansum(rejected))
            if total_count > 0:
                metrics["rlcer/train/rubric_acceptance_rate_overall_weighted"] = (
                    float(np.nansum(accepted)) / total_count
                )
                metrics["rlcer/train/correlation_unavailable_rate_weighted"] = (
                    float(np.nansum(unavailable)) / total_count
                )
            if computable_count > 0:
                metrics[
                    "rlcer/train/rubric_acceptance_rate_given_computable"
                ] = float(np.nansum(accepted)) / computable_count
    except (TypeError, ValueError):
        pass

    # Correlation means must be weighted by the number of computable rubrics.
    # Averaging per-sample zero placeholders would bias steps where one side is
    # absent (for example, no accepted rubrics in a sample).
    for mean_key, count_key, metric_name in (
        (
            "mean_accepted_correlation",
            "accepted_correlation_count",
            "mean_accepted_correlation",
        ),
        (
            "mean_rejected_correlation",
            "rejected_correlation_count",
            "mean_rejected_correlation",
        ),
    ):
        means = reward_extra_infos_dict.get(mean_key)
        counts = reward_extra_infos_dict.get(count_key)
        if means is None or counts is None or len(means) == 0:
            continue
        try:
            mean_values = np.asarray(means, dtype=np.float64)
            count_values = np.asarray(counts, dtype=np.float64)
            mask = (
                np.isfinite(mean_values)
                & np.isfinite(count_values)
                & (count_values > 0)
            )
            if np.any(mask):
                metrics[f"rlcer/train/{metric_name}"] = float(
                    np.average(mean_values[mask], weights=count_values[mask])
                )
        except (TypeError, ValueError):
            continue


def _sampled_anchor_kl(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """Non-negative sampled KL estimate on a fixed prompt/response probe."""
    mask = response_mask.to(dtype=torch.bool)
    if not torch.any(mask):
        return 0.0
    log_ratio = (new_log_probs - old_log_probs)[mask].float()
    # Schulman's non-negative Monte-Carlo KL approximation under old-policy
    # samples: exp(log_ratio) - 1 - log_ratio.
    return float((torch.exp(log_ratio) - 1.0 - log_ratio).mean().item())


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld
    token_level_kl_penalty = -beta * kld
    data.batch["token_level_kl_penalty"] = token_level_kl_penalty

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_custom_metrics(data: DataProto, prefix: str = "custom_metrics") -> dict:
    """Compute all custom metrics registered in METRIC_REGISTRY.

    Args:
        data (DataProto): The data containing batch information.
        prefix (str): Prefix for metric names in the returned dictionary.

    Returns:
        dict: A dictionary containing all computed custom metrics with appropriate prefixes.
    """
    custom_metrics = {}

    for metric_name, metric_fn in METRIC_REGISTRY.items():
        try:
            metric_value = metric_fn(data)
            custom_metrics[f"{prefix}/{metric_name}"] = metric_value
        except Exception as e:
            print(f"Warning: Failed to compute custom metric '{metric_name}': {e}")

    return custom_metrics

def _default_eps(
    x: torch.Tensor,
    small_eps: float = 1e-2,
    large_eps: float = 1e-6,
) -> float:
    """
    Choose a comparison tolerance (eps) based on tensor dtype.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return small_eps
    return large_eps


def compute_value_mask(
    data: DataProto,
    ignore_value: float = -100.0,
    eps: float | None = None,
) -> torch.Tensor:
    """
    Compute value-function loss mask from token-level returns.

    Value loss is only computed at positions where `returns` is valid.
    Invalid / ignored positions are marked by a float sentinel
    `ignore_value` (default: -100.0), similar in spirit to
    CrossEntropy's `ignore_index`.

    If you do NOT want a certain token position to participate in
    value-function training, simply write:

        returns[..., pos] = ignore_value

    This mask will then automatically exclude that position from
    value loss computation.
    """
    returns = data.batch["returns"]

    if eps is None:
        eps = _default_eps(returns)

    # Identify ignored positions via approximate comparison
    is_ignored = (returns - ignore_value).abs() < eps

    # Mask dtype is aligned with response_mask for numerical stability
    return (~is_ignored).to(dtype=data.batch["response_mask"].dtype)


    

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    whiten: bool = True,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
            whiten=whiten,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "data": data,
            "config": config,
            "gamma": gamma,
            "lam": lam,
            "num_repeat": num_repeat,
            "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
            "whiten": whiten,
        }
        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._image_dump_actors = {}
        self._pending_dump_futures = []
        self._log_image_cfg = self.config.trainer.get("log_image", {})
        self._log_image_enable = self._log_image_cfg.get("enable", False)
        self._max_pending_dumps = self._log_image_cfg.get("max_pending", 2)

        # HuggingFace Hub upload
        self._hf_upload_manager = HFUploadManager(config)

        self._rlcer_rubric_gen_cache = None
        self._rubric_replay_batches = []

        anchor_cfg = (
            config.get("custom_reward_function", {})
            .get("reward_kwargs", {})
            .get("anchor_kl", {})
        )
        self._anchor_kl_enabled = bool(anchor_cfg.get("enabled", False))
        self._anchor_kl_interval = max(1, int(anchor_cfg.get("interval", 20)))
        self._anchor_kl_samples = max(1, int(anchor_cfg.get("samples", 8)))

        # Rubricator scheduler: anti-Nash equilibrium / plateau detection / warmup
        self.rubricator_scheduler = RubricatorScheduler(config)
        self._latest_val_metrics = None

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, images, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_safe_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

        # Save images to subfolders
        if images and self._log_image_enable:
            actor = self._image_dump_actors.get(dump_path)
            if actor is None:
                actor = ImageDumpActor.remote(base_dir=dump_path)
                self._image_dump_actors[dump_path] = actor

            compress_level = self._log_image_cfg.get("png_compress_level", 0)
            fut = actor.dump_images.remote(
                step=self.global_steps,
                images=images,
                compress_level=compress_level,
            )
            self._pending_dump_futures.append(fut)

            if self._max_pending_dumps > 0 and len(self._pending_dump_futures) > self._max_pending_dumps:
                done, rest = ray.wait(self._pending_dump_futures, num_returns=1)
                ray.get(done)
                self._pending_dump_futures = rest

    def _flush_image_dumps(self):
        if not self._pending_dump_futures:
            return
        ray.get(self._pending_dump_futures)
        self._pending_dump_futures = []

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            
            inputs = batch.batch["prompts"]
            outputs = batch.batch["responses"]
            
            # remove pad tokens for logging (keeps other special tokens like <|endoftext|>
            # visible so we can spot degenerate model outputs)
            pad_token_id = self.tokenizer.pad_token_id
            skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)

            if self.config.trainer.get("replace_image_tokens_for_logging", False):
                inputs = replace_image_tokens_for_logging(inputs, processor=self.processor, tokenizer=self.tokenizer)
                outputs = replace_image_tokens_for_logging(outputs, processor=self.processor, tokenizer=self.tokenizer)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            # Extract images from non_tensor_batch (extra_fields are stored there)
            sample_images=[]
            if "image_data" in batch.non_tensor_batch:
                batch_images = batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))
            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                images=sample_images,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score, image) and sort by input text
        if images is None or len(images) == 0:
            images = [None] * len(inputs)
        else:
            non_none_count = sum(1 for img in images if img is not None)
            print(f"Logging {non_none_count}/{len(images)} validation samples with images to wandb")

        samples = list(zip(inputs, outputs, scores, images, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _assign_group_and_traj_idx(self, gen_batch: DataProto, num_traj_per_sample: int) -> None:
        """Assign group_idx and traj_idx for no-concat mode.

        Args:
            gen_batch: The generated batch after repeat operation
            num_traj_per_sample: Number of trajectories per sample (repeat_times)
        """
        # Assign group_idx from uid
        gen_batch.non_tensor_batch["group_idx"] = gen_batch.non_tensor_batch["uid"]

        # Assign traj_idx based on repeat pattern
        # Since repeat with interleave=True creates [A, A, A, B, B, B, C, C, C] pattern,
        # traj_idx should be [0, 1, 2, 0, 1, 2, 0, 1, 2]
        batch_size = len(gen_batch.non_tensor_batch["uid"])
        traj_idx = np.tile(np.arange(num_traj_per_sample), batch_size // num_traj_per_sample)
        gen_batch.non_tensor_batch["traj_idx"] = traj_idx


    def _post_process_no_concat_batch(self, batch: DataProto, gen_batch_output: DataProto) -> DataProto:
        """Re-align and union batch with gen_batch_output in no-concat mode.

        In no-concat mode, each trajectory has multiple prompt-response pairs with varying lengths.
        Each original batch item may correspond to a different number of gen_batch_output items
        depending on how many turns were generated for that trajectory.

        The key insight: we build a selection index list that maps each gen_batch_output item
        to its corresponding original batch item, then use select_idxs to replicate and reorder
        the batch to match gen_batch_output's uid sequence.

        Args:
            batch: Original batch with reward model keys (uid, data_source, etc.)
            gen_batch_output: Generated output with sequences and uid (variable items per original uid)

        Returns:
            DataProto: Aligned and unified batch ready for downstream processing

        Example:
            Original batch: [item_0 (uid=C), item_1 (uid=B), item_2 (uid=A)]
            gen_batch_output uids: [A, A, A, B, B, C, C, C, C]
            -> selection_indices: [2, 2, 2, 1, 1, 0, 0, 0, 0]
            -> select_idxs([2,2,2,1,1,0,0,0,0]): [A, A, A, B, B, C, C, C, C]
            -> Perfectly aligned with gen_batch_output!
        """
        # Step 1: Verify uid exists in both batches
        assert "uid" in batch.non_tensor_batch, "batch must contain 'uid' in non_tensor_batch for alignment"
        gen_batch_output.non_tensor_batch["uid"]=gen_batch_output.non_tensor_batch["group_idx"]
        assert "uid" in gen_batch_output.non_tensor_batch, (
            "gen_batch_output must contain 'uid' in non_tensor_batch for alignment"
        )

        # Step 2: Build uid to index mapping for original batch
        batch_uid_to_idx = {str(uid): idx for idx, uid in enumerate(batch.non_tensor_batch["uid"])}

        # Step 3: Build selection indices by mapping each gen_batch_output uid to its batch index
        # This automatically handles:
        # - Variable repetition (each uid can appear different number of times)
        # - Arbitrary ordering (gen_batch_output uids can be in any order)
        selection_indices = []

        for gen_uid in gen_batch_output.non_tensor_batch["uid"]:
            gen_uid_str = str(gen_uid)
            if gen_uid_str not in batch_uid_to_idx:
                raise ValueError(
                    f"uid '{gen_uid_str}' from gen_batch_output not found in batch. "
                    f"Available uids: {list(batch_uid_to_idx.keys())[:5]}... "
                    f"This suggests a data alignment issue in agent loop."
                )
            batch_idx = batch_uid_to_idx[gen_uid_str]
            selection_indices.append(batch_idx)

        # Step 4: Use select_idxs to replicate and reorder batch to match gen_batch_output
        # This single operation handles both repetition and reordering
        batch = batch.select_idxs(selection_indices)

        # Step 5: Verify the size matches
        assert len(batch) == len(gen_batch_output), (
            f"After alignment, batch size ({len(batch)}) should match gen_batch_output size ({len(gen_batch_output)}). "
            f"selection_indices length: {len(selection_indices)}"
        )

        # Step 6: Union the aligned batches
        batch = batch.union(gen_batch_output)

        return batch

    def _build_terminal_reward_tensor_from_scores(self, batch: DataProto, scores: torch.Tensor) -> torch.Tensor:
        """Create token-level reward tensor with scalar rewards placed on terminal response token."""
        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        prompt_len = batch.batch["prompts"].shape[-1]
        valid_response_lengths = batch.batch["attention_mask"][:, prompt_len:].sum(dim=-1).long()

        if scores.dtype != torch.float32:
            scores = scores.to(dtype=torch.float32)
        scores = scores.to(device=reward_tensor.device)

        row_idx = torch.arange(reward_tensor.shape[0], device=reward_tensor.device)
        col_idx = torch.clamp(valid_response_lengths - 1, min=0)
        reward_tensor[row_idx, col_idx] = scores
        return reward_tensor

    def _build_rlcer_policy_rubricator_prompts(self, batch: DataProto) -> tuple[list[list[int]], list[Optional[list]]]:
        """Build rubricator model inputs from current trajectories.

        Each input includes:
        - rubrics_generation-style system prompt
        - rubrics_generation-style user prompt (with current observation image)
        - current solver response context
        """
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        prompt_len = batch.batch["prompts"].shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        responses: list[str] = []
        for i in range(len(batch)):
            valid_len = int(valid_response_lengths[i].item())
            responses.append(self.tokenizer.decode(response_ids[i][:valid_len], skip_special_tokens=True))

        image_batch = batch.non_tensor_batch.get("image_data", [None] * len(batch))
        data_source_batch = batch.non_tensor_batch.get("data_source", [None] * len(batch))
        prompt_id_list: list[list[int]] = []
        image_payloads: list[Optional[list]] = []
        # print('****'*10, '848', 'response',responses[0][:400])
        for i, r in enumerate(responses):
            img_item = image_batch[i] if i < len(image_batch) else None
            if isinstance(img_item, list):
                current_image = img_item[-1] if len(img_item) > 0 else None
            else:
                current_image = img_item

            data_source = data_source_batch[i] if i < len(data_source_batch) else None
            is_frozenlake = "frozenlake" in str(data_source).lower()
            rubricator_system_prompt = (
                FROZENLAKE_RUBRICATOR_SYSTEM_PROMPT
                if is_frozenlake
                else RLCER_RUBRICATOR_SYSTEM_PROMPT
            )
            rubricator_user_prompt = (
                FROZENLAKE_RUBRICATOR_USER_PROMPT
                if is_frozenlake
                else RLCER_RUBRICATOR_USER_PROMPT
            )
            task_instruction = (
                "\nAnalyze the current FrozenLake board state and generate evaluation rubrics."
                if is_frozenlake
                else "\nAnalyze the current Sokoban board state and generate evaluation rubrics."
            )

            user_tail = (
                f"\n\n[Turn ID]\nturn_{int(self.global_steps):06d}\n"
                f"\n[Solver Response]\n{_strip_multimodal_placeholders(r)}\n"
                "\n[Constraint]\nReturn ONLY the JSON in a json block."
            )

            if self.processor is not None:
                messages = [
                    {"role": "system", "content": rubricator_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "[Initial Observation]:\n"},
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": task_instruction + user_tail,
                            },
                        ],
                    },
                ]
                raw_prompt = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                
                model_inputs = self.processor(
                    text=[raw_prompt],
                    images=[current_image] if current_image is not None else None,
                    return_tensors="pt",
                )
                ids = model_inputs["input_ids"].squeeze(0).tolist()
            else:
                flat_messages = [
                    {"role": "system", "content": rubricator_system_prompt},
                    {
                        "role": "user",
                        "content": f"{rubricator_user_prompt}\n{user_tail}",
                    },
                ]
                ids = self.tokenizer.apply_chat_template(
                    flat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
            
            ids = [int(x) for x in ids][-int(self.config.data.max_prompt_length) :]
            prompt_id_list.append(ids)
            image_payloads.append([current_image] if current_image is not None else None)
        
        return prompt_id_list, image_payloads

    def _generate_rlcer_policy_rubrics_with_current_actor(self, batch: DataProto) -> list[str] | None:
        """Generate rubric proposals using the CURRENT actor policy.

        This makes rubricator and solver share the same synchronized policy weights.
        """
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        rubricator_cfg = reward_cfg.get("rubricator", {})
        rubric_mode = str(rubricator_cfg.get("mode", "")).lower()

        self._rlcer_rubric_gen_cache = None

        if rubric_mode != "policy":
            return None

        use_actor_policy = bool(reward_cfg.get("policy_rubricator_use_actor_generation", True))
        if not use_actor_policy:
            return None
        
        prompt_id_list, image_payloads = self._build_rlcer_policy_rubricator_prompts(batch)
        if len(prompt_id_list) == 0:
            return []
        
        # Async rollout path: call server handles directly.
        if self.async_rollout_mode:
            server_handles = getattr(self.async_rollout_manager, "server_handles", None)
            if not server_handles:
                return None

            rollout_cfg = self.config.actor_rollout_ref.rollout
            if rollout_cfg.get("free_cache_engine", False):
                self.async_rollout_manager.wake_up()

            try:
                max_new_tokens = int(reward_cfg.get("policy_rubricator_max_new_tokens", 4096))
                rubric_temperature = float(reward_cfg.get("policy_rubricator_temperature", 0.3))
                rubric_top_p = float(reward_cfg.get("policy_rubricator_top_p", 0.9))
                sampling_params = {
                    "temperature": rubric_temperature,
                    "top_p": rubric_top_p,
                    "max_new_tokens": max_new_tokens,
                }

                reqs = []
                for i, ids in enumerate(prompt_id_list):
                    server = server_handles[i % len(server_handles)]
                    req_id = f"rlcer-rubric-{self.global_steps}-{i}-{uuid.uuid4().hex[:8]}"
                    reqs.append(
                        server.generate.remote(
                            prompt_ids=ids,
                            sampling_params=sampling_params,
                            request_id=req_id,
                            image_data=image_payloads[i],
                        )
                    )
                
                outs = ray.get(reqs)
                rubric_raws: list[str] = []
                rubric_response_ids: list[list[int]] = []
                for out in outs:
                    toks = list(out.token_ids)
                    rubric_response_ids.append(toks)
                    rubric_raws.append(self.tokenizer.decode(toks, skip_special_tokens=True))

                self._rlcer_rubric_gen_cache = {
                    "source": "async",
                    "prompt_id_list": prompt_id_list,
                    "response_ids_list": rubric_response_ids,
                    "image_payloads": image_payloads,
                    "pad_token_id": self.tokenizer.pad_token_id or 0,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }

                return rubric_raws
            finally:
                if rollout_cfg.get("free_cache_engine", False):
                    self.async_rollout_manager.sleep()
       
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max(len(x) for x in prompt_id_list)
        input_ids = torch.full((len(prompt_id_list), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompt_id_list), max_len), dtype=torch.long)
        for i, ids in enumerate(prompt_id_list):
            l = len(ids)
            if l == 0:
                continue
            input_ids[i, :l] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :l] = 1
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
      
        rubric_gen_batch = DataProto.from_single_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        rubric_gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": False,
            "global_steps": self.global_steps,
        }
        rubric_gen_batch.non_tensor_batch["multi_modal_data"] = np.array(
            [{"image": imgs} if imgs is not None else None for imgs in image_payloads],
            dtype=object,
        )

        # Use actor rollout worker directly so rubricator always follows latest actor updates.
        rubric_out = self.actor_rollout_wg.generate_sequences(rubric_gen_batch)
       
        generated = rubric_out.batch["responses"]
        gen_mask = rubric_out.batch["attention_mask"][:, input_ids.shape[-1] :]
        gen_lens = gen_mask.sum(dim=-1)
        rubric_raws: list[str] = []

        for i in range(generated.shape[0]):
            l = int(gen_lens[i].item())
            rubric_raws.append(self.tokenizer.decode(generated[i][:l], skip_special_tokens=True))

        self._rlcer_rubric_gen_cache = {
            "source": "sync",
            "prompt_id_list": prompt_id_list,
            "input_ids": input_ids,
            "prompt_attention_mask": attention_mask,
            "responses": generated,
            "response_mask": gen_mask,
            "full_attention_mask": rubric_out.batch["attention_mask"],
            "meta_info": dict(rubric_gen_batch.meta_info),
            "image_payloads": image_payloads,
        }

        return rubric_raws

    def _build_rlcer_joint_training_batch(
        self,
        reasoner_batch: DataProto,
        reasoner_reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict,
        metrics: dict,
    ) -> DataProto:
        """Optionally append rubricator-role trajectories for joint PPO update.

        This function uses rubricator rewards returned by custom reward function
        (`reward_extra_info['rubricator_reward']`) and constructs a second role batch
        with role-specific advantages, then concatenates it with reasoner batch.
        """
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        enable_dual_role_update = reward_cfg.get("enable_dual_role_update", False)

        reasoner_batch.non_tensor_batch["rlcer_role"] = np.array(["reasoner"] * len(reasoner_batch), dtype=object)
        if not enable_dual_role_update:
            return reasoner_batch

        update_interval = max(1, int(reward_cfg.get("rubricator_update_interval", 1)))
        if self.global_steps % update_interval != 0:
            metrics["rlcer/train/dual_role_skipped_update_interval"] = 1.0
            metrics["rlcer/train/rubricator_update_interval"] = float(update_interval)
            return reasoner_batch

        rubricator_rewards = reward_extra_infos_dict.get("rubricator_reward", None)
        if rubricator_rewards is None:
            metrics["rlcer/train/dual_role_skipped_no_rubricator_reward"] = 1.0
            return reasoner_batch
        if len(rubricator_rewards) != len(reasoner_batch):
            metrics["rlcer/train/dual_role_skipped_len_mismatch"] = 1.0
            return reasoner_batch

        # Signal gate: skip dual-role when rubricator has no valid signal
        valid_ratios = reward_extra_infos_dict.get("valid_ratio", None)
        if valid_ratios is not None:
            mean_valid_ratio = float(np.mean(valid_ratios))
            metrics["rlcer/train/mean_valid_ratio"] = mean_valid_ratio
            min_valid_ratio = float(reward_cfg.get("dual_role_min_valid_ratio", 0.01))
            if mean_valid_ratio < min_valid_ratio:
                metrics["rlcer/train/dual_role_skipped_no_signal"] = 1.0
                return reasoner_batch

        rubricator_scores = torch.tensor(rubricator_rewards, dtype=torch.float32, device=reasoner_reward_tensor.device)
        rubricator_batch = deepcopy(reasoner_batch)
        rubricator_batch.non_tensor_batch["rlcer_role"] = np.array(["rubricator"] * len(rubricator_batch), dtype=object)

        rubricator_reward_tensor = self._build_terminal_reward_tensor_from_scores(rubricator_batch, rubricator_scores)
        rubricator_batch.batch["token_level_scores"] = rubricator_reward_tensor
        rubricator_batch.batch["token_level_rewards"] = rubricator_reward_tensor

        # Role-specific advantage for rubricator
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        rubricator_batch = compute_advantage(
            rubricator_batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
            rubricator_batch.batch["value_mask"] = compute_value_mask(rubricator_batch)

        joint_batch = DataProto.concat([reasoner_batch, rubricator_batch])

        # Shuffle so rubricator samples distribute evenly across DP workers during chunk
        shuffle_indices = np.random.permutation(len(joint_batch))
        joint_batch = joint_batch[shuffle_indices]

        metrics["rlcer/train/dual_role_enabled"] = 1.0
        metrics["rlcer/train/rubricator_update_interval"] = float(update_interval)
        metrics["rlcer/train/rubricator_reward_mean"] = float(np.mean(rubricator_rewards))
        metrics["rlcer/train/rubricator_reward_max"] = float(np.max(rubricator_rewards))
        metrics["rlcer/train/rubricator_reward_min"] = float(np.min(rubricator_rewards))
        return joint_batch

    def _build_rubricator_trajectory_batch(
        self,
        rubricator_rewards: list,
        reasoner_batch: DataProto,
        metrics: dict,
    ) -> DataProto | None:
        """Build a training-ready DataProto from cached rubric generation data.

        Constructs a full trajectory batch (prompts + responses) for the rubricator
        role using the cached prompt/response token IDs from
        _generate_rlcer_policy_rubrics_with_current_actor.  The rubricator tokens
        are the ACTUAL generation tokens, so PPO gradients will flow through them.

        The returned batch has token_level_scores/rewards set but **does not** have
        old_log_probs, values, or advantages yet — those must be computed by the
        caller before PPO update.

        Returns None if cached generation data is unavailable.
        """
        cache = getattr(self, "_rlcer_rubric_gen_cache", None)
        if cache is None:
            metrics["rlcer/train/rubricator_traj_no_cache"] = 1.0
            return None

        n_reward = len(rubricator_rewards)

        # ---------- Build input_ids / attention_mask / position_ids ----------
        pad_token_id = cache.get("pad_token_id", self.tokenizer.pad_token_id or 0)

        if cache["source"] == "sync":
            prompt_ids = cache["input_ids"]
            prompt_mask = cache["prompt_attention_mask"]
            responses = cache["responses"]
            resp_mask = cache["response_mask"]
            full_attn = cache["full_attention_mask"]
        else:
            prompt_id_list = cache["prompt_id_list"]
            response_ids_list = cache["response_ids_list"]
            max_prompt_len = max(len(p) for p in prompt_id_list)
            max_resp_len = max(len(r) for r in response_ids_list)

            B = len(prompt_id_list)
            prompt_ids = torch.full((B, max_prompt_len), pad_token_id, dtype=torch.long)
            prompt_mask = torch.zeros((B, max_prompt_len), dtype=torch.long)
            responses = torch.full((B, max_resp_len), pad_token_id, dtype=torch.long)
            resp_mask = torch.zeros((B, max_resp_len), dtype=torch.long)

            for i in range(B):
                plen = len(prompt_id_list[i])
                prompt_ids[i, :plen] = torch.tensor(prompt_id_list[i], dtype=torch.long)
                prompt_mask[i, :plen] = 1
                rlen = len(response_ids_list[i])
                responses[i, :rlen] = torch.tensor(response_ids_list[i], dtype=torch.long)
                resp_mask[i, :rlen] = 1

            full_attn = torch.cat([prompt_mask, resp_mask], dim=1)

        B_rub = prompt_ids.shape[0]
        if B_rub != n_reward:
            metrics["rlcer/train/rubricator_traj_len_mismatch"] = 1.0
            return None

        input_ids = torch.cat([prompt_ids, responses], dim=1)

        reasoner_pos_ids = reasoner_batch.batch.get("position_ids", None)
        if reasoner_pos_ids is not None and reasoner_pos_ids.dim() == 3:
            pos_2d = full_attn.long().cumsum(dim=-1) - 1
            pos_2d = pos_2d.masked_fill(full_attn == 0, 0)
            n_rope_dims = reasoner_pos_ids.shape[1]
            position_ids = pos_2d.unsqueeze(1).expand(-1, n_rope_dims, -1)
        else:
            position_ids = full_attn.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(full_attn == 0, 0)

        # ---------- Build reward tensors ----------
        device = reasoner_batch.batch["responses"].device
        rubricator_scores = torch.tensor(rubricator_rewards, dtype=torch.float32, device=device)

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        valid_resp_lens = resp_mask.sum(dim=-1).long()
        row_idx = torch.arange(B_rub, device=device)
        col_idx = torch.clamp(valid_resp_lens - 1, min=0)
        reward_tensor[row_idx, col_idx] = rubricator_scores

        # ---------- Assemble DataProto ----------
        rub_batch_dict = {
            "input_ids": input_ids,
            "attention_mask": full_attn,
            "position_ids": position_ids,
            "prompts": prompt_ids,
            "responses": responses,
            "response_mask": resp_mask,
            "token_level_scores": reward_tensor,
            "token_level_rewards": reward_tensor,
        }

        rubricator_batch = DataProto.from_single_dict(rub_batch_dict)
        rubricator_batch.meta_info = dict(reasoner_batch.meta_info)

        rubricator_batch.non_tensor_batch["rlcer_role"] = np.array(
            ["rubricator"] * B_rub, dtype=object
        )

        for key in ("uid", "group_idx", "traj_idx"):
            if key in reasoner_batch.non_tensor_batch:
                src = reasoner_batch.non_tensor_batch[key]
                if len(src) == B_rub:
                    rubricator_batch.non_tensor_batch[key] = np.array(src, dtype=object)

        image_payloads = cache.get("image_payloads", None)
        if image_payloads is not None and len(image_payloads) == B_rub:
            rubricator_batch.non_tensor_batch["multi_modal_data"] = np.array(
                [{"image": imgs} if imgs is not None else None for imgs in image_payloads],
                dtype=object,
            )

        # ---------- Pad to match reasoner_batch sequence length ----------
        reasoner_seq_len = reasoner_batch.batch["input_ids"].shape[1]
        rub_seq_len = input_ids.shape[1]

        if rub_seq_len < reasoner_seq_len:
            pad_len = reasoner_seq_len - rub_seq_len
            for key in list(rubricator_batch.batch.keys()):
                tensor = rubricator_batch.batch[key]
                if key == "position_ids" and tensor.dim() == 3:
                    rubricator_batch.batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=0)
                elif key == "prompts" and tensor.dim() == 2:
                    pass
                elif tensor.dim() == 2 and tensor.shape[0] == B_rub:
                    if key == "input_ids":
                        rubricator_batch.batch[key] = torch.nn.functional.pad(
                            tensor, (0, pad_len), value=pad_token_id
                        )
                    else:
                        rubricator_batch.batch[key] = torch.nn.functional.pad(
                            tensor, (0, pad_len), value=0
                        )
            rub_P = prompt_ids.shape[1]
            rubricator_batch.batch["responses"] = rubricator_batch.batch["input_ids"][:, rub_P:]
        elif rub_seq_len > reasoner_seq_len:
            metrics["rlcer/train/rubricator_traj_truncated"] = 1.0
            max_resp = reasoner_seq_len - prompt_ids.shape[1]
            if max_resp <= 0:
                metrics["rlcer/train/rubricator_traj_prompt_too_long"] = 1.0
                return None
            for key in list(rubricator_batch.batch.keys()):
                tensor = rubricator_batch.batch[key]
                if key == "position_ids" and tensor.dim() == 3:
                    rubricator_batch.batch[key] = tensor[:, :, :reasoner_seq_len]
                elif tensor.dim() == 2 and tensor.shape[0] == B_rub:
                    rubricator_batch.batch[key] = tensor[:, :reasoner_seq_len]
            rubricator_batch.batch["responses"] = rubricator_batch.batch["input_ids"][:, prompt_ids.shape[1]:]
            rubricator_batch.batch["response_mask"] = rubricator_batch.batch["attention_mask"][:, prompt_ids.shape[1]:]

        return rubricator_batch

    def _concat_rubricator_batch_before_fwd(
        self,
        reasoner_batch: DataProto,
        reward_extra_infos_dict: dict,
        metrics: dict,
    ) -> DataProto:
        """Concatenate rubricator-role trajectories with reasoner batch BEFORE
        compute_log_prob / compute_values so both roles share the same ρ_t(θ).

        Falls back to deepcopy when the generation cache is unavailable.
        """
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        enable_dual_role_update = reward_cfg.get("enable_dual_role_update", False)

        reasoner_batch.non_tensor_batch["rlcer_role"] = np.array(["reasoner"] * len(reasoner_batch), dtype=object)
        if not enable_dual_role_update:
            return reasoner_batch

        rubricator_rewards = reward_extra_infos_dict.get("rubricator_reward", None)
        if rubricator_rewards is None:
            metrics["rlcer/train/dual_role_skipped_no_rubricator_reward"] = 1.0
            return reasoner_batch
        if len(rubricator_rewards) != len(reasoner_batch):
            metrics["rlcer/train/dual_role_skipped_len_mismatch"] = 1.0
            return reasoner_batch

        # Signal gate: skip dual-role when rubricator has no valid signal
        valid_ratios = reward_extra_infos_dict.get("valid_ratio", None)
        if valid_ratios is not None:
            mean_valid_ratio = float(np.mean(valid_ratios))
            metrics["rlcer/train/mean_valid_ratio"] = mean_valid_ratio
            min_valid_ratio = float(reward_cfg.get("dual_role_min_valid_ratio", 0.01))
            if mean_valid_ratio < min_valid_ratio:
                metrics["rlcer/train/dual_role_skipped_no_signal"] = 1.0
                return reasoner_batch

        rubricator_batch = self._build_rubricator_trajectory_batch(
            rubricator_rewards=rubricator_rewards,
            reasoner_batch=reasoner_batch,
            metrics=metrics,
        )

        if rubricator_batch is not None:
            metrics["rlcer/train/dual_role_rubricator_trajectory"] = 1.0
        else:
            metrics["rlcer/train/dual_role_fallback_deepcopy"] = 1.0
            rubricator_scores = torch.tensor(
                rubricator_rewards, dtype=torch.float32,
                device=reasoner_batch.batch["responses"].device,
            )
            rubricator_batch = deepcopy(reasoner_batch)
            rubricator_batch.non_tensor_batch["rlcer_role"] = np.array(
                ["rubricator"] * len(rubricator_batch), dtype=object
            )
            rubricator_reward_tensor = self._build_terminal_reward_tensor_from_scores(
                rubricator_batch, rubricator_scores
            )
            rubricator_batch.batch["token_level_scores"] = rubricator_reward_tensor
            rubricator_batch.batch["token_level_rewards"] = rubricator_reward_tensor

        # --- Subsample rubricator to 1 mini_batch worth of samples ---
        rub_subsample = int(reward_cfg.get("rubricator_subsample_size", 0))
        if rub_subsample <= 0:
            rub_subsample = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if len(rubricator_batch) > rub_subsample:
            indices = np.random.choice(len(rubricator_batch), rub_subsample, replace=False)
            indices.sort()
            rubricator_batch = rubricator_batch[indices]
            metrics["rlcer/train/rubricator_subsampled_to"] = float(rub_subsample)

        # --- Ensure all tensor keys are aligned ---
        reasoner_keys = set(reasoner_batch.batch.keys())
        rubricator_keys = set(rubricator_batch.batch.keys())
        for key in reasoner_keys - rubricator_keys:
            ref_tensor = reasoner_batch.batch[key]
            shape = list(ref_tensor.shape)
            shape[0] = len(rubricator_batch)
            rubricator_batch.batch[key] = torch.zeros(shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for key in rubricator_keys - reasoner_keys:
            ref_tensor = rubricator_batch.batch[key]
            shape = list(ref_tensor.shape)
            shape[0] = len(reasoner_batch)
            reasoner_batch.batch[key] = torch.zeros(shape, dtype=ref_tensor.dtype, device=ref_tensor.device)

        # --- Pad sequence dim to match ---
        r_seq = reasoner_batch.batch["input_ids"].shape[1]
        u_seq = rubricator_batch.batch["input_ids"].shape[1]
        pad_token_id = self.tokenizer.pad_token_id or 0

        if r_seq != u_seq:
            target_len = max(r_seq, u_seq)
            for batch in [reasoner_batch, rubricator_batch]:
                cur_len = batch.batch["input_ids"].shape[1]
                if cur_len < target_len:
                    pad_len = target_len - cur_len
                    for key in list(batch.batch.keys()):
                        tensor = batch.batch[key]
                        if key == "position_ids" and tensor.dim() == 3:
                            batch.batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=0)
                        elif tensor.dim() == 2 and tensor.shape[0] == len(batch):
                            if key == "input_ids":
                                batch.batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=pad_token_id)
                            else:
                                batch.batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=0)

        # --- Align non_tensor_batch keys ---
        r_nt_keys = set(reasoner_batch.non_tensor_batch.keys())
        u_nt_keys = set(rubricator_batch.non_tensor_batch.keys())
        common_keys = r_nt_keys & u_nt_keys

        for key in r_nt_keys - common_keys:
            del reasoner_batch.non_tensor_batch[key]
        for key in u_nt_keys - common_keys:
            del rubricator_batch.non_tensor_batch[key]

        for key in list(common_keys):
            r_val = reasoner_batch.non_tensor_batch[key]
            u_val = rubricator_batch.non_tensor_batch[key]
            if hasattr(r_val, 'shape') and hasattr(u_val, 'shape') and r_val.shape[1:] != u_val.shape[1:]:
                del reasoner_batch.non_tensor_batch[key]
                del rubricator_batch.non_tensor_batch[key]

        joint_batch = DataProto.concat([reasoner_batch, rubricator_batch])

        # Shuffle so rubricator samples distribute evenly across DP workers during chunk
        shuffle_indices = np.random.permutation(len(joint_batch))
        joint_batch = joint_batch[shuffle_indices]

        metrics["rlcer/train/dual_role_enabled"] = 1.0
        metrics["rlcer/train/rubricator_reward_mean"] = float(np.mean(rubricator_rewards))
        metrics["rlcer/train/rubricator_reward_max"] = float(np.max(rubricator_rewards))
        metrics["rlcer/train/rubricator_reward_min"] = float(np.min(rubricator_rewards))
        return joint_batch

    def _compute_rlcer_role_specific_advantages(self, batch: DataProto, metrics: dict) -> DataProto:
        """Compute advantages independently for each role in the joint batch.

        Splits by rlcer_role, computes GAE per role, scales rubricator advantages
        by rubricator_loss_weight, and restores original sample order.
        """
        role_arr = batch.non_tensor_batch.get("rlcer_role", None)
        if role_arr is None:
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )
            if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
                batch.batch["value_mask"] = compute_value_mask(batch)
            return batch

        is_rubricator = np.array([r == "rubricator" for r in role_arr])
        is_reasoner = ~is_rubricator
        n_rub = int(is_rubricator.sum())

        if n_rub == 0:
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                config=self.config.algorithm,
            )
            if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
                batch.batch["value_mask"] = compute_value_mask(batch)
            return batch

        rea_indices = np.where(is_reasoner)[0]
        rub_indices = np.where(is_rubricator)[0]

        reasoner_sub = batch[rea_indices]
        rubricator_sub = batch[rub_indices]

        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        adv_cfg = dict(
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        reasoner_sub = compute_advantage(reasoner_sub, **adv_cfg)

        # For rubricator: use REINFORCE (reward - mean) instead of GAE
        # because the critic was never trained on rubricator data and produces random values
        rub_rewards = rubricator_sub.batch["token_level_rewards"]
        rub_mask = rubricator_sub.batch["response_mask"]
        reward_per_sample = (rub_rewards * rub_mask).sum(dim=-1)
        reward_var = reward_per_sample.var().item() if len(reward_per_sample) > 1 else 0.0
        metrics["rlcer/train/rubricator_reward_var"] = reward_var

        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        rub_var_threshold = float(reward_cfg.get("rubricator_var_threshold", 0.01))
        if reward_var < rub_var_threshold:
            rubricator_sub.batch["advantages"] = torch.zeros_like(rub_rewards)
            rubricator_sub.batch["returns"] = rub_rewards.clone()
            metrics["rlcer/train/rubricator_adv_zeroed"] = 1.0
        else:
            # REINFORCE: advantage = reward - mean_reward, broadcast to token level
            mean_reward = reward_per_sample.mean()
            token_adv = torch.zeros_like(rub_rewards)
            # Place advantage on the last response token (where reward is)
            valid_lens = rub_mask.sum(dim=-1).long()
            for idx in range(len(rubricator_sub)):
                last_pos = max(valid_lens[idx].item() - 1, 0)
                token_adv[idx, last_pos] = reward_per_sample[idx] - mean_reward
            rubricator_sub.batch["advantages"] = masked_whiten(token_adv, rub_mask)
            rubricator_sub.batch["returns"] = rub_rewards.clone()
            metrics["rlcer/train/rubricator_adv_zeroed"] = 0.0

        if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
            reasoner_sub.batch["value_mask"] = compute_value_mask(reasoner_sub)
            rubricator_sub.batch["value_mask"] = compute_value_mask(rubricator_sub)

        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        rub_loss_weight = float(reward_cfg.get("rubricator_loss_weight", 0.5))
        if rub_loss_weight != 1.0 and "advantages" in rubricator_sub.batch:
            rubricator_sub.batch["advantages"] = (
                rubricator_sub.batch["advantages"] * rub_loss_weight
            )
        metrics["rlcer/train/rubricator_loss_weight"] = rub_loss_weight

        # --- Merge back in original order ---
        rea_keys = set(reasoner_sub.batch.keys())
        rub_keys = set(rubricator_sub.batch.keys())
        for key in rea_keys - rub_keys:
            ref = reasoner_sub.batch[key]
            shape = list(ref.shape)
            shape[0] = len(rubricator_sub)
            rubricator_sub.batch[key] = torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        for key in rub_keys - rea_keys:
            ref = rubricator_sub.batch[key]
            shape = list(ref.shape)
            shape[0] = len(reasoner_sub)
            reasoner_sub.batch[key] = torch.zeros(shape, dtype=ref.dtype, device=ref.device)

        rea_nt = set(reasoner_sub.non_tensor_batch.keys())
        rub_nt = set(rubricator_sub.non_tensor_batch.keys())
        for key in rea_nt - rub_nt:
            del reasoner_sub.non_tensor_batch[key]
        for key in rub_nt - rea_nt:
            del rubricator_sub.non_tensor_batch[key]

        joint_batch = DataProto.concat([reasoner_sub, rubricator_sub])

        original_order = np.concatenate([rea_indices, rub_indices])
        restore_perm = np.empty_like(original_order)
        restore_perm[original_order] = np.arange(len(original_order))

        for key in list(joint_batch.batch.keys()):
            tensor = joint_batch.batch[key]
            if tensor.shape[0] == len(joint_batch):
                joint_batch.batch[key] = tensor[restore_perm]
        for key in list(joint_batch.non_tensor_batch.keys()):
            arr = joint_batch.non_tensor_batch[key]
            if len(arr) == len(joint_batch):
                joint_batch.non_tensor_batch[key] = arr[restore_perm]

        metrics["rlcer/train/advantage_computed_per_role"] = 1.0
        return joint_batch

    def _build_rubric_recovery_batch(self, metrics: dict, timing_raw: dict):
        """Create an on-policy imitation batch from a healthy rubric replay."""
        if not self._rubric_replay_batches:
            metrics["rlcer/train/recovery_skipped_empty_replay"] = 1.0
            return None
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        replay_cfg = reward_cfg.get("rubric_replay", {})
        recovery_batch = deepcopy(self._rubric_replay_batches[-1])
        sample_size = max(1, int(replay_cfg.get("recovery_batch_size", 8)))
        if len(recovery_batch) > sample_size:
            indices = np.random.choice(len(recovery_batch), sample_size, replace=False)
            indices.sort()
            recovery_batch = recovery_batch[indices]
        recovery_batch.non_tensor_batch["rlcer_role"] = np.array(
            ["rubricator"] * len(recovery_batch), dtype=object
        )
        with marked_timer("rubric_recovery_log_prob", timing_raw, color="purple"):
            current = self.actor_rollout_wg.compute_log_prob(recovery_batch)
            recovery_batch.batch["old_log_probs"] = current.batch["old_log_probs"]
        mask = recovery_batch.batch["response_mask"].float()
        advantage = float(replay_cfg.get("recovery_advantage", 1.0))
        recovery_batch.batch["advantages"] = mask * advantage
        recovery_batch.batch["returns"] = torch.zeros_like(mask)
        metrics["rlcer/train/recovery_replay_active"] = 1.0
        metrics["rlcer/train/recovery_replay_batch_size"] = float(len(recovery_batch))
        return recovery_batch

    def _align_and_concat_role_batches(self, reasoner_batch: DataProto, rubricator_batch: DataProto) -> DataProto:
        """Align heterogeneous role tensors while keeping forwards separable downstream."""
        # Padding must not mutate the original reasoner object: its value
        # predictions were computed before rubricator construction and the
        # critic must see those original sequence dimensions.
        reasoner_batch = deepcopy(reasoner_batch)
        rubricator_batch = deepcopy(rubricator_batch)
        reasoner_keys = set(reasoner_batch.batch.keys())
        rubricator_keys = set(rubricator_batch.batch.keys())
        for key in reasoner_keys - rubricator_keys:
            ref_tensor = reasoner_batch.batch[key]
            shape = list(ref_tensor.shape)
            shape[0] = len(rubricator_batch)
            rubricator_batch.batch[key] = torch.zeros(shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
        for key in rubricator_keys - reasoner_keys:
            ref_tensor = rubricator_batch.batch[key]
            shape = list(ref_tensor.shape)
            shape[0] = len(reasoner_batch)
            reasoner_batch.batch[key] = torch.zeros(shape, dtype=ref_tensor.dtype, device=ref_tensor.device)

        pad_token_id = self.tokenizer.pad_token_id or 0
        # DataProto.concat requires every non-batch dimension of every common
        # TensorDict field to match.  Reasoner and rubricator have independently
        # padded prompts/responses, so aligning only ``input_ids`` is
        # insufficient (e.g. ``prompts`` can be 1000 vs 994 tokens).
        for key in set(reasoner_batch.batch.keys()) & set(rubricator_batch.batch.keys()):
            r_tensor = reasoner_batch.batch[key]
            u_tensor = rubricator_batch.batch[key]
            if r_tensor.dim() != u_tensor.dim():
                raise ValueError(
                    f"Cannot align role field {key!r}: rank {r_tensor.dim()} vs {u_tensor.dim()}"
                )
            if r_tensor.shape[1:] == u_tensor.shape[1:]:
                continue
            target_shape = tuple(max(a, b) for a, b in zip(r_tensor.shape[1:], u_tensor.shape[1:]))
            value = pad_token_id if key in {"input_ids", "prompts", "responses"} else 0
            for role_batch in (reasoner_batch, rubricator_batch):
                tensor = role_batch.batch[key]
                # torch.nn.functional.pad lists dimensions from last to first.
                padding = []
                for current, target in reversed(list(zip(tensor.shape[1:], target_shape))):
                    padding.extend((0, target - current))
                role_batch.batch[key] = torch.nn.functional.pad(tensor, tuple(padding), value=value)

        common_nt = set(reasoner_batch.non_tensor_batch) & set(rubricator_batch.non_tensor_batch)
        for key in set(reasoner_batch.non_tensor_batch) - common_nt:
            del reasoner_batch.non_tensor_batch[key]
        for key in set(rubricator_batch.non_tensor_batch) - common_nt:
            del rubricator_batch.non_tensor_batch[key]
        for key in list(common_nt):
            r_val = reasoner_batch.non_tensor_batch[key]
            u_val = rubricator_batch.non_tensor_batch[key]
            if hasattr(r_val, "shape") and hasattr(u_val, "shape") and r_val.shape[1:] != u_val.shape[1:]:
                del reasoner_batch.non_tensor_batch[key]
                del rubricator_batch.non_tensor_batch[key]
        joint_batch = DataProto.concat([reasoner_batch, rubricator_batch])
        return joint_batch[np.random.permutation(len(joint_batch))]

    def _append_rubricator_with_isolated_fwd(
        self,
        reasoner_batch: DataProto,
        reward_extra_infos_dict: dict,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        """Build rubricator batch with isolated forward pass and append to reasoner.

        Unlike the old shared_rho approach, this method:
        1. Checks signal gate (valid_ratio)
        2. Builds rubricator trajectory from cached generation tokens
        3. Computes old_log_probs on rubricator batch SEPARATELY
        4. Computes REINFORCE advantage for rubricator
        5. Concatenates with reasoner batch for actor update
        """
        reward_cfg = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
        enable_dual_role_update = reward_cfg.get("enable_dual_role_update", False)

        reasoner_batch.non_tensor_batch["rlcer_role"] = np.array(["reasoner"] * len(reasoner_batch), dtype=object)
        if not enable_dual_role_update:
            if getattr(self.rubricator_scheduler, "_health_state", "normal") == "recovery":
                recovery_batch = self._build_rubric_recovery_batch(metrics, timing_raw)
                if recovery_batch is not None:
                    metrics["rlcer/train/rubricator_update_mode"] = 2.0
                    return self._align_and_concat_role_batches(reasoner_batch, recovery_batch)
            return reasoner_batch

        update_interval = max(1, int(reward_cfg.get("rubricator_update_interval", 1)))
        if self.global_steps % update_interval != 0:
            metrics["rlcer/train/dual_role_skipped_update_interval"] = 1.0
            metrics["rlcer/train/rubricator_update_interval"] = float(update_interval)
            return reasoner_batch

        rubricator_rewards = reward_extra_infos_dict.get("rubricator_reward", None)
        if rubricator_rewards is None:
            metrics["rlcer/train/dual_role_skipped_no_rubricator_reward"] = 1.0
            return reasoner_batch
        if len(rubricator_rewards) != len(reasoner_batch):
            metrics["rlcer/train/dual_role_skipped_len_mismatch"] = 1.0
            return reasoner_batch

        # Signal gate: skip when rubricator has no valid signal
        valid_ratios = reward_extra_infos_dict.get("valid_ratio", None)
        if valid_ratios is not None:
            mean_valid_ratio = float(np.mean(valid_ratios))
            metrics["rlcer/train/mean_valid_ratio"] = mean_valid_ratio
            min_valid_ratio = float(reward_cfg.get("dual_role_min_valid_ratio", 0.01))
            if mean_valid_ratio < min_valid_ratio:
                metrics["rlcer/train/dual_role_skipped_no_signal"] = 1.0
                return reasoner_batch

        # Build rubricator batch from cached generation tokens
        rubricator_batch = self._build_rubricator_trajectory_batch(
            rubricator_rewards=rubricator_rewards,
            reasoner_batch=reasoner_batch,
            metrics=metrics,
        )

        if rubricator_batch is None:
            metrics["rlcer/train/dual_role_skipped_no_cache"] = 1.0
            return reasoner_batch

        # Subsample rubricator to limit compute
        rub_subsample = int(reward_cfg.get("rubricator_subsample_size", 0))
        if rub_subsample <= 0:
            rub_subsample = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        if len(rubricator_batch) > rub_subsample:
            indices = np.random.choice(len(rubricator_batch), rub_subsample, replace=False)
            indices.sort()
            rubricator_batch = rubricator_batch[indices]
            metrics["rlcer/train/rubricator_subsampled_to"] = float(rub_subsample)

        rubricator_batch.non_tensor_batch["rlcer_role"] = np.array(
            ["rubricator"] * len(rubricator_batch), dtype=object
        )

        # Compute old_log_probs on rubricator batch SEPARATELY (isolated forward pass)
        with marked_timer("rubricator_log_prob", timing_raw, color="purple"):
            rub_log_prob = self.actor_rollout_wg.compute_log_prob(rubricator_batch)
            rubricator_batch = rubricator_batch.union(rub_log_prob)

        # Compute REINFORCE advantage for rubricator
        rub_rewards = rubricator_batch.batch["token_level_rewards"]
        rub_mask = rubricator_batch.batch["response_mask"]
        reward_per_sample = (rub_rewards * rub_mask).sum(dim=-1)
        reward_var = reward_per_sample.var().item() if len(reward_per_sample) > 1 else 0.0
        metrics["rlcer/train/rubricator_reward_var"] = reward_var

        rub_var_threshold = float(reward_cfg.get("rubricator_var_threshold", 0.01))
        if reward_var < rub_var_threshold:
            rubricator_batch.batch["advantages"] = torch.zeros_like(rub_rewards)
            rubricator_batch.batch["returns"] = rub_rewards.clone()
            metrics["rlcer/train/rubricator_adv_zeroed"] = 1.0
        else:
            mean_reward = reward_per_sample.mean()
            token_adv = torch.zeros_like(rub_rewards)
            valid_lens = rub_mask.sum(dim=-1).long()
            for idx in range(len(rubricator_batch)):
                last_pos = max(valid_lens[idx].item() - 1, 0)
                token_adv[idx, last_pos] = reward_per_sample[idx] - mean_reward
            rubricator_batch.batch["advantages"] = masked_whiten(token_adv, rub_mask)
            rubricator_batch.batch["returns"] = rub_rewards.clone()
            metrics["rlcer/train/rubricator_adv_zeroed"] = 0.0

        # Scale rubricator advantages
        rub_loss_weight = float(reward_cfg.get("rubricator_loss_weight", 0.5))
        if rub_loss_weight != 1.0 and "advantages" in rubricator_batch.batch:
            rubricator_batch.batch["advantages"] = (
                rubricator_batch.batch["advantages"] * rub_loss_weight
            )
        metrics["rlcer/train/rubricator_loss_weight"] = rub_loss_weight
        metrics["rlcer/train/dual_role_enabled"] = 1.0
        metrics["rlcer/train/rubricator_update_interval"] = float(update_interval)
        metrics["rlcer/train/rubricator_reward_mean"] = float(np.mean(rubricator_rewards))

        replay_cfg = reward_cfg.get("rubric_replay", {})
        if bool(replay_cfg.get("enabled", False)):
            format_values = reward_extra_infos_dict.get("proposal_format_ok", [])
            nonempty_values = reward_extra_infos_dict.get("proposal_nonempty", [])
            replay_min_valid = float(replay_cfg.get("min_valid_ratio", 0.05))
            healthy = bool(
                format_values
                and nonempty_values
                and float(np.mean(format_values)) >= 0.95
                and float(np.mean(nonempty_values)) >= 0.95
                and float(np.mean(valid_ratios or [0.0])) >= replay_min_valid
            )
            if healthy:
                max_batches = max(1, int(replay_cfg.get("max_batches", 4)))
                self._rubric_replay_batches.append(deepcopy(rubricator_batch).to("cpu"))
                self._rubric_replay_batches = self._rubric_replay_batches[-max_batches:]
                metrics["rlcer/train/rubric_replay_size"] = float(len(self._rubric_replay_batches))

        # ============================================================
        # PLAN B (isolate_rubricator_update): run rubricator as a FULLY
        # INDEPENDENT second update_actor call instead of concatenating it
        # into the reasoner batch.  Both roles still evolve (true dual-role),
        # but they never share a forward pass / batch statistics, so the
        # actor-KL / forward-stat contamination that dragged traj_success to
        # 0.54 is eliminated while keeping the rubricator gradient (plan A
        # dropped it entirely; plan B keeps it, cleanly).
        #
        # Return signature changes to a tuple (reasoner_batch, rubricator_batch)
        # when enabled; trainer runs two independent update_actor calls.
        # rubricator advantages are already pre-scaled by rub_loss_weight above
        # (single scaling) — the vanilla single-role path in dp_actor applies NO
        # additional rub_w, so there is no double-scaling (bug 2 fixed).
        # ============================================================
        rubricator_update_mode = str(reward_cfg.get("rubricator_update_mode", "joint")).lower()
        if rubricator_update_mode in ("isolate", "isolated", "plan_b"):
            metrics["rlcer/train/rubricator_update_mode"] = 1.0  # isolated
            metrics["rlcer/train/rubricator_subsampled_to"] = float(len(rubricator_batch))
            # Mark rubricator batch as pure-rubricator (no rlcer_role mixing so
            # dp_actor takes the single-role vanilla path). reasoner_batch keeps
            # its all-"reasoner" rlcer_role for per-role metric attribution.
            rubricator_batch.non_tensor_batch["rlcer_role"] = np.array(
                ["rubricator"] * len(rubricator_batch), dtype=object
            )
            return reasoner_batch, rubricator_batch
        metrics["rlcer/train/rubricator_update_mode"] = 0.0  # joint (legacy)

        return self._align_and_concat_role_batches(reasoner_batch, rubricator_batch)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        custom_metrics_accumulator: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_images = []

        pad_token_id = self.tokenizer.pad_token_id
        skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            if not self.concat_multi_turn:
                # we need to create group_idx, traj_idx for each traj in no-concat mode
                num_traj_per_sample = self.config.actor_rollout_ref.rollout.val_kwargs.n
                self._assign_group_and_traj_idx(test_gen_batch, num_traj_per_sample)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            # In no-concat mode, save original uids before padding for filtering later
            if not self.concat_multi_turn:
                original_uids = set(test_gen_batch.non_tensor_batch["uid"])

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            if self.concat_multi_turn:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            else:
                # In no-concat mode, filter by uid since each input generates variable number of outputs
                # We need to keep only outputs whose uid is in the original (pre-padding) uid set
                valid_indices = [
                    i for i, uid in enumerate(test_output_gen_batch_padded.non_tensor_batch["group_idx"]) # uid in test_gen become group index in test_output_gen
                    if uid in original_uids
                ]
                test_output_gen_batch = test_output_gen_batch_padded.select_idxs(valid_indices)
                # Concatenate multi-turn trajectories into single entries
                test_output_gen_batch = concat_val_multi_turn(test_output_gen_batch, test_gen_batch,self.tokenizer)
                # after this, we can assume no-concat mode and concat_multi_turn can be handled equally


            print("validation generation end")
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # Store generated outputs
            
            inputs = test_batch.batch["prompts"]
            outputs = test_batch.batch["responses"]
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)
           
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)

            # Extract images from non_tensor_batch (extra_fields are stored there)
            if "image_data" in test_batch.non_tensor_batch:
                batch_images = test_batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))

            # RLCER: generate rubric proposals for validation batch
            val_reward_kwargs = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
            val_rubric_mode = str(val_reward_kwargs.get("rubricator", {}).get("mode", "")).lower()
            if val_rubric_mode == "policy":
                try:
                    val_rubric_raws = self._generate_rlcer_policy_rubrics_with_current_actor(test_batch)
                    if val_rubric_raws is not None and len(val_rubric_raws) == len(test_batch):
                        test_batch.non_tensor_batch["rlcer_policy_rubric_raw"] = np.array(val_rubric_raws, dtype=object)
                except Exception as e:
                    print(f"[RLCER] validation rubricator generation failed: {e}")

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Add token_level_scores to batch for custom metrics computation
            test_batch.batch["token_level_scores"] = reward_tensor

            # Compute custom metrics for validation
            custom_val_metrics = compute_custom_metrics(test_batch, prefix="custom_metrics")
            for metric_name, metric_value in custom_val_metrics.items():
                custom_metrics_accumulator[metric_name].append(metric_value)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        
        if self.config.trainer.get("replace_image_tokens_for_logging", False):
            sample_inputs = replace_image_tokens_for_logging(sample_inputs, processor=self.processor, tokenizer=self.tokenizer)
            sample_outputs = replace_image_tokens_for_logging(sample_outputs, processor=self.processor, tokenizer=self.tokenizer)
            
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, images=sample_images)
        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                images=sample_images,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        numeric_reward_extra = {
            k: v for k, v in reward_extra_infos_dict.items()
            if v and not isinstance(v[0], (dict, list))
        }
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, numeric_reward_extra)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else: 
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            # Cast numpy scalars to native float for downstream serialization.
            metric_dict["val-aux/num_turns/min"] = float(sample_turns.min())
            metric_dict["val-aux/num_turns/max"] = float(sample_turns.max())
            metric_dict["val-aux/num_turns/mean"] = float(sample_turns.mean())

        # Add aggregated custom metrics to metric_dict
        for metric_name, values in custom_metrics_accumulator.items():
            # Filter None / non-numeric: custom metric fns may return None on
            # some samples (e.g. first-turn / not-applicable fields in
            # frozenlake), and np.mean on a list with None raises
            # `unsupported operand type(s) for /: 'NoneType' and 'int'`.
            numeric_vals = [
                v for v in values
                if v is not None and isinstance(v, (int, float, np.integer, np.floating, np.bool_))
            ]
            if numeric_vals:
                metric_dict[f"custom_metrics/val/{metric_name.split('/')[-1]}"] = float(np.mean(numeric_vals))

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.concat_multi_turn = True # whether to concat history in async rollout --> if not, one traj has multiple prompt-response pairs
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            if self.config.trainer.get("concat_multi_turn", True):
                from verl.experimental.agent_loop import AgentLoopManager
            else:
                from .agent_loop.agent_loop_no_concat import AgentLoopManager
                self.concat_multi_turn = False
            self.async_rollout_manager = AgentLoopManager(
                    config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
                )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        compute_score_fn = getattr(self.reward_fn, "compute_score", None)
        tracker_state_getter = getattr(
            compute_score_fn, "get_tracker_state", None
        )
        if callable(tracker_state_getter):
            tracker_local_path = os.path.join(
                local_global_step_folder, "rlcer_tracker.pt"
            )
            torch.save(tracker_state_getter(), tracker_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        tracker_local_path = os.path.join(
            global_step_folder, "rlcer_tracker.pt"
        )
        compute_score_fn = getattr(self.reward_fn, "compute_score", None)
        tracker_state_loader = getattr(
            compute_score_fn, "load_tracker_state", None
        )
        if callable(tracker_state_loader) and os.path.exists(tracker_local_path):
            tracker_state_loader(
                torch.load(tracker_local_path, weights_only=False)
            )
            print(f"Loaded RLCER tracker state from {tracker_local_path}")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._flush_image_dumps()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if not self.concat_multi_turn:
                    # we need to create group_idx, traj_idx for each traj in no-concat mode
                    num_traj_per_sample = self.config.actor_rollout_ref.rollout.n
                    self._assign_group_and_traj_idx(gen_batch_output, num_traj_per_sample)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if not self.concat_multi_turn:
                            raise NotImplementedError("REMAX advantage estimation is not supported in no-concat mode yet.")
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    if self.concat_multi_turn:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        # In no-concat mode, each trajectory has multiple prompt-response pairs.
                        # We need to re-generate batch to align with gen_batch_output.
                        batch = self._post_process_no_concat_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        if not self.concat_multi_turn: # pad to divisor of dp_size
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # RLCER policy-sync rubricator: generate rubric proposals with current actor policy.
                        # Skip when pure_outcome mode is enabled (no rubric needed).
                        reward_kwargs = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
                        batch.non_tensor_batch["rlcer_global_step"] = np.full(
                            len(batch), self.global_steps, dtype=np.int64
                        )
                        if not reward_kwargs.get("pure_outcome", False):
                            try:
                                rubric_raws = self._generate_rlcer_policy_rubrics_with_current_actor(batch)
                                if rubric_raws is not None and len(rubric_raws) == len(batch):
                                    batch.non_tensor_batch["rlcer_policy_rubric_raw"] = np.array(rubric_raws, dtype=object)
                            except Exception as e:
                                metrics["rlcer/train/policy_rubricator_generation_failed"] = 1.0
                                print(f"[RLCER] policy rubricator generation failed: {e}")
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                    reward_kwargs = self.config.get("custom_reward_function", {}).get("reward_kwargs", {})
                    dual_role_strategy = str(reward_kwargs.get("dual_role_strategy", "post_advantage")).lower()
                    reasoner_critic_batch = None

                    if dual_role_strategy == "shared_rho":
                        # ============================================================
                        # SHARED RHO STRATEGY (ISOLATED FORWARD PASSES)
                        # Forward passes run on reasoner only. Rubricator batch is
                        # built separately with its own log_probs, then appended.
                        # ============================================================

                        # --- Resolve rewards early ---
                        with marked_timer("reward_retrieve", timing_raw, color="brown"):
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            _record_rlcer_reward_diagnostics(metrics, reward_extra_infos_dict)

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # --- Rubricator scheduler ---
                        if self.rubricator_scheduler._any_active:
                            self.rubricator_scheduler.step(
                                config=self.config,
                                global_steps=self.global_steps,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                metrics=metrics,
                                val_metrics=self._latest_val_metrics,
                            )
                            self._latest_val_metrics = None

                        # --- Forward passes on REASONER ONLY ---
                        if bypass_recomputing_logprobs:
                            from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction
                            apply_rollout_correction(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    from verl.utils.debug.metrics import calculate_debug_metrics
                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # --- compute_values on REASONER ONLY ---
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        # --- Compute reasoner advantage ---
                        with marked_timer("adv", timing_raw, color="brown"):
                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                metrics.update(is_metrics)

                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                        # --- Build and append rubricator batch with isolated forward pass ---
                        # Reset isolated rubricator state each step to avoid stale carry-over
                        # (plan-B safety: a previous step's rubricator batch must never leak
                        # into a step where the signal gate skipped rubricator generation).
                        self._isolated_rubricator_batch = None
                        reasoner_critic_batch = batch
                        _append_ret = self._append_rubricator_with_isolated_fwd(
                            reasoner_batch=batch,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            metrics=metrics,
                            timing_raw=timing_raw,
                        )
                        if isinstance(_append_ret, tuple):
                            # Plan B (isolate): (reasoner_batch, rubricator_batch)
                            batch, self._isolated_rubricator_batch = _append_ret
                        else:
                            # Legacy joint mode: single batch (rubricator already concatenated)
                            batch = _append_ret

                    else:
                        # ============================================================
                        # POST-ADVANTAGE STRATEGY (default, original 0605 flow)
                        # Forward passes on reasoner only, compute advantage on
                        # reasoner, then append rubricator batch after.
                        # ============================================================
                        if bypass_recomputing_logprobs:
                            from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                            apply_rollout_correction(
                                batch=batch,
                                rollout_corr_config=rollout_corr_config,
                                policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                            )
                        else:
                            with marked_timer("old_log_prob", timing_raw, color="blue"):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["entropys"]
                                response_masks = batch.batch["response_mask"]
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_agg = agg_loss(
                                    loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                                )
                                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                                metrics.update(old_log_prob_metrics)
                                old_log_prob.batch.pop("entropys")
                                batch = batch.union(old_log_prob)
                                if "rollout_log_probs" in batch.batch.keys():
                                    from verl.utils.debug.metrics import calculate_debug_metrics

                                    metrics.update(calculate_debug_metrics(batch))

                        assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                        if self.use_reference_policy:
                            with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            batch.batch["token_level_scores"] = reward_tensor
                            _record_rlcer_reward_diagnostics(metrics, reward_extra_infos_dict)

                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()})

                            if self.config.algorithm.use_kl_in_reward:
                                batch, kl_metrics = apply_kl_penalty(
                                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                            # --- Rubricator scheduler ---
                            if self.rubricator_scheduler._any_active:
                                self.rubricator_scheduler.step(
                                    config=self.config,
                                    global_steps=self.global_steps,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    metrics=metrics,
                                    val_metrics=self._latest_val_metrics,
                                )
                                self._latest_val_metrics = None

                            if (
                                rollout_corr_config is not None
                                and "rollout_log_probs" in batch.batch
                                and not bypass_recomputing_logprobs
                            ):
                                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                                metrics.update(is_metrics)

                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )

                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )

                            reasoner_critic_batch = batch
                            batch = self._build_rlcer_joint_training_batch(
                                reasoner_batch=batch,
                                reasoner_reward_tensor=reward_tensor,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                metrics=metrics,
                            )

                    if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first"]:
                        batch.batch["value_mask"] = compute_value_mask(batch)

                    if "rlcer_role" in batch.non_tensor_batch:
                        rub_w = float(reward_kwargs.get("rubricator_loss_weight", 0.0))
                        batch.meta_info["rubricator_loss_weight"] = rub_w
                        batch.meta_info["batch_has_dual_role"] = any(
                            r == "rubricator" for r in batch.non_tensor_batch["rlcer_role"]
                        )
                        # Optional per-role KL coef (validation-1 extension). Forwarded only when
                        # explicitly set, so default dual-role behaviour is unchanged otherwise.
                        _rub_kl_coef = reward_kwargs.get("rubricator_kl_loss_coef", None)
                        if _rub_kl_coef is not None:
                            batch.meta_info["rubricator_kl_loss_coef"] = float(_rub_kl_coef)

                    # compute custom metrics
                    with marked_timer("custom_metrics", timing_raw, color="magenta"):
                        custom_train_metrics = compute_custom_metrics(batch, prefix="custom_metrics/train")
                        metrics.update(custom_train_metrics)

                    
                    
                    # filter the training batch for effective update (Refer to STARPO-S and DAPO)
                    if self.config.filter.get("enable", False):
                        batch,metrics = FILTER_REGISTRY.get(self.config.filter.name)(batch, metrics,**self.config.filter.filter_kwargs)
                        if self.config.trainer.balance_batch:
                            # re-balance after filtering
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"After filtering: Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                            self._balance_batch(batch, metrics=metrics, logging_prefix="filtered_global_seqlen")
                    
                    # update critic — only on reasoner samples when dual-role is active
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_batch = reasoner_critic_batch if reasoner_critic_batch is not None else batch
                            if reasoner_critic_batch is None and "rlcer_role" in batch.non_tensor_batch:
                                rea_mask = np.array([r == "reasoner" for r in batch.non_tensor_batch["rlcer_role"]])
                                if not rea_mask.all():
                                    critic_batch = batch[np.where(rea_mask)[0]]
                            critic_output = self.critic_wg.update_critic(critic_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        anchor_probe_step = bool(
                            self._anchor_kl_enabled
                            and self.global_steps % self._anchor_kl_interval == 0
                        )
                        reasoner_probe = None
                        rubricator_probe = None
                        reasoner_pre_log_probs = None
                        rubricator_pre_log_probs = None
                        reasoner_after_reasoner_log_probs = None
                        rubricator_after_reasoner_log_probs = None
                        rub_batch = getattr(self, "_isolated_rubricator_batch", None)
                        protected_joint_rows = bool(
                            "rlcer_role" in batch.non_tensor_batch
                            and any(r == "rubricator" for r in batch.non_tensor_batch["rlcer_role"])
                        )
                        if anchor_probe_step:
                            if protected_joint_rows:
                                role_arr = np.asarray(batch.non_tensor_batch["rlcer_role"])
                                rea_idx = np.where(role_arr == "reasoner")[0][: self._anchor_kl_samples]
                                rub_idx = np.where(role_arr == "rubricator")[0][: self._anchor_kl_samples]
                                reasoner_probe = batch[rea_idx]
                                rubricator_probe = batch[rub_idx]
                            else:
                                reasoner_probe = batch[: min(len(batch), self._anchor_kl_samples)]
                            reasoner_pre_log_probs = reasoner_probe.batch["old_log_probs"].clone()
                            if rub_batch is not None:
                                rubricator_probe = rub_batch[: min(len(rub_batch), self._anchor_kl_samples)]
                            if rubricator_probe is not None:
                                rubricator_pre_log_probs = rubricator_probe.batch["old_log_probs"].clone()

                        # update actor — reasoner (clean, no rubricator contamination)
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            update_mode = str(
                                reward_kwargs.get("rubricator_update_mode", "joint")
                            ).lower()
                            has_rubricator_rows = protected_joint_rows
                            if update_mode in ("protected", "pcgrad", "single_step") and has_rubricator_rows:
                                batch.meta_info["rubricator_grad_norm_ratio_cap"] = float(
                                    reward_kwargs.get("rubricator_grad_norm_ratio_cap", 0.02)
                                )
                                actor_output = self.actor_rollout_wg.update_actor_protected_multirole(batch)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        if anchor_probe_step and reasoner_probe is not None and has_rubricator_rows:
                            with marked_timer("anchor_kl_after_protected_update", timing_raw, color="olive"):
                                rea_after = self.actor_rollout_wg.compute_log_prob(reasoner_probe)
                                rub_after = self.actor_rollout_wg.compute_log_prob(rubricator_probe)
                                metrics["rlcer/anchor_kl/reasoner_after_protected_update"] = _sampled_anchor_kl(
                                    reasoner_pre_log_probs,
                                    rea_after.batch["old_log_probs"],
                                    reasoner_probe.batch["response_mask"],
                                )
                                metrics["rlcer/anchor_kl/rubricator_after_protected_update"] = _sampled_anchor_kl(
                                    rubricator_pre_log_probs,
                                    rub_after.batch["old_log_probs"],
                                    rubricator_probe.batch["response_mask"],
                                )

                        if anchor_probe_step and reasoner_probe is not None and not has_rubricator_rows:
                            with marked_timer("anchor_kl_after_reasoner", timing_raw, color="olive"):
                                rea_after = self.actor_rollout_wg.compute_log_prob(reasoner_probe)
                                reasoner_after_reasoner_log_probs = rea_after.batch["old_log_probs"]
                                metrics["rlcer/anchor_kl/reasoner_after_reasoner"] = _sampled_anchor_kl(
                                    reasoner_pre_log_probs,
                                    reasoner_after_reasoner_log_probs,
                                    reasoner_probe.batch["response_mask"],
                                )
                                if rubricator_probe is not None:
                                    rub_after_rea = self.actor_rollout_wg.compute_log_prob(rubricator_probe)
                                    rubricator_after_reasoner_log_probs = rub_after_rea.batch["old_log_probs"]
                                    metrics["rlcer/anchor_kl/rubricator_after_reasoner"] = _sampled_anchor_kl(
                                        rubricator_pre_log_probs,
                                        rubricator_after_reasoner_log_probs,
                                        rubricator_probe.batch["response_mask"],
                                    )

                        # ============================================================
                        # PLAN B: rubricator as an INDEPENDENT second update_actor.
                        # Its batch was detached in _append (isolate mode) and stored
                        # on self._isolated_rubricator_batch.  Run a fully separate
                        # actor update so rubricator never shares a forward pass /
                        # batch-stat with reasoner (eliminates KL / forward-stat
                        # contamination) while STILL updating the shared policy with
                        # the rubricator REINFORCE gradient (plan A dropped it; plan B
                        # keeps it, cleanly).  Advantages are already pre-scaled by
                        # rub_loss_weight upstream, so the vanilla single-role path
                        # applies no extra scaling (no double-scaling).
                        # ============================================================
                        if rub_batch is not None:
                            with marked_timer("update_actor_rubricator", timing_raw, color="red"):
                                # Force single-role vanilla path: dp_actor branches on
                                # batch_has_dual_role, which must be False (rubricator
                                # batch has no reasoner rows to split against).
                                rub_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                rub_batch.meta_info["batch_has_dual_role"] = False
                                rub_batch.meta_info.pop("rubricator_loss_weight", None)
                                rub_batch.meta_info.pop("rubricator_kl_loss_coef", None)

                                # Bug-1 fix: vanilla path needs ref_log_prob when
                                # use_kl_loss=True (rubricator batch was never run
                                # through the ref policy). Compute it here so the
                                # rubricator update does not KeyError.
                                if self.config.actor_rollout_ref.actor.use_kl_loss and self.use_reference_policy:
                                    if "ref_log_prob" not in rub_batch.batch.keys():
                                        with marked_timer("rubricator_ref_log_prob", timing_raw, color="olive"):
                                            if not self.ref_in_actor:
                                                rub_ref = self.ref_policy_wg.compute_ref_log_prob(rub_batch)
                                            else:
                                                rub_ref = self.actor_rollout_wg.compute_ref_log_prob(rub_batch)
                                            rub_batch = rub_batch.union(rub_ref)

                                # Ensure batch size is divisible by DP world_size so the
                                # rubricator update distributes evenly across workers.
                                _rub_divisor = self.actor_rollout_wg.world_size
                                if len(rub_batch) % _rub_divisor != 0:
                                    rub_batch, _rub_pad = pad_dataproto_to_divisor(
                                        rub_batch, _rub_divisor
                                    )
                                rub_output = self.actor_rollout_wg.update_actor(rub_batch)
                            rub_metrics = reduce_metrics(rub_output.meta_info["metrics"])
                            # Prefix to avoid colliding with reasoner metrics in wandb.
                            for _k, _v in rub_metrics.items():
                                _suffix = _k.split("/", 1)[1] if "/" in _k else _k
                                metrics[f"actor_rub/{_suffix}"] = _v

                            if anchor_probe_step and reasoner_probe is not None:
                                with marked_timer("anchor_kl_after_rubricator", timing_raw, color="olive"):
                                    rea_after_rub = self.actor_rollout_wg.compute_log_prob(reasoner_probe)
                                    metrics["rlcer/anchor_kl/reasoner_after_rubricator"] = _sampled_anchor_kl(
                                        reasoner_after_reasoner_log_probs,
                                        rea_after_rub.batch["old_log_probs"],
                                        reasoner_probe.batch["response_mask"],
                                    )
                                    if rubricator_probe is not None:
                                        rub_after_rub = self.actor_rollout_wg.compute_log_prob(rubricator_probe)
                                        metrics["rlcer/anchor_kl/rubricator_after_rubricator"] = _sampled_anchor_kl(
                                            rubricator_after_reasoner_log_probs,
                                            rub_after_rub.batch["old_log_probs"],
                                            rubricator_probe.batch["response_mask"],
                                        )
                            self._isolated_rubricator_batch = None

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                        self._latest_val_metrics = val_metrics  # capture for rubricator scheduler
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                should_save_ckpt = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )
                should_upload_hf = self._hf_upload_manager.should_upload(self.global_steps)

                if should_save_ckpt or should_upload_hf:
                    # Flush pending HF uploads before saving to avoid conflicts
                    # with checkpoint deletion (max_actor_ckpt_to_keep)
                    self._hf_upload_manager.flush()
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                if should_upload_hf:
                    self._hf_upload_manager.maybe_upload(self.global_steps)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                data_metrics_batch = (
                    reasoner_critic_batch if reasoner_critic_batch is not None else batch
                )
                metrics.update(compute_data_metrics(batch=data_metrics_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._flush_image_dumps()
                    self._hf_upload_manager.flush()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
