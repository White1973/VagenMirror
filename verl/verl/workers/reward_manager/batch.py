# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        # Mark whether the compute_score is a custom (user-provided) reward function.
        # Custom reward functions (e.g. RLCER) should always be invoked even when
        # rm_scores from the environment already exist, because they may combine
        # the outcome reward with additional signals (cot_reward, rubricator_reward, etc.).
        self._is_custom_reward_fn = getattr(compute_score, '_is_custom_reward_fn', False)

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        prompts_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            prompts_str.append(prompt_str)

        ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        # Environment-side fields (for example traj_success, dino_score and
        # dreamsim_score) must be visible to the custom reward function, not
        # only merged into the returned logging dictionary after scoring.
        # RLCER uses these values as the grounded target z for rubric
        # validation.
        score_context_keys = set(reward_extra_keys) | {
            "traj_success",
            "dino_score",
            "dreamsim_score",
            "turn_env_states",
            "initial_env_state",
            "verifier_context",
        }

        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]
            extras[i]["prompt_str"] = prompts_str[i]
            for key in score_context_keys:
                if key in data.non_tensor_batch:
                    extras[i][key] = data.non_tensor_batch[key][i]
            # Pass environment reward (rm_scores) to custom reward functions so
            # they can use it as outcome reward instead of relying on
            # default_compute_score which only recognises built-in data sources.
            if "rm_scores" in data.batch.keys():
                prompt_len = data.batch["prompts"].shape[-1]
                resp_mask = data.batch["attention_mask"][i, prompt_len:].sum().item()
                if resp_mask > 0:
                    extras[i]["env_reward"] = float(data.batch["rm_scores"][i, resp_mask - 1].item())
            if "image_data" in data.non_tensor_batch:
                extras[i]["image_data"] = data.non_tensor_batch["image_data"][i]
            if "rlcer_policy_rubric_raw" in data.non_tensor_batch:
                extras[i]["rlcer_policy_rubric_raw"] = data.non_tensor_batch["rlcer_policy_rubric_raw"][i]
            if "uid" in data.non_tensor_batch:
                extras[i]["uid"] = data.non_tensor_batch["uid"][i]
            if "group_idx" in data.non_tensor_batch:
                extras[i]["group_idx"] = data.non_tensor_batch["group_idx"][i]
            if "traj_idx" in data.non_tensor_batch:
                extras[i]["traj_idx"] = data.non_tensor_batch["traj_idx"][i]
            if "turn_idx" in data.non_tensor_batch:
                extras[i]["turn_idx"] = data.non_tensor_batch["turn_idx"][i]
            if "turn_env_states" in data.non_tensor_batch:
                extras[i]["turn_env_states"] = data.non_tensor_batch["turn_env_states"][i]
            if "initial_env_state" in data.non_tensor_batch:
                extras[i]["initial_env_state"] = data.non_tensor_batch["initial_env_state"][i]
            if "verifier_context" in data.non_tensor_batch:
                extras[i]["verifier_context"] = data.non_tensor_batch["verifier_context"][i]
            # Optional, behavior-neutral metadata used only by sparse RLCER
            # mechanism auditing. It is injected by vagen/ray_trainer.py only
            # when rubric_audit.enabled=true.
            if "rlcer_global_step" in data.non_tensor_batch:
                extras[i]["rlcer_global_step"] = data.non_tensor_batch["rlcer_global_step"][i]

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score and compute_score is the default one, we directly
        # return rm score.  Custom reward functions (e.g. RLCER) are always invoked
        # because they may combine outcome reward with additional signals.
        if "rm_scores" in data.batch.keys() and not self._is_custom_reward_fn:
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: dict[str, Any] = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # Merge reward_extra_keys from agent loop (e.g. traj_success) into reward_extra_info
        # so that custom reward functions don't lose these fields.
        reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        for key in reward_extra_keys:
            if key not in reward_extra_info and key in data.non_tensor_batch:
                values = data.non_tensor_batch[key]
                reward_extra_info[key] = list(values) if hasattr(values, '__len__') else [values] * len(data)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
