# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from verl import DataProto
import torch
import numpy as np

class EpisodeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Vectorized reward placement
        prompt_length = data.batch['prompts'].shape[-1]
        attention_mask = data.batch['attention_mask']
        valid_response_lengths = attention_mask[:, prompt_length:].sum(dim=-1)  # (bs,)

        episode_rewards = np.array([data.non_tensor_batch['episode_rewards'][i] for i in range(len(data))])
        episode_lengths_arr = np.array([data.non_tensor_batch['episode_lengths'][i] for i in range(len(data))])

        if self.normalize_by_length:
            scores = episode_rewards / np.maximum(episode_lengths_arr, 1)
        else:
            scores = episode_rewards

        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=data.batch['responses'].device)
        indices = (valid_response_lengths - 1).clamp(min=0).long()
        reward_tensor[torch.arange(len(data), device=reward_tensor.device), indices] = scores_tensor

        # Sampling for logging
        if self.num_examine > 0:
            already_print_data_sources = {}
            for i in range(len(data)):
                data_source = data.non_tensor_batch['data_source'][i] if isinstance(data.non_tensor_batch['data_source'], (list, np.ndarray)) else data[i].non_tensor_batch['data_source']
                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                    already_print_data_sources[data_source] += 1
                    data_item = data[i]
                    prompt_ids = data_item.batch['prompts']
                    pl = prompt_ids.shape[-1]
                    valid_pl = data_item.batch['attention_mask'][:pl].sum()
                    valid_prompt_ids = prompt_ids[-valid_pl:]
                    response_ids = data_item.batch['responses']
                    valid_rl = data_item.batch['attention_mask'][pl:].sum()
                    valid_response_ids = response_ids[:valid_rl]
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                    print(f"[{data_source}][prompt]", prompt_str)
                    print(f"[{data_source}][response]", response_str)
                    print(f"[{data_source}][score]", scores[i])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor
