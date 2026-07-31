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

import numpy as np
import torch
from collections import defaultdict, Counter
from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


"""
Core functions to implement the GiGPO algorithm (https://arxiv.org/abs/2505.10978).
The function implemented in this file should be used by trainer with different distributed strategies to implement GiGPO.
"""

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        # Calculate returns
        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0
        
        # Calculate returns from the end to the start
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return
        
        # Store the results
        returns_by_traj[uid] = traj_returns
    
    # Recombine the returns into the original batch order (O(N) via precomputed mapping)
    all_returns = np.zeros_like(rewards)
    uid_to_positions = {}
    for i, uid in enumerate(traj_uids):
        uid_to_positions.setdefault(uid, []).append(i)
    for uid, positions in uid_to_positions.items():
        traj_returns = returns_by_traj[uid]
        for idx_in_traj, global_idx in enumerate(positions):
            all_returns[global_idx] = traj_returns[idx_in_traj]

    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   eta: float = 0.0,
                                   teacher_reward: torch.Tensor = None,
                                   wn_clip: float = 0.0,
                                   wn_sign: float = 1.0,
                                   wn_adaptive_clip: bool = False,
                                   wn_mode: str = "additive",
                                   student_log_probs: torch.Tensor = None,
                                   teacher_log_probs: torch.Tensor = None,
                                   wn_neg_reset: bool = False,
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages

    if eta > 0 and teacher_reward is not None:
        if wn_mode == "multiplicative" and student_log_probs is not None and teacher_log_probs is not None:
            from verl.trainer.ppo.rlsd_utils import compute_rlsd_token_advantage
            scores = compute_rlsd_token_advantage(
                seq_advantages=scores,
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                response_mask=response_mask,
                rlsd_lambda=eta,
                rlsd_clip_eps=wn_clip if wn_clip > 0 else 0.2,
                neg_reset=wn_neg_reset,
            )
        else:
            from verl.utils.torch_functional import masked_mean, masked_var
            with torch.no_grad():
                bsz = scores.shape[0]
                id2indices = defaultdict(list)
                for i in range(bsz):
                    id2indices[index[i]].append(i)
                median_strength = None
                if wn_adaptive_clip and wn_clip > 0:
                    valid_strengths = [scores[indices].abs().max() for indices in id2indices.values() if len(indices) > 1 and scores[indices].abs().max() >= 1e-6]
                    median_strength = torch.stack(valid_strengths).median() if valid_strengths else torch.tensor(1.0)
                for idx, indices in id2indices.items():
                    if len(indices) <= 1:
                        continue
                    group_strength = scores[indices].abs().max()
                    if group_strength < 1e-6:
                        continue
                    group_teacher = teacher_reward[indices]
                    group_mask = response_mask[indices]
                    group_scores = scores[indices]
                    if wn_neg_reset:
                        first_valid = group_mask.int().argmax(dim=-1)
                        sample_adv = group_scores[torch.arange(len(indices), device=group_scores.device), first_valid]
                        pos_mask = (sample_adv > 0).unsqueeze(-1)
                    raw_masked = group_teacher * group_mask
                    w_mean = masked_mean(raw_masked, group_mask)
                    w_var = masked_var(raw_masked, group_mask)
                    within_norm = (group_teacher - w_mean) * torch.rsqrt(w_var + 1e-8)
                    wn_contribution = wn_sign * eta * within_norm * group_mask
                    if wn_clip > 0:
                        effective_clip = wn_clip
                        if wn_adaptive_clip:
                            # Compress teacher clip when env signal is below batch median.
                            relative_strength = torch.clamp(group_strength / (median_strength + 1e-6), max=1.0)
                            effective_clip = wn_clip * relative_strength
                        wn_contribution = wn_contribution.clamp(-effective_clip, effective_clip)
                    if wn_neg_reset:
                        wn_contribution = wn_contribution * pos_mask
                    scores[indices] = scores[indices] + wn_contribution

    return scores, scores


def compute_gigpo_rtg_advantage(token_level_rewards: torch.Tensor,
                                step_rewards: torch.Tensor,
                                response_mask: torch.Tensor,
                                anchor_obs: np.array,
                                index: np.array,
                                traj_index: np.array,
                                epsilon: float = 1e-6,
                                step_advantage_w: float = 1.0,
                                mode: str = "mean_norm",
                                enable_similarity: bool = False,
                                similarity_thresh: float = 0.95,
                                eta: float = 0.0,
                                teacher_reward: torch.Tensor = None,
                                ):
    """GiGPO with token-level return-to-go for the episode branch.

    Episode branch: trajectory-level GRPO advantage plus optional per-token
    modulation scaled by eta.  When teacher_reward is provided, the per-token
    modulation uses the TVA-modulated teacher signal directly.
    Step branch: unchanged.
    """
    from verl.utils.torch_functional import masked_mean, masked_var
    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])

        bsz = returns.shape[0]
        episode_advantages = torch.zeros_like(returns)

        id2indices = defaultdict(list)
        for i in range(bsz):
            id2indices[index[i]].append(i)

        for idx, indices in id2indices.items():
            group_returns = returns[indices]
            group_mask = response_mask[indices]
            if len(indices) == 1:
                episode_advantages[indices] = torch.zeros_like(group_returns)
            else:
                traj_sum = (group_returns * group_mask).sum(dim=-1)
                traj_count = group_mask.sum(dim=-1).clamp(min=1)
                traj_mean = traj_sum / traj_count
                g_mean = traj_mean.mean()
                g_var = traj_mean.var()
                standard_adv = (traj_mean - g_mean) * torch.rsqrt(g_var + 1e-8)
                group_adv = standard_adv.unsqueeze(-1).expand_as(group_returns)

                if eta > 0:
                    if teacher_reward is not None:
                        raw = teacher_reward[indices]
                    else:
                        raw = group_returns - traj_mean.unsqueeze(-1)
                    raw_masked = raw * group_mask
                    w_mean = masked_mean(raw_masked, group_mask)
                    w_var = masked_var(raw_masked, group_mask)
                    within_norm = (raw - w_mean) * torch.rsqrt(w_var + 1e-8)
                    group_adv = group_adv + eta * within_norm

                episode_advantages[indices] = group_adv * group_mask

    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, returns


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        compute_mean_std_cross_steps: bool = True,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages

