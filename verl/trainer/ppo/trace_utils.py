"""
TRACE: Teacher-Reward Aligned Contrastive distillation for agentic Environments.

Key innovation: Teacher-side Contrastive Attribution (TCA).
- TCA gate (novel): measures whether TEACHER's logprob at each token position
  PREDICTS trajectory success within anchor state groups. This directly
  estimates teacher knowledge relevance — distinct from GCPO/DelTA which
  compute RL advantages.
- Confidence gate (from SDAR): teacher-student logprob gap.

The dual gate ensures we distill ONLY where:
1. Teacher's knowledge is reward-relevant (high TCA) — not just style
2. Teacher has information the student lacks (high confidence)

Distinct from:
- GCPO/DelTA: compute token-level RL advantages (single-turn)
- TRACE: computes token-level DISTILLATION gates (multi-turn agentic)
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Optional

from verl.trainer.ppo.core_algos import agg_loss


# ================================================================
# CORE INNOVATION: Skill-Conditional Teacher Contrastive Attribution
# ================================================================
#
# The key insight that differentiates TRACE from ALL existing methods:
#
# - ActFocus (2605.14558): Corr(student_logprob, reward) → RL reweighting
# - GCPO (2605.29198): Contrast under pos/neg prompts → RL advantages
# - SDAR: |teacher - student| → confidence gate (no reward check)
# - OPTO-Agent: step_adv × KL_share → indirect attribution
#
# TRACE: Corr(teacher_logprob - student_logprob, reward) within anchor state
# = "Does the teacher's SKILL-SPECIFIC knowledge predict trajectory success?"
#
# This is the ONLY method that directly answers:
# "Where does the teacher KNOW BETTER than the student, AND that
#  extra knowledge actually HELPS achieve reward?"
# ================================================================


def compute_skill_conditional_tca(
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Skill-Conditional Teacher Contrastive Attribution (SC-TCA).

    THE CORE INNOVATION OF TRACE.

    Measures: Corr(δ_{i,t}, R_i) for i ∈ G(s_k)
    where δ = teacher_logprob - student_logprob = "skill contribution"

    SC-TCA answers: "At this token position, does the teacher's ADDITIONAL
    knowledge (beyond what the student already knows) predict success?"

    - SC-TCA high: teacher's skill knowledge here → success → DISTILL HERE
    - SC-TCA ≈ 0: teacher's extra knowledge doesn't predict outcome → STYLE → SKIP
    - SC-TCA negative: teacher's preference here → FAILURE → SUPPRESS

    Why this is strictly better than alternatives:
    - vs Teacher-only TCA: removes base model knowledge (which student already has)
    - vs ActFocus: measures TEACHER's added value, not student's own behavior
    - vs SDAR confidence gate: checks reward-relevance, not just magnitude
    - vs KL-share heuristic: direct correlation, not proxy

    Args:
        teacher_log_probs: (bs, response_length) — teacher's per-token logprobs.
        student_log_probs: (bs, response_length) — student's per-token logprobs.
        response_mask: (bs, response_length) — valid token mask.
        step_group_ids: (bs,) — integer anchor state group IDs from GiGPO.
        step_rewards: (bs,) — per-step/trajectory rewards.

    Returns:
        sc_tca: (bs, response_length) — per-token SC-TCA scores (Pearson r).
    """
    bs, seq_len = teacher_log_probs.shape
    device = teacher_log_probs.device

    # scatter_add_ requires self.dtype == src.dtype (no auto-promotion);
    # response_mask may be bf16/bool/int, so cast it to float32 to match the
    # float32 accumulator tensors created by torch.zeros below.
    response_mask = response_mask.to(device=device, dtype=torch.float32)

    # Skill contribution = teacher - student (how much MORE teacher likes this token)
    delta = (teacher_log_probs - student_log_probs).detach()
    masked_delta = delta * response_mask

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1
    group_ids_exp = group_ids.unsqueeze(1).expand(bs, seq_len)

    # Group counts
    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    # Reward statistics per group
    group_reward_sum = torch.zeros(num_groups, device=device)
    group_reward_sq_sum = torch.zeros(num_groups, device=device)
    group_reward_sum.scatter_add_(0, group_ids, step_rewards)
    group_reward_sq_sum.scatter_add_(0, group_ids, step_rewards ** 2)
    group_reward_mean = group_reward_sum / group_count.clamp(min=1)
    group_reward_var = (group_reward_sq_sum / group_count.clamp(min=1)) - group_reward_mean ** 2
    group_reward_std = group_reward_var.clamp(min=0).sqrt().clamp(min=eps)

    # Delta (skill contribution) statistics per group per token
    group_delta_sum = torch.zeros(num_groups, seq_len, device=device)
    group_delta_sq_sum = torch.zeros(num_groups, seq_len, device=device)
    group_mask_sum = torch.zeros(num_groups, seq_len, device=device)

    group_delta_sum.scatter_add_(0, group_ids_exp, masked_delta)
    group_delta_sq_sum.scatter_add_(0, group_ids_exp, masked_delta ** 2)
    group_mask_sum.scatter_add_(0, group_ids_exp, response_mask)

    group_delta_mean = group_delta_sum / group_mask_sum.clamp(min=1)
    group_delta_var = (group_delta_sq_sum / group_mask_sum.clamp(min=1)) - group_delta_mean ** 2
    group_delta_std = group_delta_var.clamp(min=0).sqrt().clamp(min=eps)

    # Cross-correlation: Cov(delta_t, reward) per group per token
    centered_rewards = step_rewards - group_reward_mean[group_ids]
    centered_delta = (masked_delta - group_delta_mean[group_ids]) * response_mask

    cross_prod = centered_delta * centered_rewards.unsqueeze(1)
    group_cross_sum = torch.zeros(num_groups, seq_len, device=device)
    group_cross_sum.scatter_add_(0, group_ids_exp, cross_prod)

    # Pearson correlation
    group_cov = group_cross_sum / group_count.unsqueeze(1).clamp(min=1)
    group_corr = group_cov / (group_delta_std * group_reward_std.unsqueeze(1) + eps)
    group_corr = group_corr.clamp(-1.0, 1.0)

    # Only valid for groups >= 2
    valid = (group_count >= 2).unsqueeze(1).expand(num_groups, seq_len)
    group_corr = group_corr * valid.float()

    sc_tca = group_corr[group_ids] * response_mask
    return sc_tca


def compute_teacher_contrastive_attribution(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Teacher-side Contrastive Attribution (TCA) — TRACE's core innovation.

    Measures whether the TEACHER's logprob at each token position PREDICTS
    the trajectory's eventual success/failure within an anchor state group.

    TCA_t = Corr(teacher_logprob_{i,t}, reward_i) for i in G(s_k)

    Interpretation:
    - TCA high: teacher's confidence at this position correlates with success
      → teacher KNOWS the right answer here → DISTILL
    - TCA ≈ 0: teacher's confidence is uncorrelated with outcome
      → teacher's "knowledge" is just style preference → DON'T DISTILL
    - TCA negative: teacher's confidence ANTI-correlates with success
      → teacher gives WRONG advice here → SUPPRESS

    This is fundamentally different from GCPO/DelTA which compute RL
    advantages (what's important for the student). TCA measures TEACHER
    KNOWLEDGE QUALITY (where the teacher is actually helpful).

    Args:
        teacher_log_probs: (bs, response_length) — teacher's per-token logprobs.
        response_mask: (bs, response_length) — valid token mask.
        step_group_ids: (bs,) — integer anchor state group IDs.
        step_rewards: (bs,) — per-step/trajectory rewards.

    Returns:
        tca_scores: (bs, response_length) — per-token TCA (Pearson correlation).
            Positive = teacher is reward-relevant; Zero = style; Negative = harmful.
    """
    bs, seq_len = teacher_log_probs.shape
    device = teacher_log_probs.device

    # scatter_add_ requires self.dtype == src.dtype (no auto-promotion);
    # response_mask may be bf16/bool/int, so cast it to float32 to match the
    # float32 accumulator tensors created by torch.zeros below.
    response_mask = response_mask.to(device=device, dtype=torch.float32)

    logp = teacher_log_probs.detach()
    masked_logp = logp * response_mask

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1
    group_ids_exp = group_ids.unsqueeze(1).expand(bs, seq_len)

    # Group counts
    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    # Reward statistics per group
    group_reward_sum = torch.zeros(num_groups, device=device)
    group_reward_sq_sum = torch.zeros(num_groups, device=device)
    group_reward_sum.scatter_add_(0, group_ids, step_rewards)
    group_reward_sq_sum.scatter_add_(0, group_ids, step_rewards ** 2)

    group_reward_mean = group_reward_sum / group_count.clamp(min=1)
    group_reward_var = (group_reward_sq_sum / group_count.clamp(min=1)) - group_reward_mean ** 2
    group_reward_std = group_reward_var.clamp(min=0).sqrt().clamp(min=eps)

    # Teacher logprob statistics per group per token
    group_logp_sum = torch.zeros(num_groups, seq_len, device=device)
    group_logp_sq_sum = torch.zeros(num_groups, seq_len, device=device)
    group_mask_sum = torch.zeros(num_groups, seq_len, device=device)

    group_logp_sum.scatter_add_(0, group_ids_exp, masked_logp)
    group_logp_sq_sum.scatter_add_(0, group_ids_exp, masked_logp ** 2)
    group_mask_sum.scatter_add_(0, group_ids_exp, response_mask)

    group_logp_mean = group_logp_sum / group_mask_sum.clamp(min=1)
    group_logp_var = (group_logp_sq_sum / group_mask_sum.clamp(min=1)) - group_logp_mean ** 2
    group_logp_std = group_logp_var.clamp(min=0).sqrt().clamp(min=eps)

    # Cross-correlation: E[(logp - mean_logp)(reward - mean_reward)]
    centered_rewards = step_rewards - group_reward_mean[group_ids]  # (bs,)
    centered_logp = (masked_logp - group_logp_mean[group_ids]) * response_mask  # (bs, seq_len)

    cross_prod = centered_logp * centered_rewards.unsqueeze(1)  # (bs, seq_len)
    group_cross_sum = torch.zeros(num_groups, seq_len, device=device)
    group_cross_sum.scatter_add_(0, group_ids_exp, cross_prod)

    # Pearson correlation per group per token
    group_cov = group_cross_sum / group_count.unsqueeze(1).clamp(min=1)
    group_corr = group_cov / (group_logp_std * group_reward_std.unsqueeze(1) + eps)
    group_corr = group_corr.clamp(-1.0, 1.0)

    # Only valid for groups with size >= 2
    valid = (group_count >= 2).unsqueeze(1).expand(num_groups, seq_len)
    group_corr = group_corr * valid.float()

    # Map back to per-sample
    tca_scores = group_corr[group_ids] * response_mask
    return tca_scores


def compute_state_level_teacher_trust(
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    State-Level Teacher Trust (STT).

    Estimates overall teacher reliability at each anchor state by computing
    the correlation between the teacher's AVERAGE logprob for the entire step
    and the trajectory outcome.

    High STT = teacher is generally reliable at this state
    Low STT = teacher is unreliable → reduce distillation for the whole step

    Args:
        teacher_log_probs: (bs, response_length) — teacher's per-token logprobs.
        response_mask: (bs, response_length) — valid token mask.
        step_group_ids: (bs,) — integer anchor state group IDs.
        step_rewards: (bs,) — per-step/trajectory rewards.

    Returns:
        stt_scores: (bs,) — per-sample state-level trust score.
    """
    bs = teacher_log_probs.shape[0]
    device = teacher_log_probs.device

    # Average teacher logprob per step
    masked_logp = teacher_log_probs.detach() * response_mask
    step_mean_logp = masked_logp.sum(dim=-1) / response_mask.sum(dim=-1).clamp(min=1)  # (bs,)

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1

    # Group statistics
    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    group_logp_sum = torch.zeros(num_groups, device=device)
    group_logp_sq_sum = torch.zeros(num_groups, device=device)
    group_reward_sum = torch.zeros(num_groups, device=device)
    group_reward_sq_sum = torch.zeros(num_groups, device=device)
    group_cross_sum = torch.zeros(num_groups, device=device)

    group_logp_sum.scatter_add_(0, group_ids, step_mean_logp)
    group_logp_sq_sum.scatter_add_(0, group_ids, step_mean_logp ** 2)
    group_reward_sum.scatter_add_(0, group_ids, step_rewards)
    group_reward_sq_sum.scatter_add_(0, group_ids, step_rewards ** 2)

    group_logp_mean = group_logp_sum / group_count.clamp(min=1)
    group_reward_mean = group_reward_sum / group_count.clamp(min=1)

    centered_logp = step_mean_logp - group_logp_mean[group_ids]
    centered_reward = step_rewards - group_reward_mean[group_ids]
    group_cross_sum.scatter_add_(0, group_ids, centered_logp * centered_reward)

    group_logp_var = (group_logp_sq_sum / group_count.clamp(min=1)) - group_logp_mean ** 2
    group_reward_var = (group_reward_sq_sum / group_count.clamp(min=1)) - group_reward_mean ** 2
    group_logp_std = group_logp_var.clamp(min=0).sqrt().clamp(min=eps)
    group_reward_std = group_reward_var.clamp(min=0).sqrt().clamp(min=eps)

    group_cov = group_cross_sum / group_count.clamp(min=1)
    group_trust = group_cov / (group_logp_std * group_reward_std + eps)
    group_trust = group_trust.clamp(-1.0, 1.0)

    valid = group_count >= 2
    group_trust = group_trust * valid.float()

    stt_scores = group_trust[group_ids]
    return stt_scores


# ================================================================
# Student-side CTA (supplementary, for ablation comparison)
# ================================================================


def compute_contrastive_token_attribution(
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_uids: np.ndarray,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Contrastive Token Attribution (CTA) scores.

    For each token position t within a step group, CTA measures how much
    the student's log-probability at that position differs between
    trajectories with positive vs negative outcomes.

    High CTA = token position is causally important (action token)
    Low CTA  = token position is causally irrelevant (style token)

    Args:
        student_log_probs: (bs, response_length) — student's per-token log probs.
        response_mask: (bs, response_length) — valid token mask.
        step_group_uids: (bs,) — anchor state group UIDs from GiGPO.
            Trajectories sharing the same UID passed through the same state.
        step_rewards: (bs,) — step-level rewards for each trajectory-step.

    Returns:
        cta_scores: (bs, response_length) — per-token CTA z-scores.
    """
    bs, seq_len = student_log_probs.shape
    cta_scores = torch.zeros_like(student_log_probs)
    logp = student_log_probs.detach()
    mask = response_mask

    uid_to_indices = defaultdict(list)
    for i in range(bs):
        uid_to_indices[step_group_uids[i]].append(i)

    for uid, indices in uid_to_indices.items():
        if len(indices) < 2:
            continue

        idx = torch.tensor(indices, device=logp.device)
        group_rewards = step_rewards[idx]
        group_logp = logp[idx]
        group_mask = mask[idx]

        median_reward = group_rewards.median()
        pos_mask_flag = group_rewards >= median_reward
        neg_mask_flag = group_rewards < median_reward

        n_pos = pos_mask_flag.sum().item()
        n_neg = neg_mask_flag.sum().item()

        if n_pos == 0 or n_neg == 0:
            if group_rewards.unique().numel() <= 1:
                continue
            sorted_idx = group_rewards.argsort()
            half = len(sorted_idx) // 2
            neg_mask_flag = torch.zeros_like(pos_mask_flag)
            pos_mask_flag = torch.zeros_like(pos_mask_flag)
            neg_mask_flag[sorted_idx[:half]] = True
            pos_mask_flag[sorted_idx[half:]] = True
            n_pos = pos_mask_flag.sum().item()
            n_neg = neg_mask_flag.sum().item()
            if n_pos == 0 or n_neg == 0:
                continue

        masked_logp = group_logp * group_mask

        pos_mean = (masked_logp[pos_mask_flag] * group_mask[pos_mask_flag]).sum(dim=0) / (
            group_mask[pos_mask_flag].sum(dim=0).clamp(min=1)
        )
        neg_mean = (masked_logp[neg_mask_flag] * group_mask[neg_mask_flag]).sum(dim=0) / (
            group_mask[neg_mask_flag].sum(dim=0).clamp(min=1)
        )

        all_mean = (masked_logp * group_mask).sum(dim=0) / group_mask.sum(dim=0).clamp(min=1)
        all_var = ((masked_logp - all_mean.unsqueeze(0)) ** 2 * group_mask).sum(dim=0) / group_mask.sum(dim=0).clamp(min=1)
        all_std = all_var.clamp(min=0).sqrt().clamp(min=eps)

        cta = (pos_mean - neg_mean).abs() / (all_std + eps)

        any_valid = group_mask.sum(dim=0) > 0
        cta = cta * any_valid.float()

        for local_i, global_i in enumerate(indices):
            cta_scores[global_i] = cta * mask[global_i]

    return cta_scores


def compute_contrastive_token_attribution_vectorized(
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized CTA computation using scatter operations.

    Args:
        student_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        step_group_ids: (bs,) — integer group IDs (0, 1, 2, ...).
        step_rewards: (bs,) — per-step rewards.

    Returns:
        cta_scores: (bs, response_length) — per-token CTA z-scores.
    """
    bs, seq_len = student_log_probs.shape
    device = student_log_probs.device

    # scatter_add_ requires self.dtype == src.dtype (no auto-promotion);
    # response_mask may be bf16/bool/int, so cast it to float32 to match the
    # float32 accumulator tensors created by torch.zeros below.
    response_mask = response_mask.to(device=device, dtype=torch.float32)

    logp = student_log_probs.detach()
    masked_logp = logp * response_mask

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1

    group_median_reward = torch.zeros(num_groups, device=device)
    group_count = torch.zeros(num_groups, device=device)
    group_reward_sum = torch.zeros(num_groups, device=device)

    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))
    group_reward_sum.scatter_add_(0, group_ids, step_rewards)
    valid_groups = group_count > 0
    group_median_reward[valid_groups] = group_reward_sum[valid_groups] / group_count[valid_groups]

    per_sample_group_median = group_median_reward[group_ids]
    is_positive = (step_rewards >= per_sample_group_median).float()
    is_negative = 1.0 - is_positive

    group_pos_count = torch.zeros(num_groups, device=device)
    group_neg_count = torch.zeros(num_groups, device=device)
    group_pos_count.scatter_add_(0, group_ids, is_positive)
    group_neg_count.scatter_add_(0, group_ids, is_negative)

    group_ids_expanded = group_ids.unsqueeze(1).expand(bs, seq_len)
    mask_expanded = response_mask

    pos_weighted_logp = masked_logp * is_positive.unsqueeze(1)
    neg_weighted_logp = masked_logp * is_negative.unsqueeze(1)
    pos_mask_sum = mask_expanded * is_positive.unsqueeze(1)
    neg_mask_sum = mask_expanded * is_negative.unsqueeze(1)

    group_pos_logp_sum = torch.zeros(num_groups, seq_len, device=device)
    group_neg_logp_sum = torch.zeros(num_groups, seq_len, device=device)
    group_pos_mask_sum = torch.zeros(num_groups, seq_len, device=device)
    group_neg_mask_sum = torch.zeros(num_groups, seq_len, device=device)
    group_logp_sq_sum = torch.zeros(num_groups, seq_len, device=device)
    group_logp_sum = torch.zeros(num_groups, seq_len, device=device)
    group_mask_sum = torch.zeros(num_groups, seq_len, device=device)

    group_pos_logp_sum.scatter_add_(0, group_ids_expanded, pos_weighted_logp)
    group_neg_logp_sum.scatter_add_(0, group_ids_expanded, neg_weighted_logp)
    group_pos_mask_sum.scatter_add_(0, group_ids_expanded, pos_mask_sum)
    group_neg_mask_sum.scatter_add_(0, group_ids_expanded, neg_mask_sum)
    group_logp_sum.scatter_add_(0, group_ids_expanded, masked_logp)
    group_logp_sq_sum.scatter_add_(0, group_ids_expanded, masked_logp ** 2)
    group_mask_sum.scatter_add_(0, group_ids_expanded, mask_expanded)

    pos_mean = group_pos_logp_sum / (group_pos_mask_sum.clamp(min=1))
    neg_mean = group_neg_logp_sum / (group_neg_mask_sum.clamp(min=1))

    all_mean = group_logp_sum / (group_mask_sum.clamp(min=1))
    all_var = (group_logp_sq_sum / group_mask_sum.clamp(min=1)) - all_mean ** 2
    all_std = all_var.clamp(min=0).sqrt().clamp(min=eps)

    group_cta = (pos_mean - neg_mean).abs() / (all_std + eps)

    valid_mask = (group_pos_mask_sum > 0) & (group_neg_mask_sum > 0)
    group_cta = group_cta * valid_mask.float()

    cta_scores = group_cta[group_ids] * response_mask

    return cta_scores


def compute_soft_contrastive_attribution(
    student_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Soft CTA: Pearson correlation between token log-prob and reward within group.

    Works with any group size >= 2 (no hard positive/negative split needed).
    Backup Plan B: use when group sizes are small.

    Args:
        student_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        step_group_ids: (bs,) — integer group IDs.
        step_rewards: (bs,) — per-step rewards.

    Returns:
        cta_scores: (bs, response_length) — absolute Pearson correlation.
    """
    bs, seq_len = student_log_probs.shape
    device = student_log_probs.device

    # scatter_add_ requires self.dtype == src.dtype (no auto-promotion);
    # response_mask may be bf16/bool/int, so cast it to float32 to match the
    # float32 accumulator tensors created by torch.zeros below.
    response_mask = response_mask.to(device=device, dtype=torch.float32)

    logp = student_log_probs.detach()
    masked_logp = logp * response_mask

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1

    group_ids_exp = group_ids.unsqueeze(1).expand(bs, seq_len)

    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    group_reward_sum = torch.zeros(num_groups, device=device)
    group_reward_sq_sum = torch.zeros(num_groups, device=device)
    group_reward_sum.scatter_add_(0, group_ids, step_rewards)
    group_reward_sq_sum.scatter_add_(0, group_ids, step_rewards ** 2)

    group_reward_mean = group_reward_sum / group_count.clamp(min=1)
    group_reward_var = (group_reward_sq_sum / group_count.clamp(min=1)) - group_reward_mean ** 2
    group_reward_std = group_reward_var.clamp(min=0).sqrt().clamp(min=eps)

    group_logp_sum = torch.zeros(num_groups, seq_len, device=device)
    group_logp_sq_sum = torch.zeros(num_groups, seq_len, device=device)
    group_mask_sum = torch.zeros(num_groups, seq_len, device=device)
    group_logp_sum.scatter_add_(0, group_ids_exp, masked_logp)
    group_logp_sq_sum.scatter_add_(0, group_ids_exp, masked_logp ** 2)
    group_mask_sum.scatter_add_(0, group_ids_exp, response_mask)

    group_logp_mean = group_logp_sum / group_mask_sum.clamp(min=1)
    group_logp_var = (group_logp_sq_sum / group_mask_sum.clamp(min=1)) - group_logp_mean ** 2
    group_logp_std = group_logp_var.clamp(min=0).sqrt().clamp(min=eps)

    centered_rewards = step_rewards - group_reward_mean[group_ids]
    centered_logp = (masked_logp - group_logp_mean[group_ids]) * response_mask

    cross_prod = centered_logp * centered_rewards.unsqueeze(1)
    group_cross_sum = torch.zeros(num_groups, seq_len, device=device)
    group_cross_sum.scatter_add_(0, group_ids_exp, cross_prod)

    group_cov = group_cross_sum / group_count.unsqueeze(1).clamp(min=1)
    group_corr = group_cov.abs() / (
        group_logp_std * group_reward_std.unsqueeze(1) + eps
    )
    group_corr = group_corr.clamp(max=1.0)

    valid = (group_count >= 2).unsqueeze(1).expand(num_groups, seq_len)
    group_corr = group_corr * valid.float()

    cta_scores = group_corr[group_ids] * response_mask
    return cta_scores


def compute_trace_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    cta_scores: torch.Tensor,
    gate_beta: float = 5.0,
    tau: float = 2.0,
    weight_min: float = 0.05,
    use_dual_gate: bool = True,
    use_adaptive_tau: bool = True,
    tau_running_std: Optional[float] = None,
    beta_running_std: Optional[float] = None,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    TRACE contrastive-gated token distillation loss.

    L = agg( w_t * (log_teacher - log_student) )
    w_t = g_cta * g_conf   (dual gate)

    where:
        g_cta  = sigmoid(tau * CTA_t)         — contrastive attribution gate
        g_conf = sigmoid(beta * delta_t)      — confidence gate (from SDAR)

    Args:
        student_log_probs: (bs, response_length) — current policy log probs.
        teacher_log_probs: (bs, response_length) — teacher log probs (no grad).
        response_mask: (bs, response_length) — valid token mask.
        cta_scores: (bs, response_length) — pre-computed CTA scores.
        gate_beta: temperature for confidence gate.
        tau: temperature for CTA gate.
        weight_min: minimum gate value.
        use_dual_gate: if False, CTA gate only (ablation).
        use_adaptive_tau: auto-normalize tau by CTA running std.
        tau_running_std: externally tracked running std of CTA scores.
        beta_running_std: externally tracked running std of delta_t.
        loss_agg_mode: aggregation mode.

    Returns:
        loss: scalar loss.
        metrics: diagnostic statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()
    if use_adaptive_tau and beta_running_std is not None and beta_running_std > 0:
        effective_beta = gate_beta / (beta_running_std + 1e-6)
    else:
        effective_beta = gate_beta
    g_conf = torch.sigmoid(effective_beta * delta_t).detach()

    if use_adaptive_tau and tau_running_std is not None and tau_running_std > 0:
        effective_tau = tau / (tau_running_std + 1e-6)
    else:
        effective_tau = tau
    g_cta = torch.sigmoid(effective_tau * cta_scores.detach())

    if use_dual_gate:
        w_t = (g_cta * g_conf).clamp(min=weight_min, max=1.0)
    else:
        w_t = g_cta.clamp(min=weight_min, max=1.0)

    kl_per_token = teacher_log_probs - student_log_probs
    gated_kl = w_t * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        g_cta_mean = (g_cta * response_mask).sum() / mask_sum
        g_conf_mean = (g_conf * response_mask).sum() / mask_sum
        dual_gate_mean = (w_t * response_mask).sum() / mask_sum
        dual_gate_active = ((w_t > 0.5).float() * response_mask).sum() / mask_sum
        cta_mean = (cta_scores * response_mask).sum() / mask_sum
        cta_std = ((cta_scores * response_mask) ** 2).sum() / mask_sum - cta_mean ** 2
        cta_std = cta_std.clamp(min=0).sqrt()
        delta_mean = (delta_t * response_mask).sum() / mask_sum
        delta_std = (((delta_t * response_mask) ** 2).sum() / mask_sum - delta_mean ** 2).clamp(min=0).sqrt()

    metrics = {
        "trace/g_cta_mean": g_cta_mean.item(),
        "trace/g_conf_mean": g_conf_mean.item(),
        "trace/dual_gate_mean": dual_gate_mean.item(),
        "trace/dual_gate_active_ratio": dual_gate_active.item(),
        "trace/cta_mean": cta_mean.item(),
        "trace/cta_std": cta_std.item(),
        "trace/delta_std": delta_std.item(),
        "trace/effective_tau": effective_tau if isinstance(effective_tau, float) else effective_tau.item(),
        "trace/loss": loss.detach().item(),
    }

    return loss, metrics


def compute_trace_loss_with_fallback(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    step_advantages: torch.Tensor,
    step_boundary_mask: Optional[torch.Tensor] = None,
    gate_beta: float = 5.0,
    tau: float = 2.0,
    weight_min: float = 0.05,
    min_group_size: int = 3,
    use_soft_cta: bool = False,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    TRACE loss with automatic fallback to KL-share attribution for small groups.

    When anchor state group size >= min_group_size: use CTA (direct causal).
    When group size < min_group_size: fallback to OPTO-Agent's KL-share.

    Args:
        student_log_probs: (bs, response_length)
        teacher_log_probs: (bs, response_length)
        response_mask: (bs, response_length)
        step_group_ids: (bs,) — integer anchor state group IDs.
        step_rewards: (bs,) — per-step rewards.
        step_advantages: (bs, response_length) — step-level advantages (for fallback).
        step_boundary_mask: (bs, response_length) — step boundary IDs (for fallback).
        gate_beta: confidence gate temperature.
        tau: CTA/reward gate temperature.
        weight_min: minimum gate value.
        min_group_size: minimum group size for CTA.
        use_soft_cta: use Pearson correlation instead of hard split.
        loss_agg_mode: aggregation mode.

    Returns:
        loss: scalar loss.
        metrics: diagnostic statistics.
    """
    bs, seq_len = student_log_probs.shape
    device = student_log_probs.device

    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1
    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))

    per_sample_group_size = group_count[group_ids]
    use_cta_mask = (per_sample_group_size >= min_group_size).float()
    use_fallback_mask = 1.0 - use_cta_mask

    cta_ratio = use_cta_mask.sum().item() / bs
    fallback_ratio = use_fallback_mask.sum().item() / bs

    if use_soft_cta:
        cta_scores = compute_soft_contrastive_attribution(
            student_log_probs, response_mask, step_group_ids, step_rewards
        )
    else:
        cta_scores = compute_contrastive_token_attribution_vectorized(
            student_log_probs, response_mask, step_group_ids, step_rewards
        )

    from verl.trainer.ppo.opto_agent_utils import compute_step_kl_share_vectorized

    teacher_log_probs_detached = teacher_log_probs.detach()
    delta_t = teacher_log_probs_detached - student_log_probs.detach()
    token_kl_fallback = delta_t.clamp(min=0)

    if step_boundary_mask is not None:
        kl_share = compute_step_kl_share_vectorized(
            token_kl_fallback, response_mask, step_boundary_mask
        )
    else:
        kl_sum = (token_kl_fallback * response_mask).sum(dim=-1, keepdim=True) + 1e-8
        kl_share = (token_kl_fallback * response_mask) / kl_sum

    fallback_attr = step_advantages * kl_share

    g_cta = torch.sigmoid(tau * cta_scores.detach())
    g_fallback = torch.sigmoid(tau * fallback_attr.detach())

    blended_gate = (
        use_cta_mask.unsqueeze(1) * g_cta
        + use_fallback_mask.unsqueeze(1) * g_fallback
    )

    g_conf = torch.sigmoid(gate_beta * delta_t).detach()
    w_t = (blended_gate * g_conf).clamp(min=weight_min, max=1.0)

    kl_per_token = teacher_log_probs_detached - student_log_probs
    gated_kl = w_t * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        metrics = {
            "trace/g_cta_mean": (g_cta * response_mask).sum().item() / mask_sum.item(),
            "trace/g_conf_mean": (g_conf * response_mask).sum().item() / mask_sum.item(),
            "trace/g_fallback_mean": (g_fallback * response_mask).sum().item() / mask_sum.item(),
            "trace/dual_gate_mean": (w_t * response_mask).sum().item() / mask_sum.item(),
            "trace/dual_gate_active_ratio": ((w_t > 0.5).float() * response_mask).sum().item() / mask_sum.item(),
            "trace/cta_ratio": cta_ratio,
            "trace/fallback_ratio": fallback_ratio,
            "trace/cta_mean": (cta_scores * response_mask).sum().item() / mask_sum.item(),
            "trace/loss": loss.detach().item(),
        }

    return loss, metrics


# ================================================================
# PRIMARY API: Teacher-CTA-based TRACE Loss (recommended entry point)
# ================================================================


def compute_trace_tca_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: torch.Tensor,
    step_rewards: torch.Tensor,
    gate_beta: float = 5.0,
    tau: float = 3.0,
    weight_min: float = 0.05,
    use_stt: bool = True,
    stt_temperature: float = 2.0,
    min_group_size: int = 2,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    TRACE primary loss: Teacher-CTA gated distillation.

    Computes:
    1. Teacher-side Contrastive Attribution (TCA) — per-token
    2. State-Level Teacher Trust (STT) — per-step (optional)
    3. Confidence gate (from SDAR) — per-token
    4. Combined gated KL distillation loss

    Key difference from GCPO/DelTA:
    - They compute token-level RL advantages for policy gradient
    - We compute token-level DISTILLATION GATES for selective KD
    - We use TEACHER logprobs to measure teacher knowledge relevance
    - We leverage multi-turn anchor state structure

    Args:
        student_log_probs: (bs, response_length) — student logprobs. Retains grad.
        teacher_log_probs: (bs, response_length) — teacher logprobs. No grad.
        response_mask: (bs, response_length) — valid token mask.
        step_group_ids: (bs,) — integer anchor state group IDs from GiGPO.
        step_rewards: (bs,) — per-step/trajectory rewards.
        gate_beta: temperature for confidence gate.
        tau: temperature for TCA gate.
        weight_min: minimum gate value.
        use_stt: whether to apply State-Level Teacher Trust.
        stt_temperature: temperature for STT sigmoid.
        min_group_size: minimum group size for TCA.
        loss_agg_mode: aggregation mode.

    Returns:
        loss: scalar loss.
        metrics: diagnostic statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()
    bs = student_log_probs.shape[0]
    device = student_log_probs.device

    # --- Teacher-side Contrastive Attribution (core innovation) ---
    group_ids = step_group_ids.long()
    num_groups = group_ids.max().item() + 1
    group_count = torch.zeros(num_groups, device=device)
    group_count.scatter_add_(0, group_ids, torch.ones(bs, device=device))
    per_sample_group_size = group_count[group_ids]

    valid_for_tca = (per_sample_group_size >= min_group_size).float()

    # Use SC-TCA (skill-conditional) as primary signal — the core innovation
    tca_scores = compute_skill_conditional_tca(
        teacher_log_probs, student_log_probs.detach(),
        response_mask, step_group_ids, step_rewards
    )

    # TCA gate: |TCA| indicates relevance (positive or negative both informative)
    g_tca = torch.sigmoid(tau * tca_scores.abs()).detach()
    g_tca = g_tca * valid_for_tca.unsqueeze(1) + 0.5 * (1 - valid_for_tca.unsqueeze(1))
    g_tca = g_tca * response_mask

    # --- State-Level Teacher Trust (optional) ---
    if use_stt:
        stt = compute_state_level_teacher_trust(
            teacher_log_probs, response_mask, step_group_ids, step_rewards
        )
        stt_gate = torch.sigmoid(stt_temperature * stt.clamp(min=0)).detach()
        stt_gate = stt_gate * (per_sample_group_size >= min_group_size).float() + \
                   0.7 * (per_sample_group_size < min_group_size).float()
    else:
        stt_gate = torch.ones(bs, device=device)

    # --- Confidence gate (from SDAR) ---
    delta_t = teacher_log_probs - student_log_probs.detach()
    g_conf = torch.sigmoid(gate_beta * delta_t).detach()

    # --- Combined gate: TCA × conf × STT ---
    w_t = (g_tca * g_conf * stt_gate.unsqueeze(1)).clamp(min=weight_min, max=1.0)

    # Suppress anti-correlated teacher positions
    anti_corr = (tca_scores < -0.1).float() * valid_for_tca.unsqueeze(1)
    w_t = w_t * (1.0 - 0.8 * anti_corr)
    w_t = w_t.clamp(min=weight_min)

    # --- Gated KL loss ---
    kl_per_token = teacher_log_probs - student_log_probs
    gated_kl = w_t * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # --- Metrics ---
    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        tca_mean = (tca_scores.abs() * response_mask).sum() / mask_sum
        tca_pos = ((tca_scores > 0.1).float() * response_mask).sum() / mask_sum
        tca_neg = ((tca_scores < -0.1).float() * response_mask).sum() / mask_sum

    metrics = {
        "trace/tca_abs_mean": tca_mean.item(),
        "trace/tca_positive_ratio": tca_pos.item(),
        "trace/tca_negative_ratio": tca_neg.item(),
        "trace/g_tca_mean": (g_tca * response_mask).sum().item() / mask_sum.item(),
        "trace/g_conf_mean": (g_conf * response_mask).sum().item() / mask_sum.item(),
        "trace/stt_mean": stt_gate.mean().item() if use_stt else 1.0,
        "trace/dual_gate_mean": (w_t * response_mask).sum().item() / mask_sum.item(),
        "trace/dual_gate_active": ((w_t > 0.5).float() * response_mask).sum().item() / mask_sum.item(),
        "trace/valid_group_ratio": valid_for_tca.mean().item(),
        "trace/loss": loss.detach().item(),
    }

    return loss, metrics
