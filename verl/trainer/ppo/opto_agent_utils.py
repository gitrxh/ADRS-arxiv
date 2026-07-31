"""
OPTO-Agent: Step-Level Reward-Attributed Token Optimization for Agentic RL.

Dual-gated token distillation: confidence gate × reward-attribution gate.
- Confidence gate (from SDAR): tokens where teacher is more confident
- Reward-attribution gate (novel): tokens whose KL contributes to step reward

The dual gate ensures we only distill tokens that are BOTH:
1. Where the teacher has useful knowledge (high confidence)
2. Where that knowledge is reward-relevant (high step advantage × KL share)
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_step_kl_share(
    token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    step_boundary_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute KL share within each step (normalized KL contribution per token).

    For each token t in step k: kl_share_t = KL_t / sum(KL in step k)

    Args:
        token_kl: (bs, response_length) — per-token KL values.
        response_mask: (bs, response_length) — valid token mask.
        step_boundary_mask: (bs, response_length) — integer step IDs (0, 1, 2, ...)
            for each token, indicating which step it belongs to.

    Returns:
        kl_share: (bs, response_length) — normalized KL share within each step.
    """
    bs, seq_len = token_kl.shape
    masked_kl = token_kl * response_mask

    kl_share = torch.zeros_like(token_kl)

    for b in range(bs):
        step_ids = step_boundary_mask[b]
        unique_steps = step_ids[response_mask[b].bool()].unique()

        for step_id in unique_steps:
            step_mask = (step_ids == step_id) & response_mask[b].bool()
            step_kl_sum = masked_kl[b][step_mask].sum() + eps
            kl_share[b][step_mask] = masked_kl[b][step_mask] / step_kl_sum

    return kl_share


def compute_step_kl_share_vectorized(
    token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    step_boundary_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Vectorized version of step-internal KL share computation.
    Uses scatter_add for efficiency on GPU.

    Args:
        token_kl: (bs, response_length) — per-token KL values.
        response_mask: (bs, response_length) — valid token mask.
        step_boundary_mask: (bs, response_length) — integer step IDs.

    Returns:
        kl_share: (bs, response_length) — normalized KL within each step.
    """
    bs, seq_len = token_kl.shape
    masked_kl = token_kl * response_mask

    max_steps = int(step_boundary_mask.max().item()) + 1

    step_kl_sums = torch.zeros(bs, max_steps, device=token_kl.device)
    step_kl_sums.scatter_add_(1, step_boundary_mask.long(), masked_kl)

    per_token_step_sum = step_kl_sums.gather(1, step_boundary_mask.long()) + eps

    kl_share = masked_kl / per_token_step_sum
    kl_share = kl_share * response_mask

    return kl_share


def compute_opto_agent_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_advantages: torch.Tensor,
    step_boundary_mask: torch.Tensor = None,
    gate_beta: float = 5.0,
    tau: float = 2.0,
    weight_min: float = 0.05,
    use_dual_gate: bool = True,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    OPTO-Agent dual-gated token distillation loss.

    L = agg( w_t * (log_teacher - log_student) )
    w_t = g_conf * g_rew   (dual gate)

    where:
        g_conf = sigmoid(beta * delta_t)          — confidence gate
        g_rew  = sigmoid(tau * step_adv * kl_share) — reward-attribution gate

    Args:
        student_log_probs: (bs, response_length) — current policy log probs. Retains grad.
        teacher_log_probs: (bs, response_length) — teacher (skill-augmented) log probs. No grad.
        response_mask: (bs, response_length) — valid token mask.
        step_advantages: (bs, response_length) — step-level advantage for each token
            (from GiGPO; same value for all tokens within a step).
        step_boundary_mask: (bs, response_length) — integer step IDs per token.
            If None, treats entire response as one step (falls back to episode-level).
        gate_beta: temperature for confidence gate sigmoid.
        tau: temperature for reward-attribution gate sigmoid.
        weight_min: minimum gate value (prevents complete gradient death).
        use_dual_gate: if False, only uses reward gate (ablation).
        loss_agg_mode: how to aggregate the loss.

    Returns:
        loss: scalar loss.
        metrics: dict with diagnostic statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    # --- Confidence gate (from SDAR) ---
    delta_t = teacher_log_probs - student_log_probs.detach()
    g_conf = torch.sigmoid(gate_beta * delta_t).detach()

    # --- Token-level KL (base signal for attribution) ---
    token_kl = delta_t.clamp(min=0)  # positive part = teacher more confident

    # --- Reward-attribution gate ---
    if step_boundary_mask is not None:
        kl_share = compute_step_kl_share_vectorized(
            token_kl, response_mask, step_boundary_mask
        )
        # Token-level reward attribution: step advantage × KL share within step
        token_reward_attr = step_advantages * kl_share
    else:
        # Completion-level mode: use advantage directly (no KL decomposition)
        # Get per-sequence advantage (first valid token's value)
        seq_adv = step_advantages[:, 0:1]  # (bs, 1) — same for all tokens
        # Scale by normalized token KL to differentiate within sequence
        kl_sum = (token_kl * response_mask).sum(dim=-1, keepdim=True) + 1e-8
        kl_normalized = token_kl / (kl_sum / response_mask.sum(dim=-1, keepdim=True).clamp(min=1))
        token_reward_attr = seq_adv * kl_normalized

    g_rew = torch.sigmoid(tau * token_reward_attr).detach()

    # --- Dual gate ---
    if use_dual_gate:
        w_t = (g_conf * g_rew).clamp(min=weight_min, max=1.0)
    else:
        w_t = g_rew.clamp(min=weight_min, max=1.0)

    # --- Gated KL loss ---
    kl_per_token = teacher_log_probs - student_log_probs
    gated_kl = w_t * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # --- Metrics ---
    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        g_conf_mean = (g_conf * response_mask).sum() / mask_sum
        g_rew_mean = (g_rew * response_mask).sum() / mask_sum
        dual_gate_mean = (w_t * response_mask).sum() / mask_sum
        dual_gate_active = ((w_t > 0.5).float() * response_mask).sum() / mask_sum
        teacher_gap_mean = (delta_t * response_mask).sum() / mask_sum
        step_adv_abs_mean = (step_advantages.abs() * response_mask).sum() / mask_sum

    metrics = {
        "opto/g_conf_mean": g_conf_mean.item(),
        "opto/g_rew_mean": g_rew_mean.item(),
        "opto/dual_gate_mean": dual_gate_mean.item(),
        "opto/dual_gate_active_ratio": dual_gate_active.item(),
        "opto/teacher_gap_mean": teacher_gap_mean.item(),
        "opto/step_adv_abs_mean": step_adv_abs_mean.item(),
        "opto/loss": loss.detach().item(),
    }

    return loss, metrics


def compute_opto_agent_loss_reward_only(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    step_advantages: torch.Tensor,
    step_boundary_mask: torch.Tensor = None,
    tau: float = 2.0,
    weight_min: float = 0.05,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Ablation: reward gate only (no confidence gate).
    Tests the value of step-level reward attribution alone.
    """
    return compute_opto_agent_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        step_advantages=step_advantages,
        step_boundary_mask=step_boundary_mask,
        gate_beta=5.0,  # not used
        tau=tau,
        weight_min=weight_min,
        use_dual_gate=False,
        loss_agg_mode=loss_agg_mode,
    )


def compute_opto_agent_loss_episode_level(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    episode_advantages: torch.Tensor,
    gate_beta: float = 5.0,
    tau: float = 2.0,
    weight_min: float = 0.05,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    Ablation: uses episode-level advantage instead of step-level.
    Tests the value of step-level granularity.
    """
    return compute_opto_agent_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        step_advantages=episode_advantages,
        step_boundary_mask=None,
        gate_beta=gate_beta,
        tau=tau,
        weight_min=weight_min,
        use_dual_gate=True,
        loss_agg_mode=loss_agg_mode,
    )
