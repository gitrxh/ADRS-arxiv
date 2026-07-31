"""
OPTO-SDAR: Reward-Attributed Confidence-Gated Distillation.

Builds ON TOP of SDAR's confidence gate, adding a reward-attribution
layer that modulates per-token distillation based on completion advantage.

Key difference from standalone OPTO-Agent:
  - SDAR's confidence gate handles teacher quality (teacher_gap < 0 → low gate)
  - Reward attribution handles token selection (which tokens matter for reward)
  - The two are multiplicative: w_t = g_conf × g_rew

Key difference from SDAR:
  - SDAR treats all tokens in a good completion equally
  - OPTO-SDAR up-weights tokens where teacher-student disagree AND
    the completion has positive advantage, down-weights the rest
"""

import torch

from verl.trainer.ppo.core_algos import agg_loss


def compute_opto_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    gate_beta: float = 5.0,
    tau: float = 5.0,
    weight_min: float = 0.05,
    loss_agg_mode: str = "token-mean",
) -> tuple[torch.Tensor, dict]:
    """
    OPTO-SDAR loss = SDAR confidence gate × reward-attributed weight.

    For positive-advantage completions: high-KL tokens get extra weight
    For negative-advantage completions: distillation is suppressed
    For zero-advantage: falls back to vanilla SDAR behavior

    Args:
        student_log_probs: (bs, response_length) — retains grad.
        teacher_log_probs: (bs, response_length) — no grad.
        response_mask: (bs, response_length)
        advantages: (bs, response_length) — per-token advantage from GRPO.
        gate_beta: temperature for SDAR confidence gate.
        tau: temperature for reward gate.
        weight_min: minimum weight floor.
        loss_agg_mode: aggregation mode.
    """
    teacher_log_probs = teacher_log_probs.detach()

    # === SDAR confidence gate (unchanged) ===
    delta_t = teacher_log_probs - student_log_probs.detach()
    g_conf = torch.sigmoid(gate_beta * delta_t).detach()

    # === Reward-attributed gate ===
    # Get per-sequence advantage (same value across all tokens in GRPO)
    # Use first valid token's advantage value
    first_valid = response_mask.int().argmax(dim=-1)  # (bs,)
    batch_idx = torch.arange(advantages.size(0), device=advantages.device)
    seq_adv = advantages[batch_idx, first_valid]  # (bs,)

    # Reward gate: positive advantage → distill, negative → suppress
    # Scale by tau so sigmoid has meaningful range
    g_rew = torch.sigmoid(tau * seq_adv).unsqueeze(-1).expand_as(response_mask).detach()

    # === Combined gate ===
    w_t = (g_conf * g_rew).clamp(min=weight_min, max=1.0)

    # === Gated KL loss ===
    kl_per_token = teacher_log_probs - student_log_probs
    gated_kl = w_t * kl_per_token

    loss = agg_loss(loss_mat=gated_kl, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    # === Metrics ===
    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        pos_adv_ratio = (seq_adv > 0).float().mean()

        metrics = {
            "opto_sdar/g_conf_mean": (g_conf * response_mask).sum().item() / mask_sum.item(),
            "opto_sdar/g_rew_mean": (g_rew * response_mask).sum().item() / mask_sum.item(),
            "opto_sdar/combined_gate_mean": (w_t * response_mask).sum().item() / mask_sum.item(),
            "opto_sdar/gate_active_ratio": ((w_t > 0.5).float() * response_mask).sum().item() / mask_sum.item(),
            "opto_sdar/teacher_gap_mean": (delta_t * response_mask).sum().item() / mask_sum.item(),
            "opto_sdar/seq_adv_mean": seq_adv.mean().item(),
            "opto_sdar/pos_adv_ratio": pos_adv_ratio.item(),
            "opto_sdar/loss": loss.detach().item(),
        }

    return loss, metrics
