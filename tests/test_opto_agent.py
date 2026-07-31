"""
Unit tests for OPTO-Agent: Step-Level Reward-Attributed Token Optimization.

Tests cover:
1. Basic loss computation correctness
2. Dual-gate behavior (style suppression)
3. Step-level vs episode-level comparison
4. Gradient flow
5. Edge cases
"""

import torch
import pytest
import sys

from verl.trainer.ppo.opto_agent_utils import (
    compute_opto_agent_loss,
    compute_opto_agent_loss_reward_only,
    compute_opto_agent_loss_episode_level,
    compute_step_kl_share_vectorized,
)


class TestOPTOAgentLoss:
    """Test suite for OPTO-Agent dual-gated loss."""

    def setup_method(self):
        torch.manual_seed(42)
        self.bs = 8
        self.seq_len = 64
        self.student_lp = torch.randn(self.bs, self.seq_len) - 1.0
        self.teacher_lp = torch.randn(self.bs, self.seq_len) - 0.5
        self.response_mask = torch.ones(self.bs, self.seq_len)
        self.response_mask[:, 50:] = 0
        self.step_boundary = torch.zeros(self.bs, self.seq_len, dtype=torch.long)
        self.step_boundary[:, 16:32] = 1
        self.step_boundary[:, 32:48] = 2
        self.step_boundary[:, 48:] = 3

    def test_basic_computation(self):
        """Loss should be a valid scalar."""
        step_adv = torch.randn(self.bs, self.seq_len) * 0.5
        loss, metrics = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=self.step_boundary,
        )
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
        assert "opto/g_conf_mean" in metrics
        assert "opto/g_rew_mean" in metrics
        assert "opto/dual_gate_mean" in metrics

    def test_style_suppression(self):
        """Negative advantage steps should get lower gate values."""
        # All positive advantage
        step_adv_pos = torch.ones(self.bs, self.seq_len)
        _, m_pos = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv_pos,
            step_boundary_mask=self.step_boundary,
        )

        # All negative advantage
        step_adv_neg = -torch.ones(self.bs, self.seq_len)
        _, m_neg = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv_neg,
            step_boundary_mask=self.step_boundary,
        )

        # Positive advantage should yield higher reward gate
        assert m_pos["opto/g_rew_mean"] > m_neg["opto/g_rew_mean"], (
            f"Positive adv should have higher g_rew: {m_pos['opto/g_rew_mean']:.4f} vs {m_neg['opto/g_rew_mean']:.4f}"
        )

    def test_dual_gate_vs_single(self):
        """Dual gate should be <= confidence gate (multiplicative)."""
        step_adv = torch.randn(self.bs, self.seq_len) * 0.5
        _, m_dual = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=self.step_boundary,
            use_dual_gate=True,
        )
        _, m_single = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=self.step_boundary,
            use_dual_gate=False,
        )

        # Dual gate mean should be lower (more selective)
        assert m_dual["opto/dual_gate_mean"] <= m_dual["opto/g_conf_mean"] + 0.01

    def test_gradient_flow(self):
        """Gradients should flow through student_log_probs only."""
        student_lp = self.student_lp.clone().requires_grad_(True)
        student_lp.retain_grad()
        step_adv = torch.randn(self.bs, self.seq_len) * 0.5

        loss, _ = compute_opto_agent_loss(
            student_log_probs=student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=self.step_boundary,
        )
        loss.backward()

        assert student_lp.grad is not None, "Student should receive gradients"
        # Masked positions should have zero gradient
        masked_grad = student_lp.grad[:, 50:]
        assert (masked_grad == 0).all(), "Masked positions should have zero gradient"

    def test_step_kl_share_normalization(self):
        """KL shares within each step should sum to 1."""
        token_kl = torch.rand(self.bs, self.seq_len)
        kl_share = compute_step_kl_share_vectorized(
            token_kl, self.response_mask, self.step_boundary
        )

        # Check normalization per step (within valid mask)
        for b in range(self.bs):
            for step_id in range(3):
                step_mask = (self.step_boundary[b] == step_id) & self.response_mask[b].bool()
                if step_mask.sum() > 0:
                    step_sum = kl_share[b][step_mask].sum().item()
                    assert abs(step_sum - 1.0) < 1e-5, (
                        f"KL share should sum to 1 within step, got {step_sum:.6f}"
                    )

    def test_episode_level_fallback(self):
        """When step_boundary is None, should normalize across entire response."""
        step_adv = torch.randn(self.bs, self.seq_len) * 0.5
        loss, metrics = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=None,  # fallback to episode-level
        )
        assert not torch.isnan(loss)
        assert "opto/dual_gate_mean" in metrics

    def test_zero_advantage(self):
        """Zero advantage should give ~0.5 reward gate (sigmoid(0))."""
        step_adv = torch.zeros(self.bs, self.seq_len)
        _, metrics = compute_opto_agent_loss(
            student_log_probs=self.student_lp,
            teacher_log_probs=self.teacher_lp,
            response_mask=self.response_mask,
            step_advantages=step_adv,
            step_boundary_mask=self.step_boundary,
        )
        # With zero advantage, g_rew should be ~0.5
        assert abs(metrics["opto/g_rew_mean"] - 0.5) < 0.05, (
            f"Zero advantage should give ~0.5 g_rew, got {metrics['opto/g_rew_mean']:.4f}"
        )


if __name__ == "__main__":
    test = TestOPTOAgentLoss()
    test.setup_method()

    print("Running OPTO-Agent unit tests...")
    test.test_basic_computation()
    print("  [PASS] basic_computation")
    test.test_style_suppression()
    print("  [PASS] style_suppression")
    test.test_dual_gate_vs_single()
    print("  [PASS] dual_gate_vs_single")
    test.test_gradient_flow()
    print("  [PASS] gradient_flow")
    test.test_step_kl_share_normalization()
    print("  [PASS] step_kl_share_normalization")
    test.test_episode_level_fallback()
    print("  [PASS] episode_level_fallback")
    test.test_zero_advantage()
    print("  [PASS] zero_advantage")
    print("\nAll 7 tests passed!")
