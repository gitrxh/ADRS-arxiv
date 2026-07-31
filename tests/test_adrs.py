"""
Unit tests for ADRS: teacher reward shaping + TVA modulation.
"""

import numpy as np
import torch
import unittest


class TestComputeStepMean(unittest.TestCase):
    def test_basic(self):
        from verl.trainer.ppo.adrs_utils import compute_step_mean

        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        step_ids = torch.tensor([[0, 0, 1, 1]])

        result = compute_step_mean(values, mask, step_ids)
        self.assertAlmostEqual(result[0, 0].item(), 1.5, places=4)
        self.assertAlmostEqual(result[0, 1].item(), 1.5, places=4)
        self.assertAlmostEqual(result[0, 2].item(), 3.5, places=4)
        self.assertAlmostEqual(result[0, 3].item(), 3.5, places=4)

    def test_with_mask(self):
        from verl.trainer.ppo.adrs_utils import compute_step_mean

        values = torch.tensor([[1.0, 2.0, 0.0, 4.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
        step_ids = torch.tensor([[0, 0, 1, 1]])

        result = compute_step_mean(values, mask, step_ids)
        self.assertAlmostEqual(result[0, 0].item(), 1.5, places=4)
        self.assertAlmostEqual(result[0, 3].item(), 4.0, places=4)


class TestComputeTeacherReward(unittest.TestCase):
    def test_global_baseline(self):
        from verl.trainer.ppo.adrs_utils import compute_teacher_reward

        teacher_lp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.ones(1, 4)
        step_ids = torch.tensor([[0, 0, 1, 1]])

        reward = compute_teacher_reward(teacher_lp, mask, step_ids, baseline_mode="global")
        self.assertAlmostEqual(reward[mask.bool()].mean().item(), 0.0, places=4)

    def test_step_baseline(self):
        from verl.trainer.ppo.adrs_utils import compute_teacher_reward

        teacher_lp = torch.tensor([[-1.0, -3.0, -2.0, -4.0]])
        mask = torch.ones(1, 4)
        step_ids = torch.tensor([[0, 0, 1, 1]])

        reward = compute_teacher_reward(teacher_lp, mask, step_ids, baseline_mode="step", normalize=False)
        self.assertAlmostEqual(reward[0, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(reward[0, 1].item(), -1.0, places=4)
        self.assertAlmostEqual(reward[0, 2].item(), 1.0, places=4)
        self.assertAlmostEqual(reward[0, 3].item(), -1.0, places=4)


class TestTVAComputation(unittest.TestCase):
    def test_tva_basic(self):
        from verl.trainer.ppo.adrs_utils import compute_tva_vectorized

        alignment = torch.tensor([0.9, 0.1, 0.8, 0.2])
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        uids = np.array(["g1", "g1", "g1", "g1"])

        tva = compute_tva_vectorized(alignment, rewards, uids)
        self.assertTrue(tva[0].item() > 0, "TVA should be positive when aligned → good reward")

    def test_tva_single_group(self):
        from verl.trainer.ppo.adrs_utils import compute_tva_vectorized

        alignment = torch.tensor([0.5])
        rewards = torch.tensor([1.0])
        uids = np.array(["g1"])

        tva = compute_tva_vectorized(alignment, rewards, uids)
        self.assertAlmostEqual(tva[0].item(), 0.0, places=4)

    def test_tva_teacher_useless(self):
        from verl.trainer.ppo.adrs_utils import compute_tva_vectorized

        alignment = torch.tensor([0.9, 0.9, 0.1, 0.1])
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        uids = np.array(["g1", "g1", "g1", "g1"])

        tva = compute_tva_vectorized(alignment, rewards, uids)
        self.assertTrue(abs(tva[0].item()) < 0.5, "TVA should be near 0 when teacher is not predictive")


class TestCompletionLevelTVA(unittest.TestCase):
    def test_l2_basic(self):
        from verl.trainer.ppo.adrs_utils import compute_completion_level_tva

        bs, seq_len = 4, 6
        teacher_lp = torch.tensor([
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # aligned, reward=1
            [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0],  # divergent, reward=0
            [-1.2, -1.2, -1.2, -1.2, -1.2, -1.2],  # aligned, reward=1
            [-2.8, -2.8, -2.8, -2.8, -2.8, -2.8],  # divergent, reward=0
        ])
        mask = torch.ones(bs, seq_len)
        rewards = torch.zeros(bs, seq_len)
        rewards[0, -1] = 1.0
        rewards[2, -1] = 1.0
        uids = np.array(["p1", "p1", "p1", "p1"])

        tva, metrics = compute_completion_level_tva(teacher_lp, mask, rewards, uids)
        self.assertTrue(tva[0].item() > 0, "TVA should be positive: aligned completions succeed")
        self.assertIn("tva_l2/positive_ratio", metrics)

    def test_l2_teacher_useless(self):
        from verl.trainer.ppo.adrs_utils import compute_completion_level_tva

        bs, seq_len = 4, 6
        teacher_lp = torch.tensor([
            [-1.0] * seq_len,  # aligned, reward=1
            [-1.0] * seq_len,  # aligned, reward=0
            [-3.0] * seq_len,  # divergent, reward=1
            [-3.0] * seq_len,  # divergent, reward=0
        ])
        mask = torch.ones(bs, seq_len)
        rewards = torch.zeros(bs, seq_len)
        rewards[0, -1] = 1.0
        rewards[2, -1] = 1.0
        uids = np.array(["p1", "p1", "p1", "p1"])

        tva, _ = compute_completion_level_tva(teacher_lp, mask, rewards, uids)
        self.assertTrue(abs(tva[0].item()) < 0.3, "TVA should be near 0: teacher not predictive")


class TestPAS(unittest.TestCase):
    def test_pas_basic(self):
        from verl.trainer.ppo.adrs_utils import compute_pas

        teacher_lp = torch.tensor([[-1.0, -2.0, -1.5, -3.0]])
        ref_lp = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])
        mask = torch.ones(1, 4)

        pas, metrics = compute_pas(teacher_lp, ref_lp, mask)
        self.assertTrue(pas[0, 0].item() > 0, "Teacher prefers token 0 more than ref → positive PAS")
        self.assertTrue(pas[0, 3].item() < 0, "Teacher prefers token 3 less than ref → negative PAS")
        self.assertIn("pas/mean", metrics)


class TestSTARTVACombined(unittest.TestCase):
    def test_level_none(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        teacher_lp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.ones(1, 4)

        reward, metrics = compute_adrs_reward(
            teacher_lp, mask, eta=0.1, tva_level="none",
        )
        self.assertEqual(reward.shape, (1, 4))
        self.assertEqual(metrics["star/tva_level"], 0)
        self.assertTrue(reward.abs().sum().item() > 0)

    def test_level_l1(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        teacher_lp = torch.tensor([[-1.0, -2.0, -1.5, -3.0]])
        ref_lp = torch.tensor([[-2.0, -2.0, -2.0, -2.0]])
        mask = torch.ones(1, 4)

        reward, metrics = compute_adrs_reward(
            teacher_lp, mask, ref_log_probs=ref_lp,
            eta=0.1, tva_level="L1",
        )
        self.assertEqual(metrics["star/tva_level"], 1)
        self.assertIn("pas/mean", metrics)

    def test_level_l2(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        bs, seq_len = 4, 6
        teacher_lp = torch.randn(bs, seq_len) - 2.0
        mask = torch.ones(bs, seq_len)
        rewards = torch.zeros(bs, seq_len)
        rewards[0, -1] = 1.0
        rewards[2, -1] = 1.0
        uids = np.array(["p1", "p1", "p1", "p1"])

        reward, metrics = compute_adrs_reward(
            teacher_lp, mask,
            token_level_rewards=rewards, prompt_uids=uids,
            eta=0.1, tva_level="L2",
        )
        self.assertEqual(metrics["star/tva_level"], 2)
        self.assertIn("tva_l2/mean", metrics)

    def test_level_l3(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        bs, seq_len = 4, 8
        teacher_lp = torch.randn(bs, seq_len) - 2.0
        mask = torch.ones(bs, seq_len)
        step_boundary = torch.zeros(bs, seq_len, dtype=torch.long)
        for i in range(bs):
            step_boundary[i, seq_len // 2:] = 1

        step_rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.8, 0.2])
        step_uids = np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"])

        reward, metrics = compute_adrs_reward(
            teacher_lp, mask,
            step_rewards=step_rewards, step_group_uids=step_uids,
            step_boundary_mask=step_boundary,
            eta=0.1, tva_level="L3", tva_tau=2.0,
        )
        self.assertEqual(reward.shape, (bs, seq_len))
        self.assertEqual(metrics["star/tva_level"], 3)
        self.assertIn("tva/mean", metrics)

    def test_auto_selects_l3(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        bs, seq_len = 4, 8
        teacher_lp = torch.randn(bs, seq_len) - 2.0
        mask = torch.ones(bs, seq_len)
        step_boundary = torch.zeros(bs, seq_len, dtype=torch.long)
        step_rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.8, 0.2])
        step_uids = np.array(["g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2"])

        _, metrics = compute_adrs_reward(
            teacher_lp, mask,
            step_rewards=step_rewards, step_group_uids=step_uids,
            step_boundary_mask=step_boundary,
            eta=0.1, tva_level="auto",
        )
        self.assertEqual(metrics["star/tva_level"], 3, "Auto should select L3 when all data available")

    def test_auto_falls_back_to_l2(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        bs, seq_len = 4, 6
        teacher_lp = torch.randn(bs, seq_len) - 2.0
        mask = torch.ones(bs, seq_len)
        rewards = torch.zeros(bs, seq_len)
        rewards[0, -1] = 1.0
        uids = np.array(["p1", "p1", "p1", "p1"])

        _, metrics = compute_adrs_reward(
            teacher_lp, mask,
            token_level_rewards=rewards, prompt_uids=uids,
            eta=0.1, tva_level="auto",
        )
        self.assertEqual(metrics["star/tva_level"], 2, "Auto should select L2 when no step data")

    def test_auto_falls_back_to_none(self):
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        teacher_lp = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
        mask = torch.ones(1, 4)

        _, metrics = compute_adrs_reward(
            teacher_lp, mask, eta=0.1, tva_level="auto",
        )
        self.assertEqual(metrics["star/tva_level"], 0, "Auto should select none when no data")

    def test_style_suppression_via_gigpo(self):
        """
        Verify that style tokens (same teacher logprob across all trajectories
        in a group) get zero contribution after GiGPO group normalization.
        """
        from verl.trainer.ppo.adrs_utils import compute_teacher_reward

        bs = 4
        seq_len = 4
        teacher_lp = torch.tensor([
            [-1.0, -1.0, -2.0, -3.0],
            [-1.0, -1.0, -3.0, -2.0],
            [-1.0, -1.0, -2.5, -2.5],
            [-1.0, -1.0, -4.0, -1.0],
        ])
        mask = torch.ones(bs, seq_len)
        step_boundary = torch.zeros(bs, seq_len, dtype=torch.long)

        reward = compute_teacher_reward(teacher_lp, mask, step_boundary, baseline_mode="global", normalize=False)

        style_col = reward[:, 0]
        action_col = reward[:, 2]

        style_variance = style_col.var().item()
        action_variance = action_col.var().item()

        self.assertTrue(
            style_variance < action_variance,
            f"Style token variance ({style_variance:.4f}) should be less than "
            f"action token variance ({action_variance:.4f}) — GiGPO group norm "
            f"will further reduce style contribution to ~0"
        )


if __name__ == "__main__":
    unittest.main()
