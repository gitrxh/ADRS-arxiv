"""
ADRS Trainer.

Extends SkillSDRayTrainer. Instead of computing a distillation loss in the actor,
ADRS injects TVA-modulated teacher reward into token_level_rewards BEFORE
advantage computation. The actor's update_policy runs standard policy gradient
with NO distillation loss.

Key difference from SDAR/SkillSD:
    SDAR:     L = L_RL + λ·L_KD(gate × KL)       → two losses, gradient conflict
    ADRS: L = L_RL(r_env + η·TVA·r_teacher)   → one loss, GiGPO handles credit
"""

from pprint import pprint

import json
import os

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_rtg_advantage
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.trainer.ppo.rlsd_ray_trainer import RLSDRayTrainer, build_teacher_batch
from verl.trainer.ppo.skillsd_ray_trainer import SkillSDRayTrainer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)

from agent_system.multi_turn_rollout import adjust_batch


class ADRSRayTrainer(SkillSDRayTrainer):
    """
    ADRS: Teacher-competence-modulated reward shaping.

    Injects teacher reward into token_level_rewards before GiGPO advantage
    computation. No distillation loss in the actor.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, skill_provider=skill_provider, **kwargs)
        adrs_cfg = self.config.algorithm.get("adrs", {})
        self.adrs_eta = adrs_cfg.get("eta", 0.1)
        self.tva_temperature = adrs_cfg.get("tva_temperature", 1.0)
        self.tva_tau = adrs_cfg.get("tva_tau", 2.0)
        self.baseline_mode = adrs_cfg.get("baseline_mode", "step")
        self.tva_level = adrs_cfg.get("tva_level", "auto")
        # Std-normalize TVA before sigmoid so small magnitudes (~1e-3) activate the gate.

        self.tva_gate_norm = adrs_cfg.get("tva_gate_norm", False)
        self.use_rtg_advantage = adrs_cfg.get("use_rtg_advantage", False)
        self._wn_warmup_steps = adrs_cfg.get("wn_warmup_steps", 20)
        self._wn_active_steps = adrs_cfg.get("wn_active_steps", 50)
        self._wn_clip = adrs_cfg.get("wn_clip", 0.2)
        self._wn_sign = adrs_cfg.get("wn_sign", 1.0)
        self._wn_adaptive_clip = adrs_cfg.get("wn_adaptive_clip", False)
        self._wn_mode = adrs_cfg.get("wn_mode", "additive")
        # Boost KL coefficient during teacher-active window to counteract extra per-token divergence.

        self._kl_boost_mult = adrs_cfg.get("kl_boost_mult", 1.0)
        # Enhanced entropy_coeff only during teacher-active window; falls back to config default otherwise.

        self._ent_coeff_active = adrs_cfg.get("ent_coeff_active", 0)
        self._wn_neg_reset = adrs_cfg.get("wn_neg_reset", False)

        self.eta_mode = adrs_cfg.get("eta_mode", "fixed")
        # auto-eta v2 config
        self._eta_warmup_steps = adrs_cfg.get("eta_warmup_steps", 20)
        self._eta_floor_ratio = adrs_cfg.get("eta_floor_ratio", 0.1)
        self._eta_ema_alpha = adrs_cfg.get("eta_ema_alpha", 0.05)
        self._competence_ema_alpha = adrs_cfg.get("competence_ema_alpha", 0.1)
        # auto-eta v2 running state
        self._competence_ema = 0.5
        self._smoothed_eta = self.adrs_eta

    def fit(self):
        """
        Training loop with ADRS teacher reward injection.

        Identical to SkillSDRayTrainer.fit() except:
        1. After token_level_rewards is set and before compute_advantage(),
           we inject TVA-modulated teacher reward into token_level_rewards.
        2. No distillation loss flags are set — actor runs pure policy gradient.
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
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    if self.config.algorithm.adv_estimator in ('gigpo', 'grpo'):
                        from gigpo.core_gigpo import compute_step_discounted_returns
                        step_rewards_tensor = compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.get('gamma', 0.95)
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # Teacher forward pass (reuse SkillSD/RLSD infrastructure)
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ============================================================
                        # ADRS: Inject teacher reward BEFORE advantage computation
                        # ============================================================
                        with _timer("adrs", timing_raw):
                            adrs_metrics = self._inject_teacher_reward(batch)
                            metrics.update(adrs_metrics)

                        if self.eta_mode == "auto":
                            pos_ratio = adrs_metrics.get("tva_l2/positive_ratio") or adrs_metrics.get("tva/positive_ratio")
                            if pos_ratio is not None:
                                a = self._competence_ema_alpha
                                self._competence_ema = a * pos_ratio + (1 - a) * self._competence_ema

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        wn_end = self._wn_warmup_steps + self._wn_active_steps
                        if self.global_steps <= self._wn_warmup_steps:
                            wn_eta = 0.0
                        elif self._wn_active_steps > 0 and self.global_steps > wn_end:
                            wn_eta = 0.0
                        else:
                            wn_eta = self.adrs_eta
                        if self.use_rtg_advantage and self.config.algorithm.adv_estimator == "grpo":
                            grpo_mask = batch.batch["response_mask"]
                            if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                                resp_len = grpo_mask.size(1)
                                grpo_mask = batch.batch["loss_mask"][:, -resp_len:]
                            advantages, returns = compute_grpo_rtg_advantage(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                response_mask=grpo_mask,
                                index=batch.non_tensor_batch["uid"],
                                eta=wn_eta,
                                teacher_reward=batch.batch.get("teacher_reward_raw", None),
                            )
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns
                        elif self.use_rtg_advantage and self.config.algorithm.adv_estimator == "gigpo":
                            from gigpo.core_gigpo import compute_gigpo_rtg_advantage
                            gigpo_mask = batch.batch["response_mask"]
                            if self.config.actor_rollout_ref.rollout.multi_turn.enable:
                                resp_len = gigpo_mask.size(1)
                                gigpo_mask = batch.batch["loss_mask"][:, -resp_len:]
                            advantages, returns = compute_gigpo_rtg_advantage(
                                token_level_rewards=batch.batch["token_level_rewards"],
                                step_rewards=batch.batch["step_rewards"],
                                response_mask=gigpo_mask,
                                anchor_obs=batch.non_tensor_batch["anchor_obs"],
                                index=batch.non_tensor_batch["uid"],
                                traj_index=batch.non_tensor_batch["traj_uid"],
                                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                                mode=self.config.algorithm.gigpo.mode,
                                enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                                similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                                eta=wn_eta,
                                teacher_reward=batch.batch.get("teacher_reward_raw", None),
                            )
                            batch.batch["advantages"] = advantages
                            batch.batch["returns"] = returns
                        else:
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                                gigpo_mode=self.config.algorithm.gigpo.mode,
                                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                                eta=wn_eta,
                                teacher_reward=batch.batch.get("teacher_reward_raw", None),
                                wn_clip=self._wn_clip,
                                wn_sign=self._wn_sign,
                                wn_adaptive_clip=self._wn_adaptive_clip,
                                wn_mode=self._wn_mode,
                                student_log_probs=batch.batch.get("old_log_probs", None),
                                teacher_log_probs=batch.batch.get("teacher_log_probs", None),
                                wn_neg_reset=self._wn_neg_reset,
                            )

                        # No distillation loss — advantages go directly to actor
                        batch, metrics = self._post_advantage_hook(batch, metrics)


                        # Log teacher-student gap metrics (for analysis only)
                        response_mask = batch.batch["response_mask"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_lp = batch.batch["teacher_log_probs"]
                        delta_t = (teacher_lp - student_log_probs) * response_mask
                        metrics["adrs/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["adrs/eta"] = wn_eta
                        metrics["adrs/wn_clip"] = self._wn_clip
                        metrics["adrs/wn_sign"] = self._wn_sign
                        metrics["adrs/wn_adaptive_clip"] = self._wn_adaptive_clip
                        metrics["adrs/wn_mode"] = self._wn_mode
                        metrics["adrs/wn_neg_reset"] = self._wn_neg_reset

                        base_kl_coef = self.config.actor_rollout_ref.actor.kl_loss_coef
                        if self._kl_boost_mult != 1.0 and wn_eta > 0:
                            batch.meta_info["kl_loss_coef_override"] = base_kl_coef * self._kl_boost_mult
                        metrics["adrs/kl_coef_effective"] = base_kl_coef * self._kl_boost_mult if (self._kl_boost_mult != 1.0 and wn_eta > 0) else base_kl_coef

                        if self._ent_coeff_active > 0 and wn_eta > 0:
                            batch.meta_info["entropy_coeff_override"] = self._ent_coeff_active
                        metrics["adrs/ent_coeff_effective"] = self._ent_coeff_active if (self._ent_coeff_active > 0 and wn_eta > 0) else self.config.actor_rollout_ref.actor.entropy_coeff

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    test_start_step = self.config.trainer.get("test_start_step", 0)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or (self.global_steps >= test_start_step and self.global_steps % self.config.trainer.test_freq == 0)):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def _inject_teacher_reward(self, batch: DataProto) -> dict:
        """
        Compute and inject TVA-modulated teacher reward into token_level_rewards.

        Auto-selects the finest available TVA level:
            L3 (step-level):       when anchor_obs + step_rewards available (GiGPO)
            L2 (completion-level): when prompt_uids available (GRPO with K>1)
            L1 (PAS):              when ref_log_prob available
            none:                  fixed eta
        """
        from verl.trainer.ppo.adrs_utils import compute_adrs_reward

        teacher_log_probs = batch.batch["teacher_log_probs"]
        response_mask = batch.batch["response_mask"]
        bs = response_mask.shape[0]
        seq_len = response_mask.shape[1]

        # Build step boundary mask from turn_step (for L3 and step-baseline)
        turn_steps = batch.non_tensor_batch.get("turn_step", None)
        step_boundary_mask = None
        if turn_steps is not None:
            step_boundary_mask = torch.zeros(bs, seq_len, device=response_mask.device, dtype=torch.long)
            for i in range(bs):
                step_boundary_mask[i] = int(turn_steps[i])

        # L3 data: step rewards + anchor state group UIDs
        step_rewards = batch.batch.get("step_rewards", None)
        step_group_uids = None
        anchor_obs = batch.non_tensor_batch.get("anchor_obs", None)
        index = batch.non_tensor_batch.get("uid", None)

        if anchor_obs is not None and step_rewards is not None:
            from gigpo.core_gigpo import build_step_group
            gigpo_cfg = self.config.algorithm.get("gigpo", {})
            step_group_uids = build_step_group(
                anchor_obs=anchor_obs,
                index=index,
                enable_similarity=gigpo_cfg.get("enable_similarity", False),
                similarity_thresh=gigpo_cfg.get("similarity_thresh", 0.95),
            )

            # Convert string UIDs to integer group IDs for SC-TCA in actor
            uid_to_int = {}
            int_ids = torch.zeros(bs, dtype=torch.long, device=response_mask.device)
            for i, uid in enumerate(step_group_uids):
                if uid not in uid_to_int:
                    uid_to_int[uid] = len(uid_to_int)
                int_ids[i] = uid_to_int[uid]
            batch.batch["step_group_ids"] = int_ids

        # L2 data: prompt UIDs + token-level rewards
        prompt_uids = batch.non_tensor_batch.get("uid", None)
        token_level_rewards = batch.batch.get("token_level_rewards", None)

        # L1 data: reference logprobs
        ref_log_probs = batch.batch.get("ref_log_prob", None)

        if self.eta_mode == "auto":
            eta_floor = self.adrs_eta * self._eta_floor_ratio
            if self.global_steps <= self._eta_warmup_steps:
                target_eta = self.adrs_eta
            else:
                target_eta = eta_floor + (self.adrs_eta - eta_floor) * self._competence_ema
            self._smoothed_eta += self._eta_ema_alpha * (target_eta - self._smoothed_eta)
            effective_eta = max(eta_floor, self._smoothed_eta)
        else:
            effective_eta = self.adrs_eta

        teacher_reward, tva_modulation, adrs_metrics = compute_adrs_reward(
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            step_rewards=step_rewards,
            step_group_uids=step_group_uids,
            step_boundary_mask=step_boundary_mask,
            token_level_rewards=token_level_rewards,
            prompt_uids=prompt_uids,
            ref_log_probs=ref_log_probs,
            eta=effective_eta,
            tva_temperature=self.tva_temperature,
            tva_tau=self.tva_tau,
            baseline_mode=self.baseline_mode,
            normalize_mode=self.config.algorithm.adrs.get("normalize_mode", "global"),
            tva_gate_norm=self.tva_gate_norm,
            tva_level=self.tva_level,
        )

        # DEBUG: verify whether ADRS contribution survives sum(dim=-1) in GRPO/GiGPO
        _adrs_row_sum = teacher_reward.sum(dim=-1)
        _scores_before = batch.batch["token_level_rewards"].sum(dim=-1)
        batch.batch["token_level_rewards"] = batch.batch["token_level_rewards"] + teacher_reward
        _scores_after = batch.batch["token_level_rewards"].sum(dim=-1)
        _delta = (_scores_after - _scores_before).abs()
        print(f"[ADRS-DEBUG] row_sum: max={_adrs_row_sum.abs().max():.6e}, mean={_adrs_row_sum.abs().mean():.6e} | "
              f"score_delta: max={_delta.max():.6e}, mean={_delta.mean():.6e} | "
              f"eta={effective_eta:.4f}, bs={teacher_reward.shape[0]}")
        batch.batch["tva_modulation"] = tva_modulation.contiguous()
        batch.batch["teacher_reward_raw"] = teacher_reward.detach()

        if self.eta_mode == "auto":
            adrs_metrics["auto_eta/competence_ema"] = self._competence_ema
            adrs_metrics["auto_eta/target_eta"] = target_eta
            adrs_metrics["auto_eta/smoothed_eta"] = self._smoothed_eta

        # ============================================================
        # CCE (Stage 0): bootstrap counterfactual-credit diagnostics + dump.
        # Off by default (env DUMP_CCE). Reuses the GiGPO anchor groups; needs
        # no extra rollout and does NOT modify rewards/advantages.
        # ============================================================
        import os as _os
        if _os.environ.get("DUMP_CCE") and step_group_uids is not None and step_rewards is not None:
            try:
                from verl.trainer.ppo.cce_utils import compute_cce_bootstrap, dump_cce_records

                student_lp = batch.batch.get("old_log_probs", None)
                if student_lp is not None:
                    cce_per_row, delta_mean, delta_sum, ess_per_row, cce_metrics = compute_cce_bootstrap(
                        teacher_log_probs=teacher_log_probs,
                        student_log_probs=student_lp,
                        response_mask=response_mask,
                        step_rewards=step_rewards,
                        group_uids=step_group_uids,
                        is_temperature=float(self.config.algorithm.adrs.get("cce_is_temperature", 1.0)),
                    )
                    adrs_metrics.update(cce_metrics)

                    # Trajectory-level success, broadcast to each step-row (for plot coloring).
                    won_per_row = None
                    tls = batch.batch.get("token_level_scores", None)
                    traj_uid = batch.non_tensor_batch.get("uid", None)
                    if tls is not None and traj_uid is not None:
                        row_score = tls.sum(dim=-1)
                        won_map = {}
                        for i, u in enumerate(traj_uid):
                            won_map[u] = max(won_map.get(u, 0.0), float(row_score[i].item() > 0))
                        won_per_row = [won_map[u] for u in traj_uid]

                    dump_dir = _os.environ.get("CCE_DUMP_DIR", "./cce_dump")
                    dump_cce_records(
                        path=_os.path.join(dump_dir, "cce_records.jsonl"),
                        global_step=getattr(self, "global_steps", -1),
                        group_uids=step_group_uids,
                        delta_mean=delta_mean,
                        delta_sum=delta_sum,
                        returns=step_rewards,
                        cce_per_row=cce_per_row,
                        ess_per_row=ess_per_row,
                        won=won_per_row,
                        tag="real_skill",
                    )
            except Exception as _e:  # never let diagnostics break training
                print(f"[CCE] dump skipped due to: {_e}")

        return adrs_metrics
