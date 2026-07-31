"""
Main entry point for TRACE (SC-TCA) training.

TRACE = GiGPO (step-level advantage) + SC-TCA gated distillation.

SC-TCA (Skill-Conditional Teacher Contrastive Attribution):
    Corr(teacher_logprob - student_logprob, reward) within anchor state groups.
    Directly measures whether the teacher's SKILL knowledge predicts success.

Uses ADRSRayTrainer for:
    - Teacher forward pass (skill-augmented)
    - GiGPO anchor state grouping
    - Step reward computation

The SC-TCA loss is computed inside the actor's update_policy() alongside
the standard GRPO/GiGPO policy gradient loss.
"""

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_trace(config)


def run_trace(config) -> None:
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TRACETaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TRACETaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        trace_cfg = config.algorithm.get("trace", {})

        with open_dict(config):
            # Enable TRACE (SC-TCA) distillation loss in actor
            config.actor_rollout_ref.actor.use_trace_loss = True
            config.actor_rollout_ref.actor.trace_loss_coef = trace_cfg.get("trace_coef", 0.01)
            config.actor_rollout_ref.actor.trace_gate_beta = trace_cfg.get("gate_beta", 5.0)
            config.actor_rollout_ref.actor.trace_tau = trace_cfg.get("tau", 3.0)
            config.actor_rollout_ref.actor.trace_weight_min = trace_cfg.get("weight_min", 0.05)
            config.actor_rollout_ref.actor.trace_use_stt = trace_cfg.get("use_stt", True)
            config.actor_rollout_ref.actor.trace_stt_temperature = trace_cfg.get("stt_temperature", 2.0)
            config.actor_rollout_ref.actor.trace_min_group_size = trace_cfg.get("min_group_size", 2)

            # Disable other distillation losses to isolate SC-TCA
            config.actor_rollout_ref.actor.use_sdar_loss = False
            config.actor_rollout_ref.actor.use_sdl_loss = False
            config.actor_rollout_ref.actor.use_opto_agent_loss = False
            config.actor_rollout_ref.actor.use_opto_sdar_loss = False

        adrs_cfg = config.algorithm.get("adrs", {})
        print(f"[TRACE/SC-TCA] trace_coef: {config.actor_rollout_ref.actor.trace_loss_coef}")
        print(f"[TRACE/SC-TCA] gate_beta: {config.actor_rollout_ref.actor.trace_gate_beta}")
        print(f"[TRACE/SC-TCA] tau: {config.actor_rollout_ref.actor.trace_tau}")
        print(f"[TRACE/SC-TCA] use_stt: {config.actor_rollout_ref.actor.trace_use_stt}")
        print(f"[TRACE/SC-TCA] stt_temperature: {config.actor_rollout_ref.actor.trace_stt_temperature}")
        print(f"[TRACE/SC-TCA] min_group_size: {config.actor_rollout_ref.actor.trace_min_group_size}")
        print(f"[TRACE/SC-TVA] adrs_eta: {adrs_cfg.get('eta', 0.0)} (0=SC-TCA only, >0=SC-TCA+ADRS)")
        print(f"[TRACE/SC-TCA] adv_estimator: {config.algorithm.adv_estimator}")

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from agent_system.environments import make_envs

        envs, val_envs = make_envs(config)

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager import EpisodeRewardManager

            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, (
            "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. "
            "In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"
        )

        from agent_system.multi_turn_rollout import TrajectoryCollector

        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        from verl.trainer.ppo.rlsd_utils import SkillProvider

        skills_dir = trace_cfg.get("skills_dir", adrs_cfg.get("skills_dir", "skills/alfworld"))
        skill_all = trace_cfg.get("skill_all", adrs_cfg.get("skill_all", False))
        skill_provider = SkillProvider(skills_dir=skills_dir, skill_all=skill_all)
        print(f"[TRACE/SC-TCA] Loaded skills from {skills_dir}")
        print(f"[TRACE/SC-TCA] Available skills: {list(skill_provider.skill_contents.keys())}")

        from verl.trainer.ppo.adrs_ray_trainer import ADRSRayTrainer

        trainer = ADRSRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
            skill_provider=skill_provider,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
