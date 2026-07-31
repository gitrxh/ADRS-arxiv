set -x
ENGINE=${1:-vllm}

# OPTO-Agent v2: tuned hyperparameters based on v1 diagnostics
# Changes from v1:
#   opto_coef: 0.01 → 0.05 (5x stronger distillation)
#   tau: 2.0 → 5.0 (sharper reward gate, more tokens pass)
#   gate_beta: 5.0 → 3.0 (softer confidence gate, since teacher_gap is negative)
#   weight_min: 0.05 → 0.1 (higher floor to ensure gradient flow)

num_cpus_per_env_worker=0.1
train_data_size=64
val_data_size=128
group_size=8

opto_coef=0.05
gate_beta=3.0
tau=5.0
weight_min=0.1
use_dual_gate=true
skill_all=false

experiment_name="opto_agent_v2_sokoban_coef${opto_coef}_tau${tau}_beta${gate_beta}"

python3 -m verl.trainer.main_opto_agent \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.005 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    +algorithm.opto_agent.opto_coef=$opto_coef \
    +algorithm.opto_agent.gate_beta=$gate_beta \
    +algorithm.opto_agent.tau=$tau \
    +algorithm.opto_agent.weight_min=$weight_min \
    +algorithm.opto_agent.use_dual_gate=$use_dual_gate \
    +algorithm.opto_agent.skills_dir=skills/sokoban \
    +algorithm.opto_agent.skill_all=$skill_all \
    env.env_name=Sokoban \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.sokoban.mode='list' \
    env.sokoban.dim_room=6 \
    env.sokoban.num_boxes=1 \
    env.sokoban.search_depth=30 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='opto_agent_sokoban' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=60 \
    trainer.val_before_train=True $@
