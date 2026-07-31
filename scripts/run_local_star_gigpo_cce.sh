#!/usr/bin/env bash
# Local 4-GPU ADRS (GiGPO) with CCE-bootstrap dump for delta-CCE plots.
# Disable val_before_train so CCE dump starts from step 1.
# Usage: bash scripts/run_local_star_gigpo_cce.sh [vllm|sglang]
set -x
ENGINE=${1:-vllm}
cd "$(dirname "$0")/.."

# ---- Environment ----
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

# Local output directory
LOCAL_OUT=$HOME/star_gigpo_cce_local
mkdir -p "$LOCAL_OUT/metrics" "$LOCAL_OUT/wandb"
export WANDB_DIR=$LOCAL_OUT/wandb
export METRICS_LOG_DIR=$LOCAL_OUT/metrics

# Local alfworld data
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}

# ---- CCE dump config ----
export DUMP_CCE=1
export CCE_DUMP_DIR=$LOCAL_OUT/cce_dump
mkdir -p "$CCE_DUMP_DIR"

# ---- Hyperparams ----
num_cpus_per_env_worker=0.1
adrs_eta=0.1
tva_temperature=1.0
tva_tau=2.0
tva_level=auto
skill_all=false
mode="mean_norm"
train_data_size=16
val_data_size=128
group_size=8
total_steps=30
timestamp=$(date +%Y%m%d_%H%M%S)
experiment_name="star_gigpo_cce_local_qwen2.5_3b_${timestamp}"

# ---- Data ----
python3 examples/data_preprocess/prepare.py \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    || echo "[warn] prepare.py failed — reusing existing parquet at \$HOME/data/verl-agent/text/"

python3 -m verl.trainer.main_adrs \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=False \
    algorithm.gigpo.similarity_thresh=0.95 \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.adrs.eta=$adrs_eta \
    +algorithm.adrs.tva_temperature=$tva_temperature \
    +algorithm.adrs.tva_tau=$tva_tau \
    +algorithm.adrs.tva_level=$tva_level \
    +algorithm.adrs.baseline_mode=step \
    +algorithm.adrs.cce_is_temperature=1.0 \
    +algorithm.adrs.skills_dir=skills/alfworld \
    +algorithm.adrs.skill_all=$skill_all \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$LOCAL_OUT/checkpoints/$experiment_name \
    trainer.test_freq=999 \
    trainer.total_epochs=300 \
    trainer.total_training_steps=$total_steps \
    trainer.val_before_train=False
