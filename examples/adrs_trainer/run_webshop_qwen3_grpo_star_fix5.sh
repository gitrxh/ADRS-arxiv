set -x
ENGINE=${1:-vllm}

num_cpus_per_env_worker=0.1

# STAR fix5: fix4 + entropy protection (wn_clip + 3-phase schedule) — Qwen3-1.7B
#   step 1-20:  eta=0 (pure GRPO)
#   step 21-70: eta=target (teacher active, clipped to [-0.2, +0.2])
#   step 71+:   eta=0 (back to GRPO)
adrs_eta=${STAR_ETA:-0.02}
tva_level=L2
skill_all=false
TVA_GATE_NORM=${TVA_GATE_NORM:-False}
USE_RTG_ADVANTAGE=${USE_RTG_ADVANTAGE:-false}
eta_mode=${ETA_MODE:-fixed}
WN_CLIP=${WN_CLIP:-0.2}
WN_SIGN=${WN_SIGN:-1.0}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
WN_ADAPTIVE_CLIP=${WN_ADAPTIVE_CLIP:-False}
KL_BOOST_MULT=${KL_BOOST_MULT:-1.0}
WN_MODE=${WN_MODE:-additive}
ENT_COEFF_ACTIVE=${ENT_COEFF_ACTIVE:-0}
WN_NEG_RESET=${WN_NEG_RESET:-False}

train_data_size=16
val_data_size=128
group_size=8
timestamp=$(date +%Y%m%d_%H%M%S)
GATE_TAG=""; [ "$TVA_GATE_NORM" = "True" ] && GATE_TAG="_gatenorm"
RTG_TAG=""; [ "$USE_RTG_ADVANTAGE" = "true" ] && RTG_TAG="_rtg"
CLIP_TAG=""; [ "$WN_CLIP" != "0.2" ] && CLIP_TAG="_clip${WN_CLIP}"
SIGN_TAG=""; [ "$WN_SIGN" != "1.0" ] && SIGN_TAG="_sign${WN_SIGN}"
ENT_TAG=""; [ "$ENTROPY_COEF" != "0.001" ] && ENT_TAG="_ent${ENTROPY_COEF}"
ADAPT_TAG=""; [ "$WN_ADAPTIVE_CLIP" = "True" ] && ADAPT_TAG="_adaptclip"
KLBOOST_TAG=""; [ "$KL_BOOST_MULT" != "1.0" ] && KLBOOST_TAG="_klboost${KL_BOOST_MULT}"
MODE_TAG=""; [ "$WN_MODE" != "additive" ] && MODE_TAG="_${WN_MODE}"
NEGRESET_TAG=""; [ "$WN_NEG_RESET" = "True" ] && NEGRESET_TAG="_negreset"
experiment_name="grpo_star_fix5${GATE_TAG}${RTG_TAG}${CLIP_TAG}${SIGN_TAG}${ENT_TAG}${ADAPT_TAG}${KLBOOST_TAG}${MODE_TAG}${NEGRESET_TAG}_webshop_qwen3_1.7b_eta${adrs_eta}_${timestamp}"

python3 examples/data_preprocess/prepare.py \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    || echo "[warn] prepare.py failed (HF unreachable) — reusing existing parquet"

python3 -m verl.trainer.main_adrs \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    algorithm.use_kl_in_reward=False \
    +algorithm.adrs.eta=$adrs_eta \
    +algorithm.adrs.tva_level=$tva_level \
    +algorithm.adrs.tva_temperature=1.0 \
    +algorithm.adrs.tva_tau=2.0 \
    +algorithm.adrs.baseline_mode=step \
    +algorithm.adrs.normalize_mode=per_seq \
    +algorithm.adrs.tva_gate_norm=$TVA_GATE_NORM \
    +algorithm.adrs.use_rtg_advantage=$USE_RTG_ADVANTAGE \
    +algorithm.adrs.eta_mode=$eta_mode \
    +algorithm.adrs.wn_warmup_steps=20 \
    +algorithm.adrs.wn_active_steps=50 \
    +algorithm.adrs.wn_clip=$WN_CLIP \
    +algorithm.adrs.wn_sign=$WN_SIGN \
    +algorithm.adrs.wn_adaptive_clip=$WN_ADAPTIVE_CLIP \
    +algorithm.adrs.kl_boost_mult=$KL_BOOST_MULT \
    +algorithm.adrs.wn_mode=$WN_MODE \
    +algorithm.adrs.ent_coeff_active=$ENT_COEFF_ACTIVE \
    +algorithm.adrs.wn_neg_reset=$WN_NEG_RESET \
    +algorithm.adrs.skills_dir=skills/webshop \
    +algorithm.adrs.skill_all=$skill_all \
    env.env_name=webshop/WebShopEnv \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$HOME/data/adrs/checkpoints/$experiment_name \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
