#!/bin/bash
# STAR fix5 on SearchQA -- Qwen3-1.7B
# fix5 = fix4 + entropy protection (wn_clip + three-phase schedule)
#   step 1-20:  eta=0 (pure GRPO)
#   step 21-70: eta=target (teacher active, clipped to [-0.2, +0.2])
#   step 71+:   eta=0 (back to GRPO)
#
# Self-contained: setup -> retrieval server -> training -> cleanup.
#
# Prerequisites -- download the following data before running:
#   $OSS_BASE/search_data/processed/train.parquet  (preprocessed QA data)
#   $OSS_BASE/search_data/processed/test.parquet
#   $OSS_BASE/search_data/wiki-18.jsonl            (Wikipedia corpus)
#   $OSS_BASE/search_index/e5_Flat.index            (FAISS index)
#   $OSS_BASE/search_model/e5-base-v2/              (E5 retriever weights)
#   faiss-gpu wheel installed (or extracted to /dev/shm/faiss_gpu)
set -x
ENGINE=${1:-vllm}

# ============================================================
# 1. Environment variables & hyperparameters
# ============================================================
export TOKENIZERS_PARALLELISM=false

num_cpus_per_env_worker=0.1
adrs_eta=${STAR_ETA:-0.02}
eta_mode=${ETA_MODE:-fixed}
TVA_GATE_NORM=${TVA_GATE_NORM:-False}
USE_RTG_ADVANTAGE=${USE_RTG_ADVANTAGE:-false}
WN_CLIP=${WN_CLIP:-0.2}
WN_SIGN=${WN_SIGN:-1.0}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
WN_ADAPTIVE_CLIP=${WN_ADAPTIVE_CLIP:-True}
KL_BOOST_MULT=${KL_BOOST_MULT:-1.0}
WN_MODE=${WN_MODE:-additive}
ENT_COEFF_ACTIVE=${ENT_COEFF_ACTIVE:-0.003}
WN_NEG_RESET=${WN_NEG_RESET:-False}
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
experiment_name="grpo_star_fix5${GATE_TAG}${RTG_TAG}${CLIP_TAG}${SIGN_TAG}${ENT_TAG}${ADAPT_TAG}${KLBOOST_TAG}${MODE_TAG}${NEGRESET_TAG}_search_qwen3_1.7b_eta${adrs_eta}_${timestamp}"

# Data and model paths (adjust to your local layout)
OSS_BASE=$HOME/data/adrs
SEARCH_DATA_DIR=$OSS_BASE/search_data
SEARCH_INDEX_DIR=$OSS_BASE/search_index
LOG_DIR=$OSS_BASE/logs
METRICS_DIR=$OSS_BASE/metrics
CKPT_DIR=$OSS_BASE/checkpoints/$experiment_name
MODEL_PATH=Qwen/Qwen3-1.7B

mkdir -p $SEARCH_DATA_DIR $SEARCH_INDEX_DIR $LOG_DIR $METRICS_DIR $CKPT_DIR

export WANDB_MODE=offline
export WANDB_DIR=$OSS_BASE/wandb
export METRICS_LOG_DIR=$METRICS_DIR
mkdir -p $WANDB_DIR $METRICS_LOG_DIR 2>/dev/null || true

# ============================================================
# 2-5. Dependencies, data, index, retrieval server
# ============================================================
CONDA_PREFIX=${CONDA_PREFIX:-/opt/conda/envs/python3.10.13}
PIP="${CONDA_PREFIX}/bin/pip"

echo "[$(date)] Installing search dependencies..."
$PIP install "numpy<2" -q 2>/dev/null || true
$PIP install cbor2 tensordict accelerate -q 2>/dev/null || true
$PIP install datasets transformers -q 2>/dev/null || true
$PIP install flash-attn --no-build-isolation --no-cache-dir -q 2>/dev/null || true
$PIP install -e . -q 2>/dev/null || true
$PIP install "numpy<2" -q 2>/dev/null || true
$PIP uninstall -y faiss faiss-cpu 2>/dev/null || true

PROCESSED_DATA_DIR=$OSS_BASE/search_data/processed
RAW_QA_DIR=$OSS_BASE/search_data/qa_dataset

if [ -f "$PROCESSED_DATA_DIR/train.parquet" ]; then
    echo "[$(date)] Processed QA data found: $PROCESSED_DATA_DIR"
elif [ -f "$RAW_QA_DIR/train.parquet" ]; then
    echo "[$(date)] Raw QA data found, processing..."
    mkdir -p $PROCESSED_DATA_DIR
    python3 examples/data_preprocess/preprocess_search_r1_dataset.py \
        --hf_repo_id PeterJinGo/nq_hotpotqa_train \
        --local_dir $PROCESSED_DATA_DIR 2>/dev/null || \
        echo "[$(date)] WARNING: preprocess failed; expecting processed parquet"
else
    echo "[$(date)] ERROR: No QA data found. Please download data to $PROCESSED_DATA_DIR"
    exit 1
fi

WIKI_CORPUS=$SEARCH_DATA_DIR/wiki-18.jsonl
E5_INDEX=$SEARCH_INDEX_DIR/e5_Flat.index
E5_MODEL=$OSS_BASE/search_model/e5-base-v2

if [ ! -f "$E5_INDEX" ]; then echo "[$(date)] ERROR: E5 index not found at $E5_INDEX"; exit 1; fi
if [ ! -f "$WIKI_CORPUS" ]; then echo "[$(date)] ERROR: Wiki corpus not found at $WIKI_CORPUS"; exit 1; fi
if [ ! -d "$E5_MODEL" ]; then
    echo "[$(date)] WARNING: local e5 model not found at $E5_MODEL -- falling back to HF id"
    E5_MODEL=intfloat/e5-base-v2
fi

# FAISS GPU setup (adjust path if your faiss-gpu package is installed elsewhere)
FAISS_LOCAL=/dev/shm/faiss_gpu
if [ -d "$FAISS_LOCAL/pkg/faiss" ]; then
    export PYTHONPATH="$FAISS_LOCAL/pkg:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$FAISS_LOCAL/libs:${LD_LIBRARY_PATH:-}"
fi

SEARCH_PORT=8000
SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}/retrieve"
RETRIEVAL_LOG=$LOG_DIR/retrieval_server_${experiment_name}.log
if ! python3 -c "import faiss; assert hasattr(faiss,'GpuMultipleClonerOptions')" 2>/dev/null; then
    echo "[$(date)] FATAL: faiss has no GPU support"; exit 1
fi
echo "[$(date)] Starting retrieval server on port $SEARCH_PORT..."
python3 -u examples/search/retriever/retrieval_server.py \
    --index_path $E5_INDEX --corpus_path $WIKI_CORPUS \
    --retriever_name e5 --retriever_model $E5_MODEL \
    --faiss_gpu --topk 3 --port $SEARCH_PORT > "$RETRIEVAL_LOG" 2>&1 &
SEARCH_PID=$!

echo "[$(date)] Waiting for retrieval server (up to 1800s)..."
for i in $(seq 1 1800); do
    if ! kill -0 $SEARCH_PID 2>/dev/null; then
        echo "[$(date)] ERROR: retrieval server died."; tail -40 "$RETRIEVAL_LOG"; exit 1
    fi
    if curl -s -o /dev/null -w "%{http_code}" -X POST $SEARCH_URL \
        -H "Content-Type: application/json" -d '{"query":"test","topk":1}' | grep -q "200"; then
        echo "[$(date)] Retrieval server ready after ${i}s!"; break
    fi
    if [ $i -eq 1800 ]; then
        echo "[$(date)] ERROR: Retrieval server timeout."; tail -40 "$RETRIEVAL_LOG"; kill $SEARCH_PID 2>/dev/null; exit 1
    fi
    sleep 1
done

# ============================================================
# 6. Training -- fix5, Qwen3-1.7B
# ============================================================
echo "[$(date)] Starting GRPO-STAR fix5 SearchQA Qwen3-1.7B: $experiment_name"

python3 -m verl.trainer.main_adrs \
    algorithm.adv_estimator=grpo \
    data.train_files=$PROCESSED_DATA_DIR/train.parquet \
    data.val_files=${VAL_FILES:-$PROCESSED_DATA_DIR/test.parquet} \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +algorithm.adrs.eta=$adrs_eta \
    +algorithm.adrs.tva_level=L2 \
    +algorithm.adrs.tva_temperature=1.0 \
    +algorithm.adrs.tva_tau=2.0 \
    +algorithm.adrs.baseline_mode=step \
    +algorithm.adrs.tva_gate_norm=$TVA_GATE_NORM \
    +algorithm.adrs.use_rtg_advantage=$USE_RTG_ADVANTAGE \
    +algorithm.adrs.normalize_mode=per_seq \
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
    +algorithm.adrs.skills_dir=skills/search \
    +algorithm.adrs.skill_all=false \
    env.env_name=search/SearchQAEnv \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.search.search_url=$SEARCH_URL \
    env.search.topk=3 \
    env.search.timeout=60 \
    env.search.log_requests=False \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.test_freq=${TEST_FREQ:-25} \
    trainer.total_epochs=150 \
    trainer.total_training_steps=${TOTAL_STEPS:-150} \
    trainer.val_only=${VAL_ONLY:-False} \
    trainer.resume_mode=${RESUME_MODE:-auto} \
    trainer.resume_from_path=${RESUME_PATH:-null} \
    trainer.val_before_train=${VAL_BEFORE:-True} $@

EXIT_CODE=$?
echo "[$(date)] Training finished (exit=$EXIT_CODE). Stopping retrieval server..."
kill $SEARCH_PID 2>/dev/null; wait $SEARCH_PID 2>/dev/null
echo "[$(date)] Done."
exit $EXIT_CODE
