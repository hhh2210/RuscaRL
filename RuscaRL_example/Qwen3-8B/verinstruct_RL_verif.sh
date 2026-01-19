#!/bin/bash
# 从 .env 文件加载所有环境变量
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Model configuration
MODEL_PATH="/data/MODEL/Qwen3-8B"

# Data configuration (VerIF-format parquet built from THU-KEG/VerInstruct)
DATA_TRAIN_PATH="data/verinstruct_verif/verinstruct_verif_train.parquet"
DATA_VAL_PATH="data/verinstruct_verif/verinstruct_verif_val.parquet"
OUTPUT_DIR="/data/haozy/${EXPERIMENT_NAME}"
# Experiment configuration
VLLM_MODEL_TAG="${VLLM_MODEL:-unset}"
# If VLLM_MODEL is a path, keep only the last segment; then sanitize to be filesystem-safe.
VLLM_MODEL_TAG="${VLLM_MODEL_TAG##*/}"
VLLM_MODEL_TAG="$(printf '%s' "$VLLM_MODEL_TAG" | tr -cs 'A-Za-z0-9._-' '_')"
EXPERIMENT_NAME="Qwen3-8B_verinstruct_RL_verif_${VLLM_MODEL_TAG}"

export WANDB_MODE=online

# Workaround for vLLM engine selection mismatch:
# "Using V1 LLMEngine, but envs.VLLM_USE_V1=False."
# Explicitly enable vLLM V1 engine to match vllm>=0.8 defaults.
export VLLM_USE_V1=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
# Fix for: "CUDASymmetricMemoryAllocator::rendezvous: detected allocations from overlapping devices from different ranks"
# Ray by default sets per-actor CUDA_VISIBLE_DEVICES, so each rank often sees only one GPU as cuda:0.
# PyTorch symmetric-memory rendezvous may interpret this as overlapping devices across ranks.
# This flag tells Ray not to set CUDA_VISIBLE_DEVICES; verl will then set LOCAL_RANK from RAY_LOCAL_RANK
# and call torch.cuda.set_device(local_rank) during worker init.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_TRAIN_PATH} \
    data.val_files=${DATA_VAL_PATH} \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    custom_reward_function.path=verl/utils/reward_score/verif_reward_fn.py \
    custom_reward_function.name=compute_score_batched \
    +custom_reward_function.reward_kwargs.max_workers_per_url=512 \
    +custom_reward_function.reward_kwargs.skip_rules=true \
    reward_model.reward_manager=batch_verif \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_graded_system_prompt=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_general' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=999999 \
    trainer.test_freq=5 \
    trainer.rollout_data_dir="${OUTPUT_DIR}/log/rollout_log/${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/log/validation_log/${EXPERIMENT_NAME}" \
    trainer.total_training_steps=350 \
    trainer.total_epochs=5 $@
