#!/usr/bin/env bash
set -euo pipefail

# Offline score a saved FSDP checkpoint on VerInstruct validation split.
#
# This script runs:
#  1) `verl.trainer.main_generation` to generate responses from the checkpoint
#  2) `verl.trainer.main_eval` to score the generated file with the same custom reward function
#
# Usage:
#   bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_eval_ckpt.sh \
#     /root/aicloud-data/checkpoints/Qwen2.5-7B-Instruct_verinstruct_RL/global_step_350/actor \
#     /root/aicloud-data/eval_outputs
#
# Environment variables (optional):
#   BASE_MODEL_PATH : base HF model path for tokenizer/config (default: Qwen2.5-7B-Instruct path used in training)
#   DATA_PATH       : parquet with prompts + reward_model (default: data/verinstruct/verinstruct_val.parquet)
#   N_GPUS          : world size for loading the FSDP ckpt (default: 4; must match ckpt world size)
#   N_SAMPLES       : number of responses per prompt (default: 8)
#   BATCH_SIZE      : generation batch size (default: 8)
#   ROLLOUT_NAME    : hf or vllm (default: hf)
#   TP_SIZE         : vLLM tensor parallel size (default: 1). Must satisfy TP_SIZE <= N_GPUS and N_GPUS % TP_SIZE == 0
#   GPU_MEM_UTIL    : vLLM gpu_memory_utilization (default: 0.6)
#   MAX_NUM_BATCHED_TOKENS : vLLM max_num_batched_tokens (default: 8192 for 4096+4096)
#   LOAD_FORMAT     : vLLM load_format (default: dummy_dtensor when syncing weights from FSDP ckpt)
#   ENABLE_CHUNKED_PREFILL : vLLM enable_chunked_prefill (default: True)
#   PROMPT_LEN      : max prompt length for chat template tokenization (default: 4096)
#   RESP_LEN        : max generated tokens (default: 4096)
#   TEMP/TOP_P/TOP_K: sampling params (defaults align with training)

CKPT_ACTOR_PATH="${1:-}"
OUT_DIR="${2:-/root/aicloud-data/eval_outputs}"

if [[ -z "${CKPT_ACTOR_PATH}" ]]; then
  echo "ERROR: missing ckpt path."
  echo "Usage: $0 /path/to/global_step_xxx/actor [output_dir]"
  exit 1
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/root/aicloud-fs/wangxuekang/model/Qwen2.5-7B-Instruct}"
DATA_PATH="${DATA_PATH:-data/verinstruct/verinstruct_val.parquet}"

N_GPUS="${N_GPUS:-4}"
N_SAMPLES="${N_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"

ROLLOUT_NAME="${ROLLOUT_NAME:-hf}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
# Must be >= (PROMPT_LEN + RESP_LEN) when ENABLE_CHUNKED_PREFILL=True in vLLM rollout.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy_dtensor}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
PROMPT_LEN="${PROMPT_LEN:-4096}"
RESP_LEN="${RESP_LEN:-4096}"
TEMP="${TEMP:-0.7}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"

mkdir -p "${OUT_DIR}"

CKPT_TAG="$(basename "$(dirname "${CKPT_ACTOR_PATH}")")_$(basename "${CKPT_ACTOR_PATH}")"
GEN_OUT_PATH="${OUT_DIR}/${CKPT_TAG}_verinstruct_val_gen_ns${N_SAMPLES}.parquet"

set -x

python3 -m verl.trainer.main_generation \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS}" \
  data.path="${DATA_PATH}" \
  data.output_path="${GEN_OUT_PATH}" \
  data.n_samples="${N_SAMPLES}" \
  data.batch_size="${BATCH_SIZE}" \
  model.path="${BASE_MODEL_PATH}" \
  model.ckpt_path="${CKPT_ACTOR_PATH}" \
  rollout.name="${ROLLOUT_NAME}" \
  rollout.tensor_model_parallel_size="${TP_SIZE}" \
  rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  rollout.load_format="${LOAD_FORMAT}" \
  rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  rollout.prompt_length="${PROMPT_LEN}" \
  rollout.response_length="${RESP_LEN}" \
  rollout.temperature="${TEMP}" \
  rollout.top_p="${TOP_P}" \
  rollout.top_k="${TOP_K}" \
  rollout.do_sample=True

python3 -m verl.trainer.main_eval \
  data.path="${GEN_OUT_PATH}" \
  custom_reward_function.path=health_bench/scaleai_batch_reward_fn.py \
  custom_reward_function.name=compute_score

echo "Generated parquet: ${GEN_OUT_PATH}"
