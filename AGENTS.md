# 8*H800如何开启训练
conda activate verl
# 终端1：RL baseline，使用第一个 API key
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RAY_PORT=6379 \
VLLM_API_KEY="sk-7f9f79ef433c42eeb96a4aaa74b92aee" \
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_RL.sh

# 终端2：RuscaRL，使用第二个 API key
CUDA_VISIBLE_DEVICES=4,5,6,7 \
RAY_PORT=6380 \
VLLM_API_KEY="sk-8065d51e783f4ad79c72c54079b7d19d" \
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_RuscaRL.sh

# VerIF reward（仅 LLM judge，跳过 rule code）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RAY_PORT=6379 \
VLLM_API_KEY="sk-7f9f79ef433c42eeb96a4aaa74b92aee" \
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_RL_verif.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 \
RAY_PORT=6380 \
VLLM_API_KEY="sk-8065d51e783f4ad79c72c54079b7d19d" \
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_RuscaRL_verif.sh

# 数据对应关系
# - verinstruct_RL.sh / verinstruct_RuscaRL.sh -> data/verinstruct (rubric 格式)
# - verinstruct_*_verif.sh -> data/verinstruct_verif (VerIF checkers/functions 格式)

# evalatuation
### RL baseline:
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_eval_ckpt.sh /root/aicloud-data/checkpoints/Qwen2.5-7B-Instruct_verinstruct_RL/global_step_350/actor

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/generate_parquet_vllm_local.py \
--model /root/aicloud-data/merged_models/Qwen2.5-7B-Instruct_verinstruct_RuscaRL_step350 \
--data-path data/verinstruct/verinstruct_val.parquet \
--output-path /root/aicloud-data/eval_outputs/ruscarl_step350_gen.parquet \
--tp-size 4 \
--gpu-mem-util 0.95 \
--max-model-len 8192 \
--batch-size 8 \
--n-samples 8 \
--max-tokens 4096 \
--dtype bfloat16

python -m verl.trainer.main_eval \
  data.path=/root/aicloud-data/eval_outputs/ruscarl_step350_gen.parquet \
  custom_reward_function.path=health_bench/scaleai_batch_reward_fn.py \
  custom_reward_function.name=compute_score

### RuscaRL:
bash RuscaRL_example/Qwen2.5-7B-Instruct/verinstruct_eval_ckpt.sh /root/aicloud-data/checkpoints/Qwen2.5-7B-Instruct_verinstruct_RuscaRL/global_step_350/actor

# IFEval (instruction_following_eval) 一键评测
## 依赖与数据
# 代码与数据：/root/google-research/instruction_following_eval
# 依赖安装（absl 需要 absl-py）：
pip install absl-py langdetect nltk immutabledict
python - <<'PY'
import nltk; nltk.download("punkt")
PY

## vLLM 生成 + IFEval 评分（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/ifeval_eval_vllm.sh \
  --model /root/aicloud-data/merged_models/Qwen2.5-7B-Instruct_verinstruct_RuscaRL_step350 \
  --responses /root/google-research/instruction_following_eval/data/input_response_data_ruscarl_step350.jsonl

## 直接从 FSDP ckpt 合并 + 评测
CUDA_VISIBLE_DEVICES=0,1,2,3 \
scripts/ifeval_eval_vllm.sh \
  --ckpt-dir /root/aicloud-data/checkpoints/Qwen2.5-7B-Instruct_verinstruct_RuscaRL/global_step_350/actor \
  --merge-target /root/aicloud-data/merged_models/Qwen2.5-7B-Instruct_verinstruct_RuscaRL_step350

## 结果指标对应关系（IFEval strict/loose）
# Pr(S) = prompt-level (strict)
# Ins(S) = instruction-level (strict)
# Pr(L) = prompt-level (loose)
# Ins(L) = instruction-level (loose)

# 需要修复
1. LLM grading failure count reached limit (3 times), marking all as NOT_PRESEN 会导致分数被判定为 0
