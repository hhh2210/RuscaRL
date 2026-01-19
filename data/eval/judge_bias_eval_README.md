# Judge-bias small eval set (human-curated)

This directory holds **small evaluation sets** intended to diagnose **LLM-as-a-judge bias** with controlled response variants.

## Dataset file

- `judge_bias_eval_10.jsonl`: 10 prompts sampled from:
  - `data/health_bench/healthbench_val.parquet`
  - `data/verinstruct_verif/verinstruct_verif_val.parquet`

Each JSONL row contains:
- **prompt**: flattened conversation/instruction text
- **spec**:
  - `healthbench`: `rubrics`
  - `verinstruct_verif`: `checkers`/`functions` parsed from `reward_model.ground_truth`
- **responses**: placeholders you should fill (or generate) for the following types:
  - `yshort`: concise but correct
  - `yverbose`: verbose but semantically same as correct
  - `ystructured`: add headers / bullet points (format bias)
  - `ysycophant`: add a flattering opener like “Great question!” (sycophancy bias)
  - `ycorrect`: the reference correct answer
  - `yconfident_wrong`: confident tone but incorrect (confidence bias)

## Build / refresh

### re-generate (deterministic sampling):

```bash
python scripts/build_judge_bias_eval_dataset.py \
  --n-total 500 \
  --n-healthbench 250 \
  --seed 0 \
  --out data/eval/judge_bias_eval_10.jsonl
```

## Policy rollout generation (V1)

### TODO: Rewriter
使用 policy模型 本身，通过 prompt 改写得到有 bias 的 rollout,然后再 judge 得到分数

### Remote (OpenAI-compatible endpoint)

```bash
python scripts/generate_policy_rollouts_for_eval.py \
  --in data/eval/judge_bias_eval_10.jsonl \
  --out data/eval/judge_bias_eval_10_with_ypolicy.jsonl \
  --policy-base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" \
  --policy-model "qwen3-8b"
```

### Local (vLLM Python API)

If `--policy-base-url` is empty, the script falls back to **local vLLM generate** and requires `--policy-model-path`:

```bash
export CUDA_VISIBLE_DEVICES=4 
python scripts/generate_policy_rollouts_for_eval.py \
  --in data/eval/judge_bias_eval_10.jsonl \
  --out data/eval/judge_bias_eval_10_with_ypolicy.jsonl \
  --policy-model-path /data/MODEL/Qwen3-8B/ \
  --local-batch-size 8 \
  --tp-size 1 \
  --max-model-len 8192 \
  --gpu-mem-util 0.9 \
  --dtype bfloat16
```

Notes:
- Remote endpoints typically cannot batch multiple prompts into one request; use `--remote-max-workers` to increase concurrency.
- Local vLLM can batch prompts efficiently; use `--local-batch-size` to control per-call batch size.

## Judge scoring (V1) + saving judge raw responses

Create a judge config JSON, e.g. `data/eval/judge_config_example.json`:

```json
[
  {
    "name": "qwen-flash",
    "vllm_base_url": "http://localhost:8001/v1",
    "vllm_model": "grader",
    "api_key_env": "VLLM_API_KEY",
    "temperature": 0.0
  }
]
```

Run scoring (writes `scores.jsonl` + `summary.json`):

```bash
python scripts/score_judge_bias_eval.py \
  --in data/eval/judge_bias_eval_10_with_ypolicy.jsonl \
  --judge-config data/eval/judge_config_example.json \
  --out-dir data/eval/judge_scores_out
```

If you want to inspect the **raw judge output**, add `--save-judge-responses`:

```bash
python scripts/score_judge_bias_eval.py \
  --in data/eval/judge_bias_eval_10_with_ypolicy.jsonl \
  --judge-config data/eval/judge_config_example.json \
  --out-dir data/eval/judge_scores_out \
  --save-judge-responses
```

The raw texts are stored (best-effort) under `judge_debug` in `data/eval/judge_scores_out/scores.jsonl`.
