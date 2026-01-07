#!/usr/bin/env python3
import argparse
import json
from typing import List

from vllm import LLM, SamplingParams


def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IFEval responses with vLLM.")
    parser.add_argument("--model", required=True, help="HF model path (merged).")
    parser.add_argument(
        "--input",
        default="/root/google-research/instruction_following_eval/data/input_data.jsonl",
        help="IFEval input_data.jsonl path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output jsonl path with fields: prompt, response.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.95)
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw prompts without applying chat template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code to tokenizer/LLM.",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.input)

    if args.no_chat_template:
        vllm_prompts = prompts
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        vllm_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=args.trust_remote_code,
    )
    params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    with open(args.output, "w") as out:
        for i in range(0, len(prompts), args.batch_size):
            chunk_prompts = vllm_prompts[i : i + args.batch_size]
            outputs = llm.generate(chunk_prompts, params)
            for j, o in enumerate(outputs):
                out.write(
                    json.dumps(
                        {"prompt": prompts[i + j], "response": o.outputs[0].text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
