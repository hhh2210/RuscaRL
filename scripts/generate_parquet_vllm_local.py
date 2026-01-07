#!/usr/bin/env python3

import argparse
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    if messages is None:
        return None
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, tuple):
        messages = list(messages)
    return messages


def _extract_texts(request_output: Any, n: int) -> List[str]:
    outputs = getattr(request_output, "outputs", None) or []
    texts: List[str] = []
    for out in outputs:
        text = getattr(out, "text", None)
        if text is None:
            text = str(out)
        texts.append(text)
    if len(texts) < n:
        texts.extend([""] * (n - len(texts)))
    return texts[:n]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate chat responses using vLLM Python API and write a parquet.")
    p.add_argument("--model", required=True, help="HF model path (e.g. merged hf model dir).")
    p.add_argument("--data-path", required=True, help="Input parquet path (must contain a chat-style prompt column).")
    p.add_argument("--output-path", required=True, help="Output parquet path.")

    p.add_argument("--prompt-key", default="prompt", help="Column name that stores messages (list[dict]).")
    p.add_argument("--response-key", default="responses", help="Column name to write outputs to (list[str]).")
    p.add_argument("--start", type=int, default=0, help="Start row index (0-based).")
    p.add_argument("--limit", type=int, default=None, help="Limit number of rows to generate (useful for smoke tests).")

    p.add_argument("--n-samples", type=int, default=8, help="Number of responses per prompt.")
    p.add_argument("--batch-size", type=int, default=8)

    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per response.")

    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size (e.g. 4 for 4 GPUs).")
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "half", "float16", "bfloat16", "float", "float32"])
    p.add_argument("--max-model-len", type=int, default=None, help="Override max_model_len for vLLM engine.")
    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--disable-custom-all-reduce", action="store_true", help="Pass disable_custom_all_reduce=True to vLLM.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.data_path)
    if args.prompt_key not in df.columns:
        raise KeyError(f"Missing prompt column '{args.prompt_key}' in {args.data_path}")

    start = max(0, int(args.start))
    if args.limit is None:
        df = df.iloc[start:].reset_index(drop=True)
    else:
        limit = max(0, int(args.limit))
        df = df.iloc[start : start + limit].reset_index(drop=True)

    engine_kwargs: Dict[str, Any] = {
        "tensor_parallel_size": int(args.tp_size),
        "gpu_memory_utilization": float(args.gpu_mem_util),
        "dtype": args.dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "disable_custom_all_reduce": bool(args.disable_custom_all_reduce),
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = int(args.max_model_len)

    llm = LLM(args.model, **engine_kwargs)
    sampling_params = SamplingParams(
        n=int(args.n_samples),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
    )

    prompts = df[args.prompt_key].tolist()
    outputs: List[List[str]] = []

    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[start : start + batch_size]
        batch_msgs = []
        for p in batch:
            msgs = _normalize_messages(p)
            if not msgs:
                msgs = [{"role": "user", "content": ""}]
            batch_msgs.append(msgs)

        req_outs = llm.chat(batch_msgs, sampling_params=sampling_params, use_tqdm=False)
        for req_out in req_outs:
            outputs.append(_extract_texts(req_out, n=int(args.n_samples)))

    if len(outputs) != len(df):
        raise RuntimeError(f"Output count mismatch: got {len(outputs)} outputs for {len(df)} rows.")

    out_df = df.copy()
    out_df[args.response_key] = outputs

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_parquet(args.output_path)
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
