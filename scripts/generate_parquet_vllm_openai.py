#!/usr/bin/env python3

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


def _normalize_messages(messages: Any) -> Optional[List[Dict[str, Any]]]:
    if messages is None:
        return None
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if isinstance(messages, tuple):
        messages = list(messages)
    return messages


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _call_chat_completions(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    n: int,
    temperature: float,
    top_p: float,
    top_k: Optional[int],
    max_tokens: int,
    timeout_s: float,
    api_key: Optional[str],
    strict_openai: bool,
    max_retries: int,
) -> List[str]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = _build_headers(api_key)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if top_k is not None and not strict_openai:
        payload["top_k"] = int(top_k)

    last_err: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", []) or []
            outputs: List[str] = []
            for choice in choices:
                msg = choice.get("message") or {}
                outputs.append(msg.get("content", ""))
            if len(outputs) < n:
                outputs.extend([""] * (n - len(outputs)))
            return outputs[:n]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(min(60.0, 1.5**attempt))
                continue
            raise last_err
    raise RuntimeError("unreachable")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate chat responses via an OpenAI-compatible vLLM endpoint and write a parquet.")
    p.add_argument("--data-path", required=True, help="Input parquet path (must contain a chat-style prompt column).")
    p.add_argument("--output-path", required=True, help="Output parquet path.")
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model", default=os.getenv("VLLM_MODEL") or "actor", help="Served model name as exposed by the OpenAI endpoint.")
    p.add_argument("--api-key", default=os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY"), help="Optional API key for the endpoint.")
    p.add_argument("--prompt-key", default="prompt", help="Column name that stores messages (list[dict]).")
    p.add_argument("--response-key", default="responses", help="Column name to write model outputs to (list[str]).")

    p.add_argument("--n-samples", type=int, default=8, help="Number of responses per prompt (OpenAI 'n').")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="Non-standard OpenAI param; ignored if --strict-openai is set.")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max new tokens per response.")

    p.add_argument("--max-workers", type=int, default=16, help="Client-side concurrency.")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--strict-openai", action="store_true", help="Only send standard OpenAI fields (disable top_k).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    df = pd.read_parquet(args.data_path)
    if args.prompt_key not in df.columns:
        raise KeyError(f"Missing prompt column '{args.prompt_key}' in {args.data_path}")

    total = len(df)
    outputs: List[Optional[List[str]]] = [None] * total

    def run_one(idx_and_prompt: Tuple[int, Any]) -> Tuple[int, List[str]]:
        idx, prompt = idx_and_prompt
        messages = _normalize_messages(prompt)
        if not messages:
            return idx, ["" for _ in range(args.n_samples)]
        texts = _call_chat_completions(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout_s,
            api_key=args.api_key,
            strict_openai=args.strict_openai,
            max_retries=args.max_retries,
        )
        return idx, texts

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [ex.submit(run_one, item) for item in enumerate(df[args.prompt_key].tolist())]
        for fut in tqdm(as_completed(futures), total=total):
            idx, texts = fut.result()
            outputs[idx] = texts

    if any(o is None for o in outputs):
        raise RuntimeError("Some generation jobs did not produce outputs.")

    df = df.copy()
    df[args.response_key] = outputs

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(args.output_path)
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()

