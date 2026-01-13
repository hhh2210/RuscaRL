#!/usr/bin/env python3
"""
Generate ypolicy_raw rollouts for a judge-bias eval JSONL.

This script is intentionally simple:
  - Reads `data/eval/judge_bias_eval_*.jsonl`
  - For each item, calls an OpenAI-compatible `/chat/completions` endpoint
  - Writes a new JSONL with `responses.ypolicy_raw` filled

Why: V1 experiment focuses on judge bias on the *real policy distribution*.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _normalize_messages(msgs: Any) -> List[Dict[str, str]]:
    if isinstance(msgs, str):
        return [{"role": "user", "content": msgs}]
    if not isinstance(msgs, list):
        return [{"role": "user", "content": str(msgs)}]
    out: List[Dict[str, str]] = []
    for m in msgs:
        if isinstance(m, dict):
            role = str(m.get("role") or "user")
            content = str(m.get("content") or "")
            if content.strip():
                out.append({"role": role, "content": content})
        elif m is not None and str(m).strip():
            out.append({"role": "user", "content": str(m)})
    return out or [{"role": "user", "content": ""}]


def _messages_to_prompt_text(messages: List[Dict[str, str]], model_path: str, trust_remote_code: bool) -> str:
    """
    Convert chat messages to a single prompt string for local vLLM generation.
    Prefer HF tokenizer's chat template when available.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Local vLLM generation requires `transformers` to build chat prompts. "
            "Please install it or use --policy-base-url for remote generation."
        ) from e

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except TypeError:
            # Some tokenizers use different argument names.
            return tok.apply_chat_template(messages, tokenize=False)

    # Fallback: simple transcript. (Less ideal, but better than crashing.)
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}:\n{content}")
    lines.append("assistant:\n")
    return "\n\n".join(lines)


class RRClient:
    def __init__(self, base_urls: List[str], api_key: str, timeout_s: int):
        self.base_urls = [u.rstrip("/") for u in base_urls if u.strip()]
        if not self.base_urls:
            raise ValueError("No base URLs provided")
        self.timeout_s = timeout_s
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer dummy",
        }
        self._i = 0

    def _next_url(self) -> str:
        url = self.base_urls[self._i % len(self.base_urls)]
        self._i += 1
        return url

    def chat(self, model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]) -> str:
        url = self._next_url()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        resp = requests.post(
            f"{url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


class LocalVLLMGenerator:
    def __init__(
        self,
        model_path: str,
        tp_size: int,
        max_model_len: int,
        gpu_mem_util: float,
        dtype: str,
        trust_remote_code: bool,
    ):
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local vLLM generation requested but `vllm` is not importable. "
                "Please install vLLM or provide --policy-base-url for remote generation."
            ) from e
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local vLLM generation requires `transformers` to build chat prompts. "
                "Please install it or use --policy-base-url for remote generation."
            ) from e

        # Keep params minimal; users can tune via args.
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    def messages_to_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except TypeError:
                return tok.apply_chat_template(messages, tokenize=False)

        # Fallback: simple transcript.
        lines: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}:\n{content}")
        lines.append("assistant:\n")
        return "\n\n".join(lines)

    def chat(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]) -> str:
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError("vllm is missing SamplingParams; please check your vLLM install.") from e

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
        )
        prompt = self.messages_to_prompt_text(messages)
        outputs = self.llm.generate([prompt], sp)
        if not outputs:
            return ""
        out0 = outputs[0]
        if not getattr(out0, "outputs", None):
            return ""
        text = out0.outputs[0].text or ""
        return text.strip()

    def generate_batch(self, prompts: List[str], max_tokens: int, temperature: float, top_p: float, top_k: Optional[int]) -> List[str]:
        try:
            from vllm import SamplingParams  # type: ignore
        except Exception as e:
            raise RuntimeError("vllm is missing SamplingParams; please check your vLLM install.") from e

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
        )
        outs = self.llm.generate(prompts, sp)
        texts: List[str] = []
        for o in outs:
            if not getattr(o, "outputs", None):
                texts.append("")
            else:
                texts.append((o.outputs[0].text or "").strip())
        return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input eval JSONL")
    ap.add_argument("--out", dest="out_path", required=True, help="Output eval JSONL (with ypolicy_raw filled)")
    ap.add_argument(
        "--policy-base-url",
        default="",
        help="If set: OpenAI-compatible base URL(s), comma-separated (remote). If empty: use local vLLM generate.",
    )
    ap.add_argument(
        "--policy-model",
        default="",
        help="Served model name for remote policy. Required when --policy-base-url is provided.",
    )
    ap.add_argument(
        "--policy-model-path",
        default="",
        help="Local model path for vLLM generate. Required when --policy-base-url is empty.",
    )
    ap.add_argument("--tp-size", type=int, default=1, help="Local vLLM tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=8192, help="Local vLLM max_model_len")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9, help="Local vLLM gpu_memory_utilization")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="Local vLLM dtype (e.g. bfloat16/float16)")
    ap.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to tokenizer/vLLM")
    ap.add_argument(
        "--local-batch-size",
        type=int,
        default=8,
        help="Local vLLM batch size (number of prompts per llm.generate call). Only applies when --policy-base-url is empty.",
    )
    ap.add_argument(
        "--remote-max-workers",
        type=int,
        default=16,
        help="Remote concurrency (threads) when --policy-base-url is provided.",
    )
    ap.add_argument("--api-key-env", default="VLLM_API_KEY", help="Env var name that stores API key (default: VLLM_API_KEY)")
    ap.add_argument("--timeout-s", type=int, default=180, help="Request timeout seconds")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max tokens for policy generation")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.8, help="Top-p")
    ap.add_argument("--top-k", type=int, default=20, help="Top-k (set <0 to disable)")
    ap.add_argument("--sleep-s", type=float, default=0.0, help="Sleep seconds between requests (to avoid bursts)")
    args = ap.parse_args()

    top_k = None if int(args.top_k) < 0 else int(args.top_k)
    policy_base_url = str(args.policy_base_url or "").strip()
    use_remote = bool(policy_base_url)
    if use_remote:
        api_key = (os.getenv(args.api_key_env, "") or "").strip()
        base_urls = [u.strip() for u in policy_base_url.split(",") if u.strip()]
        if not str(args.policy_model or "").strip():
            raise SystemExit("--policy-model is required when --policy-base-url is provided.")
        client: Any = RRClient(base_urls=base_urls, api_key=api_key, timeout_s=int(args.timeout_s))
    else:
        if not str(args.policy_model_path or "").strip():
            raise SystemExit("--policy-model-path is required when --policy-base-url is empty (local vLLM).")
        client = LocalVLLMGenerator(
            model_path=str(args.policy_model_path),
            tp_size=int(args.tp_size),
            max_model_len=int(args.max_model_len),
            gpu_mem_util=float(args.gpu_mem_util),
            dtype=str(args.dtype),
            trust_remote_code=bool(args.trust_remote_code),
        )

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    pending: List[int] = [i for i, it in enumerate(items) if not (it.get("responses", {}) or {}).get("ypolicy_raw")]

    if use_remote:
        # Remote: concurrent requests (OpenAI-compatible endpoints generally do not support multi-prompt batching in one request).
        def _work(idx: int) -> tuple[int, str]:
            it = items[idx]
            prompt_messages = (it.get("meta") or {}).get("prompt_messages")
            messages = _normalize_messages(prompt_messages or it.get("prompt") or "")
            text = client.chat(
                model=str(args.policy_model),
                messages=messages,
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=top_k,
            )
            if float(args.sleep_s) > 0:
                time.sleep(float(args.sleep_s))
            return idx, text

        max_workers = max(1, int(args.remote_max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_work, idx) for idx in pending]
            for fut in as_completed(futures):
                idx, text = fut.result()
                items[idx].setdefault("responses", {})["ypolicy_raw"] = text
    else:
        # Local: batch vLLM generate
        batch_size = max(1, int(args.local_batch_size))
        # Prebuild prompts
        prompts: List[str] = []
        for idx in pending:
            it = items[idx]
            prompt_messages = (it.get("meta") or {}).get("prompt_messages")
            messages = _normalize_messages(prompt_messages or it.get("prompt") or "")
            prompts.append(client.messages_to_prompt_text(messages))

        # Chunked generation
        for start in range(0, len(pending), batch_size):
            end = min(len(pending), start + batch_size)
            chunk_prompts = prompts[start:end]
            chunk_idxs = pending[start:end]
            texts = client.generate_batch(
                prompts=chunk_prompts,
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=top_k,
            )
            for idx, text in zip(chunk_idxs, texts):
                items[idx].setdefault("responses", {})["ypolicy_raw"] = text
            if float(args.sleep_s) > 0:
                time.sleep(float(args.sleep_s))

    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()

