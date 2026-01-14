#!/usr/bin/env python3
"""
Rescore a list of already-constructed VerIF judge prompts using the *VerlF* (VerIF) reward code.

Why this script exists
----------------------
RuscaRL and VerIF have slightly different judge / parsing behavior. This script helps you
isolate "judge model issue" vs "reward function / parsing issue" by re-scoring the *same*
stored `prompt` field with VerlF's VerIF judge logic.

Input
-----
`--input` expects a JSON array (or JSONL) where each item contains at least:
  - prompt: str

Config
------
By default we load `/home/haozy/RuscaRL/.env` and use:
  - VLLM_BASE_URL (or BASE_URL)
  - VLLM_MODEL    (or MODEL)
  - VLLM_API_KEY  (or API_KEY / OPENAI_API_KEY)

Important: We force-import VerlF's `verl` package by prepending `--verlf-root` to sys.path,
so this script won't accidentally import RuscaRL's `verl` fork.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        return value[1:-1]
    return value


def _load_dotenv_if_present(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v)
        if not k:
            continue
        # Do not override shell env by default.
        os.environ.setdefault(k, v)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if suffix == ".jsonl":
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    raise ValueError(f"Unsupported JSON structure in {path}: expected list or dict with 'data' list.")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_llm_score_prompt(prompt: str) -> Tuple[str, str, Any]:
    """
    Parse a prompt that was constructed by `llm_call.llm_score` into:
      - instruction (string)
      - response (string)
      - checkers (usually a list[str])

    Expected format includes these markers:
      [指令] ... [回复] ... [约束] ...
    """
    text = str(prompt)

    m_instr = re.search(r"\[指令\]\s*(.*?)\n\s*\[回复\]", text, flags=re.DOTALL)
    m_resp = re.search(r"\[回复\]\s*(.*?)\n\s*\[约束\]", text, flags=re.DOTALL)

    idx_constraints = text.find("[约束]")
    if idx_constraints >= 0:
        tail = text[idx_constraints:]
        m_cons = re.search(
            r"\[约束\]\s*(.*?)\n\s*请判断给定的回复是否遵循指令中的约束",
            tail,
            flags=re.DOTALL,
        )
    else:
        m_cons = None

    if not m_instr or not m_resp or not m_cons:
        raise ValueError("Cannot parse llm_score prompt into [指令]/[回复]/[约束] sections.")

    instruction = m_instr.group(1).strip()
    response = m_resp.group(1).strip()
    checkers_raw = m_cons.group(1).strip()

    # Best-effort parsing for the checker list literal (often printed as Python list).
    checkers: Any = checkers_raw
    try:
        parsed = ast.literal_eval(checkers_raw)
        checkers = parsed
    except Exception:
        try:
            parsed = json.loads(checkers_raw)
            checkers = parsed
        except Exception:
            checkers = checkers_raw

    return instruction, response, checkers


def _build_verif_llm_score_prompt(instruction: str, response: str, checkers: Any) -> str:
    # Match VerlF VerIF implementation (see VerlF/verl/utils/reward_score/local_server/llm_call.py:llm_score).
    return f"""
    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    
    [指令]
    {instruction}

    [回复]
    {response}

    [约束]
    {checkers}

    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    请在回答的最开始用[[score]]格式输出你的分数。
    如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
    """


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score a list of already-constructed VerIF judge prompts by calling VerlF's VerIF judge.\n"
            "Input format: JSON array (or JSONL) with at least a 'prompt' field per item."
        )
    )
    parser.add_argument(
        "--input",
        default="/home/haozy/RuscaRL/llm_score_extracted_Rusca_sorted.json",
        help="Input JSON/JSONL file. Each item must contain a 'prompt' string.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Output JSONL for incremental saving. Default: <output-json>.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="llm_score_extracted_Rusca_sorted.verlf_verif_rescored.json",
        help="Final output JSON array path (written at end).",
    )
    parser.add_argument(
        "--dotenv",
        default="/home/haozy/RuscaRL/.env",
        help="Path to dotenv file for BASE_URL/model/api key (default: RuscaRL/.env).",
    )
    parser.add_argument(
        "--verlf-root",
        default="/home/haozy/VerlF",
        help="Path to VerlF repo root (must contain 'verl/' folder).",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel judge workers (default: 1).")
    parser.add_argument("--max-items", type=int, default=0, help="Process only first N items (0 means all).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL by skipping already-scored prompt hashes.",
    )
    parser.add_argument(
        "--call-mode",
        choices=["raw_prompt", "llm_score"],
        default="raw_prompt",
        help=(
            "How to score each record.\n"
            "  raw_prompt: send the stored `prompt` directly to the judge model\n"
            "  llm_score: parse stored `prompt` into (instruction/response/checkers), then rebuild the VerlF llm_score prompt\n"
            "            (useful if the stored prompt was modified by another codepath)."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for judge generation.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=0.0,
        help="Optional sleep between requests per worker (seconds). Default: 0.",
    )
    parser.add_argument("--base-url", default=None, help="Override BASE_URL/VLLM_BASE_URL for judge endpoint.")
    parser.add_argument("--model", default=None, help="Override MODEL/VLLM_MODEL for judge model name.")
    parser.add_argument("--api-key", default=None, help="Override API_KEY/VLLM_API_KEY for judge auth.")

    args = parser.parse_args()

    # Load env first (so overrides can come from args).
    _load_dotenv_if_present(Path(args.dotenv))

    # Force `import verl` to point to VerlF's code.
    verlf_root = Path(args.verlf_root).resolve()
    if not (verlf_root / "verl").exists():
        raise FileNotFoundError(f"--verlf-root does not look like a repo root: missing {verlf_root / 'verl'}")
    sys.path.insert(0, str(verlf_root))

    try:
        from verl.utils.reward_score.local_server import llm_call as verif_llm_call  # type: ignore
        from verl.utils.reward_score.local_server.build_model import APIModel  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import VerlF VerIF judge code. "
            "Make sure you are running inside `conda activate verl` and that VerlF dependencies are installed."
        ) from e

    base_url = (
        args.base_url
        or os.getenv("VLLM_BASE_URL")
        or os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()
    model = (
        args.model
        or os.getenv("VLLM_MODEL")
        or os.getenv("MODEL")
        or os.getenv("VERIF_MODEL_NAME")
        or ""
    ).strip()
    api_key = (
        args.api_key
        or os.getenv("VLLM_API_KEY")
        or os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    if base_url and model:
        # Override the global API model inside VerlF's llm_call module.
        verif_llm_call.api_url = base_url
        verif_llm_call.model_name = model
        verif_llm_call.api_model = APIModel(base_url, model, api_key=api_key or "EMPTY")

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
    else:
        output_jsonl = output_json.with_suffix(".jsonl") if output_json.suffix else Path(str(output_json) + ".jsonl")

    records = _load_records(input_path)
    if args.max_items and args.max_items > 0:
        records = records[: args.max_items]

    # Build resume set.
    done_hashes = set()
    if args.resume and output_jsonl.exists():
        for rec in _iter_jsonl(output_jsonl):
            h = rec.get("prompt_sha256")
            if isinstance(h, str) and h:
                done_hashes.add(h)

    prompts: List[Tuple[int, str, Dict[str, Any]]] = []
    for idx, rec in enumerate(records):
        prompt = rec.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        h = _sha256(prompt)
        if h in done_hashes:
            continue
        prompts.append((idx, prompt, rec))

    total = len(prompts)
    if total == 0:
        print("No prompts to score (either empty input or all done).")
        return

    # Print minimal run context (no secrets).
    print(f"Scoring {total} prompts with VerlF VerIF judge")
    print(f"  verlf_root={verlf_root}")
    print(f"  base_url={base_url or getattr(verif_llm_call, 'api_url', '') or 'unset'}")
    print(f"  model={model or getattr(verif_llm_call, 'model_name', '') or 'unset'}")
    print(f"  api_key={'set' if api_key else 'unset'}")
    print(f"  call_mode={args.call_mode} | max_workers={args.max_workers}")
    print(f"Output (jsonl): {output_jsonl}")
    print(f"Output (json):  {output_json}")

    scored: List[Dict[str, Any]] = []
    start_t = time.time()

    def _work(item: Tuple[int, str, Dict[str, Any]]) -> Dict[str, Any]:
        idx, prompt, rec = item
        h = _sha256(prompt)
        old_score = rec.get("score")
        old_llm_response = rec.get("llm_response")
        source_file = rec.get("source_file")

        try:
            if args.call_mode == "raw_prompt":
                judge_prompt = prompt
            else:
                instruction, response, checkers = _parse_llm_score_prompt(prompt)
                judge_prompt = _build_verif_llm_score_prompt(instruction, response, checkers)

            judge_text = verif_llm_call.generate_chat(
                [{"role": "user", "content": judge_prompt}],
                max_tokens=int(args.max_tokens),
                temperature=float(args.temperature),
            )
            verif_score = int(verif_llm_call.extract_score(judge_text))

            delta = None
            try:
                if isinstance(old_score, (int, float, bool)):
                    delta = float(verif_score) - float(old_score)
            except Exception:
                delta = None

            if args.min_interval_s and args.min_interval_s > 0:
                time.sleep(float(args.min_interval_s))

            return {
                "idx": idx,
                "prompt": prompt,
                "prompt_sha256": h,
                "source_file": source_file,
                "old_score": old_score,
                "old_llm_response": old_llm_response,
                "verif_score": verif_score,
                "delta": delta,
                "verif_llm_response": judge_text,
            }
        except Exception as e:
            return {
                "idx": idx,
                "prompt": prompt,
                "prompt_sha256": h,
                "source_file": source_file,
                "old_score": old_score,
                "old_llm_response": old_llm_response,
                "verif_score": 0,
                "delta": None,
                "verif_llm_response": "",
                "error": repr(e),
            }

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [ex.submit(_work, item) for item in prompts]
        for fut in as_completed(futures):
            rec = fut.result()
            scored.append(rec)
            _write_jsonl(output_jsonl, rec)

            if len(scored) % 25 == 0 or len(scored) == total:
                done = len(scored)
                mean = sum(r.get("verif_score", 0) for r in scored) / done
                elapsed = time.time() - start_t
                rate = done / max(1e-9, elapsed)
                print(f"[{done}/{total}] mean(verif_score)={mean:.6f} ({rate:.3f} items/s)")

    # Final: consolidate from JSONL (supports --resume and crash-safe partial runs).
    by_sha: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(_iter_jsonl(output_jsonl)):
        sha = rec.get("prompt_sha256")
        if not isinstance(sha, str) or not sha:
            sha = f"__line_{i}"
        by_sha[sha] = rec
    all_scored = list(by_sha.values())

    all_scored.sort(key=lambda r: (r.get("idx", 10**18),))
    output_json.write_text(json.dumps(all_scored, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mean = sum(r.get("verif_score", 0) for r in all_scored) / max(1, len(all_scored))
    elapsed = time.time() - start_t
    print("\nDone.")
    print(f"scored_n={len(all_scored)}")
    print(f"verif_score_mean={mean}")
    print(f"elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
