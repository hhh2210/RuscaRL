#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_dotenv_if_present() -> None:
    env_path = _REPO_ROOT / ".env"
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
        v = v.strip()
        if not k:
            continue
        os.environ.setdefault(k, v)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_score_anywhere(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\[\[\s*([01])\s*\]\]", str(text))
    if not m:
        return None
    return int(m.group(1))


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


@dataclass(frozen=True)
class JudgeResult:
    score: int
    score_anywhere: Optional[int]
    parse_ok: bool
    attempts: int
    llm_response: str


def _judge_prompt_with_retry(
    prompt: str,
    *,
    max_retries: int,
    retry_backoff_s: float,
    parse_mode: str,
    strict_suffix: str,
) -> JudgeResult:
    from verl.utils.reward_score.local_server import llm_call

    messages = [{"role": "user", "content": prompt}]
    last_text = ""
    score_anywhere_last: Optional[int] = None

    for attempt in range(max_retries + 1):
        last_text = llm_call.generate_chat(messages, max_tokens=4096, temperature=0.0)
        score_anywhere_last = _extract_score_anywhere(last_text)
        strict_score = llm_call.extract_score(last_text)

        score: Optional[int] = None
        parse_ok = False

        if parse_mode == "strict":
            score = strict_score
            parse_ok = strict_score is not None
        elif parse_mode == "anywhere":
            score = score_anywhere_last
            parse_ok = score_anywhere_last is not None
        elif parse_mode == "either":
            score = strict_score if strict_score is not None else score_anywhere_last
            parse_ok = score is not None
        else:
            raise ValueError(f"Unknown parse_mode={parse_mode!r}")

        if score is not None:
            return JudgeResult(
                score=int(score),
                score_anywhere=score_anywhere_last,
                parse_ok=parse_ok,
                attempts=attempt + 1,
                llm_response=last_text,
            )

        # Retry with a stricter formatting instruction.
        messages = [{"role": "user", "content": prompt + strict_suffix}]
        if retry_backoff_s > 0:
            time.sleep(retry_backoff_s * (attempt + 1))

    # Match RuscaRL llm_call.llm_score contract: unparseable => 0.
    return JudgeResult(
        score=0,
        score_anywhere=score_anywhere_last,
        parse_ok=False,
        attempts=max_retries + 1,
        llm_response=last_text,
    )


def _parse_llm_score_prompt(prompt: str) -> Tuple[str, str, Any]:
    """
    Parse a prompt that was constructed by `llm_call.llm_score` into:
      - instruction (string)
      - response (string)
      - checkers (usually a list[str])

    This enables re-scoring via the *original* `llm_call.llm_score` function,
    instead of directly sending the already-constructed prompt.
    """
    text = str(prompt)

    m_instr = re.search(r"\[指令\]\s*(.*?)\n\s*\[回复\]", text, flags=re.DOTALL)
    m_resp = re.search(r"\[回复\]\s*(.*?)\n\s*\[约束\]", text, flags=re.DOTALL)
    # For constraints, search from the [约束] marker forward to avoid matching the header.
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
        if isinstance(parsed, (list, tuple)):
            checkers = list(parsed)
        else:
            checkers = parsed
    except Exception:
        # Some dumps may contain JSON-style list strings.
        try:
            parsed = json.loads(checkers_raw)
            checkers = parsed
        except Exception:
            checkers = checkers_raw

    return instruction, response, checkers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score a list of already-constructed VerIF judge prompts by re-calling the judge model.\n"
            "Input format: JSON array (or JSONL) with at least a 'prompt' field per item."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON/JSONL file. Each item must contain a 'prompt' string.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Output JSONL for incremental saving (recommended). Default: <output-json>.jsonl",
    )
    parser.add_argument(
        "--output-json",
        default="llm_score_extracted_Rusca_sorted.rescored.json",
        help=(
            "Final output JSON array path (written at end). "
            "Default: llm_score_extracted_Rusca_sorted.rescored.json"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel judge workers (default: 1).")
    parser.add_argument("--max-items", type=int, default=0, help="Process only first N items (0 means all).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL by skipping already-scored prompt hashes.",
    )
    parser.add_argument(
        "--parse-mode",
        choices=["strict", "anywhere", "either"],
        default="strict",
        help=(
            "How to parse judge output into 0/1.\n"
            "  strict: use RuscaRL llm_call.extract_score (expects [[0]]/[[1]] at start)\n"
            "  anywhere: accept [[0]]/[[1]] anywhere in the response\n"
            "  either: strict if possible else anywhere"
        ),
    )
    parser.add_argument(
        "--call-mode",
        choices=["raw_prompt", "llm_score"],
        default="llm_score",
        help=(
            "How to re-score each record.\n"
            "  raw_prompt: send the stored `prompt` directly to the judge model\n"
            "  llm_score: parse stored `prompt` into (instruction/response/checkers) and call "
            "`verl.utils.reward_score.local_server.llm_call.llm_score` (matches RL training path)"
        ),
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Parse-level retries (default: 2).")
    parser.add_argument(
        "--retry-backoff-s",
        type=float,
        default=0.0,
        help="Optional parse-level retry backoff (seconds). Default: 0.",
    )
    parser.add_argument(
        "--strict-suffix",
        default="\n\n请只输出[[0]]或[[1]]，不要输出其他内容。",
        help="Appended on retries to force strict output format.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["idx", "new_score", "delta"],
        default="idx",
        help=(
            "How to sort the final JSON output.\n"
            "  idx: keep input order\n"
            "  new_score: group failures first (0 then 1)\n"
            "  delta: sort by (new_score - old_score) descending"
        ),
    )

    args = parser.parse_args()

    _load_dotenv_if_present()

    input_path = Path(args.input)
    output_json = Path(args.output_json)
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
    else:
        # Prefer a clean sibling JSONL file name like `foo.jsonl` instead of `foo.json.jsonl`.
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

    # Pre-validate prompts.
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
    backend = os.getenv("VERIF_LLM_BACKEND", "").strip() or "auto"
    model = os.getenv("VLLM_MODEL", "").strip() or os.getenv("VERIF_MODEL_NAME", "").strip() or "unknown"
    base_url = (os.getenv("VLLM_BASE_URL", "") or os.getenv("VERIF_API_URL", "")).strip()
    print(f"Scoring {total} prompts | backend={backend} | model={model} | base_url={base_url or 'unset'}")
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
                jr = _judge_prompt_with_retry(
                    prompt,
                    max_retries=args.max_retries,
                    retry_backoff_s=args.retry_backoff_s,
                    parse_mode=args.parse_mode,
                    strict_suffix=args.strict_suffix,
                )
            else:
                from verl.utils.reward_score.local_server import llm_call

                instruction, response, checkers = _parse_llm_score_prompt(prompt)
                new_score = llm_call.llm_score(instruction, response, checkers)
                # Best-effort capture of judge output via the debug hook (only reliable in single-thread).
                new_judge_text = ""
                if os.getenv("RUSCARL_CAPTURE_JUDGE_TEXT", "").strip().lower() in ("1", "true", "yes", "y", "on"):
                    try:
                        new_judge_text = (llm_call._LAST_VERIF_JUDGE_DEBUG.get("judge_text") or "")
                    except Exception:
                        new_judge_text = ""
                jr = JudgeResult(
                    score=int(new_score),
                    score_anywhere=_extract_score_anywhere(new_judge_text) if new_judge_text else None,
                    parse_ok=True,
                    attempts=1,
                    llm_response=new_judge_text,
                )
            delta = None
            try:
                if isinstance(old_score, (int, float, bool)):
                    delta = float(jr.score) - float(old_score)
            except Exception:
                delta = None
            return {
                "idx": idx,
                "prompt": prompt,
                "prompt_sha256": h,
                "source_file": source_file,
                "old_score": old_score,
                "old_llm_response": old_llm_response,
                "new_score": jr.score,
                "new_score_anywhere": jr.score_anywhere,
                "delta": delta,
                "parse_ok": jr.parse_ok,
                "attempts": jr.attempts,
                "new_llm_response": jr.llm_response,
            }
        except Exception as e:
            return {
                "idx": idx,
                "prompt": prompt,
                "prompt_sha256": h,
                "source_file": source_file,
                "old_score": old_score,
                "old_llm_response": old_llm_response,
                "new_score": 0,
                "new_score_anywhere": None,
                "delta": None,
                "parse_ok": False,
                "attempts": 0,
                "new_llm_response": "",
                "error": repr(e),
            }

    # Execute with optional concurrency.
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futures = [ex.submit(_work, item) for item in prompts]
        for fut in as_completed(futures):
            rec = fut.result()
            scored.append(rec)
            _write_jsonl(output_jsonl, rec)

            # Periodic progress.
            if len(scored) % 25 == 0 or len(scored) == total:
                done = len(scored)
                mean = sum(r["new_score"] for r in scored) / done
                elapsed = time.time() - start_t
                rate = done / max(1e-9, elapsed)
                print(f"[{done}/{total}] mean={mean:.6f} ({rate:.3f} items/s)")

    # Final: consolidate from JSONL (supports --resume and crash-safe partial runs).
    by_sha: Dict[str, Dict[str, Any]] = {}
    for i, rec in enumerate(_iter_jsonl(output_jsonl)):
        sha = rec.get("prompt_sha256")
        if not isinstance(sha, str) or not sha:
            sha = f"__line_{i}"
        by_sha[sha] = rec
    all_scored = list(by_sha.values())

    def _sort_key(rec: Dict[str, Any]):
        idx = rec.get("idx", 10**18)
        if args.sort_by == "idx":
            return (idx,)
        if args.sort_by == "new_score":
            old = rec.get("old_score", 0)
            old_v = float(old) if isinstance(old, (int, float, bool)) else 0.0
            return (rec.get("new_score", 0), old_v, idx)
        if args.sort_by == "delta":
            delta = rec.get("delta")
            delta_v = float(delta) if isinstance(delta, (int, float)) else 0.0
            return (-delta_v, idx)
        return (idx,)

    scored_sorted = sorted(all_scored, key=_sort_key)
    output_json.write_text(json.dumps(scored_sorted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mean = sum(r["new_score"] for r in scored_sorted) / len(scored_sorted)
    elapsed = time.time() - start_t
    print("\nDone.")
    print(f"scored_n={len(scored_sorted)}")
    print(f"score_mean={mean}")
    print(f"elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
