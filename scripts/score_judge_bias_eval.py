#!/usr/bin/env python3
"""
Score judge-bias eval items for multiple judge configurations, focusing on ypolicy_raw.

This is V1: evaluate judge bias on the *real policy output distribution*.

Supported sources:
  - healthbench: uses `health_bench/scaleai_batch_reward_fn.py::compute_score`
  - verinstruct_verif: uses `verl/utils/reward_score/local_server/constraint_analyzer.py::evaluate_if_reward_multi`

Input: eval JSONL with `responses.ypolicy_raw` filled.
Output:
  - JSONL per-judge per-item scores
  - Summary JSON with mean scores + pass rates

Judge config JSON example:
[
  {
    "name": "qwen-flash",
    "vllm_base_url": "http://host:8001/v1,http://host:8002/v1",
    "vllm_model": "grader",
    "api_key_env": "VLLM_API_KEY"
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is importable (so `health_bench`, `verl`, etc. work even when running from elsewhere).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _reset_cached_graders() -> None:
    # Reset VerIF verifier sampler cache
    try:
        from verl.utils.reward_score.local_server import llm_call
        llm_call._global_sampler = None  # type: ignore[attr-defined]
    except Exception:
        pass

    # Reset HealthBench grader cache
    try:
        from health_bench import scaleai_batch_reward_fn
        scaleai_batch_reward_fn._global_grader = None  # type: ignore[attr-defined]
    except Exception:
        pass


def _configure_judge_env(j: Dict[str, Any]) -> None:
    # Ensure VerIF uses vLLM backend when possible.
    os.environ["VERIF_LLM_BACKEND"] = "vllm"

    if "vllm_base_url" in j and j["vllm_base_url"]:
        os.environ["VLLM_BASE_URL"] = str(j["vllm_base_url"])
    if "vllm_model" in j and j["vllm_model"]:
        os.environ["VLLM_MODEL"] = str(j["vllm_model"])

    # Judge sampling controls (applies to both HealthBench grader and VerIF verifier backend).
    if "temperature" in j and j["temperature"] is not None:
        os.environ["VLLM_TEMPERATURE"] = str(j["temperature"])
        os.environ["VERIF_LLM_TEMPERATURE"] = str(j["temperature"])

    api_key_env = str(j.get("api_key_env") or "VLLM_API_KEY")
    if api_key_env and os.getenv(api_key_env):
        # VLLMSampler reads VLLM_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY; keep user choice.
        os.environ.setdefault("VLLM_API_KEY", os.getenv(api_key_env, "") or "")


def _score_healthbench(item: Dict[str, Any], answer: str) -> float:
    from health_bench.scaleai_batch_reward_fn import compute_score

    # Build extra_info in the same shape as parquet provides.
    prompt_messages = (item.get("meta") or {}).get("prompt_messages")
    rubrics = (item.get("spec") or {}).get("rubrics") or []
    extra_info = {
        "prompt": prompt_messages,
        "reward_model": {"rubrics": rubrics},
    }
    return float(compute_score(data_source="healthbench", solution_str=answer, ground_truth="", extra_info=extra_info))


def _score_verinstruct_verif(item: Dict[str, Any], answer: str) -> float:
    from verl.utils.reward_score.local_server.constraint_analyzer import evaluate_if_reward_multi

    spec = item.get("spec") or {}
    checkers = spec.get("checkers") or []
    functions = spec.get("functions") or []
    instruction = str(item.get("prompt") or "")
    result = evaluate_if_reward_multi(instruction, [answer], checker_names=checkers, functions=functions, skip_rules=True)
    overall = result.get("overall") or [0.0]
    try:
        return float(overall[0])
    except Exception:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Eval JSONL with ypolicy_raw filled")
    ap.add_argument("--judge-config", required=True, help="Path to JSON list of judge configs")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional override: set temperature for ALL judges (unless a judge explicitly sets temperature in judge-config).",
    )
    ap.add_argument(
        "--save-judge-responses",
        action="store_true",
        help="Also dump raw judge prompts/responses (best-effort) into scores.jsonl. Sets RUSCARL_CAPTURE_JUDGE_TEXT=1.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    judges = json.loads(Path(args.judge_config).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    summary: Dict[str, Any] = {"input": str(in_path), "judges": {}}
    out_jsonl = out_dir / "scores.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fo:
        for j in judges:
            name = str(j.get("name") or "judge")
            if args.temperature is not None and (j.get("temperature") is None):
                j = dict(j)
                j["temperature"] = float(args.temperature)
            _configure_judge_env(j)
            _reset_cached_graders()
            if args.save_judge_responses:
                os.environ["RUSCARL_CAPTURE_JUDGE_TEXT"] = "1"

            scores: List[Tuple[str, str, float]] = []
            for it in items:
                ans = (it.get("responses") or {}).get("ypolicy_raw") or ""
                src = it.get("source")
                if not ans.strip():
                    sc = 0.0
                elif src == "healthbench":
                    sc = _score_healthbench(it, ans)
                elif src == "verinstruct_verif":
                    sc = _score_verinstruct_verif(it, ans)
                else:
                    sc = 0.0

                scores.append((str(it.get("id")), str(src), float(sc)))
                debug: Dict[str, Any] | None = None
                if args.save_judge_responses:
                    if src == "healthbench":
                        try:
                            from health_bench import scaleai_batch_reward_fn
                            debug = dict(getattr(scaleai_batch_reward_fn, "_LAST_JUDGE_DEBUG", {}) or {})
                        except Exception:
                            debug = None
                    elif src == "verinstruct_verif":
                        try:
                            from verl.utils.reward_score.local_server import llm_call
                            debug = dict(getattr(llm_call, "_LAST_VERIF_JUDGE_DEBUG", {}) or {})
                        except Exception:
                            debug = None
                fo.write(
                    json.dumps(
                        {"judge": name, "id": it.get("id"), "source": src, "score": float(sc), "judge_debug": debug},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Summaries
            by_src: Dict[str, List[float]] = {}
            for _, src, sc in scores:
                by_src.setdefault(src, []).append(sc)

            judge_sum: Dict[str, Any] = {"n": len(scores), "by_source": {}}
            for src, arr in by_src.items():
                mean = sum(arr) / max(1, len(arr))
                pass_rate = sum(1 for x in arr if x > 0.5) / max(1, len(arr))
                judge_sum["by_source"][src] = {"n": len(arr), "mean": mean, "pass_rate@0.5": pass_rate}
            summary["judges"][name] = judge_sum

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote per-item scores -> {out_jsonl}")
    print(f"Wrote summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

