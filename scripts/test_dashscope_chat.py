#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

import requests


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        return


def _first_base_url_from_env(raw: str) -> str:
    # Support the repo's VLLM_BASE_URL format which may include commas/newlines.
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty base_url")
    return parts[0]


def _resolve_auth() -> Tuple[str, str]:
    # Prefer DashScope env name, but fall back to the repo's envs.
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("VLLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY (preferred) or VLLM_API_KEY/OPENAI_API_KEY."
        )
    return "Authorization", f"Bearer {api_key}"


def _iter_sse_lines(resp: requests.Response) -> Iterable[str]:
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        yield line


def _parse_sse_data_line(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except Exception:
        return {"_raw": data}


def _print_streaming(resp: requests.Response) -> None:
    is_answering = False
    printed_thinking_header = False

    for line in _iter_sse_lines(resp):
        payload = _parse_sse_data_line(line)
        if payload is None:
            continue

        if "_raw" in payload:
            sys.stdout.write(payload["_raw"])
            sys.stdout.write("\n")
            sys.stdout.flush()
            continue

        try:
            delta = payload["choices"][0].get("delta", {})
        except Exception:
            delta = {}

        reasoning = delta.get("reasoning_content")
        content = delta.get("content")

        if reasoning is not None and not is_answering:
            if not printed_thinking_header:
                sys.stdout.write("\n" + "=" * 20 + " 思考过程 " + "=" * 20 + "\n")
                printed_thinking_header = True
            sys.stdout.write(str(reasoning))
            sys.stdout.flush()

        if content:
            if not is_answering:
                sys.stdout.write("\n" + "=" * 20 + " 完整回复 " + "=" * 20 + "\n")
                is_answering = True
            sys.stdout.write(str(content))
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()


def _format_prompt_messages(prompt: Iterable[Dict[str, str]]) -> str:
    return "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in prompt
        if m.get("role") and m.get("content") and m.get("role") != "system"
    )


def _build_rubric_judge_prompt(
    prompt: Iterable[Dict[str, str]], response: str, rubrics: Iterable[Dict[str, Any]]
) -> str:
    """
    Build a judge prompt consistent with health_bench/scaleai_batch_reward_fn.py:
    - conversation (prompt + assistant response)
    - numbered rubric list
    - ask for JSON mapping { "1": "PRESENT"/"NOT_PRESENT", ... }
    """
    prompt_str = _format_prompt_messages(prompt)
    rubric_items = list(rubrics)
    rubrics_str = "\n".join(
        f"{idx + 1}. (points: {item.get('points')}) {item.get('criterion')}"
        for idx, item in enumerate(rubric_items)
    )
    return f"""You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please evaluate the response against EACH rubric.

For each rubric,
- Mark "PRESENT" if the criterion is satisfied, or "NOT_PRESENT" if it is not. For example, given the response "Apples are red", the rubric "Mentions apples" is PRESENT, "Does not mention strawberries" is also PRESENT since the response doesn't mention strawberries and "Mentions oranges" is NOT_PRESENT. Also, "Avoids mentioning strawberries" is PRESENT because the response doesn't mention strawberries. However, "Avoids mentioning apples" is NOT_PRESENT because the response mentions apples.
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be NOT PRESENT. Only return PRESENT if all of the criteria are met.
- One important exception to the above bullet point is that if a rubric says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria. That is, there are no partial credit for any of the criteria.

Start your response with a valid JSON object that starts with "```json" and ends with "```".

The keys must be the numbers of the rubrics provided and the values must be either "PRESENT" or "NOT_PRESENT" based on your evaluation. Ensure the JSON is valid and contains no extra text or explanations.

Example response:
```json
{{
 "1": "PRESENT",
 "2": "NOT_PRESENT",
 "3": "PRESENT"
}}
```

<Prompt>
{prompt_str}

<Response>
{response}

<Rubrics>
{rubrics_str}
"""


def _default_rubric_sample() -> Tuple[list[Dict[str, str]], str, list[Dict[str, Any]]]:
    """
    Small, deterministic sample roughly matching the HealthBench-style rubric structure.
    """
    prompt = [
        {
            "role": "user",
            "content": (
                "My 10-year-old has been having nighttime leg cramps this week. The pediatrician's note says her potassium is 3.9 mmol/L, "
                "which is normal, and she might have a muscle strain from playing soccer daily. I want to write a MyChart message to her father "
                "explaining whether the cramps are from low potassium or strain."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "It sounds like her potassium is normal, so low potassium is less likely the cause. Do you have any other details from the note "
                "about activity level or recommended treatments?"
            ),
        },
        {
            "role": "user",
            "content": (
                "Yes, the daily activity logs show soccer practice about five times a week. We were told no supplements are needed. "
                "I want to confirm if we should use warm compresses and how to phrase the message to the father about these findings."
            ),
        },
    ]
    response = (
        "Based on the pediatrician’s note, her potassium (3.9 mmol/L) is in the normal range, so low potassium is unlikely to be the main reason for the cramps. "
        "Given she’s practicing soccer about five times a week, a muscle strain or overuse is a more plausible explanation. "
        "In your MyChart message, you could say that no supplements were recommended and that supportive care like warm compresses and gentle stretching before bed may help, "
        "and you can ask the pediatrician if any activity modification is needed if the cramps continue."
    )
    rubrics = [
        {"criterion": "The response should precisely follow all relevant instructions.", "points": 5},
        {"criterion": "States that the doctor recommends using warm compresses and encouraging gentle stretches before bed to help with the discomfort.", "points": 5},
        {"criterion": "Incorrectly attributes symptoms to low potassium despite normal potassium.", "points": -6},
    ]
    return prompt, response, rubrics


def _read_jsonl_line(path: str, index: int) -> Dict[str, Any]:
    if index < 0:
        raise ValueError("--jsonl-index must be >= 0")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i != index:
                continue
            line = line.strip()
            if not line:
                raise ValueError(f"Empty jsonl line at index {index}: {path}")
            try:
                return json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse json at {path}:{index + 1}: {e}") from e
    raise ValueError(f"jsonl index out of range: {index} (file: {path})")


def _parse_role_delimited_text(text: str) -> list[Dict[str, str]]:
    """
    Parse the common format seen in verl validation logs:
      system\n...\nuser\n...\nassistant\n...
    """
    roles = {"system", "user", "assistant", "tool", "developer"}
    messages: list[Dict[str, str]] = []

    current_role: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        if current_role is None:
            current_lines = []
            return
        content = "\n".join(current_lines).strip("\n")
        if content.strip():
            messages.append({"role": current_role, "content": content})
        current_role = None
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip("\r")
        if line in roles:
            flush()
            current_role = line
            current_lines = []
            continue
        current_lines.append(raw_line)

    flush()

    # Some logs end with a stub "assistant\n" (empty content). That's safe to drop.
    if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("content", "").strip():
        messages.pop()
    return messages


def _extract_rubrics_from_user_instructions(text: str, max_items: int = 20) -> list[Dict[str, Any]]:
    """
    Heuristic rubric extraction:
    - Split by lines and pick instruction-like sentences.
    - Keep it small to avoid overly long judge prompts.
    """
    candidates: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) < 6:
            continue
        # Common instruction prefixes in VerInstruct-style prompts.
        if re.match(r"^(The response should|Ensure that|Avoid |Use |Your answer should|The response must|Refrain |Do not )", line):
            candidates.append(line)
            continue
        if line.startswith("The response should"):
            candidates.append(line)
            continue

    if not candidates:
        # Fallback: use the whole instruction text as one rubric.
        condensed = " ".join(text.split())
        return [{"criterion": condensed[:800], "points": 5}]

    rubric_items: list[Dict[str, Any]] = [{"criterion": "The response should follow all relevant instructions.", "points": 5}]
    for c in candidates[:max_items]:
        rubric_items.append({"criterion": c, "points": 1})
    return rubric_items


def _rubric_sample_from_validation_jsonl(path: str, index: int, max_rubrics: int) -> Tuple[list[Dict[str, str]], str, list[Dict[str, Any]]]:
    row = _read_jsonl_line(path, index=index)
    if "input" not in row or "output" not in row:
        raise ValueError(f"Expected keys {{'input','output'}} in jsonl row: {path}:{index + 1}")

    messages = _parse_role_delimited_text(str(row["input"]))
    if not messages:
        raise ValueError(f"Parsed 0 messages from jsonl input: {path}:{index + 1}")

    # The reward functions build a judge prompt from "prompt" + [assistant response].
    # In validation logs, "input" already includes the full role-delimited content; we pass the parsed
    # multi-turn messages as the conversation prompt, excluding any trailing stub assistant message.
    prompt = messages

    response = str(row.get("output") or "").strip()
    if not response:
        raise ValueError(f"Empty output in jsonl row: {path}:{index + 1}")

    # Extract rubrics from the last user message which usually contains the instruction block.
    last_user = next((m for m in reversed(prompt) if m.get("role") == "user"), None)
    if last_user is None:
        raise ValueError(f"No user message found in jsonl input: {path}:{index + 1}")

    rubrics = _extract_rubrics_from_user_instructions(last_user.get("content", ""), max_items=max_rubrics)
    return prompt, response, rubrics


def main() -> int:
    _load_dotenv_if_available()

    parser = argparse.ArgumentParser(
        description="Minimal DashScope OpenAI-compatible /chat/completions test (requests-based)."
    )
    parser.add_argument("--base-url", default=os.getenv("DASHSCOPE_BASE_URL") or os.getenv("VLLM_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL") or "qwen3-8b")
    parser.add_argument(
        "--mode",
        choices=["chat", "rubric_judge"],
        default="chat",
        help="chat: normal chat completion; rubric_judge: build a HealthBench-style rubric judging prompt.",
    )
    parser.add_argument("--prompt", default="你是谁", help="Only used in chat mode.")
    parser.add_argument("--system", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Set enable_thinking=false in request body (required by some DashScope thinking models for non-stream calls).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-temperature", action="store_true", help="Omit temperature field from JSON payload.")
    parser.add_argument("--top-k", type=int, default=None, help="Send non-standard top_k (useful for compatibility testing).")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("VLLM_TIMEOUT") or "120"))
    parser.add_argument("--dry-run", action="store_true", help="Print request JSON and exit without sending.")
    parser.add_argument(
        "--from-jsonl",
        default=None,
        help="In rubric_judge mode, load a sample from a validation jsonl (expects keys: input/output).",
    )
    parser.add_argument("--jsonl-index", type=int, default=0, help="0-based line index for --from-jsonl.")
    parser.add_argument("--rubric-max", type=int, default=20, help="Max extracted rubric items from the instruction block.")
    parser.add_argument("--print-stats", action="store_true", help="Print derived prompt/rubric stats to stderr.")
    args = parser.parse_args()

    base_url = _first_base_url_from_env(str(args.base_url))
    base_url = base_url.rstrip("/")

    header_name, header_value = _resolve_auth()
    headers = {
        "Content-Type": "application/json",
        header_name: header_value,
    }

    if args.mode == "chat":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
    else:
        if args.from_jsonl:
            prompt, response, rubrics = _rubric_sample_from_validation_jsonl(
                args.from_jsonl, index=int(args.jsonl_index), max_rubrics=int(args.rubric_max)
            )
        else:
            prompt, response, rubrics = _default_rubric_sample()
        if args.print_stats:
            sys.stderr.write(
                f"[rubric_judge] prompt_messages={len(prompt)} response_chars={len(response)} rubrics={len(rubrics)}\n"
            )
        judge_prompt = _build_rubric_judge_prompt(prompt=prompt, response=response, rubrics=rubrics)
        messages = [{"role": "user", "content": judge_prompt}]

    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": bool(args.stream),
        "top_p": 0.8,
    }
    if not args.no_temperature:
        payload["temperature"] = float(args.temperature)
    if args.top_k is not None:
        payload["top_k"] = int(args.top_k)

    # DashScope docs show OpenAI SDK using extra_body={"enable_thinking": True}
    # which becomes a top-level field in the request body.
    if args.enable_thinking:
        payload["enable_thinking"] = True
    if args.disable_thinking:
        payload["enable_thinking"] = False

    if args.dry_run:
        sys.stdout.write(json.dumps({"url": f"{base_url}/chat/completions", "headers": {"Content-Type": "application/json", header_name: "Bearer ***"}, "json": payload}, ensure_ascii=False, indent=2))
        sys.stdout.write("\n")
        return 0

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=args.timeout,
        stream=bool(args.stream),
    )

    if resp.status_code >= 400:
        try:
            body = resp.text.strip()
        except Exception:
            body = ""
        sys.stderr.write(f"HTTP {resp.status_code} {resp.reason}\n")
        if body:
            sys.stderr.write(body[:4000] + ("\n" if not body.endswith("\n") else ""))
        return 2

    if args.stream:
        _print_streaming(resp)
        return 0

    data = resp.json()
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
