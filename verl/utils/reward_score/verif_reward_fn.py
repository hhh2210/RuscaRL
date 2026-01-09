import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from verl.utils.reward_score.local_server import local_serve
except Exception:
    from .local_server import local_serve


_DEFAULT_MAX_THREADS = 128


def _get_env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _count_urls_from_env() -> int:
    url_env = os.getenv("VLLM_BASE_URL", "").strip()
    if not url_env:
        return 1
    urls = [u.strip() for u in url_env.split(",") if u.strip()]
    return max(1, len(urls))


def _resolve_max_threads(max_threads: int, max_workers_per_url: int) -> int:
    # Highest priority: explicit override from env.
    max_workers_override = _get_env_int("GRADER_MAX_WORKERS", 0)
    if max_workers_override > 0:
        return max_workers_override

    # Next: explicit VerIF override.
    max_threads_env = _get_env_int("VERIF_MAX_THREADS", 0)
    if max_threads_env > 0 and max_workers_per_url <= 0:
        return max_threads_env

    # Derive from workers-per-URL if provided.
    if max_workers_per_url > 0:
        url_count = _count_urls_from_env()
        return max(1, int(max_workers_per_url) * url_count)

    return max(1, int(max_threads))


def _parse_solution(text: Any) -> str:
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    if text is None:
        return ""
    return str(text).split("</think>")[-1].strip()


def _extract_instruction(extra_info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(extra_info, dict):
        return None

    prompt = (
        extra_info.get("prompt")
        or extra_info.get("instruction")
        or extra_info.get("prompt_str")
        or extra_info.get("prompt_text")
    )
    if prompt is None:
        return None

    if isinstance(prompt, list):
        parts: List[str] = []
        for item in prompt:
            if isinstance(item, dict):
                content = item.get("content")
                if content:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip() if parts else None

    if isinstance(prompt, dict):
        content = prompt.get("content")
        return str(content).strip() if content is not None else None

    return str(prompt).strip()


def _normalize_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    if isinstance(label, str):
        return label
    try:
        return json.dumps(label, ensure_ascii=False)
    except Exception:
        return None


def _parse_if_verifier_rm_reward(response: Dict[str, Any]) -> float:
    try:
        return float(response["result"][0])
    except Exception:
        return -1.0


def _local_request(data_list: List[Dict[str, Any]], max_threads: int) -> List[float]:
    results: List[Optional[float]] = [None] * len(data_list)
    if not data_list:
        return []

    with ThreadPoolExecutor(max_threads) as executor:
        future_to_index = {
            executor.submit(local_serve, data): idx for idx, data in enumerate(data_list)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = _parse_if_verifier_rm_reward(future.result())
            except Exception:
                results[idx] = -1.0

    # Fill any missing results
    return [r if r is not None else -1.0 for r in results]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: Dict[str, Any] = None,
    max_threads: int = _DEFAULT_MAX_THREADS,
    max_workers_per_url: int = 0,
    skip_rules: bool = False,
    **kwargs,
) -> float:
    scores = compute_score_batched(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info],
        max_threads=max_threads,
        max_workers_per_url=max_workers_per_url,
        skip_rules=skip_rules,
        **kwargs,
    )
    return float(scores[0]) if scores else 0.0


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
    max_threads: int = _DEFAULT_MAX_THREADS,
    max_workers_per_url: int = 0,
    skip_rules: bool = False,
    **kwargs,
) -> List[float]:
    max_threads = _resolve_max_threads(max_threads=max_threads, max_workers_per_url=max_workers_per_url)
    if not skip_rules:
        skip_rules = bool(_get_env_int("VERIF_SKIP_RULES", 0))

    # Build requests
    data_list: List[Dict[str, Any]] = []
    index_map: List[int] = []
    scores: List[float] = [0.0] * len(solution_strs)

    for i, (solution, label, extra_info) in enumerate(
        zip(solution_strs, ground_truths, extra_infos)
    ):
        instruction = _extract_instruction(extra_info)
        label_str = _normalize_label(label)
        if not instruction or not label_str:
            scores[i] = 0.0
            continue

        data_list.append(
            {
                "instruction": instruction,
                "answers": [_parse_solution(solution)],
                "labels": label_str,
                "skip_rules": skip_rules,
            }
        )
        index_map.append(i)

    if data_list:
        batch_scores = _local_request(data_list, max_threads=max_threads)
        for j, score in enumerate(batch_scores):
            scores[index_map[j]] = score

    return scores
