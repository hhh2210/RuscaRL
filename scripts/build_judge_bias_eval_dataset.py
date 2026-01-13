#!/usr/bin/env python3
"""
Build a small, human-curated evaluation dataset for diagnosing LLM-as-a-judge bias.

Goal:
  - Start tiny (default: 10 prompts total) so you can hand-author controlled responses.
  - Use existing sources first: health_bench + verinstruct_verif.

Output format: JSONL, one item per line:
  {
    "id": "...",
    "source": "healthbench" | "verinstruct_verif",
    "split": "val",
    "ability": "...",
    "prompt": "<flattened conversation/instruction>",
    "spec": { ... dataset-specific info needed for judge ... },
    "responses": {
      "ycorrect": "",
      "yshort": "",
      "yverbose": "",
      "ystructured": "",
      "ysycophant": "",
      "yconfident_wrong": ""
    },
    "notes": ""
  }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq


@dataclass(frozen=True)
class Sample:
    source: str
    split: str
    ability: str
    prompt_text: str
    spec: Dict[str, Any]
    meta: Dict[str, Any]


def _flatten_prompt(prompt: Any) -> str:
    """
    Input parquet uses list-of-messages: [{"role": "...", "content": "..."}] (sometimes role is absent).
    We flatten to a readable transcript to keep context for judge bias evaluation.
    """
    if isinstance(prompt, str):
        return prompt.strip()
    if not isinstance(prompt, list):
        return str(prompt).strip()

    lines: List[str] = []
    for msg in prompt:
        if isinstance(msg, dict):
            role = str(msg.get("role", "user")).strip() or "user"
            content = str(msg.get("content", "")).strip()
            if content:
                lines.append(f"{role}:\n{content}")
        elif msg is not None:
            lines.append(str(msg).strip())
    return "\n\n".join(lines).strip()


def _read_parquet_rows(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    pf = pq.ParquetFile(str(path))
    table = pf.read()
    # to_pylist preserves nested dict/list structures better than pandas for our needs here.
    rows = table.to_pylist()
    colnames = [f.name for f in pf.schema_arrow]
    return rows, colnames


def _stable_sample_indices(n_total: int, k: int, seed: int) -> List[int]:
    # Simple LCG-style deterministic shuffle w/o extra deps.
    # Enough for reproducible sampling for a small k.
    if k >= n_total:
        return list(range(n_total))
    order = list(range(n_total))
    x = (seed ^ 0x9E3779B9) & 0xFFFFFFFF
    for i in range(n_total - 1, 0, -1):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        j = x % (i + 1)
        order[i], order[j] = order[j], order[i]
    return order[:k]


def _build_healthbench_samples(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Sample]:
    idxs = _stable_sample_indices(len(rows), k=k, seed=seed + 101)
    out: List[Sample] = []
    for idx in idxs:
        r = rows[idx]
        prompt_messages = r.get("prompt")
        prompt_text = _flatten_prompt(prompt_messages)
        reward_model = r.get("reward_model") or {}
        rubrics = reward_model.get("rubrics") or []
        out.append(
            Sample(
                source="healthbench",
                split="val",
                ability=str(r.get("ability", "")),
                prompt_text=prompt_text,
                spec={"rubrics": rubrics},
                meta={"row_index": idx, "prompt_messages": prompt_messages},
            )
        )
    return out


def _build_verinstruct_verif_samples(rows: List[Dict[str, Any]], k: int, seed: int) -> List[Sample]:
    idxs = _stable_sample_indices(len(rows), k=k, seed=seed + 202)
    out: List[Sample] = []
    for idx in idxs:
        r = rows[idx]
        prompt_messages = r.get("prompt")
        prompt_text = _flatten_prompt(prompt_messages)
        reward_model = r.get("reward_model") or {}
        gt = reward_model.get("ground_truth") or ""
        # In this dataset, ground_truth is a JSON string like {"checkers":[...], "functions":[...]}
        spec: Dict[str, Any] = {}
        try:
            spec = json.loads(gt) if isinstance(gt, str) and gt.strip().startswith("{") else {"ground_truth": gt}
        except Exception:
            spec = {"ground_truth": gt}
        out.append(
            Sample(
                source="verinstruct_verif",
                split="val",
                ability=str(r.get("ability", "")),
                prompt_text=prompt_text,
                spec=spec,
                meta={"row_index": idx, "prompt_messages": prompt_messages, **(r.get("extra_info") or {})},
            )
        )
    return out


def build_dataset(
    repo_root: Path,
    out_path: Path,
    n_total: int,
    n_healthbench: int,
    seed: int,
) -> List[Dict[str, Any]]:
    hb_path = repo_root / "data" / "health_bench" / "healthbench_val.parquet"
    vv_path = repo_root / "data" / "verinstruct_verif" / "verinstruct_verif_val.parquet"
    if not hb_path.exists():
        raise FileNotFoundError(f"Missing {hb_path}")
    if not vv_path.exists():
        raise FileNotFoundError(f"Missing {vv_path}")

    hb_rows, _ = _read_parquet_rows(hb_path)
    vv_rows, _ = _read_parquet_rows(vv_path)

    n_healthbench = max(0, min(n_total, n_healthbench))
    n_verif = max(0, n_total - n_healthbench)

    samples: List[Sample] = []
    samples.extend(_build_healthbench_samples(hb_rows, k=n_healthbench, seed=seed))
    samples.extend(_build_verinstruct_verif_samples(vv_rows, k=n_verif, seed=seed))

    items: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        item_id = f"{s.source}:{s.split}:{s.meta.get('row_index', i)}"
        items.append(
            {
                "id": item_id,
                "source": s.source,
                "split": s.split,
                "ability": s.ability,
                "prompt": s.prompt_text,
                "spec": s.spec,
                "meta": s.meta,
                "response_types": [
                    "yshort",
                    "yverbose",
                    "ystructured",
                    "ysycophant",
                    "ycorrect",
                    "yconfident_wrong",
                    "ypolicy_raw",
                ],
                "responses": {
                    "ycorrect": "",
                    "yshort": "",
                    "yverbose": "",
                    "ystructured": "",
                    "ysycophant": "",
                    "yconfident_wrong": "",
                    "ypolicy_raw": "",
                },
                "notes": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=str, default=".", help="Path to RuscaRL repo root.")
    ap.add_argument(
        "--out",
        type=str,
        default="data/eval/judge_bias_eval_10.jsonl",
        help="Output JSONL path (relative to repo-root).",
    )
    ap.add_argument("--n-total", type=int, default=10, help="Total prompts in the eval set.")
    ap.add_argument(
        "--n-healthbench",
        type=int,
        default=5,
        help="How many prompts to sample from health_bench (rest from verinstruct_verif).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed for reproducibility.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = (repo_root / args.out).resolve()
    items = build_dataset(
        repo_root=repo_root,
        out_path=out_path,
        n_total=int(args.n_total),
        n_healthbench=int(args.n_healthbench),
        seed=int(args.seed),
    )
    print(f"Wrote {len(items)} items -> {out_path}")


if __name__ == "__main__":
    main()

