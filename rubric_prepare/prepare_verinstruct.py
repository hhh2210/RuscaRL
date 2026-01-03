import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datasets


@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "points": self.points, "tags": self.tags}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _normalize_checker_text(text: str) -> str:
    text = str(text).strip()
    # Remove VerIF-style prefix tags like "[llm]" / "[rule]".
    text = re.sub(r"^\s*\[(llm|rule)\]\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _is_soft_checker(text: str) -> bool:
    return "[rule]" not in str(text).lower()


def _looks_like_natural_language_constraint(text: str) -> bool:
    """
    Heuristic: treat as a usable constraint if it looks like a sentence rather than a short label.
    """
    text = str(text).strip()
    if len(text) < 20:
        return False
    if any(ch.isspace() for ch in text):
        return True
    if ":" in text or ";" in text:
        return True
    return False


def _split_numbered_constraints(text: str) -> List[str]:
    """
    Split strings like:
      "1. A ... 2. B ... 3. C ..."
    into ["A ...", "B ...", "C ..."].

    If the pattern doesn't look like a numbered list, return [text].
    """
    text = str(text).strip()
    if not text:
        return []

    # Match "1." / "2." / "3)" etc. We only split if the first item starts at the beginning.
    matches = list(re.finditer(r"(\d{1,3})\s*[\.\)]\s*", text))
    if len(matches) < 2 or matches[0].start() > 2:
        return [text]

    # If it doesn't start with "1.", it's likely not the intended format.
    try:
        first_num = int(matches[0].group(1))
    except Exception:
        return [text]
    if first_num != 1:
        return [text]

    parts: List[str] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = text[start:end].strip()
        if part:
            parts.append(part)
    return parts or [text]


def _get_called_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _extract_llm_constraints_from_function_code(function_code: str) -> List[str]:
    """
    Extract constraint strings from VerIF function code without executing it.

    We look for calls like:
      llm_score(instruction, response, "constraint ...")
      llm_judge(instruction, response, "constraint ...")
    """
    function_code = str(function_code)
    try:
        tree = ast.parse(function_code)
    except SyntaxError:
        return []

    constraints: List[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        fn_name = _get_called_name(node.func)
        if fn_name not in {"llm_score", "llm_judge"}:
            continue

        # Both llm_score and llm_judge expect the constraint to be the 3rd positional argument.
        if len(node.args) < 3:
            continue

        constraint_arg = node.args[2]
        if isinstance(constraint_arg, ast.Constant) and isinstance(constraint_arg.value, str):
            constraints.append(constraint_arg.value)

    return constraints


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_soft_constraints(example: Dict[str, Any], max_constraints: int) -> List[str]:
    """
    Prefer extracting natural-language constraints from `checkers`.
    If checkers are not informative, fall back to parsing `functions` for llm_score/llm_judge strings.
    """
    checkers = example.get("checkers") or []
    functions = example.get("functions") or []

    constraints: List[str] = []

    # 1) Try checkers first (most direct, avoids parsing code strings).
    for checker in checkers:
        if not isinstance(checker, str):
            continue
        if not _is_soft_checker(checker):
            continue
        norm = _normalize_checker_text(checker)
        if _looks_like_natural_language_constraint(norm):
            constraints.extend(_split_numbered_constraints(norm))

    constraints = _dedupe_keep_order(constraints)

    # 2) Fallback: parse code strings to get the natural-language constraints passed to llm_score/llm_judge.
    if not constraints:
        parsed: List[str] = []
        for fn_code in functions:
            if not isinstance(fn_code, str):
                continue
            parsed.extend(_extract_llm_constraints_from_function_code(fn_code))

        # Some VerIF function strings contain multiple numbered constraints in one big string.
        expanded: List[str] = []
        for c in parsed:
            expanded.extend(_split_numbered_constraints(c))
        constraints = _dedupe_keep_order(expanded)

    if max_constraints > 0:
        constraints = constraints[:max_constraints]
    return constraints


def _to_prompt_messages(prompt_text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": prompt_text}]


def _make_map_fn(data_source: str, max_constraints: int, points: float):
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        prompt_text = str(example.get("prompt", "")).strip()
        prompt = _to_prompt_messages(prompt_text)

        soft_constraints = _extract_soft_constraints(example, max_constraints=max_constraints)
        rubrics = [RubricItem(criterion=c, points=points, tags={}) for c in soft_constraints]

        reward_model = {
            "style": "rubric",
            "rubrics": [r.to_dict() for r in rubrics],
            "ground_truth": "",
        }

        data = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "instruction_following",
            "reward_model": reward_model,
            "extra_info": {
                # The reward fn in RuscaRL reads prompt + reward_model from extra_info.
                "prompt": prompt,
                "reward_model": reward_model,
                # Keep a stable reference for debugging.
                "original_id": example.get("id", idx),
            },
        }
        return data

    return process_fn


def _load_hf_dataset(repo: str) -> Tuple[datasets.Dataset, Optional[datasets.Dataset]]:
    ds_dict = datasets.load_dataset(repo)
    if isinstance(ds_dict, datasets.Dataset):
        return ds_dict, None

    # Prefer explicit validation, else test, else None.
    train = ds_dict.get("train")
    val = ds_dict.get("validation") or ds_dict.get("eval") or ds_dict.get("test")
    if train is None:
        # Fallback: first split as train
        first_split = list(ds_dict.keys())[0]
        train = ds_dict[first_split]
        # If there are more splits, use the 2nd one as val.
        if val is None and len(ds_dict.keys()) > 1:
            second_split = list(ds_dict.keys())[1]
            val = ds_dict[second_split]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo", default="THU-KEG/VerInstruct")
    parser.add_argument("--local_jsonl", default=None, help="Optional local jsonl path with fields: id/prompt/checkers/functions")
    parser.add_argument("--output_dir", default="data/verinstruct")
    parser.add_argument("--data_source", default="verinstruct")
    parser.add_argument("--val_ratio", type=float, default=0.01, help="Used only when no validation/test split is available")
    parser.add_argument("--max_constraints", type=int, default=32, help="Cap soft constraints per example (0 means no cap)")
    parser.add_argument("--points", type=float, default=1.0, help="Points per soft constraint (use 1.0 to make reward = mean of 0/1)")
    args = parser.parse_args()

    if args.local_jsonl:
        raw = _load_jsonl(args.local_jsonl)
        ds = datasets.Dataset.from_list(raw)
        # Create a small validation split from train for convenience.
        split = ds.train_test_split(test_size=args.val_ratio, seed=42, shuffle=True)
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        train_ds, val_ds = _load_hf_dataset(args.hf_repo)
        if val_ds is None:
            split = train_ds.train_test_split(test_size=args.val_ratio, seed=42, shuffle=True)
            train_ds = split["train"]
            val_ds = split["test"]

    train_processed = train_ds.map(
        function=_make_map_fn(args.data_source, max_constraints=args.max_constraints, points=args.points),
        with_indices=True,
        remove_columns=train_ds.column_names,
    ).shuffle(seed=42)

    val_processed = val_ds.map(
        function=_make_map_fn(args.data_source, max_constraints=args.max_constraints, points=args.points),
        with_indices=True,
        remove_columns=val_ds.column_names,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "verinstruct_train.parquet")
    val_path = os.path.join(args.output_dir, "verinstruct_val.parquet")
    train_processed.to_parquet(train_path)
    val_processed.to_parquet(val_path)

    print("\nDataset information:")
    print(f"Train: {len(train_processed)} -> {train_path}")
    print(f"Val:   {len(val_processed)} -> {val_path}")
    print("\nTrain sample:")
    print(json.dumps(train_processed[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

