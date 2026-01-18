#!/usr/bin/env python3
"""
HealthBench Official Evaluation Script

Uses OpenAI's official simple-evals grading logic to evaluate HealthBench responses.
This script grades pre-generated responses from a parquet file.

Usage:
    # Set environment variables
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_BASE_URL="https://api.uniapi.io/v1"  # or your proxy

    # Run evaluation
    python scripts/healthbench_official_eval.py \
        --data-path /data/haozy/eval_outputs/healthbench_RuscaRL_step350_gen.parquet \
        --output-path /data/haozy/eval_outputs/healthbench_RuscaRL_step350_official_scores.json \
        --grader-model gpt-4.1-2025-04-14 \
        --n-threads 32
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add simple-evals to path if available
SIMPLE_EVALS_PATH = "/home/haozy/simple-evals"
if os.path.exists(SIMPLE_EVALS_PATH):
    sys.path.insert(0, SIMPLE_EVALS_PATH)


from openai import OpenAI

# Official HealthBench grader template (from simple-evals)
GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: List[str]

    def __str__(self):
        return f"[{self.points}] {self.criterion}"

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags = d.get("tags", [])
        if isinstance(tags, dict):
            tags = list(tags.keys())
        elif not isinstance(tags, list):
            tags = []
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=tags,
        )


def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON from grader response, handling markdown formatting."""
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


def calculate_score(rubric_items: List[RubricItem], grading_response_list: List[dict]) -> Optional[float]:
    """Calculate score using official HealthBench formula."""
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(rubric_items, grading_response_list)
        if grading_response.get("criteria_met", False)
    )
    overall_score = achieved_points / total_possible_points
    return overall_score


class OfficialGrader:
    """Official HealthBench grader using OpenAI API."""
    
    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        
        print(f"Initialized grader with model: {model}")
        if base_url:
            print(f"Using base URL: {base_url}")
    
    def grade_rubric_item(
        self,
        prompt: List[Dict[str, str]],
        response_text: str,
        rubric_item: RubricItem,
    ) -> dict:
        """Grade a single rubric item (official logic)."""
        # Build conversation string
        convo_with_response = prompt + [{"content": response_text, "role": "assistant"}]
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response if m.get("role") != "system"]
        )
        
        # Build grader prompt
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": grader_prompt}
        ]
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                grading_response = response.choices[0].message.content
                grading_response_dict = parse_json_to_dict(grading_response)
                
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        return grading_response_dict
                
                print(f"Grading failed (attempt {attempt + 1}), bad JSON output, retrying...")
            except Exception as e:
                print(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
        
        # Return default on failure
        return {"criteria_met": False, "explanation": "Grading failed after retries"}
    
    def grade_sample(
        self,
        prompt: List[Dict[str, str]],
        response_text: str,
        rubric_items: List[RubricItem],
        n_threads: int = 8,
    ) -> Tuple[float, List[dict]]:
        """Grade all rubric items for a sample (parallel)."""
        grading_response_list = []
        
        # Grade each rubric item (can parallelize within sample)
        with ThreadPoolExecutor(max_workers=min(n_threads, len(rubric_items))) as executor:
            futures = {
                executor.submit(self.grade_rubric_item, prompt, response_text, item): i
                for i, item in enumerate(rubric_items)
            }
            results = [None] * len(rubric_items)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
            grading_response_list = results
        
        # Calculate score
        score = calculate_score(rubric_items, grading_response_list)
        return score if score is not None else 0.0, grading_response_list


def normalize_data(data):
    """Convert numpy arrays to lists if needed."""
    if hasattr(data, 'tolist'):
        return data.tolist()
    return data


def main():
    parser = argparse.ArgumentParser(description="Official HealthBench Evaluation")
    parser.add_argument("--data-path", required=True, help="Path to parquet with responses")
    parser.add_argument("--output-path", required=True, help="Path to save evaluation results")
    parser.add_argument("--grader-model", default="gpt-4.1-2025-04-14", help="Grader model name")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or use OPENAI_API_KEY env)")
    parser.add_argument("--base-url", default=None, help="OpenAI base URL (or use OPENAI_BASE_URL env)")
    parser.add_argument("--n-threads", type=int, default=32, help="Number of parallel threads")
    parser.add_argument("--n-samples", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--response-key", default="responses", help="Column name for responses")
    parser.add_argument("--response-idx", type=int, default=0, help="Index of response to use (if multiple)")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    if args.n_samples:
        df = df.head(args.n_samples)
        print(f"Limited to {len(df)} samples")
    
    # Initialize grader
    grader = OfficialGrader(
        model=args.grader_model,
        api_key=args.api_key or os.getenv("OPENAI_API_KEY"),
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL"),
    )
    
    # Evaluate each sample
    results = []
    all_scores = []
    
    for idx in tqdm(range(len(df)), desc="Evaluating"):
        row = df.iloc[idx]
        
        # Extract prompt
        prompt = normalize_data(row.get("prompt", []))
        if not prompt:
            print(f"Warning: Sample {idx} has no prompt, skipping")
            continue
        
        # Extract response
        responses = normalize_data(row.get(args.response_key, []))
        if isinstance(responses, list) and len(responses) > args.response_idx:
            response_text = responses[args.response_idx]
        elif isinstance(responses, str):
            response_text = responses
        else:
            print(f"Warning: Sample {idx} has no valid response, skipping")
            continue
        
        # Extract rubrics
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, np.ndarray):
            reward_model = reward_model.item() if reward_model.size == 1 else {}
        rubrics_raw = normalize_data(reward_model.get("rubrics", []))
        
        if not rubrics_raw:
            print(f"Warning: Sample {idx} has no rubrics, skipping")
            continue
        
        rubric_items = [RubricItem.from_dict(r) for r in rubrics_raw]
        
        # Grade sample
        score, grading_details = grader.grade_sample(
            prompt=prompt,
            response_text=response_text,
            rubric_items=rubric_items,
            n_threads=args.n_threads,
        )
        
        all_scores.append(score)
        results.append({
            "idx": idx,
            "score": score,
            "prompt": prompt,
            "response": response_text,
            "rubric_grades": [
                {
                    "criterion": item.criterion,
                    "points": item.points,
                    "criteria_met": grade.get("criteria_met", False),
                    "explanation": grade.get("explanation", ""),
                }
                for item, grade in zip(rubric_items, grading_details)
            ],
        })
        
        # Print progress
        if (idx + 1) % 10 == 0:
            current_mean = np.mean(all_scores)
            print(f"  Progress: {idx + 1}/{len(df)}, Mean score: {current_mean:.4f}")
    
    # Calculate final metrics
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    
    # Bootstrap standard error
    bootstrap_means = [
        np.mean(np.random.choice(all_scores, len(all_scores)))
        for _ in range(1000)
    ]
    bootstrap_std = np.std(bootstrap_means)
    
    final_metrics = {
        "overall_score": float(np.clip(mean_score, 0, 1)),
        "std": float(std_score),
        "bootstrap_std": float(bootstrap_std),
        "n_samples": len(all_scores),
        "grader_model": args.grader_model,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "metrics": final_metrics,
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("HealthBench Official Evaluation Results")
    print(f"{'='*60}")
    print(f"Overall Score: {final_metrics['overall_score']:.4f} ± {final_metrics['bootstrap_std']:.4f}")
    print(f"Samples: {final_metrics['n_samples']}")
    print(f"Grader Model: {final_metrics['grader_model']}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
