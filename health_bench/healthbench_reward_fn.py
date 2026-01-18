import json
import re
import os
import requests
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from datetime import datetime
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from dotenv import load_dotenv
import importlib
from health_bench.vllm_sampler import VLLMSampler

# Import verification function module
from verl.utils.reward_score.rule_fn import get_verification_function, VERIFICATION_FUNCTIONS

# Concurrent workers control variable for single URL
MAX_CONCURRENT_WORKERS = 512

# Default rubric batch size for a single judge request (override via env)
DEFAULT_RUBRIC_BATCH_SIZE = 10

# Judge debug log directory
JUDGE_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "judge_log")

def _ensure_log_dir():
    """Ensure the judge log directory exists."""
    os.makedirs(JUDGE_LOG_DIR, exist_ok=True)

def _log_judge_debug(message: str):
    """Write debug message to judge log file."""
    _ensure_log_dir()
    log_file = os.path.join(JUDGE_LOG_DIR, f"judge_debug_{datetime.now().strftime('%Y%m%d')}.log")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# Load .env file
load_dotenv()

# Verification functions have been moved to verification_functions.py module

@dataclass
class RubricItem:
    criterion: str
    points: float
    tags: Dict[str, Any]

    def __str__(self) -> str:
        return self.criterion

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        tags_data = d.get("tags", [])
        # If tags is in list format, try to parse as dictionary
        if isinstance(tags_data, list):
            tags_dict = {}
            for tag in tags_data:
                if isinstance(tag, str) and ":" in tag:
                    key, value = tag.split(":", 1)
                    tags_dict[key] = value
                elif isinstance(tag, str):
                    # For tags without colon, use the tag itself as key with value True
                    tags_dict[tag] = True
            tags_data = tags_dict
        elif not isinstance(tags_data, dict):
            # If neither list nor dict, set to empty dict
            tags_data = {}
            
        return cls(
            criterion=d["criterion"],
            points=d["points"],
            tags=tags_data
        )

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "points": self.points,
            "tags": self.tags
        }

@dataclass
class SamplerResponse:
    """Sampler response"""
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]

class SamplerBase:
    """Base sampler class"""
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": content}

class ChatCompletionSampler(SamplerBase):
    """OpenAI API sampler"""
    def __init__(
        self,
        model: str = "gpt-4.1-2025-04-14",
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 2048,
    ):
        if OpenAI is None:
            raise ImportError("openai package not installed; ChatCompletionSampler requires it.")
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message(self.system_message, "system")
            ] + message_list
        
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response, retrying...")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = min(2**trial, 300)  # Exponential backoff, max wait time 300 seconds
                print(
                    f"Rate limit exception, waiting {exception_backoff} seconds before retry {trial}",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1

# VLLMSampler moved to health_bench.vllm_sampler
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

def _get_rubric_batch_size() -> int:
    raw_value = os.getenv("GRADER_RUBRIC_BATCH_SIZE", str(DEFAULT_RUBRIC_BATCH_SIZE)).strip()
    try:
        batch_size = int(raw_value)
    except Exception:
        batch_size = DEFAULT_RUBRIC_BATCH_SIZE
    return max(1, batch_size)

def _chunk_indices(indices: List[int], chunk_size: int):
    for start in range(0, len(indices), chunk_size):
        yield indices[start:start + chunk_size]

def _build_batch_grader_prompt(
    prompt: List[Dict[str, str]],
    response: str,
    rubric_items: List[RubricItem],
) -> str:
    convo_with_response = prompt + [dict(content=response, role="assistant")]
    convo_str = "\n\n".join(
        [f"{m['role']}: {m['content']}" for m in convo_with_response if m.get("role") != "system"]
    )
    rubrics_str = "\n".join(
        f"{idx + 1}. {item.criterion}" for idx, item in enumerate(rubric_items)
    )
    return f"""Your job is to look at a conversation and a list of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows EACH rubric item.

# Conversation
{convo_str}

# Rubric items
{rubrics_str}

# Instructions
Return a json object in markdown format with the following structure:
- Keys are the rubric numbers (1-based).
- Values are booleans (true/false) indicating whether the response meets that rubric item.
- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.
- If the criteria is undesirable (e.g., a negative-point rubric), you should still return true if the response meets that undesirable criterion.

# Example response
```json
{{
  "1": true,
  "2": false,
  "3": true
}}
```

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def _parse_batch_grading_response(resp_text: str, expected_count: int) -> Dict[int, bool]:
    cleaned = re.sub(r"^```json\s*|\s*```$", "", resp_text.strip())
    try:
        data = json.loads(cleaned)
    except Exception:
        return {}

    results: Dict[int, bool] = {}
    for key, val in data.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if idx < 1 or idx > expected_count:
            continue
        if isinstance(val, bool):
            results[idx] = val
            continue
        if isinstance(val, (int, float)) and val in (0, 1):
            results[idx] = bool(val)
            continue
        if isinstance(val, str):
            norm = val.strip().upper()
            if norm in ("TRUE", "PRESENT", "YES", "Y"):
                results[idx] = True
            elif norm in ("FALSE", "NOT_PRESENT", "NO", "N"):
                results[idx] = False
    return results

def _grade_llm_rubric_batch(
    prompt: List[Dict[str, str]],
    response: str,
    rubric_items: List[RubricItem],
    grader_model,
    max_retries: int = 3,
) -> Dict[int, bool]:
    if not rubric_items:
        return {}
    retry_count = 0
    llm_results: Dict[int, bool] = {}
    last_response_text = ""
    while retry_count < max_retries:
        prompt_text = _build_batch_grader_prompt(prompt, response, rubric_items)
        sampler_response = grader_model([dict(content=prompt_text, role="user")])
        last_response_text = sampler_response.response_text
        llm_results = _parse_batch_grading_response(
            last_response_text,
            expected_count=len(rubric_items),
        )
        if len(llm_results) == len(rubric_items):
            return llm_results
        retry_count += 1

    print(
        f"Batch grading failure count reached limit ({max_retries} times), marking all as NOT_PRESENT"
    )
    # Log debug info to file
    _log_judge_debug("=" * 60)
    _log_judge_debug(f"Batch grading FAILED after {max_retries} retries")
    _log_judge_debug(f"Expected {len(rubric_items)} results, got {len(llm_results)}: {llm_results}")
    _log_judge_debug(f"Rubric items: {[str(r) for r in rubric_items]}")
    _log_judge_debug(f"Last judge response:\n{last_response_text}")
    _log_judge_debug("=" * 60)
    return {i + 1: False for i in range(len(rubric_items))}

def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string, handling markdown format"""
    original_string = json_string
    
    # Method 1: Original matching approach - Remove markdown-style ```json``` markers if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())

    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        # JSON parsing failed, but don't print details to avoid log pollution
        pass
    
    # Backup method: more lenient approach - extract anything that looks like JSON content
    # New: try to fix double quote escaping issues in JSON
    try:
        # Find the first complete JSON object
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(original_string):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    json_content = original_string[start_idx:i+1]
                    
                    # Try to fix double quote escaping issues
                    # Find all "explanation": "..." and "criteria_met": ... patterns
                    def fix_quotes_in_json(json_str):
                        # Use regular expressions to fix double quotes in explanation field
                        import re
                        
                        # Match "explanation": "..." part
                        explanation_pattern = r'("explanation"\s*:\s*")(.*?)("(?:\s*,|\s*}))'
                        
                        def fix_explanation(match):
                            prefix = match.group(1)
                            content = match.group(2)
                            suffix = match.group(3)
                            
                            # Escape double quotes in content while preserving already escaped ones
                            # First replace already escaped quotes with temporary placeholder
                            content = content.replace('\\"', '###ESCAPED_QUOTE###')
                            # Escape unescaped double quotes
                            content = content.replace('"', '\\"')
                            # Restore already escaped quotes
                            content = content.replace('###ESCAPED_QUOTE###', '\\"')
                            
                            return prefix + content + suffix
                        
                        fixed_json = re.sub(explanation_pattern, fix_explanation, json_str, flags=re.DOTALL)
                        return fixed_json
                    
                    fixed_json = fix_quotes_in_json(json_content)
                    return json.loads(fixed_json)
    except Exception as e:
        # Don't print detailed information to avoid log pollution
        pass
        
        # Directly extract values of explanation and criteria_met fields
        try:
            # Extract explanation field value (content within quotes)
            explanation_pattern = r'"explanation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            explanation_match = re.search(explanation_pattern, original_string)
            explanation = explanation_match.group(1) if explanation_match else ""
            
            # Extract criteria_met field value (true or false)
            criteria_pattern = r'"criteria_met"\s*:\s*(true|false)'
            criteria_match = re.search(criteria_pattern, original_string)
            criteria_met = criteria_match.group(1) == 'true' if criteria_match else False
            
            # Construct JSON object
            result = {
                "explanation": explanation,
                "criteria_met": criteria_met
            }
            # Don't print detailed information to avoid log pollution
            return result
        except Exception as e:
            print(f"Field extraction exception occurred: {e}")

        

        print("All JSON parsing methods failed, returning empty dictionary")
        print("="*80)
        return {}

    # Fallback: if try block completed without returning (no valid JSON structure found)
    return {}

def calculate_score(rubric_items: List[RubricItem], grading_response_list: List[dict]) -> float:
    """Calculate total score"""
    total_possible_points = sum(
        rubric_item.points for rubric_item in rubric_items if rubric_item.points > 0
    )
    if total_possible_points == 0:
        return 0.0

    achieved_points = sum(
        rubric_item.points
        for rubric_item, grading_response in zip(rubric_items, grading_response_list)
        if grading_response["criteria_met"]
    )
    overall_score = achieved_points / total_possible_points
    return overall_score

def grade_single_example(
    prompt: List[Dict[str, str]], 
    response: str,
    rubric_items: List[RubricItem],
    grader_model,
    executor=None,  # New parameter: external thread pool
) -> Tuple[float, str, List[Dict]]:
    """Evaluate a single example with rule checks + batched LLM grading for non-rule criteria"""
    def run_rule_check(rubric_item: RubricItem) -> dict | None:
        verifier_type = rubric_item.tags.get("verifier") if rubric_item.tags else None
        if verifier_type == "rule":
            function_name = rubric_item.tags.get("function")
            parameter_value = rubric_item.tags.get("parameters")
            if function_name and parameter_value is not None:
                verify_func = get_verification_function(function_name) if get_verification_function else None
                if verify_func:
                    criteria_met = verify_func(response, parameter_value)
                    return {
                        "criteria_met": criteria_met,
                        "explanation": f"Rule-based verification using {function_name} with parameter '{parameter_value}': {'PASS' if criteria_met else 'FAIL'}"
                    }
                raise ValueError(f"Verification function '{function_name}' not found")
        return None

    # Split rubric items into rule-based and LLM-based
    rule_indices = []
    llm_indices = []
    for idx, item in enumerate(rubric_items):
        if (item.tags and item.tags.get("verifier") == "rule" and
            item.tags.get("function") and item.tags.get("parameters") is not None):
            rule_indices.append(idx)
        else:
            llm_indices.append(idx)

    grading_response_list = [None] * len(rubric_items)

    # Run rule checks sequentially (or via executor if provided)
    if rule_indices:
        if executor is not None:
            futures = {}
            for idx in rule_indices:
                futures[executor.submit(run_rule_check, rubric_items[idx])] = idx
            for future in futures:
                res = future.result()
                grading_response_list[futures[future]] = res or {
                    "criteria_met": False,
                    "explanation": "Rule verification failed or not applicable"
                }
        else:
            print("No external thread pool, executing sequentially")
            for idx in rule_indices:
                res = run_rule_check(rubric_items[idx])
                grading_response_list[idx] = res or {
                    "criteria_met": False,
                    "explanation": "Rule verification failed or not applicable"
                }

    # Batched LLM grading for remaining criteria
    if llm_indices:
        batch_size = _get_rubric_batch_size()
        for chunk in _chunk_indices(llm_indices, batch_size):
            llm_items = [rubric_items[i] for i in chunk]
            llm_results = _grade_llm_rubric_batch(prompt, response, llm_items, grader_model)
            for local_idx, rubric_global_idx in enumerate(chunk):
                present = llm_results.get(local_idx + 1, False)
                grading_response_list[rubric_global_idx] = {
                    "criteria_met": present,
                    "explanation": f"Batch LLM evaluation: {'PRESENT' if present else 'NOT_PRESENT'}"
                }

    # Calculate total score
    overall_score = calculate_score(rubric_items, grading_response_list)

    # Generate detailed explanation
    rubric_items_with_grades = []
    readable_explanation_list = []
    for rubric_item, grading_response in zip(rubric_items, grading_response_list):
        explanation = grading_response.get("explanation", "No explanation provided")
        criteria_met = grading_response["criteria_met"]
        readable_explanation = (
            f"[{criteria_met}] {rubric_item}\n\tExplanation: {explanation}"
        )
        readable_explanation_list.append(readable_explanation)
        rubric_items_with_grades.append(
            {
                **rubric_item.to_dict(),
                "criteria_met": criteria_met,
                "explanation": explanation,
            }
        )

    # Display in original rubric order
    readable_explanation_str = "\n\n".join(readable_explanation_list)
    readable_explanation_str = f"\n\n{readable_explanation_str}"

    return overall_score, readable_explanation_str, rubric_items_with_grades

def compute_score(data_source: str, solution_str: str, ground_truth: str = None, extra_info: Dict[str, Any] = None) -> float:
    """
    Calculate healthbench reward score
    
    Args:
        data_source: Dataset name (obtained from DataProto.non_tensor_batch['data_source'])
        solution_str: Model-generated response
        ground_truth: Not used
        extra_info: Contains prompt and reward_model information
        
    Returns:
        float: Reward score [0, 1]
    """
    
    try:
        # Check if extra_info is None
        if extra_info is None:
            return 0.0
        
        # Extract data from extra_info
        prompt = extra_info.get("prompt", [])
        reward_model = extra_info.get("reward_model", {})
        rubrics = reward_model.get("rubrics", [])
        
        # Convert numpy arrays to lists if needed (parquet stores as numpy arrays)
        if hasattr(prompt, 'tolist'):
            prompt = prompt.tolist()
        if hasattr(rubrics, 'tolist'):
            rubrics = rubrics.tolist()
        
        if not prompt or not rubrics:
            return 0.0
            
        # Rebuild rubrics
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        # Use VLLM as scoring model
        grader = get_global_grader()  # Use global grader instance
        
        score, _, _ = grade_single_example(prompt, solution_str, rubric_items, grader)
        return score  # Already normalized score [0,1]
        
    except Exception as e:
        print(f"Error calculating reward score: {e}")
        return 0.0

def compute_score_batched(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[Dict[str, Any]], max_workers_per_url: int = MAX_CONCURRENT_WORKERS, **kwargs) -> List[Dict[str, Any]]:
    """
    Batch calculate reward scores for multiple responses
    
    Args:
        data_sources: List of dataset names
        solution_strs: List of model-generated responses
        ground_truths: List of ground truth answers (will be included in the result)
        extra_infos: List containing prompt and reward_model information
        max_workers_per_url: Concurrency per URL, defaults to MAX_CONCURRENT_WORKERS
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing score, acc, and ground_truth fields
    """
    batch_data = list(zip(data_sources, solution_strs, ground_truths, extra_infos))
    scores = batch_compute_scores(batch_data, max_workers_per_url=max_workers_per_url)
    
    # Convert scores to dictionary format containing score and acc fields
    results = []
    for i, score in enumerate(scores):
        results.append({
            "score": score,
            "acc": score > 0.5,  # Convert score to accuracy (boolean value)
            "ground_truth": ground_truths[i]  # Add ground_truth to the result
        })
    
    return results

# Global grader instance to avoid repeated creation
_global_grader = None

def get_global_grader():
    """Get or create global grader instance"""
    global _global_grader
    if _global_grader is None:
        _global_grader = VLLMSampler(
            max_tokens=2048,
            enable_thinking=False,
            filter_think_tags=True
        )
    return _global_grader

def batch_compute_scores(batch_data: List[Tuple[str, str, str, Dict[str, Any]]], max_workers_per_url: int = MAX_CONCURRENT_WORKERS) -> List[float]:
    """
    Batch calculate reward scores for multiple responses
    
    New optimized architecture:
    - Separate rule-based and LLM-based criteria processing
    - Rule criteria: batch sequential processing (thread-safe)
    - LLM criteria: multi-threaded concurrent processing (network I/O optimized)
    - Final score aggregation from both types
    
    Args:
        batch_data: List, each item contains (data_source, solution_str, ground_truth, extra_info)
        max_workers_per_url: Concurrency per URL, defaults to MAX_CONCURRENT_WORKERS. Total requests = max_workers_per_url × number of URLs
        
    Returns:
        List[float]: List of reward scores
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    # Record start time
    start_time = time.time()
    
    # Statistics
    total_criteria = 0
    rule_based_criteria = 0
    llm_criteria = 0
    
    # Get global grader
    grader = get_global_grader()
    
    # Reload environment variables from .env and update URL configuration at the start of each batch
    print(f"Batch started: Re-reading .env file and updating URL configuration...")
    url_changed = grader._reload_urls_from_env()
    if url_changed:
        print("URL configuration updated")
    else:
        print("URL configuration unchanged")
    
    # Re-get URL count (after re-reading URLs)
    url_count = len(grader.base_urls)
    
    # Re-update load information
    grader._update_loads()
    print(f"Load update completed, current available URL count: {len([url for url in grader.base_urls if grader.url_loads[url].get('available', False)])}")
    
    # Print average load information
    print("\nCurrent load information:")
    available_urls = [url for url in grader.base_urls if grader.url_loads[url].get('available', False)]
    if available_urls:
        total_running = sum(grader.url_loads[url].get('running', 0) for url in available_urls)
        total_waiting = sum(grader.url_loads[url].get('waiting', 0) for url in available_urls)
        total_load = sum(grader.url_loads[url].get('total', 0) for url in available_urls)
        print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total load: {total_load/len(available_urls):.1f}")
    print(f"Available server count: {len(available_urls)}/{len(grader.base_urls)}")
    print()
    
    # Reset virtual load counters
    for url in grader.base_urls:
        grader.virtual_loads[url] = 0
    
    # Total concurrency: max_workers_per_url × available URL count (or override via GRADER_MAX_WORKERS).
    try:
        max_workers_override = int(os.getenv("GRADER_MAX_WORKERS", "0").strip() or "0")
    except Exception:
        max_workers_override = 0

    effective_url_count = len(available_urls) if available_urls else max(1, url_count)
    per_url_workers = max(1, int(max_workers_per_url))
    actual_max_workers = max_workers_override if max_workers_override > 0 else (per_url_workers * effective_url_count)
    print(
        f"Configured {url_count} URLs (available={effective_url_count}), max_workers_per_url={per_url_workers}, total_concurrency={actual_max_workers}"
    )

    rubric_batch_size = _get_rubric_batch_size()
    
    # Separate rule and LLM tasks
    rule_tasks = []
    llm_tasks = []
    
    for sample_idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(batch_data):
        if extra_info is None:
            continue
            
        prompt = extra_info.get("prompt", [])
        reward_model = extra_info.get("reward_model", {})
        rubrics = reward_model.get("rubrics", [])
        
        if not prompt or not rubrics:
            continue
            
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        # Classify tasks by type
        llm_indices = []
        for rubric_idx, rubric_item in enumerate(rubric_items):
            # Check if this is a rule-based task
            if (rubric_item.tags and 
                rubric_item.tags.get("verifier") == "rule" and 
                rubric_item.tags.get("function")):
                rule_tasks.append({
                    'sample_idx': sample_idx,
                    'rubric_idx': rubric_idx,
                    'prompt': prompt,
                    'response': solution_str,
                    'rubric_item': rubric_item,
                    'rubric_items': rubric_items
                })
            else:
                llm_indices.append(rubric_idx)

        for chunk in _chunk_indices(llm_indices, rubric_batch_size):
            llm_tasks.append({
                'sample_idx': sample_idx,
                'prompt': prompt,
                'response': solution_str,
                'rubric_items': [rubric_items[i] for i in chunk],
                'rubric_indices': chunk
            })
    
    # Dictionary for tracking function call counts
    function_call_stats = {}
    
    def process_rule_task(task):
        """Process single rule-based task"""
        current_function_name = None
        try:
            rubric_item = task['rubric_item']
            response = task['response']
            
            function_name = rubric_item.tags.get("function")
            parameter_value = rubric_item.tags.get("parameters")
            current_function_name = function_name
            
            if function_name:
                # If parameter is null, use empty dictionary
                if parameter_value is None:
                    parameter_value = {}
                
                # Track function call counts
                if function_name not in function_call_stats:
                    function_call_stats[function_name] = 0
                function_call_stats[function_name] += 1
                
                # Get function from verification function registry
                verify_func = get_verification_function(function_name) if get_verification_function else None
                if verify_func:
                    criteria_met = verify_func(response, parameter_value)
                    return {
                        'sample_idx': task['sample_idx'],
                        'rubric_idx': task['rubric_idx'],
                        'result': {
                            "criteria_met": criteria_met,
                            "explanation": f"Rule-based verification using {function_name} with parameters {parameter_value}: {'PASS' if criteria_met else 'FAIL'}"
                        },
                        'verification_type': 'rule'
                    }
                else:
                    raise ValueError(f"Verification function '{function_name}' not found")
            else:
                raise ValueError("Rule task missing function name")
                
        except Exception as e:
            error_msg = f"Rule task error: {e}"
            if current_function_name:
                error_msg += f" (using rule function: {current_function_name})"
                print(f"Error details: {error_msg}")
                print(f"Input parameter - function_name: {current_function_name}")
                print(f"Input parameter - parameter_value: {parameter_value if 'parameter_value' in locals() else 'N/A'}")
            else:
                print(error_msg)
            return {
                'sample_idx': task['sample_idx'],
                'rubric_idx': task['rubric_idx'],
                'result': {
                    "criteria_met": False,
                    "explanation": f"Rule processing error: {str(e)}" + (f" (rule function: {current_function_name})" if current_function_name else "")
                },
                'verification_type': 'rule_failed'
            }
    
    def process_llm_task(task):
        """Process batched LLM-based criteria"""
        try:
            rubric_items = task['rubric_items']
            rubric_indices = task['rubric_indices']
            llm_results = _grade_llm_rubric_batch(
                task['prompt'],
                task['response'],
                rubric_items,
                grader,
            )
            results = []
            for local_idx, global_idx in enumerate(rubric_indices):
                present = llm_results.get(local_idx + 1, False)
                results.append({
                    'sample_idx': task['sample_idx'],
                    'rubric_idx': global_idx,
                    'result': {
                        "criteria_met": present,
                        "explanation": f"Batch LLM evaluation: {'PRESENT' if present else 'NOT_PRESENT'}"
                    },
                    'verification_type': 'llm'
                })
            return results
        except Exception as e:
            print(f"LLM task error: {e}")
            return [{
                'sample_idx': task['sample_idx'],
                'rubric_idx': idx,
                'result': {
                    "criteria_met": False,
                    "explanation": f"LLM processing error: {str(e)}"
                },
                'verification_type': 'llm_failed'
            } for idx in task.get('rubric_indices', [])]
    
    # Count total criteria
    total_criteria = len(rule_tasks) + sum(len(task.get('rubric_indices', [])) for task in llm_tasks)
    
    # Store all results
    sample_results = {}  # sample_idx -> {rubric_idx: result}
    failed_rule_criteria = 0
    
    print(f"\nProcessing {len(rule_tasks)} rule tasks and {len(llm_tasks)} LLM batch tasks (batch_size={rubric_batch_size})...")
    
    # Phase 1: Process rule tasks sequentially (batch processing)
    print("Phase 1: Processing rule-based criteria sequentially...")
    rule_start_time = time.time()
    
    for task in rule_tasks:
        result = process_rule_task(task)
        sample_idx = result['sample_idx']
        if sample_idx not in sample_results:
            sample_results[sample_idx] = {}
        sample_results[sample_idx][result['rubric_idx']] = result['result']
        
        # Count verification types
        verification_type = result.get('verification_type', 'unknown')
        if verification_type == 'rule':
            rule_based_criteria += 1
        else:
            failed_rule_criteria += 1
    
    rule_end_time = time.time()
    print(f"Rule processing completed in {rule_end_time - rule_start_time:.2f} seconds")
    
    # Phase 2: Process LLM tasks concurrently
    print("Phase 2: Processing LLM-based criteria concurrently...")
    llm_start_time = time.time()
    
    if llm_tasks:
        with ThreadPoolExecutor(max_workers=actual_max_workers) as executor:
            # Submit all LLM tasks
            future_to_task = {executor.submit(process_llm_task, task): task for task in llm_tasks}
            
            # Collect results
            for future in as_completed(future_to_task):
                result_list = future.result()
                for result in result_list:
                    sample_idx = result['sample_idx']
                    if sample_idx not in sample_results:
                        sample_results[sample_idx] = {}
                    sample_results[sample_idx][result['rubric_idx']] = result['result']
                    
                    # Count verification types
                    verification_type = result.get('verification_type', 'unknown')
                    if verification_type == 'llm':
                        llm_criteria += 1
    
    llm_end_time = time.time()
    print(f"LLM processing completed in {llm_end_time - llm_start_time:.2f} seconds")
    
    # Calculate final scores for each sample
    final_scores = []
    for sample_idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(batch_data):
        try:
            if extra_info is None:
                final_scores.append(0.0)
                continue
                
            prompt = extra_info.get("prompt", [])
            reward_model = extra_info.get("reward_model", {})
            rubrics = reward_model.get("rubrics", [])
            
            if not prompt or not rubrics:
                final_scores.append(0.0)
                continue
                
            rubric_items = [RubricItem.from_dict(r) for r in rubrics]
            
            # Get all grading results for this sample
            if sample_idx in sample_results:
                grading_response_list = []
                for rubric_idx in range(len(rubric_items)):
                    if rubric_idx in sample_results[sample_idx]:
                        grading_response_list.append(sample_results[sample_idx][rubric_idx])
                    else:
                        # If a rubric has no result, use default failure result
                        grading_response_list.append({
                            "criteria_met": False,
                            "explanation": "Grading task not completed"
                        })
                
                # Calculate total score
                overall_score = calculate_score(rubric_items, grading_response_list)
                final_scores.append(overall_score)
            else:
                final_scores.append(0.0)
                
        except Exception as e:
            print(f"Error calculating score for sample {sample_idx}: {e}")
            final_scores.append(0.0)
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print statistics
    print("\n" + "="*60)
    print("Batch grading statistics:")
    print(f"Total criterion count: {total_criteria}")
    print(f"Rule-based verification count: {rule_based_criteria}")
    print(f"LLM as a Judge count: {llm_criteria}")
    print(f"LLM API calls (batch tasks): {len(llm_tasks)} (batch_size={rubric_batch_size})")
    if failed_rule_criteria > 0:
        print(f"Failed Rule Function count: {failed_rule_criteria}")
    
    # Print function call counts (only print non-zero counts)
    if function_call_stats:
        print("\nRule Function call statistics:")
        for func_name, count in function_call_stats.items():
            if count > 0:
                print(f"  {func_name}: {count} times")

    # Print judge output length statistics
    grader.print_output_stats(prefix="\n")
    grader.reset_output_stats()

    print(f"\nTotal time: {total_time:.2f} seconds")
    if total_criteria > 0:
        print(f"Average time per criterion: {total_time/total_criteria:.3f} seconds")
    print("="*60 + "\n")
    
    return final_scores
