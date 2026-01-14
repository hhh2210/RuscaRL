import json
import re
import os
import requests
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
from dotenv import load_dotenv
import importlib
from health_bench.vllm_sampler import VLLMSampler, VLLMDataInspectionError

# Import verification function module
from verl.utils.reward_score.rule_fn import get_verification_function, VERIFICATION_FUNCTIONS

# Concurrent workers control variable for single URL
MAX_CONCURRENT_WORKERS = 512

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


# Optional debug capture for judge responses (used by small eval scripts).
# Enabled when env `RUSCARL_CAPTURE_JUDGE_TEXT=1`.
_LAST_JUDGE_DEBUG: Dict[str, Any] = {}

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

def _format_prompt_messages(prompt: List[Dict[str, str]]) -> str:
    """Convert prompt messages to a readable string, skipping system role"""
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in prompt if m.get("role") != "system"
    )

def _build_batch_grader_prompt(prompt: List[Dict[str, str]], response: str, rubric_items: List[RubricItem]) -> str:
    """Build the unified grading prompt for LLM-as-a-judge"""
    prompt_str = _format_prompt_messages(prompt)
    rubrics_str = "\n".join(
        f"{idx + 1}. (points: {item.points}) {item.criterion}"
        for idx, item in enumerate(rubric_items)
    )
    return f'''You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please evaluate the response against EACH rubric.

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
</Prompt>

<Response>
{response}
</Response>

<Rubrics>
{rubrics_str}
</Rubrics>'''

def _parse_presence_response(resp_text: str, expected_count: int) -> Dict[int, bool]:
    """
    Parse batch LLM grading JSON. Keys are 1-based rubric indices, values PRESENT/NOT_PRESENT.
    Returns mapping of 1-based index -> bool.
    """
    cleaned = re.sub(r"^```json\s*|\s*```$", "", resp_text.strip())
    try:
        data = json.loads(cleaned)
    except Exception:
        return {}
    results = {}
    for key, val in data.items():
        try:
            idx = int(key)
        except Exception:
            continue
        if idx < 1 or idx > expected_count:
            continue
        if isinstance(val, str):
            norm = val.strip().upper()
            if norm == "PRESENT":
                results[idx] = True
            elif norm == "NOT_PRESENT":
                results[idx] = False
    return results

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
    return max(0,overall_score)

def grade_single_example(
    prompt: List[Dict[str, str]], 
    response: str,
    rubric_items: List[RubricItem],
    grader_model,
    executor=None,  # New parameter: external thread pool
) -> Tuple[float, str, List[Dict]]:
    """Evaluate a single example with rule checks + batched LLM grading for non-rule criteria"""
    def run_rule_check(rubric_item: RubricItem) -> dict:
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
                else:
                    raise ValueError(f"Verification function '{function_name}' not found")
        return None  # Not a rule task

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
        llm_items = [rubric_items[i] for i in llm_indices]
        max_retries = 3
        retry_count = 0
        llm_results = {}
        while retry_count < max_retries:
            prompt_text = _build_batch_grader_prompt(prompt, response, llm_items)
            sampler_response = grader_model([dict(content=prompt_text, role="user")])
            if os.getenv("RUSCARL_CAPTURE_JUDGE_TEXT", "").strip().lower() in ("1", "true", "yes", "y", "on"):
                _LAST_JUDGE_DEBUG.clear()
                _LAST_JUDGE_DEBUG.update(
                    {
                        "prompt_text": prompt_text,
                        "response_text": getattr(sampler_response, "response_text", None),
                        "actual_queried_message_list": getattr(sampler_response, "actual_queried_message_list", None),
                    }
                )
            llm_results = _parse_presence_response(
                sampler_response.response_text,
                expected_count=len(llm_items)
            )
            if len(llm_results) == len(llm_items):
                break
            retry_count += 1

        if len(llm_results) != len(llm_items):
            print(f"Batch grading failure count reached limit ({max_retries} times), marking all LLM criteria as NOT_PRESENT")
            llm_results = {i + 1: False for i in range(len(llm_items))}

        for local_idx, rubric_global_idx in enumerate(llm_indices):
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
        if reward_model is None:
            return 0.0

        # Normalize prompt / rubrics (handle numpy arrays from parquet)
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, tuple):
            prompt = list(prompt)

        rubrics = reward_model.get("rubrics", [])
        if hasattr(rubrics, "tolist"):
            rubrics = rubrics.tolist()
        if isinstance(rubrics, tuple):
            rubrics = list(rubrics)

        if prompt is None or len(prompt) == 0 or rubrics is None or len(rubrics) == 0:
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
    scores, skipped_reasons = batch_compute_scores(batch_data, max_workers_per_url=max_workers_per_url)
    
    # Convert scores to dictionary format containing score and acc fields
    results = []
    for i, score in enumerate(scores):
        results.append({
            "score": score,
            "acc": score > 0.5,  # Convert score to accuracy (boolean value)
            "skipped": i in skipped_reasons,
            "skip_reason": skipped_reasons.get(i),
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

def batch_compute_scores(batch_data: List[Tuple[str, str, str, Dict[str, Any]]], max_workers_per_url: int = MAX_CONCURRENT_WORKERS) -> Tuple[List[float], Dict[int, str]]:
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
        Tuple[List[float], Dict[int, str]]: (reward scores, skipped sample reasons keyed by sample_idx)
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
            if (rubric_item.tags and 
                rubric_item.tags.get("verifier") == "rule" and 
                rubric_item.tags.get("function") and 
                rubric_item.tags.get("parameters") is not None):
                rule_tasks.append({
                    'sample_idx': sample_idx,
                    'rubric_idx': rubric_idx,
                    'prompt': prompt,
                    'response': solution_str,
                    'rubric_item': rubric_item
                })
            else:
                llm_indices.append(rubric_idx)
        if llm_indices:
            llm_tasks.append({
                'sample_idx': sample_idx,
                'prompt': prompt,
                'response': solution_str,
                'rubric_items': [rubric_items[i] for i in llm_indices],
                'rubric_indices': llm_indices
            })
    
    # Dictionary for tracking function call counts
    function_call_stats = {}

    # Track samples skipped due to provider-side moderation / data inspection blocks.
    skipped_reasons: Dict[int, str] = {}
    try:
        import threading
        _skip_lock = threading.Lock()
    except Exception:
        _skip_lock = None

    def _mark_sample_skipped(sample_idx: int, reason: str) -> None:
        if sample_idx is None or sample_idx < 0:
            return
        if _skip_lock is None:
            skipped_reasons.setdefault(sample_idx, reason)
            return
        with _skip_lock:
            skipped_reasons.setdefault(sample_idx, reason)
    
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
        """Process batched LLM-based criteria for a single sample"""
        try:
            rubric_items = task['rubric_items']
            rubric_indices = task['rubric_indices']
            max_retries = 3
            retry_count = 0
            llm_results = {}
            while retry_count < max_retries:
                batch_prompt = _build_batch_grader_prompt(task['prompt'], task['response'], rubric_items)
                sampler_response = grader([dict(content=batch_prompt, role="user")])
                llm_results = _parse_presence_response(
                    sampler_response.response_text,
                    expected_count=len(rubric_items)
                )
                if len(llm_results) == len(rubric_items):
                    break
                retry_count += 1
            if len(llm_results) != len(rubric_items):
                print(f"LLM grading failure count reached limit ({max_retries} times), marking all as NOT_PRESENT")
                llm_results = {i + 1: False for i in range(len(rubric_items))}
            
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
            sample_idx = task.get('sample_idx', -1)
            err_text = str(e)
            if isinstance(e, VLLMDataInspectionError) or "data_inspection_failed" in err_text.lower():
                _mark_sample_skipped(sample_idx, err_text)
                return [{
                    'sample_idx': sample_idx,
                    'rubric_idx': idx,
                    'result': {
                        "criteria_met": False,
                        "explanation": "Skipped: provider data inspection blocked the request (data_inspection_failed)"
                    },
                    'verification_type': 'llm_skipped'
                } for idx in task.get('rubric_indices', [])]

            print(f"LLM task error: {e}")
            return [{
                'sample_idx': sample_idx,
                'rubric_idx': idx,
                'result': {
                    "criteria_met": False,
                    "explanation": f"LLM processing error: {str(e)}"
                },
                'verification_type': 'llm_failed'
            } for idx in task.get('rubric_indices', [])]
    
    # Count total criteria
    total_criteria = len(rule_tasks) + sum(len(task['rubric_indices']) for task in llm_tasks)
    
    # Store all results
    sample_results = {}  # sample_idx -> {rubric_idx: result}
    failed_rule_criteria = 0
    
    print(f"\nProcessing {len(rule_tasks)} rule tasks and {len(llm_tasks)} LLM tasks...")
    
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
                    
                    verification_type = result.get('verification_type', 'unknown')
                    if verification_type == 'llm':
                        llm_criteria += 1
    
    llm_end_time = time.time()
    print(f"LLM processing completed in {llm_end_time - llm_start_time:.2f} seconds")
    
    # Calculate final scores for each sample
    final_scores = []
    for sample_idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(batch_data):
        try:
            if sample_idx in skipped_reasons:
                final_scores.append(0.0)
                continue

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
    if skipped_reasons:
        print(f"Skipped sample count (data inspection): {len(skipped_reasons)}")
    if failed_rule_criteria > 0:
        print(f"Failed Rule Function count: {failed_rule_criteria}")
    
    # Print function call counts (only print non-zero counts)
    if function_call_stats:
        print("\nRule Function call statistics:")
        for func_name, count in function_call_stats.items():
            if count > 0:
                print(f"  {func_name}: {count} times")
    
    print(f"\nTotal time: {total_time:.2f} seconds")
    if total_criteria > 0:
        print(f"Average time per criterion: {total_time/total_criteria:.3f} seconds")
    print("="*60 + "\n")
    
    return final_scores, skipped_reasons
