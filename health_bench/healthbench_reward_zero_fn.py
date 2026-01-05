import json
import re
import os
import requests
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from openai import OpenAI
from dotenv import load_dotenv
import importlib

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

class VLLMSampler(SamplerBase):
    """Local VLLM service sampler"""
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        enable_thinking: bool = False,
        filter_think_tags: bool = True,
    ):
        # Support multiple URL configuration
        if base_url:
            self.base_urls = [base_url]
        else:
            # Read URL list from environment variable, support comma separation
            url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
            self.base_urls = [url.strip() for url in url_env.split(',') if url.strip()]
        
        # Load balancing related variables
        self.current_url_index = 0  # Index for round-robin
        self.url_loads = {}  # Store load information for each URL
        self.virtual_loads = {}  # Store virtual load for each URL (including allocated requests)
        
        # Initialize load information for each URL
        for url in self.base_urls:
            self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
            self.virtual_loads[url] = 0  # Initial virtual load is 0
        
        # Perform load statistics once during initialization
        self._update_loads()
        
        self.model = model or os.getenv("VLLM_MODEL", "default")
        self.system_message = system_message
        self.temperature = temperature if temperature is not None else float(os.getenv("VLLM_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "2048"))
        self.timeout = timeout if timeout is not None else int(os.getenv("VLLM_TIMEOUT", "120"))
        self.enable_thinking = enable_thinking
        self.filter_think_tags = filter_think_tags

        metrics_enabled = os.getenv("VLLM_METRICS_ENABLED", "auto").strip().lower()
        if metrics_enabled == "auto":
            metrics_enabled = "true" if any(url.startswith(p) for p in ("http://localhost", "http://127.0.0.1")) else "false"
        self.metrics_enabled = metrics_enabled in ("1", "true", "yes", "y", "on")

        # Rate limiting / throttling (important for remote OpenAI-compatible endpoints).
        # - If MIN_INTERVAL_S > 0: enforce a fixed minimum interval between requests per URL.
        # - If MIN_INTERVAL_S < 0: use a random jitter interval in [0, abs(MIN_INTERVAL_S)] per request (per URL).
        # The throttle is applied per URL and coordinated across concurrent threads.
        min_interval_env = os.getenv("MIN_INTERVAL_S", "0").strip()
        try:
            self.min_interval_s = float(min_interval_env)
        except Exception:
            self.min_interval_s = 0.0
        self._min_interval_is_jitter = self.min_interval_s < 0
        self._min_interval_abs_s = abs(self.min_interval_s)
        # Next allowed request timestamp per URL (monotonic clock).
        self._next_request_ts_by_url = {url: 0.0 for url in self.base_urls}
        try:
            import threading
            self._throttle_lock = threading.Lock()
        except Exception:
            self._throttle_lock = None

        api_key = (
            os.getenv("VLLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        ).strip()
        auth_header = f"Bearer {api_key}" if api_key else "Bearer dummy"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": auth_header,
        }
        
        print(f"VLLMSampler initialization completed, configured {len(self.base_urls)} URLs: {self.base_urls}")
        
        # Print load information during initialization
        print("\n=== Initialization Load Information ===")
        available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
        if available_urls:
             total_running = sum(self.url_loads[url]['running'] for url in available_urls)
             total_waiting = sum(self.url_loads[url]['waiting'] for url in available_urls)
             total_load = sum(self.url_loads[url]['total'] for url in available_urls)
             print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total load: {total_load/len(available_urls):.1f}")
        print(f"Available servers: {len(available_urls)}/{len(self.base_urls)}")
        print("========================\n")

    def _throttle(self, url: str) -> None:
        interval_abs_s = getattr(self, "_min_interval_abs_s", 0.0)
        if interval_abs_s <= 0:
            return

        is_jitter = getattr(self, "_min_interval_is_jitter", False)
        lock = getattr(self, "_throttle_lock", None)

        now = time.monotonic()
        if lock is None:
            next_ts = self._next_request_ts_by_url.get(url, 0.0)
            scheduled_ts = max(now, next_ts)
            interval_s = random.uniform(0.0, interval_abs_s) if is_jitter else interval_abs_s
            self._next_request_ts_by_url[url] = scheduled_ts + interval_s
            wait_s = scheduled_ts - now
            if wait_s > 0:
                time.sleep(wait_s)
            return

        with lock:
            now = time.monotonic()
            next_ts = self._next_request_ts_by_url.get(url, 0.0)
            scheduled_ts = max(now, next_ts)
            interval_s = random.uniform(0.0, interval_abs_s) if is_jitter else interval_abs_s
            self._next_request_ts_by_url[url] = scheduled_ts + interval_s
            wait_s = scheduled_ts - now

        if wait_s > 0:
            time.sleep(wait_s)

    def _filter_think_tags(self, text: str) -> str:
        """Remove <think></think> tags and their content"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _get_url_load(self, url: str) -> dict:
        """Get load information for a single URL"""
        if not getattr(self, "metrics_enabled", True):
            return {'running': 0, 'waiting': 0, 'total': 0, 'available': True}
        try:
            # Build metrics endpoint URL
            if url.endswith('/v1'):
                base_url = url[:-3]  # Remove trailing '/v1'
            else:
                base_url = url.rstrip('/')
            metrics_url = f"{base_url}/metrics"
            
            response = requests.get(metrics_url, timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Parse Prometheus format metrics
                running = self._parse_metric_value(metrics_text, 'vllm:num_requests_running')
                waiting = self._parse_metric_value(metrics_text, 'vllm:num_requests_waiting')
                
                return {
                    'running': running,
                    'waiting': waiting,
                    'total': running + waiting,
                    'available': True
                }
        except Exception as e:
            # If retrieval fails, return default values
            pass
        
        return {'running': 0, 'waiting': 0, 'total': 0, 'available': False}
    
    def _parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        """Parse the value of specified metric from Prometheus format metrics text"""
        try:
            pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)'
            matches = re.findall(pattern, metrics_text, re.MULTILINE)
            if matches:
                return int(float(matches[0]))
        except Exception:
            pass
        return 0
    
    def _reload_urls_from_env(self):
        """Reload URL configuration from environment variables"""
        
        # Reload .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)  # Force reload
        except ImportError:
            # If python-dotenv is not available, manually read .env file
            env_file = '.env'
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()
        
        # Re-read URL configuration
        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        new_urls = [url.strip() for url in url_env.split(',') if url.strip()]
        
        # Check if URLs have changed
        old_urls = set(self.base_urls)
        new_urls_set = set(new_urls)
        url_changed = old_urls != new_urls_set
        
        if url_changed:
            print(f"Detected URL configuration change: {self.base_urls} -> {new_urls}")
            
            # Remove URLs that are no longer used
            for url in old_urls - new_urls_set:
                if url in self.url_loads:
                    del self.url_loads[url]
                if url in self.virtual_loads:
                    del self.virtual_loads[url]
            
            # Add new URLs
            for url in new_urls_set - old_urls:
                self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
                self.virtual_loads[url] = 0
            
            # Update URL list
            self.base_urls = new_urls
            print(f"URL configuration updated, current configuration: {self.base_urls}")
        
        return url_changed
    
    def _update_loads(self):
        """Update load information for all URLs"""
        # Get current load for all URLs
        for url in self.base_urls:
            load_info = self._get_url_load(url)
            self.url_loads[url] = load_info
    
    def _get_next_url(self) -> str:
        """Select the URL with the lowest load based on fill-the-gap algorithm"""
        # Get all available URLs
        available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
        
        if not available_urls:
            # If no URLs are available, wait for server recovery
            while True:
                # Reload URL configuration from .env
                self._reload_urls_from_env()
                    # Reload environment variables and update URL configuration at the beginning of each batch
                self._update_loads()
                

                # Recheck available URLs
                available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
                
                if available_urls:
                    break
                
                time.sleep(10)
        
        # Find the URL with the lowest virtual load (fill-the-gap algorithm)
        min_virtual_load = float('inf')
        selected_url = available_urls[0]
        
        for url in available_urls:
            # Virtual load = actual load + allocated but unfinished requests
            virtual_load = self.url_loads[url]['total'] + self.virtual_loads[url]
            if virtual_load < min_virtual_load:
                min_virtual_load = virtual_load
                selected_url = url
        
        # After selecting URL, increase its virtual load
        self.virtual_loads[selected_url] += 1
        
        return selected_url

    def __call__(self, message_list: List[Dict[str, str]]) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message(self.system_message, "system")
            ] + message_list

        trial = 0
        current_url = None
        while True:
            try:
                # Select URL
                current_url = self._get_next_url()
                self._throttle(current_url)
                strict_env = os.getenv("VLLM_STRICT_OPENAI_COMPAT", "auto").strip().lower()
                if strict_env in ("1", "true", "yes", "y", "on"):
                    strict_openai_compat = True
                elif strict_env in ("0", "false", "no", "n", "off"):
                    strict_openai_compat = False
                else:
                    strict_openai_compat = not getattr(self, "metrics_enabled", True)

                # Use different configurations based on enable_thinking
                if self.enable_thinking:
                    payload = {
                        "model": self.model,
                        "messages": message_list,
                        "temperature": 0.6,
                        "top_p": 0.95,
                        "presence_penalty": 1.5,
                        "max_tokens": self.max_tokens,
                    }
                    if not strict_openai_compat:
                        payload.update(
                            {
                                "top_k": 20,
                                "min_p": 0,
                                "chat_template_kwargs": {"enable_thinking": True},
                            }
                        )
                else:
                    payload = {
                        "model": self.model,
                        "messages": message_list,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "presence_penalty": 1.5,
                        "max_tokens": self.max_tokens,
                    }
                    if not strict_openai_compat:
                        payload.update(
                            {
                                "top_k": 20,
                                "min_p": 0,
                                "chat_template_kwargs": {"enable_thinking": False},
                            }
                        )

                # DashScope compatibility: some "thinking" models require explicitly disabling thinking
                # for non-streaming calls, otherwise the API returns HTTP 400.
                if "dashscope.aliyuncs.com/compatible-mode" in current_url:
                    payload.setdefault("enable_thinking", False)

                response = requests.post(
                    f"{current_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                if content is None:
                    raise ValueError("VLLM service returned empty response, retrying...")
                
                if self.filter_think_tags:
                    content = self._filter_think_tags(content)
                
                # Request successful, decrease virtual load
                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)
                
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response_data.get("usage", {})},
                    actual_queried_message_list=message_list,
                )
            
            except requests.exceptions.RequestException as e:
                # Request failed, also decrease virtual load
                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                status_code = None
                retry_after_s = None
                resp = getattr(e, "response", None)
                if resp is not None:
                    status_code = getattr(resp, "status_code", None)
                    if status_code == 429:
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                retry_after_s = float(ra)
                            except Exception:
                                retry_after_s = None
                    elif status_code is not None and 400 <= int(status_code) < 500:
                        try:
                            resp_text = (resp.text or "").strip()
                        except Exception:
                            resp_text = ""
                        if resp_text:
                            print(f"Non-retriable HTTP {status_code} body: {resp_text[:800]}")
                        raise

                exception_backoff = min(2**trial, 300)  # Maximum wait time 300 seconds
                sleep_s = exception_backoff
                if retry_after_s is not None:
                    sleep_s = max(sleep_s, retry_after_s)
                sleep_s += random.uniform(0.0, 0.25)

                print(
                    f"Request failed (URL: {current_url}, status={status_code}), retrying attempt {trial} after {sleep_s:.2f} seconds",
                    str(e),
                )
                if isinstance(e, requests.exceptions.ConnectionError):
                    print(f"Connection error: Please ensure VLLM service is running at {current_url}")
                    # On connection error, try to reload .env file and update URL configuration
                    print("Detected connection error, attempting to reload .env file and update URL configuration...")
                    url_changed = self._reload_urls_from_env()
                    if url_changed:
                        print("URL configuration updated, will retry with new URL configuration")
                    else:
                        print("No change in URL configuration")
                    
                    # Re-detect load status of all servers (like initialization)
                    self._update_loads()
                    
                    # Print average load information
                    available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
                    if available_urls:
                         total_running = sum(self.url_loads[url]['running'] for url in available_urls)
                         total_waiting = sum(self.url_loads[url]['waiting'] for url in available_urls)
                         total_load = sum(self.url_loads[url]['total'] for url in available_urls)
                         print(f"Average load - Running: {total_running/len(available_urls):.1f}, Waiting: {total_waiting/len(available_urls):.1f}, Total: {total_load/len(available_urls):.1f}")
                    
                    print(f"Load detection completed, current available servers: {len(available_urls)}/{len(self.base_urls)}")
                    print("================================\n")
                elif isinstance(e, requests.exceptions.Timeout):
                    print(f"Timeout error: Request exceeded {self.timeout} seconds")
                time.sleep(sleep_s)
                trial += 1
                # Remove retry limit to implement infinite retries

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
    """Evaluate a single example
    
    Args:
        prompt: Conversation history
        response: Model response
        rubric_items: List of grading criteria
        grader_model: Grading model
        executor: External thread pool for global concurrency control
        
    Returns:
        tuple: (score, detailed explanation, grading results for each criterion)
    """
    # Build complete conversation
    convo_with_response = prompt + [dict(content=response, role="assistant")]
    
    def grade_rubric_item(rubric_item: RubricItem) -> dict:
        # Check if tags contain rule-based verification
        if rubric_item.tags:
            # Check if there's a verifier tag with value 'rule'
            verifier_type = rubric_item.tags.get("verifier")
            if verifier_type == "rule":
                # Get function and parameters directly from dictionary
                function_name = rubric_item.tags.get("function")
                parameter_value = rubric_item.tags.get("parameters")
                
                # Dynamically call verification function
                if function_name and parameter_value is not None:
                    # Get function from verification function registry
                    verify_func = get_verification_function(function_name) if get_verification_function else None
                    if verify_func:
                        criteria_met = verify_func(response, parameter_value)
                        return {
                            "criteria_met": criteria_met,
                            "explanation": f"Rule-based verification using {function_name} with parameter '{parameter_value}': {'PASS' if criteria_met else 'FAIL'}"
                        }
                    else:
                        raise ValueError(f"Verification function '{function_name}' not found")
        
        # Model-based verification
        # If no rule-based verification or verification failed, use traditional LLM grading
        # Build conversation string, filtering out system role messages
        convo_str = "\n\n".join(
            [f"{m['role']}: {m['content']}" for m in convo_with_response if m['role'] != 'system']
        )
        # Build grading prompt
        grader_prompt = GRADER_TEMPLATE.replace(
            "<<conversation>>", convo_str
        ).replace("<<rubric_item>>", str(rubric_item))
        messages = [dict(content=grader_prompt, role="user")]
        
        # Call grading model
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            sampler_response = grader_model(messages)
            # Get response_text from SamplerResponse object
            grading_response_dict = parse_json_to_dict(sampler_response.response_text)
            if "criteria_met" in grading_response_dict:
                label = grading_response_dict["criteria_met"]
                if label is True or label is False:
                    break
            # Grading failed, JSON output error, retrying (not printing details to avoid log pollution)
            pass
            retry_count += 1
            
        # If retry count reaches limit, return failure result
        if retry_count >= max_retries:
            print(f"Grading failure count reached limit ({max_retries} times), returning failure result")
            return {
                "criteria_met": False,
                "explanation": "Too many JSON parsing failures"
            }
            
        return grading_response_dict

    # Evaluate each criterion - only use external thread pool (single-layer global concurrency)
    if executor is not None:
        # Use external thread pool (global concurrency control)
        futures = [executor.submit(grade_rubric_item, rubric_item) for rubric_item in rubric_items]
        grading_response_list = [future.result() for future in futures]
    else:
        # If no external thread pool, execute sequentially (for single example testing)
        print("No external thread pool, executing sequentially")
        grading_response_list = [grade_rubric_item(rubric_item) for rubric_item in rubric_items]

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
        for rubric_idx, rubric_item in enumerate(rubric_items):
            task = {
                'sample_idx': sample_idx,
                'rubric_idx': rubric_idx,
                'prompt': prompt,
                'response': solution_str,
                'rubric_item': rubric_item,
                'rubric_items': rubric_items
            }
            
            # Check if this is a rule-based task
            if (rubric_item.tags and 
                rubric_item.tags.get("verifier") == "rule" and 
                rubric_item.tags.get("function")):
                rule_tasks.append(task)
            else:
                llm_tasks.append(task)
    
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
        """Process single LLM-based task"""
        try:
            # Build complete conversation
            convo_with_response = task['prompt'] + [dict(content=task['response'], role="assistant")]
            
            # Build conversation string, filtering out system role messages
            convo_str = "\n\n".join(
                [f"{m['role']}: {m['content']}" for m in convo_with_response if m['role'] != 'system']
            )
            
            # Build grading prompt
            grader_prompt = GRADER_TEMPLATE.replace(
                "<<conversation>>", convo_str
            ).replace("<<rubric_item>>", str(task['rubric_item']))
            messages = [dict(content=grader_prompt, role="user")]
            
            # Call grading model
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                sampler_response = grader(messages)
                grading_response_dict = parse_json_to_dict(sampler_response.response_text)
                if "criteria_met" in grading_response_dict:
                    label = grading_response_dict["criteria_met"]
                    if label is True or label is False:
                        return {
                            'sample_idx': task['sample_idx'],
                            'rubric_idx': task['rubric_idx'],
                            'result': grading_response_dict,
                            'verification_type': 'llm'
                        }
                # Grading failed, JSON output error, retrying (not printing details to avoid log pollution)
                pass
                retry_count += 1
                
            # If retry count reaches limit, return failure result
            print(f"LLM grading failure count reached limit ({max_retries} times), returning failure result")
            return {
                'sample_idx': task['sample_idx'],
                'rubric_idx': task['rubric_idx'],
                'result': {
                    "criteria_met": False,
                    "explanation": "JSON parsing failed too many times"
                },
                'verification_type': 'llm_failed'
            }
            
        except Exception as e:
            print(f"LLM task error: {e}")
            return {
                'sample_idx': task['sample_idx'],
                'rubric_idx': task['rubric_idx'],
                'result': {
                    "criteria_met": False,
                    "explanation": f"LLM processing error: {str(e)}"
                },
                'verification_type': 'llm_failed'
            }
    
    # Count total criteria
    total_criteria = len(rule_tasks) + len(llm_tasks)
    
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
                result = future.result()
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
    
    return final_scores
