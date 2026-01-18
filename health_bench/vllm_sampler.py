import json
import os
import re
import time
import random
import requests
from dataclasses import dataclass
from typing import List, Dict

from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Only needed when ChatCompletionSampler is used.

# Load env so VLLM_BASE_URL / keys are available even when imported standalone.
load_dotenv()


@dataclass
class SamplerResponse:
    """Uniform response container for sampler outputs."""
    response_text: str
    response_metadata: dict
    actual_queried_message_list: List[Dict[str, str]]


class SamplerBase:
    """Base sampler class."""
    def _pack_message(self, content: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": content}


class VLLMStreamError(RuntimeError):
    """Raised when an OpenAI-compatible SSE stream contains an error payload."""

    def __init__(self, error_payload: object):
        self.error_payload = error_payload
        super().__init__(f"SSE stream returned error payload: {error_payload}")


class VLLMDataInspectionError(VLLMStreamError):
    """
    Raised when the serving platform blocks the request via content/data inspection.

    This is typically a non-transient error and should NOT be retried with exponential backoff.
    """


def _is_data_inspection_error_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    code = str(payload.get("code") or payload.get("type") or "").strip().lower()
    return code in ("data_inspection_failed", "inappropriate_content")


def _exception_mentions_data_inspection(exc: BaseException) -> bool:
    try:
        text = str(exc).lower()
    except Exception:
        return False
    return "data_inspection_failed" in text or "inappropriate content" in text


class ChatCompletionSampler(SamplerBase):
    """Plain OpenAI-compatible chat sampler (fallback)."""

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
                exception_backoff = min(2 ** trial, 300)  # Exponential backoff, max 300s
                print(
                    f"Rate limit exception, waiting {exception_backoff} seconds before retry {trial}",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1


class VLLMSampler(SamplerBase):
    """OpenAI-compatible sampler with multi-URL load balancing and metrics probing."""

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
            url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
            self.base_urls = [url.strip() for url in url_env.split(',') if url.strip()]

        # Load balancing related variables
        self.current_url_index = 0
        self.url_loads: Dict[str, dict] = {}
        self.virtual_loads: Dict[str, int] = {}

        for url in self.base_urls:
            self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
            self.virtual_loads[url] = 0

        # Perform load statistics once during initialization
        self._update_loads()

        self.model = model or os.getenv("VLLM_MODEL", "default")
        self.system_message = system_message
        self.temperature = temperature if temperature is not None else float(os.getenv("VLLM_TEMPERATURE", "0.7"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("VLLM_MAX_TOKENS", "4096"))
        self.timeout = timeout if timeout is not None else int(os.getenv("VLLM_TIMEOUT", "120"))
        self.enable_thinking = enable_thinking
        self.filter_think_tags = filter_think_tags

        metrics_enabled = os.getenv("VLLM_METRICS_ENABLED", "auto").strip().lower()
        if metrics_enabled == "auto":
            metrics_enabled = "true" if any(url.startswith(p) for p in ("http://localhost", "http://127.0.0.1")) else "false"
        self.metrics_enabled = metrics_enabled in ("1", "true", "yes", "y", "on")

        min_interval_env = os.getenv("MIN_INTERVAL_S", "0").strip()
        try:
            self.min_interval_s = float(min_interval_env)
        except Exception:
            self.min_interval_s = 0.0
        self._min_interval_is_jitter = self.min_interval_s < 0
        self._min_interval_abs_s = abs(self.min_interval_s)
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

        # Output length statistics
        self.output_lengths: List[int] = []

        print(f"VLLMSampler initialization completed, configured {len(self.base_urls)} URLs: {self.base_urls}")
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
        """Remove <think></think> tags and their content."""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def _iter_sse_lines(self, resp: requests.Response):
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            yield line

    def _consume_stream_content(self, resp: requests.Response) -> tuple[str, dict | None]:
        content_parts: List[str] = []
        usage = None
        last_error = None
        for line in self._iter_sse_lines(resp):
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except Exception:
                continue
            if isinstance(payload, dict) and "error" in payload:
                # Some providers return errors inside the SSE stream with 200 OK.
                last_error = payload.get("error")
                continue
            if isinstance(payload, dict) and "usage" in payload:
                usage = payload.get("usage")
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if not choices:
                continue
            choice0 = choices[0] if isinstance(choices, list) else {}
            delta = choice0.get("delta") or choice0.get("message") or {}
            piece = delta.get("content")
            if piece:
                content_parts.append(piece)
        content = "".join(content_parts)
        if not content.strip():
            if last_error is not None:
                if _is_data_inspection_error_payload(last_error):
                    raise VLLMDataInspectionError(last_error)
                raise VLLMStreamError(last_error)
            raise RuntimeError("SSE stream returned empty content (no choices).")
        return content, usage

    def _get_url_load(self, url: str) -> dict:
        """Get load information for a single URL."""
        if not getattr(self, "metrics_enabled", True):
            return {'running': 0, 'waiting': 0, 'total': 0, 'available': True}
        try:
            if url.endswith('/v1'):
                base_url = url[:-3]
            else:
                base_url = url.rstrip('/')
            metrics_url = f"{base_url}/metrics"

            response = requests.get(metrics_url, timeout=5)
            if response.status_code == 200:
                metrics_text = response.text

                running = self._parse_metric_value(metrics_text, 'vllm:num_requests_running')
                waiting = self._parse_metric_value(metrics_text, 'vllm:num_requests_waiting')

                return {
                    'running': running,
                    'waiting': waiting,
                    'total': running + waiting,
                    'available': True
                }
        except Exception:
            pass

        return {'running': 0, 'waiting': 0, 'total': 0, 'available': False}

    def _parse_metric_value(self, metrics_text: str, metric_name: str) -> int:
        """Parse the value of specified metric from Prometheus format metrics text."""
        try:
            pattern = rf'^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9.]+)'
            matches = re.findall(pattern, metrics_text, re.MULTILINE)
            if matches:
                return int(float(matches[0]))
        except Exception:
            pass
        return 0

    def _reload_urls_from_env(self):
        """Reload URL configuration from environment variables."""
        try:
            from dotenv import load_dotenv as _reload_dotenv
            _reload_dotenv(override=True)
        except ImportError:
            env_file = '.env'
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip()

        url_env = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        new_urls = [url.strip() for url in url_env.split(',') if url.strip()]

        old_urls = set(self.base_urls)
        new_urls_set = set(new_urls)
        url_changed = old_urls != new_urls_set

        if url_changed:
            print(f"Detected URL configuration change: {self.base_urls} -> {new_urls}")

            for url in old_urls - new_urls_set:
                self.url_loads.pop(url, None)
                self.virtual_loads.pop(url, None)

            for url in new_urls_set - old_urls:
                self.url_loads[url] = {'running': 0, 'waiting': 0, 'total': 0}
                self.virtual_loads[url] = 0

            self.base_urls = new_urls
            print(f"URL configuration updated, current configuration: {self.base_urls}")

        return url_changed

    def _update_loads(self):
        """Update load information for all URLs."""
        for url in self.base_urls:
            load_info = self._get_url_load(url)
            self.url_loads[url] = load_info

    def _get_next_url(self) -> str:
        """Select the URL with the lowest load based on fill-the-gap algorithm."""
        available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]

        if not available_urls:
            while True:
                self._reload_urls_from_env()
                self._update_loads()
                available_urls = [url for url in self.base_urls if self.url_loads[url].get('available', False)]
                if available_urls:
                    break
                time.sleep(10)

        min_virtual_load = float('inf')
        selected_url = available_urls[0]

        for url in available_urls:
            virtual_load = self.url_loads[url]['total'] + self.virtual_loads[url]
            if virtual_load < min_virtual_load:
                min_virtual_load = virtual_load
                selected_url = url

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
                current_url = self._get_next_url()
                self._throttle(current_url)
                strict_env = os.getenv("VLLM_STRICT_OPENAI_COMPAT", "auto").strip().lower()
                if strict_env in ("1", "true", "yes", "y", "on"):
                    strict_openai_compat = True
                elif strict_env in ("0", "false", "no", "n", "off"):
                    strict_openai_compat = False
                else:
                    strict_openai_compat = not getattr(self, "metrics_enabled", True)

                payload = {
                    "model": self.model,
                    "messages": message_list,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if not strict_openai_compat:
                    # Do not send top_p/top_k by default; let the serving platform decide.
                    pass

                # Always pass enable_thinking parameter to control thinking output
                payload["enable_thinking"] = self.enable_thinking

                use_stream = False
                if "dashscope.aliyuncs.com/compatible-mode" in current_url:
                    payload.setdefault("stream", True)
                    use_stream = True

                response = requests.post(
                    f"{current_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                    stream=use_stream,
                )
                response.raise_for_status()
                if use_stream:
                    content, usage = self._consume_stream_content(response)
                else:
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]
                    usage = response_data.get("usage")

                if content is None or not str(content).strip():
                    raise ValueError("VLLM service returned empty response, retrying...")

                if self.filter_think_tags:
                    content = self._filter_think_tags(content)

                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                # Record output length for statistics
                self.output_lengths.append(len(content))

                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": usage},
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                if current_url and current_url in self.virtual_loads:
                    self.virtual_loads[current_url] = max(0, self.virtual_loads[current_url] - 1)

                # Provider-side data inspection blocks are not transient; don't retry forever.
                if isinstance(e, VLLMDataInspectionError) or _exception_mentions_data_inspection(e):
                    raise

                print(f"Request to VLLM service failed, retrying... {e}")
                trial += 1
                self._reload_urls_from_env()
                self._update_loads()
                time.sleep(min(2 ** trial, 300))

    def get_output_stats(self) -> dict:
        """Get output length statistics."""
        if not self.output_lengths:
            return {'count': 0, 'avg_len': 0, 'min_len': 0, 'max_len': 0, 'total_chars': 0}
        return {
            'count': len(self.output_lengths),
            'avg_len': sum(self.output_lengths) / len(self.output_lengths),
            'min_len': min(self.output_lengths),
            'max_len': max(self.output_lengths),
            'total_chars': sum(self.output_lengths),
        }

    def reset_output_stats(self):
        """Reset output length statistics."""
        self.output_lengths = []

    def print_output_stats(self, prefix: str = ""):
        """Print output length statistics."""
        stats = self.get_output_stats()
        if stats['count'] == 0:
            print(f"{prefix}[Judge Output Stats] No data collected")
            return
        print(f"{prefix}[Judge Output Stats] count={stats['count']}, "
              f"avg_len={stats['avg_len']:.1f}, min={stats['min_len']}, max={stats['max_len']}, "
              f"total_chars={stats['total_chars']}")
