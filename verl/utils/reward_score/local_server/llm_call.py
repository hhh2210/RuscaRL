import os
import re
import time

from .build_model import APIModel

try:
    from health_bench.vllm_sampler import VLLMSampler
except Exception:
    VLLMSampler = None


api_url = os.getenv("VERIF_API_URL", "http://localhost:8000/v1")
model_name = os.getenv("VERIF_MODEL_NAME", "models/IF-Verifier-7B")  # 或 "QwQ-32B-Preview"
api_model = APIModel(api_url, model_name)

_global_sampler = None


def _select_backend() -> str:
    backend = os.getenv("VERIF_LLM_BACKEND", "").strip().lower()
    if backend:
        if backend in {"vllm", "sampler", "grader"} and VLLMSampler is not None:
            return "vllm"
        if backend in {"openai", "api", "apimodel", "remote"}:
            return "openai"
    if VLLMSampler is not None and os.getenv("VLLM_BASE_URL"):
        return "vllm"
    return "openai"


def _get_sampler() -> "VLLMSampler":
    global _global_sampler
    if _global_sampler is None:
        max_tokens = int(os.getenv("VERIF_LLM_MAX_TOKENS", "4096"))
        temperature = float(os.getenv("VERIF_LLM_TEMPERATURE", "0"))
        _global_sampler = VLLMSampler(
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
            filter_think_tags=True,
        )
    return _global_sampler


def generate_chat(messages, max_tokens=1280, temperature=0.0):
    backend = _select_backend()
    if backend == "vllm":
        sampler = _get_sampler()
        response = sampler(messages).response_text
        return (response or "").strip()
    response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


def extract_chat(messages, max_tokens=128, temperature=0.0):
    backend = _select_backend()
    if backend == "vllm":
        sampler = _get_sampler()
        response = sampler(messages).response_text
        return (response or "").strip()
    response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


prompt_template = """
请判断以下文本是否满足给定的约束，仅回答是或否，不要输出其他内容。


原始指令：$I$

文本：$R$

约束：$C$

原始指令描述了基本的任务信息，给定的约束介绍了应该满足的具体的一个约束。
请判断以下文本是否满足给定的这个约束（仅仅判断是否满足给定的约束），仅回答是或否，不要输出其他内容。
"""


def llm_judge(instruction, response, constraint):
    if isinstance(response, list):
        response = "\n\n".join(response)
    prompt = prompt_template.replace("$R$", response).replace("$C$", constraint).replace("$I", instruction)
    data = [
        {"role": "user", "content": prompt}
    ]
    response = generate_chat(data)
    print(response)
    print(response[0])
    return response[0] == "是"


def llm_extract(instruction, response, specific_prompt):
    prompt_suffix = "\n请直接输出文本中的原文信息，不要改写，不要添加任何额外的信息。"

    prompt = f"文本：{response}\n抽取要求：{specific_prompt}" + prompt_suffix
    data = [
        {"role": "user", "content": prompt}
    ]
    response = extract_chat(data, max_tokens=1024)
    return response


def extract_score(text):
    if not text:
        return None
    match = re.search(r"\[\[\s*([01])\s*\]\]", text)
    if match:
        return int(match.group(1))
    # Fallback: accept a leading 0/1 token if brackets are missing.
    match = re.search(r"^\s*([01])\b", text)
    if match:
        return int(match.group(1))
    # Fallback for yes/no in Chinese if it's unambiguous.
    if "是" in text and "否" not in text:
        return 1
    if "否" in text and "是" not in text:
        return 0
    return None


def llm_score(instruction, response, checkers):
    prompt = f"""
    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    
    [指令]
    {instruction}

    [回复]
    {response}

    [约束]
    {checkers}

    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    请在回答的最开始用[[score]]格式输出你的分数。
    如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
    """
    data = [
        {"role": "user", "content": prompt}
    ]
    max_retries = int(os.getenv("VERIF_LLM_MAX_RETRIES", "2"))
    backoff_s = float(os.getenv("VERIF_LLM_RETRY_BACKOFF", "0"))

    strict_prompt = prompt + "\n\n请只输出[[0]]或[[1]]，不要输出其他内容。"
    messages = data

    for attempt in range(max_retries + 1):
        response = generate_chat(messages, max_tokens=4096)
        score = extract_score(response)
        if score is not None:
            return score
        # Retry with a stricter format instruction.
        messages = [{"role": "user", "content": strict_prompt}]
        if backoff_s > 0:
            time.sleep(backoff_s * (attempt + 1))
    return 0



if __name__ == "__main__":
    result = llm_judge(
        "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
        "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
        "Your response should contain exactly 3 sentences",
    )
    print(result)
    result_score = llm_score(
        "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
        "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
        "Your response should contain exactly 3 sentences",
    )
    print(result_score)

    
