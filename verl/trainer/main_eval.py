# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict
import hashlib
import importlib.util
import json
import os
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local


_REWARD_FN_CACHE: Dict[Tuple[str, str, str], Callable[..., Any]] = {}


def _load_reward_fn(file_path: str, function_name: str, reward_kwargs: Optional[dict] = None) -> Callable[..., Any]:
    if not file_path:
        raise ValueError("custom_reward_function.path is empty")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")
    if not function_name:
        raise ValueError("custom_reward_function.name is empty")

    reward_kwargs = reward_kwargs or {}
    reward_kwargs_key = json.dumps(reward_kwargs, sort_keys=True, default=str)
    cache_key = (file_path, function_name, reward_kwargs_key)
    if cache_key in _REWARD_FN_CACHE:
        return _REWARD_FN_CACHE[cache_key]

    # Use a stable, per-file module name so that Ray workers can import/deserialize
    # the callable without relying on the driver-side dynamic module name.
    digest = hashlib.md5(os.path.abspath(file_path).encode(), usedforsecurity=False).hexdigest()
    module_name = f"custom_reward_module_{digest}"

    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for '{file_path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    raw_fn = getattr(module, function_name)

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    _REWARD_FN_CACHE[cache_key] = wrapped_fn
    return wrapped_fn


@ray.remote
def process_item(reward_fn_path, reward_fn_name, reward_fn_kwargs, data_source, prompt, response_lst, reward_data):
    reward_fn = _load_reward_fn(reward_fn_path, reward_fn_name, reward_kwargs=reward_fn_kwargs)

    ground_truth = None
    if isinstance(reward_data, dict):
        ground_truth = reward_data.get("ground_truth", None)

    # Many custom reward functions (e.g. VerInstruct/HealthBench graders) expect an `extra_info`
    # dict that contains both `prompt` and `reward_model` (rubrics, ground truth, etc.).
    extra_info = None
    if prompt is not None and reward_data is not None:
        extra_info = {"prompt": prompt, "reward_model": reward_data}

    score_lst = []
    for r in response_lst:
        try:
            # Preferred: keyword arg for forward-compatibility.
            score = reward_fn(data_source, r, ground_truth, extra_info=extra_info)
        except TypeError:
            try:
                # Fallback: positional 4th arg.
                score = reward_fn(data_source, r, ground_truth, extra_info)
            except TypeError:
                # Fallback: legacy signature without extra_info.
                score = reward_fn(data_source, r, ground_truth)
        score_lst.append(score)
    return data_source, np.mean(score_lst)


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    dataset = pd.read_parquet(local_path)
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]
    prompts = dataset[config.data.prompt_key] if config.data.prompt_key in dataset.columns else None

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    reward_fn_cfg = config.get("custom_reward_function") or {}
    reward_fn_path = reward_fn_cfg.get("path")
    reward_fn_name = reward_fn_cfg.get("name")
    reward_fn_kwargs = dict(reward_fn_cfg.get("reward_kwargs", {}))
    if reward_fn_path is None:
        raise ValueError("custom_reward_function.path is required for main_eval")
    reward_fn_path = os.path.abspath(reward_fn_path)

    def _normalize_prompt(p):
        if p is None:
            return None
        # Parquet may store list-like columns as numpy arrays / pandas objects.
        if hasattr(p, "tolist"):
            p = p.tolist()
        return p

    # Create remote tasks
    remote_tasks = [
        process_item.remote(
            reward_fn_path,
            reward_fn_name,
            reward_fn_kwargs,
            data_sources[i],
            _normalize_prompt(prompts[i]) if prompts is not None else None,
            responses[i],
            reward_model_data[i],
        )
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, score = ray.get(result_id)
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)


if __name__ == "__main__":
    main()
