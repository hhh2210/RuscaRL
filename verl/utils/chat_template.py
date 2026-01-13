"""Utilities for loading/applying chat templates.

Why this exists:
- Some model families (e.g., Qwen2.5) ship a default system prompt via their
  built-in chat template (e.g. "You are Qwen, created by Alibaba Cloud...").
- For RL training/eval we often want to override that behavior without
  mutating the model directory on disk (e.g. copying `chat_template.jinja`).

This module adds a lightweight, runtime way to load a chat template either
from:
  1) an inline config string, or
  2) a file path (recommended for long Jinja templates).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _coerce_to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _read_template_file(path: Path) -> str:
    # Explicit utf-8 to be consistent across environments.
    return path.read_text(encoding="utf-8")


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute():
        return candidate

    # First, resolve relative to CWD (common when running from repo root).
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    # Fallback: resolve relative to the repo root (Ray workers might not keep CWD).
    repo_candidate = _REPO_ROOT / candidate
    return repo_candidate


def load_custom_chat_template(model_config: Any) -> Optional[str]:
    """Load chat template content from config.

    Supported config keys under `model_config`:
    - `custom_chat_template_path`: Path to a Jinja template file.
    - `custom_chat_template`: Inline Jinja template string.

    Additionally, `custom_chat_template` can be prefixed with `file:` to force
    file loading, e.g. `file:/abs/path/chat_template.jinja`.
    """
    if model_config is None:
        return None

    template_path = model_config.get("custom_chat_template_path", None)
    if template_path:
        resolved = _resolve_path(_coerce_to_str(template_path).strip())
        if not resolved.exists():
            raise FileNotFoundError(f"custom_chat_template_path not found: {resolved}")
        return _read_template_file(resolved)

    template = model_config.get("custom_chat_template", None)
    if not template:
        return None

    template_str = _coerce_to_str(template).strip()
    if not template_str:
        return None

    if template_str.startswith("file:"):
        resolved = _resolve_path(template_str[len("file:") :].strip())
        if not resolved.exists():
            raise FileNotFoundError(f"custom_chat_template file not found: {resolved}")
        return _read_template_file(resolved)

    return template_str

