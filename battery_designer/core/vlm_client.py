"""Unified VLM (Vision Language Model) client.

Merges duplicate VLM infrastructure that previously existed in both
``vision.py`` and ``vlm_detection.py``:

* API-key resolution (``_get_api_key``)
* DashScope SDK availability check (``_check_vlm_available``)
* Rate-limited multi-modal call with exponential back-off (``vlm_call``)
* Robust JSON extraction from model output (``extract_json``)

All modules should import from here instead of maintaining private copies.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

from .errors import DesignError
from .logger import get_logger

_log = get_logger(__name__)

# ── DashScope SDK (optional dependency) ──────────────────────────────

try:
    import dashscope
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None  # type: ignore[assignment]
    dashscope = None  # type: ignore[assignment]

# ── Constants ────────────────────────────────────────────────────────

MODEL_NAME: str = os.getenv("VLM_MODEL_NAME", "qwen3.7-plus")
TEMPERATURE: float = 0.05
MAX_TOKENS: int = 2048
ENABLE_THINKING: bool = False

# ── Rate limiting & retry (single source of truth) ───────────────────

_VLM_CALL_LOCK = threading.Lock()
_VLM_LAST_CALL_TS: float = 0.0
_VLM_MIN_INTERVAL: float = float(os.getenv("VLM_MIN_INTERVAL", "0.8"))
_VLM_MAX_RETRIES: int = int(os.getenv("VLM_MAX_RETRIES", "4"))
_VLM_RETRY_BASE_DELAY: float = float(os.getenv("VLM_RETRY_BASE_DELAY", "2.0"))


# ── Public API ───────────────────────────────────────────────────────


def get_api_key() -> str:
    """Resolve ``DASHSCOPE_API_KEY`` from env (process → user → machine)."""
    key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if key:
        return key
    try:
        import winreg

        for hive, subkey in (
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ):
            try:
                with winreg.OpenKey(hive, subkey) as reg:
                    key, _ = winreg.QueryValueEx(reg, "DASHSCOPE_API_KEY")
                    if key:
                        return key.strip()
            except OSError:
                continue
    except Exception:
        pass
    return ""


def check_vlm_available() -> None:
    """Verify that the DashScope API key and SDK are available.

    Raises :class:`DesignError` (code ``"VLM_UNAVAILABLE"``) if either is missing.
    """
    if not get_api_key():
        raise DesignError("VLM_UNAVAILABLE", "DASHSCOPE_API_KEY not set")
    if MultiModalConversation is None:
        raise DesignError("VLM_UNAVAILABLE", "dashscope SDK not installed")


def vlm_call(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool = False,
    max_retries: int = _VLM_MAX_RETRIES,
    base_delay: float = _VLM_RETRY_BASE_DELAY,
):
    """Call ``MultiModalConversation.call`` with rate limiting and 429 retry.

    * Enforces a minimum interval between calls (``_VLM_MIN_INTERVAL``) to avoid 429.
    * On 429 (rate limit), retries with exponential back-off up to *max_retries* times.
    * On other non-200 status codes, retries once with a short delay.
    * Returns the response object on success, or ``None`` after all retries fail.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        # ── Rate limit: wait if needed ──
        global _VLM_LAST_CALL_TS
        with _VLM_CALL_LOCK:
            now = time.monotonic()
            wait = _VLM_LAST_CALL_TS + _VLM_MIN_INTERVAL - now
            if wait > 0:
                pass  # sleep outside the lock
            _VLM_LAST_CALL_TS = now + max(wait, 0)

        if wait > 0:
            _log.debug("VLM rate limit: waiting %.1fs before next call", wait)
            time.sleep(wait)

        try:
            response = MultiModalConversation.call(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
            )

            if response.status_code == 429:
                delay = base_delay * (2 ** attempt)
                _log.warning(
                    "VLM 429 rate limit (attempt %d/%d), retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                )
                time.sleep(delay)
                last_exc = RuntimeError(
                    f"HTTP 429 rate limit (attempt {attempt + 1})"
                )
                continue

            if response.status_code != 200:
                _log.error(
                    "VLM API error: status=%s, code=%s, message=%s",
                    response.status_code,
                    getattr(response, "code", "?"),
                    getattr(response, "message", "?"),
                )
                if attempt < max_retries:
                    time.sleep(base_delay)
                    last_exc = RuntimeError(f"HTTP {response.status_code}")
                    continue
                return None

            return response

        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                _log.warning(
                    "VLM call exception (attempt %d/%d): %s, retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                _log.error("VLM call failed after %d retries: %s", max_retries + 1, exc)
                return None

    _log.error("VLM call failed after all retries: %s", last_exc)
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """Robustly extract a JSON object from model output.

    Handles plain JSON, markdown fences (```json … ```), bare arrays,
    and regex-extracted JSON embedded in prose.

    Returns a ``dict``. If the model returned a JSON array, it is wrapped as
    ``{"items": [...]}``. Returns ``None`` if no valid JSON can be parsed.
    """
    text = text.strip()

    # Try direct parse first
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return {"items": value}
        return value
    except json.JSONDecodeError:
        pass

    # Try stripping markdown fences
    for fence in ("```json", "```"):
        if fence in text:
            stripped = text.split(fence, 1)[-1]
            stripped = stripped.rsplit("```", 1)[0].strip()
            try:
                value = json.loads(stripped)
                if isinstance(value, list):
                    return {"items": value}
                return value
            except json.JSONDecodeError:
                text = stripped
                break

    # Try regex: extract first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: extract first [...] array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return {"items": json.loads(match.group(0))}
        except json.JSONDecodeError:
            pass

    _log.error("Failed to parse JSON from VLM response: %s", text[:300])
    return None
