"""Vision model client.

`VisionModel` is the only seam the agent depends on, so swapping Gemini for Claude
or GPT is one new class. The Gemini implementation adds what the free tier needs:
retries with backoff on 429/503 and a fallback model chain.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Free-tier quotas are per model per day (as low as ~20 requests), so a chain of
# models stretches the budget. Stronger models first; "lite" models as last resort.
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-flash-latest", "gemini-3.7-flash",
                  "gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class ModelError(RuntimeError):
    pass


class BadOutput(ValueError):
    """The model answered, but not with JSON matching the schema."""


class VisionModel(Protocol):
    calls: int

    def generate(self, prompt: str, images: list[bytes], schema: type[T]) -> T: ...


class GeminiModel:
    def __init__(self, models: list[str] | None = None, api_key: str | None = None,
                 max_retries: int = 5, temperature: float = 0.1, request_timeout_s: int = 90):
        from google import genai
        from google.genai import types

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ModelError("Set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)")
        self._types = types
        self._client = genai.Client(api_key=key,
                                    http_options=types.HttpOptions(timeout=request_timeout_s * 1000))
        self.models = models or DEFAULT_MODELS
        self.max_retries = max_retries
        self.temperature = temperature
        self.calls = 0
        self.usage: dict[str, int] = {}      # successful calls per model
        self.exhausted: set[str] = set()     # models out of daily quota / unavailable for this key
        self.cooldown_until: dict[str, float] = {}   # overloaded (503) models are skipped for a while

    def generate(self, prompt: str, images: list[bytes], schema: type[T]) -> T:
        types = self._types
        parts = [types.Part.from_bytes(data=img, mime_type="image/png") for img in images]
        parts.append(types.Part.from_text(text=prompt))
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        last_err: Exception | None = None
        for model in self._candidates():
            for attempt in range(self.max_retries):
                try:
                    self.calls += 1
                    resp = self._client.models.generate_content(model=model, contents=parts, config=config)
                    result = parse_json(resp.text, schema)
                    self.usage[model] = self.usage.get(model, 0) + 1
                    return result
                except (BadOutput, ValidationError, json.JSONDecodeError) as e:
                    # Malformed output: retrying the same prompt usually fixes it.
                    last_err = e
                    log.warning("%s returned invalid JSON (attempt %d): %s", model, attempt + 1, e)
                except Exception as e:  # SDK raises APIError subclasses; also network errors
                    last_err = e
                    code = getattr(e, "code", None)
                    if code in (400, 401, 403, 404):
                        if code == 404:      # model not available for this key -> next model
                            self.exhausted.add(model)
                            break
                        raise ModelError(f"{model}: {e}") from e
                    if code == 429 and "PerDay" in str(e):
                        log.warning("%s daily quota exhausted; falling back", model)
                        self.exhausted.add(model)
                        break
                    if code == 503 and attempt >= 1:
                        log.warning("%s overloaded; falling back", model)
                        self.cooldown_until[model] = time.monotonic() + 60
                        break
                    delay = retry_delay(e, attempt)
                    log.warning("%s error %s; retrying in %.1fs", model, code or type(e).__name__, delay)
                    time.sleep(delay)
        raise ModelError(f"all models failed; last error: {last_err}")

    def _candidates(self) -> list[str]:
        live = [m for m in self.models if m not in self.exhausted]
        now = time.monotonic()
        cool = [m for m in live if self.cooldown_until.get(m, 0) <= now]
        # If everything is cooling down, try them anyway rather than failing.
        return cool + [m for m in live if m not in cool]


def retry_delay(err: Exception, attempt: int) -> float:
    """Honour the server's retryDelay hint on 429s, else exponential backoff with jitter."""
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", str(err))
    if m:
        return min(float(m.group(1)) + 1, 90)
    return min(2 ** attempt + random.random(), 30)


def parse_json(text: str | None, schema: type[T]) -> T:
    """JSON mode is reliable but not perfect -- tolerate code fences / leading prose."""
    if not text:
        raise BadOutput("empty response")
    try:
        return schema.model_validate_json(text)
    except ValidationError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise BadOutput(f"no JSON object in response: {text[:200]!r}")
        return schema.model_validate(json.loads(m.group(0)))
