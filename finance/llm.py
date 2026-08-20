"""Shared chat-completion plumbing (OpenAI-compatible APIs), used by
finance.newsloop/tickerthesis/fundamentals/critic (structured thesis JSON).
Callers own their own prompt, caching, and response post-processing -- this
only owns the provider/API key, the model list, and the HTTP call.

Two providers are supported (both OpenAI-compatible chat-completion APIs,
just a different base URL/API key): Groq and OpenRouter. `complete` defaults
to Groq + MODEL_CANDIDATES below if a caller doesn't pass its own models.
Loop A's own calls (finance.newsloop, finance.tickerthesis,
finance.fundamentals, finance.critic) instead pass
provider/models/reasoning_models sourced from
finance.loop_a_config.llm_config(), so config_loop_a.json's "llm" section
controls all of Loop A from one place.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# provider name -> {chat-completion URL, .env/environment variable holding its API key}
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {"url": GROQ_URL, "api_key_env": "GROQ_API_KEY"},
    "openrouter": {"url": OPENROUTER_URL, "api_key_env": "OPENROUTER_API_KEY"},
}
DEFAULT_PROVIDER = "groq"

# Default model list, tried in order -- only used by a caller that doesn't pass its own `models`.
# llama-3.3-70b-versatile is a plain instruct model -- no hidden reasoning-token budget to manage,
# so it can't hit max_tokens with empty visible content the way a reasoning model can (see gpt-oss
# below). gpt-oss-20b is kept as a fallback for resilience if the first is down/rate-limited.
MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
]

# Groq (and OpenRouter, for the same underlying models) rejects `reasoning_effort` outright (400)
# for models that don't support it, so it's only added to the request for models that actually
# are reasoning models -- currently just the gpt-oss family. Only the default model list's own
# reasoning models; a caller passing its own `models` should pass its own `reasoning_models` too.
_REASONING_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}


def _load_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if not ENV_PATH.exists():
        return None
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == name:
            return val.strip()
    return None


def has_api_key(provider: str = DEFAULT_PROVIDER) -> bool:
    return _load_env_var(PROVIDERS[provider]["api_key_env"]) is not None


def _rate_limit_message(response: requests.Response) -> str | None:
    """The provider's own explanation of *which* limit was hit and by how
    much -- e.g. Groq's "Rate limit reached for model llama-3.3-70b-versatile
    ... on tokens per day (TPD): Limit 100000, Used 99820, Requested 1800."
    Only appears in a 429's JSON error body, never in headers -- the
    x-ratelimit-* headers only ever describe the per-minute (RPM/TPM)
    buckets, so a request that's actually blocked by a daily cap is
    otherwise indistinguishable from one that's merely per-minute-throttled
    unless this is read. OpenRouter uses the same {"error": {"message":...}}
    shape, being OpenAI-compatible, so this works for either provider.
    """
    try:
        return response.json().get("error", {}).get("message")
    except (ValueError, AttributeError):
        return None


class RateLimited(Exception):
    """Raised by `complete` when every candidate model came back 429 with
    retries exhausted -- distinct from returning None, which means a genuine
    (non-rate-limit) failure or rejection. Signals callers that this specific
    prompt is fine, the account is just out of quota right now, so it should
    be retried later rather than treated as a permanent extraction failure.
    `message`, if present, is Groq's own explanation of which specific limit
    was hit (see `_rate_limit_message`) -- per-minute limits recover in
    seconds/minutes, the daily one doesn't recover until the next UTC day, and
    the two are otherwise indistinguishable from the outside.
    """

    def __init__(self, message: str | None = None):
        super().__init__(message or "rate limited")
        self.message = message


_MAX_RETRIES_PER_MODEL = 2  # covers a transient per-minute rate limit without giving up on real errors

# Running token-usage totals across calls to `complete`, for callers that want to report cost
# (e.g. newsloop's per-article progress logging) without every call site having to plumb a
# richer return type through -- reset_usage_counter() at whatever granularity you want to
# measure (e.g. once per article), then read get_usage_counter() after.
_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def reset_usage_counter() -> None:
    _usage_totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def get_usage_counter() -> dict:
    return dict(_usage_totals)


def get_rate_limit_status(model: str | None = None, provider: str = DEFAULT_PROVIDER) -> dict | None:
    """The provider's current rate-limit standing for `model` (defaults to
    the primary candidate). Neither provider has a dedicated quota-check
    endpoint, so this is a real minimal request (max_tokens=1) purely to
    read the response, not to use the answer -- costs a handful of tokens,
    unavoidable given there's no free way to ask. Returns None if there's no
    API key for `provider` or the request itself fails outright.

    The returned dict has every x-ratelimit-* header (lowercased) the
    provider sent, if any (Groq always sends them; OpenRouter often doesn't,
    especially on free/low tiers) -- but note these only ever describe the
    per-minute (RPM/TPM) buckets, and can look perfectly healthy even while
    the account is blocked by a *daily* cap, which providers only ever
    report inside a 429's error body, never in headers. So: if the ping
    itself got a 429, the dict also has "_blocked" (True) and "_message"
    (the provider's own explanation, if it sent one) -- check "_blocked"
    before trusting the headers as "there's room". The dict always has
    "_status_code" (the raw HTTP status of the probe) so a caller can tell
    "200 with no headers" (fine, this provider just doesn't report them)
    apart from "400/401/etc with no headers" (a real problem -- bad model
    name, bad key, ...), which an empty dict alone can't distinguish.
    Returns None only if there's no API key for `provider` or the request
    itself couldn't complete at all (network failure).
    """
    provider_cfg = PROVIDERS[provider]
    api_key = _load_env_var(provider_cfg["api_key_env"])
    if not api_key:
        return None
    payload = {"model": model or MODEL_CANDIDATES[0], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}
    try:
        response = requests.post(
            provider_cfg["url"], headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=10
        )
    except requests.RequestException:
        return None
    status = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit-")}
    status["_status_code"] = response.status_code
    if response.status_code == 429:
        status["_blocked"] = True
        status["_message"] = _rate_limit_message(response)
    elif response.status_code >= 400:
        status["_message"] = _rate_limit_message(response)
    return status


def complete(
    prompt: str, max_tokens: int = 500, accept=lambda text: True,
    provider: str = DEFAULT_PROVIDER, models: list[str] | None = None,
    reasoning_models: set[str] | None = None,
) -> str | None:
    """Raw completion text for `prompt`, trying each model in `models` (default
    MODEL_CANDIDATES) against `provider` (default Groq) until one returns
    non-empty content that `accept` approves of (e.g. a caller-specific
    "doesn't look like leaked reasoning" filter). Returns None if there's no
    API key for `provider`, or every model fails for a real (non-rate-limit)
    reason, or every model's content is rejected -- callers decide what "no
    answer" means for them. Raises RateLimited instead if every model came
    back 429 with retries exhausted (the account, not this particular
    prompt, is out of quota) -- callers that want to retry later rather than
    treat that as a permanent failure should catch it.

    A caller that doesn't pass provider/models/reasoning_models gets the Groq + MODEL_CANDIDATES
    default above; Loop A's own calls (finance.newsloop/tickerthesis/fundamentals/critic) instead
    pass finance.loop_a_config.llm_config()'s choice, so config_loop_a.json's "llm" section
    controls all of them from one place.
    """
    models = models if models is not None else MODEL_CANDIDATES
    reasoning_models = reasoning_models if reasoning_models is not None else _REASONING_MODELS
    provider_cfg = PROVIDERS[provider]
    api_key = _load_env_var(provider_cfg["api_key_env"])
    if not api_key:
        return None

    all_rate_limited = True
    last_rate_limit_message = None
    for model in models:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        if model in reasoning_models:
            # Otherwise it burns its whole token budget on hidden deliberation before ever
            # writing the answer, hitting the max_tokens cap with empty visible content --
            # these are classification/extraction/summarization tasks, none need deep reasoning.
            # The two providers use different shapes for the same idea: Groq's gpt-oss family
            # takes a flat "reasoning_effort" string; OpenRouter takes a nested
            # {"reasoning": {"effort": ...}} object -- sending the wrong shape to a provider is
            # silently ignored (no error), not rejected, so a model can end up reasoning at its
            # own default effort with no visible symptom except truncated/malformed output.
            if provider == "openrouter":
                payload["reasoning"] = {"effort": "low"}
            else:
                payload["reasoning_effort"] = "low"

        model_rate_limited = False
        for attempt in range(_MAX_RETRIES_PER_MODEL):
            model_rate_limited = False
            try:
                response = requests.post(
                    provider_cfg["url"], headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=30,
                )
                if response.status_code == 429:
                    model_rate_limited = True
                    last_rate_limit_message = _rate_limit_message(response)
                    if attempt + 1 < _MAX_RETRIES_PER_MODEL:
                        retry_after = float(response.headers.get("Retry-After", 3))
                        time.sleep(min(retry_after, 10))
                        continue
                    break  # retries exhausted, still rate limited -- fall through to the next model
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"].get("content")
                usage = body.get("usage") or {}
                for key in _usage_totals:
                    _usage_totals[key] += usage.get(key, 0)
            except (requests.RequestException, KeyError, IndexError):
                all_rate_limited = False
                break  # a real error, not a rate limit -- no point retrying this model
            content = (content or "").strip()
            if content and accept(content):
                return content
            all_rate_limited = False
            break

        if not model_rate_limited:
            all_rate_limited = False

    if all_rate_limited:
        raise RateLimited(last_rate_limit_message)
    return None
