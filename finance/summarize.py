"""AI-generated article summaries via OpenRouter's free-tier models, as a
richer alternative to an RSS feed's own teaser (see finance.news). Summaries
are cached per article (by link) so a re-fetch never re-summarizes the same
article twice -- important given OpenRouter's free-tier daily request cap
(50/day with no credits ever purchased, 1000/day once you have).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
SUMMARY_CACHE_PATH = CACHE_DIR / "news_summaries.json"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Tried in order; a reasoning model (gpt-oss-20b) sometimes burns its whole
# token budget on hidden reasoning and returns empty content, so a plain
# instruct model goes first for reliability, with fallbacks if it's down.
MODEL_CANDIDATES = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

_TAG_RE = re.compile(r"<[^>]+>")


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


def _strip_html(html: str) -> str:
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", html)).strip()


# These free models expose hidden chain-of-thought via a separate
# "reasoning" field, but sometimes leak that same monologue into "content"
# instead of (or as well as) the actual answer -- this filters those out
# rather than trying to detect it upstream, since it's provider-routing
# dependent and not consistent per model.
_LEAKED_REASONING_MARKERS = (
    "we need to", "let's craft", "let's write", "provide a concise",
    "sentence 1", "sentence1", "we must", "i need to summarize",
)


def _looks_like_leaked_reasoning(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _LEAKED_REASONING_MARKERS)


# These models like typographic Unicode punctuation (narrow no-break space,
# non-breaking hyphen/space) between numbers and units -- fine in isolation,
# but some renderers mangle runs of them, collapsing "12 B/year cost margin"
# into "12B/yearcostmargin". Normalize to plain ASCII before it's ever cached.
_UNICODE_NORMALIZE = {
    " ": " ",  # narrow no-break space
    " ": " ",  # non-breaking space
    "‑": "-",  # non-breaking hyphen
}


def _normalize_text(text: str) -> str:
    for bad, good in _UNICODE_NORMALIZE.items():
        text = text.replace(bad, good)
    return text


def _load_cache() -> dict[str, str]:
    if SUMMARY_CACHE_PATH.exists():
        return json.loads(SUMMARY_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def has_api_key() -> bool:
    return _load_env_var("OPENROUTER_API_KEY") is not None


def summarize_article(link: str, title: str, html_content: str) -> str | None:
    """A 2-3 sentence summary of one article, cached by link. Returns None
    (never raises) if there's no API key configured, every candidate model's
    call fails, or the free tier is exhausted -- callers should fall back to
    the feed's own teaser (finance.news's `summary` column) in that case.
    """
    cache = _load_cache()
    if link in cache:
        return _normalize_text(cache[link])

    api_key = _load_env_var("OPENROUTER_API_KEY")
    if not api_key:
        return None

    text = _strip_html(html_content)
    if not text:
        return None

    prompt = (
        "Summarize this article in 2-3 sentences for a semiconductor/tech industry reader. "
        "Be concrete (numbers, names, what changed) -- no fluff, no 'this article discusses'.\n\n"
        f"Title: {title}\n\n{text[:6000]}"
    )

    for model in MODEL_CANDIDATES:
        try:
            response = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 500},
                timeout=30,
            )
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"].get("content")
        except (requests.RequestException, KeyError, IndexError):
            continue
        summary = (summary or "").strip()
        if summary and not _looks_like_leaked_reasoning(summary):
            summary = _normalize_text(summary)
            cache[link] = summary
            _save_cache(cache)
            return summary

    return None
