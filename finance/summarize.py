"""AI-generated article summaries via OpenRouter's free-tier models, as a
richer alternative to an RSS feed's own teaser (see finance.news). Summaries
are cached per article (by link) so a re-fetch never re-summarizes the same
article twice -- important given OpenRouter's free-tier daily request cap
(50/day with no credits ever purchased, 1000/day once you have).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from finance.llm import RateLimited, complete as _complete
from finance.llm import has_api_key

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "news"
SUMMARY_CACHE_PATH = CACHE_DIR / "summaries.json"

_TAG_RE = re.compile(r"<[^>]+>")


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


def get_cached_summary(link: str) -> str | None:
    """The AI summary already cached for `link`, if one exists -- a pure
    cache lookup, never calls the LLM. For callers (like Loop A's claim
    cards) that want to show a summary when one's already been generated
    (e.g. from browsing the This Week tab) but shouldn't spend a fresh LLM
    call just to produce one on demand.
    """
    cache = _load_cache()
    if link in cache:
        return _normalize_text(cache[link])
    return None


def summarize_article(link: str, title: str, html_content: str) -> str | None:
    """A 2-3 sentence summary of one article, cached by link. Returns None
    (never raises) if there's no API key configured, every candidate model's
    call fails, or the free tier is exhausted -- callers should fall back to
    the feed's own teaser (finance.news's `summary` column) in that case.
    """
    cache = _load_cache()
    if link in cache:
        return _normalize_text(cache[link])

    text = _strip_html(html_content)
    if not text:
        return None

    prompt = (
        "Summarize this article in 2-3 sentences for a semiconductor/tech industry reader. "
        "Be concrete (numbers, names, what changed) -- no fluff, no 'this article discusses'.\n\n"
        f"Title: {title}\n\n{text[:6000]}"
    )

    try:
        summary = _complete(prompt, max_tokens=500, accept=lambda text: not _looks_like_leaked_reasoning(text))
    except RateLimited:
        return None  # same fallback as any other failure -- caller uses the feed's own teaser instead
    if summary:
        summary = _normalize_text(summary)
        cache[link] = summary
        _save_cache(cache)
        return summary

    return None
