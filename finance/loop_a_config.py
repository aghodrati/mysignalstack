"""Loop A's own config: which tickers it tracks, which news sources it
processes, which LLM provider/models it calls, and how much of an article's
text it sends -- deliberately separate
from finance.universe's QUICK_PICK_CATEGORIES/custom_tickers.json (the
sidebar's comparison-tab ticker picker, free to browse -- just price
charts), finance.news's NEWS_SOURCES (the "This Week" tab's full browsing
list), and finance.llm's own defaults (still used unchanged by
finance.summarize's article summaries). Every ticker/source/model
configured here directly drives Stage A/B/C LLM cost, since Loop A spends a
real call per (article, ticker) it decides to look at -- adding a ticker to
a chart comparison shouldn't silently start spending quota tracking its
news forever, and switching Loop A's model/provider for cost or quality
reasons shouldn't also change what powers article summaries.

A plain, hand-editable JSON file rather than anything fancier -- everything
else this app persists (portfolios, theses, claims) is JSON too, and this is
meant to be edited directly, not through a UI (yet).
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_loop_a.json"


def _load() -> dict:
    if not CONFIG_PATH.exists():
        return {"universe": {}, "news_sources": []}
    return json.loads(CONFIG_PATH.read_text())


def tracked_universe() -> dict[str, str]:
    """ticker -> display name, exactly as configured -- the set of companies
    Loop A looks for in every article and the only tickers its extracted
    claims are ever validated against.
    """
    return dict(_load().get("universe", {}))


def active_news_sources() -> list[tuple[str, str]]:
    """(name, feed_url) for every source with "active": true -- same shape
    finance.news.get_news_from_sources expects. A source left in the config
    but marked inactive stays there as a record of what's available, same
    spirit as the commented-out entries finance.news.NEWS_SOURCES used to
    carry before this file existed.
    """
    return [(s["name"], s["url"]) for s in _load().get("news_sources", []) if s.get("active", False)]


# Loop A's own default provider/models if the config's "llm" section is missing or incomplete --
# identical to finance.llm's own Groq defaults, so leaving "llm" out of the config changes nothing.
# Each model carries its own "reasoning" flag rather than a second parallel list of reasoning-model
# names -- one source of truth, so adding a reasoning model here can't silently forget to also flag
# it elsewhere (which used to just get it a 400 from the provider and get skipped, unnoticed).
_DEFAULT_LLM_CONFIG = {
    "provider": "groq",
    "models": [
        {"name": "llama-3.3-70b-versatile", "reasoning": False},
        {"name": "openai/gpt-oss-20b", "reasoning": True},
    ],
}


def llm_config() -> dict:
    """Loop A's own provider/model choice, converted into the shape
    finance.llm.complete's provider/models/reasoning_models parameters
    expect (`complete(prompt, **llm_config())`) -- the config file itself
    stores "models" as a single ordered list of {"name", "reasoning"}
    objects (see _DEFAULT_LLM_CONFIG), not two parallel lists, so there's
    nothing to keep in sync by hand. Separate from finance.llm's module-
    level defaults, which finance.summarize's article summaries still use
    unchanged -- switching Loop A's provider/models here doesn't touch
    summaries, and vice versa.
    """
    cfg = _load().get("llm", {})
    models = cfg.get("models", _DEFAULT_LLM_CONFIG["models"])
    return {
        "provider": cfg.get("provider", _DEFAULT_LLM_CONFIG["provider"]),
        "models": [m["name"] for m in models],
        "reasoning_models": {m["name"] for m in models if m.get("reasoning", False)},
    }


def max_article_chars() -> int | None:
    """Cap on how many characters of an article's text Loop A sends to the
    LLM (Stage A, Stage B, the SEC 8-K classifier) -- None (the default,
    absent from the config or explicitly null) means unlimited, send the
    full article. Set this when pointed at a smaller-context-window model
    (e.g. a free tier) that can't reliably handle a full long-form article;
    a large paid-model context window has no need for it. A single shared
    cap, not per-stage, since it's really "how much can this model handle,"
    the same answer for every stage that embeds article text.
    """
    value = _load().get("llm", {}).get("max_article_chars")
    return int(value) if value is not None else None
