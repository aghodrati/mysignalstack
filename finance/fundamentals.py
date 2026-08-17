"""Fundamental second opinion on a Loop A news-driven thesis: given the same
draft claim/direction Stage B produced, judges it purely from the company's
underlying business/valuation picture -- deliberately independent of recent
price action (excludes finance.ranking's Momentum category entirely; that's
pure price-action and belongs to a separate, future, LLM-free technical
check, not this one).

Two parts, mirroring finance.newsloop's own two-stage split:

`compute_fundamental_factors` (no LLM): reuses finance.ranking's own factor
definitions (Growth/Valuation/Quality/Sentiment/Ownership) for a single
ticker, plus a fair-value anchor -- analysts' own mean target price, not a
DCF/multiple estimate invented here, since that would just be a heuristic
with no more claim to correctness than what analysts already publish. Treat
it as a rough directional anchor, not a precise target.

`fundamental_opinion` (one LLM call): given a draft thesis + those factors,
asks for a single unambiguous `fundamental_confidence` -- 0 means fundamentals
strongly argue AGAINST the thesis's claimed direction, 1 means they strongly
SUPPORT it, 0.5 means no strong bearing either way. Deliberately a single
number, not a separate "assessment" label plus a same-direction-but-oddly-
named "confidence" -- asking for both invited the model to answer two subtly
different questions (how sure am I fundamentals disagree? vs. how much do
fundamentals support this direction?) that don't always agree, which used to
let a "contradicts" verdict paired with a high self-reported number
perversely *raise* blended confidence instead of lowering it. The
supports/contradicts/neutral label shown in the UI is now derived in code
from this one score (see `_derive_assessment`), not asked from the LLM, so
there's exactly one source of truth. finance.tickerthesis blends this score
with Stage B's own confidence (soft influence only, News-weighted -- this
never vetoes a trade outright, just moves the number that already gates
CONFIDENCE_MIN/FULL) and folds risks into the thesis's accumulating
invalidation list, same mechanism as everything else there.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd

from finance.data import get_stock_info
from finance.llm import complete
from finance.loop_a_config import llm_config
from finance.ranking import FACTOR_CATEGORIES, build_factor_table

# Every category except Momentum -- see module docstring.
_FUNDAMENTAL_CATEGORIES = {c: fs for c, fs in FACTOR_CATEGORIES.items() if c != "Momentum"}

_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def compute_fundamental_factors(ticker: str) -> dict:
    """Deterministic (no LLM) fundamental snapshot for `ticker`: raw factor
    values (not percentile-ranked -- ranking is relative to a whole universe,
    which isn't meaningful for a single-ticker check) grouped by category,
    plus the analyst-consensus fair-value anchor. `finance.ranking.
    build_factor_table` normally also needs price history for Momentum --
    passed empty here since that category is excluded anyway, so it's simply
    never computed.
    """
    table = build_factor_table([ticker], pd.DataFrame(), pd.Series(dtype=float))
    row = table.loc[ticker].to_dict() if ticker in table.index else {}
    factors = {
        category: {f: row.get(f) for f in factor_names if row.get(f) is not None}
        for category, factor_names in _FUNDAMENTAL_CATEGORIES.items()
    }

    info = get_stock_info(ticker)
    target = info.get("targetMeanPrice")
    current = info.get("currentPrice") or info.get("regularMarketPrice")
    return {
        "factors": factors,
        "fair_value_estimate": target,
        "current_price": current,
        "implied_return_pct": round((target / current - 1) * 100, 1) if target and current else None,
    }


_FUNDAMENTAL_REQUIRED_FIELDS = {"fundamental_confidence", "risks", "reasoning"}

# Thresholds for deriving the supports/contradicts/neutral label shown in the UI from the single
# fundamental_confidence score -- symmetric around 0.5 (no strong bearing either way).
_SUPPORTS_THRESHOLD = 0.6
_CONTRADICTS_THRESHOLD = 0.4


def _derive_assessment(fundamental_confidence: float) -> str:
    if fundamental_confidence >= _SUPPORTS_THRESHOLD:
        return "supports"
    if fundamental_confidence <= _CONTRADICTS_THRESHOLD:
        return "contradicts"
    return "neutral"


def fundamental_opinion(
    ticker: str, thesis: str, direction: str, news_confidence: float, factors: dict, as_of: dt.date
) -> dict | None:
    """One LLM call: how well does `ticker`'s fundamental picture support the
    news-driven thesis's claimed `direction`? Never raises for a genuine
    parse failure (returns None, same "soft influence only" philosophy as the
    rest of Loop A -- a failed second opinion shouldn't block a trade the
    news thesis alone already earned). Raises RateLimited (propagated from
    finance.llm.complete) if every model is currently out of quota, same as
    every other Stage call.
    """
    prompt = (
        f"You are a fundamental equity analyst giving a second opinion on a news-driven investment "
        f"thesis for {ticker}, as of {as_of.isoformat()}. Judge purely from the company's underlying "
        f"business/valuation picture below -- deliberately independent of recent price action or the "
        f"news itself, which the thesis below already accounts for.\n\n"
        f"News-driven thesis: {json.dumps({'thesis': thesis, 'direction': direction, 'confidence': news_confidence})}\n\n"
        f"Fundamental factors for {ticker} (raw values, None = data unavailable): "
        f"{json.dumps(factors['factors'])}\n"
        f"Analyst consensus fair value estimate: {factors['fair_value_estimate']}, current price: "
        f"{factors['current_price']} (implied return to that estimate: {factors['implied_return_pct']}%) "
        f"-- treat this as a rough directional anchor, not a precise target.\n\n"
        f"How well do the company's fundamentals support this thesis's claimed direction "
        f'("{direction}")? Respond with ONLY a JSON object (no markdown fences, no commentary):\n'
        f'{{"fundamental_confidence": number between 0 and 1 -- 0 means fundamentals strongly argue '
        f'AGAINST the "{direction}" direction, 1 means fundamentals strongly SUPPORT the "{direction}" '
        f"direction, 0.5 means no strong bearing either way. This is a single support score, not your "
        f"confidence in your own reasoning -- if you think fundamentals clearly argue against this "
        f"direction, this number should be LOW, not high, "
        f'"risks": ["...", ...] (fundamental risks to this thesis not already implied by it -- '
        f"empty list if none), "
        f'"reasoning": "one sentence explaining the assessment"}}'
    )

    raw = complete(prompt, max_tokens=400, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not _FUNDAMENTAL_REQUIRED_FIELDS.issubset(data):
        return None

    data["fundamental_confidence"] = max(0.0, min(1.0, float(data["fundamental_confidence"])))
    data["risks"] = [str(r) for r in (data.get("risks") or [])]
    data["assessment"] = _derive_assessment(data["fundamental_confidence"])
    return data
