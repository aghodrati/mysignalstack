"""The synthesized, tradable view of a ticker: one per ticker, global and
portfolio-independent, recomputed from every finance.claims.ArticleClaim
collected so far whenever a new one lands (finance.newsloop.update_research
calls `update_ticker_thesis`). Portfolios don't run their own version of
this; they just watch the shared snapshot and react deterministically
(finance.newsloop.run_loop_a).

Two LLM calls per update:

`aggregate_claims` (Stage C): given every claim ever collected for this
ticker, synthesizes ONE current thesis/direction/confidence -- weighing each
claim by its own confidence/importance rather than treating the list flatly,
which is the (deliberately soft, not a hard numeric rule) protection against
any single new or low-importance claim overriding an established consensus
from many prior ones.

`finance.fundamentals.fundamental_opinion`: the same fundamental second
opinion used before, just now running once globally per ticker-thesis update
instead of once per portfolio -- blended into confidence (News-weighted) and
folded into invalidation, same as before, recorded as its own "fundamental"
event for audit but never the source of truth for confidence (the preceding
"aggregated" event already carries the final blended number).

Third and last: a critic pass on Stage C's own synthesis -- deterministic
guardrails (`_deterministic_critic_flags`: source concentration, evidence
thinness, staleness) plus one LLM red-team call (`finance.critic`), combined
into a pure downward multiplier on the already-blended confidence (never
able to raise it). Recorded as its own "critic" event; the deterministic
half always applies even if the LLM call is rate-limited or misses the
issue.

Stored as an append-only per-ticker event log, same convention as
finance.thesis and finance.portfolio -- but simpler, since each update is a
full recompute (not an incremental merge): "aggregated" events are complete
snapshots, the latest one is the current state, and every one ever appended
is kept as history for display.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from finance.claims import ArticleClaim, Direction, HORIZON_MAX_DAYS, HORIZON_MIN_DAYS, load_claims
from finance.critic import critic_review
from finance.fundamentals import compute_fundamental_factors, fundamental_opinion
from finance.llm import complete
from finance.loop_a_config import llm_config

THESES_DIR = Path("output/theses")

# Blend weights for the fundamental second opinion against the aggregator's own ("News") confidence
# -- News-weighted since it's the primary, always-present signal; fundamentals are a corroborating/
# contradicting check on top, not a replacement. Soft influence only, same as before.
NEWS_CONFIDENCE_WEIGHT = 0.6
FUNDAMENTAL_CONFIDENCE_WEIGHT = 0.4

# Caps how many of a ticker's claims ever reach one aggregate_claims prompt -- claims accumulate
# forever (finance.claims.append_claims is pure-append, nothing prunes it), so without a cap a
# heavily-covered ticker's prompt would keep growing every day the tool runs, eventually getting
# slow/expensive. Keeps the most recent N; older claims still exist on disk (finance.claims never
# deletes anything), they just stop feeding new syntheses. Real cost is small (~100 tokens/claim,
# measured), so this is set generously rather than as a tight cost control -- 200 is comfortably
# ahead of where even a heavily-covered ticker lands for a long while, since it's Stage C's *input*
# (its response, not this, is what's max_tokens-capped -- more claims add context, not truncation risk).
MAX_CLAIMS_FOR_AGGREGATION = 200

# Critic guardrails (see finance.critic for the LLM half): deterministic dampeners on the
# just-blended confidence, computed in code so they apply even if the LLM critic call itself is
# rate-limited or its own judgment misses the issue -- a thesis built on one source, thin
# evidence, or stale claims shouldn't need quota available to catch it. Each condition that
# triggers multiplies confidence down a bit; multiple conditions compound rather than picking
# the worst one, since e.g. "one source" and "stale" together are worse than either alone.
SOURCE_CONCENTRATION_PENALTY = 0.9
EVIDENCE_THINNESS_PENALTY = 0.85
STALENESS_PENALTY = 0.85
EVIDENCE_THINNESS_CONFIDENCE_THRESHOLD = 0.75
EVIDENCE_THINNESS_CLAIMS_THRESHOLD = 3

_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class TickerThesis:
    ticker: str
    updated: dt.date
    thesis: str
    direction: Direction | None
    confidence: float
    expected_return_pct: float
    expected_horizon_days: int
    catalysts: list[str] = field(default_factory=list)
    invalidation: list[str] = field(default_factory=list)
    claims_considered: int = 0
    history: list[dict] = field(default_factory=list)  # every "aggregated"/"fundamental" event ever appended


def _path(ticker: str) -> Path:
    return THESES_DIR / f"{ticker}.json"


def _load_events(ticker: str) -> list[dict]:
    path = _path(ticker)
    return json.loads(path.read_text()) if path.exists() else []


def _append(ticker: str, event: dict) -> None:
    path = _path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = _load_events(ticker)
    events.append(json.loads(json.dumps(event, default=str)))
    path.write_text(json.dumps(events, indent=2))


def load_ticker_thesis(ticker: str) -> TickerThesis | None:
    events = _load_events(ticker)
    latest = next((e for e in reversed(events) if e["event"] == "aggregated"), None)
    if latest is None:
        return None
    return TickerThesis(
        ticker=ticker, updated=dt.date.fromisoformat(latest["date"]),
        thesis=latest["thesis"], direction=latest["direction"], confidence=float(latest["confidence"]),
        expected_return_pct=float(latest["expected_return_pct"]),
        expected_horizon_days=int(latest["expected_horizon_days"]), catalysts=list(latest["catalysts"]),
        invalidation=list(latest["invalidation"]), claims_considered=int(latest.get("claims_considered", 0)),
        history=list(events),
    )


def list_tickers_with_thesis() -> list[str]:
    """Every ticker with at least one TickerThesis snapshot -- the universe a
    portfolio should scan each run, regardless of whether *this* portfolio's
    own call to update_research was the one that produced them (they may have
    been produced by a different portfolio's run earlier the same day, since
    research is shared -- see finance.newsloop.update_research).
    """
    if not THESES_DIR.exists():
        return []
    return sorted(p.stem for p in THESES_DIR.glob("*.json"))


_AGGREGATE_REQUIRED_FIELDS = {
    "thesis", "direction", "confidence", "expected_return_pct", "expected_horizon_days",
    "catalysts", "invalidation", "reasoning",
}


def aggregate_claims(ticker: str, claims: list[ArticleClaim], as_of: dt.date) -> dict | None:
    """Stage C, one LLM call: synthesizes every claim ever collected for
    `ticker` into one current thesis/direction/confidence. Never raises for a
    genuine parse failure (returns None). Raises RateLimited (propagated from
    finance.llm.complete) if every model is currently out of quota, same as
    every other Stage call.
    """
    claim_summaries = [
        {
            "claim": c.claim, "direction": c.direction, "confidence": c.confidence, "importance": c.importance,
            "trade_worthy": c.trade_worthy, "source": c.source_title, "date": c.created.isoformat(),
            "expected_horizon_days": c.expected_horizon_days,
        }
        for c in sorted(claims, key=lambda c: c.created)
    ]
    prompt = (
        f"You are an investment research assistant synthesizing a single current thesis for {ticker} "
        f"as of {as_of.isoformat()}, from every distinct claim collected about it so far. Reason only "
        f"from the claims given -- no outside knowledge of what happened after {as_of.isoformat()}.\n\n"
        f"Claims (oldest first, each with its own confidence, importance, and expected horizon -- weigh "
        f"accordingly, don't treat them as equally significant, and don't let a single new or "
        f"low-importance claim override an established consensus from many prior claims):\n"
        f"{json.dumps(claim_summaries)}\n\n"
        f"Synthesize ONE current thesis, direction, and confidence for {ticker} that best reflects this "
        f"whole body of evidence. If the claims genuinely conflict or are too thin to support a clear "
        f"call, reflect that with a low confidence rather than picking a direction arbitrarily.\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary):\n"
        f'{{"thesis": "one sentence: the current synthesized view", '
        f'"direction": "long" or "short", '
        f'"confidence": number between 0 and 1, '
        f'"expected_return_pct": number, '
        f'"expected_horizon_days": integer between {HORIZON_MIN_DAYS} and {HORIZON_MAX_DAYS}, '
        f'"catalysts": ["...", ...] (drawn from the claims above -- events/developments that would '
        f'CONFIRM or DRIVE the direction you chose; for a "short" direction these must themselves be '
        f"bearish, not a restatement of the bullish case), "
        f'"invalidation": ["...", ...] (the opposite standard -- events/developments that would '
        f'CONTRADICT or DISPROVE the direction you chose; for a "short" direction these are the '
        f"bullish claims/outcomes that would prove it wrong), "
        f'"reasoning": "one sentence explaining the synthesis"}}'
    )

    raw = complete(prompt, max_tokens=700, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not _AGGREGATE_REQUIRED_FIELDS.issubset(data):
        return None
    if data["direction"] not in ("long", "short"):
        return None

    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    data["expected_horizon_days"] = int(
        max(HORIZON_MIN_DAYS, min(HORIZON_MAX_DAYS, data["expected_horizon_days"]))
    )
    data["catalysts"] = [str(c) for c in data["catalysts"]]
    data["invalidation"] = [str(c) for c in data["invalidation"]]
    return data


def _deterministic_critic_flags(
    claims: list[ArticleClaim], confidence: float, expected_horizon_days: int, as_of: dt.date
) -> tuple[float, list[str]]:
    """The free half of the critic: hard-rule checks on the evidence behind
    a just-synthesized thesis, independent of any LLM call. Returns a
    confidence multiplier (1.0 = no concerns) and the human-readable flags
    that produced it.
    """
    multiplier = 1.0
    flags: list[str] = []

    distinct_sources = len({c.source_link for c in claims})
    if distinct_sources == 1:
        multiplier *= SOURCE_CONCENTRATION_PENALTY
        flags.append("all supporting claims come from a single source article")

    if confidence >= EVIDENCE_THINNESS_CONFIDENCE_THRESHOLD and len(claims) < EVIDENCE_THINNESS_CLAIMS_THRESHOLD:
        multiplier *= EVIDENCE_THINNESS_PENALTY
        flags.append(f"confidence is high ({confidence:.0%}) but only {len(claims)} claim(s) support it")

    days_since_newest_claim = (as_of - max(c.created for c in claims)).days
    if days_since_newest_claim > expected_horizon_days:
        multiplier *= STALENESS_PENALTY
        flags.append(
            f"newest supporting claim is {days_since_newest_claim}d old, older than the "
            f"thesis's own {expected_horizon_days}d horizon"
        )

    return multiplier, flags


def update_ticker_thesis(ticker: str, as_of: dt.date) -> TickerThesis | None:
    """Recomputes `ticker`'s current synthesized view from every claim
    collected so far, plus a fresh fundamental second opinion and a critic
    pass (deterministic guardrails + one LLM red-team call, see
    finance.critic), and appends all of it to the permanent, global
    (portfolio-independent) log. Returns the updated TickerThesis, or None
    if there are no claims yet or the aggregator produced nothing usable --
    callers should leave whatever the previous snapshot was in place rather
    than losing it. Propagates RateLimited from any of the LLM calls, same
    as every other Stage.
    """
    claims = load_claims(ticker)
    if not claims:
        return None
    if len(claims) > MAX_CLAIMS_FOR_AGGREGATION:
        claims = sorted(claims, key=lambda c: c.created, reverse=True)[:MAX_CLAIMS_FOR_AGGREGATION]
    result = aggregate_claims(ticker, claims, as_of)
    if result is None:
        return None

    news_confidence = result["confidence"]
    invalidation = list(result["invalidation"])
    factors = compute_fundamental_factors(ticker)
    fundamental = fundamental_opinion(ticker, result["thesis"], result["direction"], news_confidence, factors, as_of)
    fundamental_confidence = fundamental["fundamental_confidence"] if fundamental is not None else None
    if fundamental is not None:
        blended_confidence = (
            news_confidence * NEWS_CONFIDENCE_WEIGHT
            + fundamental_confidence * FUNDAMENTAL_CONFIDENCE_WEIGHT
        )
        invalidation = list(dict.fromkeys(invalidation + fundamental["risks"]))
    else:
        blended_confidence = news_confidence

    deterministic_multiplier, deterministic_flags = _deterministic_critic_flags(
        claims, blended_confidence, result["expected_horizon_days"], as_of
    )
    critic = critic_review(
        ticker, result["thesis"], result["direction"], claims, blended_confidence, deterministic_flags, as_of
    )
    llm_multiplier = critic["multiplier"] if critic is not None else None
    critic_multiplier = deterministic_multiplier * (llm_multiplier if llm_multiplier is not None else 1.0)
    final_confidence = max(0.0, min(1.0, blended_confidence * critic_multiplier))

    _append(
        ticker,
        {
            "event": "aggregated", "date": as_of.isoformat(), "thesis": result["thesis"],
            "direction": result["direction"], "confidence": final_confidence,
            "news_confidence": news_confidence, "fundamental_confidence": fundamental_confidence,
            "pre_critic_confidence": blended_confidence, "critic_multiplier": critic_multiplier,
            "expected_return_pct": result["expected_return_pct"],
            "expected_horizon_days": result["expected_horizon_days"], "catalysts": result["catalysts"],
            "invalidation": invalidation, "reasoning": result["reasoning"], "claims_considered": len(claims),
        },
    )
    if fundamental is not None:
        _append(
            ticker,
            {
                "event": "fundamental", "date": as_of.isoformat(), "assessment": fundamental["assessment"],
                "thesis": result["thesis"], "direction": result["direction"],
                "news_confidence": news_confidence, "fundamental_confidence": fundamental_confidence,
                "blended_confidence": blended_confidence, "fair_value_estimate": factors["fair_value_estimate"],
                "current_price": factors["current_price"], "implied_return_pct": factors["implied_return_pct"],
                "factors": factors["factors"], "risks": fundamental["risks"],
                "reasoning": fundamental["reasoning"],
            },
        )
    _append(
        ticker,
        {
            "event": "critic", "date": as_of.isoformat(),
            "deterministic_flags": deterministic_flags, "deterministic_multiplier": deterministic_multiplier,
            "llm_concerns": critic["concerns"] if critic is not None else [],
            "llm_multiplier": llm_multiplier,
            "llm_reasoning": critic["reasoning"] if critic is not None else None,
            "final_multiplier": critic_multiplier,
            "confidence_before": blended_confidence, "confidence_after": final_confidence,
        },
    )

    return load_ticker_thesis(ticker)


def refresh_fundamentals(ticker: str, as_of: dt.date) -> TickerThesis | None:
    """Re-runs only the fundamental second opinion for `ticker`'s current
    thesis and re-blends it into confidence, without a new Stage C
    aggregation -- for callers that want fundamentals refreshed independent
    of whether any new claim has landed (namely finance.newsloop's
    earnings-window trigger: the underlying numbers move around an earnings
    report whether or not a qualifying news article does). Returns None if
    there's no existing thesis to refresh, or the fundamental call itself
    produced nothing usable. Propagates RateLimited, same as every other
    Stage call.
    """
    events = _load_events(ticker)
    latest = next((e for e in reversed(events) if e["event"] == "aggregated"), None)
    if latest is None:
        return None

    news_confidence = float(latest["news_confidence"]) if latest.get("news_confidence") is not None else float(latest["confidence"])
    thesis, direction = latest["thesis"], latest["direction"]
    factors = compute_fundamental_factors(ticker)
    fundamental = fundamental_opinion(ticker, thesis, direction, news_confidence, factors, as_of)
    if fundamental is None:
        return None

    fundamental_confidence = fundamental["fundamental_confidence"]
    blended_confidence = (
        news_confidence * NEWS_CONFIDENCE_WEIGHT + fundamental_confidence * FUNDAMENTAL_CONFIDENCE_WEIGHT
    )
    invalidation = list(dict.fromkeys(list(latest["invalidation"]) + fundamental["risks"]))
    # Carries forward the critic's last dampening rather than dropping it -- an earnings-only
    # refresh doesn't re-run the critic (the underlying claims/thesis text haven't changed), but
    # whatever structural concern it flagged last time (thin evidence, single source, staleness)
    # still applies to the same thesis text now getting a fresh confidence number.
    critic_multiplier = float(latest.get("critic_multiplier", 1.0))
    final_confidence = max(0.0, min(1.0, blended_confidence * critic_multiplier))

    _append(
        ticker,
        {
            "event": "aggregated", "date": as_of.isoformat(), "thesis": thesis, "direction": direction,
            "confidence": final_confidence, "news_confidence": news_confidence,
            "fundamental_confidence": fundamental_confidence,
            "pre_critic_confidence": blended_confidence, "critic_multiplier": critic_multiplier,
            "expected_return_pct": latest["expected_return_pct"],
            "expected_horizon_days": latest["expected_horizon_days"], "catalysts": latest["catalysts"],
            "invalidation": invalidation, "reasoning": latest["reasoning"],
            "claims_considered": latest.get("claims_considered", 0), "trigger": "earnings",
        },
    )
    _append(
        ticker,
        {
            "event": "fundamental", "date": as_of.isoformat(), "assessment": fundamental["assessment"],
            "thesis": thesis, "direction": direction, "news_confidence": news_confidence,
            "fundamental_confidence": fundamental_confidence, "blended_confidence": blended_confidence,
            "fair_value_estimate": factors["fair_value_estimate"], "current_price": factors["current_price"],
            "implied_return_pct": factors["implied_return_pct"], "factors": factors["factors"],
            "risks": fundamental["risks"], "reasoning": fundamental["reasoning"], "trigger": "earnings",
        },
    )
    return load_ticker_thesis(ticker)
