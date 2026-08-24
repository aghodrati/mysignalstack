"""The synthesized, tradable view of a ticker: one per ticker, global and
portfolio-independent, recomputed from every finance.claims.ArticleClaim
collected so far whenever a new one lands (finance.newsloop.update_research
calls `update_ticker_thesis`). Portfolios don't run their own version of
this; they just watch the shared snapshot and react deterministically
(finance.newsloop.run_loop_a).

One LLM call per update (plus the critic's own, see below):

`aggregate_claims` (Stage C): given every claim ever collected for this
ticker, PLUS whichever independent fundamental snapshot is currently latest
(see finance.fundamentals.load_fundamental_history/latest_fundamental -- NOT
regenerated here; fundamentals refresh on their own schedule, either
finance.newsloop's earnings-window trigger or a manual
`--refresh-fundamentals` run_loop_a.py pass, since they don't meaningfully
change often enough to justify a fresh LLM call on every single new claim),
PLUS whichever earnings-call snapshot is currently latest (see
finance.earnings_calls.latest_earnings_call -- also NOT regenerated here, same
"refreshes on its own schedule" reasoning, and now including that call's own
key_qa_moments, not just its summary/guidance -- the Q&A highlights are where
hedging/deflection signal actually lives, per finance.earnings_calls' own
docstring), PLUS the last *reported* quarter's real numbers (finance.data.
get_earnings_history's own reported_eps/eps_estimate/surprise_pct -- real,
never LLM-guessed, same "only ever real numbers" discipline finance.
earnings_calls.refresh_earnings_preview already uses), synthesizes ONE
current thesis/direction/confidence -- weighing each claim by its own
confidence/importance rather than treating the list flatly (deliberately
soft, not a hard numeric rule), and weighing the fundamental/earnings-call/
reported-earnings pictures in the same prompt rather than a separate numeric
blend applied after the fact. The fundamental/earnings-call snapshots live
entirely in their own per-ticker stores (output/fundamentals/{ticker}.json,
output/earnings/{ticker}.json), genuinely separate from this module's own
event log. None of the three carries a "confidence" that gets arithmetically
combined with anything -- Stage C's own "aggregated" event here is the sole
source of truth for confidence.

Third and last: a critic pass on Stage C's own synthesis -- deterministic
guardrails (`_deterministic_critic_flags`: source concentration, evidence
thinness, staleness) plus one LLM red-team call (`finance.critic`), combined
into a pure downward multiplier on the already-blended confidence (never
able to raise it). Recorded as its own "critic" event; the deterministic
half always applies even if the LLM call is rate-limited or misses the
issue.

Stored as an append-only per-ticker event log, same convention as
finance.positions and finance.portfolio -- but simpler, since each update is a
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

import pandas as pd

from finance.claims import ArticleClaim, Direction, HORIZON_MAX_DAYS, HORIZON_MIN_DAYS, load_claims
from finance.critic import critic_review
from finance.data import get_earnings_history
from finance.earnings_calls import latest_earnings_call
from finance.fundamentals import latest_fundamental
from finance.llm import complete
from finance.loop_a_config import llm_config

THESES_DIR = Path("output/theses")

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


def aggregate_claims(
    ticker: str, claims: list[ArticleClaim], as_of: dt.date, fundamental: dict | None = None,
    earnings_call: dict | None = None, reported_earnings: dict | None = None,
) -> dict | None:
    """Stage C, one LLM call: synthesizes every claim ever collected for
    `ticker` into one current thesis/direction/confidence, weighing in
    `fundamental` (the latest finance.fundamentals.fundamental_snapshot, if
    one exists), `earnings_call` (the latest
    finance.earnings_calls.latest_earnings_call, if one exists), and
    `reported_earnings` (the last actually-reported quarter's real
    eps_estimate/reported_eps/surprise_pct -- see _reported_earnings_dict,
    None if yfinance has nothing reported yet) alongside the claims rather
    than blending a number in afterward -- this call's own "confidence" is
    the final say, no further arithmetic combination happens once it
    returns. Never raises for a genuine parse failure (returns None).
    Raises RateLimited (propagated from finance.llm.complete) if every model
    is currently out of quota, same as every other Stage call.
    """
    claim_summaries = [
        {
            "claim": c.claim, "direction": c.direction, "confidence": c.confidence, "importance": c.importance,
            "trade_worthy": c.trade_worthy, "source": c.source_title, "date": c.created.isoformat(),
            "expected_horizon_days": c.expected_horizon_days,
        }
        for c in sorted(claims, key=lambda c: c.created)
    ]
    if fundamental is None:
        fundamental_block = "No fundamental picture available for this ticker.\n\n"
    else:
        fundamental_block = (
            f"Independent fundamental picture (from the company's underlying business/valuation "
            f"data, NOT derived from the news claims above -- a corroborating or contradicting "
            f"check, weigh it alongside the claims but don't let it alone override a strong, "
            f"well-supported news consensus, and vice versa):\n"
            f"{json.dumps({k: fundamental[k] for k in ('fundamental_direction', 'fundamental_confidence', 'summary', 'key_changes', 'risks')})}"
            f"\n\n"
        )
    if earnings_call is None:
        earnings_call_block = "No earnings-call picture available for this ticker.\n\n"
    else:
        earnings_call_block = (
            f"Latest earnings-call picture (from the company's own most recent earnings call, NOT "
            f"derived from the news claims above -- another independent corroborating or "
            f"contradicting check, same weighing rule as the fundamental picture; key_qa_moments is "
            f"unscripted analyst Q&A, weight it for tone/hedging over the scripted guidance fields):\n"
            f"{json.dumps({k: earnings_call[k] for k in ('earnings_direction', 'earnings_confidence', 'summary', 'guidance_summary', 'guidance_change', 'management_tone', 'key_qa_moments', 'risks')})}"
            f"\n\n"
        )
    if reported_earnings is None:
        reported_earnings_block = "No reported quarterly earnings on record for this ticker.\n\n"
    else:
        reported_earnings_block = (
            f"Last reported quarter's real numbers (yfinance, NOT an LLM estimate -- ground any "
            f"beat/miss framing in these, don't restate a claim's own characterization of the "
            f"quarter as if it were the number itself):\n"
            f"{json.dumps(reported_earnings)}\n\n"
        )
    prompt = (
        f"You are an investment research assistant synthesizing a single current thesis for {ticker} "
        f"as of {as_of.isoformat()}, from every distinct claim collected about it so far plus an "
        f"independent fundamental picture, the latest earnings-call picture, and the last reported "
        f"quarter's real numbers. Reason only from "
        f"what's given below -- no outside knowledge of what happened after {as_of.isoformat()}.\n\n"
        f"Claims (oldest first, each with its own confidence, importance, and expected horizon -- weigh "
        f"accordingly, don't treat them as equally significant, and don't let a single new or "
        f"low-importance claim override an established consensus from many prior claims):\n"
        f"{json.dumps(claim_summaries)}\n\n"
        f"{fundamental_block}"
        f"{earnings_call_block}"
        f"{reported_earnings_block}"
        f"Synthesize ONE current thesis, direction, and confidence for {ticker} that best reflects this "
        f"whole body of evidence. If the claims genuinely conflict, are too thin to support a clear "
        f"call, or the fundamental/earnings-call pictures strongly contradict the news-driven "
        f"direction, reflect that with a low confidence rather than picking a direction arbitrarily.\n\n"
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
        f"bullish claims/outcomes that would prove it wrong -- fold in any fundamental/earnings-call "
        f"risks above that would count as invalidation too), "
        f'"reasoning": "one sentence explaining the synthesis, including how the fundamental, '
        f'earnings-call, and reported-earnings pictures factored in if given"}}'
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


def _reported_earnings_dict(ticker: str) -> dict | None:
    """The last actually-*reported* quarter's real numbers for `ticker` (eps_estimate,
    reported_eps, surprise_pct, earnings_date), or None if yfinance has nothing reported yet --
    same finance.data.get_earnings_history source, same dropna-on-reported-fields convention,
    app.py's _latest_reported_earnings already uses for the "Just Reported" card. A live
    yfinance-backed read, not persisted anywhere of its own -- cheap/deterministic, no LLM
    involved, so nothing to cache beyond finance.data's own.
    """
    history = get_earnings_history(ticker).dropna(subset=["reported_eps", "surprise_pct"])
    if history.empty:
        return None
    row = history.sort_values("earnings_date").iloc[-1]
    return {
        "earnings_date": row["earnings_date"].date().isoformat(),
        "eps_estimate": None if pd.isna(row["eps_estimate"]) else float(row["eps_estimate"]),
        "reported_eps": float(row["reported_eps"]),
        "surprise_pct": float(row["surprise_pct"]),
    }


def update_ticker_thesis(ticker: str, as_of: dt.date) -> TickerThesis | None:
    """Recomputes `ticker`'s current synthesized view from every claim
    collected so far, plus whichever independent fundamental snapshot
    (finance.fundamentals.latest_fundamental) and earnings-call snapshot
    (finance.earnings_calls.latest_earnings_call) are currently latest, plus
    the last reported quarter's real numbers (_reported_earnings_dict) --
    all three pure reads, NOT regenerated here; the fundamental/earnings-call
    snapshots don't meaningfully change often enough to justify a fresh LLM
    call on every single new claim, so they refresh on their own separate
    schedules -- fed into Stage C together, then a critic pass (deterministic
    guardrails + one LLM red-team call, see finance.critic) on the result.
    Appends the "aggregated"/"critic" events to the permanent, global
    (portfolio-independent) log. Returns the updated TickerThesis, or None if
    there are no claims yet or the aggregator produced nothing usable --
    callers should leave whatever the previous snapshot was in place rather
    than losing it. Propagates RateLimited from any of the LLM calls, same as
    every other Stage.
    """
    claims = load_claims(ticker)
    if not claims:
        return None
    if len(claims) > MAX_CLAIMS_FOR_AGGREGATION:
        claims = sorted(claims, key=lambda c: c.created, reverse=True)[:MAX_CLAIMS_FOR_AGGREGATION]

    fundamental = latest_fundamental(ticker)
    earnings_call = latest_earnings_call(ticker)
    reported_earnings = _reported_earnings_dict(ticker)

    result = aggregate_claims(
        ticker, claims, as_of, fundamental=fundamental, earnings_call=earnings_call,
        reported_earnings=reported_earnings,
    )
    if result is None:
        return None

    pre_critic_confidence = result["confidence"]
    invalidation = list(result["invalidation"])

    deterministic_multiplier, deterministic_flags = _deterministic_critic_flags(
        claims, pre_critic_confidence, result["expected_horizon_days"], as_of
    )
    critic = critic_review(
        ticker, result["thesis"], result["direction"], claims, pre_critic_confidence, deterministic_flags, as_of
    )
    llm_multiplier = critic["multiplier"] if critic is not None else None
    critic_multiplier = deterministic_multiplier * (llm_multiplier if llm_multiplier is not None else 1.0)
    final_confidence = max(0.0, min(1.0, pre_critic_confidence * critic_multiplier))

    _append(
        ticker,
        {
            "event": "aggregated", "date": as_of.isoformat(), "thesis": result["thesis"],
            "direction": result["direction"], "confidence": final_confidence,
            "fundamental_direction": fundamental["fundamental_direction"] if fundamental else None,
            "fundamental_confidence": fundamental["fundamental_confidence"] if fundamental else None,
            "earnings_direction": earnings_call["earnings_direction"] if earnings_call else None,
            "earnings_confidence": earnings_call["earnings_confidence"] if earnings_call else None,
            "pre_critic_confidence": pre_critic_confidence, "critic_multiplier": critic_multiplier,
            "expected_return_pct": result["expected_return_pct"],
            "expected_horizon_days": result["expected_horizon_days"], "catalysts": result["catalysts"],
            "invalidation": invalidation, "reasoning": result["reasoning"], "claims_considered": len(claims),
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
            "confidence_before": pre_critic_confidence, "confidence_after": final_confidence,
        },
    )

    return load_ticker_thesis(ticker)
