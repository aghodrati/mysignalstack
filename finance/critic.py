"""A red-team second opinion on Stage C's own synthesis: given the exact
same claims `aggregate_claims` saw and the thesis it produced, judges
whether the synthesis is actually well-supported -- does it overreach the
evidence, is it internally consistent, are any catalysts already stale or
resolved per the claims' own dates. Deliberately closed-book: no outside
data, no web search, nothing beyond what's already in `output/claims/` --
its job is to check our own synthesis against our own evidence, not to go
find new information (that's what Stage A/B are for; see finance.newsloop).

Blended into confidence as a pure downward multiplier (never able to raise
it), same soft-influence-only philosophy as finance.fundamentals -- this
never vetoes a trade outright, just tempers the number that already gates
one. finance.tickerthesis also runs a set of free, deterministic guardrails
(source concentration, evidence thinness, staleness) independent of this
LLM call, so a thesis with an obvious structural weakness still gets
dampened even if this call is rate-limited or the LLM's own judgment misses
it.
"""

from __future__ import annotations

import datetime as dt
import json
import re

from finance.claims import ArticleClaim
from finance.llm import complete
from finance.loop_a_config import llm_config

_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)

# Clamp on the LLM's own multiplier -- one call should temper confidence, never gut it; a thesis
# that's genuinely this weak should have been caught by the deterministic guardrails or Stage C
# itself, not swing entirely on one critic call's judgment.
CRITIC_MULTIPLIER_MIN = 0.7

_CRITIC_REQUIRED_FIELDS = {"multiplier", "concerns", "reasoning"}


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def critic_review(
    ticker: str, thesis: str, direction: str, claims: list[ArticleClaim], confidence: float,
    deterministic_flags: list[str], as_of: dt.date,
) -> dict | None:
    """One LLM call: red-teams `thesis` against the same claims Stage C used
    to produce it. Never raises for a genuine parse failure (returns None,
    same fallback as finance.fundamentals.fundamental_snapshot -- callers
    should fall back to the deterministic guardrails alone). Raises
    RateLimited (propagated from finance.llm.complete) if every model is
    currently out of quota, same as every other Stage call.
    """
    claim_summaries = [
        {
            "claim": c.claim, "direction": c.direction, "confidence": c.confidence, "importance": c.importance,
            "source": c.source_title, "date": c.created.isoformat(),
        }
        for c in sorted(claims, key=lambda c: c.created)
    ]
    flags_text = "; ".join(deterministic_flags) if deterministic_flags else "none"
    prompt = (
        f"You are a skeptical second reviewer for an investment thesis on {ticker}, as of "
        f"{as_of.isoformat()}. Another analyst already synthesized this thesis from the claims below "
        f"-- your job is to red-team THAT SYNTHESIS against THOSE SAME CLAIMS, not to form your own "
        f"independent view. Use only the claims given; no outside knowledge.\n\n"
        f"Synthesized thesis ({direction}, confidence {confidence:.0%}): {thesis}\n\n"
        f"Claims it was built from (oldest first):\n{json.dumps(claim_summaries)}\n\n"
        f"Automated checks already flagged: {flags_text}. Don't repeat these -- focus on what they "
        f"wouldn't catch: does the thesis actually follow from the claims, or does it overreach them? "
        f"Is it internally consistent? Is any catalyst already resolved or stale per the claims' own "
        f"dates? If you find nothing beyond what's already flagged, say so plainly.\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary):\n"
        f'{{"multiplier": number between {CRITIC_MULTIPLIER_MIN} and 1.0 '
        f"(1.0 = no additional concerns, lower = the more this synthesis overreaches its own evidence), "
        f'"concerns": ["...", ...] (specific, evidence-grounded; empty list if none), '
        f'"reasoning": "one sentence"}}'
    )

    raw = complete(prompt, max_tokens=500, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not _CRITIC_REQUIRED_FIELDS.issubset(data):
        return None

    data["multiplier"] = max(CRITIC_MULTIPLIER_MIN, min(1.0, float(data["multiplier"])))
    data["concerns"] = [str(c) for c in data["concerns"]]
    data["reasoning"] = str(data["reasoning"])
    return data
