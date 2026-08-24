"""An independent fundamental read on a company -- deliberately NOT scored against any particular
news-driven thesis or direction (contrast the old design, where this judged "support" for a draft
Stage B thesis). finance.thesis instead feeds the latest snapshot straight into Stage C's own
aggregation prompt as context, so the thesis LLM weighs claims and fundamentals together itself,
rather than a separate numeric blend applied after the fact. This also means a snapshot can be
generated for any tracked ticker at any time, independent of whether it has any claims or a
synthesized thesis yet.

Two parts, mirroring finance.newsloop's own two-stage split:

`compute_fundamental_factors` (no LLM): reuses finance.ranking's own factor definitions
(Growth/Valuation/Quality/Sentiment/Ownership) for a single ticker, plus a fair-value anchor --
analysts' own mean target price, not a DCF/multiple estimate invented here, since that would just
be a heuristic with no more claim to correctness than what analysts already publish. Treat it as a
rough directional anchor, not a precise target. Excludes Momentum entirely -- that's pure
price-action and belongs to a separate, future, LLM-free technical check, not this one.

`fundamental_snapshot` (one LLM call): given the current factors and (if one exists) the previous
snapshot, produces a standalone fundamental_direction/fundamental_confidence, a narrative summary,
and -- the whole point of tracking these over time -- `key_changes`: what's different since the
last check. The deltas themselves are computed deterministically (`_factor_changes`) and handed to
the model as grounding, so "key_changes" narrates real numbers rather than guessing from two raw
JSON blobs.

Persisted in this module's own per-ticker store (output/fundamentals/{ticker}.json), a plain
append-only list same convention as finance.positions/finance.thesis/finance.portfolio -- but a
genuinely separate file from finance.thesis's own per-ticker event log, not just a different
event type mixed into it. That split (not just the LLM call itself) is what makes fundamentals
truly independent: they can be inspected, cleared, or backed up without touching a ticker's
aggregated-thesis/critic history at all, and finance.thesis never has to know this file
exists to blend its result in -- it just calls `refresh_fundamentals`.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from finance.data import get_stock_info
from finance.llm import complete
from finance.loop_a_config import llm_config
from finance.ranking import FACTOR_CATEGORIES, build_factor_table
from finance import read_state

FUNDAMENTALS_DIR = Path("output/fundamentals")

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


# Below this, a factor's move vs. the previous check is treated as noise, not worth surfacing in
# the prompt or narrating as a "change" -- e.g. a forward_pe drifting from 24.1 to 24.3 shouldn't
# generate a key_changes bullet.
_MIN_NOTABLE_DELTA_PCT = 5.0


def _factor_changes(current: dict, previous: dict | None) -> list[dict]:
    """Deterministic per-factor deltas between `current` and `previous` (both shaped like
    compute_fundamental_factors's "factors" -- category -> {factor: value}), for every factor
    present in both with a move of at least _MIN_NOTABLE_DELTA_PCT. Returns [] if `previous` is
    None (first-ever check for this ticker) -- grounds fundamental_snapshot's "key_changes" in real
    numbers instead of asking the model to eyeball two raw JSON blobs.
    """
    if not previous:
        return []
    changes = []
    for category, values in current.items():
        prev_values = previous.get(category) or {}
        for factor, value in values.items():
            prev_value = prev_values.get(factor)
            if prev_value is None or value is None or prev_value == 0:
                continue
            delta_pct = round((value - prev_value) / abs(prev_value) * 100, 1)
            if abs(delta_pct) < _MIN_NOTABLE_DELTA_PCT:
                continue
            changes.append({
                "category": category, "factor": factor,
                "previous": prev_value, "current": value, "delta_pct": delta_pct,
            })
    return changes


_FUNDAMENTAL_SNAPSHOT_REQUIRED_FIELDS = {
    "fundamental_direction", "fundamental_confidence", "summary", "key_changes", "risks", "reasoning",
}


def fundamental_snapshot(ticker: str, factors: dict, previous: dict | None, as_of: dt.date) -> dict | None:
    """One LLM call: an independent read of `ticker`'s current fundamental picture, as of
    `as_of` -- NOT scored against any particular news thesis or price action, purely the factors
    below. Never raises for a genuine parse failure (returns None, same "soft influence only"
    philosophy as the rest of Loop A). Raises RateLimited (propagated from finance.llm.complete) if
    every model is currently out of quota, same as every other Stage call.

    `previous` is the last fundamental snapshot event stored for this ticker (or None for a
    first-ever check) -- its "factors"/"summary" ground this call's computed deltas and give the
    model something to reference ("summary" from last time) rather than starting cold every check.
    """
    changes = _factor_changes(factors["factors"], previous.get("factors") if previous else None)
    if previous is None:
        history_note = "No previous check on record for this ticker -- this is the first."
    else:
        history_note = f"Previous check ({previous['date']}) summary: {previous.get('summary', '')}"

    prompt = (
        f"You are a fundamental equity analyst giving an independent read on {ticker}'s underlying "
        f"business/valuation picture as of {as_of.isoformat()} -- NOT scored against any particular "
        f"news thesis or recent price action, purely the fundamentals below.\n\n"
        f"Fundamental factors (raw values, None = data unavailable): {json.dumps(factors['factors'])}\n"
        f"Analyst consensus fair value estimate: {factors['fair_value_estimate']}, current price: "
        f"{factors['current_price']} (implied return to that estimate: {factors['implied_return_pct']}%) "
        f"-- treat this as a rough directional anchor, not a precise target.\n\n"
        f"{history_note}\n"
        f"Computed changes vs. the previous check (factor: previous -> current, % move -- already "
        f"filtered to moves worth mentioning): {json.dumps(changes) if changes else 'none notable'}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary):\n"
        f'{{"fundamental_direction": "long", "short", or "neutral" (the view the fundamentals ALONE '
        f"imply, independent of any news or thesis -- \"neutral\" if genuinely mixed or unclear), "
        f'"fundamental_confidence": number between 0 and 1 (conviction in that direction; 0.5 = no '
        f'strong lean either way), '
        f'"summary": "2-3 sentence overview of the current fundamental picture", '
        f'"key_changes": ["...", ...] (plain-English narration of the computed changes above, most '
        f"significant first -- empty list if nothing notable changed or this is the first check), "
        f'"risks": ["...", ...] (fundamental risks worth flagging right now -- empty list if none), '
        f'"reasoning": "one sentence explaining the direction/confidence"}}'
    )

    raw = complete(prompt, max_tokens=500, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not _FUNDAMENTAL_SNAPSHOT_REQUIRED_FIELDS.issubset(data):
        return None
    if data["fundamental_direction"] not in ("long", "short", "neutral"):
        data["fundamental_direction"] = "neutral"

    data["fundamental_confidence"] = max(0.0, min(1.0, float(data["fundamental_confidence"])))
    data["key_changes"] = [str(c) for c in (data.get("key_changes") or [])]
    data["risks"] = [str(r) for r in (data.get("risks") or [])]
    return data


def _path(ticker: str) -> Path:
    return FUNDAMENTALS_DIR / f"{ticker}.json"


def load_fundamental_history(ticker: str) -> list[dict]:
    """Every fundamental snapshot ever stored for `ticker`, oldest first -- independent of whether
    this ticker has any thesis/claims at all. The one place callers should read snapshots from;
    app.py's Ticker page renders these as their own foldable card grid, and
    finance.thesis.aggregate_claims takes the latest one as context (see refresh_fundamentals).
    """
    path = _path(ticker)
    return json.loads(path.read_text()) if path.exists() else []


def latest_fundamental(ticker: str) -> dict | None:
    history = load_fundamental_history(ticker)
    return history[-1] if history else None


def _append(ticker: str, event: dict) -> None:
    path = _path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_fundamental_history(ticker)
    # app.py's card id for a fundamental snapshot is keyed off ticker+date (_card_id(ticker,
    # "fundamental", ev["date"])), not its position in this list -- so a second refresh landing on
    # the same date (e.g. a manual --refresh-fundamental re-run the same day as the earnings-window
    # auto-refresh) produces fresh content under an id the user may have already marked read. Mark
    # it unread again so the new content isn't silently hidden behind a stale read-mark.
    if any(e.get("date") == event.get("date") for e in history):
        read_state.mark_unread(read_state.CURRENT_USER, read_state.card_id(ticker, "fundamental", event["date"]))
    history.append(json.loads(json.dumps(event, default=str)))
    path.write_text(json.dumps(history, indent=2))


def refresh_fundamentals(ticker: str, as_of: dt.date, trigger: str | None = None) -> dict | None:
    """The one entry point every caller needs: computes `ticker`'s current factors, generates a
    fresh standalone fundamental_snapshot (grounded in deltas vs. the last stored one), persists it
    as the newest entry in this ticker's own fundamentals file, and returns it. Never touches
    finance.thesis's aggregated-thesis/critic log at all -- callers that want the result
    folded into a thesis (finance.thesis.update_ticker_thesis) pass the returned dict into
    aggregate_claims themselves; finance.newsloop's earnings-window trigger calls this directly and
    just lets the snapshot sit here for display/next Stage C run.

    `trigger` is purely an audit tag on the stored record (e.g. "earnings") -- optional, has no
    effect on behavior. Returns None (and stores nothing) if the LLM call itself produced nothing
    usable -- same "soft influence only" philosophy as the rest of Loop A. Propagates RateLimited,
    same as every other Stage call.
    """
    factors = compute_fundamental_factors(ticker)
    previous = latest_fundamental(ticker)
    snapshot = fundamental_snapshot(ticker, factors, previous, as_of)
    if snapshot is None:
        return None
    event = {
        "date": as_of.isoformat(),
        "fundamental_direction": snapshot["fundamental_direction"],
        "fundamental_confidence": snapshot["fundamental_confidence"],
        "summary": snapshot["summary"], "key_changes": snapshot["key_changes"],
        "risks": snapshot["risks"], "reasoning": snapshot["reasoning"],
        "fair_value_estimate": factors["fair_value_estimate"], "current_price": factors["current_price"],
        "implied_return_pct": factors["implied_return_pct"], "factors": factors["factors"],
    }
    if trigger is not None:
        event["trigger"] = trigger
    _append(ticker, event)
    return snapshot
