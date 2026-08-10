"""Automated buy/sell rules for portfolio simulations.

A rule is a plain function `(as_of: date) -> pd.DataFrame` with columns
`ticker, score` (higher score = more conviction) -- e.g. diverged_pairs_candidates
below. `run_rules` turns one or more of those, each paired with a `RuleConfig`,
into actual trades against a portfolio's trade log:

1. Exits first, per rule: a position this rule opened gets sold once it's
   been held `hold_days` or it no longer appears in that rule's candidate
   list (the signal that justified holding it is gone).
2. Entries second, across all rules at once: every rule's remaining
   candidates (already-held tickers dropped, capped at that rule's `top_n`)
   are merged into one list ranked by each candidate's percentile rank
   *within its own rule* (so rules whose raw scores live on different scales
   are still comparable), then filled greedily from a single shared cash
   pool at a fixed dollar size per position -- until either the ranked list
   or the cash runs out. Nothing is force-filled or split into fractions of
   a position; a candidate that doesn't fit today just waits for tomorrow.

Re-running this on an unchanged signal is a no-op by construction: a
candidate already held under its own rule is excluded from that rule's
entry list, so "the same rule is still true tomorrow" doesn't re-buy it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from finance.correlation import divergence_now, pairwise_correlation
from finance.data import get_earnings_history, get_prices, get_upgrades_downgrades
from finance.portfolio import append_trade, current_state, execution_price, rule_positions

RuleFunction = Callable[[dt.date], pd.DataFrame]


@dataclass
class RuleConfig:
    name: str
    top_n: int  # max concurrent positions this rule may hold
    hold_days: int  # force-exit a position after this many calendar days
    unit_size: float  # $ spent per new position
    exit_on_signal_loss: bool = True  # also exit early if the ticker drops out of today's candidates
    # (True fits a *condition* the position should keep satisfying, e.g. "still diverged" --
    # False fits a one-time *event* trigger, e.g. "fell 5% today", where the entry was a single
    # day's signal and re-checking it daily would just cause an exit the very next day.)


def diverged_pairs_candidates(
    as_of: dt.date,
    universe_tickers: list[str],
    lookback_months: int = 6,
    min_correlation: float = 0.8,
    min_abs_z: float = 1.5,
) -> pd.DataFrame:
    """Buy-the-laggard candidates: among normally-correlated pairs (correlation
    >= min_correlation) whose return spread has diverged by >= min_abs_z
    standard deviations from its own mean, buy whichever leg has fallen
    behind (expecting it to catch up), scored by |z-score|. If a ticker shows
    up as the laggard in more than one pair, it keeps its highest score.
    """
    start = as_of - dt.timedelta(days=lookback_months * 31)
    end = as_of + dt.timedelta(days=1)  # yfinance end is exclusive
    prices = get_prices(universe_tickers, start=start.isoformat(), end=end.isoformat())
    prices = prices.loc[prices.index <= pd.Timestamp(as_of)].dropna(axis=1, how="all")
    if prices.shape[1] < 2:
        return pd.DataFrame(columns=["ticker", "score"])

    daily_returns = prices.pct_change(fill_method=None).dropna(how="all")
    cumulative_return = prices / prices.bfill().iloc[0] - 1

    pairs = pairwise_correlation(daily_returns)
    pairs = pairs[pairs["correlation"] >= min_correlation]
    if pairs.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    diverged = divergence_now(cumulative_return, pairs)
    if diverged.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    diverged = diverged[diverged["z_score"].abs() >= min_abs_z]
    if diverged.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    diverged["ticker"] = diverged.apply(
        lambda r: r["ticker_b"] if r["z_score"] > 0 else r["ticker_a"], axis=1
    )
    diverged["score"] = diverged["z_score"].abs()
    candidates = diverged.groupby("ticker", as_index=False)["score"].max()
    return candidates.sort_values("score", ascending=False).reset_index(drop=True)


def dip_candidates(
    as_of: dt.date,
    universe_tickers: list[str],
    drop_pct: float = 0.05,
) -> pd.DataFrame:
    """Buy-the-dip candidates: tickers whose close-to-close return on `as_of`
    itself was <= -drop_pct, scored by how far past the threshold they fell.
    A one-time event, not an ongoing condition -- pair with
    RuleConfig(exit_on_signal_loss=False) so a position isn't sold the very
    next day just because it isn't *also* dropping 5% again.
    """
    start = as_of - dt.timedelta(days=10)
    end = as_of + dt.timedelta(days=1)  # yfinance end is exclusive
    prices = get_prices(universe_tickers, start=start.isoformat(), end=end.isoformat())
    prices = prices.loc[prices.index <= pd.Timestamp(as_of)]
    if len(prices) < 2 or prices.index[-1] != pd.Timestamp(as_of):
        return pd.DataFrame(columns=["ticker", "score"])  # market wasn't open on as_of, or no history yet

    daily_return = prices.pct_change(fill_method=None).iloc[-1]
    dips = daily_return[daily_return <= -drop_pct].dropna()
    if dips.empty:
        return pd.DataFrame(columns=["ticker", "score"])
    candidates = pd.DataFrame({"ticker": dips.index, "score": dips.abs().values})
    return candidates.sort_values("score", ascending=False).reset_index(drop=True)


def earnings_streak_candidates(
    as_of: dt.date,
    universe_tickers: list[str],
    low_pct: float = 0.0,
    high_pct: float = 2.0,
    earnings_history_limit: int = 8,
) -> pd.DataFrame:
    """Mild-beat-streak candidates: a ticker qualifies only on the single
    trading day before its next scheduled earnings report, and only if its
    last two *reported* quarters (as of that report) both surprised by
    between low_pct and high_pct (a modest, consistent beat -- not a
    blowout, not a miss). Scored by closeness to the midpoint of that range
    (most "textbook mild beat" ranks highest). A one-time event tied to a
    specific date, same as dip_candidates -- pair with
    RuleConfig(exit_on_signal_loss=False).

    "Next earnings report" is the earliest earnings_date strictly after
    as_of, regardless of whether that report has since happened by the time
    this function is actually called -- point-in-time correct, so this also
    works for a historical as_of (a backtest), not just today. Using
    "whatever is still unreported right now" instead would silently miss
    real historical instances once that report is old news.

    Because this only fires on that one specific day, it only catches a
    ticker if the rule happens to be run on/after that day (and before the
    earnings report) -- there's no lookback correction for a missed run.
    """
    as_of_ts = pd.Timestamp(as_of)
    midpoint = (low_pct + high_pct) / 2
    half_range = max((high_pct - low_pct) / 2, 1e-9)
    rows = []
    for t in universe_tickers:
        earnings = get_earnings_history(t, limit=earnings_history_limit)
        if earnings.empty:
            continue
        earnings = earnings.sort_values("earnings_date").reset_index(drop=True)

        future = earnings[earnings["earnings_date"] > as_of_ts]
        if future.empty:
            continue
        next_earnings_date = future["earnings_date"].min()

        # as_of only qualifies as "the day before" if no business day falls
        # strictly between it and the earnings date (approximates the actual
        # trading calendar without needing future price data to confirm it).
        gap = pd.bdate_range(as_of_ts + pd.Timedelta(days=1), next_earnings_date - pd.Timedelta(days=1))
        if len(gap) > 0:
            continue

        reported = earnings[
            (earnings["earnings_date"] < next_earnings_date) & earnings["reported_eps"].notna()
        ].sort_values("earnings_date")
        if len(reported) < 2:
            continue
        prev1, prev2 = reported.iloc[-1], reported.iloc[-2]
        if pd.isna(prev1.surprise_pct) or pd.isna(prev2.surprise_pct):
            continue
        if not (low_pct <= prev1.surprise_pct <= high_pct and low_pct <= prev2.surprise_pct <= high_pct):
            continue

        avg_surprise = (prev1.surprise_pct + prev2.surprise_pct) / 2
        score = max(1.0 - abs(avg_surprise - midpoint) / half_range, 0.01)
        rows.append({"ticker": t, "score": score})

    if not rows:
        return pd.DataFrame(columns=["ticker", "score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def analyst_momentum_candidates(
    as_of: dt.date,
    universe_tickers: list[str],
    lookback_days: int = 7,
    min_upside_pct: float = 20.0,
) -> pd.DataFrame:
    """Candidates from recent analyst activity, same grouping as the "Recent
    analyst actions" section of the This Week tab: over the trailing
    `lookback_days`, a ticker qualifies if (upgrades + maintains) >=
    downgrades, and its average analyst price target is >= min_upside_pct
    above its last close. Scored by upside %, so the biggest implied gaps
    rank first when several qualify the same day.

    This is an ongoing condition (like diverged pairs, not a single-day
    event like a dip) -- pair with RuleConfig(exit_on_signal_loss=True) if
    you want an early exit once analysts stop supporting the thesis, or
    False for a pure fixed-hold-then-sell bet.
    """
    as_of_ts = pd.Timestamp(as_of)
    window_start = as_of_ts - pd.Timedelta(days=lookback_days)

    start = (as_of - dt.timedelta(days=10)).isoformat()
    end = (as_of + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    prices = get_prices(universe_tickers, start=start, end=end)
    prices = prices.loc[prices.index <= as_of_ts]

    rows = []
    for t in universe_tickers:
        ud = get_upgrades_downgrades(t)
        if ud.empty:
            continue
        recent = ud[(ud["grade_date"] >= window_start) & (ud["grade_date"] <= as_of_ts)]
        if recent.empty:
            continue

        up = int((recent["action"] == "up").sum())
        down = int((recent["action"] == "down").sum())
        maintain = len(recent) - up - down
        if up + maintain < down:
            continue

        priced = recent[recent["current_price_target"] > 0]
        if priced.empty:
            continue
        avg_target = priced["current_price_target"].mean()

        if t not in prices.columns or prices[t].dropna().empty:
            continue
        current_price = prices[t].dropna().iloc[-1]
        upside_pct = (avg_target / current_price - 1) * 100
        if upside_pct < min_upside_pct:
            continue

        rows.append({"ticker": t, "score": upside_pct})

    if not rows:
        return pd.DataFrame(columns=["ticker", "score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def run_rules(
    portfolio_name: str,
    rules: dict[str, tuple[RuleFunction, RuleConfig]],
    as_of: dt.date | None = None,
) -> list[dict]:
    """Runs every rule's exit pass, then a single cross-rule entry pass drawing
    from one shared cash pool. Returns a log of every trade actually made:
    [{"rule": ..., "action": "BUY"/"SELL", "ticker": ..., "shares": ..., "price": ..., "reason": ...}, ...]
    """
    as_of = as_of or dt.date.today()
    log: list[dict] = []

    # 1. Exits -- per rule, frees up cash before any new entries are decided.
    for rule_name, (rule_fn, config) in rules.items():
        candidates = rule_fn(as_of)
        candidate_tickers = set(candidates["ticker"]) if not candidates.empty else set()
        open_pos = rule_positions(portfolio_name, rule_name)
        for ticker, row in open_pos.iterrows():
            held_days = (pd.Timestamp(as_of) - row["entry_date"]).days
            expired = held_days >= config.hold_days
            dropped = config.exit_on_signal_loss and ticker not in candidate_tickers
            if not (expired or dropped):
                continue
            reason = "max hold reached" if expired else "no longer qualifies"
            try:
                price = execution_price(ticker, as_of)
            except ValueError:
                continue
            append_trade(
                portfolio_name, as_of, ticker, "SELL", row["shares"], price,
                note=f"rule exit: {reason}", rule=rule_name,
            )
            log.append(
                {"rule": rule_name, "action": "SELL", "ticker": ticker, "shares": row["shares"],
                 "price": price, "reason": reason}
            )

    # 2. Entries -- one merged, ranked pass across all rules, one shared cash pool.
    ranked_candidates = []
    for rule_name, (rule_fn, config) in rules.items():
        candidates = rule_fn(as_of)
        if candidates.empty:
            continue
        already_held = set(rule_positions(portfolio_name, rule_name).index)
        slots_remaining = config.top_n - len(already_held)
        if slots_remaining <= 0:
            continue
        fresh = candidates[~candidates["ticker"].isin(already_held)].sort_values("score", ascending=False)
        fresh = fresh.head(slots_remaining)
        n = len(fresh)
        for rank, row in enumerate(fresh.itertuples(index=False)):
            percentile = 1.0 - (rank / n) if n > 1 else 1.0
            ranked_candidates.append(
                {"rule": rule_name, "config": config, "ticker": row.ticker, "raw_score": row.score,
                 "percentile": percentile}
            )

    ranked_candidates.sort(key=lambda c: c["percentile"], reverse=True)

    cash, _ = current_state(portfolio_name)
    for candidate in ranked_candidates:
        config = candidate["config"]
        if config.unit_size > cash + 1e-6:
            continue  # doesn't fit today's remaining shared cash; wait for tomorrow
        try:
            price = execution_price(candidate["ticker"], as_of)
        except ValueError:
            continue
        shares = config.unit_size / price
        append_trade(
            portfolio_name, as_of, candidate["ticker"], "BUY", shares, price,
            note=f"rule entry: score={candidate['raw_score']:.2f}", rule=candidate["rule"],
        )
        cash -= config.unit_size
        log.append(
            {"rule": candidate["rule"], "action": "BUY", "ticker": candidate["ticker"], "shares": shares,
             "price": price, "reason": f"score={candidate['raw_score']:.2f}"}
        )

    return log
