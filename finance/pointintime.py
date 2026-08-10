"""Point-in-time factor values from data sources that are already fully
historical/dated -- unlike yfinance's `.info` and `institutional_holders`
(today's snapshot only), Form 4 insider filings, analyst upgrade/downgrade
actions, and SEC 13F institutional filings all carry real historical dates,
so a factor's value "as of any past date" can be reconstructed by windowing
to data that existed by that date. No new network calls beyond what
`finance.data` and `finance.sec13f` already cache.

One casualty: yfinance's `recommendations` buy/hold/sell count trend has no
historical depth (only current + prior 3 months), so there's no honest way
to reconstruct it as of an arbitrary past date. `revisions_trend_as_of` below
is a *different* (but point-in-time-safe) proxy built from the fully-dated
upgrades/downgrades log instead: the change in upgrade momentum between two
consecutive trailing windows, rather than a change in aggregate analyst
ratings.
"""

from __future__ import annotations

import pandas as pd

from finance.data import get_insider_transactions, get_splits, get_upgrades_downgrades
from finance.sec13f import datasets_as_of, get_institutional_history


def split_adjustment_factor(ticker: str, as_of: pd.Timestamp) -> float:
    """Cumulative product of every split ratio for `ticker` that happened
    *after* `as_of`. `finance.data.get_prices`'s "Close" is always
    split-adjusted at the source using the full split history -- including
    splits that hadn't happened yet as of a given historical date -- so any
    nominal-dollar-or-share-count figure reported *as of* that date (raw EPS,
    an analyst's price target, shares outstanding) needs to be scaled by this
    factor before it's comparable to a price pulled from `get_prices` for the
    same date. Not a look-ahead violation: a split ratio is a fixed unit
    conversion, not information about future company performance.
    """
    splits = get_splits(ticker)
    if splits.empty:
        return 1.0
    future_splits = splits[splits.index > as_of]
    return float(future_splits.prod()) if not future_splits.empty else 1.0


def insider_flow_as_of(ticker: str, as_of: pd.Timestamp, window_days: int = 180) -> float | None:
    """Net insider $ (buys - sales) in the trailing `window_days` ending at
    `as_of`. Only uses filings with start_date <= as_of, so it's safe to use
    as a backtest feature.
    """
    tx = get_insider_transactions(ticker)
    if tx.empty:
        return None
    window = tx[(tx["start_date"] <= as_of) & (tx["start_date"] > as_of - pd.Timedelta(days=window_days))]
    if window.empty:
        return 0.0
    is_buy = window["transaction"].str.contains("Buy|Purchase", case=False, na=False, regex=True)
    is_sale = window["transaction"].str.contains("Sale", case=False, na=False)
    return float(window.loc[is_buy, "value"].sum() - window.loc[is_sale, "value"].sum())


def net_upgrades_as_of(ticker: str, as_of: pd.Timestamp, window_days: int = 90) -> float:
    """Upgrades minus downgrades in the trailing `window_days` ending at
    `as_of`.
    """
    ud = get_upgrades_downgrades(ticker)
    if ud.empty:
        return 0.0
    window = ud[(ud["grade_date"] <= as_of) & (ud["grade_date"] > as_of - pd.Timedelta(days=window_days))]
    if window.empty:
        return 0.0
    return float((window["action"] == "up").sum() - (window["action"] == "down").sum())


def revisions_trend_as_of(ticker: str, as_of: pd.Timestamp, window_days: int = 90) -> float | None:
    """Change in upgrade momentum: net upgrades in the trailing `window_days`
    ending at `as_of`, minus net upgrades in the `window_days` before that.
    This is a point-in-time-safe *substitute* for yfinance's buy/hold/sell
    revisions trend (which has no historical depth), not the same metric --
    it tracks the direction of analyst grade activity rather than the
    aggregate rating distribution.
    """
    ud = get_upgrades_downgrades(ticker)
    if ud.empty:
        return None
    recent = net_upgrades_as_of(ticker, as_of, window_days)
    prior_asof = as_of - pd.Timedelta(days=window_days)
    prior = net_upgrades_as_of(ticker, prior_asof, window_days)
    return recent - prior


def analyst_upside_as_of(
    ticker: str, as_of: pd.Timestamp, price_as_of: float, max_staleness_days: int = 540
) -> float | None:
    """% upside of the reconstructed analyst price-target consensus vs
    `price_as_of`: for each firm, its most recent price target as of `as_of`
    (from the upgrade/downgrade log, which records a price target with every
    action), averaged across firms with a usable target.

    Firms whose latest action is older than `max_staleness_days` (default ~18
    months) are excluded -- a firm that hasn't reissued a rating in that long
    isn't meaningfully covering the stock anymore, and an old, effectively
    abandoned target can be wildly wrong for reasons that have nothing to do
    with the stock's actual outlook (e.g. it predates a stock split).

    Each remaining target is split-adjusted individually (using the split
    ratios that happened after *that action's own date*, not `as_of`) before
    averaging, since `price_as_of` is itself split-adjusted -- see
    `split_adjustment_factor`.
    """
    if not price_as_of:
        return None
    ud = get_upgrades_downgrades(ticker)
    if ud.empty:
        return None
    known = ud[ud["grade_date"] <= as_of]
    if known.empty:
        return None
    latest_per_firm = known.sort_values("grade_date").groupby("firm").tail(1)
    fresh = latest_per_firm[latest_per_firm["grade_date"] > as_of - pd.Timedelta(days=max_staleness_days)]
    fresh = fresh[fresh["current_price_target"] > 0]
    if fresh.empty:
        return None
    adjusted_targets = [
        row["current_price_target"] / split_adjustment_factor(ticker, row["grade_date"])
        for _, row in fresh.iterrows()
    ]
    return float((sum(adjusted_targets) / len(adjusted_targets)) / price_as_of - 1) * 100


def institutional_flow_as_of(ticker: str, company_name: str, as_of: pd.Timestamp, quarters: int = 1) -> float | None:
    """Share-count %% change of the top-10 institutional holders for the most
    recent SEC 13F reporting quarter fully knowable as of `as_of` (filing
    deadline is ~45 days after quarter-end, so this naturally lags `as_of` by
    up to a full quarter plus that grace period -- it can't see a quarter's
    holdings before they were actually filed).
    """
    datasets = datasets_as_of(as_of, quarters=quarters)
    hist = get_institutional_history(ticker, company_name, quarters=quarters, datasets=datasets)
    if hist.empty:
        return None
    return float(hist.iloc[-1]["pct_change"])
