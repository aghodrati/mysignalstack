"""Generic rebalancing backtest engine.

The core idea: you supply a `weight_func(date, prices_so_far) -> {ticker: weight}`
that decides portfolio weights at each rebalance date, using only data available
up to that date (no lookahead). The engine holds those weights until the next
rebalance date and compounds daily returns. Transaction costs are optional
(`cost_bps`) and off by default — every rebalance is otherwise assumed to be a
full sell-everything/buy-everything swap with no slippage, which is fine for
testing a hypothesis's direction but not for production trading decisions.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

WeightFunc = Callable[[pd.Timestamp, pd.DataFrame], dict[str, float]]


def rebalance_dates(prices: pd.DataFrame, freq: str | int = "ME", anchor: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Rebalance dates: always the first trading day on/after a target date, so
    everything lands on real trading days. freq as a string is a calendar period
    ('ME'=monthly, 'QE'=quarterly, 'YE'=yearly). freq as an int is every N
    *calendar* days starting from `anchor` (not N trading days) — e.g. with
    anchor=2026-06-01 and freq=30, targets are June 1, July 1, July 31, ...
    `anchor` defaults to the start of `prices` if omitted.
    """
    if isinstance(freq, int):
        start = anchor if anchor is not None else prices.index[0]
        period_starts = pd.date_range(start, prices.index[-1], freq=f"{freq}D")
    else:
        period_starts = prices.resample(freq).first().index
    dates = []
    for ts in period_starts:
        candidates = prices.index[prices.index >= ts]
        if len(candidates):
            dates.append(candidates[0])
    return pd.DatetimeIndex(sorted(set(dates)))


def run_backtest(
    prices: pd.DataFrame,
    weight_func: WeightFunc,
    freq: str | int = "ME",
    start_date: pd.Timestamp | None = None,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Returns the equity curve (starting at 1.0) for a strategy defined by weight_func.

    `prices` may include history before `start_date` purely to give weight_func
    enough lookback for its first decision — the equity curve itself only
    starts accruing from `start_date` (or the first available date if omitted).

    `cost_bps`: round-trip transaction cost in basis points, charged on every
    rebalance day (since a rebalance here is always a full sell-everything/
    buy-everything swap — see module docstring). E.g. cost_bps=10 charges 0.10%
    of the portfolio on each rebalance date, split evenly across the sell and
    buy legs.
    """
    daily_returns = prices.pct_change(fill_method=None).fillna(0)
    dates = rebalance_dates(prices, freq, anchor=start_date)
    cost_per_rebalance = cost_bps / 10_000

    start_idx = 0
    if start_date is not None:
        on_or_after = prices.index[prices.index >= start_date]
        if len(on_or_after) == 0:
            raise ValueError("start_date is after the available price history")
        start_idx = prices.index.get_loc(on_or_after[0])

    equity_index = prices.index[start_idx:]
    equity = pd.Series(index=equity_index, dtype=float)
    equity.iloc[0] = 1.0

    weights: dict[str, float] = {}
    prior_dates = dates[dates <= equity_index[0]]
    if len(prior_dates):
        weights = weight_func(prior_dates[-1], prices.loc[: prior_dates[-1]])

    for i in range(1, len(equity_index)):
        today = equity_index[i]
        rebalanced = False
        if today in dates:
            weights = weight_func(today, prices.loc[:today])
            rebalanced = True
        if weights:
            day_ret = sum(w * daily_returns.loc[today].get(t, 0.0) for t, w in weights.items())
        else:
            day_ret = 0.0
        if rebalanced:
            day_ret -= cost_per_rebalance
        equity.iloc[i] = equity.iloc[i - 1] * (1 + day_ret)

    return equity


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Equity curve (starting at 1.0) for holding a single price series from day one."""
    return prices / prices.iloc[0]


def equal_weight(tickers: list[str]) -> dict[str, float]:
    w = 1 / len(tickers)
    return {t: w for t in tickers}
