"""Momentum strategy: rank a universe by trailing return, hold the top (or bottom)
N equal-weighted. Also provides two benchmark strategies over the same eligible
universe and rebalance schedule, for a fair comparison: equal-weighting everything,
and randomly picking N stocks.
"""

from __future__ import annotations

import random
from typing import Literal

import pandas as pd

Direction = Literal["best", "worst"]


def _trailing_window(history: pd.DataFrame, lookback_days: int) -> pd.DataFrame | None:
    """The slice of `history` covering the trailing `lookback_days` *calendar* days,
    or None if `history` doesn't yet go back far enough to fill that window.
    """
    if history.empty:
        return None
    window_start = history.index[-1] - pd.Timedelta(days=lookback_days)
    if history.index[0] > window_start:
        return None
    return history.loc[history.index >= window_start]


def eligible_universe(history: pd.DataFrame, lookback_days: int) -> list[str]:
    """Tickers with a full trailing `lookback_days` of history as of `history`'s last
    date — the same eligibility gate `top_n_momentum_weight_func` uses, so benchmark
    strategies draw from an identical pool at every rebalance.
    """
    window = _trailing_window(history, lookback_days)
    if window is None:
        return []
    return list(window.columns[window.notna().all()])


def _rank_top_n(window: pd.DataFrame, n: int, direction: Direction = "best") -> pd.Series:
    """Top (or bottom) n tickers in `window` by return from its first to its last
    row. "best" is descending (highest return first), "worst" is ascending
    (most negative return first).
    """
    valid = window.columns[window.notna().all()]
    if len(valid) == 0:
        return pd.Series(dtype=float)
    trailing_return = window[valid].iloc[-1] / window[valid].iloc[0] - 1
    n = min(n, len(valid))
    return trailing_return.nsmallest(n) if direction == "worst" else trailing_return.nlargest(n)


def top_n_momentum_weight_func(lookback_days: int, n: int, direction: Direction = "best"):
    """Builds a weight_func for `finance.backtest.run_backtest`: at each call, ranks
    every ticker in `history` by its return over the trailing `lookback_days`
    calendar days and equal-weights the top (or bottom) `n`.
    """

    def weight_func(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        window = _trailing_window(history, lookback_days)
        if window is None:
            return {}
        picks = _rank_top_n(window, n, direction)
        if picks.empty:
            return {}
        w = 1 / len(picks)
        return {t: w for t in picks.index}

    return weight_func


def equal_weight_universe_weight_func(lookback_days: int):
    """Benchmark: equal-weight every eligible ticker in the universe at each
    rebalance (same eligibility gate and schedule as the momentum strategy) —
    "what if I just owned the whole universe instead of picking winners/losers."
    """

    def weight_func(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        valid = eligible_universe(history, lookback_days)
        if not valid:
            return {}
        w = 1 / len(valid)
        return {t: w for t in valid}

    return weight_func


def random_n_weight_func(lookback_days: int, n: int, seed: int | None = None):
    """Benchmark: at each rebalance, equal-weight `n` randomly chosen eligible
    tickers — a control for whether the momentum ranking itself adds value
    versus just holding *some* n stocks on the same schedule.
    """
    rng = random.Random(seed)

    def weight_func(date: pd.Timestamp, history: pd.DataFrame) -> dict[str, float]:
        valid = eligible_universe(history, lookback_days)
        if not valid:
            return {}
        picks = rng.sample(valid, min(n, len(valid)))
        w = 1 / len(picks)
        return {t: w for t in picks}

    return weight_func


def random_n_average_equity(
    prices: pd.DataFrame,
    lookback_days: int,
    n: int,
    freq: int,
    start_date: pd.Timestamp,
    trials: int = 1000,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Runs the random-N control `trials` times (different seed each time) and
    averages the resulting equity curves — a Monte Carlo estimate of what randomly
    picking N stocks on this schedule would do on average, smoothing out the luck
    of any single draw. Deterministic: always uses seeds 0..trials-1.

    Vectorized: eligibility is computed once per rebalance period (not once per
    trial), and each trial only has to average the picked tickers' daily returns
    over each period — a plain day-by-day `run_backtest` call per trial would be
    far too slow at `trials=1000` over multi-year windows.

    `cost_bps`: round-trip transaction cost in basis points charged on every
    rebalance day (matching `run_backtest`'s `cost_bps`), skipped on the seed day.
    """
    from finance.backtest import rebalance_dates

    daily_returns = prices.pct_change(fill_method=None).fillna(0)
    dates = rebalance_dates(prices, freq, anchor=start_date)

    on_or_after = prices.index[prices.index >= start_date]
    if len(on_or_after) == 0:
        raise ValueError("start_date is after the available price history")
    equity_index = prices.index[prices.index.get_loc(on_or_after[0]) :]

    prior_dates = dates[dates <= equity_index[0]]
    seed_date = prior_dates[-1] if len(prior_dates) else None
    active_dates = dates[dates >= equity_index[0]]

    # If seed_date coincides with the first active rebalance (the usual case,
    # since `dates` is anchored to start_date), only count it once — matching
    # run_backtest, which calls weight_func a single time for that date.
    periods: list[tuple[int, list[str]]] = []
    if seed_date is not None and (len(active_dates) == 0 or seed_date < active_dates[0]):
        periods.append((0, eligible_universe(prices.loc[:seed_date], lookback_days)))
    for d in active_dates:
        idx = equity_index.get_loc(d)
        periods.append((idx, eligible_universe(prices.loc[:d], lookback_days)))

    tickers = list(prices.columns)
    ticker_idx = {t: i for i, t in enumerate(tickers)}
    returns_matrix = daily_returns.loc[equity_index, tickers].to_numpy()
    n_days = len(equity_index)
    cost_per_rebalance = cost_bps / 10_000

    equity_sum = pd.Series(0.0, index=equity_index)
    for seed in range(trials):
        rng = random.Random(seed)
        day_ret = pd.Series(0.0, index=equity_index)
        for i, (start_i, eligible) in enumerate(periods):
            end_i = periods[i + 1][0] if i + 1 < len(periods) else n_days
            if not eligible:
                continue
            picks = rng.sample(eligible, min(n, len(eligible)))
            idxs = [ticker_idx[t] for t in picks]
            day_ret.iloc[start_i:end_i] = returns_matrix[start_i:end_i, idxs].mean(axis=1)
            if start_i > 0:
                day_ret.iloc[start_i] -= cost_per_rebalance
        day_ret.iloc[0] = 0.0  # day 0 is the fixed baseline (equity=1.0), same as run_backtest
        equity_sum += (1 + day_ret).cumprod()

    return equity_sum / trials


def picks_by_rebalance(
    prices: pd.DataFrame, dates: pd.DatetimeIndex, lookback_days: int, n: int, direction: Direction = "best"
) -> pd.DataFrame:
    """For each rebalance date, the top (or bottom) n tickers picked and the
    trailing lookback return (over `lookback_days` calendar days) that earned
    them the pick. Columns: date, rank, ticker, trailing_return.
    """
    rows = []
    for d in dates:
        history = prices.loc[:d]
        window = _trailing_window(history, lookback_days)
        if window is None:
            continue
        picks = _rank_top_n(window, n, direction)
        for rank, (ticker, ret) in enumerate(picks.items(), start=1):
            rows.append({"date": d, "rank": rank, "ticker": ticker, "trailing_return": ret})
    return pd.DataFrame(rows)
