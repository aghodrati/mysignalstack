"""Post-earnings-announcement drift (PEAD): buy stocks whose earnings surprised by
more than X% (beats) or missed by more than X% (misses), hold for T days, and
compare against the S&P 500 over the identical window.

Surprise% (as reported by the data provider): (Reported EPS - EPS Estimate) /
|EPS Estimate| * 100.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Direction = Literal["beat", "miss"]

TRADE_COLUMNS = [
    "ticker",
    "earnings_date",
    "surprise_pct",
    "buy_date",
    "buy_price",
    "sell_date",
    "sell_price",
    "stock_return",
    "benchmark_return",
    "excess_return",
]


def find_pead_trades(
    ticker: str,
    prices: pd.Series,
    benchmark: pd.Series,
    earnings: pd.DataFrame,
    surprise_threshold_pct: float,
    hold_days: int,
    direction: Direction = "beat",
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """For every earnings report matching the surprise filter, a trade buying at
    the close of the first trading day *after* the earnings date (whether
    reported before or after market hours, day+1 avoids any same-day look-ahead)
    and selling at the close on/after `hold_days` calendar days later. Includes
    the benchmark's return over the identical window for comparison.

    `surprise_threshold_pct` is always a positive magnitude: direction="beat"
    keeps reports with surprise_pct > +threshold; direction="miss" keeps reports
    with surprise_pct < -threshold.

    `cost_bps`: round-trip transaction cost in basis points, subtracted from
    each trade's stock_return (the benchmark side is a passive hold, not a
    trade, so it isn't charged).
    """
    cost = cost_bps / 10_000
    series = prices.dropna()
    rows = []
    for row in earnings.itertuples(index=False):
        if pd.isna(row.surprise_pct):
            continue
        if direction == "beat" and row.surprise_pct <= surprise_threshold_pct:
            continue
        if direction == "miss" and row.surprise_pct >= -surprise_threshold_pct:
            continue

        entry_candidates = series.index[series.index.date > row.earnings_date.date()]
        if len(entry_candidates) == 0:
            continue
        buy_date = entry_candidates[0]

        sell_target = buy_date + pd.Timedelta(days=hold_days)
        exit_candidates = series.index[series.index >= sell_target]
        if len(exit_candidates) == 0:
            continue
        sell_date = exit_candidates[0]

        bench_window = benchmark.loc[(benchmark.index >= buy_date) & (benchmark.index <= sell_date)]
        if bench_window.empty:
            continue
        benchmark_return = bench_window.iloc[-1] / bench_window.iloc[0] - 1

        buy_price = series.loc[buy_date]
        sell_price = series.loc[sell_date]
        stock_return = sell_price / buy_price - 1 - cost

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": row.earnings_date,
                "surprise_pct": row.surprise_pct,
                "buy_date": buy_date,
                "buy_price": buy_price,
                "sell_date": sell_date,
                "sell_price": sell_price,
                "stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "excess_return": stock_return - benchmark_return,
            }
        )

    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)


STREAK_TRADE_COLUMNS = [
    "ticker",
    "earnings_date",
    "prior_surprise_1",
    "prior_surprise_2",
    "buy_date",
    "buy_price",
    "sell_date",
    "sell_price",
    "stock_return",
    "benchmark_return",
    "excess_return",
]


def find_earnings_streak_trades(
    ticker: str,
    prices: pd.Series,
    benchmark: pd.Series,
    earnings: pd.DataFrame,
    streak_threshold_pct: float,
    hold_days: int,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Alternative hypothesis (anticipatory, not reactive like PEAD above): a
    stock whose last two reported quarters *both* beat estimates by more than
    `streak_threshold_pct` tends to keep beating. Buy at the close of the last
    trading day before its next earnings report and sell at the close on/after
    `hold_days` calendar days after that report -- a bet the streak continues
    into the next print, taken before that print happens (so it's a bet on the
    streak, not a reaction to how that next report actually goes).

    `cost_bps`: round-trip transaction cost in basis points, subtracted from
    each trade's stock_return.
    """
    cost = cost_bps / 10_000
    series = prices.dropna()
    earnings = earnings.sort_values("earnings_date").reset_index(drop=True)
    rows = []
    for i in range(2, len(earnings)):
        prev1 = earnings.iloc[i - 1]
        prev2 = earnings.iloc[i - 2]
        if pd.isna(prev1.surprise_pct) or pd.isna(prev2.surprise_pct):
            continue
        if prev1.surprise_pct <= streak_threshold_pct or prev2.surprise_pct <= streak_threshold_pct:
            continue

        target = earnings.iloc[i]
        earnings_date = target.earnings_date

        entry_candidates = series.index[series.index.date < earnings_date.date()]
        if len(entry_candidates) == 0:
            continue
        buy_date = entry_candidates[-1]

        sell_target = earnings_date + pd.Timedelta(days=hold_days)
        exit_candidates = series.index[series.index >= sell_target]
        if len(exit_candidates) == 0:
            continue
        sell_date = exit_candidates[0]

        bench_window = benchmark.loc[(benchmark.index >= buy_date) & (benchmark.index <= sell_date)]
        if bench_window.empty:
            continue
        benchmark_return = bench_window.iloc[-1] / bench_window.iloc[0] - 1

        buy_price = series.loc[buy_date]
        sell_price = series.loc[sell_date]
        stock_return = sell_price / buy_price - 1 - cost

        rows.append(
            {
                "ticker": ticker,
                "earnings_date": earnings_date,
                "prior_surprise_1": prev1.surprise_pct,
                "prior_surprise_2": prev2.surprise_pct,
                "buy_date": buy_date,
                "buy_price": buy_price,
                "sell_date": sell_date,
                "sell_price": sell_price,
                "stock_return": stock_return,
                "benchmark_return": benchmark_return,
                "excess_return": stock_return - benchmark_return,
            }
        )

    if not rows:
        return pd.DataFrame(columns=STREAK_TRADE_COLUMNS)
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)
