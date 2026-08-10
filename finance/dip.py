"""'Buy the dip' hypothesis: whenever a stock falls more than T% in a single day,
buy it and sell D days later. How does that compare to the S&P 500 over the same
holding window?
"""

from __future__ import annotations

import pandas as pd


def find_dip_trades(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    drop_pct: float,
    hold_days: int,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """For every ticker, every day its close-to-close return is <= -drop_pct, records
    a trade: buy at that close, sell at the close on the first trading day on/after
    `hold_days` calendar days later. Also records the benchmark's return over that
    identical window, for an apples-to-apples comparison. Trades whose exit would
    fall beyond the available data are skipped (incomplete).

    `cost_bps`: round-trip transaction cost in basis points, subtracted from each
    trade's stock_return (the benchmark side is a passive hold, not a trade, so
    it isn't charged).

    Columns: ticker, buy_date, buy_price, sell_date, sell_price, drop_that_day,
    stock_return, benchmark_return, excess_return.
    """
    cost = cost_bps / 10_000
    daily_returns = prices.pct_change(fill_method=None)
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        rets = daily_returns[ticker]
        for buy_date in series.index:
            r = rets.get(buy_date)
            if r is None or pd.isna(r) or r > -drop_pct:
                continue
            sell_target = buy_date + pd.Timedelta(days=hold_days)
            sell_candidates = series.index[series.index >= sell_target]
            if len(sell_candidates) == 0:
                continue
            sell_date = sell_candidates[0]
            buy_price = series.loc[buy_date]
            sell_price = series.loc[sell_date]

            bench_window = benchmark.loc[(benchmark.index >= buy_date) & (benchmark.index <= sell_date)]
            if bench_window.empty:
                continue
            benchmark_return = bench_window.iloc[-1] / bench_window.iloc[0] - 1
            stock_return = sell_price / buy_price - 1 - cost

            rows.append(
                {
                    "ticker": ticker,
                    "buy_date": buy_date,
                    "buy_price": buy_price,
                    "sell_date": sell_date,
                    "sell_price": sell_price,
                    "drop_that_day": r,
                    "stock_return": stock_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": stock_return - benchmark_return,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "buy_date",
                "buy_price",
                "sell_date",
                "sell_price",
                "drop_that_day",
                "stock_return",
                "benchmark_return",
                "excess_return",
            ]
        )
    return pd.DataFrame(rows).sort_values("buy_date").reset_index(drop=True)
