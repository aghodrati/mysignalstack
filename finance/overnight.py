"""Overnight (previous close -> today's open) vs intraday (today's open -> today's
close) return decomposition. Their product on any given day reproduces that day's
close-to-close return: (1 + overnight) * (1 + intraday) = close[t] / close[t-1].
"""

from __future__ import annotations

import pandas as pd


def decompose_returns(opens: pd.DataFrame, closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (overnight_returns, intraday_returns), aligned on the same dates and
    tickers as `closes`. The first row's overnight return is NaN (no prior close).
    """
    prev_close = closes.shift(1)
    overnight = opens / prev_close - 1
    intraday = closes / opens - 1
    return overnight, intraday


def compounded_curve(daily_returns: pd.Series) -> pd.Series:
    """Equity curve (starting at 1.0) compounding a series of period returns."""
    return (1 + daily_returns.fillna(0)).cumprod()


def summarize(overnight: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker: compounded overnight-only return, compounded intraday-only
    return, and the compounded total (their product) over the whole window.

    Overnight and intraday are aligned to the same valid dates *before*
    compounding (both drop day 1, whose overnight leg is undefined without a
    prior close) — compounding them independently would multiply mismatched-
    length series and silently misstate the total.
    """
    rows = []
    for ticker in overnight.columns:
        paired = pd.concat([overnight[ticker], intraday[ticker]], axis=1, keys=["on", "id"]).dropna()
        if paired.empty:
            continue
        overnight_return = (1 + paired["on"]).prod() - 1
        intraday_return = (1 + paired["id"]).prod() - 1
        total_return = (1 + overnight_return) * (1 + intraday_return) - 1
        rows.append(
            {
                "ticker": ticker,
                "overnight_return": overnight_return,
                "intraday_return": intraday_return,
                "total_return": total_return,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")
