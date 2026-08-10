"""Average shape of the trading day: for each bar, the return since that day's
open, averaged across every trading day in the period.
"""

from __future__ import annotations

import pandas as pd


def average_intraday_path(intraday_closes: pd.Series) -> pd.Series:
    """`intraday_closes`: tz-aware, datetime-indexed intraday bar closes for one
    ticker. Returns a Series indexed by time-of-day ("HH:MM", exchange local
    time) giving the average return since that day's opening bar, across every
    day present.
    """
    df = intraday_closes.dropna().to_frame("close")
    df["day"] = df.index.date
    df["time"] = df.index.strftime("%H:%M")
    day_open = df.groupby("day")["close"].transform("first")
    df["return_since_open"] = df["close"] / day_open - 1
    return df.groupby("time")["return_since_open"].mean().sort_index()
