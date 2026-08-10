"""Pairwise correlation and divergence-from-usual-relationship analysis.

The idea: over some lookback window, some tickers move together (high
correlation of daily returns). Among those normally-correlated pairs, the
"spread" between their cumulative returns should hover around some typical
level. A pair is "diverged right now" when today's spread is many standard
deviations away from its own average over the window — i.e. the relationship
that usually holds has recently broken down.
"""

from __future__ import annotations

import pandas as pd


def pairwise_correlation(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """All unique ticker pairs with their return correlation over the given window."""
    corr_matrix = daily_returns.corr()
    cols = corr_matrix.columns
    rows = [
        {"ticker_a": cols[i], "ticker_b": cols[j], "correlation": corr_matrix.iloc[i, j]}
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]
    return pd.DataFrame(rows).dropna(subset=["correlation"])


def divergence_now(cumulative_return: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """For each pair, how many standard deviations today's return spread is from
    its own mean over the window (a simple mean-reversion-style divergence score).
    """
    rows = []
    for row in pairs.itertuples(index=False):
        spread = (cumulative_return[row.ticker_a] - cumulative_return[row.ticker_b]).dropna()
        if len(spread) < 10 or spread.std() == 0:
            continue
        z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
        rows.append(
            {
                "ticker_a": row.ticker_a,
                "ticker_b": row.ticker_b,
                "correlation": row.correlation,
                "current_spread": spread.iloc[-1],
                "mean_spread": spread.mean(),
                "z_score": z_score,
            }
        )
    return pd.DataFrame(rows)
