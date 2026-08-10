"""Performance stats for an equity curve (a Series of portfolio value over time)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(equity_curve: pd.Series) -> float:
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if n_years <= 0:
        return float("nan")
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    if excess.std() == 0:
        return float("nan")
    return np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std()


def summarize(equity_curve: pd.Series, label: str = "") -> dict:
    returns = equity_curve.pct_change(fill_method=None).dropna()
    return {
        "label": label,
        "total_return": equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        "cagr": cagr(equity_curve),
        "volatility": returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
    }


def summary_table(equity_curves: dict[str, pd.Series]) -> pd.DataFrame:
    rows = [summarize(curve, label) for label, curve in equity_curves.items()]
    return pd.DataFrame(rows).set_index("label")
