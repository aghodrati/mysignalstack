"""Relative rank of stocks across a multi-factor scoring model: Growth,
Valuation, Quality, Momentum, Sentiment, and Ownership/Flow. Each raw factor is
converted to a 0-100 percentile rank within the universe (100 = best),
factors within a category are averaged equally (no factor is obviously more
important than another without a backtest to justify it), and categories are
combined into one composite score using caller-supplied weights -- so *which*
factors matter is fixed, but *how much* each category matters is a choice the
caller makes, not something baked in here.

Every value here comes from yfinance (via `finance.data`) -- today's live
snapshot, not a historical record. Compare to `finance.panel`, which blends
yfinance with SEC XBRL and SEC 13F bulk data specifically to be point-in-time
reconstructible for a past date.
"""

from __future__ import annotations

import pandas as pd

from finance.data import (
    get_insider_transactions,
    get_institutional_ownership,
    get_recommendations_trend,
    get_stock_info,
    get_upgrades_downgrades,
)

FACTOR_CATEGORIES: dict[str, list[str]] = {
    "Growth": ["revenue_growth", "earnings_growth"],
    "Valuation": ["forward_pe", "peg_ratio", "ev_to_revenue"],
    "Quality": ["operating_margin", "fcf_margin"],
    "Momentum": ["momentum", "relative_strength", "relative_volume"],
    "Sentiment": ["analyst_upside", "revisions_trend", "net_upgrades"],
    "Ownership": ["institutional_flow", "insider_flow", "low_short_interest"],
}

# Whether a higher raw value is better (True) or lower is better (False) --
# controls percentile-rank direction per factor.
HIGHER_IS_BETTER: dict[str, bool] = {
    "revenue_growth": True,
    "earnings_growth": True,
    "forward_pe": False,
    "peg_ratio": False,
    "ev_to_revenue": False,
    "operating_margin": True,
    "fcf_margin": True,
    "momentum": True,
    "relative_strength": True,
    "relative_volume": True,
    "analyst_upside": True,
    "revisions_trend": True,
    "net_upgrades": True,
    "institutional_flow": True,
    "insider_flow": True,
    "low_short_interest": True,
}

# category -> {factor: one-line description}, for a "what does this mean" reference in the UI.
FACTOR_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "Growth": {
        "revenue_growth": (
            "Most recent reported quarter's revenue vs. the same quarter a year ago (yfinance's own "
            "definition -- a single quarter, not a trailing-twelve-months figure)."
        ),
        "earnings_growth": (
            "Most recent reported quarter's earnings vs. the same quarter a year ago (yfinance's own "
            "definition -- a single quarter, not a trailing-twelve-months figure)."
        ),
    },
    "Valuation": {
        "forward_pe": "Price divided by next-12-months consensus EPS estimate. Lower is better.",
        "peg_ratio": "Forward P/E divided by expected earnings growth rate. Lower is better.",
        "ev_to_revenue": "Enterprise value divided by trailing revenue. Lower is better.",
    },
    "Quality": {
        "operating_margin": "Operating income divided by revenue.",
        "fcf_margin": "Operating cash flow divided by revenue (a free-cash-flow-margin proxy).",
    },
    "Momentum": {
        "momentum": "Raw price return over the chosen lookback window.",
        "relative_strength": "That same return minus the S&P 500's return over the identical window.",
        "relative_volume": "Today's volume divided by average volume.",
    },
    "Sentiment": {
        "analyst_upside": (
            "% gap between analyst mean target price and the current price. Positive = target is "
            "above current price (analysts expect it to rise); negative = target is below current "
            "price (analysts think it's already overvalued)."
        ),
        "revisions_trend": "Change in % of analysts rating the stock buy/strong-buy, now vs. 3 months ago.",
        "net_upgrades": "Upgrades minus downgrades in the trailing 90 days.",
    },
    "Ownership": {
        "institutional_flow": "Average % change in institutional/mutual-fund holdings vs. the prior quarter.",
        "insider_flow": "Net $ of insider buys minus sells (Form 4 filings).",
        "low_short_interest": "Short interest as % of float, sign-flipped so higher = less shorted = better.",
    },
}


def _revisions_trend(rec: pd.DataFrame) -> float | None:
    """Change in analyst bullishness: % rated buy/strongBuy now vs 3 months ago,
    in percentage points.
    """
    if rec.empty or "0m" not in rec["period"].values or "-3m" not in rec["period"].values:
        return None
    now = rec[rec["period"] == "0m"].iloc[0]
    then = rec[rec["period"] == "-3m"].iloc[0]
    now_total = now[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum()
    then_total = then[["strongBuy", "buy", "hold", "sell", "strongSell"]].sum()
    if now_total == 0 or then_total == 0:
        return None
    now_bullish = (now["strongBuy"] + now["buy"]) / now_total
    then_bullish = (then["strongBuy"] + then["buy"]) / then_total
    return (now_bullish - then_bullish) * 100


def _net_upgrades(ud: pd.DataFrame, days: int = 90) -> float:
    """Upgrades minus downgrades in the trailing `days`."""
    if ud.empty:
        return 0.0
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    recent = ud[ud["grade_date"] >= cutoff]
    if recent.empty:
        return 0.0
    return float((recent["action"] == "up").sum() - (recent["action"] == "down").sum())


def build_factor_table(
    tickers: list[str],
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    momentum_weeks: int = 12,
) -> pd.DataFrame:
    """Raw (unranked) factor values, one row per ticker, indexed by ticker.

    `prices` must cover enough daily-close history for every ticker plus the
    benchmark to compute the requested momentum lookback.
    """
    rows = []
    for t in tickers:
        info = get_stock_info(t)
        rec = get_recommendations_trend(t)
        ud = get_upgrades_downgrades(t)
        insider = get_insider_transactions(t)
        ownership = get_institutional_ownership(t)

        row: dict[str, float | None] = {"ticker": t}
        row["revenue_growth"] = info.get("revenueGrowth")
        row["earnings_growth"] = info.get("earningsGrowth")
        row["forward_pe"] = info.get("forwardPE")
        row["peg_ratio"] = info.get("pegRatio")
        row["ev_to_revenue"] = info.get("enterpriseToRevenue")
        row["operating_margin"] = info.get("operatingMargins")
        op_cf, revenue = info.get("operatingCashflow"), info.get("totalRevenue")
        row["fcf_margin"] = (op_cf / revenue) if op_cf and revenue else None

        if t in prices.columns:
            series = prices[t].dropna()
            if len(series) > 1:
                lookback_days = momentum_weeks * 7
                window = series[series.index >= series.index[-1] - pd.Timedelta(days=lookback_days)]
                if len(window) > 1:
                    stock_ret = window.iloc[-1] / window.iloc[0] - 1
                    row["momentum"] = stock_ret * 100
                    bench_window = benchmark_prices[
                        (benchmark_prices.index >= window.index[0]) & (benchmark_prices.index <= window.index[-1])
                    ]
                    if len(bench_window) > 1:
                        bench_ret = bench_window.iloc[-1] / bench_window.iloc[0] - 1
                        row["relative_strength"] = (stock_ret - bench_ret) * 100

        avg_vol, cur_vol = info.get("averageVolume"), (info.get("volume") or info.get("regularMarketVolume"))
        row["relative_volume"] = (cur_vol / avg_vol) if avg_vol and cur_vol else None

        target = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        row["analyst_upside"] = ((target / current - 1) * 100) if target and current else None

        row["revisions_trend"] = _revisions_trend(rec)
        row["net_upgrades"] = _net_upgrades(ud)

        row["institutional_flow"] = (ownership["pct_change"].mean() * 100) if not ownership.empty else None

        if not insider.empty:
            is_buy = insider["transaction"].str.contains("Buy|Purchase", case=False, na=False, regex=True)
            is_sale = insider["transaction"].str.contains("Sale", case=False, na=False)
            row["insider_flow"] = insider.loc[is_buy, "value"].sum() - insider.loc[is_sale, "value"].sum()
        else:
            row["insider_flow"] = None

        short_pct = info.get("shortPercentOfFloat")
        row["low_short_interest"] = (-short_pct * 100) if short_pct is not None else None

        rows.append(row)

    return pd.DataFrame(rows).set_index("ticker")


def percentile_rank_table(factors: pd.DataFrame) -> pd.DataFrame:
    """Each factor column converted to a 0-100 percentile rank within the
    universe (100 = best), respecting HIGHER_IS_BETTER direction per factor.
    Missing values stay NaN (excluded from that stock's category average).
    """
    ranked = pd.DataFrame(index=factors.index)
    for col in factors.columns:
        ascending = HIGHER_IS_BETTER.get(col, True)
        ranked[col] = factors[col].rank(pct=True, ascending=ascending) * 100
    return ranked


def category_scores(ranked: pd.DataFrame) -> pd.DataFrame:
    """Average percentile rank per category, one column per category. A stock
    missing every factor in a category gets NaN for that category.
    """
    scores = pd.DataFrame(index=ranked.index)
    for category, factor_names in FACTOR_CATEGORIES.items():
        present = [f for f in factor_names if f in ranked.columns]
        scores[category] = ranked[present].mean(axis=1) if present else float("nan")
    return scores


def composite_score(scores: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average of category scores. Weights are normalized to sum to 1
    (a category missing from `weights` gets 0 weight). A stock missing a
    category's score falls back to that category's universe-average score for
    the weighting, rather than being penalized for missing data.
    """
    total_weight = sum(weights.get(c, 0.0) for c in scores.columns)
    if total_weight == 0:
        return pd.Series(float("nan"), index=scores.index)
    composite = pd.Series(0.0, index=scores.index)
    for c in scores.columns:
        w = weights.get(c, 0.0) / total_weight
        composite += scores[c].fillna(scores[c].mean()) * w
    return composite
