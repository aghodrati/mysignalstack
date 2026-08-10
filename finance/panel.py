"""Builds a historical (rebalance date x ticker) factor + forward-return panel
for backtesting or learning factor weights -- ties together `finance.xbrl`
(point-in-time fundamentals), `finance.pointintime` (point-in-time
insider/analyst/institutional windowing), and price history (point-in-time by
construction) into rows a regression/ranking model can train on.

Every *feature* column is computed using only information knowable as of that
row's date. `forward_return` is the deliberate exception -- it looks forward
by design, since it's the training target, not a feature.

Compared to the live Ranking tab (`finance.ranking`), a few factors are
dropped or substituted here because there's no honest point-in-time source
for them yet:

- Gross margin: already dropped from Ranking itself (coverage reasons).
- Short interest: no free historical archive wired up yet (FINRA bi-monthly
  data was scoped in discussion but not built).
- Forward P/E, PEG, EV/Revenue: need forward analyst estimates (no free
  historical source) or debt/cash XBRL tags (not pulled yet). Trailing P/E
  and Price/Sales are used as point-in-time-safe stand-ins for Valuation.
"""

from __future__ import annotations

import pandas as pd

from finance.pointintime import (
    analyst_upside_as_of,
    insider_flow_as_of,
    institutional_flow_as_of,
    net_upgrades_as_of,
    revisions_trend_as_of,
    split_adjustment_factor,
)
from finance.xbrl import as_of_value, get_concept_history

CONCEPTS = ["revenue", "net_income", "eps_diluted", "shares_outstanding", "operating_income", "operating_cash_flow"]

FACTOR_COLUMNS = [
    "revenue_growth",
    "earnings_growth",
    "trailing_pe",
    "price_to_sales",
    "operating_margin",
    "fcf_margin",
    "momentum",
    "relative_strength",
    "price_vs_ma",
    "relative_volume",
    "analyst_upside",
    "net_upgrades",
    "revisions_trend",
    "institutional_flow",
    "insider_flow",
]


def _yoy_growth(history: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """% change in trailing-twelve-month value vs the same window one year
    earlier, both computed only from filings known by their respective dates.
    """
    now = as_of_value(history, as_of, ttm=True)
    prior = as_of_value(history, as_of - pd.DateOffset(years=1), ttm=True)
    if now is None or prior is None or prior == 0:
        return None
    return (now / prior - 1) * 100


def build_panel(
    tickers: list[str],
    company_names: dict[str, str],
    rebalance_dates: list[pd.Timestamp],
    prices: pd.DataFrame,
    benchmark: str,
    volumes: pd.DataFrame | None = None,
    momentum_weeks: int = 12,
    ma_period: str = "50d",
    forward_days: int = 30,
) -> pd.DataFrame:
    """One row per (rebalance date, ticker): every factor in `FACTOR_COLUMNS`
    plus `forward_return` (the ticker's return from that date to the first
    trading day on/after `forward_days` later -- the training target).

    `prices` must be a daily-close DataFrame covering every ticker plus
    `benchmark`, spanning from well before the first rebalance date (at least
    a year, for YoY growth and the 200-day MA) to at least `forward_days`
    past the last one. `volumes` (from `finance.data.get_volume`, same shape
    as `prices` minus the benchmark) is optional -- `relative_volume` is left
    NaN if omitted.
    """
    concept_history = {t: {c: get_concept_history(t, c) for c in CONCEPTS} for t in tickers}
    ma_days = 50 if ma_period == "50d" else 200

    rows = []
    for d in rebalance_dates:
        price_history = prices.loc[prices.index <= d]
        bench_series = price_history[benchmark].dropna() if benchmark in price_history.columns else pd.Series(dtype=float)
        volume_history = volumes.loc[volumes.index <= d] if volumes is not None else None

        for t in tickers:
            if t not in prices.columns:
                continue
            c = concept_history[t]
            row: dict[str, object] = {"date": d, "ticker": t}

            rev_ttm = as_of_value(c["revenue"], d, ttm=True)
            row["revenue_growth"] = _yoy_growth(c["revenue"], d)
            row["earnings_growth"] = _yoy_growth(c["net_income"], d)

            price_series = price_history[t].dropna() if t in price_history.columns else pd.Series(dtype=float)
            price_at_d = float(price_series.iloc[-1]) if len(price_series) else None

            # EPS and shares outstanding are reported in nominal, un-split-adjusted
            # terms, but `price_at_d` (from `prices`) is always split-adjusted --
            # see `split_adjustment_factor`. Adjusted per-row (by each
            # contributing quarter's own end date, not `d`) since a TTM sum's
            # four quarters can straddle a split even when `d` itself is safely
            # after it.
            eps_ttm_adj = as_of_value(
                c["eps_diluted"], d, ttm=True, adjust=lambda end, tk=t: 1 / split_adjustment_factor(tk, end)
            )
            row["trailing_pe"] = (price_at_d / eps_ttm_adj) if price_at_d and eps_ttm_adj and eps_ttm_adj > 0 else None

            shares_adj = as_of_value(
                c["shares_outstanding"], d, ttm=False, adjust=lambda end, tk=t: split_adjustment_factor(tk, end)
            )
            market_cap = (price_at_d * shares_adj) if price_at_d and shares_adj else None
            row["price_to_sales"] = (market_cap / rev_ttm) if market_cap and rev_ttm else None

            oi_ttm = as_of_value(c["operating_income"], d, ttm=True)
            row["operating_margin"] = (oi_ttm / rev_ttm) if oi_ttm is not None and rev_ttm else None
            ocf_ttm = as_of_value(c["operating_cash_flow"], d, ttm=True)
            row["fcf_margin"] = (ocf_ttm / rev_ttm) if ocf_ttm is not None and rev_ttm else None

            if price_at_d is not None and len(price_series) > 1:
                lookback_start = price_series.index[-1] - pd.Timedelta(days=momentum_weeks * 7)
                window = price_series[price_series.index >= lookback_start]
                if len(window) > 1:
                    stock_ret = window.iloc[-1] / window.iloc[0] - 1
                    row["momentum"] = stock_ret * 100
                    bench_window = bench_series[(bench_series.index >= window.index[0]) & (bench_series.index <= window.index[-1])]
                    if len(bench_window) > 1:
                        bench_ret = bench_window.iloc[-1] / bench_window.iloc[0] - 1
                        row["relative_strength"] = (stock_ret - bench_ret) * 100
                if len(price_series) >= ma_days:
                    ma = price_series.tail(ma_days).mean()
                    row["price_vs_ma"] = (price_series.iloc[-1] / ma - 1) * 100

            row["relative_volume"] = None
            if volume_history is not None and t in volume_history.columns:
                vol_series = volume_history[t].dropna()
                if len(vol_series) >= 60:
                    short_avg, long_avg = vol_series.tail(10).mean(), vol_series.tail(60).mean()
                    if long_avg:
                        row["relative_volume"] = (short_avg / long_avg - 1) * 100

            row["analyst_upside"] = analyst_upside_as_of(t, d, price_at_d) if price_at_d else None
            row["net_upgrades"] = net_upgrades_as_of(t, d)
            row["revisions_trend"] = revisions_trend_as_of(t, d)
            row["institutional_flow"] = institutional_flow_as_of(t, company_names.get(t, t), d)
            row["insider_flow"] = insider_flow_as_of(t, d)

            entry_candidates = prices.index[prices.index >= d]
            row["forward_return"] = None
            if len(entry_candidates) and t in prices.columns:
                entry_date = entry_candidates[0]
                exit_candidates = prices.index[prices.index >= d + pd.Timedelta(days=forward_days)]
                if len(exit_candidates):
                    exit_date = exit_candidates[0]
                    p0, p1 = prices.loc[entry_date, t], prices.loc[exit_date, t]
                    if pd.notna(p0) and pd.notna(p1) and p0:
                        row["forward_return"] = (p1 / p0 - 1) * 100

            rows.append(row)

    return pd.DataFrame(rows, columns=["date", "ticker", *FACTOR_COLUMNS, "forward_return"])


def add_tickers_to_panel(
    new_tickers: list[str],
    company_names: dict[str, str],
    benchmark: str,
    panel_path: str = "output/panel.csv",
    momentum_weeks: int = 12,
    ma_period: str = "50d",
    forward_days: int = 20,
    price_buffer_days: int = 200,
) -> pd.DataFrame:
    """Appends `new_tickers` to an existing panel CSV without rebuilding it --
    reuses the existing panel's own rebalance dates, computes factors only for
    the new tickers over that same date range, and concatenates.

    `momentum_weeks`/`ma_period`/`forward_days` must match whatever the
    existing panel was built with (they aren't stored in the CSV), or the new
    tickers' rows won't be computed the same way as the rest of the panel.
    Cheap relative to a full rebuild: cost scales with the number of new
    tickers, not the whole universe, since most of the per-ticker time is
    XBRL/point-in-time lookups that don't touch existing rows.

    Overlapping (date, ticker) rows -- i.e. re-adding a ticker already in the
    panel -- are resolved in favor of the newly computed row.
    """
    from finance.data import get_prices, get_volume  # local import: avoid a hard dependency for callers that don't need it

    existing = pd.read_csv(panel_path, parse_dates=["date"])
    rebalance_dates = sorted(existing["date"].unique())

    price_start = (pd.Timestamp(rebalance_dates[0]) - pd.Timedelta(days=price_buffer_days)).date().isoformat()
    price_end = (pd.Timestamp(rebalance_dates[-1]) + pd.Timedelta(days=forward_days + 30)).date().isoformat()

    prices = get_prices([*new_tickers, benchmark], start=price_start, end=price_end)
    volumes = get_volume(new_tickers, start=price_start, end=price_end)

    new_panel = build_panel(
        new_tickers,
        company_names,
        list(rebalance_dates),
        prices,
        benchmark,
        volumes=volumes,
        momentum_weeks=momentum_weeks,
        ma_period=ma_period,
        forward_days=forward_days,
    )

    combined = pd.concat([existing, new_panel], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_csv(panel_path, index=False)
    return combined
