"""Learning-to-rank on top of `output/panel.csv`: instead of regressing
`forward_return` directly (which optimizes for predicting a noisy magnitude),
train a model to get the *relative order* of stocks right on each rebalance
date -- LightGBM's LambdaMART ranker (`LGBMRanker`, `objective="lambdarank"`)
treats each date as a "query" and each ticker on that date as a candidate to
rank, learning directly from pairwise ordering mistakes within a date rather
than from any date's absolute return level (which varies a lot with market
regime and isn't comparable across dates anyway).

Split is time-based (train on earlier dates, test on later ones), not random
-- panel.py's factors are point-in-time but still autocorrelated within a
ticker across nearby dates (see finance/panel.py's module docstring), so a
random split would leak information from a stock's near-future into its
near-past.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURE_COLUMNS = [
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


def load_panel(path: str = "output/panel.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def add_relevance_labels(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Relevance label per row: `forward_return` rank within its own date,
    bucketed into `n_bins` integer grades (0 = worst, n_bins-1 = best) via
    `qcut`. LightGBM's lambdarank objective wants small integer relevance
    grades, not the raw continuous return -- bucketing also matches the
    actual goal (get the ranking right) better than asking the model to
    preserve exact return magnitudes.
    """
    df = df.copy()
    df["relevance"] = df.groupby("date")["forward_return"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_bins, labels=False)
    )
    return df


def time_split(df: pd.DataFrame, test_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest dates, test on the most recent ones -- a plain
    date-ordered split, not random, since factors are autocorrelated within
    a ticker across nearby dates (see module docstring).
    """
    dates = sorted(df["date"].unique())
    n_test = max(1, round(len(dates) * test_fraction))
    train_dates, test_dates = dates[:-n_test], dates[-n_test:]
    return df[df["date"].isin(train_dates)], df[df["date"].isin(test_dates)]


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("date", sort=False).size().tolist()


def train_ranker(train_df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS, **lgb_kwargs) -> lgb.LGBMRanker:
    """`train_df` must already be sorted by date (matching `_group_sizes`'s
    grouping) and carry a `relevance` column (see `add_relevance_labels`).
    """
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=lgb_kwargs.pop("n_estimators", 200),
        learning_rate=lgb_kwargs.pop("learning_rate", 0.05),
        num_leaves=lgb_kwargs.pop("num_leaves", 15),
        min_child_samples=lgb_kwargs.pop("min_child_samples", 5),
        verbosity=-1,
        **lgb_kwargs,
    )
    model.fit(
        train_df[feature_cols],
        train_df["relevance"],
        group=_group_sizes(train_df),
    )
    return model


def evaluate(model: lgb.LGBMRanker, test_df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Per-date Spearman rank correlation between the model's predicted score
    and the actual `forward_return` -- the direct measure of "did it rank
    stocks in roughly the right order that day", independent of the
    relevance-bucketing used for training.
    """
    test_df = test_df.copy()
    test_df["pred_score"] = model.predict(test_df[feature_cols])

    rows = []
    for date, group in test_df.groupby("date"):
        corr, _ = spearmanr(group["pred_score"], group["forward_return"])
        rows.append({"date": date, "n": len(group), "spearman": corr})
    return pd.DataFrame(rows)


def feature_importance(model: lgb.LGBMRanker, feature_cols: list[str] = FEATURE_COLUMNS) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
