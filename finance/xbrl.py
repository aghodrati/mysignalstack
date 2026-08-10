"""Point-in-time fundamentals from SEC's XBRL Company Facts API -- unlike
yfinance's `.info` (today's snapshot only), every value here is tagged with
its actual filing date, so a factor's value "as of any past date" can be
reconstructed by filtering to filings that existed by that date (see
`as_of_value` / `as_of_series`).

Company financials aren't tagged with one universal XBRL concept name -- the
same line item (e.g. revenue) is reported under different tags depending on
the company and filing year (accounting standard changes, company-specific
extensions, etc.). `CONCEPT_TAGS` lists a fallback chain per concept; the
first tag with data wins.

Raw API responses are cached indefinitely under `cache/xbrl/raw/` (small,
one JSON per ticker+tag, never deleted) since they're a slow-changing
historical record worth keeping around for future use, not a throwaway
intermediate like the SEC 13F bulk downloads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "xbrl"
RAW_DIR = CACHE_DIR / "raw"
PARSED_DIR = CACHE_DIR / "parsed"
CIK_MAP_PATH = CACHE_DIR / "cik_map.json"

HEADERS = {"User-Agent": "finance-hypothesis-tool ghodrati@gmail.com"}

# Fallback chain per concept: SEC/GAAP tagging isn't consistent across
# companies or years, so try each tag in order and use the first with data.
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "shares_outstanding": ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}


def get_cik_map(refresh: bool = False) -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, from SEC's official mapping file.
    Cached indefinitely (refresh only if a ticker you need is missing).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CIK_MAP_PATH.exists() and not refresh:
        with open(CIK_MAP_PATH) as f:
            return json.load(f)

    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    mapping = {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in raw.values()}
    with open(CIK_MAP_PATH, "w") as f:
        json.dump(mapping, f)
    return mapping


def _fetch_concept_raw(cik: str, tag: str, refresh: bool = False) -> dict | None:
    """Raw `companyconcept` API response for one (CIK, tag), cached indefinitely.
    Returns None if this tag doesn't exist for this company (a normal outcome --
    callers should try the next tag in the fallback chain).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{cik}_{tag}.json"
    if raw_path.exists() and not refresh:
        with open(raw_path) as f:
            cached = json.load(f)
        return cached if cached else None

    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 404:
        with open(raw_path, "w") as f:
            json.dump({}, f)
        return None
    resp.raise_for_status()
    data = resp.json()
    with open(raw_path, "w") as f:
        json.dump(data, f)
    time.sleep(0.1)  # stay well under SEC's fair-access rate limit across a full-universe sweep
    return data


def get_concept_history(ticker: str, concept: str, refresh: bool = False) -> pd.DataFrame:
    """Full reported history for one concept (e.g. "revenue"), merged across
    every tag in `CONCEPT_TAGS[concept]` that has data -- companies commonly
    switch XBRL tags mid-history (e.g. most switched revenue tags around 2018
    when ASC 606 took effect), so using only the first tag with *any* data
    would silently miss everything reported under a later tag. Columns: start
    (period start, NaT for instant values like shares outstanding), end
    (period end), val, filed (filing date -- this is what makes it
    point-in-time-safe), form, fy, fp, tag (which XBRL tag that row came from).

    Parsed result is cached per (ticker, concept); pass refresh=True to
    re-derive it from the raw JSON (already-cached raw responses are still
    reused unless refresh also re-fetches those).
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_path = PARSED_DIR / f"{ticker}_{concept}.csv"
    columns = ["start", "end", "val", "filed", "form", "fy", "fp", "tag"]
    date_cols = ["start", "end", "filed"]
    if parsed_path.exists() and not refresh:
        present_dates = [c for c in date_cols if c in pd.read_csv(parsed_path, nrows=0).columns]
        return pd.read_csv(parsed_path, parse_dates=present_dates)

    cik_map = get_cik_map()
    cik = cik_map.get(ticker.upper())
    if cik is None:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(parsed_path, index=False)
        return empty

    frames = []
    for tag in CONCEPT_TAGS.get(concept, [concept]):
        data = _fetch_concept_raw(cik, tag, refresh=refresh)
        if not data:
            continue
        units = data.get("units", {})
        rows = units.get("USD") or units.get("shares") or units.get("USD/shares")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["tag"] = tag
        # Reindex to the full column set (not just whichever happen to be
        # present) -- "instant" facts like shares outstanding have no "start"
        # date, and without this a concept mixing instant/duration facts (or
        # just this one alone) would write an inconsistent CSV schema.
        frames.append(df.reindex(columns=columns))

    if not frames:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(parsed_path, index=False)
        return empty

    combined = pd.concat(frames, ignore_index=True)
    for date_col in ["start", "end", "filed"]:
        if date_col in combined.columns:
            combined[date_col] = pd.to_datetime(combined[date_col])
    # Different tags can double-report the same period during a transition
    # (both the old and new tag briefly present in the same filing); keep one
    # row per (start, end, form) -- last tag in the fallback list wins, which
    # is fine since fallback order doesn't encode any particular preference
    # here, just candidate tags.
    dedup_keys = [c for c in ["start", "end", "form", "fy", "fp"] if c in combined.columns]
    combined = combined.sort_values("filed").drop_duplicates(subset=dedup_keys, keep="last")
    combined = _derive_missing_quarter(combined)
    combined = combined.sort_values(["end", "filed"]).reset_index(drop=True)
    combined.to_csv(parsed_path, index=False)
    return combined


def _derive_missing_quarter(combined: pd.DataFrame) -> pd.DataFrame:
    """For each fiscal year with an annual (10-K) total but only 3 of its 4
    quarters reported as standalone ~90-day facts, derives the missing one
    (almost always Q4 -- a 10-K reports the full fiscal year, not an isolated
    Q4 duration, so nearly every company has this gap) as
    (FY total) - (sum of the 3 known quarters), appended as a synthetic row.

    Uses each period's *earliest* filed occurrence, not the latest: SEC's API
    repeats a prior period's figure as a same-period comparative column in a
    later filing (e.g. a Q3 2023 10-Q includes Q3 2022 again, dated by *that*
    filing), and point-in-time analysis should use what was knowable when the
    period was first disclosed, not a later comparative appearance or
    restatement.

    Groups quarters to their fiscal year by actual date-range containment
    (quarter.start/end falling inside the annual period's start/end), not by
    SEC's own "fy"/"fp" tags -- those describe the *filing's* fiscal context,
    not the reported period's, so a 10-Q's prior-year comparative quarter gets
    tagged with the current filing's fy, not the comparative period's real
    one. Grouping by that tag mixes unrelated periods together.
    """
    if combined.empty or "start" not in combined.columns:
        return combined

    duration_days = (combined["end"] - combined["start"]).dt.days
    quarterly = combined[duration_days.between(80, 100)].sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")
    annual = combined[duration_days.between(350, 380)].sort_values("filed").drop_duplicates(subset=["start", "end"], keep="first")

    synthetic = []
    for _, fy_row in annual.iterrows():
        q_rows = quarterly[(quarterly["start"] >= fy_row["start"]) & (quarterly["end"] <= fy_row["end"])]
        if len(q_rows) != 3:
            continue
        synthetic.append(
            {
                "start": q_rows["end"].max(),
                "end": fy_row["end"],
                "val": fy_row["val"] - q_rows["val"].sum(),
                "filed": fy_row["filed"],
                "form": "derived",
                "fy": fy_row["fy"],
                "fp": "Q4-derived",
                "tag": "derived",
            }
        )
    if not synthetic:
        return combined
    return pd.concat([combined, pd.DataFrame(synthetic)], ignore_index=True)


def as_of_value(
    history: pd.DataFrame,
    as_of: pd.Timestamp,
    ttm: bool = False,
    adjust: Callable[[pd.Timestamp], float] | None = None,
) -> float | None:
    """The most recent value known as of `as_of` (only uses rows filed on or
    before that date -- the point-in-time guarantee). If `ttm`, sums the
    trailing four quarterly (3-month) periods ending at or before `as_of`
    instead of returning the latest single reported period (for flow items
    like revenue/net income, where a single quarter isn't a great magnitude
    to compare across companies with different fiscal-quarter timing).

    `adjust`, if given, is called with each *contributing row's own* period-end
    date and multiplies that row's value by the result before use -- pass
    e.g. `lambda end: 1 / split_adjustment_factor(ticker, end)` for a
    per-share figure to correct for splits that happened between that specific
    quarter's report and `as_of` (not just between `as_of` and today). This
    must be per-row, not a single factor applied to the aggregate, because a
    TTM sum's four quarters can straddle a split even when `as_of` itself is
    safely after it.
    """
    if history.empty:
        return None
    known = history[history["filed"] <= as_of]
    if known.empty:
        return None

    if not ttm:
        # Most recent period; among duplicates for that same period (see
        # `_derive_missing_quarter`'s docstring), the earliest-filed one.
        max_end = known["end"].max()
        row = known[known["end"] == max_end].sort_values("filed").iloc[0]
        val = float(row["val"])
        return val * adjust(row["end"]) if adjust else val

    quarterly = known.copy()
    if "start" in quarterly.columns:
        quarterly = quarterly[(quarterly["end"] - quarterly["start"]).dt.days.between(80, 100)]
    # Prefer each period's earliest-filed occurrence -- SEC's API can repeat a
    # period as a later comparative figure (see `_derive_missing_quarter`'s
    # docstring); point-in-time analysis should use what was knowable when the
    # period was first disclosed.
    quarterly = quarterly.sort_values("filed").drop_duplicates(subset=["end"], keep="first").sort_values("end")
    if len(quarterly) < 4:
        return None
    quarterly = quarterly.tail(4)
    if adjust:
        return float(sum(row["val"] * adjust(row["end"]) for _, row in quarterly.iterrows()))
    return float(quarterly["val"].sum())
