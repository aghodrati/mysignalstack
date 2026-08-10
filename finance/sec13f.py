"""Multi-quarter institutional ownership from SEC's bulk Form 13F structured
datasets. Unlike yfinance's institutional_holders (a single latest-quarter
snapshot with one quarter-over-quarter %% change), this can build that same
kind of %% change across as many past quarters as are cached.

To stay directly comparable to yfinance's own number, this uses the same
methodology: rank each quarter's filers by position size, take the top 10
("large institutional holders" -- same as yfinance's institutional_holders
table), and for each of those 10, compute *that specific holder's* %% change
from its own prior-quarter position (100%% if it wasn't a holder last
quarter -- yfinance's own convention for a new position). The overall %%
change is the plain average of those 10 numbers, same as yfinance.

Each quarterly dataset is a ~100-400MB zip (SEC publishes rolling 3-month
filing-date buckets). We download it once, aggregate every filer's position
by (CUSIP, CIK, period of report), cache that aggregate to disk, and discard
the raw files -- so repeat use only re-reads a small CSV per quarter.

There's no ticker/CUSIP field in the raw data, so the target company's CUSIP
is resolved by matching NAMEOFISSUER text against the company name and
picking whichever CUSIP has the largest aggregate $ value under that name
(the real common-stock listing dominates by orders of magnitude over
mis-typed variants or unrelated securities sharing a word).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "13f"
LISTING_URL = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
BASE_URL = "https://www.sec.gov"
HEADERS = {"User-Agent": "finance-hypothesis-tool ghodrati@gmail.com"}


def list_available_datasets() -> list[str]:
    """Zip file URLs for SEC's bulk 13F structured-data dumps, newest first."""
    resp = requests.get(LISTING_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    hrefs = re.findall(r'href="(/files/structureddata/data/form-13f-data-sets/[^"]+\.zip)"', resp.text)
    seen: list[str] = []
    for h in hrefs:
        if h not in seen:
            seen.append(h)
    return [BASE_URL + h for h in seen]


def _dataset_bucket_end(url: str) -> pd.Timestamp:
    """The end date of a dataset's filing-date bucket, parsed from its filename
    -- either the newer "01mon<year>-31mon<year>" range format or the older
    "<year>qN" quarter format. Used to pick datasets that don't extend past a
    given historical `as_of` date (see `datasets_as_of`).
    """
    name = _dataset_name(url).replace("_form13f", "")
    if re.match(r"^\d{4}q[1-4]$", name.lower()):
        year, q = name.lower().split("q")
        month = int(q) * 3
        return pd.Timestamp(year=int(year), month=month, day=1) + pd.offsets.MonthEnd(0)
    end_str = name.split("-")[-1]
    return pd.to_datetime(end_str, format="%d%b%Y")


def datasets_as_of(as_of: pd.Timestamp, quarters: int = 5, buffer: int = 2) -> list[str]:
    """Dataset URLs whose filing-date bucket ends on or before `as_of` (so none
    of them could contain information not yet knowable on that date), newest
    first, enough to cover `quarters` reporting periods plus a buffer for the
    same late-amendment-bucket disambiguation `get_institutional_history` does
    for the "now" case. Pass the result as `get_institutional_history`'s
    `datasets` argument to get a point-in-time-safe institutional history as of
    a past date instead of today.
    """
    urls = list_available_datasets()
    eligible = [u for u in urls if _dataset_bucket_end(u) <= as_of]
    return eligible[: quarters + buffer]


def _dataset_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].replace(".zip", "")


def _cusip_agg_path(url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"agg_{_dataset_name(url)}.csv"


def _holder_agg_path(url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"holders_{_dataset_name(url)}.csv"


def _download_and_aggregate(url: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads one quarterly 13F bulk zip, aggregates common-stock positions,
    caches two small tables, and deletes the raw ~100-400MB files:

    - cusip_agg: one row per (cusip, period of report), across every filer --
      used to resolve a company name to its CUSIP and to identify the primary
      reporting bucket for a period. Columns: cusip, period_of_report, name
      (most common NAMEOFISSUER spelling), total_shares, total_value,
      num_filers.
    - holder_agg: one row per (cusip, filer CIK, period of report) -- used to
      rank individual filers by position size. Columns: cusip, cik,
      period_of_report, shares, value.
    """
    agg_path = _cusip_agg_path(url)
    holder_path = _holder_agg_path(url)
    if agg_path.exists() and holder_path.exists() and not refresh:
        return (
            pd.read_csv(agg_path, parse_dates=["period_of_report"]),
            pd.read_csv(holder_path, parse_dates=["period_of_report"]),
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "_tmp_download.zip"
    with requests.get(url, headers=HEADERS, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        # SEC's zip layout is inconsistent across quarters: some archives have
        # top-level files, others nest everything under a folder -- match by
        # basename instead of assuming a flat layout.
        names = {name.rsplit("/", 1)[-1]: name for name in zf.namelist()}
        with zf.open(names["SUBMISSION.tsv"]) as f:
            submissions = pd.read_csv(f, sep="\t", usecols=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT"])
        with zf.open(names["INFOTABLE.tsv"]) as f:
            info = pd.read_csv(
                f,
                sep="\t",
                usecols=["ACCESSION_NUMBER", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "VALUE", "SSHPRNAMT"],
            )
    zip_path.unlink()

    info = info[info["TITLEOFCLASS"] == "COM"]
    merged = info.merge(submissions, on="ACCESSION_NUMBER", how="left")
    merged["PERIODOFREPORT"] = pd.to_datetime(merged["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")

    cusip_agg = (
        merged.groupby(["CUSIP", "PERIODOFREPORT"])
        .agg(
            name=("NAMEOFISSUER", lambda s: s.value_counts().idxmax() if s.notna().any() else ""),
            total_shares=("SSHPRNAMT", "sum"),
            total_value=("VALUE", "sum"),
            num_filers=("ACCESSION_NUMBER", "nunique"),
        )
        .reset_index()
        .rename(columns={"CUSIP": "cusip", "PERIODOFREPORT": "period_of_report"})
    )
    cusip_agg.to_csv(agg_path, index=False)

    holder_agg = (
        merged.groupby(["CUSIP", "CIK", "PERIODOFREPORT"])
        .agg(shares=("SSHPRNAMT", "sum"), value=("VALUE", "sum"))
        .reset_index()
        .rename(columns={"CUSIP": "cusip", "CIK": "cik", "PERIODOFREPORT": "period_of_report"})
    )
    holder_agg.to_csv(holder_path, index=False)

    return cusip_agg, holder_agg


def _resolve_cusip(company_name: str, agg: pd.DataFrame) -> str | None:
    """Picks the CUSIP whose reported name best matches `company_name`: every
    CUSIP whose most-common name contains the company name's first word, ranked
    by aggregate $ value (the real listing dominates noise by orders of
    magnitude).
    """
    token = company_name.upper().split()[0]
    candidates = agg[agg["name"].str.upper().str.contains(re.escape(token), na=False)]
    if candidates.empty:
        return None
    totals = candidates.groupby("cusip")["total_value"].sum()
    return str(totals.idxmax())


def get_institutional_history(
    ticker: str,
    company_name: str,
    quarters: int = 5,
    top_n: int = 10,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Top-`top_n` institutional holders' aggregate position for one company,
    one row per reporting quarter, most recent `quarters` periods. Columns:
    period_of_report, total_shares (sum of the top holders' shares),
    total_value, num_filers (holders counted, <= top_n), pct_change (plain
    average of each top holder's own %% change vs its prior-quarter position;
    a holder with no prior-quarter position counts as +100%%, same convention
    yfinance uses for a new position).

    Downloads and caches SEC's bulk 13F datasets to cover enough periods (large:
    ~100-400MB per quarter on first use, then just small cached aggregates).
    `datasets` lets callers reuse an already-fetched dataset URL list instead of
    re-listing SEC's page for every ticker.
    """
    urls = datasets if datasets is not None else list_available_datasets()
    columns = ["period_of_report", "total_shares", "total_value", "num_filers", "pct_change"]

    pairs = [_download_and_aggregate(u) for u in urls[: quarters + 2]]
    cusip_frames = [p[0] for p in pairs]
    holder_frames = [p[1] for p in pairs]

    tagged = []
    for i, f in enumerate(cusip_frames):
        f = f.copy()
        f["_src"] = i
        tagged.append(f)
    combined_cusip = pd.concat(tagged, ignore_index=True)
    if combined_cusip.empty:
        return pd.DataFrame(columns=columns)

    cusip = _resolve_cusip(company_name, cusip_frames[0])
    if cusip is None:
        return pd.DataFrame(columns=columns)

    # A quarter can appear in more than one dataset bucket (late amendments
    # trickling into the next bucket); keep whichever bucket saw the most
    # filers for that quarter -- the primary bucket, not the stragglers. Track
    # which bucket that was so the holder-level lookup below stays consistent
    # (duplicate (cusip, cik) rows across buckets would otherwise appear).
    company_cusip_rows = combined_cusip[combined_cusip["cusip"] == cusip]
    best_idx = company_cusip_rows.groupby("period_of_report")["num_filers"].idxmax()
    primary = company_cusip_rows.loc[best_idx].sort_values("period_of_report").tail(quarters + 1)
    periods = primary["period_of_report"].tolist()
    period_source = dict(zip(primary["period_of_report"], primary["_src"]))
    if len(periods) < 2:
        return pd.DataFrame(columns=columns)

    by_period = {}
    for p in periods:
        holders = holder_frames[period_source[p]]
        by_period[p] = holders[(holders["cusip"] == cusip) & (holders["period_of_report"] == p)].set_index("cik")

    rows = []
    for i in range(1, len(periods)):
        period = periods[i]
        current = by_period[period]
        prior = by_period[periods[i - 1]]
        top = current.nlargest(top_n, "shares")

        pct_changes = []
        for cik, row in top.iterrows():
            if cik in prior.index and prior.loc[cik, "shares"] > 0:
                pct_changes.append((row["shares"] / prior.loc[cik, "shares"] - 1) * 100)
            else:
                pct_changes.append(100.0)

        rows.append(
            {
                "period_of_report": period,
                "total_shares": top["shares"].sum(),
                "total_value": top["value"].sum(),
                "num_filers": len(top),
                "pct_change": sum(pct_changes) / len(pct_changes) if pct_changes else float("nan"),
            }
        )

    return pd.DataFrame(rows, columns=columns)
