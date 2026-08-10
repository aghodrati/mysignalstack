"""Price data access, backed by a local CSV cache so repeated backtests don't re-hit yfinance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _cache_key(tickers: list[str], start: str, end: str, interval: str, label: str = "prices") -> Path:
    raw = f"{sorted(tickers)}|{start}|{end}|{interval}|{label}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{label}_{digest}.csv"


def _download(tickers: list[str], start: str, end: str | None, interval: str) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )


def _extract_field(raw: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame({t: raw[t][field] for t in tickers if t in raw.columns.get_level_values(0)})
    return raw[[field]].rename(columns={field: tickers[0]})


def get_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices, wide format (columns = tickers, index = date).

    Cached to disk by (tickers, start, end, interval) so re-running a hypothesis
    doesn't repeatedly hit the network. Pass refresh=True to force a re-download.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(tickers, start, end or "", interval)

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = _download(tickers, start, end, interval)
    prices = _extract_field(raw, tickers, "Close").dropna(how="all")
    prices.to_csv(cache_path)
    return prices


def get_live_price(tickers: list[str]) -> dict[str, float]:
    """Most recent traded price per ticker, right now -- deliberately
    uncached (unlike everything else in this module), since the whole point
    is a fresh number, not yesterday's close. Used when a rule needs to fill
    a trade *during* today's session, before today's daily close exists yet.
    """
    raw = yf.download(tickers, period="1d", interval="1m", auto_adjust=True, progress=False, group_by="ticker", threads=False)
    closes = _extract_field(raw, tickers, "Close").dropna(how="all")
    if closes.empty:
        return {}
    return closes.ffill().iloc[-1].dropna().to_dict()


def get_volume(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """Daily share volume, wide format (columns = tickers, index = date). Same
    caching scheme as `get_prices`; a separate cache label since it's a
    different field from the same underlying download.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(tickers, start, end or "", interval, label="volume")

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    raw = _download(tickers, start, end, interval)
    volume = _extract_field(raw, tickers, "Volume").dropna(how="all")
    volume.to_csv(cache_path)
    return volume


def get_intraday_closes(
    tickers: list[str],
    period: str = "1y",
    interval: str = "60m",
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted intraday close prices, wide format, tz-aware timestamps (exchange
    local time). Yahoo limits how far back intraday bars go: 60m bars are
    available for up to ~2 years, finer intervals (5m, 15m, ...) only ~60 days.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key(tickers, period, "", interval, label="intraday")

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0)
        # A full year of intraday timestamps spans a DST change, so the raw CSV
        # column has mixed UTC offsets (-04:00 / -05:00) that pd.read_csv can't
        # parse into a single DatetimeIndex on its own. Parse as UTC first (each
        # row keeps its own offset correctly), then convert back to exchange
        # local time so wall-clock trading hours line up across the whole year.
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        return df

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    closes = _extract_field(raw, tickers, "Close").dropna(how="all")
    closes.to_csv(cache_path)
    return closes


def get_earnings_history(ticker: str, limit: int = 12, refresh: bool = False) -> pd.DataFrame:
    """EPS estimate/actual/surprise% for one ticker, oldest first. Columns:
    earnings_date (tz-naive), eps_estimate, reported_eps, surprise_pct.

    Includes the next scheduled (not-yet-reported) earnings date if yfinance
    returns one — that row has a NaN reported_eps/surprise_pct, so callers that
    only want historical, reported quarters should
    `.dropna(subset=["reported_eps", "surprise_pct"])`.

    This is a per-ticker scrape (yfinance has no batched earnings endpoint), so
    it's cached aggressively — each ticker is its own network round-trip.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], str(limit), "", "earnings", label="earnings")

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=["earnings_date"])

    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    columns = ["earnings_date", "eps_estimate", "reported_eps", "surprise_pct"]
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(cache_path)
        return empty

    df = raw.rename(
        columns={"EPS Estimate": "eps_estimate", "Reported EPS": "reported_eps", "Surprise(%)": "surprise_pct"}
    )
    df = df.reset_index().rename(columns={"Earnings Date": "earnings_date"})
    df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.tz_localize(None)
    df = df.sort_values("earnings_date").reset_index(drop=True)
    df.to_csv(cache_path)
    return df


def get_insider_transactions(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Individual insider transaction filings for one ticker, most recent first.
    Columns: insider, position, transaction, start_date, shares, value, ownership.

    `transaction` is yfinance's free-text description (e.g. "Sale at price X per
    share", "Stock Gift"). `value` and `shares` may be NaN for non-market
    transactions (grants, gifts). This is a per-ticker scrape, so it's cached
    aggressively like `get_earnings_history`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "insider", label="insider")

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=["start_date"])

    raw = yf.Ticker(ticker).insider_transactions
    columns = ["insider", "position", "transaction", "start_date", "shares", "value", "ownership"]
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(cache_path)
        return empty

    df = raw.rename(
        columns={
            "Insider": "insider",
            "Position": "position",
            "Text": "transaction",
            "Start Date": "start_date",
            "Shares": "shares",
            "Value": "value",
            "Ownership": "ownership",
        }
    )[columns]
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.tz_localize(None)
    df = df.sort_values("start_date", ascending=False).reset_index(drop=True)
    df.to_csv(cache_path)
    return df


def get_institutional_ownership(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Institutional + mutual fund holders for one ticker, combined (13F-derived,
    ~10 largest of each, quarterly). Columns: holder, holder_type ("institution" or
    "fund"), date_reported, pct_held (% of shares outstanding held by that holder),
    shares, value, pct_change (% change in that holder's position since the prior
    quarter's filing).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "ownership", label="ownership")
    columns = ["holder", "holder_type", "date_reported", "pct_held", "shares", "value", "pct_change"]

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=["date_reported"])

    tk = yf.Ticker(ticker)
    rename = {
        "Holder": "holder",
        "Date Reported": "date_reported",
        "pctHeld": "pct_held",
        "Shares": "shares",
        "Value": "value",
        "pctChange": "pct_change",
    }
    frames = []
    inst = tk.institutional_holders
    if inst is not None and not inst.empty:
        inst = inst.rename(columns=rename)
        inst["holder_type"] = "institution"
        frames.append(inst[columns])
    funds = tk.mutualfund_holders
    if funds is not None and not funds.empty:
        funds = funds.rename(columns=rename)
        funds["holder_type"] = "fund"
        frames.append(funds[columns])

    if not frames:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(cache_path)
        return empty

    df = pd.concat(frames, ignore_index=True)
    df["date_reported"] = pd.to_datetime(df["date_reported"]).dt.tz_localize(None)
    df.to_csv(cache_path)
    return df


def get_major_holders(ticker: str, refresh: bool = False) -> pd.Series:
    """insidersPercentHeld, institutionsPercentHeld, institutionsFloatPercentHeld,
    institutionsCount for one ticker, as a Series indexed by those breakdown names.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "major", label="major_holders")

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0)["value"]

    raw = yf.Ticker(ticker).major_holders
    series = raw["Value"] if raw is not None and not raw.empty else pd.Series(dtype=float)
    series.to_frame("value").to_csv(cache_path)
    return series


def get_stock_info(ticker: str, refresh: bool = False) -> dict:
    """Snapshot of yfinance's `Ticker.info` for one ticker (valuation multiples,
    growth/margin figures, price-target and beta fields, etc.) -- whatever
    fields yfinance returns, cached as-is per ticker. Callers should use
    `dict.get(key)` since coverage varies by ticker (e.g. crypto/ETFs lack
    most fundamental fields).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "info", label="info")

    if cache_path.exists() and not refresh:
        return pd.read_json(cache_path, typ="series").to_dict()

    info = yf.Ticker(ticker).info or {}
    pd.Series(info).to_json(cache_path)
    return info


def get_recommendations_trend(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Analyst recommendation counts (strongBuy/buy/hold/sell/strongSell) for the
    current month and the prior 3 months, via yfinance. Columns: period ("0m",
    "-1m", "-2m", "-3m"), strongBuy, buy, hold, sell, strongSell.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "rec_trend", label="rec_trend")
    columns = ["period", "strongBuy", "buy", "hold", "sell", "strongSell"]

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0)

    raw = yf.Ticker(ticker).recommendations
    df = raw[columns] if raw is not None and not raw.empty else pd.DataFrame(columns=columns)
    df.to_csv(cache_path)
    return df


def get_upgrades_downgrades(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Dated analyst upgrade/downgrade actions for one ticker, via yfinance.
    Columns: grade_date, firm, to_grade, from_grade, action ("up", "down",
    "init", "main", "reit"), current_price_target (0 if not given with that
    action).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "upgrades", label="upgrades")
    columns = ["grade_date", "firm", "to_grade", "from_grade", "action", "current_price_target"]

    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col=0, parse_dates=["grade_date"])

    raw = yf.Ticker(ticker).upgrades_downgrades
    if raw is None or raw.empty:
        empty = pd.DataFrame(columns=columns)
        empty.to_csv(cache_path)
        return empty

    df = raw.reset_index().rename(
        columns={
            "GradeDate": "grade_date",
            "Firm": "firm",
            "ToGrade": "to_grade",
            "FromGrade": "from_grade",
            "Action": "action",
            "currentPriceTarget": "current_price_target",
        }
    )
    if "current_price_target" not in df.columns:
        df["current_price_target"] = 0.0
    df = df[columns]
    df["grade_date"] = pd.to_datetime(df["grade_date"]).dt.tz_localize(None)
    df.to_csv(cache_path)
    return df


def get_splits(ticker: str, refresh: bool = False) -> pd.Series:
    """Dated stock-split ratios for one ticker (e.g. 10.0 for a 10-for-1
    split), via yfinance. Empty Series if it's never split.

    Needed because `get_prices`'s split-adjusted series bakes in *every*
    split Yahoo knows about (including ones after a given historical date) --
    so a nominal-dollar figure from a source that isn't split-adjusted (raw
    reported EPS, an analyst's price target at the time) needs to be scaled
    by the split ratios that happened *after* that figure's own date to be
    comparable to a price pulled from `get_prices` for the same date.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_key([ticker], "", "", "splits", label="splits")

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df["ratio"] if not df.empty else pd.Series(dtype=float)

    splits = yf.Ticker(ticker).splits
    if splits is None or splits.empty:
        pd.DataFrame(columns=["ratio"]).to_csv(cache_path)
        return pd.Series(dtype=float)

    splits = splits.copy()
    splits.index = pd.to_datetime(splits.index).tz_localize(None)
    splits.name = "ratio"
    splits.to_frame().to_csv(cache_path)
    return splits


def get_open_close(
    tickers: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adjusted open and close prices (wide format) for the same tickers/window —
    for decomposing a day into its overnight (prev close -> open) and intraday
    (open -> close) components. Cached separately from `get_prices`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    open_path = _cache_key(tickers, start, end or "", interval, label="open")
    close_path = _cache_key(tickers, start, end or "", interval, label="close")

    if open_path.exists() and close_path.exists() and not refresh:
        opens = pd.read_csv(open_path, index_col=0, parse_dates=True)
        closes = pd.read_csv(close_path, index_col=0, parse_dates=True)
        return opens, closes

    raw = _download(tickers, start, end, interval)
    opens = _extract_field(raw, tickers, "Open").dropna(how="all")
    closes = _extract_field(raw, tickers, "Close").dropna(how="all")
    opens.to_csv(open_path)
    closes.to_csv(close_path)
    return opens, closes
