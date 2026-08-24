"""A global macro dashboard: a handful of deterministic, LLM-free stat tiles -- level + period-
over-period change -- covering rates, credit risk, volatility, commodities, inflation, and labor,
plus (for the 10-Y Treasury yield so far) a weekly, LLM-grounded "why did this move" narrative.
Not tied to any ticker -- computed and shown once, globally, not per-ticker.

Two parts, deliberately kept separate in spirit even though they share this one file:

1. The stat tiles (MACRO_SERIES/series_tile/macro_snapshot) are pure numbers -- every value comes
   straight from FRED (official US macro series, via its public fredgraph.csv CSV export -- no API
   key required) or yfinance (commodity/volatility prices, reusing finance.data's existing
   get_prices), so there's no hallucination risk and no LLM cost here at all.

   Two cadences, matching how often each series' underlying data actually changes:
   - "weekly": 10Y yield, HY credit spread, VIX, oil/gold/silver/copper -- all daily-updating
     market data. The period-over-period comparison is always against ~7 calendar days back
     regardless of how often the dashboard itself happens to be viewed, since that's the cadence
     that's actually informative for these (checking more often wouldn't show anything new day to
     day; checking only monthly would smooth over -- and could entirely miss -- a fast risk-off
     move, which is the whole point of tracking credit spreads/VIX at all).
   - "monthly": core PCE, unemployment rate -- genuinely monthly-release government data. The
     period-over-period comparison is against the previous published reading, not a fixed day
     count.

   FRED series have real publication lag/revisions (a payrolls-style report can be revised after
   its initial release) -- fredgraph.csv always returns whatever FRED currently has on file for
   each date, i.e. the latest vintage, not the as-originally-published one. Fine for "what's the
   current macro picture" (this dashboard's only job), but note this is NOT leakage-safe the way
   Stage A/B's as_of-grounded reasoning is -- don't reuse these series to backtest a historical
   date without accounting for that.

2. The weekly narrative (NARRATIVE_QUERIES/FINNHUB_PROXY_SYMBOLS/refresh_macro_narrative) is the
   one place this module adds interpretation on top of the pure numbers above, and it's grounded
   the same way every other LLM stage in this app is: real, dated evidence handed to the model,
   never left to recall/guess from training data.

   The evidence is real news headlines from finance.news_sources -- two providers, chosen per
   series (see that module's own docstring for the full "why," confirmed live via side-by-side
   testing against GDELT, which this replaced entirely): Finnhub's per-symbol company-news feed
   (FINNHUB_PROXY_SYMBOLS) for any series with a real, liquid ticker/ETF behind it (gold->GLD,
   oil->USO, etc.), and GNews.io's keyword search (NARRATIVE_QUERIES) for the series with no clean
   single instrument to proxy (VIX, credit spread, PCE, unemployment). Headlines are title/domain/
   date only (no article body -- neither provider's search endpoint returns full text on the free
   tier, and a headline alone is enough for the LLM to judge relevance), capped per-domain
   (select_diverse_headlines) so one wire story echoed across a dozen syndicated domains can't
   crowd out everything else that happened that period.

   One real LLM call per series per week, deduped by ISO week (not a rolling 7-day window) so a
   mid-week rerun never reprocesses -- same "already on record, skip" contract every other Loop A
   stage uses.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from finance.data import get_prices
from finance.llm import complete
from finance.loop_a_config import llm_config, max_article_chars
from finance.news_sources import fetch_finnhub_headlines, fetch_gnews_headlines, select_diverse_headlines
from finance import read_state

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "macro"
NARRATIVES_DIR = Path("output/macro")
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_USER_AGENT = {"User-Agent": "Mozilla/5.0"}

# key -> (label, source, source_id, cadence, unit, transform, delta_format)
# unit: "pct" | "usd" | "num" -- how to format the level.
# transform: "level" (use the series as-is) | "yoy_pct" (year-over-year %% change -- PCE's raw
#   value is an index level, meaningless on its own; the standard "inflation rate" reading is its
#   12-months-back %% change).
# delta_format: "bps" (basis points -- how yields/spreads are conventionally quoted) | "pp"
#   (percentage points -- how a %% rate's own change is conventionally quoted) | "pct_change"
#   (%% change in a price) | "points" (VIX's own unit, already an index).
MACRO_SERIES: dict[str, dict] = {
    "10y_yield": {
        # ^TNX (CBOE's 10-Y Treasury yield index), not FRED's DGS10 -- a live market index rather
        # than a government-published series, confirmed live to run about a day fresher than FRED
        # (FRED's own additional publication step adds lag on top of the same underlying market
        # data). Same percentage units as DGS10 (e.g. 4.65), no conversion needed -- confirmed live.
        "label": "10-Y Treasury", "tag": "10-Y", "icon": "\U0001f3db️", "source": "yfinance", "source_id": "^TNX",
        "cadence": "weekly", "unit": "pct", "transform": "level", "delta_format": "bps",
        # Shown as a trend chart, not a plain tile -- the single number this week can hide whether
        # it's accelerating, plateauing, or reversing, which matters more here than for the other
        # series since 10Y is the discount rate behind every high-multiple name in the tracked
        # universe. See _tile's "history" field, only ever populated when this is set. 7 days to
        # match the card's own weekly cadence/narrative -- the chart shows exactly the window the
        # narrative is explaining, not a longer lookback that would mix in an earlier week's move.
        "chart_days": 7,
        # A companion "bigger picture" mini-chart, no narrative attached -- see _tile's
        # "history_extra"/app.py's _macro_mini_chart_card_html.
        "extra_chart_days": 30,
    },
    "credit_spread": {
        "label": "HY Credit Spread", "tag": "HY-Spread", "source": "fred", "source_id": "BAMLH0A0HYM2",
        "cadence": "weekly", "unit": "pct", "transform": "level", "delta_format": "bps",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "vix": {
        "label": "VIX", "tag": "VIX", "icon": "\U0001f628", "source": "yfinance", "source_id": "^VIX",
        "cadence": "weekly", "unit": "num", "transform": "level", "delta_format": "points",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "oil": {
        "label": "WTI Crude Oil", "tag": "Oil", "icon": "\U0001f6e2️", "source": "yfinance", "source_id": "CL=F",
        "cadence": "weekly", "unit": "usd", "transform": "level", "delta_format": "pct_change",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "gold": {
        # GC=F (COMEX gold futures), not the GLD ETF -- yfinance has no true spot forex-style
        # ticker for metals (XAUUSD=X et al. confirmed 404/delisted live), and the front-month
        # futures contract is the closest available proxy, and what "gold price" conventionally
        # means in financial media anyway. Same reasoning as oil's CL=F, which already used futures.
        "label": "Gold (Futures)", "tag": "Gold", "icon": "\U0001f947", "source": "yfinance", "source_id": "GC=F",
        "cadence": "weekly", "unit": "usd", "transform": "level", "delta_format": "pct_change",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "silver": {
        "label": "Silver (Futures)", "tag": "Silver", "icon": "\U0001f948", "source": "yfinance", "source_id": "SI=F",
        "cadence": "weekly", "unit": "usd", "transform": "level", "delta_format": "pct_change",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "copper": {
        "label": "Copper (Futures)", "tag": "Copper", "icon": "\U0001f949", "source": "yfinance", "source_id": "HG=F",
        "cadence": "weekly", "unit": "usd", "transform": "level", "delta_format": "pct_change",
        "chart_days": 7, "extra_chart_days": 30,
    },
    "pce": {
        "label": "Core PCE (YoY)", "tag": "PCE", "source": "fred", "source_id": "PCEPILFE",
        "cadence": "monthly", "unit": "pct", "transform": "yoy_pct", "delta_format": "pp",
        # 365 days -> ~12 monthly readings, same "give the trend some room to show" reasoning as
        # the weekly series' chart_days, just scaled to this series' own (much slower) cadence.
        "chart_days": 365,
    },
    "unemployment": {
        "label": "Unemployment Rate", "tag": "Unemployment", "source": "fred", "source_id": "UNRATE",
        "cadence": "monthly", "unit": "pct", "transform": "level", "delta_format": "pp",
        "chart_days": 365,
    },
    "bitcoin": {
        # Same treatment as gold/silver/copper -- moved out of config_loop_a.json's tracked
        # universe (no RSS source ever covered it, same as the metals) and into the Macro group
        # instead, sourced from the same BTC-USD yfinance ticker it used there.
        "label": "Bitcoin", "tag": "BTC", "icon": "\U0001fa99", "source": "yfinance", "source_id": "BTC-USD",
        "cadence": "weekly", "unit": "usd", "transform": "level", "delta_format": "pct_change",
        "chart_days": 7, "extra_chart_days": 30,
    },
}

# Series where "direction" is a coherent concept at all -- a literal tradeable instrument price,
# not a rate/spread/index/economic print (see the discussion behind this split: yields/spreads/VIX
# naturally read as "rising"/"falling" trend, not a long/short call on something you can't actually
# trade here). Only these five get the extra direction/confidence/trade_worthy fields below.
DIRECTIONAL_SERIES = {"oil", "gold", "silver", "copper", "bitcoin"}

# Series eligible for a *monthly* narrative in addition to the weekly one -- derived from
# extra_chart_days (MACRO_SERIES' own "bigger picture" 1-month mini-chart window) rather than a
# separately maintained list, so a series only ever gets a monthly narrative once it already has a
# 1-month chart to pair it with (see app.py's _macro_mini_chart_card_html, which now shows this
# narrative instead of being chart-only). PCE/unemployment are excluded since they're already
# monthly-cadence at the tile level -- a second "monthly narrative" on top would be redundant.
MONTHLY_NARRATIVE_SERIES = {key for key, config in MACRO_SERIES.items() if config.get("extra_chart_days")}

# Series with a real, liquid ticker/ETF standing in for them -- routed through Finnhub's
# per-symbol company-news feed (finance.news_sources.fetch_finnhub_headlines) rather than keyword
# search. Confirmed live (side-by-side against GDELT and GNews.io, for gold/NVDA/VIX/credit-spread)
# that a dedicated company/ETF news feed gives both higher volume and higher relevance than
# guessing at search terms whenever a clean instrument actually exists -- e.g. Finnhub's GLD
# coverage surfaced concrete technical levels and named analyst calls (Ray Dalio, Goldman's price
# target) that a keyword search on "gold price" never found.
FINNHUB_PROXY_SYMBOLS: dict[str, str] = {
    "10y_yield": "IEF",   # iShares 7-10 Year Treasury Bond ETF
    "oil": "USO",         # United States Oil Fund
    "gold": "GLD",        # SPDR Gold Shares
    "silver": "SLV",      # iShares Silver Trust
    "copper": "CPER",     # United States Copper Index Fund
    "bitcoin": "BITO",    # ProShares Bitcoin Strategy ETF
}

# Series with no single clean instrument to proxy (VIX/credit spread/PCE/unemployment are
# concepts, not something with one ticker behind it -- confirmed live that Finnhub's own company-
# news endpoint returns nothing at all for index tickers like ^VIX/^TNX, and an ETF proxy like HYG
# returns news about the ETF issuer's own ticker mentions, not the credit-spread story). Routed
# through GNews.io's keyword search instead (finance.news_sources.fetch_gnews_headlines). Kept as
# specific, quoted multi-word phrases rather than single generic buzzwords -- confirmed live that a
# looser query (bare "market volatility", "risk appetite") pulls in a lot of only-tangentially-
# related noise (regional equity selloffs, unrelated Bitcoin rallies) that quoted, more specific
# phrases mostly avoid.
NARRATIVE_QUERIES: dict[str, str] = {
    "vix": '"VIX" OR "Cboe Volatility Index" OR "volatility index" OR "market volatility spike"',
    "credit_spread": '"credit spread" OR "high-yield spread" OR "junk bond spread" OR "high yield bond market"',
    "pce": '"PCE inflation" OR "core PCE" OR "personal consumption expenditures" OR "Fed inflation gauge"',
    "unemployment": '"unemployment rate" OR "jobs report" OR "nonfarm payrolls" OR "labor market"',
}

# Every series with *some* narrative source configured, regardless of which provider -- the
# membership check app.py and refresh_macro_narrative actually care about ("does this series get a
# weekly narrative at all"), since which of the two dicts above a series lives in is an
# implementation detail of *how* it's grounded, not *whether* it is.
NARRATIVE_SERIES = set(FINNHUB_PROXY_SYMBOLS) | set(NARRATIVE_QUERIES)


# ---------------------------------------------------------------------------
# Stat tiles -- pure numbers, no LLM.
# ---------------------------------------------------------------------------


def _cache_path(source_id: str) -> Path:
    return CACHE_DIR / f"{source_id.replace('=', '_').replace('^', '')}.csv"


def _fetch_fred_series(source_id: str, refresh: bool) -> pd.Series:
    """Full history of a FRED series, date-indexed -- FRED marks missing observations (e.g. a
    holiday on a daily series) with "." rather than omitting the row, so those are dropped here
    rather than left to become a NaN surprise downstream. Returns an empty series (never raises)
    on any network failure -- same soft-failure contract as finance.news.fetch_full_page_text --
    falling back to a stale cache if one exists rather than losing the tile entirely for what's
    often a transient timeout.

    Uses urllib rather than requests/urllib3 -- empirically, requests hangs for 20-30s+ against
    fred.stlouisfed.org's Akamai-fronted endpoint (a TLS-handshake-fingerprinting quirk, most
    likely) while urllib (and curl) connect in well under a second for the exact same URL,
    confirmed directly during development. requests is used everywhere else in this codebase, but
    this one host is a specific, verified exception.
    """
    cache_path = _cache_path(source_id)
    if cache_path.exists() and not refresh:
        raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return raw["value"]

    url = f"{_FRED_CSV_URL}?{urllib.parse.urlencode({'id': source_id})}"
    try:
        request = urllib.request.Request(url, headers=_USER_AGENT)
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        if cache_path.exists():
            raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return raw["value"]
        return pd.Series(dtype=float)
    df = pd.read_csv(io.StringIO(text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df["value"]


def _fetch_yfinance_series(source_id: str, refresh: bool) -> pd.Series:
    """Full history of a yfinance-sourced series (commodity/volatility price), date-indexed --
    reuses finance.data.get_prices' own disk cache, so this only adds the macro cache dir for
    FRED series above; nothing new to invalidate for yfinance ones. Returns an empty series
    (never raises) on any fetch failure, same soft-failure contract as _fetch_fred_series.
    """
    start = (dt.date.today() - dt.timedelta(days=400)).isoformat()
    try:
        prices = get_prices([source_id], start=start, refresh=refresh)
    except Exception:
        return pd.Series(dtype=float)
    if source_id not in prices.columns:
        return pd.Series(dtype=float)
    return prices[source_id].dropna()


def _value_on_or_before(series: pd.Series, target: pd.Timestamp) -> float | None:
    eligible = series[series.index <= target]
    return float(eligible.iloc[-1]) if not eligible.empty else None


def _level_series(config: dict, refresh: bool) -> pd.Series:
    if config["source"] == "fred":
        raw = _fetch_fred_series(config["source_id"], refresh)
    else:
        raw = _fetch_yfinance_series(config["source_id"], refresh)
    if config["transform"] == "yoy_pct":
        return ((raw / raw.shift(12) - 1) * 100).dropna()
    return raw


def _tile(key: str, config: dict, refresh: bool) -> dict | None:
    """One series' current level + its period-over-period change, or None if there's no usable
    data at all (a fetch failure or an empty series -- soft failure, same as the rest of Loop A,
    so one broken series doesn't take down the whole dashboard).
    """
    series = _level_series(config, refresh)
    if series.empty:
        return None

    latest_date = series.index[-1]
    latest_value = float(series.iloc[-1])
    if config["cadence"] == "weekly":
        prior_value = _value_on_or_before(series, latest_date - pd.Timedelta(days=7))
    else:
        prior_value = float(series.iloc[-2]) if len(series) >= 2 else None

    delta = (latest_value - prior_value) if prior_value is not None else None
    if delta is not None and config["delta_format"] == "pct_change" and prior_value:
        delta = (latest_value / prior_value - 1) * 100

    tile = {
        "key": key, "label": config["label"], "tag": config.get("tag", config["label"]),
        "icon": config.get("icon", ""),
        "cadence": config["cadence"], "unit": config["unit"],
        "delta_format": config["delta_format"], "value": latest_value, "delta": delta,
        "as_of": latest_date.date().isoformat(),
    }
    chart_days = config.get("chart_days")
    if chart_days:
        tile["history"] = _history_pairs(series, latest_date, chart_days)
    # A second, longer lookback for a companion "bigger picture" mini-chart (no narrative -- see
    # app.py's _macro_mini_chart_card_html) -- kept separate from "history" above rather than just
    # widening it, since the weekly narrative is grounded in exactly the "history" window and a
    # longer one here would visually mismatch a chart with what the narrative text actually covers.
    extra_chart_days = config.get("extra_chart_days")
    if extra_chart_days:
        tile["history_extra"] = _history_pairs(series, latest_date, extra_chart_days)
    return tile


def _history_pairs(series: pd.Series, latest_date: pd.Timestamp, days: int) -> list[list]:
    """[date_str, value] pairs (not {"date":..., "value":...} dicts) for every point in `series`
    newer than `latest_date - days` -- compact on purpose, since this gets persisted verbatim into
    a narrative snapshot's JSON (finance.macro.refresh_macro_narrative): dropping the two repeated
    key names per point roughly halves the stored size for a multi-week history with no loss of
    information (position 0/1 is unambiguous once documented here).
    """
    cutoff = latest_date - pd.Timedelta(days=days)
    recent = series[series.index > cutoff]
    return [[d.date().isoformat(), float(v)] for d, v in recent.items()]


def _value_delta_from_window(history: list[list], delta_format: str) -> tuple[float | None, float | None]:
    """Latest value + change-over-the-window from a [date, value] pairs list (e.g. a tile's
    "history_extra") -- same delta-computation rule as _tile above (raw difference, except
    "pct_change" series which compare as a %% change), just applied across the full window instead
    of a fixed 7-day lookback. Used for the monthly narrative's value/delta, since a tile's own
    "value"/"delta" are always the *weekly* comparison (see MACRO_SERIES' cadence handling in
    _tile) and would misrepresent a full month's move.
    """
    if not history:
        return None, None
    latest_value = history[-1][1]
    prior_value = history[0][1] if len(history) > 1 else None
    delta = (latest_value - prior_value) if prior_value is not None else None
    if delta is not None and delta_format == "pct_change" and prior_value:
        delta = (latest_value / prior_value - 1) * 100
    return latest_value, delta


def series_tile(key: str, refresh: bool = False) -> dict | None:
    """One series' tile by key (see MACRO_SERIES), or None if it's unconfigured or its fetch
    failed. Public single-series accessor for callers (refresh_macro_narrative below) that only
    need one series' current value/delta rather than the whole dashboard.
    """
    config = MACRO_SERIES.get(key)
    return _tile(key, config, refresh) if config is not None else None


def macro_snapshot(refresh: bool = False) -> list[dict]:
    """Every configured macro tile (see MACRO_SERIES), in insertion order -- skips a series
    entirely (rather than raising) if its fetch fails or returns no usable data.
    """
    tiles = []
    for key, config in MACRO_SERIES.items():
        tile = _tile(key, config, refresh)
        if tile is not None:
            tiles.append(tile)
    return tiles


# ---------------------------------------------------------------------------
# Weekly narrative -- the one part of this module that costs an LLM call.
# ---------------------------------------------------------------------------


def _fetch_narrative_headlines(series_key: str, days: int) -> list[dict] | None:
    """Routes to whichever of finance.news_sources' two providers `series_key` uses -- Finnhub
    (FINNHUB_PROXY_SYMBOLS) or GNews.io (NARRATIVE_QUERIES), see the module docstring and each
    dict's own comment for why. Same None-vs-[] contract as both underlying fetchers: None means
    the fetch itself failed (network error, missing API key, rate limit that outlasted every
    retry) -- distinct from an empty list, which means the fetch succeeded but genuinely found
    nothing this period.
    """
    if series_key in FINNHUB_PROXY_SYMBOLS:
        return fetch_finnhub_headlines(FINNHUB_PROXY_SYMBOLS[series_key], days)
    if series_key in NARRATIVE_QUERIES:
        return fetch_gnews_headlines(NARRATIVE_QUERIES[series_key], days)
    return None


def _week_key(as_of: dt.date) -> str:
    iso = as_of.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _period_group_key(as_of: dt.date, period: str) -> str:
    """Which "bucket" `as_of` falls into for storage purposes -- ISO week for period="week",
    calendar month for period="month". Used only to decide whether a fresh narrative replaces the
    most recent stored one of the same period (still within the same bucket) or starts a new
    history row (a new bucket) -- see _upsert_narrative. Not used for skipping work: both periods
    now always regenerate a fresh LLM read on every refresh_macro_narrative call.
    """
    return _week_key(as_of) if period == "week" else f"{as_of.year}-{as_of.month:02d}"


_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _truncate(text: str) -> str:
    limit = max_article_chars()
    return text[:limit] if limit is not None else text


def macro_narrative_snapshot(
    label: str, value: float, delta: float | None, unit: str, headlines: list[dict], as_of: dt.date,
    include_direction: bool = False, period: str = "week",
) -> dict | None:
    """One LLM call: a 2-3 sentence explanation of this period's move in `label`, grounded in
    `headlines` (real, dated results from finance.news_sources -- see module docstring) -- honestly says there's no
    clear single driver rather than forcing a narrative onto an unrelated headline, same honesty
    contract finance.earnings_calls/finance.fundamentals already use for a weak/absent signal.

    `period` is "week" or "month" -- purely wording (the caller already picked headlines/value/delta
    for the right window; this just makes the prompt/response describe them accurately).

    `include_direction` (only ever True for DIRECTIONAL_SERIES -- see that set's comment) also asks
    for a bullish/bearish/neutral price lean, a confidence in it, and whether there's enough signal
    to have a view at all. Display-only -- no expected_return_pct/expected_horizon_days, since
    nothing downstream (portfolio, thesis) consumes this; a fabricated-looking magnitude/timeframe
    would just be noise without that.

    Never raises for a parse failure (returns None); raises RateLimited if every model is out of
    quota, same as every other Stage call in this app.
    """
    period_word = "week" if period == "week" else "month"
    value_str = f"{value:.2f}{unit}"
    delta_str = f"{delta:+.2f}{unit}" if delta is not None else "n/a (no prior reading on record)"
    headlines_text = (
        "\n".join(f"- [{h['date']}] ({h['domain']}) {h['title']}" for h in headlines)
        if headlines else f"(no headlines found for this query this {period_word})"
    )
    direction_window = "coming week or two" if period == "week" else "coming month or so"
    direction_instructions = (
        "\n\nAlso give a directional read: \"direction\" is \"bullish\", \"bearish\", or \"neutral\" "
        f"for the price over the {direction_window}; \"confidence\" is \"low\", \"medium\", or "
        "\"high\", reflecting how strongly the headlines above actually support that lean (not how "
        "large the move might be); \"trade_worthy\" is true only if there's a genuine, specific "
        f"signal here, false if it's a quiet/noisy {period_word} where \"neutral\" would just mean "
        "\"no real view\" rather than an actual balanced-forces call."
        if include_direction else ""
    )
    direction_json = (
        ', "direction": "bullish|bearish|neutral", "confidence": "low|medium|high", '
        '"trade_worthy": true/false'
        if include_direction else ""
    )
    prompt = (
        f"You are a macro analyst. {label} is currently {value_str}, a change of {delta_str} over "
        f"the past {period_word}, as of {as_of.isoformat()}.\n\n"
        f"Below are the most relevant real news headlines from the past {period_word}:\n"
        f"{_truncate(headlines_text)}\n\n"
        f"In 2-3 sentences, explain what most plausibly drove this {period_word}'s move, citing "
        f"specific headlines above by their content if they plausibly explain it. If nothing above "
        f"offers a clear, specific driver, say so honestly (e.g. \"no clear single catalyst this "
        f"{period_word} -- likely broad positioning/drift\") rather than forcing a narrative onto an "
        f"unrelated headline -- an honest \"unclear\" is more useful than a fabricated-sounding "
        f"explanation.{direction_instructions}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary): "
        f'{{"narrative": "...", "has_clear_driver": true/false{direction_json}}}'
    )

    raw = complete(prompt, max_tokens=400, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or "narrative" not in data:
        return None
    result = {
        "narrative": str(data["narrative"]),
        "has_clear_driver": bool(data.get("has_clear_driver", False)),
    }
    if include_direction:
        direction = data.get("direction")
        result["direction"] = direction if direction in ("bullish", "bearish", "neutral") else "neutral"
        confidence = data.get("confidence")
        result["confidence"] = confidence if confidence in ("low", "medium", "high") else "low"
        result["trade_worthy"] = bool(data.get("trade_worthy", False))
    return result


def _narrative_path(series_key: str) -> Path:
    return NARRATIVES_DIR / f"{series_key}.json"


def load_narrative_history(series_key: str) -> list[dict]:
    """Every weekly narrative snapshot ever stored for `series_key`, oldest first."""
    path = _narrative_path(series_key)
    return json.loads(path.read_text()) if path.exists() else []


def latest_narrative(series_key: str, period: str = "week") -> dict | None:
    """Most recent stored narrative for `series_key`, filtered to `period` ("week" or "month") --
    a series' narratives file can hold both kinds interleaved (see refresh_macro_narrative), each
    tagged with a "period" field. Entries predating this field default to "week" (every narrative
    was weekly-only before monthly narratives existed), so old data keeps working with no migration.
    """
    history = [e for e in load_narrative_history(series_key) if e.get("period", "week") == period]
    return history[-1] if history else None


def _upsert_narrative(series_key: str, event: dict, period: str, as_of: dt.date) -> None:
    """Stores `event` in `series_key`'s narratives file: appends a new history row only if `as_of`
    falls in a different bucket (ISO week for period="week", calendar month for period="month" --
    see _period_group_key) than the most recent stored entry of the same period type; otherwise
    overwrites that entry in place.

    This is what lets refresh_macro_narrative regenerate a fresh LLM read on *every* call -- no
    skip-based dedup for either period anymore -- while storage still caps out at roughly one row
    per week/month: several refreshes within the same week/month just keep updating that one
    row's content (each one a strictly more current read than the last), and only crossing into a
    new week/month starts a genuinely new history entry worth keeping around.

    Whenever this overwrites an existing row in place (the in-bucket branch below), also marks that
    card unread again (finance.read_state) -- its id (app.py's _macro_weekly_card_id/
    _macro_mini_card_id) is keyed off this event's own "date", so if a same-day refresh doesn't
    change that date, the id is identical to the one the user may have already read; without this
    the fresh content would silently stay hidden behind an old read-mark. A brand-new row (a new
    week/month bucket, or a same-bucket refresh with an advanced date) gets a new id automatically
    and needs no such nudge -- it was never marked read to begin with.
    """
    path = _narrative_path(series_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_narrative_history(series_key)
    new_bucket = _period_group_key(as_of, period)
    last_idx = next(
        (i for i in range(len(history) - 1, -1, -1) if history[i].get("period", "week") == period), None,
    )
    serialized = json.loads(json.dumps(event, default=str))
    if last_idx is not None:
        last_date = dt.date.fromisoformat(history[last_idx]["date"])
        if _period_group_key(last_date, period) == new_bucket:
            history[last_idx] = serialized
            path.write_text(json.dumps(history, indent=2))
            kind = "macro_week" if period == "week" else "macro_month"
            read_state.mark_unread(read_state.CURRENT_USER, read_state.card_id(series_key, kind, serialized["date"]))
            return
    history.append(serialized)
    path.write_text(json.dumps(history, indent=2))


def refresh_macro_narrative(series_key: str, as_of: dt.date, period: str = "week") -> dict:
    """The one entry point every caller needs: always regenerates a fresh LLM+news read for
    `series_key`/`period` (no skip-based dedup for either period -- every call costs a real LLM
    call, by design), pulling the series' current tile (series_tile above), fetching+filtering the
    grounding headlines, running the LLM read, and persisting the result via _upsert_narrative.

    `period` is "week" (every series in NARRATIVE_SERIES) or "month" (only
    MONTHLY_NARRATIVE_SERIES -- series with a 1-month mini-chart to ground it against). Both are
    rolling reads as-of `as_of` ("this week"/"last 30 days"), not calendar-locked -- what IS
    calendar-bucketed is storage: _upsert_narrative only starts a new history row when `as_of`
    crosses into a new ISO week (period="week") or calendar month (period="month") since the last
    stored entry of that type; multiple refreshes within the same bucket just update that one row
    in place, so calling this repeatedly in the same week/month costs LLM calls but not JSON
    clutter. Both kinds live in the same output/macro/{series_key}.json file, each entry tagged
    "period": "week"/"month" -- see latest_narrative's filtering.

    Always returns a dict with a "status" key, never bare None -- callers (run_loop_a.py) need to
    tell "the news source is unreachable right now" apart from a genuine LLM failure apart from a
    real success, rather than all three collapsing into an equally uninformative None. "status" is
    one of:
      "generated"       -- a new/updated narrative was produced and stored; every other event field
                            (see below) is also present.
      "unknown_series"  -- series_key isn't in NARRATIVE_SERIES.
      "no_monthly_series" -- period="month" but series_key isn't in MONTHLY_NARRATIVE_SERIES.
      "no_tile"         -- the underlying finance.macro series tile (or, for period="month", its
                            30-day history_extra window) couldn't be fetched.
      "news_unavailable" -- the Finnhub/GNews.io fetch itself failed (missing API key, rate-limited,
                            or a network error that outlasted every retry -- see
                            finance.news_sources) -- no LLM call was made, nothing was stored, and
                            this can be retried anytime.
      "llm_failed"      -- the LLM call ran but produced nothing usable.
    Propagates RateLimited, same as every other Stage call.
    """
    if series_key not in NARRATIVE_SERIES:
        return {"status": "unknown_series"}
    if period == "month" and series_key not in MONTHLY_NARRATIVE_SERIES:
        return {"status": "no_monthly_series"}

    tile = series_tile(series_key)
    if tile is None:
        return {"status": "no_tile"}

    if period == "week":
        value, delta, history = tile["value"], tile["delta"], tile.get("history", [])
    else:
        history = tile.get("history_extra", [])
        value, delta = _value_delta_from_window(history, tile["delta_format"])
        if value is None:
            return {"status": "no_tile"}

    unit_suffix = {"pct": "%", "usd": "", "num": ""}[tile["unit"]]
    days = 7 if period == "week" else 30
    headline_limit, max_per_domain = (25, 2) if period == "week" else (35, 3)
    raw_headlines = _fetch_narrative_headlines(series_key, days)
    if raw_headlines is None:
        return {"status": "news_unavailable"}  # no LLM call, no storage, free to retry
    headlines = select_diverse_headlines(raw_headlines, headline_limit, max_per_domain)

    include_direction = series_key in DIRECTIONAL_SERIES
    snapshot = macro_narrative_snapshot(
        tile["label"], value, delta, unit_suffix, headlines, as_of,
        include_direction=include_direction, period=period,
    )
    if snapshot is None:
        return {"status": "llm_failed"}

    event = {
        "status": "generated", "period": period,
        "date": as_of.isoformat(), "value": value, "delta": delta,
        # The actual daily numbers behind this period's chart, not just the single value/delta --
        # only present for a series configured with chart_days/extra_chart_days (see MACRO_SERIES).
        # Storing it here means the record is self-contained (numbers + the narrative explaining
        # them) and survives independent of finance.data/FRED's own rolling cache, which has no
        # obligation to keep this exact window around.
        "history": history,
        "narrative": snapshot["narrative"], "has_clear_driver": snapshot["has_clear_driver"],
        "headlines_used": headlines, "fetch_succeeded": True,
    }
    if period == "week":
        event["week"] = _week_key(as_of)
    if include_direction:
        event["direction"] = snapshot["direction"]
        event["confidence"] = snapshot["confidence"]
        event["trade_worthy"] = snapshot["trade_worthy"]
    _upsert_narrative(series_key, {k: v for k, v in event.items() if k != "status"}, period, as_of)
    return event
