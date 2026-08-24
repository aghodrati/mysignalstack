"""Real, dated news headlines for finance.macro's weekly/monthly narratives and finance.
earnings_preview's grounded "what to watch" read -- replaces GDELT's DOC 2.0 API entirely (used
through most of this project's early development). Confirmed live, side-by-side, across several
real queries (NVDA earnings, gold, VIX, credit spread) that GDELT was both frequently rate-limited
(a shared-IP 429, unrelated to this app's own tiny call volume -- see git history) and, even when
it worked, its keyword matching surfaced meaningfully more irrelevant noise than either replacement
below for the same topics.

Two providers, chosen per use case rather than one for everything:

- Finnhub's /company-news endpoint (fetch_finnhub_headlines) -- a real per-symbol news feed with
  genuine `from`/`to` date-range control (confirmed live: requesting an explicitly older window
  correctly returns older-dated articles, not just "now"). Used for anything with a real, liquid
  ticker/ETF behind it: individual stocks (finance.earnings_calls' preview refresh) and the macro series with a
  clean commodity/ETF proxy (finance.macro.FINNHUB_PROXY_SYMBOLS).

- GNews.io's /search endpoint (fetch_gnews_headlines) -- a real keyword-search API, also with
  genuine date-range control, for the macro series with no single clean instrument to proxy (VIX,
  credit spread, PCE, unemployment -- see finance.macro.NARRATIVE_QUERIES). Confirmed live it has a
  real but short burst rate limit (recovered within roughly 30-60s during testing) -- retried with
  backoff below, same reasoning GDELT's own retry had.

Both fetchers return the same shape: a list of {"title", "domain", "date"} dicts (date as
"YYYYMMDD", matching GDELT's old convention so select_diverse_headlines and every caller's
formatting code didn't need to change), or None (never raises) if the fetch itself failed -- a
missing API key, a network error, or a rate limit that outlasted every retry. Distinct from an
empty list, which means the fetch succeeded but genuinely found nothing this period -- callers
(refresh_macro_narrative/refresh_earnings_preview) treat None as "retry later, don't spend an LLM
call on nothing" and [] as "a real, complete (if quiet) answer worth reading."
"""

from __future__ import annotations

import datetime as dt
import os
import time

import requests

_FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
_GNEWS_URL = "https://gnews.io/api/v4/search"


def fetch_finnhub_headlines(symbol: str, days: int) -> list[dict] | None:
    """Up to 100 {title, domain, date} dicts for `symbol`'s company news over the last `days`
    days, via Finnhub's /company-news endpoint. Returns None (never raises) if FINNHUB_API_KEY
    isn't set or the request fails for any reason -- no retry here (unlike fetch_gnews_headlines):
    confirmed live Finnhub didn't rate-limit even across a burst of test calls, so there's nothing
    to back off from in practice.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        return None
    today = dt.date.today()
    params = {
        "symbol": symbol, "from": (today - dt.timedelta(days=days)).isoformat(),
        "to": today.isoformat(), "token": api_key,
    }
    try:
        response = requests.get(_FINNHUB_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, list):
        return None  # Finnhub returns a {"error": ...} dict, not a list, on a bad symbol/token
    return [
        {
            "title": item["headline"], "domain": item.get("source") or "unknown",
            "date": dt.datetime.utcfromtimestamp(item["datetime"]).strftime("%Y%m%d"),
        }
        for item in data if item.get("headline") and item.get("datetime")
    ]


def fetch_gnews_headlines(query: str, days: int) -> list[dict] | None:
    """Up to 10 {title, domain, date} dicts (GNews's free-tier max per request) matching `query`
    over the last `days` days, via GNews.io's /search endpoint. Retries with backoff on a 429 --
    see module docstring re: the real but short burst limit confirmed live. Returns None (never
    raises) if GNEWS_API_KEY isn't set, every retry is exhausted, or the request fails for any
    other reason.
    """
    api_key = os.environ.get("GNEWS_API_KEY")
    if not api_key:
        return None
    today = dt.date.today()
    params = {
        "q": query, "lang": "en", "max": 10,
        "from": (today - dt.timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z"),
        "to": today.strftime("%Y-%m-%dT23:59:59Z"), "apikey": api_key,
    }
    backoffs = (10, 30)
    data = None
    for attempt in range(len(backoffs) + 1):
        try:
            response = requests.get(_GNEWS_URL, params=params, timeout=20)
            if response.status_code == 429:
                if attempt < len(backoffs):
                    time.sleep(backoffs[attempt])
                    continue
                return None
            response.raise_for_status()
            data = response.json()
            break
        except (requests.RequestException, ValueError):
            return None
    if data is None or "articles" not in data:
        return None
    return [
        {
            "title": a["title"], "domain": (a.get("source") or {}).get("name") or "unknown",
            "date": a["publishedAt"][:10].replace("-", ""),
        }
        for a in (data.get("articles") or []) if a.get("title") and a.get("publishedAt")
    ]


def select_diverse_headlines(headlines: list[dict], limit: int, max_per_domain: int) -> list[dict]:
    """Keeps each provider's own relevance/recency order, but caps how many headlines from the
    same domain count toward `limit` -- so a wire story syndicated across a dozen domains (or a
    single outlet posting several near-duplicate pieces) can't crowd out everything else that
    happened this period.
    """
    selected: list[dict] = []
    per_domain: dict[str, int] = {}
    for h in headlines:
        if len(selected) >= limit:
            break
        if per_domain.get(h["domain"], 0) >= max_per_domain:
            continue
        selected.append(h)
        per_domain[h["domain"]] = per_domain.get(h["domain"], 0) + 1
    return selected
