"""RSS/Atom-based news feed access, for newsletters/publications relevant to
the tracked universe (starts with SemiAnalysis and Stratechery; the list is
meant to grow), plus SEC EDGAR's per-company 8-K Atom feeds (material-event
filings only -- earnings releases, guidance changes, M&A, leadership changes
-- so unlike the newsletters, zero-noise by construction).

Mostly no scraping: standard RSS already gives a title, the feed's own short
teaser, a full-content field, a link, and a date for free, no API key or
per-article cost required. The 8-K feeds are the exception: SEC's feed
only gives a one-line filing-type blurb (e.g. "Item 5.02: Departure of
Directors..."), not the filing's actual text -- that blurb is still useful
as a "something material happened" signal, just much thinner than a
newsletter article. Fetching the full filing document is a deliberately
separate, bigger lift not done here.

`fetch_full_page_text` is the one real exception to "no scraping" -- an
opt-in, per-source fallback (see finance.loop_a_config.full_page_fetch_sources)
for a source whose RSS is confirmed too thin (or empty) to classify from.
Not blanket-enabled: some sites (confirmed: EE Times) actively bot-block
this kind of fetch at the TLS/HTTP2 level, so it's only worth trying for a
source specifically confirmed to cooperate (confirmed: Next Platform).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
import trafilatura

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "news"

# (display name, RSS feed URL) -- the "This Week" tab's full browsing list, independent of which
# of these Loop A actually processes (see finance.loop_a_config -- turning a source off there for
# LLM-cost reasons no longer empties this list out too).
NEWS_SOURCES: list[tuple[str, str]] = [
    ("SemiAnalysis", "https://newsletter.semianalysis.com/feed"),
    ("Stratechery", "https://stratechery.com/feed/"),
    ("The Diff", "https://www.thediff.co/archive/rss/"),
    ("Citrini Research", "https://citriniresearch.com/feed"),
    ("Cassandra Unchained (Michael Burry)", "https://michaeljburry.substack.com/feed"),
]

# SEC requires a descriptive User-Agent identifying the requester (not a browser UA)
# -- see https://www.sec.gov/os/webmaster-faq#developers. Same header finance.xbrl and
# finance.sec13f already use for their own SEC requests.
SEC_HEADERS = {"User-Agent": "finance-hypothesis-tool ghodrati@gmail.com"}
_SEC_8K_FEED_TEMPLATE = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10&output=atom"
)
# Every get_sec_8k_news row's `source` starts with this + the ticker -- callers use it to
# recognize "the company is already known deterministically" without a schema change.
SEC_8K_SOURCE_PREFIX = "SEC 8-K: "

COLUMNS = ["source", "title", "link", "published", "author", "summary", "content"]

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _cache_path(feed_url: str) -> Path:
    digest = hashlib.sha1(feed_url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.csv"


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


def _rss_rows(source_name: str, root: ET.Element) -> list[dict]:
    return [
        {
            "source": source_name,
            "title": _text(item, "title"),
            "link": _text(item, "link"),
            "published": _text(item, "pubDate"),
            "author": _text(item, "{http://purl.org/dc/elements/1.1/}creator"),
            "summary": _text(item, "description"),
            "content": _text(item, "{http://purl.org/rss/1.0/modules/content/}encoded"),
        }
        for item in root.findall("./channel/item")
    ]


def _atom_rows(source_name: str, root: ET.Element) -> list[dict]:
    rows = []
    for entry in root.findall(f"./{_ATOM_NS}entry"):
        link_el = entry.find(f"{_ATOM_NS}link")
        rows.append(
            {
                "source": source_name,
                "title": _text(entry, f"{_ATOM_NS}title"),
                "link": link_el.get("href", "") if link_el is not None else "",
                "published": _text(entry, f"{_ATOM_NS}updated"),
                "author": "",
                "summary": _text(entry, f"{_ATOM_NS}summary"),
                "content": "",  # SEC's Atom feed has no separate full-text field -- see module docstring
            }
        )
    return rows


def _parse_feed(source_name: str, raw_xml: bytes) -> pd.DataFrame:
    root = ET.fromstring(raw_xml)
    rows = _atom_rows(source_name, root) if root.tag == f"{_ATOM_NS}feed" else _rss_rows(source_name, root)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def get_news(source_name: str, feed_url: str, refresh: bool = False, headers: dict | None = None) -> pd.DataFrame:
    """Articles from one RSS or Atom feed, newest first. Columns: source, title,
    link, published, author, summary. Cached to disk like everything else in
    `finance.data`; pass refresh=True to force a re-fetch.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(feed_url)

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["published"])
        return df

    response = requests.get(feed_url, timeout=10, headers=headers or {"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    df = _parse_feed(source_name, response.content)
    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_full_page_text(url: str, timeout: int = 15) -> str:
    """Fetches one article's own webpage and extracts its main body text via
    trafilatura -- the fallback for a source whose RSS gives too little to
    classify from (see finance.loop_a_config.full_page_fetch_sources).

    Best-effort and silent: returns "" on any failure (network error,
    timeout, no extractable content) rather than raising, so a caller falls
    back to whatever RSS already gave it, same as any other empty-article
    case -- one blocked/slow site should never take down the whole research
    update. Deliberately not retried or escalated (e.g. no headless browser
    fallback) -- a site actively bot-blocking a plain request (confirmed:
    EE Times, a TLS/HTTP2-level connection reset) should just be left off
    rather than worked around.
    """
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException:
        return ""
    return trafilatura.extract(response.text) or ""


def get_news_from_sources(
    sources: list[tuple[str, str]], refresh: bool = False, headers: dict | None = None
) -> pd.DataFrame:
    """Every (name, feed_url) in `sources`, concatenated and sorted newest
    first. A feed that fails to fetch is skipped rather than failing the
    whole call.
    """
    frames = []
    for name, url in sources:
        try:
            frames.append(get_news(name, url, refresh=refresh, headers=headers))
        except (requests.RequestException, ET.ParseError):
            continue
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("published", ascending=False).reset_index(drop=True)


def get_all_news(refresh: bool = False) -> pd.DataFrame:
    """Every configured newsletter source (NEWS_SOURCES), concatenated and
    sorted newest first.
    """
    return get_news_from_sources(NEWS_SOURCES, refresh=refresh)


def get_sec_8k_news(tickers: list[str], cik_map: dict[str, str], refresh: bool = False) -> pd.DataFrame:
    """Recent 8-K filings for every ticker with a known CIK, concatenated and
    sorted newest first. `content` is empty (see module docstring) -- callers
    fall back to `summary`, the feed's one-line filing-type blurb.

    Unlike a newsletter article, SEC's own title/summary text for a filing
    never names the company it's about (e.g. "8-K - Current report" / "Item
    5.02: Departure of..."), and this is the one case in this module where
    the caller (this function) already knows exactly which company each
    article concerns, deterministically, from which per-ticker feed it came
    from -- so the ticker is prefixed onto the title here rather than left
    for a downstream classifier to guess (and likely miss) from the text.
    """
    frames = []
    for ticker in tickers:
        cik = cik_map.get(ticker.upper())
        if cik is None:
            continue
        try:
            df = get_news(
                f"{SEC_8K_SOURCE_PREFIX}{ticker}", _SEC_8K_FEED_TEMPLATE.format(cik=cik),
                refresh=refresh, headers=SEC_HEADERS,
            )
        except (requests.RequestException, ET.ParseError):
            continue
        if not df.empty:
            df = df.copy()
            df["title"] = ticker + ": " + df["title"]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("published", ascending=False).reset_index(drop=True)
