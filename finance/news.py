"""RSS-based news feed access, for newsletters/publications relevant to the
tracked universe (starts with SemiAnalysis; the list is meant to grow).

No scraping: standard RSS already gives a title, the feed's own short
teaser, a full-content field, a link, and a date for free, no API key or
per-article cost required. `finance.summarize` optionally turns the
full-content field into a richer AI-generated summary; this module has no
opinion on that and works fine without it (the feed's own `summary` field is
always there as a fallback).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# (display name, RSS feed URL) -- add more sources here as they come up.
NEWS_SOURCES: list[tuple[str, str]] = [
    ("SemiAnalysis", "https://newsletter.semianalysis.com/feed"),
]

COLUMNS = ["source", "title", "link", "published", "author", "summary", "content"]


def _cache_path(feed_url: str) -> Path:
    digest = hashlib.sha1(feed_url.encode()).hexdigest()[:16]
    return CACHE_DIR / f"news_{digest}.csv"


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None else ""


def _parse_feed(source_name: str, raw_xml: bytes) -> pd.DataFrame:
    root = ET.fromstring(raw_xml)
    rows = []
    for item in root.findall("./channel/item"):
        rows.append(
            {
                "source": source_name,
                "title": _text(item, "title"),
                "link": _text(item, "link"),
                "published": _text(item, "pubDate"),
                "author": _text(item, "{http://purl.org/dc/elements/1.1/}creator"),
                "summary": _text(item, "description"),
                "content": _text(item, "{http://purl.org/rss/1.0/modules/content/}encoded"),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def get_news(source_name: str, feed_url: str, refresh: bool = False) -> pd.DataFrame:
    """Articles from one RSS feed, newest first. Columns: source, title,
    link, published, author, summary. Cached to disk like everything else in
    `finance.data`; pass refresh=True to force a re-fetch.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(feed_url)

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path, parse_dates=["published"])
        return df

    response = requests.get(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    df = _parse_feed(source_name, response.content)
    df = df.sort_values("published", ascending=False).reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    return df


def get_all_news(refresh: bool = False) -> pd.DataFrame:
    """Every configured source, concatenated and sorted newest first. A feed
    that fails to fetch is skipped rather than failing the whole call.
    """
    frames = []
    for name, url in NEWS_SOURCES:
        try:
            frames.append(get_news(name, url, refresh=refresh))
        except (requests.RequestException, ET.ParseError):
            continue
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).sort_values("published", ascending=False).reset_index(drop=True)
