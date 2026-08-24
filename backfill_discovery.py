"""TEMPORARY, one-off script -- delete after running.

Backfills discovery signal onto every article already sitting in cache/thesis/events.json whose
stored event predates that field existing. Two separate passes, since Loop A itself extracts
discovery candidates two different ways depending on which branch an article took:

1. Tracked-company branch (entry["event"] is set): backfills finance.newsloop's
   "other_companies_mentioned" (see _sanitize_other_companies' own docstring) via a standalone
   call, same shape as Stage A's own field.
2. No-tracked-company/theme branch (entry["event"] is None): these articles never got a discovery
   pass at all historically (themes/extract_themes didn't exist yet, or predate its "companies"
   field) -- backfilled via finance.newsloop.extract_themes itself (not a duplicated prompt),
   so this pass always matches whatever extract_themes currently asks for.

Neither pass re-runs full Stage A classification (that would re-spend a full classification call
and could shift event/companies/importance from whatever was already used to extract claims) --
both just reuse the article text already stored in events.json.

Writes the result into event["other_companies_mentioned"] (pass 1) or a top-level
entry["discovery_checked"] flag (pass 2, since a theme-branch entry has no "event" dict to write
into) -- both double as "already checked" markers, so a second run of this script skips whatever
the first run already covered.

Usage:
    python backfill_discovery.py [--limit N]

Saves to events.json after every processed article (not batched) so a rate-limit stop or Ctrl-C
partway through loses no progress -- just re-run the script again afterward to pick up where it
left off (already-checked articles are skipped).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from finance.llm import RateLimited, complete
from finance.loop_a_config import llm_config, tracked_universe
from finance.newsloop import (
    EVENTS_CACHE_PATH,
    TYPE_COMPANY,
    TYPE_THEME,
    _append_discovery_candidates,
    _extract_json,
    _sanitize_other_companies,
    _truncate_article,
    extract_themes,
)


def _display_names() -> dict[str, str]:
    return {ticker: name for ticker, name in tracked_universe().items()}


def _find_other_companies(title: str, text: str, display_names: dict[str, str]) -> list[dict] | None:
    tracked_display = ", ".join(f"{name} ({ticker})" for ticker, name in sorted(display_names.items()))
    prompt = (
        f"You are a financial news classifier.\n\n"
        f"Companies I track (name (ticker)): {tracked_display}\n\n"
        f"Article title: {title}\n"
        f"Article text: {_truncate_article(text)}\n\n"
        f"Extract ONLY a JSON object (no markdown fences, no commentary) listing at most the 5 MOST "
        f"significant companies NOT in the tracked list above that this article discusses in real, "
        f"specific depth -- not a passing name-drop, not a competitor mentioned only for context. "
        f"If more than 5 qualify, keep only the 5 most substantively covered:\n"
        f'{{"other_companies_mentioned": [{{"name": "...", "ticker": "TICKER or null if you don\'t '
        f'know it", "why": "1-3 short factual bullets, most important first, semicolon-separated -- '
        f"don't compress multiple distinct facts into one sentence if the article genuinely gives "
        f"more than one (e.g. Raised $200M Series C led by X; partnered with Y on Z; cited as the "
        f'main threat to a tracked ticker\'s market share)"}}, ...]}} (empty list if none, at most 5 entries)'
    )
    raw = complete(prompt, max_tokens=1200, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data:
        return None
    return data.get("other_companies_mentioned")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Stop after checking this many articles.")
    args = parser.parse_args()

    events = json.loads(EVENTS_CACHE_PATH.read_text())
    pending = [
        link for link, entry in events.items()
        if entry.get("event") and "other_companies_mentioned" not in entry["event"] and entry.get("text")
    ]
    print(f"{len(events)} articles total, {len(pending)} missing discovery data.")
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit given, processing {len(pending)}.")

    display_names = _display_names()
    valid_tickers = set(display_names.keys())

    done = 0
    found_total = 0
    for i, link in enumerate(pending, 1):
        entry = events[link]
        title = entry["title"]
        print(f"[{i}/{len(pending)}] {title[:70]}")
        try:
            raw_companies = _find_other_companies(title, entry["text"], display_names)
        except RateLimited as exc:
            reason = f" ({exc.message})" if exc.message else ""
            print(f"Rate limited{reason} -- stopping here. {done} checked this run; re-run to continue.")
            break
        if raw_companies is None:
            print("  -- no usable response, skipping (will retry next run)")
            continue

        other_companies = _sanitize_other_companies(raw_companies, valid_tickers)
        entry["event"]["other_companies_mentioned"] = other_companies
        EVENTS_CACHE_PATH.write_text(json.dumps(events, indent=2))
        done += 1

        if other_companies:
            found_total += len(other_companies)
            article_date = entry.get("article_date") or dt.date.today().isoformat()
            _append_discovery_candidates([
                {
                    "type": TYPE_COMPANY,
                    "date": article_date, "source": entry.get("source", "unknown"), "article_title": title,
                    "article_link": link, "name": c["name"], "ticker_guess": c["ticker"], "why": c["why"],
                }
                for c in other_companies
            ])
            names = ", ".join(f"{c['name']} ({c['ticker']})" if c["ticker"] else c["name"] for c in other_companies)
            print(f"  -- found: {names}")
        else:
            print("  -- nothing found")

    print(f"Done: {done} article(s) checked this run, {found_total} discovery candidate(s) found.")

    # Pass 2: theme-branch articles (no tracked company found at all) -- these never went through
    # any discovery extraction historically, so backfill both themes and companies via the real
    # extract_themes function rather than a hand-duplicated prompt.
    pending_themes = [
        link for link, entry in events.items()
        if not entry.get("event") and entry.get("text") and not entry.get("discovery_checked")
    ]
    print(f"\n{len(pending_themes)} theme-branch article(s) missing discovery data.")
    if args.limit is not None:
        pending_themes = pending_themes[: args.limit]
        print(f"--limit given, processing {len(pending_themes)}.")

    done2 = 0
    found_total2 = 0
    for i, link in enumerate(pending_themes, 1):
        entry = events[link]
        title = entry["title"]
        print(f"[{i}/{len(pending_themes)}] {title[:70]}")
        article_date = (
            dt.date.fromisoformat(entry["article_date"]) if entry.get("article_date") else dt.date.today()
        )
        try:
            result = extract_themes(
                title, entry["text"], article_date, entry.get("source", "unknown"), valid_tickers,
            )
        except RateLimited as exc:
            reason = f" ({exc.message})" if exc.message else ""
            print(f"Rate limited{reason} -- stopping here. {done2} checked this run; re-run to continue.")
            break

        themes, companies = result["themes"], result["companies"]
        entry["discovery_checked"] = True
        EVENTS_CACHE_PATH.write_text(json.dumps(events, indent=2))
        done2 += 1

        if companies:
            found_total2 += len(companies)
            _append_discovery_candidates([
                {
                    "type": TYPE_COMPANY,
                    "date": article_date.isoformat(), "source": entry.get("source", "unknown"),
                    "article_title": title, "article_link": link,
                    "name": c["name"], "ticker_guess": c["ticker"], "why": c["why"],
                }
                for c in companies
            ])
        if themes:
            found_total2 += len(themes)
            _append_discovery_candidates([
                {
                    "type": TYPE_THEME,
                    "date": article_date.isoformat(), "source": entry.get("source", "unknown"),
                    "article_title": title, "article_link": link,
                    "name": t["name"], "why": t["why"],
                }
                for t in themes
            ])
        if companies or themes:
            names = ", ".join([c["name"] for c in companies] + [t["name"] for t in themes])
            print(f"  -- found: {names}")
        else:
            print("  -- nothing found")

    print(f"Done: {done2} theme-branch article(s) checked this run, {found_total2} discovery candidate(s) found.")


if __name__ == "__main__":
    main()
