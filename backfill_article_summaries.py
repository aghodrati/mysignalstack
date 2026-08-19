"""TEMPORARY, one-off script -- delete after running.

Backfills Stage A's "summary" field (see finance.newsloop.extract_event) onto every article
already sitting in cache/thesis/events.json from before that field existed. Does NOT re-run full
Stage A classification (that would re-spend a full classification call and could shift
event/companies/importance from whatever's already been aggregated into claims) -- just a cheap,
separate summarization call over the article text already stored in events.json, written directly
into event["summary"] so app.py's claim cards (finance.newsloop.get_article_archive) pick it up.

Usage:
    python backfill_article_summaries.py [--limit N]

Saves to events.json after every successful summary (not batched) so a rate-limit stop or Ctrl-C
partway through loses no progress -- just re-run the script again afterward to pick up where it
left off (already-summarized articles are skipped).
"""

from __future__ import annotations

import argparse
import json

from finance.llm import RateLimited, complete
from finance.loop_a_config import llm_config, max_article_chars
from finance.newsloop import EVENTS_CACHE_PATH


def _truncate(text: str) -> str:
    limit = max_article_chars()
    return text[:limit] if limit is not None else text


def _summarize(title: str, text: str) -> str | None:
    prompt = (
        "Write a general one-paragraph (2-4 sentence) summary of the article below, for a reader "
        "who hasn't read it. Neutral and factual -- what it says, not why it matters to any "
        "particular company. No markdown, no preamble, just the summary.\n\n"
        f"Title: {title}\n\n{_truncate(text)}"
    )
    return complete(prompt, max_tokens=400, **llm_config())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Stop after summarizing this many articles.")
    args = parser.parse_args()

    events = json.loads(EVENTS_CACHE_PATH.read_text())
    pending = [
        link for link, entry in events.items()
        if entry.get("event") and not entry["event"].get("summary") and entry.get("text")
    ]
    print(f"{len(events)} articles total, {len(pending)} missing a summary.")
    if args.limit is not None:
        pending = pending[: args.limit]
        print(f"--limit given, processing {len(pending)}.")

    done = 0
    for i, link in enumerate(pending, 1):
        entry = events[link]
        title = entry["title"]
        print(f"[{i}/{len(pending)}] {title[:70]}")
        try:
            summary = _summarize(title, entry["text"])
        except RateLimited as exc:
            reason = f" ({exc.message})" if exc.message else ""
            print(f"Rate limited{reason} -- stopping here. {done} summarized this run; re-run the script to continue.")
            break
        if not summary:
            print("  -- no summary returned, skipping")
            continue
        entry["event"]["summary"] = summary.strip()
        EVENTS_CACHE_PATH.write_text(json.dumps(events, indent=2))
        done += 1
        print(f"  -- saved ({len(summary)} chars)")

    print(f"Done: {done} article(s) summarized this run.")


if __name__ == "__main__":
    main()
