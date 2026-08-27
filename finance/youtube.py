"""YouTube interview/podcast summaries -- a separate track from finance.newsloop's article claims
pipeline, deliberately not fed through Stage A/B (see the design discussion this came from: a long
interview transcript isn't ticker-scoped the way an article is, and forcing it through the claims
extractor loses the actual value, which is the discussion itself). Channel -> new videos -> transcript
-> one LLM summarization call per video, persisted to output/youtube_summaries.json and rendered on
app.py's own Interviews page.

Two fetches per new video: the channel's Atom feed (finance.loop_a_config.youtube_sources()) to
discover video ids/titles/publish dates -- youtube.com/feeds/videos.xml, no API key needed -- and
youtube_transcript_api to pull that video's own auto-captions. A video with captions disabled is
skipped (recorded as unprocessable so it isn't retried every run), same soft-skip spirit as
finance.news.fetch_full_page_text's silent "" return.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

from finance.llm import RateLimited, complete
from finance.loop_a_config import llm_config, max_article_chars, youtube_sources

DATA_DIR = Path("output/youtube")
SUMMARIES_PATH = DATA_DIR / "summaries.json"
# Video ids permanently skipped for being under MIN_VIDEO_MINUTES -- a fact about the video itself,
# stable forever, so this is checked/updated separately from SUMMARIES_PATH (which only ever holds
# real summaries app.py renders as cards) rather than mixing a "skipped, nothing to show" marker
# into the same list every card-rendering caller would then have to filter back out.
SKIPPED_PATH = DATA_DIR / "skipped.json"

# Same reasoning as finance.newsloop.MAX_ARTICLE_AGE_MONTHS -- a channel's feed can carry a backlog
# stretching back well beyond what's still worth a transcript fetch + LLM summarization call.
MAX_VIDEO_AGE_MONTHS = 1

# Many channels (e.g. Dwarkesh Patel's) mix full-length episodes into the same feed as sub-2-minute
# clips cut from those same episodes for shorts/social -- a clip has nothing a summary adds over
# just reading its own title, so it's not worth a transcript fetch + LLM call at all. 10 minutes
# comfortably separates "clip" from "actual episode" for every channel seen so far without risking
# cutting off a genuinely short but substantive interview.
MIN_VIDEO_MINUTES = 10

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_YT_NS = "{http://www.youtube.com/xml/schemas/2015}"
# Bypasses YouTube's EU/UK cookie-consent interstitial (which otherwise replaces the real watch
# page with a consent form containing none of the metadata below) -- a fixed, non-personalized
# consent cookie, not tied to any account.
_WATCH_PAGE_HEADERS = {"User-Agent": "Mozilla/5.0", "Cookie": "CONSENT=YES+cb; SOCS=CAI"}


def load_youtube_summaries() -> list[dict]:
    """Every video summarized so far, in whatever order they were appended (oldest first) --
    app.py sorts for display; this just returns the raw persisted list.
    """
    if not SUMMARIES_PATH.exists():
        return []
    return json.loads(SUMMARIES_PATH.read_text())


def _save_youtube_summaries(summaries: list[dict]) -> None:
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.write_text(json.dumps(summaries, indent=2))


def _load_skipped_ids() -> set[str]:
    if not SKIPPED_PATH.exists():
        return set()
    return set(json.loads(SKIPPED_PATH.read_text()))


def _save_skipped_ids(ids: set[str]) -> None:
    SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKIPPED_PATH.write_text(json.dumps(sorted(ids), indent=2))


def fetch_channel_videos(channel_id: str) -> list[dict]:
    """Newest-first {video_id, title, published (date), link} for every video currently in this
    channel's Atom feed (youtube.com/feeds/videos.xml -- no API key, no quota, but only ever
    returns the channel's most recent ~15 uploads, same rolling-window caveat finance.news's RSS
    fetch already has). Raises requests.RequestException on a network failure -- callers decide
    whether to skip this channel for the run or propagate, same as finance.news.get_news's own
    requests.raise_for_status().
    """
    response = requests.get(
        f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}",
        timeout=10, headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    videos = []
    for entry in root.findall(f"{_ATOM_NS}entry"):
        video_id = entry.findtext(f"{_YT_NS}videoId")
        title = entry.findtext(f"{_ATOM_NS}title")
        published = entry.findtext(f"{_ATOM_NS}published")
        link_el = entry.find(f"{_ATOM_NS}link")
        if not video_id or not title:
            continue
        videos.append({
            "video_id": video_id,
            "title": title,
            "published": dt.datetime.fromisoformat(published).date().isoformat() if published else None,
            "link": link_el.get("href") if link_el is not None else f"https://youtu.be/{video_id}",
        })
    return videos


def fetch_video_duration_seconds(video_id: str) -> int | None:
    """This video's own length, or None if it couldn't be found (network failure, or YouTube's page
    structure not matching -- never raises, same soft-failure spirit as fetch_transcript). Not
    available from the channel Atom feed at all (see fetch_channel_videos), so this is a second,
    per-video fetch of the plain watch page, scraping the `lengthSeconds` field straight out of its
    embedded player-response JSON rather than pulling in a full YouTube API client/key for one field.
    """
    try:
        response = requests.get(
            f"https://www.youtube.com/watch?v={video_id}", headers=_WATCH_PAGE_HEADERS, timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None
    match = re.search(r'"lengthSeconds":"(\d+)"', response.text)
    return int(match.group(1)) if match else None


def fetch_transcript(video_id: str) -> str:
    """Plain concatenated transcript text for `video_id`, or "" if captions are disabled/unavailable
    (a fact about this video, not a transient error) -- never raises for that case, matching
    finance.news.fetch_full_page_text's own "best-effort, empty string on failure" contract.
    """
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    except CouldNotRetrieveTranscript:
        return ""
    return " ".join(snippet.text for snippet in transcript)


_SUMMARY_PROMPT = """You are summarizing a long-form interview/podcast transcript for a reader who tracks AI and technology investing (semiconductors, AI infrastructure, frontier AI labs, and related public companies). Write for someone who wants the substance of a 1-2+ hour conversation without listening to it.

Video title: {title}
Speakers: {speakers}

Transcript:
{transcript}

First, one line introducing the guest(s) being interviewed (not the host) -- 1-2 sentences per guest on who they are: why they're known and where they work. If there are multiple guests (e.g. a panel), briefly introduce each one rather than picking just one. Skip this line entirely if there's no real guest at all (e.g. a solo host episode) or you can't tell who the guest(s) are from the title/speakers/transcript rather than guessing. Prefix this line with "Guest: " and put it before everything else, on its own line.

Then a blank line, then summarize the conversation as 5-10 bullet points, ordered by importance (most important first). Each bullet should be a few sentences -- enough to stand on its own without the transcript, not a one-liner. If the conversation genuinely covers more than 10 distinct main topics, go beyond 10 rather than cramming two topics into one bullet or dropping a topic entirely -- one bullet per topic, not a fixed count.

Each bullet should center on one concrete thing: a specific fact, data point, or claim (a number, date, comparison, deal, capacity figure, timeline, whatever was said), who said it, and enough surrounding context to make it useful on its own. Where relevant, note the company/technology it concerns. If a bullet is opinion/prediction rather than fact, say so and attribute it to the speaker -- don't blend the two silently.

Favor specific, surprising, or non-obvious points over generic ones. Skip pleasantries, tangents, and anything a reader following AI/investing news would already know. Don't pad to hit 10 -- if the interview only has 5 points worth keeping, stop at 5.

Respond with just the optional Guest: line and the bullet points as a markdown list, nothing else -- no preamble, no title."""


def summarize_transcript(title: str, speakers: str, transcript_text: str) -> str | None:
    """One LLM call, this module's own prompt -- returns the raw markdown bullet-list text, or None
    if the LLM produced nothing usable (caller decides whether that means "skip this video" or
    "retry next run"). Truncated to max_article_chars() if configured, same cap Stage A/B/the 8-K
    classifier already respect (finance.loop_a_config.max_article_chars) -- one shared "how much can
    this model handle" answer, not a second parallel limit to keep in sync.
    """
    cap = max_article_chars()
    text = transcript_text[:cap] if cap else transcript_text
    prompt = _SUMMARY_PROMPT.format(title=title, speakers=speakers, transcript=text)
    return complete(prompt, max_tokens=2500, **llm_config())


def _video_word_count(transcript_text: str) -> int:
    return len(transcript_text.split())


def process_video(source_name: str, video: dict, as_of: dt.date, verbose: bool = True) -> dict | None:
    """Transcript -> summary for one video, or None if there's nothing to persist (no captions, or
    the LLM call failed) -- caller decides whether "nothing to persist" should still be recorded to
    avoid retrying every run (see refresh_youtube_sources).
    """
    transcript_text = fetch_transcript(video["video_id"])
    if not transcript_text:
        if verbose:
            print(f"    [youtube] {video['title'][:60]} -- no transcript available, skipping")
        return None
    summary = summarize_transcript(video["title"], source_name, transcript_text)
    if not summary:
        if verbose:
            print(f"    [youtube] {video['title'][:60]} -- LLM produced nothing usable, will retry next run")
        return None
    return {
        "video_id": video["video_id"],
        "title": video["title"],
        "channel": source_name,
        "published": video["published"] or as_of.isoformat(),
        "link": video["link"],
        "word_count": _video_word_count(transcript_text),
        "summary": summary.strip(),
        "summarized_at": as_of.isoformat(),
    }


def refresh_youtube_sources(as_of: dt.date | None = None, verbose: bool = True) -> list[dict]:
    """Checks every active finance.loop_a_config.youtube_sources() channel for videos not already in
    output/youtube_summaries.json, transcribes + summarizes each one, and appends the results.
    Returns just the newly added entries. A video whose captions aren't available yet is silently
    skipped (not recorded) so it's retried on a later run, once YouTube has generated captions for
    it -- unlike a genuine LLM failure, "no captions yet" is often just a timing issue for a
    freshly-uploaded video.
    """
    as_of = as_of or dt.date.today()
    existing = load_youtube_summaries()
    known_ids = {e["video_id"] for e in existing}
    skipped_ids = _load_skipped_ids()
    cutoff = as_of - dt.timedelta(days=MAX_VIDEO_AGE_MONTHS * 30)
    added: list[dict] = []
    newly_skipped: set[str] = set()
    for source in youtube_sources():
        try:
            videos = fetch_channel_videos(source["channel_id"])
        except requests.RequestException as exc:
            if verbose:
                print(f"[youtube] {source['name']} -- feed fetch failed ({exc}), skipping this channel")
            continue
        # Same staleness reasoning as finance.newsloop's article cutoff -- a video with no
        # parseable publish date is kept, not dropped, rather than silently discarded.
        new_videos = [
            v for v in videos
            if v["video_id"] not in known_ids and v["video_id"] not in skipped_ids
            and (v["published"] is None or dt.date.fromisoformat(v["published"]) >= cutoff)
        ]
        if verbose and new_videos:
            print(f"[youtube] {source['name']}: {len(new_videos)} new video(s) within {MAX_VIDEO_AGE_MONTHS} month(s)")
        for video in new_videos:
            duration = fetch_video_duration_seconds(video["video_id"])
            if duration is not None and duration < MIN_VIDEO_MINUTES * 60:
                if verbose:
                    print(f"    [youtube] {video['title'][:60]} -- {duration // 60}m, under {MIN_VIDEO_MINUTES}m minimum, skipping")
                newly_skipped.add(video["video_id"])
                continue
            try:
                entry = process_video(source["name"], video, as_of, verbose=verbose)
            except RateLimited as exc:
                if verbose:
                    reason = f" ({exc.message})" if exc.message else ""
                    print(f"  [youtube] rate limited{reason}, stopping here.")
                _save_youtube_summaries(existing + added)
                if newly_skipped:
                    _save_skipped_ids(skipped_ids | newly_skipped)
                return added
            if entry:
                added.append(entry)
                existing.append(entry)
                known_ids.add(video["video_id"])
                if verbose:
                    print(f"    [youtube] {entry['title'][:60]} -- summarized ({entry['word_count']} words)")
    if added:
        _save_youtube_summaries(existing)
    if newly_skipped:
        _save_skipped_ids(skipped_ids | newly_skipped)
    return added
