"""Everything this app knows about a ticker's earnings calls, past and upcoming, in one per-ticker
store (output/earnings/{ticker}.json) -- a flat, date-ordered list of events, each tagged
with a "type" so two genuinely different kinds of record can share one file/history instead of
each needing its own directory:

  - TYPE_TRANSCRIPT ("earnings_transcript"): an independent read on a company's most recent
    earnings call, sourced from the call transcript itself (guidance, management tone, analyst
    Q&A) rather than yfinance factors. Deliberately NOT scored against any particular news thesis,
    for the same reason finance.fundamentals isn't: a single number tied to "does this support
    direction X" stops meaning anything once X changes. Entries from before this "type" field
    existed have none -- treated as TYPE_TRANSCRIPT (see _entry_type), since that was the only
    kind of entry this file ever held until preview events were folded in.
  - TYPE_PREVIEW ("earnings_preview"): a short, news-grounded "what to watch" read ahead of a
    ticker's *next* call -- forward-looking, unlike the transcript read above which only exists
    *after* a call happened. Was previously its own module/directory (finance.earnings_preview,
    output/earnings_preview/) -- merged in here since it's the same underlying thing (a snapshot
    about a ticker's earnings call at a point in time) and having every caller import from two
    near-identical per-ticker-JSON modules was more indirection than the split bought.

A ticker's *reported results* (beat/miss, surprise %) are deliberately NOT a third type here --
those already come for free, always fresh, from finance.data.get_earnings_history's own yfinance-
backed cache (see app.py's _latest_reported_earnings), with no LLM cost and nothing to dedupe/
retry, so persisting a redundant copy here would just be a second, staler source of the same
numbers to keep in sync.

Three transcript-side parts:

`discover_recent_transcripts`/`_find_latest_transcript` (no LLM): locates the most recent
transcript for a ticker on The Motley Fool (fool.com/earnings/call-transcripts/...), which
publishes full transcripts free -- verified directly against several tickers (AAPL, NVDA, RGTI)
spanning mega-cap to small-cap, no paywall. There's no per-ticker feed or search API, so the
primary mechanism is `discover_recent_transcripts`: one fetch of Fool's own reverse-chronological
transcript listing page (plain server-rendered HTML, no JS, no bot-detection hit across repeated
automated fetches -- same site as the transcript fetch itself), covering every ticker on it in one
request instead of one request per ticker. Only covers the ~20 most-recently-published transcripts
sitewide, though -- the listing's `?page=N` param turned out to have no actual effect via a plain
HTTP request (confirmed empirically: identical content regardless of page number, since real
pagination there is client-side/JS-driven), so there's currently no reliable way to reach older
entries this way. A ticker whose last report isn't among those ~20 most recent (i.e. most checks,
in practice) falls back to `_search_transcript_via_ddg` -- DuckDuckGo's HTML search endpoint, no
API key required, but confirmed (empirically, mid-development) to trigger an anti-bot CAPTCHA
challenge after just 2-3 rapid automated requests, so it's kept as a fallback only, never the
primary path for scanning the whole tracked universe.

`_fetch_transcript_sections` (no LLM): fetches the transcript page and splits it into prepared
remarks vs. Q&A on the operator's own "opening the call for questions" transition line -- a
near-universal phrase in conference-call operator scripts, not Fool-specific. Falls back to
treating the whole transcript as one section if the marker isn't found (degrades gracefully,
same "soft failure" spirit as the rest of Loop A).

`earnings_call_snapshot` (one LLM call): given both sections (plus the previous snapshot's
guidance, for a "what changed" comparison, same trick finance.fundamentals uses), produces a
standalone direction/confidence, guidance summary, and -- the reason Q&A gets its own section
in the prompt rather than being lumped in with prepared remarks -- `management_tone` and
`key_qa_moments`, weighted toward the Q&A specifically. Prepared remarks are scripted and
IR-reviewed, so they skew polished; Q&A is where analysts probe on exactly what the market is
worried about, and hedging/deflection/surprising specificity in an unscripted answer is real
signal that prepared remarks won't contain.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
import trafilatura

from finance.data import get_earnings_history
from finance.llm import complete
from finance.loop_a_config import llm_config, max_article_chars
from finance.news_sources import fetch_finnhub_headlines, select_diverse_headlines
from finance import read_state

EARNINGS_DIR = Path("output/earnings")

# Gates the discovery search (Fool listing lookup is free/sitewide, but a ticker not on it falls
# back to a live per-ticker DuckDuckGo search -- real network cost, run-loop wall-clock time, and
# the exact kind of repeated automated traffic that trips DDG's anti-bot check) to tickers actually
# near an earnings date, same spirit as finance.newsloop's fundamentals earnings-window trigger.
# No BEFORE window at all, unlike that one -- a transcript can't exist before the call happens, so
# checking early would only ever waste a search. AFTER stays open for a week since Fool/DDG
# sometimes lag the call itself by a few days before indexing it, but not the ~2 weeks an earlier
# version of this gate allowed -- see needs_transcript_check below for what actually closes the
# window early once the transcript's been found, rather than relying on a short AFTER alone.
TRANSCRIPT_WINDOW_BEFORE_DAYS = 0
TRANSCRIPT_WINDOW_AFTER_DAYS = 7


def needs_transcript_check(ticker: str, as_of: dt.date) -> bool:
    """True if `ticker` should get a live discovery search right now: `as_of` falls within
    TRANSCRIPT_WINDOW_{BEFORE,AFTER}_DAYS of one of its last few scheduled/reported earnings dates
    (checks every date get_earnings_history returns, not just the next upcoming one), AND the
    transcript already on record (if any) predates *that* earnings date -- so once this cycle's
    transcript has actually been found and stored, checking stops for the rest of the window instead
    of re-searching every single day until AFTER_DAYS closes it. Fails open (returns True) on an
    empty earnings history, so a data gap means "check it" rather than silently never checking a
    ticker again.
    """
    earnings = get_earnings_history(ticker)
    if earnings.empty:
        return True
    window_earnings_date = None
    for earnings_date in earnings["earnings_date"].dt.date:
        window_start = earnings_date - dt.timedelta(days=TRANSCRIPT_WINDOW_BEFORE_DAYS)
        window_end = earnings_date + dt.timedelta(days=TRANSCRIPT_WINDOW_AFTER_DAYS)
        if window_start <= as_of <= window_end:
            window_earnings_date = earnings_date
            break
    if window_earnings_date is None:
        return False
    previous = latest_earnings_call(ticker)
    if previous is None:
        return True
    stored_date = dt.date.fromisoformat(previous["transcript_date"])
    return stored_date < window_earnings_date

TYPE_TRANSCRIPT = "earnings_transcript"
TYPE_PREVIEW = "earnings_preview"

_PREVIEW_HEADLINE_LIMIT = 15
_PREVIEW_MAX_PER_DOMAIN = 2

_USER_AGENT = {"User-Agent": "Mozilla/5.0"}
_LISTING_URL = "https://www.fool.com/earnings-call-transcripts/"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/"

# fool.com/earnings/call-transcripts/YYYY/MM/DD/{slug}/ -- the date embedded in the URL itself is
# what "latest" is decided from, never a search engine's own result ordering (DuckDuckGo's isn't
# reliably chronological -- confirmed empirically, an older quarter sometimes ranked above a newer
# one for the same ticker). Captures the whole slug; the ticker itself is extracted from it
# separately (see _ticker_from_slug) since its position within the slug varies (a company name can
# be one word or several -- "ams-ams-q2-..." vs. "applied-industrial-ait-q4-...").
_TRANSCRIPT_PATH_RE = re.compile(
    r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/([a-z0-9-]+)/"
)
# The ticker is always the hyphen-token immediately before "-q{N}-{year}-earnings..." -- true
# whether the company-name portion of the slug is one word ("ams-ams-q2-2026-...") or several
# ("applied-industrial-ait-q4-2026-..." -> ticker AIT), confirmed against real listing entries.
_SLUG_TICKER_RE = re.compile(r"^(.+)-q\d+-\d{4}-earnings")


def _ticker_from_slug(slug: str) -> str | None:
    match = _SLUG_TICKER_RE.match(slug)
    if not match:
        return None
    return match.group(1).rsplit("-", 1)[-1]


def discover_recent_transcripts() -> dict[str, tuple[str, dt.date]]:
    """ticker (lowercase) -> (url, transcript_date) for every distinct ticker on Fool's own
    reverse-chronological transcript listing page -- see module docstring for why this is the
    primary discovery mechanism. Call this ONCE per run (one request covering every ticker at
    once), not per-ticker -- pass its result as `recent_index` to refresh_earnings_call/
    `_find_latest_transcript` for each ticker to look up.

    Only ever covers the roughly 20 most-recently-published transcripts sitewide (across every
    company Fool covers, not just tracked ones) -- confirmed empirically that `?page=N` has NO
    effect via a plain HTTP request (pagination is client-side/JS-driven, not server-rendered by
    query param, despite the URL accepting one), so there is currently no reliable way to fetch
    page 2+ without a headless browser. In practice this means: catches a tracked ticker that
    reported earnings very recently (the main "did anything new land" use case), but a ticker that
    reported longer ago always falls through to `_search_transcript_via_ddg` instead.
    """
    found: dict[str, tuple[str, dt.date]] = {}
    try:
        response = requests.get(_LISTING_URL, timeout=15, headers=_USER_AGENT)
        response.raise_for_status()
    except requests.RequestException:
        return found
    matches = _TRANSCRIPT_PATH_RE.findall(response.text)
    if matches:
        for year, month, day, slug in matches:
            ticker = _ticker_from_slug(slug)
            if ticker is None:
                continue
            try:
                transcript_date = dt.date(int(year), int(month), int(day))
            except ValueError:
                continue
            url = f"https://www.fool.com/earnings/call-transcripts/{year}/{month}/{day}/{slug}/"
            existing = found.get(ticker)
            if existing is None or transcript_date > existing[1]:
                found[ticker] = (url, transcript_date)
    return found


def _search_transcript_via_ddg(ticker: str) -> tuple[str, dt.date] | None:
    """Fallback for a ticker discover_recent_transcripts didn't find (its last report isn't among
    the ~20 most-recently-published sitewide -- in practice, most tickers on most days) -- see
    module docstring re: DuckDuckGo's anti-bot sensitivity under repeated automated use. A short
    self-imposed delay before each request is basic good-citizen throttling, not a guaranteed
    fix -- confirmed a block can still happen after just 2-3 rapid calls, so a `--refresh-earning-
    call` run over the full tracked universe (no `--ticker`) should be expected to occasionally
    return "no new transcript found" for a ticker that does have new content available (soft
    failure, same as every other Stage in this app) rather than guaranteed complete coverage.
    A single `--ticker AAPL`-style call is unaffected by any of this.
    """
    time.sleep(2)
    query = f"site:fool.com {ticker} earnings call transcript"
    try:
        response = requests.get(
            _DDG_HTML_URL, params={"q": query}, timeout=15, headers=_USER_AGENT
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    raw_links = re.findall(r"uddg=([^&\"]+)", response.text)
    candidates: list[tuple[dt.date, str]] = []
    seen_urls: set[str] = set()
    for raw in raw_links:
        url = urllib.parse.unquote(raw)
        match = re.match(r"https://www\.fool\.com" + _TRANSCRIPT_PATH_RE.pattern, url)
        if not match or url in seen_urls:
            continue
        seen_urls.add(url)
        year, month, day, slug = match.groups()
        if _ticker_from_slug(slug) != ticker.lower():
            continue
        try:
            transcript_date = dt.date(int(year), int(month), int(day))
        except ValueError:
            continue
        candidates.append((transcript_date, url))

    if not candidates:
        return None
    latest_date, latest_url = max(candidates, key=lambda c: c[0])
    return latest_url, latest_date


def _find_latest_transcript(
    ticker: str, recent_index: dict[str, tuple[str, dt.date]] | None = None
) -> tuple[str, dt.date] | None:
    """(url, transcript_date) for `ticker`'s most recent transcript -- looks up `recent_index`
    (from discover_recent_transcripts) first if given, falling back to a live DDG search if the
    ticker isn't in it (or no index was given at all).
    """
    if recent_index is not None:
        hit = recent_index.get(ticker.lower())
        if hit is not None:
            return hit
    return _search_transcript_via_ddg(ticker)


# The operator's own transition into Q&A -- checked in this order, first match wins. All are
# near-universal earnings-call operator phrasing, not specific to any one company or Fool's own
# formatting, since the transcript is a plain speaker-labeled dialogue with no structural markup
# separating the two parts (confirmed by inspecting real transcripts' HTML: no "Q&A" heading tag
# exists, it's all one "Full Conference Call Transcript" block).
_QA_TRANSITION_MARKERS = (
    "open the call for questions",
    "open the call up for questions",
    "open the call to questions",
    "opens the call for questions",
    "first question comes from",
)


def _split_transcript(text: str) -> tuple[str, str]:
    """(prepared_remarks, qa) -- splits on whichever _QA_TRANSITION_MARKERS phrase appears
    earliest, since "first question comes from" (the operator formally introducing it) can appear
    before or after management's own "we're now ready for questions" line depending on how the
    operator's script is transcribed. Falls back to (full_text, "") -- no marker found -- rather
    than raising, so the LLM call still gets *something* even if the split heuristic misses; it
    just loses the "weight Q&A more" benefit for that one call.
    """
    lowered = text.lower()
    earliest = None
    for marker in _QA_TRANSITION_MARKERS:
        idx = lowered.find(marker)
        if idx != -1 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is None:
        return text, ""
    return text[:earliest], text[earliest:]


def _fetch_transcript_sections(url: str) -> tuple[str, str] | None:
    """(prepared_remarks, qa) extracted from the transcript at `url`, or None on any fetch/parse
    failure -- same soft-failure contract as finance.news.fetch_full_page_text.
    """
    try:
        response = requests.get(url, timeout=20, headers=_USER_AGENT)
        response.raise_for_status()
    except requests.RequestException:
        return None
    text = trafilatura.extract(response.text)
    if not text:
        return None
    return _split_transcript(text)


def _truncate(text: str) -> str:
    limit = max_article_chars()
    return text[:limit] if limit is not None else text


_JSON_BLOB = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_EARNINGS_CALL_REQUIRED_FIELDS = {
    "earnings_direction", "earnings_confidence", "guidance_summary", "guidance_change",
    "management_tone", "key_qa_moments", "risks", "summary",
}
_VALID_TONES = {"confident", "cautious", "defensive", "evasive"}


def earnings_call_snapshot(
    ticker: str, prepared_remarks: str, qa: str, previous: dict | None, as_of: dt.date
) -> dict | None:
    """One LLM call: an independent read of `ticker`'s most recent earnings call. Never raises for
    a genuine parse failure (returns None). Raises RateLimited (propagated from
    finance.llm.complete) if every model is currently out of quota, same as every other Stage call.

    `previous` is the last earnings_call_snapshot event stored for this ticker (or None for a
    first-ever check) -- its "guidance_summary" grounds this call's "guidance_change" comparison.
    """
    if previous is None:
        history_note = "No previous earnings call on record for this ticker -- this is the first."
    else:
        history_note = (
            f"Previous call ({previous['date']}) guidance: {previous.get('guidance_summary', '')}"
        )

    prompt = (
        f"You are an equity analyst reviewing {ticker}'s most recent earnings call, as of "
        f"{as_of.isoformat()}. The transcript is split into two parts below -- prepared remarks "
        f"(scripted, reviewed by IR/legal, expect it to skew polished) and the analyst Q&A "
        f"(unscripted; analysts probe on exactly what the market is worried about, and hedging, "
        f"deflection, or surprising specificity in an answer is real signal prepared remarks "
        f"won't contain). Weight the Q&A more heavily than the prepared remarks for tone and "
        f"risks; use the prepared remarks primarily for guidance numbers.\n\n"
        f"PREPARED REMARKS:\n{_truncate(prepared_remarks)}\n\n"
        f"Q&A:\n{_truncate(qa) if qa else '(no distinct Q&A section found in this transcript)'}\n\n"
        f"{history_note}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary):\n"
        f'{{"earnings_direction": "long", "short", or "neutral" (the view this call ALONE '
        f"implies, independent of any other news -- \"neutral\" if genuinely mixed), "
        f'"earnings_confidence": number between 0 and 1 (conviction in that direction), '
        f'"guidance_summary": "what management guided to this call -- revenue/margin/capex, '
        f'qualitative if no hard numbers were given", '
        f'"guidance_change": "how this compares to the previous call\'s guidance above -- '
        f'\\"no prior call on record\\" if this is the first", '
        f'"management_tone": one of {sorted(_VALID_TONES)} (based mainly on how management '
        f"handled Q&A pushback, not how the prepared remarks framed things), "
        f'"key_qa_moments": ["...", ...] (2-4 bullets: the analyst\'s actual concern + how '
        f"management responded, prioritizing exchanges where the answer was hedgy, deflected, or "
        f"surprisingly specific -- empty list if the Q&A section wasn't usable), "
        f'"risks": ["...", ...] (risks raised or implied, especially from Q&A hedging -- empty '
        f"list if none), "
        f'"summary": "2-3 sentence overview of the whole call"}}'
    )

    raw = complete(prompt, max_tokens=1400, **llm_config())
    if not raw:
        return None
    data = _extract_json(raw)
    if not data or not _EARNINGS_CALL_REQUIRED_FIELDS.issubset(data):
        return None
    if data["earnings_direction"] not in ("long", "short", "neutral"):
        data["earnings_direction"] = "neutral"
    if data["management_tone"] not in _VALID_TONES:
        data["management_tone"] = "cautious"

    data["earnings_confidence"] = max(0.0, min(1.0, float(data["earnings_confidence"])))
    data["key_qa_moments"] = [str(m) for m in (data.get("key_qa_moments") or [])]
    data["risks"] = [str(r) for r in (data.get("risks") or [])]
    return data


def _path(ticker: str) -> Path:
    return EARNINGS_DIR / f"{ticker}.json"


def _entry_type(event: dict) -> str:
    """An entry predating the "type" field (every one stored before preview events were folded in
    here) was always a transcript-read event -- see this module's own docstring.
    """
    return event.get("type", TYPE_TRANSCRIPT)


def _load_all(ticker: str) -> list[dict]:
    """Every event ever stored for `ticker`, of any type, oldest first -- internal; callers want
    one of the type-filtered accessors below instead.
    """
    path = _path(ticker)
    return json.loads(path.read_text()) if path.exists() else []


def _save_all(ticker: str, history: list[dict]) -> None:
    path = _path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))


def load_earnings_call_history(ticker: str) -> list[dict]:
    """Every transcript-read snapshot ever stored for `ticker` (TYPE_TRANSCRIPT only -- excludes
    any preview events in the same file), oldest first.
    """
    return [e for e in _load_all(ticker) if _entry_type(e) == TYPE_TRANSCRIPT]


def latest_earnings_call(ticker: str) -> dict | None:
    history = load_earnings_call_history(ticker)
    return history[-1] if history else None


def load_earnings_preview_history(ticker: str) -> list[dict]:
    """Every earnings-preview read ever stored for `ticker` (TYPE_PREVIEW only), oldest first."""
    return [e for e in _load_all(ticker) if _entry_type(e) == TYPE_PREVIEW]


def latest_earnings_preview(ticker: str) -> dict | None:
    history = load_earnings_preview_history(ticker)
    return history[-1] if history else None


def _append_transcript(ticker: str, event: dict) -> None:
    event = {**event, "type": TYPE_TRANSCRIPT}
    history = _load_all(ticker)
    # Same reasoning as finance.fundamentals._append -- app.py's card id is keyed off ticker+date,
    # so a second snapshot landing on the same date would otherwise inherit a stale read-mark.
    # Scoped to TYPE_TRANSCRIPT entries only -- a preview event sharing the same "date" (when it
    # was generated, not the call date) is a coincidence, not the same card.
    if any(_entry_type(e) == TYPE_TRANSCRIPT and e.get("date") == event.get("date") for e in history):
        read_state.mark_unread(read_state.CURRENT_USER, read_state.card_id(ticker, "earnings_call", event["date"]))
    history.append(json.loads(json.dumps(event, default=str)))
    _save_all(ticker, history)


def _upsert_preview(ticker: str, event: dict) -> None:
    """Same "one row per real-world thing this describes" upsert finance.macro's narrative storage
    uses, just keyed on "call_date" (the upcoming call this preview is *for*) rather than a week/
    month bucket -- re-running before that call happens overwrites the same entry in place instead
    of appending a fresh one every time.
    """
    event = {**event, "type": TYPE_PREVIEW}
    history = _load_all(ticker)
    serialized = json.loads(json.dumps(event, default=str))
    existing_idx = next(
        (i for i, e in enumerate(history)
         if _entry_type(e) == TYPE_PREVIEW and e.get("call_date") == event["call_date"]),
        None,
    )
    if existing_idx is not None:
        history[existing_idx] = serialized
    else:
        history.append(serialized)
    _save_all(ticker, history)


def refresh_earnings_call(
    ticker: str, as_of: dt.date, recent_index: dict[str, tuple[str, dt.date]] | None = None,
    transcript_url: str | None = None,
) -> dict | None:
    """The one entry point every caller needs: finds `ticker`'s most recent earnings call
    transcript, skips entirely (zero LLM cost) if it's the same transcript already stored --
    matched by the transcript's own date, not `as_of` -- fetches and splits it if it's new, runs
    the LLM read, persists it as the newest entry in this ticker's own file, and returns it.

    `recent_index` is discover_recent_transcripts()'s result -- pass the SAME one when refreshing
    multiple tickers in a loop (call discover_recent_transcripts() once beforehand) rather than
    letting each call fall through to a fresh DDG search; a ticker not in the index (or if no
    index is given at all) still falls back to a live search on its own.

    `transcript_url` bypasses discovery (both the listing lookup and the DDG fallback) entirely --
    for the case where a ticker isn't in Fool's ~20 sitewide-newest and DuckDuckGo's anti-bot check
    is blocking the fallback search (see module docstring), but you already have the fool.com
    transcript URL yourself (found it in a browser). Must be a fool.com call-transcripts URL
    matching _TRANSCRIPT_PATH_RE, or this returns None.

    Returns None if: no transcript could be found/fetched, the transcript is the same one already
    on record, or the LLM call itself produced nothing usable. Propagates RateLimited, same as
    every other Stage call.
    """
    if transcript_url is not None:
        match = re.search(_TRANSCRIPT_PATH_RE, transcript_url)
        if not match:
            return None
        year, month, day, _slug = match.groups()
        try:
            found = (transcript_url, dt.date(int(year), int(month), int(day)))
        except ValueError:
            return None
    else:
        found = _find_latest_transcript(ticker, recent_index)
    if found is None:
        return None
    url, transcript_date = found

    previous = latest_earnings_call(ticker)
    if previous is not None and previous.get("transcript_url") == url:
        return None  # already have this exact transcript on record -- nothing new to process

    sections = _fetch_transcript_sections(url)
    if sections is None:
        return None
    prepared_remarks, qa = sections
    if not prepared_remarks and not qa:
        return None

    snapshot = earnings_call_snapshot(ticker, prepared_remarks, qa, previous, as_of)
    if snapshot is None:
        return None

    _append_transcript(ticker, {
        "date": as_of.isoformat(),
        "transcript_date": transcript_date.isoformat(),
        "transcript_url": url,
        "earnings_direction": snapshot["earnings_direction"],
        "earnings_confidence": snapshot["earnings_confidence"],
        "guidance_summary": snapshot["guidance_summary"],
        "guidance_change": snapshot["guidance_change"],
        "management_tone": snapshot["management_tone"],
        "key_qa_moments": snapshot["key_qa_moments"],
        "risks": snapshot["risks"],
        "summary": snapshot["summary"],
    })
    return snapshot


# ---------------------------------------------------------------------------
# Earnings preview -- TYPE_PREVIEW, a forward-looking "what to watch" read ahead of a ticker's
# *next* call (see module docstring). No transcript involved; grounded in real recent headlines
# from finance.news_sources instead.
# ---------------------------------------------------------------------------


def _preview_prompt(ticker: str, company_name: str, call_date: dt.date, headlines_text: str) -> str:
    return (
        f"Real recent headlines mentioning {company_name} ({ticker}), via Finnhub, from the last 7 days:\n"
        f"{headlines_text}\n\n"
        f"{ticker} reports earnings on {call_date.isoformat()}. In 3-4 sentences, based only on the "
        f"headlines above plus your own general knowledge of the company, what should investors be "
        f"watching for in this call? What are your thoughts and expectations for the upcoming "
        f"earnings? Be concrete and specific where the headlines actually support it. Do not state "
        f"specific financial figures (revenue, EPS, price targets, etc.) unless they appear verbatim "
        f"in a headline above -- if you don't have real numbers to cite, describe the dynamic in "
        f"words instead of guessing a number."
    )


def refresh_earnings_preview(ticker: str, company_name: str, call_date: dt.date, as_of: dt.date) -> dict:
    """The one entry point every caller needs: fetches this week's Finnhub company-news headlines
    for `ticker`, asks the LLM for a short grounded preview of the `call_date` earnings call, and
    upserts the result as a TYPE_PREVIEW entry (keyed on call_date -- see _upsert_preview).

    Always returns a dict with a "status" key, same convention as finance.macro.
    refresh_macro_narrative:
      "generated"          -- a preview was produced and stored; "narrative"/"headlines_used" present.
      "news_unavailable"   -- the Finnhub fetch itself failed (missing API key, rate-limited,
                              network error) -- no LLM call was made, nothing was stored, safe to
                              retry anytime.
      "llm_failed"         -- the LLM call ran but produced nothing usable.
    Propagates RateLimited, same as every other Stage call (via finance.llm.complete).
    """
    raw = fetch_finnhub_headlines(ticker, days=7)
    if raw is None:
        return {"status": "news_unavailable"}
    headlines = select_diverse_headlines(raw, limit=_PREVIEW_HEADLINE_LIMIT, max_per_domain=_PREVIEW_MAX_PER_DOMAIN)
    headlines_text = (
        "\n".join(f'- ({h["date"]}, {h["domain"]}) {h["title"]}' for h in headlines)
        if headlines else "(no relevant headlines found this week)"
    )
    prompt = _preview_prompt(ticker, company_name, call_date, headlines_text)
    narrative = complete(prompt, max_tokens=400, **llm_config())
    if narrative is None:
        return {"status": "llm_failed"}

    event = {
        "status": "generated", "date": as_of.isoformat(), "call_date": call_date.isoformat(),
        "narrative": narrative.strip(), "headlines_used": headlines,
    }
    _upsert_preview(ticker, {k: v for k, v in event.items() if k != "status"})
    return event
