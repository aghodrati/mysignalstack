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
  - TYPE_REMINDER ("earnings_reminder"): the plain, deterministic numbers half of app.py's
    earnings-reminder card (precise call date/time, consensus EPS estimate, last few reported
    quarters' beat/miss streak) -- see refresh_earnings_reminder. A frozen point-in-time snapshot,
    same spirit as TYPE_PREVIEW's own headlines_used field. Overwritten in place (one entry per
    upcoming call, like TYPE_PREVIEW) rather than accumulated, since it only ever describes
    whatever call hasn't happened yet. Written independently of whether TYPE_PREVIEW's own
    generation succeeds for the same cycle -- a Finnhub/LLM hiccup shouldn't also blank out the
    numbers half of the card.
  - TYPE_RESULT ("earnings_result"): app.py's "results are in" card for a call that already
    happened -- reported_eps/surprise_pct straight from finance.data.get_earnings_history, frozen
    at generation time. Unlike TYPE_REMINDER, accumulates one entry per report_date forever (like
    TYPE_TRANSCRIPT), since a past quarter's result stays exactly as true a year later as the day
    it landed. See refresh_earnings_result for why generating this needs a retry window rather than
    a single attempt right when the call happens (yfinance itself often lags a call by a day or two
    before reported_eps actually populates).

None of TYPE_REMINDER/TYPE_RESULT are a live mirror of finance.data.get_earnings_history in
general -- that module's own cache stays the one live source of truth for anything needing the
*full* history (backtests in finance.rules, the Weekly/Metrics tabs' multi-quarter views). These
two types exist purely so the two specific, narrow things app.py's earnings cards need (the next
call's tz-aware numbers, and the latest reported quarter's numbers) are precomputed once in the
batch job, instead of the interactive app touching that cache live on every cold page load.

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

import pandas as pd
import requests
import trafilatura
import yfinance as yf

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


def _open_transcript_window_date(ticker: str, as_of: dt.date) -> dt.date | None:
    """Which of `ticker`'s last few scheduled/reported earnings dates (get_earnings_history) has a
    TRANSCRIPT_WINDOW_{BEFORE,AFTER}_DAYS window currently covering `as_of`, or None if none does.
    Shared by needs_transcript_check (whether to bother searching at all) and refresh_earnings_call
    (to reject a find that isn't actually for this open cycle -- see that function).
    """
    earnings = get_earnings_history(ticker)
    if earnings.empty:
        return None
    for earnings_date in earnings["earnings_date"].dt.date:
        window_start = earnings_date - dt.timedelta(days=TRANSCRIPT_WINDOW_BEFORE_DAYS)
        window_end = earnings_date + dt.timedelta(days=TRANSCRIPT_WINDOW_AFTER_DAYS)
        if window_start <= as_of <= window_end:
            return earnings_date
    return None


def needs_transcript_check(ticker: str, as_of: dt.date) -> bool:
    """True if `ticker` should get a live discovery search right now: `as_of` falls within an open
    transcript window (_open_transcript_window_date) AND the transcript already on record (if any)
    predates *that* earnings date -- so once this cycle's transcript has actually been found and
    stored, checking stops for the rest of the window instead of re-searching every single day until
    AFTER_DAYS closes it. Fails open (returns True) on an empty earnings history, so a data gap means
    "check it" rather than silently never checking a ticker again.
    """
    if get_earnings_history(ticker).empty:
        return True
    window_earnings_date = _open_transcript_window_date(ticker, as_of)
    if window_earnings_date is None:
        return False
    previous = latest_earnings_call(ticker)
    if previous is None:
        return True
    stored_date = dt.date.fromisoformat(previous["transcript_date"])
    return stored_date < window_earnings_date


def _is_current_cycle_transcript(ticker: str, as_of: dt.date, transcript_date: dt.date) -> bool:
    """False if `transcript_date` clearly predates the earnings cycle currently open for `ticker`
    (see _open_transcript_window_date). _find_latest_transcript's auto-discovery just returns
    whichever transcript is most recently available for the ticker overall -- exactly right when
    that's also the call the open window is watching for, but wrong the first time a newly tracked
    ticker is ever checked and its real next call hasn't happened/been published yet: discovery then
    falls back to whatever OLD transcript already existed, and treating that as fresh news is
    misleading. Confirmed live: IREN's first-ever check found its ~3-month-old Q3 transcript instead
    of correctly finding nothing yet for its actual upcoming call.

    True (accept) whenever there's no open window at all -- an explicit manual --ticker check (see
    run_loop_a.py) can run outside the normal window trigger and should keep accepting whatever it
    finds, same as before this existed.
    """
    window_earnings_date = _open_transcript_window_date(ticker, as_of)
    if window_earnings_date is None:
        return True
    window_start = window_earnings_date - dt.timedelta(days=TRANSCRIPT_WINDOW_BEFORE_DAYS)
    return transcript_date >= window_start

TYPE_TRANSCRIPT = "earnings_transcript"
TYPE_PREVIEW = "earnings_preview"
TYPE_REMINDER = "earnings_reminder"
TYPE_RESULT = "earnings_result"

# How long after a report date refresh_earnings_result keeps retrying (see its own docstring for
# why a retry loop is needed at all: yfinance sometimes lags a day or two before reported_eps/
# surprise_pct actually populate for a call that already happened). Wider than
# app.py's own _EARNINGS_RESULT_WINDOW_DAYS (the "Just Reported" banner's display window) on
# purpose -- same separation as TRANSCRIPT_WINDOW_AFTER_DAYS vs app.py's reminder window: the
# generation side needs slack the display side doesn't.
EARNINGS_RESULT_WINDOW_AFTER_DAYS = 7

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


def load_earnings_reminder_history(ticker: str) -> list[dict]:
    """Every earnings-reminder numbers snapshot ever stored for `ticker` (TYPE_REMINDER only),
    oldest first.
    """
    return [e for e in _load_all(ticker) if _entry_type(e) == TYPE_REMINDER]


def latest_earnings_reminder(ticker: str) -> dict | None:
    history = load_earnings_reminder_history(ticker)
    return history[-1] if history else None


def load_earnings_result_history(ticker: str) -> list[dict]:
    """Every reported-quarter snapshot ever stored for `ticker` (TYPE_RESULT only), oldest first --
    one entry per report_date, accumulating forever like TYPE_TRANSCRIPT (unlike TYPE_REMINDER/
    TYPE_PREVIEW, which only ever describe the single upcoming call and get overwritten in place).
    """
    return [e for e in _load_all(ticker) if _entry_type(e) == TYPE_RESULT]


def latest_earnings_result(ticker: str) -> dict | None:
    history = load_earnings_result_history(ticker)
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


def _upsert_reminder(ticker: str, event: dict) -> None:
    """Same call_date-keyed upsert as _upsert_preview, kept as an entirely separate write (own
    TYPE_REMINDER entries) so the preview's own success/failure never gates whether the reminder
    card's numbers exist -- see refresh_earnings_reminder's docstring.
    """
    event = {**event, "type": TYPE_REMINDER}
    history = _load_all(ticker)
    serialized = json.loads(json.dumps(event, default=str))
    existing_idx = next(
        (i for i, e in enumerate(history)
         if _entry_type(e) == TYPE_REMINDER and e.get("call_date") == event["call_date"]),
        None,
    )
    if existing_idx is not None:
        history[existing_idx] = serialized
    else:
        history.append(serialized)
    _save_all(ticker, history)


def _upsert_result(ticker: str, event: dict) -> None:
    """Same call_date-keyed-upsert shape as _upsert_preview/_upsert_reminder, just keyed on
    "report_date" instead -- re-running before/after the same report doesn't duplicate the entry
    (e.g. a corrected surprise_pct a day later overwrites in place rather than appending a second
    row for the same quarter).
    """
    event = {**event, "type": TYPE_RESULT}
    history = _load_all(ticker)
    serialized = json.loads(json.dumps(event, default=str))
    existing_idx = next(
        (i for i, e in enumerate(history)
         if _entry_type(e) == TYPE_RESULT and e.get("report_date") == event["report_date"]),
        None,
    )
    if existing_idx is not None:
        history[existing_idx] = serialized
    else:
        history.append(serialized)
    _save_all(ticker, history)


def refresh_earnings_result(ticker: str, report_date: dt.date) -> dict:
    """Precomputes app.py's "Just Reported" earnings-result card as a static TYPE_RESULT record --
    a fast, LLM-free "results are in" read straight from finance.data.get_earnings_history's own
    reported_eps/surprise_pct, same numbers that module already carries, just frozen into
    output/earnings/{ticker}.json so app.py never has to touch that cache live at render time.

    yfinance sometimes lags a day or two after a call before reported_eps/surprise_pct actually
    populate for it -- calling this once, right when the call happens, could easily see NaNs and
    have nothing to store. So this writes nothing (status "not_yet_reported") until real numbers
    show up, and the caller (finance.newsloop's earnings-result loop) is meant to keep calling this
    once per run for as long as no stored record exists for `report_date` -- the same "no dedup
    file needed, the absence of stored data IS the retry signal" pattern refresh_earnings_call's own
    transcript search already uses. Bounded by EARNINGS_RESULT_WINDOW_AFTER_DAYS on the caller's
    side, not in here.

    Always returns a dict with a "status" key: "generated", "not_yet_reported", or "no_tile" (no row
    at all for `report_date` in finance.data.get_earnings_history -- shouldn't normally happen since
    the caller only calls this for a date that module itself just reported).
    """
    history = get_earnings_history(ticker)
    row = history[history["earnings_date"].dt.date == report_date]
    if row.empty:
        return {"status": "no_tile"}
    row = row.iloc[0]
    if pd.isna(row["reported_eps"]) or pd.isna(row["surprise_pct"]):
        return {"status": "not_yet_reported"}
    event = {
        "report_date": report_date.isoformat(),
        "eps_estimate": None if pd.isna(row["eps_estimate"]) else float(row["eps_estimate"]),
        "reported_eps": float(row["reported_eps"]),
        "surprise_pct": float(row["surprise_pct"]),
    }
    _upsert_result(ticker, event)
    return {**event, "status": "generated"}


def _precise_next_call(ticker: str) -> dict | None:
    """{"when": tz-aware datetime, "eps_estimate": float | None} for `ticker`'s next scheduled (not
    yet reported) earnings call, or None if yfinance has nothing queued. A live yfinance call --
    meant to be used only here, once per ticker actually inside its earnings-reminder window (see
    refresh_earnings_reminder), never on app.py's own request path. yfinance reports these in the
    exchange's own local time (typically America/New_York) -- kept tz-aware here (unlike
    finance.data.get_earnings_history's own cached "earnings_date" column, which strips tz on
    purpose for its own callers) specifically so app.py's card can convert it to Amsterdam time
    correctly.
    """
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=6)
    except Exception:
        return None
    if raw is None or raw.empty:
        return None
    now = pd.Timestamp.now(tz=raw.index.tz)
    upcoming = raw[raw.index > now].sort_index()
    if upcoming.empty:
        return None
    row = upcoming.iloc[0]
    estimate = row.get("EPS Estimate")
    return {
        "when": upcoming.index[0].to_pydatetime(),
        "eps_estimate": None if pd.isna(estimate) else float(estimate),
    }


def _reported_quarters_summary(ticker: str, n: int = 4) -> list[dict]:
    """The last `n` *reported* quarters (date/estimate/actual/surprise), oldest first -- a frozen
    snapshot of exactly what app.py's card needs for its beat/miss streak text, computed once here
    rather than the card re-deriving it from a live finance.data.get_earnings_history read.
    """
    history = get_earnings_history(ticker).dropna(subset=["reported_eps", "surprise_pct"])
    history = history.sort_values("earnings_date").tail(n)
    return [
        {
            "earnings_date": row["earnings_date"].date().isoformat(),
            "eps_estimate": None if pd.isna(row["eps_estimate"]) else float(row["eps_estimate"]),
            "reported_eps": float(row["reported_eps"]),
            "surprise_pct": float(row["surprise_pct"]),
        }
        for _, row in history.iterrows()
    ]


def refresh_earnings_reminder(ticker: str, call_date: dt.date) -> dict:
    """Precomputes everything app.py's earnings-reminder card needs for its numbers half -- the
    precise call date/time, consensus EPS estimate, and last few reported quarters' beat/miss
    streak -- as a static TYPE_REMINDER record in output/earnings/{ticker}.json, keyed on call_date
    same as refresh_earnings_preview's own TYPE_PREVIEW entries (see _upsert_reminder). Meant to be
    called whenever a ticker enters its earnings-reminder window, independently of whether
    refresh_earnings_preview succeeds for that same cycle -- a Finnhub outage or LLM failure on the
    narrative side shouldn't also blank out the plain, deterministic numbers side of the card.

    Always returns a dict with a "status" key: "generated", or "no_tile" if yfinance had nothing
    queued for this ticker right now, or its live next-call date no longer matches `call_date`
    (e.g. the call got rescheduled since `call_date` was computed) -- caller should just skip and
    retry next run rather than store a mismatched record.
    """
    precise = _precise_next_call(ticker)
    if precise is None or precise["when"].date() != call_date:
        return {"status": "no_tile"}
    event = {
        "call_date": call_date.isoformat(),
        "when": precise["when"].isoformat(),
        "eps_estimate": precise["eps_estimate"],
        "reported_quarters": _reported_quarters_summary(ticker),
    }
    _upsert_reminder(ticker, event)
    return {**event, "status": "generated"}


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
        # Reject a stale find -- see _is_current_cycle_transcript's own docstring for the exact
        # failure this closes (a first-ever check surfacing an old leftover transcript instead of
        # correctly finding nothing for the call actually being watched for). Only applies to
        # auto-discovery -- an explicit transcript_url is a deliberate manual override, always
        # accepted regardless of cycle.
        if found is not None and not _is_current_cycle_transcript(ticker, as_of, found[1]):
            found = None
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
