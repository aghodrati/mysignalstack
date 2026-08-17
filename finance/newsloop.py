"""Loop A: News -> LLM -> trade.

Global research (shared by every portfolio) -> per-portfolio trading (cheap,
deterministic, no LLM). Three LLM stages, the first two on never-before-seen
work only (tracked in persistent, permanent archives so a re-run never
reprocesses anything):

Stage A (`extract_event`, once per article, global): classifies the article
-- what happened, which tracked-universe tickers it materially concerns, how
important/novel it is, what kind of event it is. Company-agnostic and cheap
regardless of how many companies get mentioned.

Stage B (`extract_claims`, once per (article, ticker) Stage A flagged,
global): every distinct, independent claim this article makes about that
ticker -- zero, one, or several (a positive near-term catalyst and a
separate longer-term risk are two claims, not one forced synthesis). Each
claim is self-contained, permanent, and never revised afterward -- unlike
the old design, this has no portfolio-specific "prior draft" to react to, so
it's cacheable globally exactly like Stage A (see finance.claims).

Stage C (`finance.tickerthesis.aggregate_claims`, whenever a new trade-
worthy claim lands for a ticker, global): synthesizes every claim ever
collected for that ticker into one current claim/direction/confidence.
Weighs each claim by its own confidence/importance rather than treating the
list flatly; that weighting is the (deliberately soft, not a hard numeric
rule) protection against any single new or low-importance claim overriding
an established consensus. Also runs a fundamental second opinion
(finance.fundamentals) once per update, blended into confidence and folded
into invalidation, same soft-influence-only design as before.

Whenever a ticker is newly added to config_loop_a.json (finance.loop_a_config), `update_research`
also runs a one-time backfill sweep (`_backfill_new_tickers`) before its normal pass: a free text
pre-filter over the permanent article archive, then a real Stage A/B re-run only on articles from
the last BACKFILL_LOOKBACK_DAYS days that plausibly mention it -- so a newly tracked ticker picks
up recent history instead of only starting from whatever news happens to land after it's added.
Bounded and one-shot per ticker (tracked in cache/thesis/known_universe_tickers.json); removing a
ticker from the config never triggers anything or touches existing claims/theses.

Everything above is global and portfolio-independent -- computed once,
shared by every portfolio, regardless of which one's `run_loop_a` call
happens to trigger it (`update_research`). Each portfolio then runs a thin,
deterministic, LLM-free trade-decision pass (`run_loop_a`, after
`update_research`): for every ticker with a TickerThesis, open a position if
not already held and confidence clears the bar, or close one if already
held and the current TickerThesis no longer supports it. finance.thesis
tracks only these thin per-portfolio position records (opened/closed, at
what confidence) -- the actual claim/catalysts/invalidation content lives in
finance.tickerthesis and finance.claims, shared research any portfolio can
inspect.

Meant to run once a day, forward from today. Running forward (not against
historical dates) is a deliberate leakage-avoidance choice: the LLM's own
training data can't leak information about news that hasn't happened yet.

Stage A and B reason and ground their price/earnings signals as of each
article's own publish date, not the day the run happens to process it --
otherwise a backlog article processed late (e.g. after a RateLimited pause)
would get today's price action and today's leakage cutoff instead of what
was actually knowable when it came out. Stage C and the fundamental opinion
are different: they synthesize the *current* view across every claim and
query live fundamental data, so they correctly use the run's own date
(`as_of`), not any single article's.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd

from finance.claims import (
    HORIZON_MAX_DAYS,
    HORIZON_MIN_DAYS,
    ArticleClaim,
    append_claims,
    claim_id,
    list_tickers_with_claims,
)
from finance.data import get_earnings_history, get_prices
from finance.llm import RateLimited, complete, get_rate_limit_status, get_usage_counter, reset_usage_counter
from finance.loop_a_config import active_news_sources, llm_config, max_article_chars, tracked_universe
from finance.news import SEC_8K_SOURCE_PREFIX, get_news_from_sources, get_sec_8k_news
from finance.portfolio import append_trade, current_state, execution_price, load_meta, rule_positions
from finance.thesis import Position, append_closed, append_created, open_positions
from finance.tickerthesis import (
    TickerThesis,
    list_tickers_with_thesis,
    load_ticker_thesis,
    refresh_fundamentals,
    update_ticker_thesis,
)
from finance.xbrl import get_cik_map


def _truncate_article(text: str) -> str:
    """Applies Loop A's configured max_article_chars cap (see
    finance.loop_a_config), if any -- unlimited by default. Called at every
    point Stage A/B embed article text in a prompt, so switching to a
    smaller-context model is a config edit, not a code edit.
    """
    limit = max_article_chars()
    return text[:limit] if limit is not None else text

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "thesis"
# Global, shared across every portfolio: neither Stage A's nor Stage B's output depends on which
# portfolio is asking, so it's wasteful (and costs Groq quota) to redo either per portfolio.
EVENTS_CACHE_PATH = CACHE_DIR / "events.json"
CLAIMS_ATTEMPTED_PATH = CACHE_DIR / "claims_attempted.json"
# Tracks consecutive aggregation failures per ticker (aggregate_claims/fundamental_opinion returning
# nothing usable) -- lets the self-heal sweep in update_research back off a ticker that keeps failing
# instead of blindly retrying it (2 real LLM calls) every single run forever. A ticker with a genuine
# *new* claim always gets a fresh attempt regardless (see update_research) -- this only throttles
# retries with no new evidence behind them. Cleared the moment a ticker succeeds.
AGGREGATION_ATTEMPTS_PATH = CACHE_DIR / "aggregation_attempts.json"
MAX_SELF_HEAL_RETRIES = 3
# Tracks, per ticker, the earnings_date (isoformat) already refreshed for -- so a ticker sitting
# inside its earnings window across several consecutive daily runs only gets one fundamentals
# refresh per event, not one every run. Keyed on the earnings date itself (not a boolean) so a
# *later* earnings event is always free to trigger again.
EARNINGS_FUNDAMENTAL_CHECKS_PATH = CACHE_DIR / "earnings_fundamental_checks.json"
# Fundamentals (compute_fundamental_factors + fundamental_opinion) are otherwise only refreshed as
# a side effect of new news-driven claims -- a ticker with no news around its own earnings report
# would go stale exactly when the underlying numbers are most likely to have moved. This forces a
# refresh in a window around every scheduled/reported earnings date regardless of news.
EARNINGS_WINDOW_BEFORE_DAYS = 2
EARNINGS_WINDOW_AFTER_DAYS = 1

RULE_NAME = "loop_a"

# Confidence tiers -> position size as a % of current portfolio value.
# Below CONFIDENCE_MIN, a ticker-thesis is never traded.
CONFIDENCE_MIN = 0.60
CONFIDENCE_FULL = 0.75
POSITION_PCT_FULL = 0.05
POSITION_PCT_HALF = 0.025
MIN_TRADE_DOLLARS = 50.0
MAX_CONCURRENT_POSITIONS = 8

# Per-portfolio strategy knobs -- stored in meta.json under "rules"."loop_a" (see
# finance.portfolio.save_rule_settings, the same mechanism every other rule already uses to
# remember its own settings), chosen once at portfolio creation (see app.py). A portfolio with no
# such entry (every one created before this existed) resolves every key to "balanced"/"any" --
# exactly the hardcoded globals above -- so nothing about an existing portfolio's behavior changes.
RISK_PROFILES = {
    # name -> (confidence_min, confidence_full)
    "conservative": (0.75, 0.85),
    "balanced": (CONFIDENCE_MIN, CONFIDENCE_FULL),
    "aggressive": (0.50, 0.65),
}
CONCENTRATION_PROFILES = {
    # name -> (position_pct_full, position_pct_half, max_concurrent_positions)
    "diversified": (0.03, 0.015, 12),
    "balanced": (POSITION_PCT_FULL, POSITION_PCT_HALF, MAX_CONCURRENT_POSITIONS),
    "concentrated": (0.10, 0.05, 4),
}
HORIZON_PROFILES = {
    # name -> (min_days, max_days) a ticker-thesis's own expected_horizon_days must fall within to
    # be *opened* -- doesn't affect closing, review_loop_a always closes on the thesis's own
    # horizon regardless of this setting.
    "short_term": (HORIZON_MIN_DAYS, 90),
    "long_term": (90, HORIZON_MAX_DAYS),
    "any": (HORIZON_MIN_DAYS, HORIZON_MAX_DAYS),
}


def _portfolio_strategy(portfolio_name: str) -> dict:
    """Resolves a portfolio's loop_a settings from meta.json into concrete numeric thresholds."""
    settings = load_meta(portfolio_name).get("rules", {}).get(RULE_NAME, {})
    conf_min, conf_full = RISK_PROFILES[settings.get("risk_profile", "balanced")]
    pct_full, pct_half, max_positions = CONCENTRATION_PROFILES[settings.get("concentration", "balanced")]
    horizon_min, horizon_max = HORIZON_PROFILES[settings.get("horizon", "any")]
    return {
        "confidence_min": conf_min, "confidence_full": conf_full,
        "position_pct_full": pct_full, "position_pct_half": pct_half,
        "max_concurrent_positions": max_positions,
        "horizon_min_days": horizon_min, "horizon_max_days": horizon_max,
    }

# Stage A gate: an event below this on its own self-reported 0-10 importance
# scale never reaches Stage B. Treat the score as a coarse noise filter, not
# a precise ranking -- small free-tier models aren't well-calibrated on it.
IMPORTANCE_MIN = 3

EVENT_TYPES = (
    "capex", "earnings", "guidance", "product_launch", "regulatory", "m_and_a",
    "leadership_change", "supply_chain", "competitive", "macro", "other",
)
TIME_HORIZONS = ("days", "weeks", "months", "quarters")
ARTICLE_TYPES = ("news", "analysis")

_TAG_RE = re.compile(r"<[^>]+>")
_JSON_BLOB_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_html(raw_html: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", _TAG_RE.sub(" ", raw_html))).strip()


def _safe_text(value) -> str:
    """News feed fields round-trip through a CSV cache, so an empty field
    comes back as NaN (a float), not "" -- and `if value:` alone would treat
    NaN as truthy (it's non-zero), so this is not a redundant check.
    """
    return value if isinstance(value, str) else ""


def _load_events() -> dict:
    if EVENTS_CACHE_PATH.exists():
        return json.loads(EVENTS_CACHE_PATH.read_text())
    return {}


def get_article_archive() -> dict:
    """The permanent global archive keyed by article link -- {"title", "text",
    "event"} per article. Public read-only alias of `_load_events` for
    callers outside this module (namely app.py, to show a claim's original
    source text/classification) that just want to look articles up, not
    touch the cache.
    """
    return _load_events()


def _save_events(events: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_CACHE_PATH.write_text(json.dumps(events, indent=2))


def _load_claims_attempted() -> dict:
    if CLAIMS_ATTEMPTED_PATH.exists():
        return json.loads(CLAIMS_ATTEMPTED_PATH.read_text())
    return {}


def _save_claims_attempted(attempted: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CLAIMS_ATTEMPTED_PATH.write_text(json.dumps(attempted, indent=2))


def _load_aggregation_attempts() -> dict:
    if AGGREGATION_ATTEMPTS_PATH.exists():
        return json.loads(AGGREGATION_ATTEMPTS_PATH.read_text())
    return {}


def _save_aggregation_attempts(attempts: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    AGGREGATION_ATTEMPTS_PATH.write_text(json.dumps(attempts, indent=2))


def _load_earnings_fundamental_checks() -> dict:
    if EARNINGS_FUNDAMENTAL_CHECKS_PATH.exists():
        return json.loads(EARNINGS_FUNDAMENTAL_CHECKS_PATH.read_text())
    return {}


def _save_earnings_fundamental_checks(checks: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EARNINGS_FUNDAMENTAL_CHECKS_PATH.write_text(json.dumps(checks, indent=2))


def _earnings_trigger_date(ticker: str, as_of: dt.date) -> dt.date | None:
    """The earnings date `as_of` currently falls within the refresh window
    of ([earnings_date - EARNINGS_WINDOW_BEFORE_DAYS, earnings_date +
    EARNINGS_WINDOW_AFTER_DAYS]), if any -- checks every date
    get_earnings_history returns (both the next scheduled report and recent
    past ones), not just the upcoming one, so a report that already
    happened 1 day ago still triggers.
    """
    earnings = get_earnings_history(ticker, limit=4)
    if earnings.empty:
        return None
    for earnings_date in earnings["earnings_date"].dt.date:
        window_start = earnings_date - dt.timedelta(days=EARNINGS_WINDOW_BEFORE_DAYS)
        window_end = earnings_date + dt.timedelta(days=EARNINGS_WINDOW_AFTER_DAYS)
        if window_start <= as_of <= window_end:
            return earnings_date
    return None


def _universe_name_to_ticker() -> dict[str, str]:
    """Every configured display name and its ticker (finance.loop_a_config),
    plus every ticker mapped to itself (so a bare symbol mention also
    resolves) -- the candidate list `extract_event` is shown and the set its
    extracted tickers are validated against.
    """
    combined: dict[str, str] = {}
    for ticker, name in tracked_universe().items():
        combined[name] = ticker
        combined[ticker] = ticker
    return combined


def compute_signal_bundle(ticker: str, as_of: dt.date) -> dict:
    """Deterministic, LLM-free numbers handed to Stage B as grounding:
    trailing return, whether today itself was a dip, and the most recent
    earnings surprise if it happened within the last 10 days.
    """
    signals: dict = {
        "as_of": as_of.isoformat(),
        "trailing_return_5d_pct": None,
        "trailing_return_1m_pct": None,
        "dip_today_pct": None,
        "recent_earnings_surprise_pct": None,
        "days_since_earnings": None,
    }

    start = (as_of - dt.timedelta(days=40)).isoformat()
    end = (as_of + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    prices = get_prices([ticker], start=start, end=end)
    prices = prices.loc[prices.index <= pd.Timestamp(as_of)]
    series = prices[ticker].dropna() if ticker in prices.columns else pd.Series(dtype=float)

    if len(series) >= 2:
        last = float(series.iloc[-1])
        daily_return = series.iloc[-1] / series.iloc[-2] - 1
        if daily_return <= -0.03:
            signals["dip_today_pct"] = round(float(daily_return) * 100, 2)
        if len(series) >= 6:
            signals["trailing_return_5d_pct"] = round(last / float(series.iloc[-6]) - 1, 4) * 100
        if len(series) >= 21:
            signals["trailing_return_1m_pct"] = round(last / float(series.iloc[-21]) - 1, 4) * 100

    earnings = get_earnings_history(ticker, limit=4).dropna(subset=["reported_eps", "surprise_pct"])
    if not earnings.empty:
        last_report = earnings.sort_values("earnings_date").iloc[-1]
        days_since = (pd.Timestamp(as_of) - last_report["earnings_date"]).days
        if 0 <= days_since <= 10:
            signals["recent_earnings_surprise_pct"] = round(float(last_report["surprise_pct"]), 2)
            signals["days_since_earnings"] = int(days_since)

    return signals


def _extract_json(text: str) -> dict | None:
    match = _JSON_BLOB_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_EVENT_REQUIRED_FIELDS = {"event", "companies", "event_type", "importance", "time_horizon", "key_facts", "type", "novel"}


def extract_event(
    article_title: str, article_text: str, universe_map: dict[str, str], as_of: dt.date, debug: bool = False
) -> dict | None:
    """Stage A, one LLM call per article: classifies what happened, which
    tracked-universe tickers it materially concerns, how important/novel it
    is, and what kind of event it is. Never raises; returns None on any
    parse/validation failure -- pass debug=True to print *which* failure mode
    it was (no response at all / unparseable JSON / missing fields), since
    otherwise they're indistinguishable from the caller's side.
    """
    # universe_map has two entries per ticker (its real display name -> ticker, and the bare
    # ticker -> itself, so a bare symbol mention also resolves -- see _universe_name_to_ticker).
    # Naively filtering out entries where name == ticker (to skip the redundant bare-ticker one)
    # silently drops a ticker ENTIRELY whenever its configured display name equals its ticker
    # (e.g. "SPCX": "SPCX") -- both insertions collide on the same dict key, leaving only the
    # name==ticker entry, which then gets filtered away with nothing left to show it was ever
    # tracked. This instead picks one display name per ticker, preferring a real distinct name
    # if one exists, but always falling back to the ticker itself rather than dropping it.
    display_names: dict[str, str] = {}
    for name, ticker in universe_map.items():
        if ticker not in display_names or name != ticker:
            display_names[ticker] = name
    tracked_display = ", ".join(f"{name} ({ticker})" for ticker, name in sorted(display_names.items()))
    prompt = (
        f"You are a financial news classifier. Today's date is {as_of.isoformat()}. Reason only "
        f"from the article text below -- do not rely on outside knowledge of what happened after "
        f"{as_of.isoformat()}.\n\n"
        f"Companies I track (name (ticker)): {tracked_display}\n\n"
        f"Article title: {article_title}\n"
        f"Article text: {_truncate_article(article_text)}\n\n"
        f"Extract a structured summary of this article as ONLY a JSON object (no markdown "
        f"fences, no commentary):\n"
        f'{{"event": "one sentence describing what happened", '
        f'"companies": ["TICKER", ...] (only tickers from the tracked list above that are '
        f"materially and specifically affected, not passing mentions -- empty list if none), "
        f'"event_type": one of {list(EVENT_TYPES)}, '
        f'"importance": integer 0-10 (0 = pure noise, 10 = could plausibly change the affected '
        f"companies' multi-month outlook), "
        f'"time_horizon": one of {list(TIME_HORIZONS)} (how long this event\'s effects would '
        f"plausibly take to play out), "
        f'"key_facts": ["specific factual claim from the article", ...] (list every distinct '
        f"factual claim relevant to the companies above -- most articles support several; "
        f"only give one if the article genuinely makes just a single relevant claim), "
        f'"type": "news" or "analysis" (news = reporting a discrete event/fact; analysis = '
        f"commentary/opinion/synthesis), "
        f'"novel": true/false (genuinely new information, not a rehash of something already '
        f"widely known/priced in)}}"
    )

    raw = complete(prompt, max_tokens=1500, **llm_config())
    if not raw:
        if debug:
            print("      [extract_event] no response from the LLM (rate limit or API failure)")
        return None
    data = _extract_json(raw)
    if data is None:
        if debug:
            print(f"      [extract_event] response wasn't valid/parseable JSON: {raw[:200]!r}")
        return None
    if not _EVENT_REQUIRED_FIELDS.issubset(data):
        if debug:
            print(f"      [extract_event] missing fields {_EVENT_REQUIRED_FIELDS - set(data)}: {raw[:200]!r}")
        return None

    valid_tickers = set(universe_map.values())
    data["companies"] = sorted({t for t in data["companies"] if t in valid_tickers})
    if data["event_type"] not in EVENT_TYPES:
        data["event_type"] = "other"
    if data["time_horizon"] not in TIME_HORIZONS:
        data["time_horizon"] = "months"
    if data["type"] not in ARTICLE_TYPES:
        data["type"] = "news"
    try:
        data["importance"] = max(0, min(10, int(data["importance"])))
    except (TypeError, ValueError):
        data["importance"] = 0
    data["novel"] = bool(data["novel"])
    return data


_KNOWN_COMPANY_EVENT_REQUIRED_FIELDS = {"event", "event_type", "importance", "time_horizon", "key_facts", "novel"}


def extract_event_known_company(
    article_title: str, article_text: str, ticker: str, as_of: dt.date, debug: bool = False
) -> dict | None:
    """Cheaper Stage A variant for sources where the company is already known
    deterministically (currently: SEC 8-K filings, via finance.news's
    get_sec_8k_news) -- skips the tracked-universe company list and the
    company-detection instruction entirely, since there's nothing to detect.
    That list is the majority of extract_event's prompt size, so this matters
    on a high-volume per-ticker source like SEC filings. Same output shape as
    extract_event (companies is always just [ticker], type is always "news").
    See extract_event's docstring re: debug.
    """
    prompt = (
        f"You are a financial filing classifier. Today's date is {as_of.isoformat()}. This is an "
        f"SEC filing concerning {ticker}. Reason only from the text below -- do not rely on "
        f"outside knowledge of what happened after {as_of.isoformat()}.\n\n"
        f"Filing: {article_title}\nDetails: {_truncate_article(article_text)}\n\n"
        f"Respond with ONLY a JSON object (no markdown fences, no commentary):\n"
        f'{{"event": "one sentence describing what this filing is about", '
        f'"event_type": one of {list(EVENT_TYPES)}, '
        f'"importance": integer 0-10 (0 = routine/procedural, 10 = could plausibly change '
        f"{ticker}'s multi-month outlook -- most 8-Ks are routine, be conservative), "
        f'"time_horizon": one of {list(TIME_HORIZONS)}, '
        f'"key_facts": ["specific factual claim from the filing", ...], '
        f'"novel": true/false (substantive news, not just a routine/procedural filing)}}'
    )

    raw = complete(prompt, max_tokens=300, **llm_config())
    if not raw:
        if debug:
            print("      [extract_event_known_company] no response from the LLM (rate limit or API failure)")
        return None
    data = _extract_json(raw)
    if data is None:
        if debug:
            print(f"      [extract_event_known_company] response wasn't valid/parseable JSON: {raw[:200]!r}")
        return None
    if not _KNOWN_COMPANY_EVENT_REQUIRED_FIELDS.issubset(data):
        if debug:
            missing = _KNOWN_COMPANY_EVENT_REQUIRED_FIELDS - set(data)
            print(f"      [extract_event_known_company] missing fields {missing}: {raw[:200]!r}")
        return None

    data["companies"] = [ticker]
    data["type"] = "news"
    if data["event_type"] not in EVENT_TYPES:
        data["event_type"] = "other"
    if data["time_horizon"] not in TIME_HORIZONS:
        data["time_horizon"] = "months"
    try:
        data["importance"] = max(0, min(10, int(data["importance"])))
    except (TypeError, ValueError):
        data["importance"] = 0
    data["novel"] = bool(data["novel"])
    return data


_CLAIM_REQUIRED_FIELDS = {"trade_worthy", "claim", "confidence", "importance", "direction"}
_TRADE_WORTHY_REQUIRED_FIELDS = {"expected_horizon_days", "expected_return_pct"}


def extract_claims(
    article_text: str, ticker: str, event: dict, signals: dict, as_of: dt.date, debug: bool = False
) -> list[dict] | None:
    """Stage B: every distinct, independent claim this article makes about
    `ticker` -- zero, one, or several (a positive near-term catalyst and a
    separate longer-term risk are two claims, not one forced synthesis).
    Each claim is self-contained, carries its own confidence/importance, and
    is never revised afterward -- a pure function of the article alone, no
    portfolio-specific context, so it's cacheable globally (see
    finance.claims). finance.tickerthesis's aggregator synthesizes across
    every claim collected over time.

    Returns None only for a genuine failure (no LLM response, unparseable
    JSON) -- distinct from an empty list, which means the article
    legitimately made no claims about this ticker worth recording. Never
    raises for a parse failure. Raises RateLimited if every model is
    currently out of quota, same as every other Stage call. Pass debug=True
    to print *which* outcome it was (no response / unparseable JSON /
    genuinely empty / N claims kept, M dropped as malformed) -- otherwise a
    "ticker processed" log line looks identical whether it produced claims
    or not, since both consume real tokens.
    """
    event_context = {k: event[k] for k in ("event", "event_type", "importance", "time_horizon", "key_facts", "type")}
    prompt = (
        f"You are an investment research assistant. Today's date is {as_of.isoformat()}. Reason "
        f"only from the information given -- no outside knowledge of what happened after "
        f"{as_of.isoformat()}.\n\n"
        f"An article was classified as: {json.dumps(event_context)}\n\n"
        f"Full article text: {_truncate_article(article_text)}\n\n"
        f"Computed price/earnings signals for {ticker} as of {as_of.isoformat()}: {json.dumps(signals)}\n\n"
        f"List every distinct, independent investment claim this article makes about {ticker} -- a "
        f'claim is a specific, self-contained implication ("this changes the outlook up or down", '
        f'not just "this is relevant"). A single article can carry more than one: e.g. a positive '
        f"near-term catalyst AND a separate longer-term risk are two claims, not one. If the article's "
        f"relevance to {ticker} is too vague, mixed, or indirect to state as a specific claim, return "
        f"an empty list.\n\n"
        f'Respond with ONLY a JSON object (no markdown fences, no commentary): {{"claims": [...]}} '
        f"where each item is:\n"
        f'{{"claim": "one sentence: the specific claim", '
        f'"context": "one paragraph, specific to THIS claim, giving a reader enough of the '
        f"surrounding article context (what was said, why it matters) to understand the claim "
        f"without re-reading the whole article -- not a restatement of the one-sentence claim, and "
        f"not a summary of the whole article either, just this claim's own context, "
        f'"trade_worthy": true only if you could also confidently give a specific '
        f"expected_return_pct and expected_horizon_days for this claim alone -- i.e. it names a "
        f"fairly direct mechanism and a plausible timeframe, not just a general risk/opportunity "
        f'with no anchor. false if the claim is real and worth recording but too indirect, hedged '
        f'("could potentially..."), or open-ended to size a number on. This is independent of '
        f"importance/confidence below -- a claim can be important and high-confidence while still "
        f"not being precise enough to size, and vice versa, "
        f'"confidence": number between 0 and 1 (your confidence in this specific claim), '
        f'"importance": integer 0-10 (how central this claim is -- 0 = a minor aside, 10 = the '
        f"article's main point for this company), "
        f'"direction": "long" or "short" (always required, even if trade_worthy is false -- does '
        f"this claim, on its own, lean bullish or bearish for {ticker}? give your best-guess lean "
        f"even for a vague/indirect claim; trade_worthy is about whether it's precise enough to "
        f"size a trade on, not whether it has a direction at all), "
        f'"expected_horizon_days": integer between {HORIZON_MIN_DAYS} and {HORIZON_MAX_DAYS} (how '
        f"long this specific claim would plausibly take to play out -- required if trade_worthy is "
        f"true), "
        f'"expected_return_pct": number (required if trade_worthy is true)}}'
    )

    raw = complete(prompt, max_tokens=3500, **llm_config())
    if not raw:
        if debug:
            print(f"      [extract_claims:{ticker}] no response from the LLM (rate limit or API failure)")
        return None
    data = _extract_json(raw)
    if data is None or not isinstance(data.get("claims"), list):
        if debug:
            print(f"      [extract_claims:{ticker}] response wasn't valid/parseable JSON: {raw[:200]!r}")
        return None

    claims: list[dict] = []
    dropped = 0
    for item in data["claims"]:
        if (
            not isinstance(item, dict)
            or not _CLAIM_REQUIRED_FIELDS.issubset(item)
            or item.get("direction") not in ("long", "short")
        ):
            dropped += 1
            continue  # drop malformed individual entries rather than failing the whole batch
        trade_worthy = bool(item["trade_worthy"])
        cleaned = {
            "claim": str(item["claim"]), "trade_worthy": trade_worthy, "direction": item["direction"],
            "confidence": max(0.0, min(1.0, float(item["confidence"]))),
            "importance": max(0, min(10, int(item["importance"]))),
            "context": str(item.get("context", "")),  # optional -- never drop an otherwise-good claim for lacking it
        }
        if trade_worthy:
            if not _TRADE_WORTHY_REQUIRED_FIELDS.issubset(item):
                dropped += 1
                continue  # can't size a trade without these -- drop rather than half-fill
            cleaned.update(
                expected_horizon_days=int(
                    max(HORIZON_MIN_DAYS, min(HORIZON_MAX_DAYS, item["expected_horizon_days"]))
                ),
                expected_return_pct=float(item["expected_return_pct"]),
            )
        else:
            cleaned.update(expected_horizon_days=0, expected_return_pct=0.0)
        claims.append(cleaned)
    if debug:
        if not claims:
            reason = f", {dropped} dropped as malformed" if dropped else ""
            print(f"      [extract_claims:{ticker}] parsed OK but no usable claims{reason} -- article deemed too vague/indirect for {ticker}")
        elif dropped:
            print(f"      [extract_claims:{ticker}] kept {len(claims)} claim(s), dropped {dropped} malformed")
    return claims


def position_id(portfolio_name: str, ticker: str, as_of: dt.date) -> str:
    return hashlib.sha1(f"{portfolio_name}|{ticker}|{as_of.isoformat()}".encode()).hexdigest()[:16]


def _position_pct(confidence: float, strategy: dict) -> float | None:
    if confidence >= strategy["confidence_full"]:
        return strategy["position_pct_full"]
    if confidence >= strategy["confidence_min"]:
        return strategy["position_pct_half"]
    return None


def _portfolio_value(portfolio_name: str, as_of: dt.date) -> float:
    cash, positions = current_state(portfolio_name)
    value = cash
    for ticker, row in positions.iterrows():
        try:
            value += row["shares"] * execution_price(ticker, as_of)
        except ValueError:
            continue
    return value


def _try_open(portfolio_name: str, tt: TickerThesis, as_of: dt.date, strategy: dict) -> dict | None:
    """Attempts the entry trade for a ticker-thesis. Returns the trade log
    dict on success, None if it wasn't traded (short direction, confidence
    too low, horizon outside this portfolio's preference, at position cap,
    insufficient cash, or no price available).
    """
    if tt.direction != "long":
        return None
    if not (strategy["horizon_min_days"] <= tt.expected_horizon_days <= strategy["horizon_max_days"]):
        return None
    pct = _position_pct(tt.confidence, strategy)
    if pct is None:
        return None
    if len(rule_positions(portfolio_name, RULE_NAME)) >= strategy["max_concurrent_positions"]:
        return None

    value = _portfolio_value(portfolio_name, as_of)
    cash, _ = current_state(portfolio_name)
    unit_size = min(value * pct, cash)
    if unit_size < MIN_TRADE_DOLLARS:
        return None

    try:
        price = execution_price(tt.ticker, as_of)
    except ValueError:
        return None

    shares = unit_size / price
    pid = position_id(portfolio_name, tt.ticker, as_of)
    append_trade(
        portfolio_name, as_of, tt.ticker, "BUY", shares, price,
        note=(
            f"loop_a {pid} entry_conf={tt.confidence:.0%} horizon={tt.expected_horizon_days}d: "
            f"{tt.thesis[:80]}"
        ),
        rule=RULE_NAME,
    )
    position = Position(
        id=pid, ticker=tt.ticker, created=as_of,
        entry_confidence=tt.confidence, expected_horizon_days=tt.expected_horizon_days, status="open",
    )
    append_created(portfolio_name, position)
    return {"action": "BUY", "ticker": tt.ticker, "shares": shares, "price": price, "position_id": pid}


def _close_position_now(portfolio_name: str, position: Position, as_of: dt.date, reason: str) -> dict | None:
    """Sells the position and records the realized return. Shared by
    review_loop_a (horizon reached) and run_loop_a's trade-decision pass
    (the ticker-thesis no longer supports holding). Returns the trade log
    dict, or None if there's no matching live position (defensive --
    shouldn't happen given status="open" implies one).
    """
    positions = rule_positions(portfolio_name, RULE_NAME)
    if position.ticker not in positions.index:
        return None
    row = positions.loc[position.ticker]
    try:
        price = execution_price(position.ticker, as_of)
    except ValueError:
        return None

    actual_return_pct = (price / row["avg_cost"] - 1) * 100
    append_trade(
        portfolio_name, as_of, position.ticker, "SELL", row["shares"], price,
        note=f"loop_a {position.id} closed ({reason}, return={actual_return_pct:+.1f}%)", rule=RULE_NAME,
    )
    append_closed(portfolio_name, position.id, as_of, actual_return_pct, reason=reason)
    return {
        "action": "SELL", "ticker": position.ticker, "position_id": position.id,
        "actual_return_pct": actual_return_pct, "reason": reason,
    }


def _extract_and_store_claims(
    ticker: str, link: str, title: str, article_text: str, event: dict, article_date: dt.date,
    claims_attempted: dict, verbose: bool,
) -> bool:
    """Stage B for one (article, ticker) pair: extracts claims, stores them, and marks the pair
    attempted -- shared by update_research's main per-article loop and the new-ticker backfill
    sweep below, so both go through identical claim-extraction/storage logic. Returns True if at
    least one trade-worthy claim was stored (the caller's signal to (re)run Stage C for this
    ticker).
    """
    signals = compute_signal_bundle(ticker, article_date)
    if verbose:
        print(f"    [Stage B] extracting claims for {ticker}")
    extracted = extract_claims(article_text, ticker, event, signals, article_date, debug=verbose)
    # Marked immediately after the call returns (success or a genuine parse failure) -- a crash
    # mid-article never reprocesses it, same reasoning Stage A's cache uses. RateLimited
    # propagates *before* this line, so a rate-limited ticker is correctly left unmarked and
    # retried next call.
    claims_attempted.setdefault(link, []).append(ticker)
    _save_claims_attempted(claims_attempted)
    if not extracted:
        return False
    article_claims = [
        ArticleClaim(
            id=claim_id(ticker, link, c["claim"]), ticker=ticker, source_link=link,
            source_title=title, event_type=event["event_type"], created=article_date, claim=c["claim"],
            direction=c["direction"], confidence=c["confidence"], importance=c["importance"],
            expected_return_pct=c["expected_return_pct"],
            expected_horizon_days=c["expected_horizon_days"], trade_worthy=c["trade_worthy"],
            context=c["context"],
        )
        for c in extracted
    ]
    append_claims(ticker, article_claims)
    return any(c.trade_worthy for c in article_claims)


# How far back the new-ticker backfill sweep looks when a ticker is newly added to
# config_loop_a.json -- bounds the cost of adding a ticker to a fixed window rather than
# reprocessing the entire permanent article archive.
BACKFILL_LOOKBACK_DAYS = 90
# Persisted set of tickers whose backfill sweep has already run (or was never needed, e.g. a
# ticker present the first time this feature shipped) -- diffed against the current config each
# run so a ticker is only ever backfilled once, the moment it's added, never again afterward.
# Removing a ticker from the config never touches this file or any existing claims/theses.
KNOWN_UNIVERSE_TICKERS_PATH = CACHE_DIR / "known_universe_tickers.json"


def _load_known_universe_tickers() -> set[str]:
    if KNOWN_UNIVERSE_TICKERS_PATH.exists():
        return set(json.loads(KNOWN_UNIVERSE_TICKERS_PATH.read_text()))
    return set()


def _save_known_universe_tickers(tickers: set[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    KNOWN_UNIVERSE_TICKERS_PATH.write_text(json.dumps(sorted(tickers), indent=2))


def _backfill_new_tickers(
    events_cache: dict, claims_attempted: dict, universe_map: dict[str, str],
    new_tickers: dict[str, str], as_of: dt.date, verbose: bool,
) -> tuple[set[str], set[str]]:
    """For every ticker newly added to config_loop_a.json (not yet in the known-universe
    snapshot), retroactively reconsiders cached articles from the last BACKFILL_LOOKBACK_DAYS
    days: a free text pre-filter first (no LLM cost, just a substring search over the permanent
    archive), then a real Stage A re-classification only on matches, so adding a ticker doesn't
    silently reprocess the whole archive. A match that Stage A still doesn't consider materially
    about the new ticker (or fails the importance/novelty gate, same as the main loop) produces
    no claim, same as if it had been seen for the first time today.

    Returns (completed, touched) -- `completed` is every ticker whose sweep finished without
    hitting RateLimited (safe to mark as known so it's never re-swept); a ticker cut short by
    RateLimited stays out of it and is retried next run. `touched` is every ticker that picked up
    at least one new trade-worthy claim, for the caller to fold into Stage C's update set.
    """
    cutoff = as_of - dt.timedelta(days=BACKFILL_LOOKBACK_DAYS)
    completed: set[str] = set()
    touched: set[str] = set()
    for ticker, name in sorted(new_tickers.items()):
        needles = [s.lower() for s in {ticker, name} if s]
        candidates = [
            link for link, entry in events_cache.items()
            if entry.get("article_date")
            and dt.date.fromisoformat(entry["article_date"]) >= cutoff
            and ticker not in claims_attempted.get(link, [])
            and any(needle in (entry.get("text", "") + " " + entry.get("title", "")).lower() for needle in needles)
        ]
        if verbose:
            print(
                f"  [backfill] {ticker} ({name}) newly added to the universe -- {len(candidates)} "
                f"article(s) in the last {BACKFILL_LOOKBACK_DAYS}d mention it, re-classifying..."
            )
        try:
            for link in candidates:
                entry = events_cache[link]
                article_date = dt.date.fromisoformat(entry["article_date"])
                if verbose:
                    print(f"    [Stage A] re-classifying for {ticker}: {entry['title'][:60]}")
                event = extract_event(entry["title"], entry["text"], universe_map, article_date, debug=verbose)
                if event is None:
                    continue
                entry["event"] = event
                _save_events(events_cache)
                if ticker not in event["companies"] or not event["novel"] or event["importance"] < IMPORTANCE_MIN:
                    continue
                if _extract_and_store_claims(
                    ticker, link, entry["title"], entry["text"], event, article_date, claims_attempted, verbose
                ):
                    touched.add(ticker)
            completed.add(ticker)
        except RateLimited as exc:
            if verbose:
                reason = f" ({exc.message})" if exc.message else ""
                print(f"  [backfill] {ticker} -- rate limited{reason}, stopping backfill here (will retry next run).")
            break
    return completed, touched


def _print_rate_limit_status() -> None:
    config = llm_config()
    provider, model = config["provider"], config["models"][0] if config["models"] else None
    status = get_rate_limit_status(model=model, provider=provider)
    if status is None:
        print(f"{provider} quota: unavailable (no {provider!r} API key configured, or the network request failed).")
        return
    if status.get("_blocked"):
        # The x-ratelimit-* headers below only ever describe the per-minute buckets and can
        # look fine even while blocked -- providers only report a daily cap (if that's what this
        # is) inside the 429 body, so surface that directly rather than the misleading headers.
        reason = status.get("_message") or "no further detail given"
        print(f"{provider} quota: BLOCKED right now -- {reason}")
        return
    status_code = status.get("_status_code")
    if status_code is not None and status_code >= 400:
        # A real error other than 429 (bad model name, bad key, malformed request, ...) -- distinct
        # from "200 but this provider just doesn't send rate-limit headers" (OpenRouter, often).
        reason = status.get("_message") or "no further detail given"
        print(
            f"{provider} quota check got HTTP {status_code} probing model {model!r} -- {reason}. "
            f"(This means Loop A's real calls will likely fail too -- check the model name in "
            f"config_loop_a.json.)"
        )
        return
    req_limit = status.get("x-ratelimit-limit-requests")
    if req_limit is None:
        print(f"{provider} quota: OK (model {model!r} responded), but this provider doesn't report rate-limit headers.")
        return
    req_remaining = status.get("x-ratelimit-remaining-requests", "?")
    req_reset = status.get("x-ratelimit-reset-requests", "?")
    tok_limit = status.get("x-ratelimit-limit-tokens", "?")
    tok_remaining = status.get("x-ratelimit-remaining-tokens", "?")
    tok_reset = status.get("x-ratelimit-reset-tokens", "?")
    print(
        f"{provider} quota: {req_remaining}/{req_limit} requests remaining (resets in {req_reset}), "
        f"{tok_remaining}/{tok_limit} tokens remaining (resets in {tok_reset})."
    )


def update_research(
    as_of: dt.date | None = None, verbose: bool = True, include_sec_8k: bool = False
) -> set[str]:
    """Global, portfolio-independent research pass: Stage A -> Stage B ->
    ticker-thesis aggregation. Safe to call from every portfolio's
    `run_loop_a` -- everything here is cached/keyed globally, so a second or
    third portfolio calling this the same day does no redundant LLM work
    beyond whatever's genuinely new since the last call. Returns the set of
    tickers whose TickerThesis was actually updated this call.
    """
    as_of = as_of or dt.date.today()
    if verbose:
        _print_rate_limit_status()

    events_cache = _load_events()
    claims_attempted = _load_claims_attempted()
    universe_map = _universe_name_to_ticker()
    updated_tickers: set[str] = set()

    known_tickers = _load_known_universe_tickers()
    newly_added = {t: n for t, n in tracked_universe().items() if t not in known_tickers}
    if newly_added:
        completed, backfill_touched = _backfill_new_tickers(
            events_cache, claims_attempted, universe_map, newly_added, as_of, verbose
        )
        if completed:
            _save_known_universe_tickers(known_tickers | completed)
        updated_tickers |= backfill_touched

    articles = get_news_from_sources(active_news_sources())
    if include_sec_8k:
        tracked_tickers = sorted(set(universe_map.values()))
        articles = pd.concat([articles, get_sec_8k_news(tracked_tickers, get_cik_map())], ignore_index=True)
    # Skip anything older than 6 months -- a feed can carry a stale backlog entry (e.g. after a
    # source is newly activated), and Stage A/B have no business spending a real LLM call
    # classifying news that's no longer actionable. An article with no parseable published date
    # is kept (falls back to as_of below), not dropped, since that's a feed-parsing gap, not staleness.
    articles_cutoff = pd.Timestamp(as_of) - pd.DateOffset(months=6)
    articles = articles[articles["published"].isna() | (articles["published"] >= articles_cutoff)]
    articles = articles.sort_values("published", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Research update: checking {len(articles)} article(s) as of {as_of.isoformat()}.")

    for i, row in enumerate(articles.itertuples(index=False), 1):
        reset_usage_counter()  # measured per article -- includes Stage A (if not cached) + every Stage B call below
        title = _safe_text(row.title)
        # The article's own publish date, not today -- reasoning/signals should reflect what was
        # actually knowable *when the article came out*, not the day this run happens to process it
        # (a rate-limit-delayed backlog article processed days late would otherwise get today's
        # price action and a leakage-avoidance cutoff both wrongly shifted forward). Falls back to
        # the run's as_of if the feed genuinely didn't give a parseable date.
        article_date = row.published.date() if pd.notna(row.published) else as_of
        try:
            cached = events_cache.get(row.link)
            if cached is not None:
                title, article_text, event = cached["title"], cached["text"], cached["event"]
                if cached.get("article_date"):
                    article_date = dt.date.fromisoformat(cached["article_date"])
            else:
                title, summary, content = _safe_text(row.title), _safe_text(row.summary), _safe_text(row.content)
                article_text = _strip_html(content) if content else _strip_html(summary)
                known_ticker = row.source[len(SEC_8K_SOURCE_PREFIX):] if row.source.startswith(SEC_8K_SOURCE_PREFIX) else None
                if not article_text:
                    event = None
                elif known_ticker:
                    if verbose:
                        print(f"    [Stage A] classifying ({known_ticker}): {title[:60]}")
                    event = extract_event_known_company(title, article_text, known_ticker, article_date, debug=verbose)
                else:
                    if verbose:
                        print(f"    [Stage A] classifying: {title[:60]}")
                    event = extract_event(title, article_text, universe_map, article_date, debug=verbose)
                # Permanent global archive, independent of finance.news's rolling RSS cache
                # (which is just a rolling window of whatever the feed currently contains --
                # an old article can scroll out of it and disappear) -- keeps the full text
                # so it survives that, and doubles as an audit trail of every article looked
                # at (including ones that produced no event, and why).
                events_cache[row.link] = {
                    "seen_date": as_of.isoformat(), "article_date": article_date.isoformat(),
                    "title": title, "text": article_text, "event": event,
                }
                _save_events(events_cache)

            def _log(detail: str) -> None:
                if verbose:
                    tokens = get_usage_counter()["total_tokens"]
                    print(f"  [{i}/{len(articles)}] ({row.source}) {title[:70]} -- {detail} (tokens: {tokens})")

            if not article_text:
                _log("empty article, skipped")
                continue
            if event is None:
                _log("extraction failed")
                continue

            event_summary = f"importance={event['importance']}, companies={event['companies']}, novel={event['novel']}"
            if not event["novel"] or event["importance"] < IMPORTANCE_MIN or not event["companies"]:
                _log(f"{event_summary} -- filtered")
                continue

            already_done = set(claims_attempted.get(row.link, []))
            new_tickers = [t for t in event["companies"] if t not in already_done]
            if not new_tickers:
                _log(f"{event_summary} -- already processed")
                continue

            for ticker in new_tickers:
                if _extract_and_store_claims(
                    ticker, row.link, title, article_text, event, article_date, claims_attempted, verbose
                ):
                    updated_tickers.add(ticker)

            _log(f"{event_summary} -- {len(new_tickers)} ticker(s) processed")
        except RateLimited as exc:
            if verbose:
                reason = f" ({exc.message})" if exc.message else ""
                print(
                    f"  [{i}/{len(articles)}] ({row.source}) {title[:70]} -- rate limited{reason}, "
                    f"stopping research update here."
                )
            break

    # Self-heal: a ticker can have claims recorded but no current TickerThesis if a
    # prior aggregation call failed (a soft parse failure, not RateLimited) after
    # Stage B had already marked its claims attempted -- Stage B alone never
    # revisits an (article, ticker) pair once attempted, so without this sweep such
    # a ticker would stay orphaned forever. Just file-existence checks, no LLM cost
    # unless something is genuinely missing.
    #
    # A ticker with a genuine *new* claim this run (already in updated_tickers) always gets
    # attempted -- new evidence is always worth a fresh try. A ticker the sweep alone would add
    # (no new claims, just still-orphaned) backs off after MAX_SELF_HEAL_RETRIES consecutive
    # failures, so a ticker whose claims reliably break aggregation doesn't burn 2 real LLM calls
    # every single run forever with nothing new to justify it.
    attempts = _load_aggregation_attempts()
    for ticker in list_tickers_with_claims():
        if ticker in updated_tickers or load_ticker_thesis(ticker) is not None:
            continue
        if attempts.get(ticker, {}).get("consecutive_failures", 0) >= MAX_SELF_HEAL_RETRIES:
            continue
        updated_tickers.add(ticker)

    for ticker in sorted(updated_tickers):
        try:
            tt = update_ticker_thesis(ticker, as_of)
            if tt is not None:
                attempts.pop(ticker, None)
            else:
                record = attempts.setdefault(ticker, {"consecutive_failures": 0})
                record["consecutive_failures"] += 1
                record["last_attempt"] = as_of.isoformat()
            _save_aggregation_attempts(attempts)
            if verbose:
                if tt:
                    print(f"  [ticker-thesis] {ticker}: confidence={tt.confidence:.0%}, direction={tt.direction}")
                else:
                    failures = attempts[ticker]["consecutive_failures"]
                    print(f"  [ticker-thesis] {ticker}: aggregation produced nothing usable (consecutive failures: {failures})")
        except RateLimited as exc:
            if verbose:
                reason = f" ({exc.message})" if exc.message else ""
                print(f"  [ticker-thesis] {ticker} -- rate limited{reason}, stopping here.")
            break

    # Earnings-window fundamentals refresh: a ticker with no qualifying news around its own
    # earnings report never gets its fundamentals reconsidered otherwise (fundamental_opinion
    # is normally only a side effect of a new news-driven claim above) -- but earnings is exactly
    # when the underlying numbers most likely moved. One refresh per earnings event (tracked by
    # date, not a boolean), regardless of how many consecutive daily runs fall inside the window.
    earnings_checks = _load_earnings_fundamental_checks()
    for ticker in list_tickers_with_thesis():
        earnings_date = _earnings_trigger_date(ticker, as_of)
        if earnings_date is None:
            continue
        if earnings_checks.get(ticker) == earnings_date.isoformat():
            continue
        try:
            tt = refresh_fundamentals(ticker, as_of)
            earnings_checks[ticker] = earnings_date.isoformat()
            _save_earnings_fundamental_checks(earnings_checks)
            if verbose:
                if tt:
                    print(
                        f"  [earnings-fundamentals] {ticker}: refreshed around earnings "
                        f"{earnings_date.isoformat()} (confidence={tt.confidence:.0%})"
                    )
                else:
                    print(f"  [earnings-fundamentals] {ticker}: refresh produced nothing usable")
        except RateLimited as exc:
            if verbose:
                reason = f" ({exc.message})" if exc.message else ""
                print(f"  [earnings-fundamentals] {ticker} -- rate limited{reason}, stopping here.")
            break

    return updated_tickers


def run_loop_a(
    portfolio_name: str,
    as_of: dt.date | None = None,
    verbose: bool = True,
    include_sec_8k: bool = False,
) -> list[dict]:
    """Two phases: first `update_research` -- global, shared, cached; then a
    thin, portfolio-specific, LLM-free trade-decision pass reacting to
    whichever TickerThesis snapshots exist: opens a position for a ticker not
    already held once its ticker-thesis confidence clears the bar, closes
    one already held if its ticker-thesis no longer supports it. No position-
    size changes on reinforcement -- confidence moving up while already held
    doesn't add to the position, only a drop below the bar (or a direction
    flip) closes it.
    """
    as_of = as_of or dt.date.today()
    update_research(as_of=as_of, verbose=verbose, include_sec_8k=include_sec_8k)

    strategy = _portfolio_strategy(portfolio_name)
    log: list[dict] = []
    open_by_ticker = {p.ticker: p for p in open_positions(portfolio_name)}

    for ticker in list_tickers_with_thesis():
        tt = load_ticker_thesis(ticker)
        if tt is None:
            continue
        held = open_by_ticker.get(ticker)
        if held is not None:
            if tt.confidence < strategy["confidence_min"] or tt.direction != "long":
                close_trade = _close_position_now(portfolio_name, held, as_of, reason="invalidated")
                if close_trade:
                    log.append(close_trade)
                    if verbose:
                        print(
                            f"  [close] {ticker}: ticker-thesis no longer supports holding "
                            f"(confidence={tt.confidence:.0%}, direction={tt.direction})"
                        )
        else:
            trade = _try_open(portfolio_name, tt, as_of, strategy)
            if trade:
                log.append({"action": "position_opened", "ticker": ticker, "confidence": tt.confidence})
                log.append(trade)
                if verbose:
                    print(f"  [open] {ticker}: confidence={tt.confidence:.0%} -- {tt.thesis[:80]}")

    return log


def review_loop_a(portfolio_name: str, as_of: dt.date | None = None) -> list[dict]:
    """Closes every open loop_a position whose expected horizon has elapsed:
    sells and records the realized return against entry_confidence, for
    later calibration. Run this before `run_loop_a` each day, same
    exits-before-entries ordering as finance.rules.run_rules.
    """
    as_of = as_of or dt.date.today()
    log: list[dict] = []

    for position in open_positions(portfolio_name):
        if position.rule != RULE_NAME:
            continue
        held_days = (as_of - position.created).days
        if held_days < position.expected_horizon_days:
            continue

        trade = _close_position_now(portfolio_name, position, as_of, reason="horizon")
        if trade:
            log.append(trade)

    return log
