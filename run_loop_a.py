"""Runs Loop A (News -> LLM -> trade) once against an existing paper-trading portfolio, made up of
four independently-scopable research stages -- claims (Stage A/B/C), fundamentals, earnings calls,
macro narratives -- plus this portfolio's own thin, deterministic trade-decision pass (open/close
positions reacting to current TickerThesis snapshots), which always runs regardless of which
stages below are selected since it's LLM-free. Meant to be run once a day (cron or manual) --
create the portfolio first via the Streamlit app (app.py).

With no --refresh-* flag, every stage runs at its default scope: claims for every active news
source and earnings calls for the whole tracked universe (both cheap to check -- an
article/transcript already on record costs no LLM call). Macro narratives are different: they
always cost a real LLM call (see finance.macro.refresh_macro_narrative -- no skip-based dedup), so
on a default (no-flag) run they're gated by today's date instead, to avoid spending on every single
run of what's typically a daily cron -- weekly narratives only refresh on Saturdays, monthly
narratives only on the last calendar day of the month. The automatic earnings-window fundamentals
check (a ticker within a day of its own earnings date) always runs too, regardless of flags -- it's
an instant no-op for every ticker not currently in its own window. A *forced* fundamentals refresh
(bypassing that window) only ever happens via an explicit --refresh-fundamental.

Passing any of --refresh-claims / --refresh-fundamental / --refresh-earning-call / --refresh-macro /
--weekly / --monthly narrows the run to just the stage(s) named (earnings-window-fundamentals/
trade-decisions still apply as above unless explicitly excluded by naming other stages instead).
An explicit --refresh-macro (or --weekly/--monthly, scoped to just that one period) also bypasses
the Saturday/month-end date gating above -- it always refreshes, regardless of today's date, since
asking for it by name is itself the signal that a refresh is wanted right now. --weekly/--monthly
are meant for a scheduler (e.g. two separate GitHub Actions cron entries) that already decides
*when* to run each one, rather than relying on this script's own day-of-week/month-end check.
--source and --ticker further scope whichever stage they're relevant to -- for --refresh-macro,
--ticker names a finance.macro.NARRATIVE_SERIES key (e.g. "10y_yield"), not a stock ticker.

Usage:
    uv run run_loop_a.py my_portfolio
    uv run run_loop_a.py my_portfolio --include-sec-8k
    uv run run_loop_a.py my_portfolio --refresh-claims
    uv run run_loop_a.py my_portfolio --refresh-claims --source SemiAnalysis
    uv run run_loop_a.py my_portfolio --refresh-fundamental --ticker AAPL
    uv run run_loop_a.py my_portfolio --refresh-earning-call --ticker NBIS --transcript-url https://www.fool.com/earnings/call-transcripts/2026/.../nbis-...-earnings-call-transcript/
    uv run run_loop_a.py my_portfolio --refresh-macro --ticker 10y_yield
    uv run run_loop_a.py my_portfolio --weekly   # e.g. a Saturday-only cron entry
    uv run run_loop_a.py my_portfolio --monthly  # e.g. a month-end-only cron entry
"""

import argparse
import datetime as dt

from dotenv import load_dotenv

# Loads .env into os.environ explicitly -- `uv run` does NOT do this automatically (same gotcha
# app.py already works around). Every finance.* module reads its API keys via plain os.environ.get
# (finance.llm._load_env_var, finance.news_sources' FINNHUB_API_KEY/GNEWS_API_KEY lookups, etc.),
# so without this, a key that only exists in .env -- not already exported in the calling shell --
# silently resolves to None here, one stage at a time, with no error: OPENROUTER_API_KEY still
# working while a newly-added key like FINNHUB_API_KEY quietly doesn't is the exact symptom this
# fixes. load_dotenv() never overrides a var the shell already has set for real.
load_dotenv()

from finance.earnings_calls import discover_recent_transcripts, needs_transcript_check, refresh_earnings_call
from finance.llm import RateLimited
from finance.loop_a_config import active_news_sources, tracked_universe, youtube_sources
from finance.macro import MONTHLY_NARRATIVE_SERIES, NARRATIVE_SERIES, refresh_macro_narrative
from finance.newsloop import review_loop_a, run_loop_a
from finance.portfolio import list_portfolios
from finance.youtube import refresh_youtube_sources


def main():
    parser = argparse.ArgumentParser(description="Run Loop A against a paper-trading portfolio.")
    parser.add_argument("portfolio", help="Name of an existing portfolio (create via the Streamlit app first)")
    parser.add_argument(
        "--include-sec-8k", action="store_true",
        help="Also check SEC 8-K filings for every tracked ticker (off by default -- higher token cost)",
    )
    parser.add_argument(
        "--refresh-claims", action="store_true",
        help=(
            "Run the claims stage (Stage A/B/C). With no other --refresh-* flag given, this is "
            "already the default -- pass it explicitly to run ONLY claims this invocation "
            "(skips fundamentals/earnings-calls entirely, aside from the always-on earnings-window "
            "fundamentals check)."
        ),
    )
    parser.add_argument(
        "--source", default=None,
        help=(
            "Restrict --refresh-claims to just this news source (by name, as configured in "
            "config_loop_a.json's news_sources), e.g. --source SemiAnalysis. Ignored otherwise."
        ),
    )
    parser.add_argument(
        "--refresh-fundamental", action="store_true",
        help=(
            "Force-refresh fundamentals right now (every tracked ticker, or just --ticker if given), "
            "on top of whatever the always-on earnings-window trigger already covers. Passing this "
            "alone (no --refresh-claims/--refresh-earning-call) runs ONLY this stage."
        ),
    )
    parser.add_argument(
        "--refresh-earning-call", action="store_true",
        help=(
            "Check for new earnings call transcripts (every tracked ticker, or just --ticker if "
            "given) -- skipped entirely, no LLM cost, for a ticker whose latest transcript is "
            "already on record. With no other --refresh-* flag given, this is already the default; "
            "passing it alone (no --refresh-claims/--refresh-fundamental) runs ONLY this stage."
        ),
    )
    parser.add_argument(
        "--refresh-macro", action="store_true",
        help=(
            "Refresh weekly macro narratives (every configured finance.macro.NARRATIVE_SERIES "
            "series, or just --ticker if given) -- skipped entirely, no LLM/news-source cost, for a "
            "series whose narrative for this ISO week is already on record. With no other "
            "--refresh-* flag given, this is already the default; passing it alone runs ONLY this "
            "stage."
        ),
    )
    parser.add_argument(
        "--refresh-youtube", action="store_true",
        help=(
            "Check every active finance.loop_a_config.youtube_sources() channel for new videos, "
            "transcribe + summarize each one (finance.youtube.refresh_youtube_sources), and persist "
            "the results to output/youtube_summaries.json for the Streamlit app's Interviews page. "
            "With no other --refresh-* flag given, this is already the default (same as claims/"
            "earnings-calls/macro); pass it explicitly to run ONLY this stage. Unrelated to the "
            "portfolio/claims/fundamentals/earnings-call/macro stages above -- there's no ticker or "
            "trade-decision link, it just needs *a* portfolio argument to invoke this script at all."
        ),
    )
    parser.add_argument(
        "--weekly", action="store_true",
        help=(
            "Scope --refresh-macro (or a default no-flag run) to just the weekly macro narrative -- "
            "also implies --refresh-macro's own \"run macro only\" scoping and its date-gating "
            "bypass (see --refresh-macro's help), just narrowed to this one period. Meant for a "
            "scheduler (e.g. a GitHub Actions cron firing only on Saturdays) that already decides "
            "*when* to run this -- combine with --monthly to force both explicitly."
        ),
    )
    parser.add_argument(
        "--monthly", action="store_true",
        help=(
            "Same as --weekly, scoped to the monthly macro narrative instead (only "
            "finance.macro.MONTHLY_NARRATIVE_SERIES) -- meant for a scheduler that only fires this "
            "near month-end, rather than relying on this script's own month-end date check."
        ),
    )
    parser.add_argument(
        "--ticker", default=None,
        help=(
            "Restrict --refresh-fundamental/--refresh-earning-call to just this stock ticker (e.g. "
            "--ticker AAPL), or --refresh-macro to just this series key (e.g. --ticker 10y_yield)."
        ),
    )
    parser.add_argument(
        "--transcript-url", default=None,
        help=(
            "Manual override for --refresh-earning-call: skip discovery (Fool's recent listing + "
            "DuckDuckGo fallback) entirely and use this fool.com transcript URL directly. Useful "
            "when a ticker isn't in Fool's ~20 sitewide-newest and DuckDuckGo's anti-bot check is "
            "blocking the fallback search -- find the URL yourself in a browser. Requires --ticker."
        ),
    )
    args = parser.parse_args()

    if args.transcript_url and not args.ticker:
        raise SystemExit("--transcript-url requires --ticker.")
    if args.transcript_url and not args.refresh_earning_call:
        raise SystemExit("--transcript-url requires --refresh-earning-call.")

    # Presence of any --refresh-* flag narrows this run to just the stage(s) named; with none
    # given, claims/earnings-calls/macro-narratives/youtube all run at their default (whole-
    # universe/every-series/every-channel) scope, same as before this flag ever existed. A forced
    # fundamentals refresh is always purely opt-in -- --refresh-fundamental is never implied by an
    # all-stages default run. --weekly/--monthly count as a macro stage flag too (same "narrows to
    # just this stage" effect as --refresh-macro) -- a scheduler invoking just `--weekly` shouldn't
    # accidentally also kick off a full claims/earnings-call pass.
    any_stage_flag = (
        args.refresh_claims or args.refresh_fundamental or args.refresh_earning_call
        or args.refresh_macro or args.weekly or args.monthly or args.refresh_youtube
    )
    do_claims = args.refresh_claims or not any_stage_flag
    do_fundamentals = args.refresh_fundamental
    do_earning_call = args.refresh_earning_call or not any_stage_flag
    do_macro = args.refresh_macro or args.weekly or args.monthly or not any_stage_flag
    do_youtube = args.refresh_youtube or not any_stage_flag

    if args.source is not None:
        if not do_claims:
            raise SystemExit("--source only applies to the claims stage -- pass --refresh-claims (or no --refresh-* flag) to use it.")
        valid_sources = {name for name, _url in active_news_sources()}
        if args.source not in valid_sources:
            raise SystemExit(f"No active news source named {args.source!r}. Configured sources: {', '.join(sorted(valid_sources))}")
    if do_macro and args.ticker is not None and args.ticker not in NARRATIVE_SERIES:
        raise SystemExit(
            f"No macro series named {args.ticker!r}. Configured series: {', '.join(sorted(NARRATIVE_SERIES))}"
        )
    if args.ticker is not None and not (do_fundamentals or do_earning_call or do_macro):
        print("Note: --ticker only affects --refresh-fundamental/--refresh-earning-call/--refresh-macro -- ignored for this run.")

    if args.portfolio not in list_portfolios():
        existing = ", ".join(list_portfolios()) or "(none yet)"
        raise SystemExit(f"No portfolio named '{args.portfolio}'. Existing portfolios: {existing}")

    print("Reviewing open positions...")
    reviewed = review_loop_a(args.portfolio)
    for entry in reviewed:
        print(f"[review] {entry}")

    result = run_loop_a(
        args.portfolio, include_sec_8k=args.include_sec_8k,
        include_claims=do_claims, news_source=args.source,
        refresh_fundamentals=do_fundamentals, fundamentals_ticker=args.ticker,
    )
    acted = result["trades"]
    for entry in acted:
        print(f"[loop_a] {entry}")

    n_closed = sum(1 for e in reviewed + acted if e["action"] == "SELL")
    n_opened = sum(1 for e in acted if e["action"] == "BUY")
    print(f"Done for today: {n_closed} position(s) closed, {n_opened} opened.")

    research = result["research"]
    claims_by_source_ticker = research["claims_by_source_ticker"]
    thesis_changes = research["thesis_changes"]
    print("\n=== Research summary ===")
    if not claims_by_source_ticker:
        print("No new claims this run.")
    else:
        total_claims = sum(claims_by_source_ticker.values())
        print(f"Claims added ({total_claims} total, {len(claims_by_source_ticker)} source/ticker pair(s)):")
        for (source, ticker), count in sorted(claims_by_source_ticker.items()):
            print(f"  {source} -> {ticker}: {count}")

    if not thesis_changes:
        print("No thesis updates this run.")
    else:
        print(f"Thesis confidence changes ({len(thesis_changes)} ticker(s)), sorted by updated confidence:")
        ranked = sorted(thesis_changes.items(), key=lambda item: item[1]["after_confidence"], reverse=True)
        for ticker, change in ranked:
            before = change["before_confidence"]
            before_str = f"{before:.0%}" if before is not None else "new"
            print(f"  {ticker}: {before_str} -> {change['after_confidence']:.0%} ({change['direction']})")

    if do_earning_call:
        as_of = dt.date.today()
        if args.ticker:
            # Explicit --ticker is a deliberate manual check (also required for --transcript-url)
            # -- always runs regardless of earnings window, same as --refresh-fundamental --ticker
            # bypassing its own window trigger.
            tickers = [args.ticker.upper()]
            skipped = 0
        else:
            all_tickers = sorted(tracked_universe())
            tickers = [t for t in all_tickers if needs_transcript_check(t, as_of)]
            skipped = len(all_tickers) - len(tickers)
        print(f"\n=== Earnings call transcript refresh ({len(tickers)} ticker(s)) ===")
        if skipped:
            print(
                f"  ({skipped} ticker(s) outside their earnings window, skipped -- not near a "
                f"reported/upcoming earnings date, no point spending a discovery search on them)"
            )
        if args.transcript_url:
            recent_index = None
            print(f"Using manual transcript URL for {tickers[0]}, skipping discovery.")
        elif tickers:
            print("Scanning fool.com's recent transcript listing...")
            recent_index = discover_recent_transcripts()
        else:
            recent_index = None
        for ticker in tickers:
            try:
                snapshot = refresh_earnings_call(
                    ticker, dt.date.today(), recent_index=recent_index,
                    transcript_url=args.transcript_url,
                )
            except RateLimited as exc:
                reason = f" ({exc.message})" if exc.message else ""
                print(f"  {ticker} -- rate limited{reason}, stopping here.")
                break
            if snapshot:
                print(
                    f"  {ticker}: new transcript processed (direction={snapshot['earnings_direction']}, "
                    f"confidence={snapshot['earnings_confidence']:.0%}, tone={snapshot['management_tone']})"
                )
            else:
                print(f"  {ticker}: no new transcript found (or nothing usable)")

    if do_macro:
        # refresh_macro_narrative always regenerates a fresh LLM+news read on every call for both
        # periods -- no skip-based dedup (see its own docstring) -- so calling it costs real LLM
        # calls every time, by design (both narratives are rolling "as of now" reads). On an
        # explicit --refresh-macro that's exactly what's wanted (asking for it by name means "do it
        # now"), but on a default no-flag run (typically a daily cron alongside claims/earnings
        # calls, which stay cheap) that would mean spending on macro every single day for no
        # benefit -- weekly/monthly narratives are only ever displayed once a week/month apart
        # anyway. So a default run instead only refreshes weekly on Saturdays and monthly on the
        # last calendar day of the month; --refresh-macro (or --weekly/--monthly, scoped to just
        # that one period) bypasses this gating entirely -- naming a period explicitly is itself
        # the "do it now" signal, same reasoning as --refresh-macro alone always had.
        macro_explicit = args.refresh_macro or args.weekly or args.monthly
        # With no --weekly/--monthly given, both periods are still eligible (subject to the
        # gating below, or forced by a bare --refresh-macro); naming one or both explicitly
        # narrows to exactly that set, forced regardless of gating.
        periods_wanted = (
            {p for p, flag in (("week", args.weekly), ("month", args.monthly)) if flag}
            if (args.weekly or args.monthly) else {"week", "month"}
        )
        today = dt.date.today()
        is_saturday = today.weekday() == 5
        is_month_end = (today + dt.timedelta(days=1)).day == 1
        series_keys = [args.ticker] if args.ticker else list(NARRATIVE_SERIES)
        print(f"\n=== Macro narratives ({len(series_keys)} series) ===")
        if not macro_explicit:
            print(
                f"  (default run -- weekly only refreshes on Saturdays{' (today)' if is_saturday else ''}, "
                f"monthly only on the last day of the month{' (today)' if is_month_end else ''}; "
                f"pass --refresh-macro/--weekly/--monthly to force a refresh regardless of today's date)"
            )
        for series_key in series_keys:
            eligible_periods = [
                p for p in (["week", "month"] if series_key in MONTHLY_NARRATIVE_SERIES else ["week"])
                if p in periods_wanted
            ]
            periods = eligible_periods if macro_explicit else [
                p for p in eligible_periods
                if (p == "week" and is_saturday) or (p == "month" and is_month_end)
            ]
            for period in periods:
                try:
                    result = refresh_macro_narrative(series_key, dt.date.today(), period=period)
                except RateLimited as exc:
                    reason = f" ({exc.message})" if exc.message else ""
                    print(f"  {series_key} ({period}) -- rate limited{reason}, stopping here.")
                    return
                status = result["status"]
                if status == "generated":
                    print(f"  {series_key} ({period}): refreshed as of {result['date']} -- {result['narrative'][:100]}")
                elif status == "news_unavailable":
                    print(f"  {series_key} ({period}): news source is rate-limited or unreachable right now -- skipped, will retry next run.")
                elif status == "no_tile":
                    print(f"  {series_key} ({period}): couldn't fetch the underlying macro series -- skipped.")
                elif status == "llm_failed":
                    print(f"  {series_key} ({period}): LLM call produced nothing usable -- skipped.")
                else:
                    print(f"  {series_key} ({period}): unknown series key.")

    if do_youtube:
        channels = youtube_sources()
        print(f"\n=== YouTube interview summaries ({len(channels)} channel(s)) ===")
        try:
            added = refresh_youtube_sources(dt.date.today())
        except RateLimited as exc:
            reason = f" ({exc.message})" if exc.message else ""
            print(f"  rate limited{reason}, stopping here.")
            return
        # No per-video loop here -- refresh_youtube_sources already printed one line per video as
        # it processed it (verbose=True, the default), so reprinting the same `added` list here
        # would just be a second copy of the exact same lines.
        if not added:
            print("  No new videos summarized this run.")
        else:
            print(f"  {len(added)} video(s) summarized this run.")


if __name__ == "__main__":
    main()
