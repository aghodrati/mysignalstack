"""Runs Loop A (News -> LLM -> trade) once against an existing paper-trading
portfolio: first, the global research update (Stage A -> Stage B ->
ticker-thesis aggregation -- shared by every portfolio, cached, so a second
portfolio run the same day costs little extra), then closes any loop_a
positions whose horizon has elapsed, then this portfolio's own thin,
deterministic trade-decision pass (open/close positions reacting to current
TickerThesis snapshots). Meant to be run once a day (cron or manual) --
create the portfolio first via the Streamlit app (app.py).

Usage:
    uv run run_loop_a.py my_portfolio
    uv run run_loop_a.py my_portfolio --include-sec-8k
"""

import argparse

from finance.newsloop import review_loop_a, run_loop_a
from finance.portfolio import list_portfolios


def main():
    parser = argparse.ArgumentParser(description="Run Loop A against a paper-trading portfolio.")
    parser.add_argument("portfolio", help="Name of an existing portfolio (create via the Streamlit app first)")
    parser.add_argument(
        "--include-sec-8k", action="store_true",
        help="Also check SEC 8-K filings for every tracked ticker (off by default -- higher token cost)",
    )
    args = parser.parse_args()

    if args.portfolio not in list_portfolios():
        existing = ", ".join(list_portfolios()) or "(none yet)"
        raise SystemExit(f"No portfolio named '{args.portfolio}'. Existing portfolios: {existing}")

    print("Reviewing open positions...")
    reviewed = review_loop_a(args.portfolio)
    for entry in reviewed:
        print(f"[review] {entry}")

    result = run_loop_a(args.portfolio, include_sec_8k=args.include_sec_8k)
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


if __name__ == "__main__":
    main()
