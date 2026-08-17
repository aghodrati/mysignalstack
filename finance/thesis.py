"""Per-portfolio positions Loop A opened, each referencing the global,
shared TickerThesis (finance.tickerthesis) it was opened against -- an
append-only event log (same replay-not-store convention as
finance.portfolio's own trade log), so current position state is always
derived from the log rather than mutated in place.

Unlike the old per-portfolio design, a position here carries almost no
content of its own -- no claim/catalysts/invalidation, just the trading-
relevant facts (which ticker, when opened, at what confidence, when/why
closed). The actual investment thesis -- claim, direction, confidence, full
evidence trail -- lives globally in finance.tickerthesis and finance.claims,
shared by every portfolio watching that ticker; this module only tracks
whether *this* portfolio acted on it.

Two event kinds: "created" (a new position opened, status "open") and
"closed" (fills in closed_date/actual_return_pct/close_reason and flips
status to "closed", either because the horizon elapsed or because the
TickerThesis it was opened against was later invalidated -- see
finance.newsloop.run_loop_a).

Stored as a pretty-printed JSON array (not line-delimited JSONL) so the file
is directly readable when opened by hand -- appending means read-modify-write
the whole array, which is fine at this scale (a handful of positions a day,
not a high-frequency log). Lives inside finance.portfolio's own per-portfolio
directory (output/portfolios/{name}/positions.json), alongside meta.json and
trades.csv, rather than a separate top-level tree -- it's the same
portfolio's state, just Loop A's own slice of it. That also means deleting a
portfolio (finance.portfolio.delete_portfolio) removes its position log for
free, no separate cleanup call needed.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from finance.portfolio import PORTFOLIOS_DIR

Status = Literal["open", "closed"]
CloseReason = Literal["horizon", "invalidated"]


@dataclass
class Position:
    id: str  # stable hash of (portfolio, ticker, opened date) -- see newsloop.position_id
    ticker: str
    created: dt.date
    entry_confidence: float  # the TickerThesis's confidence at the moment this position opened
    expected_horizon_days: int
    rule: str = "loop_a"
    status: Status = "open"
    closed_date: dt.date | None = None
    actual_return_pct: float | None = None
    close_reason: CloseReason | None = None


def _path(portfolio_name: str) -> Path:
    return PORTFOLIOS_DIR / portfolio_name / "positions.json"


def delete_positions(portfolio_name: str) -> None:
    """Removes the portfolio's position log, if it has one -- a no-op
    otherwise (a portfolio Loop A was never run against never has one).
    Redundant with finance.portfolio.delete_portfolio (which removes the
    whole output/portfolios/{name}/ directory, positions.json included) --
    kept as its own function for callers that want to clear just the
    position log without deleting the portfolio itself.
    """
    _path(portfolio_name).unlink(missing_ok=True)


def _load_events(portfolio_name: str) -> list[dict]:
    path = _path(portfolio_name)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _append(portfolio_name: str, event: dict) -> None:
    path = _path(portfolio_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    events = _load_events(portfolio_name)
    events.append(json.loads(json.dumps(event, default=str)))  # normalize dates etc. to plain JSON types
    path.write_text(json.dumps(events, indent=2))


def append_created(portfolio_name: str, position: Position) -> None:
    _append(portfolio_name, {"event": "created", **asdict(position)})


def append_closed(
    portfolio_name: str, position_id: str, closed_date: dt.date, actual_return_pct: float, reason: CloseReason
) -> None:
    _append(
        portfolio_name,
        {
            "event": "closed", "id": position_id, "closed_date": closed_date,
            "actual_return_pct": actual_return_pct, "close_reason": reason,
        },
    )


def load_positions(portfolio_name: str) -> list[Position]:
    by_id: dict[str, Position] = {}
    for event in _load_events(portfolio_name):
        if event["event"] == "created":
            data = {k: v for k, v in event.items() if k != "event"}
            data["created"] = dt.date.fromisoformat(data["created"])
            by_id[data["id"]] = Position(**data)
        else:  # closed
            position = by_id.get(event["id"])
            if position is None:
                continue  # defensive -- shouldn't happen given append_closed always follows a create
            position.status = "closed"
            position.closed_date = dt.date.fromisoformat(event["closed_date"])
            position.actual_return_pct = float(event["actual_return_pct"])
            position.close_reason = event.get("close_reason", "horizon")

    return list(by_id.values())


def open_positions(portfolio_name: str, ticker: str | None = None) -> list[Position]:
    positions = [p for p in load_positions(portfolio_name) if p.status == "open"]
    if ticker is not None:
        positions = [p for p in positions if p.ticker == ticker]
    return positions
