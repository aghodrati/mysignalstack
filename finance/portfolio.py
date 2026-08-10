"""Paper-trading portfolio simulations: a named simulation is an append-only
trade log on disk (survives app restarts, since it's just a file), plus a
small amount of metadata (starting cash, creation date). Everything else --
current positions, cash balance, day-by-day mark-to-market value -- is
derived by replaying that log against price history, never stored directly.

Each simulation is independent, so you can run several at once with
different rules (manual trades, a rule function, or a mix) and compare them
side by side.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from finance.data import get_live_price, get_prices

PORTFOLIOS_DIR = Path("output/portfolios")

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _dir(name: str) -> Path:
    return PORTFOLIOS_DIR / name


def _meta_path(name: str) -> Path:
    return _dir(name) / "meta.json"


def _trades_path(name: str) -> Path:
    return _dir(name) / "trades.csv"


def list_portfolios() -> list[str]:
    if not PORTFOLIOS_DIR.exists():
        return []
    return sorted(p.name for p in PORTFOLIOS_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists())


def create_portfolio(name: str, initial_cash: float, created: dt.date | None = None) -> None:
    name = name.strip()
    if not name or not _NAME_RE.match(name):
        raise ValueError("Name must be non-empty and contain only letters, digits, '_' or '-'.")
    if _meta_path(name).exists():
        raise ValueError(f"A simulation named '{name}' already exists.")
    _dir(name).mkdir(parents=True, exist_ok=True)
    meta = {
        "name": name,
        "initial_cash": float(initial_cash),
        "created": (created or dt.date.today()).isoformat(),
    }
    _meta_path(name).write_text(json.dumps(meta, indent=2))
    _trades_path(name).write_text("date,ticker,action,shares,price,note,rule\n")


def delete_portfolio(name: str) -> None:
    path = _dir(name)
    if not path.exists():
        raise ValueError(f"No simulation named '{name}'.")
    shutil.rmtree(path)


def load_meta(name: str) -> dict:
    return json.loads(_meta_path(name).read_text())


def save_rule_settings(name: str, rule_name: str, settings: dict) -> None:
    """Persists a rule's configured parameters (top_n, hold_days, unit_size, ...)
    into meta.json, so they survive an app restart, same as everything else here.
    """
    meta = load_meta(name)
    meta.setdefault("rules", {})[rule_name] = settings
    _meta_path(name).write_text(json.dumps(meta, indent=2))


def _migrate_trades_file(name: str) -> None:
    """Trade logs created before the `rule` column existed have a 6-field
    header; if any row was since appended with 7 fields (e.g. a rule ran
    against an old portfolio before it was ever reloaded), the file becomes
    ragged and unparseable. Rewrite the header to include `rule` and pad any
    short old rows with an empty value -- a no-op once the file is current.
    """
    path = _trades_path(name)
    with path.open(newline="") as f:
        rows = list(csv.reader(f))
    if not rows or "rule" in rows[0]:
        return
    new_header = rows[0] + ["rule"]
    fixed_rows = [new_header]
    for row in rows[1:]:
        if len(row) < len(new_header):
            row = row + [""] * (len(new_header) - len(row))
        fixed_rows.append(row)
    with path.open("w", newline="") as f:
        csv.writer(f).writerows(fixed_rows)


def load_trades(name: str) -> pd.DataFrame:
    _migrate_trades_file(name)
    df = pd.read_csv(_trades_path(name))
    if "rule" not in df.columns:  # older trade logs predate the rule column
        df["rule"] = ""
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "action", "shares", "price", "note", "rule"])
    df["date"] = pd.to_datetime(df["date"])
    df["note"] = df["note"].fillna("")
    df["rule"] = df["rule"].fillna("")
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def current_state(name: str) -> tuple[float, pd.DataFrame]:
    """Cash balance and per-ticker positions (shares, avg_cost) as of now,
    replaying every trade in order. Average-cost basis: a SELL reduces cost
    basis proportionally to the shares sold, at the average cost per share
    held just before that sale.
    """
    meta = load_meta(name)
    trades = load_trades(name)
    cash = float(meta["initial_cash"])
    shares: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    for row in trades.itertuples():
        t, qty, price = row.ticker, float(row.shares), float(row.price)
        if row.action == "BUY":
            cash -= qty * price
            shares[t] = shares.get(t, 0.0) + qty
            cost_basis[t] = cost_basis.get(t, 0.0) + qty * price
        else:  # SELL
            held = shares.get(t, 0.0)
            avg_cost = (cost_basis.get(t, 0.0) / held) if held else 0.0
            cash += qty * price
            shares[t] = held - qty
            cost_basis[t] = cost_basis.get(t, 0.0) - avg_cost * qty

    rows = []
    for t, qty in shares.items():
        if qty > 1e-9:
            avg_cost = cost_basis[t] / qty
            rows.append({"ticker": t, "shares": qty, "avg_cost": avg_cost})
    positions = pd.DataFrame(rows, columns=["ticker", "shares", "avg_cost"])
    if not positions.empty:
        positions = positions.set_index("ticker").sort_index()
    return cash, positions


def rule_positions(name: str, rule_name: str) -> pd.DataFrame:
    """Positions currently open under a specific rule (a subset of the whole
    portfolio's holdings, tracked separately so a rule can tell what it
    itself is holding vs. manual/other-rule positions in the same ticker).
    `entry_date` is when the currently-open lot started -- it resets each
    time this rule's position in a ticker returns to zero and reopens.
    """
    trades = load_trades(name)
    trades = trades[trades["rule"] == rule_name]
    if trades.empty:
        return pd.DataFrame(columns=["shares", "avg_cost", "entry_date"]).rename_axis("ticker")

    shares: dict[str, float] = {}
    cost_basis: dict[str, float] = {}
    entry_date: dict[str, pd.Timestamp] = {}
    for row in trades.itertuples():
        t, qty, price, d = row.ticker, float(row.shares), float(row.price), row.date
        if row.action == "BUY":
            if shares.get(t, 0.0) <= 1e-9:
                entry_date[t] = d
            shares[t] = shares.get(t, 0.0) + qty
            cost_basis[t] = cost_basis.get(t, 0.0) + qty * price
        else:  # SELL
            held = shares.get(t, 0.0)
            avg_cost = (cost_basis.get(t, 0.0) / held) if held else 0.0
            shares[t] = held - qty
            cost_basis[t] = cost_basis.get(t, 0.0) - avg_cost * qty

    rows = []
    for t, qty in shares.items():
        if qty > 1e-9:
            rows.append({"ticker": t, "shares": qty, "avg_cost": cost_basis[t] / qty, "entry_date": entry_date[t]})
    df = pd.DataFrame(rows, columns=["ticker", "shares", "avg_cost", "entry_date"])
    return df.set_index("ticker").sort_index() if not df.empty else df.set_index("ticker")


def latest_price(ticker: str, as_of: dt.date) -> float:
    """Last available close on or before `as_of` -- used to fill both manual
    'use market price' trades and rule-driven trades at a consistent price.
    """
    start = (as_of - dt.timedelta(days=7)).isoformat()
    end = (as_of + dt.timedelta(days=1)).isoformat()  # yfinance end is exclusive
    prices = get_prices([ticker], start=start, end=end)
    prices = prices.loc[prices.index <= pd.Timestamp(as_of)]
    if ticker not in prices.columns or prices[ticker].dropna().empty:
        raise ValueError(f"No price data for {ticker} on/before {as_of}.")
    return float(prices[ticker].dropna().iloc[-1])


def closed_trades(name: str) -> pd.DataFrame:
    """Pairs every BUY with the SELL that closed it, FIFO per (rule, ticker) --
    valid because this engine only ever holds one open lot per (rule, ticker)
    at a time (a rule's own positions block re-entry while already held), so
    every SELL closes exactly the preceding BUY in that same (rule, ticker).
    A BUY with no matching SELL yet is included with status="open" and NaN
    sell fields -- the caller marks those to market as of whatever date it cares about.

    Columns: rule, ticker, buy_date, buy_price, shares, sell_date, sell_price,
    return_pct, pnl, hold_days, status ("closed"/"open").
    """
    trades = load_trades(name)
    open_lots: dict[tuple[str, str], list[dict]] = {}
    rows = []
    for row in trades.itertuples():
        key = (row.rule, row.ticker)
        if row.action == "BUY":
            open_lots.setdefault(key, []).append(
                {"buy_date": row.date, "buy_price": float(row.price), "shares": float(row.shares)}
            )
        else:  # SELL
            lots = open_lots.get(key, [])
            if not lots:
                continue  # defensive -- shouldn't happen given the engine's own invariants
            lot = lots.pop(0)
            rows.append(
                {
                    "rule": row.rule, "ticker": row.ticker,
                    "buy_date": lot["buy_date"], "buy_price": lot["buy_price"], "shares": lot["shares"],
                    "sell_date": row.date, "sell_price": float(row.price),
                    "return_pct": row.price / lot["buy_price"] - 1,
                    "pnl": (row.price - lot["buy_price"]) * lot["shares"],
                    "hold_days": (row.date - lot["buy_date"]).days,
                    "status": "closed",
                }
            )
    for (rule, ticker), lots in open_lots.items():
        for lot in lots:
            rows.append(
                {
                    "rule": rule, "ticker": ticker,
                    "buy_date": lot["buy_date"], "buy_price": lot["buy_price"], "shares": lot["shares"],
                    "sell_date": pd.NaT, "sell_price": float("nan"),
                    "return_pct": float("nan"), "pnl": float("nan"), "hold_days": float("nan"),
                    "status": "open",
                }
            )
    columns = [
        "rule", "ticker", "buy_date", "buy_price", "shares", "sell_date", "sell_price",
        "return_pct", "pnl", "hold_days", "status",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns].sort_values("buy_date").reset_index(drop=True)


def execution_price(ticker: str, as_of: dt.date) -> float:
    """The price a trade *dated* as_of should fill at: a live quote if as_of
    is today (so a same-day run fills at what the stock is actually trading
    at right now, not stale from before the market opened), otherwise the
    historical close on/before as_of (backdated or after-hours trades).
    """
    if as_of == dt.date.today():
        live = get_live_price([ticker])
        if ticker in live:
            return live[ticker]
    return latest_price(ticker, as_of)


def append_trade(
    name: str,
    date: dt.date,
    ticker: str,
    action: str,
    shares: float,
    price: float,
    note: str = "",
    rule: str = "",
) -> None:
    action = action.upper()
    if action not in ("BUY", "SELL"):
        raise ValueError("action must be 'BUY' or 'SELL'")
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be positive")

    _migrate_trades_file(name)
    cash, positions = current_state(name)
    if action == "BUY":
        cost = shares * price
        if cost > cash + 1e-6:
            raise ValueError(f"Not enough cash: need ${cost:,.2f}, have ${cash:,.2f}.")
    else:
        held = positions.loc[ticker, "shares"] if ticker in positions.index else 0.0
        if shares > held + 1e-6:
            raise ValueError(f"Not enough shares: trying to sell {shares:g}, holding {held:g}.")

    row = pd.DataFrame(
        [{"date": pd.Timestamp(date).date().isoformat(), "ticker": ticker.upper(), "action": action,
          "shares": shares, "price": price, "note": note, "rule": rule}]
    )
    row.to_csv(_trades_path(name), mode="a", header=False, index=False)


def undo_last_trade(name: str) -> None:
    trades = load_trades(name)
    if trades.empty:
        return
    trades = trades.iloc[:-1]
    trades.to_csv(_trades_path(name), index=False)


def valuation_history(name: str) -> pd.DataFrame:
    """Daily mark-to-market value from creation date through today: cash +
    holdings, using the actual trade sequence (not just the current state)
    so past days reflect what was actually held at the time.
    """
    meta = load_meta(name)
    trades = load_trades(name)
    initial_cash = float(meta["initial_cash"])
    created = pd.Timestamp(meta["created"])
    today = pd.Timestamp(dt.date.today())

    if trades.empty:
        full_dates = pd.date_range(created, today, freq="D")
        return pd.DataFrame(
            {"date": full_dates, "cash": initial_cash, "holdings_value": 0.0, "total_value": initial_cash}
        )

    start = min(created, trades["date"].min())
    full_dates = pd.date_range(start, today, freq="D")

    signed_cash = trades.apply(
        lambda r: -r["shares"] * r["price"] if r["action"] == "BUY" else r["shares"] * r["price"], axis=1
    )
    cash_delta_by_date = signed_cash.groupby(trades["date"]).sum().reindex(full_dates).fillna(0.0)
    cash_daily = initial_cash + cash_delta_by_date.cumsum()

    signed_shares = trades.apply(
        lambda r: r["shares"] if r["action"] == "BUY" else -r["shares"], axis=1
    )
    positions_daily = (
        pd.DataFrame({"date": trades["date"], "ticker": trades["ticker"], "signed_shares": signed_shares})
        .pivot_table(index="date", columns="ticker", values="signed_shares", aggfunc="sum")
        .reindex(full_dates)
        .fillna(0.0)
        .cumsum()
    )

    tickers = sorted(positions_daily.columns)
    prices = get_prices(tickers, start=start.date().isoformat())
    prices = prices.reindex(full_dates).ffill()

    holdings_value = (positions_daily[tickers] * prices[tickers]).sum(axis=1)
    total_value = cash_daily + holdings_value

    return pd.DataFrame(
        {
            "date": full_dates,
            "cash": cash_daily.values,
            "holdings_value": holdings_value.values,
            "total_value": total_value.values,
        }
    )
