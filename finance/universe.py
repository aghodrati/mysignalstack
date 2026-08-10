"""Static reference data for the comparison app: quick-pick baskets of well-known
tickers, grouped by category, plus the S&P 500 benchmark ticker.
"""

from __future__ import annotations

import json
from pathlib import Path

CUSTOM_TICKERS_PATH = Path("output/custom_tickers.json")


def load_custom_tickers() -> list[str]:
    """Tickers a user typed into the sidebar's "Other tickers" box, persisted
    to disk so they're still there next time the app opens (unlike the
    QUICK_PICK_CATEGORIES baskets below, which are fixed in code).
    """
    if not CUSTOM_TICKERS_PATH.exists():
        return []
    return json.loads(CUSTOM_TICKERS_PATH.read_text())


def save_custom_tickers(tickers: list[str]) -> None:
    CUSTOM_TICKERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_TICKERS_PATH.write_text(json.dumps(sorted(tickers), indent=2))


TOP_COMPANIES: dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN",
    "Meta Platforms": "META",
    "Tesla": "TSLA",
    "Oracle": "ORCL",
    "Uber": "UBER",
    "Reddit": "RDDT",
    "Berkshire Hathaway": "BRK-B",
}

SEMICONDUCTORS: dict[str, str] = {
    "Nvidia": "NVDA",
    "TSMC": "TSM",
    "ASML": "ASML",
    "AMD": "AMD",
    "ARM": "ARM",
    "Qualcomm": "QCOM",
    "Micron": "MU",
    "Intel": "INTC",
    "Nebius": "NBIS",
    "Broadcom": "AVGO",
}

CRYPTO: dict[str, str] = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
}

METALS: dict[str, str] = {
    "Gold": "GLD",
    "Silver": "SLV",
    "Copper": "CPER",
}

OIL: dict[str, str] = {
    "ExxonMobil": "XOM",
    "Chevron": "CVX",
}

DEFENSE: dict[str, str] = {
    "Palantir": "PLTR",
    "Lockheed Martin": "LMT",
}

FUTURISTIC: dict[str, str] = {
    "D-Wave Quantum": "QBTS",
    "Rigetti Computing": "RGTI",
    "Oklo": "OKLO",
    "SPCX": "SPCX",
}

# SPDR Select Sector ETFs: the standard tradable proxies for S&P 500 GICS sectors.
SP500_SECTORS: dict[str, str] = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

# Category label -> {display name: ticker}, rendered as one quick-pick row each in the app.
QUICK_PICK_CATEGORIES: dict[str, dict[str, str]] = {
    "Big": TOP_COMPANIES,
    "Semiconductors": SEMICONDUCTORS,
    "Crypto": CRYPTO,
    "Metals": METALS,
    "Oil & gas": OIL,
    "Defense": DEFENSE,
    "Futuristic": FUTURISTIC,
    "S&P sectors": SP500_SECTORS,
}

SP500_BENCHMARK = "SPY"
