"""Article-level investment claims extracted by finance.newsloop's Stage B
(`extract_claims`): a permanent, global, immutable record of what each
individual article implied about a ticker. One article can produce zero,
one, or several distinct claims about the same ticker -- each stored here
forever, never edited afterward. finance.tickerthesis's aggregator reads
across every claim ever collected for a ticker to synthesize the current
tradable view; this module only owns the raw evidence, not the synthesis.

Global, not per-portfolio -- unlike the old per-portfolio thesis confidence,
"what did this article claim about this ticker" doesn't depend on who's
asking, so it's computed once and shared by every portfolio (same reasoning
as Stage A's own global events cache).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

CLAIMS_DIR = Path("output/claims")

Direction = Literal["long", "short"]

# Valid range for any expected_horizon_days -- both a single claim's (finance.newsloop.extract_claims)
# and a ticker-thesis's aggregate one (finance.tickerthesis.aggregate_claims). A claim's own horizon
# is informational (fed to the aggregator as context), not used to route it anywhere -- every claim
# for a ticker, regardless of horizon, feeds the same single TickerThesis.
HORIZON_MIN_DAYS = 7
HORIZON_MAX_DAYS = 730


@dataclass
class ArticleClaim:
    id: str  # stable hash of (ticker, source_link, claim text) -- see claim_id()
    ticker: str
    source_link: str
    source_title: str
    event_type: str
    created: dt.date
    claim: str  # the directional claim, or the one-line reason if not trade_worthy
    direction: Direction | None
    confidence: float  # 0-1, this claim's own standalone confidence -- permanent, never revised
    importance: int  # 0-10, this specific claim's own significance (not the whole article's)
    expected_return_pct: float
    expected_horizon_days: int
    # No longer asked of Stage B (finance.newsloop.extract_claims) -- catalysts/invalidation only
    # make sense relative to a chosen direction, and Stage C (finance.tickerthesis.aggregate_claims)
    # already synthesizes its own from the full claim set regardless of whether individual claims
    # have them, so this was redundant cost/complexity. Kept as dataclass fields (always empty for
    # any claim extracted from here on) only so older stored claims with real values still load.
    catalysts: list[str] = field(default_factory=list)
    invalidation: list[str] = field(default_factory=list)
    trade_worthy: bool = False
    context: str = ""  # a paragraph from the article specific to this claim, for a reader who
    # doesn't want to re-read the whole source to understand why the claim was made


def claim_id(ticker: str, source_link: str, claim_text: str) -> str:
    return hashlib.sha1(f"{ticker}|{source_link}|{claim_text}".encode()).hexdigest()[:16]


def _path(ticker: str) -> Path:
    return CLAIMS_DIR / f"{ticker}.json"


def load_claims(ticker: str) -> list[ArticleClaim]:
    path = _path(ticker)
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    claims = []
    for row in rows:
        row = dict(row)
        row["created"] = dt.date.fromisoformat(row["created"])
        claims.append(ArticleClaim(**row))
    return claims


def append_claims(ticker: str, claims: list[ArticleClaim]) -> None:
    """Appends only -- claims are permanent evidence, never edited or removed
    once recorded. A no-op for an empty list (an article can legitimately
    yield zero claims about a ticker it merely mentions in passing).
    """
    if not claims:
        return
    path = _path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.extend(json.loads(json.dumps(asdict(c), default=str)) for c in claims)
    path.write_text(json.dumps(existing, indent=2))


def list_tickers_with_claims() -> list[str]:
    """Every ticker with at least one recorded ArticleClaim, regardless of
    whether a TickerThesis has ever been successfully aggregated for it --
    finance.newsloop.update_research sweeps this to catch tickers whose
    claims exist but whose aggregation (Stage C) failed or was never
    attempted, since Stage B alone never revisits an already-attempted
    (article, ticker) pair.
    """
    if not CLAIMS_DIR.exists():
        return []
    return sorted(p.stem for p in CLAIMS_DIR.glob("*.json"))
