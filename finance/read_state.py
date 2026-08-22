"""Which cards a user has explicitly marked "read" in the UI (app.py's keep-card grids) --
deliberately separate from every other finance.* module's storage. Claims/fundamentals/
earnings_calls/macro all hold slow-changing, batch-generated data that's fine to persist as
per-ticker/per-series JSON files and commit to git. Read state is different: it's written live, by
an interactive user click, from wherever app.py happens to be running (e.g. a cloud deployment
whose filesystem isn't guaranteed to persist across restarts) -- so it needs a real always-on
store, not a local file.

Backed by Upstash Redis's free-tier REST API (plain HTTPS, no persistent connection -- a good fit
for a serverless/Streamlit-Cloud environment) when UPSTASH_REDIS_REST_URL/
UPSTASH_REDIS_REST_TOKEN are set (same "read a plain env var" convention finance.llm's
OPENROUTER_API_KEY already uses -- see app.py for how these get bridged in from st.secrets on
Streamlit Cloud, where secrets aren't plain env vars by default). Falls back to a local JSON file
(output/read_state.json) if those aren't configured, so local dev works with zero setup -- same
"soft-degrade, never crash the page" contract finance.macro's FRED/GDELT fetches already use.

Each user's read ids live in one Redis Set, keyed "read:{user}" -- SADD to mark read, SMEMBERS to
read the whole set back in one call (cheap local membership checks against a page's worth of
cards, rather than one round-trip per card). "user" exists from day one, not bolted on later, so a
future real multi-user setup is just a new value in this same key, not a schema change. Every
caller today passes the single hardcoded user "amir" (see app.py).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests

_STORE_PATH = Path("output/read_state.json")

# The only user today -- see this module's own docstring re: why "user" exists as a real dimension
# from day one anyway. Centralized here (rather than duplicated as a local constant in app.py and
# every finance.* module that needs to mark a card unread on overwrite) so there's exactly one
# place to introduce real multi-user support later.
CURRENT_USER = "amir"


def card_id(*parts: str) -> str:
    """A stable id for a non-claim card (fundamental/earnings-call/macro) -- claims already have
    their own stable `c.id` (finance.claims.claim_id). Same recipe: a truncated sha1 of whatever
    fields make one snapshot distinct from another (ticker/series+kind+date is enough since each of
    these event types is generated at most once per ticker/series per day under normal use).
    Shared by app.py (to render/look up a card's id) and finance.macro/fundamentals/earnings_calls
    (to mark a card unread again when a fresh refresh overwrites what an id already pointed to --
    see mark_unread's callers in those modules) so both sides always compute the exact same id.
    """
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _upstash_config() -> tuple[str, str] | None:
    url = os.environ.get("UPSTASH_REDIS_REST_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN")
    return (url, token) if url and token else None


def _upstash_command(command: list[str]) -> dict | None:
    """One Redis command via Upstash's REST API, or None (never raises) on any failure -- a
    network hiccup here should degrade to "treat as unread"/"drop this mark_read", not break the
    page, same soft-failure contract as every other network call in this app.
    """
    config = _upstash_config()
    if config is None:
        return None
    url, token = config
    try:
        response = requests.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=command, timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _load_local() -> dict[str, list[str]]:
    return json.loads(_STORE_PATH.read_text()) if _STORE_PATH.exists() else {}


def _save_local(data: dict[str, list[str]]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(data, indent=2))


def read_ids(user: str) -> set[str]:
    """Every card id `user` has ever explicitly marked read, as a set."""
    if _upstash_config() is not None:
        result = _upstash_command(["SMEMBERS", f"read:{user}"])
        if result is not None:
            return set(result.get("result") or [])
        return set()  # Upstash configured but unreachable -- fail open (nothing hidden), not closed
    return set(_load_local().get(user, []))


def mark_read(user: str, card_id: str) -> None:
    """Idempotent -- SADD (and the local-file fallback's own dedup) both already no-op a repeat
    mark on an already-read card.
    """
    if _upstash_config() is not None:
        _upstash_command(["SADD", f"read:{user}", card_id])
        return
    data = _load_local()
    ids = data.setdefault(user, [])
    if card_id not in ids:
        ids.append(card_id)
        _save_local(data)


def mark_unread(user: str, card_id: str) -> None:
    """The Read page's own "Unread" action -- actually removes the id (SREM/local-list removal),
    not just a display filter, so it's really gone from the store, not merely hidden again.
    Idempotent, same as mark_read.
    """
    if _upstash_config() is not None:
        _upstash_command(["SREM", f"read:{user}", card_id])
        return
    data = _load_local()
    ids = data.get(user, [])
    if card_id in ids:
        ids.remove(card_id)
        _save_local(data)
