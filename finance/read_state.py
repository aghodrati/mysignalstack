"""Which cards a user has explicitly marked "read" or "favorite" in the UI (app.py's keep-card
grids) -- deliberately separate from every other finance.* module's storage. Claims/fundamentals/
earnings_calls/macro all hold slow-changing, batch-generated data that's fine to persist as
per-ticker/per-series JSON files and commit to git. This is different: it's written live, by
an interactive user gesture, from wherever app.py happens to be running (e.g. a cloud deployment
whose filesystem isn't guaranteed to persist across restarts) -- so it needs a real always-on
store, not a local file.

Backed by Upstash Redis's free-tier REST API (plain HTTPS, no persistent connection -- a good fit
for a serverless/Streamlit-Cloud environment) when UPSTASH_REDIS_REST_URL/
UPSTASH_REDIS_REST_TOKEN are set (same "read a plain env var" convention finance.llm's
OPENROUTER_API_KEY already uses -- see app.py for how these get bridged in from st.secrets on
Streamlit Cloud, where secrets aren't plain env vars by default). Falls back to a local JSON file
per kind (output/read_state.json, output/favorite_state.json) if those aren't configured, so local
dev works with zero setup -- same "soft-degrade, never crash the page" contract finance.macro's
FRED/news-source fetches already use.

"Read" and "favorite" are two independent tags on the same card id, not mutually exclusive and not
implying each other -- reading a card means "seen it, hide it from the normal feed"; favoriting one
means "keep this visible/easy to find later" and does NOT hide it anywhere (see app.py's
_render_keep_card_grid: only read status affects default-page visibility). Each lives in its own
Redis Sorted Set, keyed "{kind}:{user}" ("read:amir", "favorite:amir") -- ZADD to mark (score = the
Unix timestamp of when it was marked), ZRANGE to read the whole set back in one call for membership
checks (cheap local checks against a page's worth of cards, rather than one round-trip per card),
ZREVRANGE to read it back newest-marked-first for display order (see read_ids_ordered/
favorite_ids_ordered). A plain Set (SADD/SMEMBERS) was used here originally, but Sets have no
ordering at all -- SMEMBERS returns members in hash-bucket order, not insertion order, so "show my
most recently read cards first" was structurally impossible without this. "user" exists from day
one, not bolted on later, so a future real multi-user setup is just a new value in this same key,
not a schema change. Every caller today passes the single hardcoded user CURRENT_USER.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import requests

_STORE_PATHS = {
    "read": Path("output/read_state.json"),
    "favorite": Path("output/favorite_state.json"),
    "pin": Path("output/pin_state.json"),
}

# Module-level, not created fresh per call: a plain requests.post() each time pays a new TCP+TLS
# handshake per request, which is most of the latency to Upstash's REST endpoint given how small
# the actual payload is. A shared Session keeps the underlying connection alive (HTTP keep-alive)
# across the several read_state calls a single button click's rerun makes (see
# _visible_cards_and_action), so only the first call in a run pays the handshake.
_session = requests.Session()

# In-process TTL cache removed (was a plain dict + _CACHE_TTL_SECONDS): it was meant to bound
# cross-process staleness (a second device marking cards against the same Redis) while still
# collapsing the several _ids() calls a single page rerun makes into one Upstash round trip, but
# after a run_loop_a-vs-app.py staleness bug that didn't clear even on browser refresh (root cause
# not pinned down), caching anywhere in this read path is off the table for now -- every call below
# hits Upstash (or the local-file fallback) directly, live.


def _upstash_command_async(command: list[str]) -> None:
    """Fire-and-forget version of _upstash_command, for writes (SADD/SREM) whose HTTP response
    nothing downstream reads -- waiting on Upstash's own round-trip here too was pure added latency
    for a response nobody uses. Runs in a daemon thread so it can never block process exit; genuinely
    fire-and-forget, not retried or awaited anywhere, matching _upstash_command's own soft-failure
    contract (a lost write here just means this one mark doesn't persist, same as any other network
    hiccup this module already tolerates).
    """
    threading.Thread(target=_upstash_command, args=(command,), daemon=True).start()


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
    network hiccup here should degrade to "treat as unread"/"drop this mark", not break the page,
    same soft-failure contract as every other network call in this app.
    """
    config = _upstash_config()
    if config is None:
        return None
    url, token = config
    try:
        response = _session.post(
            url, headers={"Authorization": f"Bearer {token}"}, json=command, timeout=10,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def _load_local(kind: str) -> dict[str, dict[str, float]]:
    """{user: {card_id: marked_at_unix_timestamp}} -- a dict (not the old plain list of ids) so the
    local-file fallback can answer "most recently marked first" the same way the Redis Sorted Set
    backend does (see _ordered_ids), instead of relying on incidental list-append order.
    """
    path = _STORE_PATHS[kind]
    return json.loads(path.read_text()) if path.exists() else {}


def _save_local(kind: str, data: dict[str, dict[str, float]]) -> None:
    path = _STORE_PATHS[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _ids(kind: str, user: str) -> set[str]:
    cache_key = f"{kind}:{user}"
    if _upstash_config() is not None:
        # ZRANGE (not SMEMBERS -- this key is a Sorted Set now, see this module's own docstring)
        # 0 -1 returns every member regardless of score; order doesn't matter here since this is
        # only used for membership checks -- see _ordered_ids for the newest-first display query.
        result = _upstash_command(["ZRANGE", cache_key, "0", "-1"])
        # Upstash configured but unreachable -- fail open (nothing hidden), not closed.
        if result is None:
            return set()
        return set(result.get("result") or [])
    return set(_load_local(kind).get(user, {}).keys())


def _ordered_ids(kind: str, user: str) -> list[str]:
    """Every id for `kind`/`user`, most-recently-marked first -- the one query `_ids` can't answer,
    since it collapses to an unordered set (fine for membership checks, not for display order).
    """
    key = f"{kind}:{user}"
    if _upstash_config() is not None:
        result = _upstash_command(["ZREVRANGE", key, "0", "-1"])
        return list(result.get("result") or []) if result is not None else []
    # Local fallback stores {id: marked_at_timestamp} -- sort by that directly, newest first.
    data = _load_local(kind).get(user, {})
    return [cid for cid, _ in sorted(data.items(), key=lambda item: item[1], reverse=True)]


def read_ids_ordered(user: str) -> list[str]:
    """Every card id `user` has ever explicitly marked read, most-recently-marked first."""
    return _ordered_ids("read", user)


def favorite_ids_ordered(user: str) -> list[str]:
    """Every card id `user` has ever explicitly starred, most-recently-marked first."""
    return _ordered_ids("favorite", user)


def _mark(kind: str, user: str, card_id: str) -> None:
    """Idempotent -- ZADD re-scores rather than duplicating on a repeat mark (and the local-file
    fallback's dict assignment is naturally the same), so marking an already-read card again just
    bumps its timestamp to "now" rather than erroring or double-adding. No in-process cache to keep
    in sync anymore (see module-level comment above) -- the next `_ids`/`_ordered_ids` call, in this
    process or any other, just re-fetches live, so the async Upstash write below racing a same-rerun
    re-read is the one place this can show a stale value for a moment (acceptable for now).
    """
    marked_at = time.time()
    if _upstash_config() is not None:
        # ZADD score member -- score is when this mark happened, so ZREVRANGE (_ordered_ids) can
        # answer "most recently marked first". See this module's own docstring for why this is a
        # Sorted Set rather than the plain Set (SADD) it used to be.
        _upstash_command_async(["ZADD", f"{kind}:{user}", str(marked_at), card_id])
        return
    data = _load_local(kind)
    data.setdefault(user, {})[card_id] = marked_at
    _save_local(kind, data)


def _unmark(kind: str, user: str, card_id: str) -> None:
    """Actually removes the id (ZREM/local-dict removal), not just a display filter, so it's really
    gone from the store, not merely hidden again. Idempotent, same as _mark. See _mark's docstring
    re: no in-process cache to keep in sync anymore.
    """
    if _upstash_config() is not None:
        _upstash_command_async(["ZREM", f"{kind}:{user}", card_id])
        return
    data = _load_local(kind)
    ids = data.get(user, {})
    if card_id in ids:
        del ids[card_id]
        _save_local(kind, data)


def read_ids(user: str) -> set[str]:
    """Every card id `user` has ever explicitly marked read, as a set."""
    return _ids("read", user)


def mark_read(user: str, card_id: str) -> None:
    _mark("read", user, card_id)


def mark_unread(user: str, card_id: str) -> None:
    """The Read page's own "Unread" action."""
    _unmark("read", user, card_id)


def favorite_ids(user: str) -> set[str]:
    """Every card id `user` has ever explicitly starred, as a set."""
    return _ids("favorite", user)


def mark_favorite(user: str, card_id: str) -> None:
    _mark("favorite", user, card_id)


def mark_unfavorite(user: str, card_id: str) -> None:
    """The Favorites page's own "Unfavorite" action."""
    _unmark("favorite", user, card_id)


def pin_ids(user: str) -> set[str]:
    """Every card id `user` has ever explicitly pinned, as a set. A third, independent tag
    alongside read/favorite -- currently only meaningful on the Discovery page (see app.py's
    _render_discovery_page): pinning a discovery candidate keeps it from being discarded and sorts
    it to the top there, but deliberately does NOT show up on the Favorites page or count as a
    favorite -- "worth protecting from an accidental swipe" and "worth keeping visible everywhere
    else in the app" are different things, so this doesn't reuse the favorite kind/Redis key.
    """
    return _ids("pin", user)


def pin_ids_ordered(user: str) -> list[str]:
    """Every card id `user` has ever explicitly pinned, most-recently-pinned first."""
    return _ordered_ids("pin", user)


def mark_pin(user: str, card_id: str) -> None:
    _mark("pin", user, card_id)


def mark_unpin(user: str, card_id: str) -> None:
    _unmark("pin", user, card_id)
