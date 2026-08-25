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
Redis Set, keyed "{kind}:{user}" ("read:amir", "favorite:amir") -- SADD to mark, SMEMBERS to read
the whole set back in one call (cheap local membership checks against a page's worth of cards,
rather than one round-trip per card). "user" exists from day one, not bolted on later, so a future
real multi-user setup is just a new value in this same key, not a schema change. Every caller today
passes the single hardcoded user CURRENT_USER.
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
}

# Module-level, not created fresh per call: a plain requests.post() each time pays a new TCP+TLS
# handshake per request, which is most of the latency to Upstash's REST endpoint given how small
# the actual payload is. A shared Session keeps the underlying connection alive (HTTP keep-alive)
# across the several read_state calls a single button click's rerun makes (see
# _visible_cards_and_action), so only the first call in a run pays the handshake.
_session = requests.Session()

# In-process cache of each kind's full id set, keyed "{kind}:{user}" -- there's exactly one real
# user today (CURRENT_USER, see below), so this is safe to share across every session/rerun in the
# process rather than scoped per Streamlit session: on Streamlit Community Cloud specifically (the
# deployment this was actually slow on -- confirmed fast on localhost, where the difference is
# purely network latency to Upstash) a round trip to Upstash's REST API is the dominant cost of a
# Fav/Unfav click, and _ids() gets called from several places on every page rerun (both the
# page-level filter in _render_read_page/_render_favorites_page AND the shared grid's
# _visible_cards_and_action -- and the latter runs once per grid on a page, so a page with several
# grids, e.g. Recent's Upcoming/Just Reported/main sections, still means several calls per rerun).
# Mutating the cache locally on every mark/unmark (see _mark/_unmark) turns every read after the
# first into a plain in-memory set lookup.
#
# Bounded by _CACHE_TTL_SECONDS rather than cached for the whole process lifetime -- an unbounded
# per-process cache does NOT self-heal if Upstash's data changes from outside this process (e.g. a
# second real device/browser marking cards read against the same Redis, or a second localhost/Cloud
# instance), which is exactly the "marked read on iPad, still shows unread on localhost" symptom
# this was confirmed to cause live: two separate Python processes (the iPad's Streamlit Cloud pod
# and the localhost dev server) each cached their own snapshot of read:amir once and never refreshed
# it from Redis again for the rest of their process lifetime, so localhost kept serving a stale
# snapshot from before the iPad's marks landed, indefinitely. A short TTL re-fetches periodically
# instead, bounding cross-process staleness to _CACHE_TTL_SECONDS while still collapsing the several
# same-rerun calls above into one real Upstash round trip.
_cache: dict[str, set[str]] = {}
_cache_fetched_at: dict[str, float] = {}
_CACHE_TTL_SECONDS = 5.0


def _upstash_command_async(command: list[str]) -> None:
    """Fire-and-forget version of _upstash_command, for writes (SADD/SREM) whose HTTP response
    nothing downstream reads. _mark/_unmark already update `_cache` synchronously before this is
    called, so the UI-visible state is correct immediately -- waiting on Upstash's own round-trip
    here too was pure added latency for a response nobody uses. Runs in a daemon thread so it can
    never block process exit; genuinely fire-and-forget, not retried or awaited anywhere, matching
    _upstash_command's own soft-failure contract (a lost write here just means this one mark doesn't
    persist, same as any other network hiccup this module already tolerates).
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


def _load_local(kind: str) -> dict[str, list[str]]:
    path = _STORE_PATHS[kind]
    return json.loads(path.read_text()) if path.exists() else {}


def _save_local(kind: str, data: dict[str, list[str]]) -> None:
    path = _STORE_PATHS[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _ids(kind: str, user: str) -> set[str]:
    cache_key = f"{kind}:{user}"
    if cache_key in _cache and time.time() - _cache_fetched_at.get(cache_key, 0.0) < _CACHE_TTL_SECONDS:
        return _cache[cache_key]
    if _upstash_config() is not None:
        result = _upstash_command(["SMEMBERS", cache_key])
        # Upstash configured but unreachable -- fail open (nothing hidden), not closed. Keeps
        # serving whatever was last cached (better than silently un-hiding every read card on a
        # transient hiccup) rather than wiping it; only a real fetch refreshes _cache_fetched_at,
        # so the very next call retries instead of trusting this failure for the rest of the TTL.
        if result is None:
            return _cache.get(cache_key, set())
        ids = set(result.get("result") or [])
    else:
        ids = set(_load_local(kind).get(user, []))
    _cache[cache_key] = ids
    _cache_fetched_at[cache_key] = time.time()
    return ids


def _mark(kind: str, user: str, card_id: str) -> None:
    """Idempotent -- SADD (and the local-file fallback's own dedup) both already no-op a repeat
    mark on an already-marked card. Updates `_cache` synchronously so every reader sees the new
    state immediately, regardless of how long the actual Upstash write (fired async, see
    _upstash_command_async) takes to land.

    Calls `_ids` first (not a blind `_cache.setdefault`) so a mark/unmark that happens to be the
    very first read_state call in a process still gets the real existing set from Redis into the
    cache before mutating it -- `_cache.setdefault(key, set())` used to silently fabricate an empty
    "known" cache entry in that case, so a mark right after an unmark (or vice versa) with no
    intervening real read would leave `_cache` claiming this key has only the marks made THIS
    process, hiding whatever was already there until the next TTL-driven refetch (see _cache's own
    docstring).

    Also refreshes `_cache_fetched_at` to "now" -- otherwise a slow-arriving async Upstash write
    (see _upstash_command_async) racing against the next TTL expiry could see this mutation
    reverted by a refetch that ran before the write actually landed in Redis.
    """
    _ids(kind, user).add(card_id)
    _cache_fetched_at[f"{kind}:{user}"] = time.time()
    if _upstash_config() is not None:
        _upstash_command_async(["SADD", f"{kind}:{user}", card_id])
        return
    data = _load_local(kind)
    ids = data.setdefault(user, [])
    if card_id not in ids:
        ids.append(card_id)
        _save_local(kind, data)


def _unmark(kind: str, user: str, card_id: str) -> None:
    """Actually removes the id (SREM/local-list removal), not just a display filter, so it's really
    gone from the store, not merely hidden again. Idempotent, same as _mark. Updates `_cache`
    synchronously -- see _mark's docstring for why this calls `_ids` first rather than mutating a
    blind `_cache.setdefault`, and for why `_cache_fetched_at` is refreshed too.
    """
    _ids(kind, user).discard(card_id)
    _cache_fetched_at[f"{kind}:{user}"] = time.time()
    if _upstash_config() is not None:
        _upstash_command_async(["SREM", f"{kind}:{user}", card_id])
        return
    data = _load_local(kind)
    ids = data.get(user, [])
    if card_id in ids:
        ids.remove(card_id)
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
