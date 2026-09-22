"""RxNav-backed brand aliases for chat entity extraction.

Lane 2 of docs/superpowers/specs/2026-09-22-public-apis-live-path-design.md.

``EntityVocabulary.from_default`` ships a hand-curated alias table. Measured on
2026-09-22 it resolved "ribociclib" but not "iptacopan" or "Rhapsido". RxNav's
``related.json`` knows every ingredient <-> brand pair, so this module asks it
once per process and hands back lowercase aliases per canonical brand.

Degrade rules (the point of this module):
- any ``RxNavError`` returns what was gathered so far and STOPS the round, so an
  outage costs one timeout, not one per brand;
- a failed round is remembered for ``NEGATIVE_TTL_S`` before RxNav is asked
  again; a successful round for ``TTL_S``;
- ``RXNAV_BRAND_ALIASES=0`` disables the lookup entirely (offline unit runs,
  operator rollback). The curated table is untouched either way.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Protocol

from src.data.kg.rxnav import RxCUIMatch, RxNavClient, RxNavError

logger = logging.getLogger(__name__)

__all__ = ["NEGATIVE_TTL_S", "TTL_S", "reset_cache", "rxnav_brand_aliases"]

TTL_S = 24 * 60 * 60
NEGATIVE_TTL_S = 10 * 60
CLIENT_TIMEOUT_S = 2.0
MIN_ALIAS_LEN = 4
ENV_SWITCH = "RXNAV_BRAND_ALIASES"


class _RxNavLike(Protocol):
    def rxcui_for_name(self, name: str) -> Optional[RxCUIMatch]: ...

    def related_names(self, rxcui: str, *, ttys: tuple[str, ...] = ...) -> list[str]: ...


_lock = threading.Lock()
# key -> (expires_at, aliases). A failed round stores {} with the shorter TTL.
_cache: Dict[tuple[str, ...], tuple[float, Dict[str, List[str]]]] = {}


def _now() -> float:  # patched in tests
    return time.monotonic()


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def _enabled() -> bool:
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in ("0", "false", "no", "off")


def _aliases_for(brand: str, client: _RxNavLike) -> List[str]:
    match = client.rxcui_for_name(brand)
    if match is None:
        return []
    own = brand.strip().lower()
    out: List[str] = []
    for name in client.related_names(match.rxcui):
        alias = name.strip().lower()
        if len(alias) < MIN_ALIAS_LEN or alias == own or alias in out:
            continue
        out.append(alias)
    return out


def _fetch_round(brands: tuple[str, ...], client: _RxNavLike) -> tuple[Dict[str, List[str]], bool]:
    """Returns (aliases, complete). ``complete`` is False when RxNav raised."""
    gathered: Dict[str, List[str]] = {}
    for brand in brands:
        try:
            aliases = _aliases_for(brand, client)
        except RxNavError as exc:
            logger.warning(
                "brand_aliases: RxNav unavailable while resolving %r (%s); "
                "keeping curated aliases only for the remaining brands",
                brand,
                exc,
            )
            return gathered, False
        if aliases:
            gathered[brand] = aliases
    return gathered, True


def rxnav_brand_aliases(
    brands: Iterable[str], *, client: Optional[_RxNavLike] = None
) -> Dict[str, List[str]]:
    """Lowercase RxNav aliases per canonical brand; ``{}`` when disabled or down."""
    if not _enabled():
        return {}
    # Ask in the caller's order (the canonical brand list) so a partial round
    # degrades predictably; cache on the order-free set.
    ordered = tuple(dict.fromkeys(brands))
    key = tuple(sorted(ordered))
    if not key:
        return {}
    with _lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] > _now():
            return {b: list(a) for b, a in hit[1].items()}
    owns_client = client is None
    rx: _RxNavLike = client if client is not None else RxNavClient(timeout=CLIENT_TIMEOUT_S)
    try:
        gathered, complete = _fetch_round(ordered, rx)
    finally:
        if owns_client:
            rx.close()  # type: ignore[attr-defined]
    ttl = TTL_S if complete else NEGATIVE_TTL_S
    with _lock:
        _cache[key] = (_now() + ttl, gathered)
    return {b: list(a) for b, a in gathered.items()}
