"""RxNav public REST client.

RxNav is the NLM's drug terminology service. v1 of EntityLinker uses the
public ``rxnav.nlm.nih.gov/REST`` endpoint, which is unauthenticated. The
offline Docker variant (``RxNav-in-a-Box``) is the v2 target — it's a 12 GB
RAM / 100 GB disk install gated on procurement and not implementable today.

Surface used by EntityLinker:
    - ``rxcui_for_name(name)`` — drug name → ``RxCUIMatch(rxcui, approximate)``
    - ``rxcui_for_ndc(ndc)``    — NDC → RxCUI
    - ``properties(rxcui)``      — fetch the canonical name + TTY for an RxCUI

``RxCUIMatch`` distinguishes exact matches (canonical RxCUI for the name)
from approximate matches (RxNav's normalized-match fallback that silently
corrects typos). EntityLinker propagates the ``approximate`` flag into
``EntityLink.confidence`` so downstream consumers (Phase 2.6
CitationResolver, Phase 2.7 EnsembleVoter) can treat approximate matches
as lower-trust evidence.

References:
    - REST docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIREST.html
    - License: RxNorm is "unrestricted"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
RXNAV_BASE_URL_ENV = "RXNAV_BASE_URL"
DEFAULT_TIMEOUT = 10.0
_LRU_MAXSIZE = 4096


def _resolve_base_url(explicit: Optional[str]) -> str:
    """Resolve the RxNav base URL with precedence: explicit > env > default.

    Operators flip ``RXNAV_BASE_URL`` to redirect requests to a locally-hosted
    ``rxnav-in-a-box`` Docker instance (see ``docker/docker-compose.rxnav.yml``)
    without code changes. Defaults to the public NLM REST endpoint.

    The env var is read at instantiation (not import) so tests can use
    ``monkeypatch.setenv`` and per-process operator overrides work in long-
    running workers.
    """
    if explicit is not None:
        return explicit
    return os.environ.get(RXNAV_BASE_URL_ENV) or RXNAV_BASE


@dataclass(frozen=True)
class RxCUIMatch:
    """Result of a drug-name → RxCUI lookup with match-quality metadata.

    Attributes:
        rxcui: The resolved RxNorm Concept Unique Identifier.
        approximate: True if the lookup fell through to RxNav's
            approximate-match path (after exact match returned no hits).
            ``False`` indicates the input string matched an RxNorm
            canonical name or synonym verbatim.
    """

    rxcui: str
    approximate: bool


class RxNavError(Exception):
    """RxNav request failed."""


class RxNavClient:
    """Synchronous RxNav REST client."""

    def __init__(
        self,
        *,
        base: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        # Precedence: explicit `base=` > RXNAV_BASE_URL env > public NLM default.
        # Env var is read here (not at import) so tests/monkeypatch + per-worker
        # overrides take effect. See `_resolve_base_url` docstring + issue #246.
        self._base = _resolve_base_url(base).rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "RxNavClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        try:
            response = self._client.get(
                f"{self._base}{path}",
                params=params,
                # RxNav defaults to XML; force JSON.
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise RxNavError(f"RxNav transport error: {exc}") from exc
        if response.status_code >= 400:
            raise RxNavError(f"RxNav HTTP {response.status_code}: {response.text[:200]!r}")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise RxNavError(f"RxNav non-JSON body: {response.text[:200]!r}") from exc
        return payload

    def rxcui_for_name(self, name: str) -> Optional[RxCUIMatch]:
        """Return ``RxCUIMatch(rxcui, approximate)`` for a drug name, or None.

        Two-stage lookup:
            1. Exact match via ``/rxcui.json?name=...&search=0`` (matches
               canonical RxNorm names + synonyms verbatim). If hit,
               ``approximate=False``.
            2. If no exact hit, retry with ``/rxcui.json?name=...&search=2``
               which enables RxNav's normalized-match fallback (silently
               corrects typos and casing). If hit, ``approximate=True``.

        Surfacing the exact-vs-approximate distinction lets EntityLinker
        propagate match quality into ``EntityLink.confidence`` so
        downstream consumers (CitationResolver, EnsembleVoter) can weight
        approximate matches lower.
        """
        return _rxcui_for_name_cached(self, name)

    def _rxcui_for_name_uncached(self, name: str) -> Optional[RxCUIMatch]:
        if not name:
            return None
        # Stage 1: exact match only (search=0).
        payload = self._get("/rxcui.json", {"name": name, "search": 0})
        ids = payload.get("idGroup", {}).get("rxnormId") or []
        if isinstance(ids, list) and ids:
            return RxCUIMatch(rxcui=str(ids[0]), approximate=False)
        # Stage 2: fall back to normalized match (search=2).
        payload = self._get("/rxcui.json", {"name": name, "search": 2})
        ids = payload.get("idGroup", {}).get("rxnormId") or []
        if isinstance(ids, list) and ids:
            return RxCUIMatch(rxcui=str(ids[0]), approximate=True)
        return None

    def rxcui_for_ndc(self, ndc: str) -> Optional[str]:
        """Return the RxCUI mapped from an NDC code, or None."""
        return _rxcui_for_ndc_cached(self, ndc)

    def _rxcui_for_ndc_uncached(self, ndc: str) -> Optional[str]:
        if not ndc:
            return None
        payload = self._get("/ndcstatus.json", {"ndc": ndc})
        status = payload.get("ndcStatus", {})
        rxcui = status.get("rxcui")
        if isinstance(rxcui, str) and rxcui:
            return rxcui
        return None

    def properties(self, rxcui: str) -> Optional[dict[str, Any]]:
        """Return the property block for an RxCUI (name, tty, ...) or None."""
        return _properties_cached(self, rxcui)

    def _properties_uncached(self, rxcui: str) -> Optional[dict[str, Any]]:
        if not rxcui:
            return None
        payload = self._get(f"/rxcui/{rxcui}/properties.json")
        props = payload.get("properties")
        if isinstance(props, dict) and props:
            return props
        return None


@lru_cache(maxsize=_LRU_MAXSIZE)
def _rxcui_for_name_cached(client: RxNavClient, name: str) -> Optional[RxCUIMatch]:
    return client._rxcui_for_name_uncached(name)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _rxcui_for_ndc_cached(client: RxNavClient, ndc: str) -> Optional[str]:
    return client._rxcui_for_ndc_uncached(ndc)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _properties_cached(client: RxNavClient, rxcui: str) -> Optional[dict[str, Any]]:
    return client._properties_uncached(rxcui)


def reset_caches() -> None:
    """Clear RxNav caches (useful in tests)."""
    _rxcui_for_name_cached.cache_clear()
    _rxcui_for_ndc_cached.cache_clear()
    _properties_cached.cache_clear()
