"""RxNav public REST client.

RxNav is the NLM's drug terminology service. v1 of EntityLinker uses the
public ``rxnav.nlm.nih.gov/REST`` endpoint, which is unauthenticated. The
offline Docker variant (``RxNav-in-a-Box``) is the v2 target — it's a 12 GB
RAM / 100 GB disk install gated on procurement and not implementable today.

Surface used by EntityLinker:
    - ``rxcui_for_name(name)`` — drug name → RxCUI (canonical ID)
    - ``rxcui_for_ndc(ndc)``    — NDC → RxCUI
    - ``properties(rxcui)``      — fetch the canonical name + TTY for an RxCUI

References:
    - REST docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIREST.html
    - License: RxNorm is "unrestricted"
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
DEFAULT_TIMEOUT = 10.0
_LRU_MAXSIZE = 4096


class RxNavError(Exception):
    """RxNav request failed."""


class RxNavClient:
    """Synchronous RxNav REST client."""

    def __init__(
        self,
        *,
        base: str = RXNAV_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
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

    def rxcui_for_name(self, name: str) -> Optional[str]:
        """Return the canonical RxCUI for a drug name, or None.

        Uses ``/rxcui.json?name=...&search=2`` (search=2 enables approximate
        match if exact fails).
        """
        return _rxcui_for_name_cached(self, name)

    def _rxcui_for_name_uncached(self, name: str) -> Optional[str]:
        if not name:
            return None
        payload = self._get("/rxcui.json", {"name": name, "search": 2})
        ids = payload.get("idGroup", {}).get("rxnormId") or []
        if isinstance(ids, list) and ids:
            return str(ids[0])
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
def _rxcui_for_name_cached(client: RxNavClient, name: str) -> Optional[str]:
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
