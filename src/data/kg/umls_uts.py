"""UMLS UTS REST client.

Wraps the UMLS Terminology Services REST API (https://uts-ws.nlm.nih.gov/rest)
for the three operations EntityLinker needs:

1. ``search(term)`` — given a free-text term, return candidate concepts.
2. ``cui_lookup(cui)`` — given a CUI, fetch preferred name + semantic types.
3. ``crosswalk(code, source)`` — given a code in a source vocabulary
   (ICD10CM, RXNORM, LOINC, ...), return the candidate CUIs.

Auth:
    UMLS issues per-developer API keys. We read ``UMLS_UTS_API_KEY`` from the
    environment; production deployments inject this through Vault. The key is
    passed as a query parameter (the legacy CAS ticket flow has been retired by
    NLM in favor of plain key auth on the v1 REST endpoints).

Error model:
    - ``UMLSAuthError``     — 401/403; key missing or invalid.
    - ``UMLSNotFoundError`` — 404; code not in the requested source.
    - ``UMLSError``         — base class for all UMLS failures.

Caching:
    v1 uses ``functools.lru_cache`` for in-process memoization. UMLS concepts
    are stable across releases, so a process-lifetime cache with no TTL is
    safe. Production should swap to Redis with a 24h TTL — flagged in the
    docstrings of the cached helpers.

References:
    - UTS REST docs: https://documentation.uts.nlm.nih.gov/rest/
    - Cross-walk endpoint: ``/rest/crosswalk/current/source/{source}/{code}``
    - Search endpoint:    ``/rest/search/current``
    - CUI endpoint:       ``/rest/content/current/CUI/{cui}``
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

import httpx

from src.data.kg.types import KGConcept

logger = logging.getLogger(__name__)

UTS_BASE = "https://uts-ws.nlm.nih.gov/rest"
DEFAULT_TIMEOUT = 10.0
DEFAULT_VERSION = "current"


class UMLSError(Exception):
    """Base error for UMLS client failures."""


class UMLSAuthError(UMLSError):
    """API key is missing, invalid, or expired (HTTP 401/403)."""


class UMLSNotFoundError(UMLSError):
    """The requested code/CUI is not in the UMLS metathesaurus (HTTP 404)."""


class UMLSClient:
    """Thin synchronous client over the UTS REST API.

    Constructed once and reused; httpx.Client connection-pools internally.
    Methods raise ``UMLSError`` subclasses on failure; success paths return
    plain dicts/dataclasses.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        version: str = DEFAULT_VERSION,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("UMLS_UTS_API_KEY")
        if not key:
            raise UMLSAuthError(
                "UMLS_UTS_API_KEY not provided. Set the env var or pass api_key=..."
            )
        self._api_key = key
        self._version = version
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "UMLSClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        full_params = {"apiKey": self._api_key, **params}
        try:
            response = self._client.get(f"{UTS_BASE}{path}", params=full_params)
        except httpx.HTTPError as exc:
            raise UMLSError(f"UTS request failed: {exc}") from exc
        if response.status_code in (401, 403):
            raise UMLSAuthError(
                f"UMLS auth rejected (status={response.status_code}). "
                f"Verify UMLS_UTS_API_KEY is current."
            )
        if response.status_code == 404:
            raise UMLSNotFoundError(f"UTS 404: {path}")
        if response.status_code >= 400:
            raise UMLSError(
                f"UTS error: status={response.status_code} body={response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise UMLSError(f"UTS returned non-JSON body: {response.text[:200]!r}") from exc
        return payload

    def search(
        self,
        term: str,
        *,
        page_size: int = 5,
        search_type: str = "exact",
    ) -> list[dict[str, Any]]:
        """Free-text concept search.

        Returns a list of result rows, each containing ``ui`` (the CUI),
        ``name`` (preferred label), and ``rootSource`` (vocabulary).
        """
        if not term:
            return []
        payload = self._get(
            f"/search/{self._version}",
            {"string": term, "pageSize": page_size, "searchType": search_type},
        )
        results = payload.get("result", {}).get("results") or []
        # UTS returns a single sentinel row ``[{"ui": "NONE", ...}]`` for empty
        # searches; collapse that to an empty list so callers can rely on
        # falsy = no match without inspecting magic strings.
        if len(results) == 1 and results[0].get("ui") == "NONE":
            return []
        return results

    def cui_lookup(self, cui: str) -> KGConcept:
        """Resolve a CUI to its preferred name + semantic types."""
        return _cui_lookup_cached(self, cui)

    def _cui_lookup_uncached(self, cui: str) -> KGConcept:
        payload = self._get(f"/content/{self._version}/CUI/{cui}", {})
        result = payload.get("result", {})
        sem_types_raw = result.get("semanticTypes") or []
        sem_types = tuple(item.get("name", "") for item in sem_types_raw if item.get("name"))
        atom_count_raw = result.get("atomCount")
        atom_count = (
            int(atom_count_raw)
            if isinstance(atom_count_raw, (int, float, str)) and str(atom_count_raw).isdigit()
            else None
        )
        return KGConcept(
            cui=cui,
            preferred_name=result.get("name", ""),
            semantic_types=sem_types,
            atom_count=atom_count,
        )

    def crosswalk(
        self,
        code: str,
        *,
        source: str,
        target_source: Optional[str] = None,
        page_size: int = 25,
    ) -> list[dict[str, Any]]:
        """Cross-walk a source-vocabulary code to UMLS atoms / CUIs.

        Args:
            code: The code in the source vocabulary (e.g., ``"L20.9"``).
            source: The UTS source abbreviation (e.g., ``"ICD10CM"``,
                ``"RXNORM"``, ``"LOINC"``, ``"CPT"``, ``"HCPCS"``).
            target_source: Optional target vocabulary if you only want atoms
                from a particular source returned. Defaults to all sources.
            page_size: Pagination limit; UTS caps at 50 in practice.

        Returns:
            A list of atom rows; each row contains ``ui`` (atom UI),
            ``rootSource``, and ``name``. The caller typically maps these to
            CUIs by then calling ``atom_to_cui`` or ``search``.
        """
        return _crosswalk_cached(
            self, code=code, source=source, target_source=target_source, page_size=page_size
        )

    def _crosswalk_uncached(
        self,
        *,
        code: str,
        source: str,
        target_source: Optional[str],
        page_size: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"pageSize": page_size}
        if target_source:
            params["targetSource"] = target_source
        try:
            payload = self._get(f"/crosswalk/{self._version}/source/{source}/{code}", params)
        except UMLSNotFoundError:
            return []
        return list(payload.get("result", []))

    def code_to_cui(
        self,
        code: str,
        *,
        source: str,
    ) -> Optional[str]:
        """Resolve a source-vocabulary code to its metathesaurus CUI.

        Uses UTS's documented "code-to-CUI" path:
        ``/search/{version}?string={code}&inputType=sourceUi
        &sabs={source}&returnIdType=concept&searchType=exact``.

        This is preferred over ``/content/source/{source}/{code}`` because the
        atom-record endpoint returns the atom's *immediate* parent concept,
        which can be a more specific CUI than the canonical metathesaurus
        rollup. The search endpoint with ``returnIdType=concept`` returns the
        CUI the metathesaurus considers canonical for the code.

        Returns the first CUI found, or None if the code isn't in UMLS.
        """
        return _code_to_cui_cached(self, code=code, source=source)

    def _code_to_cui_uncached(
        self,
        *,
        code: str,
        source: str,
    ) -> Optional[str]:
        try:
            payload = self._get(
                f"/search/{self._version}",
                {
                    "string": code,
                    "inputType": "sourceUi",
                    "sabs": source,
                    "returnIdType": "concept",
                    "searchType": "exact",
                },
            )
        except UMLSNotFoundError:
            return None
        results = payload.get("result", {}).get("results") or []
        for row in results:
            ui = row.get("ui")
            if isinstance(ui, str) and ui.startswith("C") and ui != "NONE":
                return ui
        return None

    def cui_relations(
        self,
        cui: str,
        *,
        page_size: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the relations rows for a CUI from ``/content/CUI/{cui}/relations``.

        Each row carries:
            - ``relationLabel``: coarse relation type (e.g., ``"RB"`` related
              broader, ``"RN"`` related narrower, ``"PAR"`` parent, ``"CHD"``
              child, ``"SY"`` synonym, ``"RO"`` other related).
            - ``additionalRelationLabel``: fine-grained label (e.g.,
              ``"isa"``, ``"has_finding_site"``, ``"may_treat"``).
            - ``relatedId``: URL of the related CUI's content endpoint; the
              CUI itself is the trailing path segment.
            - ``relatedFromIdName`` / ``relatedIdName``: human-readable
              endpoint labels.
            - ``rootSource``: the source vocabulary that asserted the relation
              (e.g., ``"SNOMEDCT_US"``, ``"MSH"``).

        v1 callers typically post-process by extracting the trailing CUI from
        ``relatedId`` and grouping by ``additionalRelationLabel``.

        v2 punt — pagination: this method returns the first ``page_size``
        rows only. CUIs with hundreds of relations (rare but possible for
        broad parent concepts like ``C0011603`` "Dermatitis") will be
        silently truncated. If a downstream consumer reports missing edges,
        wire in a ``pageNumber`` loop here.
        """
        return _cui_relations_cached(self, cui=cui, page_size=page_size)

    def _cui_relations_uncached(
        self,
        *,
        cui: str,
        page_size: int,
    ) -> list[dict[str, Any]]:
        try:
            payload = self._get(
                f"/content/{self._version}/CUI/{cui}/relations",
                {"pageSize": page_size},
            )
        except UMLSNotFoundError:
            return []
        result = payload.get("result")
        return list(result) if isinstance(result, list) else []


# Module-level cache helpers. These bypass the ``self``-bound cache problem
# (mutable instance attrs make ``lru_cache`` on methods leak across instances)
# by keying on (id(client), <args>). The (id) component scopes the cache to the
# client's lifetime; when clients are short-lived (e.g., tests), the cache is
# implicitly invalidated.

_LRU_MAXSIZE = 4096


@lru_cache(maxsize=_LRU_MAXSIZE)
def _cui_lookup_cached(client: UMLSClient, cui: str) -> KGConcept:
    return client._cui_lookup_uncached(cui)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _crosswalk_cached(
    client: UMLSClient,
    *,
    code: str,
    source: str,
    target_source: Optional[str],
    page_size: int,
) -> list[dict[str, Any]]:
    return client._crosswalk_uncached(
        code=code, source=source, target_source=target_source, page_size=page_size
    )


@lru_cache(maxsize=_LRU_MAXSIZE)
def _code_to_cui_cached(
    client: UMLSClient,
    *,
    code: str,
    source: str,
) -> Optional[str]:
    return client._code_to_cui_uncached(code=code, source=source)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _cui_relations_cached(
    client: UMLSClient,
    *,
    cui: str,
    page_size: int,
) -> list[dict[str, Any]]:
    return client._cui_relations_uncached(cui=cui, page_size=page_size)


def reset_caches() -> None:
    """Clear UMLS in-process caches (useful in tests)."""
    _cui_lookup_cached.cache_clear()
    _crosswalk_cached.cache_clear()
    _code_to_cui_cached.cache_clear()
    _cui_relations_cached.cache_clear()
