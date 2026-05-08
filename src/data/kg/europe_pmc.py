"""Europe PMC REST client for PMID/DOI → abstract resolution.

Phase 2.6 ``CitationResolver`` uses this client to retrieve the abstract
text for a PMID or DOI cited in Open Targets evidence rows. Europe PMC is
zero-auth and CC0-friendly; rate limits are generous for read-only access.

Endpoints:
    - Search: ``GET /europepmc/webservices/rest/search`` with
      ``query=ext_id:{pmid}&format=json`` for PMID lookup.
    - Article: ``GET /europepmc/webservices/rest/article/MED/{pmid}`` for
      direct article fetch.

The search endpoint is preferred because it accepts both PMIDs and DOIs
uniformly via the ``query`` parameter; the article endpoint only accepts
PMIDs in the ``MED/`` source.

References:
    - REST docs: https://europepmc.org/RestfulWebService
    - License: CC0 (https://europepmc.org/Help#contentlicensing)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import httpx

from src.data.kg.types import AbstractRecord

logger = logging.getLogger(__name__)

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_TIMEOUT = 15.0
_LRU_MAXSIZE = 4096


class EuropePMCError(Exception):
    """Europe PMC request failed."""


class EuropePMCClient:
    """Synchronous Europe PMC REST client."""

    def __init__(
        self,
        *,
        base: str = EUROPE_PMC_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "EuropePMCClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def fetch_abstract(self, pmid: str) -> Optional[AbstractRecord]:
        """Fetch the abstract for a PMID. Returns None when not found.

        The PMID is queried via the search endpoint with
        ``query=ext_id:{pmid} AND src:MED`` and ``resultType=core`` so the
        response includes the abstract text. ``MED`` is the canonical
        Europe PMC source for PubMed records.
        """
        return _fetch_abstract_cached(self, pmid)

    def _fetch_abstract_uncached(self, pmid: str) -> Optional[AbstractRecord]:
        if not pmid:
            return None
        try:
            response = self._client.get(
                f"{self._base}/search",
                params={
                    "query": f"ext_id:{pmid} AND src:MED",
                    "format": "json",
                    "resultType": "core",
                    "pageSize": 1,
                },
            )
        except httpx.HTTPError as exc:
            raise EuropePMCError(f"Europe PMC transport error: {exc}") from exc
        if response.status_code >= 400:
            raise EuropePMCError(f"Europe PMC HTTP {response.status_code}: {response.text[:200]!r}")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise EuropePMCError(f"Europe PMC non-JSON body: {response.text[:200]!r}") from exc
        result_list = payload.get("resultList", {})
        results = result_list.get("result") if isinstance(result_list, dict) else None
        if not isinstance(results, list) or not results:
            return None
        # Codex review MEDIUM (2026-05-08): defensive validation. The
        # query ``ext_id:{pmid} AND src:MED`` should only ever return
        # records matching that exact PMID + MED source, but if Europe
        # PMC's relevance ranker ever changes (or returns a normalized
        # variant first) we'd silently verify the wrong abstract. Reject
        # any record whose ``id``/``pmid`` doesn't match the request and
        # whose ``source`` isn't ``MED``.
        record = next(
            (
                r
                for r in results
                if isinstance(r, dict)
                and (str(r.get("pmid") or r.get("id") or "") == str(pmid))
                and (r.get("source") in (None, "MED"))
            ),
            None,
        )
        if record is None:
            return None
        abstract_text = record.get("abstractText") or ""
        if not abstract_text:
            # The search hit may have stripped the abstract for partial
            # records (e.g., if the article was withdrawn). Return None so
            # CitationResolver treats this as "couldn't verify".
            return None
        year_raw = record.get("pubYear")
        year = int(year_raw) if isinstance(year_raw, str) and year_raw.isdigit() else None
        return AbstractRecord(
            identifier=str(pmid),
            identifier_kind="pmid",
            title=str(record.get("title") or ""),
            abstract=str(abstract_text),
            source="europe_pmc",
            journal=str(record.get("journalTitle") or "") or None,
            year=year,
            raw=record,
        )


@lru_cache(maxsize=_LRU_MAXSIZE)
def _fetch_abstract_cached(client: EuropePMCClient, pmid: str) -> Optional[AbstractRecord]:
    return client._fetch_abstract_uncached(pmid)


def reset_caches() -> None:
    """Clear Europe PMC caches (useful in tests)."""
    _fetch_abstract_cached.cache_clear()
