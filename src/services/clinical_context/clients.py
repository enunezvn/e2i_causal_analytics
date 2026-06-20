"""Public biomedical REST clients for clinical-context enrichment.

Synchronous httpx clients mirroring ``src/data/kg/europe_pmc.py`` /
``src/data/kg/chembl.py`` exactly (context-manager, in-process ``lru_cache``,
distinct error type, ``client=`` injectable for ``httpx.MockTransport`` in
tests). They call the PUBLIC REST APIs directly — the claude.ai MCP tools are
agent-only and unavailable to the FastAPI backend.

  - ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2): study search
    -> primary outcome measures (the disease's pivotal endpoints).
  - PubMed E-utilities (https://eutils.ncbi.nlm.nih.gov/entrez/eutils): esearch
    -> top PMID; esummary -> title/journal/DOI (a real-world-evidence citation).

Both surface a SINGLE error class per source; transport / HTTP / JSON failures
all raise it so the service layer can degrade per-provider on one except clause.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

CLINICAL_TRIALS_BASE: str = "https://clinicaltrials.gov/api/v2"
PUBMED_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# Short default timeout: enrichment is best-effort and must not hold the request
# open. The service layer treats a timeout as "degrade to static fallback".
DEFAULT_TIMEOUT: float = 8.0
_LRU_MAXSIZE: int = 2048


class ClinicalTrialsError(Exception):
    """ClinicalTrials.gov request failed (transport, HTTP, or JSON)."""


class PubMedError(Exception):
    """PubMed E-utilities request failed (transport, HTTP, or JSON)."""


@dataclass(frozen=True)
class PubMedArticle:
    """One PubMed article summary (a real-world-evidence citation)."""

    pmid: str
    title: str
    journal: Optional[str] = None
    pubdate: Optional[str] = None
    doi: Optional[str] = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class ClinicalTrialsClient:
    """Synchronous ClinicalTrials.gov API v2 client."""

    def __init__(
        self,
        *,
        base: str = CLINICAL_TRIALS_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "ClinicalTrialsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def primary_endpoints(self, intervention: str, condition: str, *, limit: int = 8) -> List[str]:
        """Return the distinct primary outcome measures across COMPLETED studies
        of ``intervention`` for ``condition`` (the disease's pivotal endpoints),
        first-seen order preserved. Empty intervention/condition skip the network."""
        if not intervention or not condition:
            return []
        return list(_ctgov_primary_endpoints_cached(self, intervention, condition, limit))

    def _primary_endpoints_uncached(
        self, intervention: str, condition: str, limit: int
    ) -> tuple[str, ...]:
        try:
            response = self._client.get(
                f"{self._base}/studies",
                params={
                    "query.intr": intervention,
                    "query.cond": condition,
                    "fields": "NCTId,PrimaryOutcomeMeasure",
                    "pageSize": limit,
                    "filter.overallStatus": "COMPLETED",
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ClinicalTrialsError(f"ClinicalTrials transport error: {exc}") from exc
        if response.status_code >= 400:
            raise ClinicalTrialsError(
                f"ClinicalTrials HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ClinicalTrialsError(
                f"ClinicalTrials non-JSON body: {response.text[:200]!r}"
            ) from exc
        seen: list[str] = []
        for study in payload.get("studies") or []:
            outcomes = (
                study.get("protocolSection", {}).get("outcomesModule", {}).get("primaryOutcomes")
                or []
            )
            for outcome in outcomes:
                measure = outcome.get("measure") if isinstance(outcome, dict) else None
                if isinstance(measure, str) and measure and measure not in seen:
                    seen.append(measure)
        return tuple(seen)


class PubMedClient:
    """Synchronous PubMed E-utilities client (esearch + esummary)."""

    def __init__(
        self,
        *,
        base: str = PUBMED_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "PubMedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def top_article(self, term: str) -> Optional[PubMedArticle]:
        """esearch for ``term`` (relevance-sorted) -> the top PMID -> esummary.
        Returns None when there are no hits. Empty term skips the network."""
        if not term:
            return None
        return _pubmed_top_article_cached(self, term)

    def fetch_by_pmid(self, pmid: str) -> Optional[PubMedArticle]:
        """esummary for a specific PMID -> the article summary. Empty pmid skips
        the network; an unknown pmid returns None."""
        if not pmid:
            return None
        return _pubmed_fetch_by_pmid_cached(self, pmid)

    def _esearch_top_pmid(self, term: str) -> Optional[str]:
        try:
            response = self._client.get(
                f"{self._base}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmode": "json",
                    "retmax": 1,
                    "sort": "relevance",
                },
            )
        except httpx.HTTPError as exc:
            raise PubMedError(f"PubMed esearch transport error: {exc}") from exc
        if response.status_code >= 400:
            raise PubMedError(
                f"PubMed esearch HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise PubMedError(f"PubMed esearch non-JSON body: {response.text[:200]!r}") from exc
        idlist = payload.get("esearchresult", {}).get("idlist") or []
        return str(idlist[0]) if idlist else None

    def _esummary(self, pmid: str) -> Optional[PubMedArticle]:
        try:
            response = self._client.get(
                f"{self._base}/esummary.fcgi",
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
            )
        except httpx.HTTPError as exc:
            raise PubMedError(f"PubMed esummary transport error: {exc}") from exc
        if response.status_code >= 400:
            raise PubMedError(
                f"PubMed esummary HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise PubMedError(f"PubMed esummary non-JSON body: {response.text[:200]!r}") from exc
        record = payload.get("result", {}).get(str(pmid))
        if not isinstance(record, dict):
            return None
        title = record.get("title")
        if not isinstance(title, str) or not title:
            return None
        doi: Optional[str] = None
        for aid in record.get("articleids") or []:
            if isinstance(aid, dict) and aid.get("idtype") == "doi":
                value = aid.get("value")
                if isinstance(value, str) and value:
                    doi = value
                    break
        journal = record.get("source")
        pubdate = record.get("pubdate")
        return PubMedArticle(
            pmid=str(pmid),
            title=title,
            journal=journal if isinstance(journal, str) and journal else None,
            pubdate=pubdate if isinstance(pubdate, str) and pubdate else None,
            doi=doi,
        )

    def _top_article_uncached(self, term: str) -> Optional[PubMedArticle]:
        pmid = self._esearch_top_pmid(term)
        if not pmid:
            return None
        return self._esummary(pmid)


# ---------------------------------------------------------------------------
# Module-level LRU wrappers — keyed on (id(client), args), client-lifetime scope
# (same pattern as europe_pmc.py / chembl.py).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_LRU_MAXSIZE)
def _ctgov_primary_endpoints_cached(
    client: ClinicalTrialsClient, intervention: str, condition: str, limit: int
) -> tuple[str, ...]:
    return client._primary_endpoints_uncached(intervention, condition, limit)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _pubmed_top_article_cached(client: PubMedClient, term: str) -> Optional[PubMedArticle]:
    return client._top_article_uncached(term)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _pubmed_fetch_by_pmid_cached(client: PubMedClient, pmid: str) -> Optional[PubMedArticle]:
    return client._esummary(pmid)


def reset_caches() -> None:
    """Clear the in-process clinical-trials / pubmed caches (useful in tests)."""
    _ctgov_primary_endpoints_cached.cache_clear()
    _pubmed_top_article_cached.cache_clear()
    _pubmed_fetch_by_pmid_cached.cache_clear()
