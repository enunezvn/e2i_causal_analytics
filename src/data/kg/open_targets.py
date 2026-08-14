"""Open Targets GraphQL client.

Open Targets is the EMBL-EBI / GSK / Sanofi platform that integrates
drug-disease evidence with structured provenance (Europe PMC PMIDs, clinical
trials, ChEMBL records). The GraphQL endpoint is unauthenticated and CC0
licensed; release 25.12 is current.

This v1 wrapper exposes only what ``CausalRoleClassifier`` and
``CitationResolver`` consume:

- ``drug_disease_evidence(drug_chembl_id, disease_efo_id)`` — returns a list
  of evidence rows with score, datatype, and Europe PMC ID where available.
- ``search_drug(name)`` — resolve a drug name to a ChEMBL ID.
- ``search_disease(name)`` — resolve a disease name to an EFO/MONDO ID.

The GraphQL queries are kept inline (small) to avoid a separate ``.graphql``
asset shipping concern. ``query_raw`` is exposed for callers who need a
custom fragment.

References:
    - GraphQL endpoint: https://api.platform.opentargets.org/api/v4/graphql
    - GraphiQL UI: https://api.platform.opentargets.org/api/v4/graphiql
    - License: https://platform-docs.opentargets.org/licence (CC0 1.0)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

OPEN_TARGETS_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_TIMEOUT = 15.0
_LRU_MAXSIZE = 2048


# Drug -> disease therapeutic claims, sourced from the drug's INDICATION list.
#
# Schema migration (#1607, verified by live introspection 2026-08-14). The prior
# document was broken against the live API and returned HTTP 400 on EVERY call,
# so `query_drug_disease_edges` could never yield an edge. Two independent
# upstream changes:
#
#   1. ``ClinicalIndicationFromDrug.maxPhaseForIndication`` (Int 0-4) was
#      renamed to ``maxClinicalStage`` (String: APPROVAL / PHASE_3 / ...).
#   2. The top-level ``evidences(drugIds:, diseaseIds:)`` Query field was
#      REMOVED. Evidence now hangs off ``Disease.evidences``, which REQUIRES a
#      gene ``ensemblIds`` argument and therefore cannot serve a drug->disease
#      lookup at all.
#
# Nothing detected this because every Open Targets unit test mocks the
# transport. `tests/integration/test_kg/test_kg_layer2_live_contracts.py` now
# executes this document against the real endpoint.
#
# Consequence of (2): edges no longer carry literature PMIDs or per-row scores.
# That is not a regression against a working baseline — the path that supplied
# them had been failing outright. In exchange, ``maxClinicalStage`` supplies the
# phase signal needed to gate the ``treats`` predicate (see
# ``KnowledgeGraphQuerier.query_drug_disease_edges``).
_DRUG_DISEASE_QUERY = """
query DrugDiseaseEvidence($drugId: String!) {
  drug(chemblId: $drugId) {
    id
    name
    maximumClinicalStage
    indications {
      count
      rows {
        disease {
          id
          name
        }
        maxClinicalStage
      }
    }
  }
}
"""

_DRUG_SEARCH_QUERY = """
query DrugSearch($name: String!) {
  search(queryString: $name, entityNames: ["drug"]) {
    hits {
      id
      name
      entity
    }
  }
}
"""

_DISEASE_SEARCH_QUERY = """
query DiseaseSearch($name: String!) {
  search(queryString: $name, entityNames: ["disease"]) {
    hits {
      id
      name
      entity
    }
  }
}
"""


class OpenTargetsError(Exception):
    """Open Targets request failed (transport or GraphQL-level)."""


class OpenTargetsClient:
    """Synchronous Open Targets GraphQL client."""

    def __init__(
        self,
        *,
        endpoint: str = OPEN_TARGETS_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._endpoint = endpoint
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __enter__(self) -> "OpenTargetsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def query_raw(
        self,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Execute a GraphQL query and return ``data``; raise on errors."""
        try:
            response = self._client.post(
                self._endpoint,
                json={"query": query, "variables": variables or {}},
            )
        except httpx.HTTPError as exc:
            raise OpenTargetsError(f"Open Targets transport error: {exc}") from exc
        if response.status_code >= 400:
            raise OpenTargetsError(
                f"Open Targets HTTP {response.status_code}: {response.text[:200]!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OpenTargetsError(f"Open Targets non-JSON body: {response.text[:200]!r}") from exc
        errors = payload.get("errors")
        if errors:
            raise OpenTargetsError(f"Open Targets GraphQL errors: {errors}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise OpenTargetsError(f"Open Targets payload missing 'data': {payload!r}")
        return data

    def drug_disease_evidence(
        self,
        drug_chembl_id: str,
        disease_efo_id: str,
        *,
        size: int = 25,
    ) -> dict[str, Any]:
        """Return drug → disease evidence + indication phase info.

        The output dict has two top-level keys:
            - ``drug``: drug record with declared indications and phase
            - ``evidences``: list of evidence rows with literature PMIDs
        """
        return _drug_disease_cached(
            self, drug_chembl_id=drug_chembl_id, disease_efo_id=disease_efo_id, size=size
        )

    def _drug_disease_uncached(
        self,
        *,
        drug_chembl_id: str,
        disease_efo_id: str,
        size: int,
    ) -> dict[str, Any]:
        # ``disease_efo_id`` / ``size`` are no longer query variables — the
        # indication list is returned whole for the drug and filtered by the
        # caller (see the schema-migration note on ``_DRUG_DISEASE_QUERY``).
        # They remain in the signature because they are part of this method's
        # public contract and its lru_cache key.
        return self.query_raw(_DRUG_DISEASE_QUERY, {"drugId": drug_chembl_id})

    def search_drug(self, name: str) -> Optional[str]:
        """Return the top ChEMBL ID for a drug name, or None."""
        return _search_drug_cached(self, name)

    def _search_drug_uncached(self, name: str) -> Optional[str]:
        if not name:
            return None
        data = self.query_raw(_DRUG_SEARCH_QUERY, {"name": name})
        hits = data.get("search", {}).get("hits") or []
        for hit in hits:
            hit_id = hit.get("id")
            if hit.get("entity") == "drug" and isinstance(hit_id, str):
                return hit_id
        return None

    def search_disease(self, name: str) -> Optional[str]:
        """Return the top EFO/MONDO ID for a disease name, or None."""
        return _search_disease_cached(self, name)

    def _search_disease_uncached(self, name: str) -> Optional[str]:
        if not name:
            return None
        data = self.query_raw(_DISEASE_SEARCH_QUERY, {"name": name})
        hits = data.get("search", {}).get("hits") or []
        for hit in hits:
            hit_id = hit.get("id")
            if hit.get("entity") == "disease" and isinstance(hit_id, str):
                return hit_id
        return None


@lru_cache(maxsize=_LRU_MAXSIZE)
def _drug_disease_cached(
    client: OpenTargetsClient,
    *,
    drug_chembl_id: str,
    disease_efo_id: str,
    size: int,
) -> dict[str, Any]:
    return client._drug_disease_uncached(
        drug_chembl_id=drug_chembl_id, disease_efo_id=disease_efo_id, size=size
    )


@lru_cache(maxsize=_LRU_MAXSIZE)
def _search_drug_cached(client: OpenTargetsClient, name: str) -> Optional[str]:
    return client._search_drug_uncached(name)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _search_disease_cached(client: OpenTargetsClient, name: str) -> Optional[str]:
    return client._search_disease_uncached(name)


def reset_caches() -> None:
    """Clear Open Targets caches (useful in tests)."""
    _drug_disease_cached.cache_clear()
    _search_drug_cached.cache_clear()
    _search_disease_cached.cache_clear()
