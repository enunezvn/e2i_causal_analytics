"""ChEMBL public REST client (#245).

ChEMBL (https://www.ebi.ac.uk/chembl/api/data/docs) is the EMBL-EBI
manually-curated database of bioactive molecules with drug-like
properties. v34 is current; the public REST endpoint is unauthenticated
but rate-limited (HTTP 429 on burst). Surface used by Layer-2 evidence
enrichment:

  - ``compound_search(name)``       — drug name → ChEMBL molecule ID
  - ``target_search(gene_symbol)``  — gene symbol → ChEMBL target ID
  - ``get_bioactivity(target)``     — list[Activity] (IC50/Ki canonical)
  - ``open_targets_target_to_chembl(gene)`` — Open Targets cross-walk
    helper; v1 thin wrapper around ``target_search`` that accepts
    ``None``/empty input without raising.

Design mirrors ``src/data/kg/open_targets.py`` and
``src/data/kg/umls_uts.py`` exactly:
    - Synchronous httpx-backed client; callable as context manager.
    - Module-level ``lru_cache`` wrappers keyed on ``(id(client), args)``
      so the cache scopes to the client's lifetime and tests can
      reset via ``reset_caches()``.
    - Distinct ``ChEMBLError`` exception type; transport/HTTP/JSON
      failures all surface as this one class.

Retry policy:
    HTTP 429 (Too Many Requests) is the documented ChEMBL rate-limit
    signal. The client retries up to ``DEFAULT_MAX_RETRIES`` times with
    fixed ``_retry_backoff_s`` seconds between attempts (no exponential
    backoff in v1 — keep simple, deterministic for tests). 5xx and other
    4xx errors raise immediately.

Cache namespace:
    ``CACHE_NAMESPACE = "chembl"`` parallel to the ``opentargets`` and
    ``umls`` namespaces. v1's in-process LRU does not lay files on disk
    yet; the namespace is exposed for the future disk-backed variant
    that mirrors ``cache.py``'s per-namespace directory layout.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 0.5
_LRU_MAXSIZE = 2048

# Cache namespace identifier — parallel to the ``opentargets`` / ``umls``
# namespaces. The v1 LRU is in-process; the constant is exposed so the
# future disk-backed variant can mirror ``cache.py``'s layout.
CACHE_NAMESPACE = "chembl"

# Canonical bioactivity standard types we surface in v1. ChEMBL's
# activity table carries dozens of standard_type values (IC50, Ki, EC50,
# Kd, GI50, AC50, ...). For Layer-2 evidence enrichment we only need the
# binding/inhibitory potency canonicals, so we filter at the client
# boundary to keep evidence payloads bounded and consistent.
_DEFAULT_ACTIVITY_TYPES: frozenset[str] = frozenset({"IC50", "Ki"})


class ChEMBLError(Exception):
    """ChEMBL request failed (transport, HTTP, JSON, or rate-limit)."""


@dataclass(frozen=True)
class Activity:
    """One bioactivity row from ChEMBL ``/activity.json``.

    Attributes:
        activity_id: ChEMBL's internal activity record ID.
        molecule_chembl_id: The compound's ChEMBL molecule ID.
        target_chembl_id: The target's ChEMBL target ID.
        standard_type: One of ``"IC50"``, ``"Ki"`` (v1 surface set).
        standard_value: Numeric potency value coerced to ``float``.
            ``None`` when the source string was missing or non-numeric.
        standard_units: Reported units (typically ``"nM"``); preserved
            verbatim from ChEMBL.
        pchembl_value: ChEMBL's normalized -log10 potency (the comparable
            metric across assays). ``None`` when ChEMBL did not compute
            one for this activity.
        pubmed_id: PMID of the source publication, when ChEMBL records
            one. ``None`` otherwise.
    """

    activity_id: int
    molecule_chembl_id: str
    target_chembl_id: str
    standard_type: str
    standard_value: Optional[float]
    standard_units: Optional[str]
    pchembl_value: Optional[float] = None
    pubmed_id: Optional[str] = None


@dataclass(frozen=True)
class Mechanism:
    """One mechanism-of-action row from ChEMBL ``/mechanism.json``.

    Attributes:
        mechanism_of_action: Free-text MoA string (e.g.
            ``"Cyclin-dependent kinase 4 inhibitor"``).
        action_type: ChEMBL action type (e.g. ``"INHIBITOR"``); ``None`` when
            absent.
        target_chembl_id: ChEMBL target ID the drug acts on; ``None`` when absent.
    """

    mechanism_of_action: str
    action_type: Optional[str] = None
    target_chembl_id: Optional[str] = None


class ChEMBLClient:
    """Synchronous ChEMBL REST client.

    Constructed once and reused; httpx.Client connection-pools internally.
    Methods raise ``ChEMBLError`` on transport/HTTP/JSON failure; success
    paths return plain Python values (str, list[Activity], etc.).
    """

    def __init__(
        self,
        *,
        base: str = CHEMBL_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        client: Optional[httpx.Client] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
    ) -> None:
        self._base = base.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._max_retries = max_retries
        # Exposed as a mutable attr so tests can zero the backoff without
        # patching ``time.sleep`` globally. Production callers should
        # construct via the ``retry_backoff_s`` kwarg.
        self._retry_backoff_s = retry_backoff_s

    def __enter__(self) -> "ChEMBLClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """GET ``{base}{path}?{params}`` with HTTP-429 retry.

        Returns the parsed JSON body. Raises ``ChEMBLError`` on transport
        failure, non-200 response (after retry exhaustion for 429), or
        non-JSON body. Other 4xx/5xx responses raise immediately without
        retry.
        """
        url = f"{self._base}{path}"
        attempts = 0
        last_error: Optional[str] = None
        while attempts <= self._max_retries:
            try:
                response = self._client.get(
                    url,
                    params=params,
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                raise ChEMBLError(f"ChEMBL transport error: {exc}") from exc
            if response.status_code == 429:
                attempts += 1
                last_error = f"HTTP 429: {response.text[:200]!r}"
                if attempts > self._max_retries:
                    break
                if self._retry_backoff_s > 0:
                    time.sleep(self._retry_backoff_s)
                continue
            if response.status_code >= 400:
                raise ChEMBLError(f"ChEMBL HTTP {response.status_code}: {response.text[:200]!r}")
            try:
                payload: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ChEMBLError(f"ChEMBL non-JSON body: {response.text[:200]!r}") from exc
            return payload
        # Retry budget exhausted on 429.
        raise ChEMBLError(
            f"ChEMBL rate-limit exhausted after {self._max_retries} retries: {last_error}"
        )

    # ------------------------------------------------------------------
    # Compound + target search
    # ------------------------------------------------------------------

    def compound_search(self, name: str) -> Optional[str]:
        """Return the top ChEMBL molecule ID for a drug name, or None.

        Empty/blank names skip the network round-trip and return None.
        """
        if not name:
            return None
        return _compound_search_cached(self, name)

    def _compound_search_uncached(self, name: str) -> Optional[str]:
        # ChEMBL accepts a synonym-iexact filter that matches case-insensitively
        # against any of the drug's recorded synonyms (canonical name,
        # trade names, INN). This is the simplest reliable name-resolver.
        payload = self._get(
            "/molecule.json",
            {
                "molecule_synonyms__molecule_synonym__iexact": name,
                "limit": 1,
            },
        )
        molecules = payload.get("molecules") or []
        for mol in molecules:
            mol_id = mol.get("molecule_chembl_id")
            if isinstance(mol_id, str) and mol_id:
                return mol_id
        return None

    def target_search(self, gene_symbol: str) -> Optional[str]:
        """Return the top ChEMBL target ID for a gene symbol, or None.

        Resolves via the denormalized top-level ``target_synonym__iexact``
        filter (case-insensitive exact match). Empty input skips the
        network. See ``_target_search_uncached`` for the precision
        trade-off note.
        """
        if not gene_symbol:
            return None
        return _target_search_cached(self, gene_symbol)

    def _target_search_uncached(self, gene_symbol: str) -> Optional[str]:
        # Use ChEMBL's denormalized top-level synonym filter. The nested
        # path ``target_components__component_synonyms__component_synonym__iexact``
        # we tried initially is rejected by the live API with HTTP 400
        # ("path 'component_synonyms' is not valid in the filter
        # expression"); on the JSON response the field is actually named
        # ``target_component_synonyms`` (with the ``target_`` prefix), but
        # ChEMBL exposes the simpler denormalized filter ``target_synonym``
        # for query use. Verified live: GET
        # /target.json?target_synonym__iexact=ABL1 returns 14 hits
        # including CHEMBL1862 (Tyrosine-protein kinase ABL1).
        #
        # Precision trade-off (acceptable for v1): ``target_synonym``
        # matches across all synonym types — UniProt accession, EC number,
        # gene symbol, and protein names. Future polish: narrow by
        # constraining ``syn_type`` (e.g., GENE_SYMBOL) once the worker
        # surfaces ambiguous-resolution telemetry. For now ``limit=1``
        # picks the canonical entry which empirically returns the
        # gene-symbol-keyed target for the workloads in scope.
        payload = self._get(
            "/target.json",
            {
                "target_synonym__iexact": gene_symbol,
                "limit": 1,
            },
        )
        targets = payload.get("targets") or []
        for tgt in targets:
            tgt_id = tgt.get("target_chembl_id")
            if isinstance(tgt_id, str) and tgt_id:
                return tgt_id
        return None

    def open_targets_target_to_chembl(self, gene_or_id: Optional[str]) -> Optional[str]:
        """Cross-walk an Open Targets target reference → ChEMBL target ID.

        Open Targets surfaces a target's gene symbol on the evidence row
        via ``target.approvedSymbol``. This helper:
          1. Returns None on None or empty input (no HTTP).
          2. Otherwise delegates to ``target_search``.

        Kept as a separate method so ``KnowledgeGraphQuerier`` has a
        named bridge point — future refinements (UniProt accession,
        Ensembl gene ID, ``chemicalProbes`` field) can be added here
        without changing the Querier call site.
        """
        if not gene_or_id:
            return None
        return self.target_search(gene_or_id)

    # ------------------------------------------------------------------
    # Bioactivity
    # ------------------------------------------------------------------

    def get_bioactivity(
        self,
        target_chembl_id: str,
        *,
        standard_types: Optional[frozenset[str]] = None,
        limit: int = 100,
    ) -> list[Activity]:
        """Return ``Activity`` rows for a ChEMBL target ID.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., ``"CHEMBL1862"``).
            standard_types: Optional frozenset of standard_type strings
                to include. Defaults to ``{"IC50", "Ki"}``; pass a
                broader frozenset to surface EC50/Kd/etc.
            limit: ChEMBL ``limit`` query param (default 100).

        Empty target ID returns an empty list with no HTTP call.
        """
        if not target_chembl_id:
            return []
        return _get_bioactivity_cached(
            self,
            target_chembl_id=target_chembl_id,
            standard_types=standard_types or _DEFAULT_ACTIVITY_TYPES,
            limit=limit,
        )

    def _get_bioactivity_uncached(
        self,
        *,
        target_chembl_id: str,
        standard_types: frozenset[str],
        limit: int,
    ) -> list[Activity]:
        payload = self._get(
            "/activity.json",
            {
                "target_chembl_id": target_chembl_id,
                "limit": limit,
            },
        )
        rows = payload.get("activities") or []
        out: list[Activity] = []
        for row in rows:
            std_type = row.get("standard_type")
            if not isinstance(std_type, str) or std_type not in standard_types:
                continue
            out.append(_row_to_activity(row))
        return out

    # ------------------------------------------------------------------
    # Mechanism of action (#causal-enrichment)
    # ------------------------------------------------------------------

    def get_mechanism(self, molecule_chembl_id: str) -> "list[Mechanism]":
        """Return ``Mechanism`` rows for a ChEMBL molecule ID.

        Empty molecule ID returns an empty list with no HTTP call. The
        ``/mechanism.json`` endpoint is the canonical drug-action surface (one
        row per molecular target the drug modulates).
        """
        if not molecule_chembl_id:
            return []
        return _get_mechanism_cached(self, molecule_chembl_id)

    def _get_mechanism_uncached(self, molecule_chembl_id: str) -> "list[Mechanism]":
        payload = self._get(
            "/mechanism.json",
            {"molecule_chembl_id": molecule_chembl_id, "limit": 20},
        )
        rows = payload.get("mechanisms") or []
        out: list[Mechanism] = []
        for row in rows:
            moa = row.get("mechanism_of_action")
            if not isinstance(moa, str) or not moa:
                continue
            action = row.get("action_type")
            target = row.get("target_chembl_id")
            out.append(
                Mechanism(
                    mechanism_of_action=moa,
                    action_type=action if isinstance(action, str) and action else None,
                    target_chembl_id=target if isinstance(target, str) and target else None,
                )
            )
        return out

    def mechanism_of_action(self, drug_name: str) -> Optional[str]:
        """Resolve a drug name → its first ChEMBL mechanism-of-action string.

        Convenience wrapper: ``compound_search`` (name → molecule id) then
        ``get_mechanism`` (id → MoA rows), returning the first MoA. Returns
        ``None`` when the name does not resolve or the molecule has no recorded
        mechanism. Empty name skips the network.
        """
        if not drug_name:
            return None
        molecule_id = self.compound_search(drug_name)
        if not molecule_id:
            return None
        mechs = self.get_mechanism(molecule_id)
        return mechs[0].mechanism_of_action if mechs else None


def _row_to_activity(row: dict[str, Any]) -> Activity:
    """Coerce a raw ChEMBL activity row into the typed ``Activity``."""

    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value if value else None
        return str(value)

    activity_id_raw = row.get("activity_id")
    try:
        activity_id = int(activity_id_raw) if activity_id_raw is not None else 0
    except (TypeError, ValueError):
        activity_id = 0
    return Activity(
        activity_id=activity_id,
        molecule_chembl_id=str(row.get("molecule_chembl_id") or ""),
        target_chembl_id=str(row.get("target_chembl_id") or ""),
        standard_type=str(row.get("standard_type") or ""),
        standard_value=_as_float(row.get("standard_value")),
        standard_units=_as_str(row.get("standard_units")),
        pchembl_value=_as_float(row.get("pchembl_value")),
        pubmed_id=_as_str(row.get("pubmed_id")),
    )


# ---------------------------------------------------------------------------
# Module-level LRU wrappers — keyed on (id(client), args) so cache scopes
# to the client's lifetime (same pattern as ``open_targets.py``).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=_LRU_MAXSIZE)
def _compound_search_cached(client: ChEMBLClient, name: str) -> Optional[str]:
    return client._compound_search_uncached(name)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _target_search_cached(client: ChEMBLClient, gene_symbol: str) -> Optional[str]:
    return client._target_search_uncached(gene_symbol)


@lru_cache(maxsize=_LRU_MAXSIZE)
def _get_bioactivity_cached(
    client: ChEMBLClient,
    *,
    target_chembl_id: str,
    standard_types: frozenset[str],
    limit: int,
) -> list[Activity]:
    return client._get_bioactivity_uncached(
        target_chembl_id=target_chembl_id,
        standard_types=standard_types,
        limit=limit,
    )


@lru_cache(maxsize=_LRU_MAXSIZE)
def _get_mechanism_cached(client: ChEMBLClient, molecule_chembl_id: str) -> list[Mechanism]:
    return client._get_mechanism_uncached(molecule_chembl_id)


def reset_caches() -> None:
    """Clear ChEMBL in-process caches (useful in tests)."""
    _compound_search_cached.cache_clear()
    _target_search_cached.cache_clear()
    _get_bioactivity_cached.cache_clear()
    _get_mechanism_cached.cache_clear()
