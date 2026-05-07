"""EntityLinker — Phase 2.1 of the adaptive temporal-validity redesign.

Single public surface that the rest of Layer 2 consumes. Given a code in any
of the supported source vocabularies (ICD-10, RxCUI, LOINC, CPT, HCPCS) or a
free-text drug name, returns an ``EntityLink`` with the resolved UMLS concept.

Design:
    EntityLinker composes three clients (``UMLSClient``, ``OpenTargetsClient``,
    ``RxNavClient``) but the only one EntityLinker itself uses for resolution
    is UMLS. Open Targets and RxNav are exposed as attributes so that
    downstream consumers (CausalRoleClassifier, CitationResolver) can borrow
    the same connection-pooled client instances.

Resolution strategy:
    - ``resolve_icd10`` / ``resolve_loinc`` / ``resolve_cpt`` / ``resolve_hcpcs``
      — single ``code_to_cui`` call against the named UTS source.
    - ``resolve_rxcui`` — same, source=``RXNORM``.
    - ``resolve_drug_name`` — first normalize through RxNav to RxCUI, then
      cross-walk the RxCUI to a UMLS CUI. Falls back to UMLS free-text search
      if RxNav has no match.

Why a class, not free functions:
    The wrapper clients hold ``httpx.Client`` instances; instantiating a fresh
    pool per call defeats keep-alive. EntityLinker keeps the pool alive for
    the lifetime of the linker.

Errors:
    Resolution methods do NOT raise on missing concepts — they return an
    ``EntityLink`` with ``concept=None``. They only raise ``EntityLinkerError``
    when the underlying UMLS auth fails or transport breaks. The "auth"
    distinction matters: a 401 from UMLS means the linker is broken for ALL
    codes, while a 404 just means this one code isn't in UMLS.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.data.kg.open_targets import OpenTargetsClient
from src.data.kg.rxnav import RxNavClient
from src.data.kg.types import CodeSystem, EntityLink, KGConcept
from src.data.kg.umls_uts import (
    UMLSAuthError,
    UMLSClient,
    UMLSError,
    UMLSNotFoundError,
)

logger = logging.getLogger(__name__)


class EntityLinkerError(Exception):
    """Raised on linker-fatal failures (e.g., UMLS auth dead).

    Per-code resolution failures do NOT raise — they return an EntityLink
    with ``error`` populated and ``concept=None``.
    """


_UTS_SOURCE_BY_SYSTEM: dict[CodeSystem, str] = {
    "ICD10CM": "ICD10CM",
    "ICD10": "ICD10",
    "RXNORM": "RXNORM",
    "LOINC": "LNC",  # UTS uses LNC, not LOINC, as the source abbreviation
    "CPT": "CPT",
    "HCPCS": "HCPCS",
    "SNOMEDCT_US": "SNOMEDCT_US",
    "MESH": "MSH",
}


class EntityLinker:
    """Compose UMLS + RxNav + Open Targets clients into a single resolution API.

    Args:
        umls: UMLS client. Constructed from ``UMLS_UTS_API_KEY`` env if None.
        rxnav: RxNav client. Constructed with defaults if None.
        open_targets: Open Targets client. Constructed with defaults if None.
    """

    def __init__(
        self,
        *,
        umls: Optional[UMLSClient] = None,
        rxnav: Optional[RxNavClient] = None,
        open_targets: Optional[OpenTargetsClient] = None,
    ) -> None:
        self.umls = umls if umls is not None else UMLSClient()
        self.rxnav = rxnav if rxnav is not None else RxNavClient()
        self.open_targets = open_targets if open_targets is not None else OpenTargetsClient()

    def __enter__(self) -> "EntityLinker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.umls.close()
        self.rxnav.close()
        self.open_targets.close()

    def resolve_icd10(self, code: str) -> EntityLink:
        return self._resolve_via_uts(code, "ICD10CM")

    def resolve_loinc(self, code: str) -> EntityLink:
        return self._resolve_via_uts(code, "LOINC")

    def resolve_cpt(self, code: str) -> EntityLink:
        return self._resolve_via_uts(code, "CPT")

    def resolve_hcpcs(self, code: str) -> EntityLink:
        return self._resolve_via_uts(code, "HCPCS")

    def resolve_rxcui(self, rxcui: str) -> EntityLink:
        return self._resolve_via_uts(rxcui, "RXNORM")

    def resolve(self, code: str, system: CodeSystem) -> EntityLink:
        """Generic dispatch by system."""
        return self._resolve_via_uts(code, system)

    def resolve_drug_name(self, name: str) -> EntityLink:
        """Resolve a free-text drug name to a UMLS concept.

        Two-step: RxNav normalizes name → RxCUI; UMLS cross-walks RxCUI →
        CUI. If RxNav has no match, falls back to UMLS search.
        """
        if not name:
            return EntityLink(input_code="", input_system="RXNORM", error="empty name")
        try:
            rxcui = self.rxnav.rxcui_for_name(name)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            logger.warning("RxNav lookup failed for %r: %s", name, exc)
            rxcui = None
        if rxcui:
            link = self.resolve_rxcui(rxcui)
            if link.resolved:
                return link
        return self._resolve_via_search(name)

    def _resolve_via_uts(self, code: str, system: CodeSystem) -> EntityLink:
        if not code:
            return EntityLink(input_code=code, input_system=system, error="empty code")
        source = _UTS_SOURCE_BY_SYSTEM.get(system)
        if source is None:
            return EntityLink(
                input_code=code,
                input_system=system,
                error=f"unsupported system: {system}",
            )
        try:
            cui = self.umls.code_to_cui(code, source=source)
        except UMLSAuthError as exc:
            raise EntityLinkerError(f"UMLS auth failed: {exc}") from exc
        except UMLSNotFoundError:
            return EntityLink(input_code=code, input_system=system)
        except UMLSError as exc:
            return EntityLink(input_code=code, input_system=system, error=str(exc))
        if not cui:
            return EntityLink(input_code=code, input_system=system)
        return self._link_from_cui(code=code, system=system, cui=cui, sources=(source,))

    def _resolve_via_search(self, name: str) -> EntityLink:
        try:
            results = self.umls.search(name, page_size=1, search_type="words")
        except UMLSAuthError as exc:
            raise EntityLinkerError(f"UMLS auth failed: {exc}") from exc
        except UMLSError as exc:
            return EntityLink(input_code=name, input_system="RXNORM", error=str(exc))
        if not results:
            return EntityLink(input_code=name, input_system="RXNORM")
        first = results[0]
        cui = first.get("ui")
        if not isinstance(cui, str) or cui == "NONE":
            return EntityLink(input_code=name, input_system="RXNORM")
        root_source = first.get("rootSource")
        sources: tuple[str, ...] = (root_source,) if isinstance(root_source, str) else ()
        return self._link_from_cui(code=name, system="RXNORM", cui=cui, sources=sources)

    def _link_from_cui(
        self,
        *,
        code: str,
        system: CodeSystem,
        cui: str,
        sources: tuple[str, ...],
    ) -> EntityLink:
        try:
            concept: Optional[KGConcept] = self.umls.cui_lookup(cui)
        except UMLSAuthError as exc:
            raise EntityLinkerError(f"UMLS auth failed: {exc}") from exc
        except UMLSError as exc:
            # CUI was found upstream but lookup failed; still return the CUI.
            logger.warning("CUI lookup failed for %s: %s", cui, exc)
            concept = KGConcept(cui=cui, preferred_name="")
        return EntityLink(
            input_code=code,
            input_system=system,
            concept=concept,
            sources=sources,
        )
