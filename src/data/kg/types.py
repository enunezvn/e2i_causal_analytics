"""Shared dataclasses for Layer 2 KG clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

CodeSystem = Literal[
    "ICD10CM",
    "ICD10",
    "RXNORM",
    "LOINC",
    "CPT",
    "HCPCS",
    "SNOMEDCT_US",
    "MESH",
]


@dataclass(frozen=True)
class KGConcept:
    """A single UMLS concept after cross-walk.

    Returned by both ``UMLSClient.cui_lookup`` and as the canonical payload
    inside ``EntityLink.concept``. ``semantic_types`` and ``atom_count`` come
    from the UTS ``content/CUI`` endpoint; ``preferred_name`` is the canonical
    English label.
    """

    cui: str
    preferred_name: str
    semantic_types: tuple[str, ...] = ()
    atom_count: Optional[int] = None


@dataclass(frozen=True)
class EntityLink:
    """The result of resolving a single code → UMLS concept.

    ``input_code`` and ``input_system`` are the caller's inputs. ``concept`` is
    None when no UMLS concept maps from the code; ``error`` captures why.
    ``sources`` lists which UTS source vocabularies the cross-walk traversed
    (helps audit "this CSU ICD-10 code resolved via SNOMEDCT_US"). The
    distinction between "no result" (``concept is None`` and ``error is None``)
    and "API error" (``error`` populated) lets the caller decide whether to
    retry or accept the absence.
    """

    input_code: str
    input_system: CodeSystem
    concept: Optional[KGConcept] = None
    sources: tuple[str, ...] = ()
    error: Optional[str] = None
    raw: Optional[dict] = field(default=None, repr=False)

    @property
    def resolved(self) -> bool:
        return self.concept is not None
