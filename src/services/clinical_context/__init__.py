"""Clinical Context enrichment — a brand-faithful, sourced NARRATIVE layer over
each discovered causal effect (drug + mechanism of action, the disease's real
pivotal endpoints, a real-world-evidence citation).

This package NEVER touches the causal math, adjustment sets, or estimation
frames. It calls the PUBLIC biomedical REST APIs directly (ChEMBL, ClinicalTrials
.gov v2, PubMed E-utilities — the claude.ai MCP tools are agent-only) best-effort,
degrades gracefully to static fallbacks when an API is down/slow, caches per
(brand, disease), and labels every payload: estimate = synthetic cohort;
clinical context = real, cited.

Extensible by design: add a ``ClinicalContextProvider`` subclass to enrich with a
new source (the deferred openFDA / UMLS tasks slot in here) without changing the
service or endpoint.
"""

from src.services.clinical_context.brand_map import (
    BRAND_CLINICAL_MAP,
    BrandClinicalProfile,
    endpoint_mapping_for_outcome,
    resolve_brand_profile,
)

__all__ = [
    "BRAND_CLINICAL_MAP",
    "BrandClinicalProfile",
    "endpoint_mapping_for_outcome",
    "resolve_brand_profile",
]
