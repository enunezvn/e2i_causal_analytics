"""Lane E — the mart comorbidity flags carry KG entity codes.

Cheapest disproof of the spec's item 1 (2026-09-22, evidence
``docs/demos/results/2026-09-22_lane_e_feature_role_voters/README.md``): before
this lane ``OPTUM_MART_FEATURES`` had ZERO entity-bearing features, and
``scripts/build_kg_cache.py`` only emits a record for a feature with
``kg_entity_codes`` — so a ``--live`` cache for ``optum_mart`` would have been an
empty file and Layer 2 would have stayed dark on the causal cohort while looking
merely quiet. The 45 Charlson/Elixhauser flags ARE disease concepts; each now
carries the (ICD10CM, UMLS) codes verified live against UTS on 2026-09-22
(``umls_code_verification.txt`` in the evidence dir: one candidate CUI was a 404
and two resolved to the wrong concept; those were dropped, not committed).
"""

from __future__ import annotations

import pytest

from src.data.feature_contract import FeatureContract
from src.data.manifests.optum_mart_feature_manifest import (
    MART_COMORBIDITY_KG_CODES,
    MART_SAFE_FEATURES,
    OPTUM_MART_FEATURES,
)

_BY_NAME = {fc.name: fc for fc in OPTUM_MART_FEATURES}
_FLAGS = sorted(n for n in MART_SAFE_FEATURES if n.startswith(("cci_", "elx_")))


def test_every_comorbidity_flag_carries_entity_codes() -> None:
    assert len(_FLAGS) == 48, _FLAGS  # 17 Charlson + 31 Elixhauser
    missing = [n for n in _FLAGS if not _BY_NAME[n].kg_entity_codes]
    assert not missing, f"comorbidity flags without kg_entity_codes: {missing}"


def test_codes_are_the_mapping_and_the_mapping_is_the_codes() -> None:
    """The dict is the single authoring surface; the contracts reference it."""
    assert set(MART_COMORBIDITY_KG_CODES) == set(_FLAGS)
    for name in _FLAGS:
        assert _BY_NAME[name].kg_entity_codes == tuple(MART_COMORBIDITY_KG_CODES[name])


@pytest.mark.parametrize("name", _FLAGS)
def test_each_flag_has_a_umls_cui_or_an_icd10cm_code(name: str) -> None:
    systems = {system for system, _code in _BY_NAME[name].kg_entity_codes}
    assert systems <= {"ICD10CM", "UMLS"}, systems
    assert systems, name


def test_the_dropped_candidates_never_returned() -> None:
    """The three live-disproved codes must stay out (UTS 404 / wrong concept)."""
    all_codes = {code for fc in OPTUM_MART_FEATURES for code in fc.kg_entity_codes}
    assert ("UMLS", "C0042990") not in all_codes  # UTS 404
    assert ("UMLS", "C0522224") not in all_codes  # 'Paralysed' is a Finding, not the disease
    assert ("ICD10CM", "I38") not in all_codes  # crosswalks to 'Valvular regurgitation'


def test_non_disease_columns_stay_without_codes() -> None:
    """Scores, counts, bands and demographics are not disease concepts; an
    honest ``no record`` is right for them (they are not KG-answerable)."""
    for name in (
        "charlson_score",
        "elixhauser_van_walraven_score",
        "age_at_index",
        "payer_category",
    ):
        assert not _BY_NAME[name].kg_entity_codes, name


def test_chronic_pulmonary_carries_both_asthma_and_copd() -> None:
    """Both Lane A drugs are approved for asthma and dupilumab for COPD; the
    Charlson/Elixhauser 'chronic pulmonary' bucket spans both, so the flag
    must name both concepts for the drug-disease pass to see them."""
    for name in ("cci_chronic_pulmonary", "elx_chronic_pulmonary"):
        codes = set(_BY_NAME[name].kg_entity_codes)
        assert ("UMLS", "C0004096") in codes, codes  # Asthma
        assert ("UMLS", "C0024117") in codes, codes  # Chronic Obstructive Airway Disease


def test_contracts_still_validate() -> None:
    """FeatureContract.__post_init__ rejects unknown systems / empty codes."""
    for fc in OPTUM_MART_FEATURES:
        assert isinstance(fc, FeatureContract)
