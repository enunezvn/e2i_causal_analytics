"""clinical_context (insights) — digit-free clinical grounding for insight surfaces.

Why this exists (2026-07-12): the /ai-insights review found the HTE strategic
insight and the Executive Brief consumed NO clinical context even though the
platform integrates OpenFDA / ChEMBL / ClinicalTrials.gov / PubMed
(src/services/clinical_context). Commercial outputs do not take place in a
clinical vacuum — the user asked for the clinical setting (mechanism,
indicated population, label constraints, competitors) to ground both surfaces.

The formatter output must be DIGIT-FREE BY CONSTRUCTION: the executive brief's
placeholder guard rejects ANY numeric character in LM output, and the HTE
guard rejects any numeric claim its grounding cannot vouch for. A
digit-bearing fragment (pivotal endpoints like "UAS7 at week 12", label
populations like "12 years and older") is DROPPED, never paraphrased.
"""

from unittest.mock import MagicMock, patch

from src.insights.clinical_context import (
    fetch_clinical_payload,
    format_clinical_context,
)


def _payload(**overrides):
    """Mirror of ClinicalContextService.get_context()'s payload shape."""
    p = {
        "brand": "Remibrutinib",
        "drug_name": "remibrutinib",
        "disease": "chronic spontaneous urticaria",
        "our_outcome": "persistent_180d",
        "mapped_endpoint": "UAS7 change from baseline",
        "mechanism": {
            "mechanism_of_action": "Bruton tyrosine kinase (BTK) inhibitor",
            "source": "chembl",
        },
        "pivotal_endpoints": {
            "endpoints": ["UAS7 change at week 12"],
            "source": "clinicaltrials.gov",
        },
        "real_world_evidence": None,
        "seminal_real_world_evidence": None,
        "approved_indications": {
            "indications": [
                "treatment of chronic spontaneous urticaria in adults and "
                "pediatric patients 12 years of age and older"
            ],
            "limitations_of_use": None,
            "boxed_warning": None,
            "source": "openfda",
        },
        "competitor_landscape": {
            "competitors": ["Xolair (omalizumab)", "second-generation antihistamines"],
            "count": 2,
            "source": "curated",
        },
        "honesty_label": "label",
    }
    p.update(overrides)
    return p


def test_format_is_digit_free_and_names_mechanism_disease_competitors():
    text = format_clinical_context(_payload())
    assert "BTK" in text
    assert "chronic spontaneous urticaria" in text
    assert "omalizumab" in text
    assert not any(ch.isnumeric() for ch in text), text


def test_digit_bearing_indication_text_is_dropped_not_paraphrased():
    # The OpenFDA indication carries "12 years of age" — it must not appear;
    # the digit-free disease name stands in for the indicated population.
    text = format_clinical_context(_payload())
    assert "12 years" not in text
    assert "UAS7" not in text  # digit-bearing endpoint names never leak


def test_boxed_warning_stated_only_when_present():
    without = format_clinical_context(_payload())
    assert "boxed warning" not in without.lower()

    with_warning = format_clinical_context(
        _payload(
            approved_indications={
                "indications": [],
                "limitations_of_use": None,
                "boxed_warning": "WARNING: SERIOUS INFECTIONS",
                "source": "openfda",
            }
        )
    )
    assert "boxed warning" in with_warning.lower()
    # The warning BODY may carry digits/case obligations — only its presence
    # is stated, never its text.
    assert "SERIOUS INFECTIONS" not in with_warning


def test_sources_are_labeled_qualitatively():
    text = format_clinical_context(_payload())
    # Provenance must ride along (public biomedical/regulatory sources vs
    # curated competitor list) without figures.
    assert "curated" in text.lower()
    assert not any(ch.isnumeric() for ch in text)


def test_none_or_empty_payload_formats_empty():
    assert format_clinical_context(None) == ""
    assert format_clinical_context({}) == ""


def test_adversarial_all_digit_payload_formats_empty():
    # Belt and braces: if EVERY candidate fragment carries digits, nothing may
    # survive — an empty string, not a digit-bearing one.
    text = format_clinical_context(
        _payload(
            drug_name="drug-2000",
            disease="type 2 diabetes",
            mechanism={"mechanism_of_action": "IL-17 antagonist", "source": "chembl"},
            approved_indications={
                "indications": ["patients 12 years and older"],
                "limitations_of_use": None,
                "boxed_warning": None,
                "source": "openfda",
            },
            competitor_landscape={
                "competitors": ["GLP-1 agonists"],
                "count": 1,
                "source": "curated",
            },
        )
    )
    assert text == ""
    assert not any(ch.isnumeric() for ch in text)


async def test_fetch_returns_payload_from_service():
    payload = _payload()
    service = MagicMock()
    service.get_context.return_value = payload
    with patch(
        "src.services.clinical_context.service.ClinicalContextService", return_value=service
    ):
        got = await fetch_clinical_payload("Remibrutinib", "persistent_180d")
    assert got == payload
    service.get_context.assert_called_once_with("Remibrutinib", "persistent_180d")


async def test_fetch_swallows_unknown_brand_and_errors_returns_none():
    service = MagicMock()
    service.get_context.side_effect = KeyError("unknown brand")
    with patch(
        "src.services.clinical_context.service.ClinicalContextService", return_value=service
    ):
        assert await fetch_clinical_payload("All", "TRx") is None

    service.get_context.side_effect = RuntimeError("fan-out exploded")
    with patch(
        "src.services.clinical_context.service.ClinicalContextService", return_value=service
    ):
        assert await fetch_clinical_payload("Kisqali", "TRx") is None


async def test_fetch_without_brand_short_circuits_to_none():
    # No brand -> no service construction, no network.
    with patch("src.services.clinical_context.service.ClinicalContextService") as ctor:
        assert await fetch_clinical_payload(None, "TRx") is None
        assert await fetch_clinical_payload("", "TRx") is None
        ctor.assert_not_called()
