"""Validation tests for the literature-derived causal-role golden set.

Issue #358 — these tests assert structural integrity of the fixture so
schema drift is caught at PR time rather than silently breaking
downstream consumers (``scripts/measure_layer4_precision.py``,
unit-test parametrizations, future #240 evaluator-promotion work).

The tests do NOT exercise the DSPy classifier — that lives in
``scripts/measure_layer4_precision.py`` (offline-skippable) and any
LM-bearing unit test gated by ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from typing import Any

_REQUIRED_FIELDS = {
    "cohort",
    "dataset_context",
    "derivation_pseudocode",
    "feature_name",
    "ground_truth_role",
    "rationale",
    "provenance",
    "scenario",
    "treatment_explicit",
}
_VALID_ROLES = {"ancestor", "confounder", "instrument", "mediator", "collider", "descendant"}
_EXPECTED_COHORTS = {"CSU_remibrutinib", "PNH_fabhalta", "BC_kisqali"}
_MIN_ENTRIES_PER_COHORT = 30
_MIN_INSTRUMENTS_PER_COHORT = 6


def test_golden_set_loads(causal_role_golden_set: dict[str, Any]) -> None:
    """Fixture file exists, parses, and has the schema-version header."""
    assert "entries" in causal_role_golden_set
    assert "cohorts" in causal_role_golden_set
    assert "total_entries" in causal_role_golden_set
    assert causal_role_golden_set["total_entries"] == len(causal_role_golden_set["entries"])
    assert causal_role_golden_set["issue"] == "#358"
    assert causal_role_golden_set["fixture_kind"] == "literature_derived_golden_set"


def test_golden_set_covers_all_three_cohorts(causal_role_golden_set: dict[str, Any]) -> None:
    """Per #358 acceptance: must cover CSU + PNH + BreastCancer cohorts."""
    seen = {e["cohort"] for e in causal_role_golden_set["entries"]}
    assert _EXPECTED_COHORTS <= seen, f"Missing cohorts: {_EXPECTED_COHORTS - seen}"


def test_golden_set_meets_per_cohort_size_floor(
    causal_role_golden_set: dict[str, Any],
) -> None:
    """Per #358 acceptance: ≥30 entries per cohort."""
    from collections import Counter

    counts = Counter(e["cohort"] for e in causal_role_golden_set["entries"])
    for cohort in _EXPECTED_COHORTS:
        assert counts.get(cohort, 0) >= _MIN_ENTRIES_PER_COHORT, (
            f"{cohort}: only {counts.get(cohort, 0)} entries (need ≥{_MIN_ENTRIES_PER_COHORT})"
        )


def test_golden_set_meets_per_cohort_instrument_floor(
    causal_role_golden_set: dict[str, Any],
) -> None:
    """Phase 4 precision-power floor: ≥6 instrument labels per cohort."""
    from collections import Counter

    by_cohort_role: dict[tuple[str, str], int] = Counter(
        (e["cohort"], e["ground_truth_role"]) for e in causal_role_golden_set["entries"]
    )
    for cohort in _EXPECTED_COHORTS:
        n_instrument = by_cohort_role.get((cohort, "instrument"), 0)
        assert n_instrument >= _MIN_INSTRUMENTS_PER_COHORT, (
            f"{cohort}: only {n_instrument} instrument labels "
            f"(need ≥{_MIN_INSTRUMENTS_PER_COHORT} for Phase 4 precision power)"
        )


def test_every_entry_has_required_fields(
    causal_role_golden_set_entry: dict[str, Any],
) -> None:
    """Each entry must carry the 8 required schema fields."""
    missing = _REQUIRED_FIELDS - set(causal_role_golden_set_entry.keys())
    assert not missing, f"Entry missing fields: {sorted(missing)}"


def test_every_entry_role_in_vocabulary(
    causal_role_golden_set_entry: dict[str, Any],
) -> None:
    """Each entry's ground_truth_role must be one of the 6 CausalRole values."""
    assert causal_role_golden_set_entry["ground_truth_role"] in _VALID_ROLES, (
        f"Invalid role: {causal_role_golden_set_entry['ground_truth_role']!r}"
    )


def test_every_entry_has_provenance(
    causal_role_golden_set_entry: dict[str, Any],
) -> None:
    """Each entry's provenance must have at least one citation form (pmid/doi/url/citation)."""
    prov = causal_role_golden_set_entry.get("provenance", {})
    assert isinstance(prov, dict), f"provenance must be dict, got {type(prov)}"
    assert any(prov.get(k) for k in ("pmid", "doi", "url", "citation")), (
        f"Entry has no usable citation form: {prov}"
    )


def test_every_entry_has_substantive_rationale(
    causal_role_golden_set_entry: dict[str, Any],
) -> None:
    """Rationales must be substantive (≥50 chars) — the agent contract."""
    rationale = causal_role_golden_set_entry.get("rationale", "")
    assert isinstance(rationale, str) and len(rationale) >= 50, (
        f"Rationale too short ({len(rationale)} chars): {rationale!r}"
    )


def test_every_instrument_entry_has_verifiable_pmid_or_doi(
    causal_role_golden_set: dict[str, Any],
) -> None:
    """Instrument entries (gating use case) must carry a verifiable identifier.

    Per #358 Phase 4 precision-routing risk: a mislabeled instrument
    silently biases IV estimates. Every positive (label=instrument)
    entry must therefore be backed by either a PMID or DOI so a reviewer
    can audit the IV-validity reasoning against a real paper.
    """
    instruments = [
        e for e in causal_role_golden_set["entries"] if e["ground_truth_role"] == "instrument"
    ]
    assert len(instruments) >= 3 * _MIN_INSTRUMENTS_PER_COHORT, (
        f"Only {len(instruments)} instrument entries across all cohorts"
    )
    weak = [
        e
        for e in instruments
        if not (e.get("provenance", {}).get("pmid") or e.get("provenance", {}).get("doi"))
    ]
    assert not weak, (
        f"{len(weak)} instrument entries lack pmid/doi: {[e['feature_name'] for e in weak[:5]]}"
    )
