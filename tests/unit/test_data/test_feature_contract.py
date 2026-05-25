"""Tests for FeatureContract — Layer 1 of adaptive-temporal-validity redesign.

The FeatureContract is a refinement-type-style declarative spec. Every feature
in the pipeline carries one. The framework propagates `knowable_at` through
derivation chains and rejects features whose derivation pulls from data that
isn't knowable at prediction time.

This is the FIRST layer of the four-layer adaptive leakage defense:
  1. THIS — declarative temporal contracts (catches semantic leaks at author time)
  2. Causal-DAG + KG-grounded LLM (catches semantic leaks at pipeline time)
  3. Adversarial discriminator (catches statistical leaks at training time)
  4. DSPy-compiled adaptive prompts (catches future leaks via continuous learning)
"""

from __future__ import annotations

import pytest


def test_feature_contract_minimal_construction():
    """A minimum-viable FeatureContract has: name, knowable_at, source."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    contract = FeatureContract(
        name="age_at_index",
        knowable_at=KnowableAt(reference="index_date", offset_days=0),
        source="demo",
        derivation_inputs=["birth_date", "index_date"],
        aggregation=None,
    )
    assert contract.name == "age_at_index"
    assert contract.knowable_at.reference == "index_date"


def test_feature_contract_rejects_post_index_aggregation_without_window():
    """A feature aggregating events without window_days fails contract validation
    when the source events extend post-index. This catches the
    journey_duration_days mechanism: max(eligend, last_med_date+supply, ...)
    where the events table extends past index.
    """
    from src.data.feature_contract import ContractViolation, FeatureContract, KnowableAt

    with pytest.raises(ContractViolation, match="aggregation without window_days"):
        FeatureContract(
            name="bad_journey_duration",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date", "days_supply"],
            aggregation="max",
            window_days=None,  # <-- forbidden when source is event table
        )


def test_feature_contract_accepts_aggregation_with_window():
    """A feature with explicit window_days is contract-valid."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    contract = FeatureContract(
        name="med_fill_count_180d",
        knowable_at=KnowableAt(reference="index_date"),
        source="medication_events",
        derivation_inputs=["medication_date"],
        aggregation="count",
        window_days=180,
    )
    assert contract.window_days == 180


def test_contract_validates_knowable_at_propagation():
    """A derived feature's knowable_at must be >= max(inputs.knowable_at).

    journey_duration_days = end_date - index_date
      end_date.knowable_at = "post_index" (it includes future events)
      index_date.knowable_at = "index_date" (it IS the prediction time)
      journey_duration_days.knowable_at must be >= max(post_index, index_date) = post_index
      → This feature is post-prediction-time → REJECT
    """
    from src.data.feature_contract import FeatureContract, KnowableAt, validate_contract_chain

    end_date_contract = FeatureContract(
        name="end_date",
        knowable_at=KnowableAt(reference="post_index"),  # Honest declaration
        source="medication_events",
        derivation_inputs=["medication_date"],
        aggregation="max",
        window_days=None,  # Will fail contract validation, but exists for the test
        _allow_unwindowed_for_test=True,  # Test-only escape hatch
    )
    index_date_contract = FeatureContract(
        name="index_date",
        knowable_at=KnowableAt(reference="index_date"),
        source="cohort",
        derivation_inputs=[],
        aggregation=None,
    )
    # Now derive journey_duration_days
    duration_contract = FeatureContract(
        name="journey_duration_days",
        knowable_at=KnowableAt(reference="index_date"),  # CLAIMS to be pre-index
        source="derived",
        derivation_inputs=["end_date", "index_date"],
        aggregation=None,
    )

    chain = {
        "end_date": end_date_contract,
        "index_date": index_date_contract,
        "journey_duration_days": duration_contract,
    }
    violations = validate_contract_chain(chain)
    # Should detect that journey_duration_days CLAIMS knowable_at=index_date
    # but its input end_date has knowable_at=post_index → claim is false
    assert any(
        v.feature == "journey_duration_days" and "knowable_at" in v.reason for v in violations
    ), f"Expected violation on journey_duration_days; got {violations}"


def test_contract_chain_passes_for_legitimate_feature():
    """A legitimate pre-index feature (e.g., age) passes the chain."""
    from src.data.feature_contract import FeatureContract, KnowableAt, validate_contract_chain

    age_contract = FeatureContract(
        name="age_at_index",
        knowable_at=KnowableAt(reference="index_date"),
        source="demo",
        derivation_inputs=["birth_date", "index_date"],
        aggregation=None,
    )
    birth_date_contract = FeatureContract(
        name="birth_date",
        knowable_at=KnowableAt(reference="enrollment"),  # Always pre-index
        source="demo",
        derivation_inputs=[],
        aggregation=None,
    )
    index_date_contract = FeatureContract(
        name="index_date",
        knowable_at=KnowableAt(reference="index_date"),
        source="cohort",
        derivation_inputs=[],
        aggregation=None,
    )
    chain = {
        "age_at_index": age_contract,
        "birth_date": birth_date_contract,
        "index_date": index_date_contract,
    }
    violations = validate_contract_chain(chain)
    assert violations == []


def test_aggregation_must_specify_event_source():
    """A feature with `aggregation` must declare a `source` that's an event table."""
    from src.data.feature_contract import ContractViolation, FeatureContract, KnowableAt

    with pytest.raises(ContractViolation, match="aggregation requires event-typed source"):
        FeatureContract(
            name="invalid_agg",
            knowable_at=KnowableAt(reference="index_date"),
            source="demo",  # demo is not an event source
            derivation_inputs=["age"],
            aggregation="sum",
            window_days=180,
        )


def test_window_days_must_be_positive():
    """window_days < 1 is invalid."""
    from src.data.feature_contract import ContractViolation, FeatureContract, KnowableAt

    with pytest.raises(ContractViolation, match="window_days must be >= 1"):
        FeatureContract(
            name="bad_window",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=0,
        )


def test_compile_set_18_incidents_caught_by_layer_1():
    """Each of the 18 documented past-leakage incidents from the compile set
    SHOULD be catchable by Layer 1 (declarative temporal contracts) IF authored
    with honest contract metadata.

    This test verifies that for each incident in the compile set, attempting to
    construct its leaky derivation as a FeatureContract either:
    (a) fails contract validation outright, OR
    (b) requires misrepresenting the knowable_at to pass — which Layer 2's
        causal-DAG check would then catch.
    """
    from src.data.feature_contract import ContractViolation, FeatureContract, KnowableAt

    # Incident 1: disease_severity — sums med_fill events without window
    with pytest.raises(ContractViolation):
        FeatureContract(
            name="disease_severity",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date", "j2357_proc_count", "abnormal_lab_count"],
            aggregation="sum",
            window_days=None,  # The actual code derivation lacks window
        )

    # Incident 3: days_on_therapy — sum med.days_sup over entire panel
    with pytest.raises(ContractViolation):
        FeatureContract(
            name="days_on_therapy",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["days_supply"],
            aggregation="sum",
            window_days=None,
        )

    # Incident 4: medication_claim_count — count of rows in med_panel
    with pytest.raises(ContractViolation):
        FeatureContract(
            name="medication_claim_count",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=None,
        )


# --- Codex audit follow-ups (Layer 1 — item D) ------------------------------


def test_validate_chain_catches_post_index_offset_input():
    """A feature claiming knowable_at=index_date+0 with an input at
    index_date+30 is a chain violation: the input is 30 days post-index even
    though it nominally shares the ``index_date`` reference. Codex audit
    Bug 1: ``rank_of`` ignored ``offset_days``, returning equal ranks for
    both, so the violation slipped through.
    """
    from src.data.feature_contract import (
        FeatureContract,
        KnowableAt,
        validate_contract_chain,
    )

    parent = FeatureContract(
        name="parent_at_index",
        knowable_at=KnowableAt(reference="index_date", offset_days=0),
        source="derived",
        derivation_inputs=["future_input"],
    )
    future_input = FeatureContract(
        name="future_input",
        knowable_at=KnowableAt(reference="index_date", offset_days=30),
        source="derived",
    )
    violations = validate_contract_chain({"parent_at_index": parent, "future_input": future_input})
    assert len(violations) == 1, (
        f"Expected exactly one chain violation; got {violations}. "
        f"rank_of must include offset_days so input(index_date+30) > parent(index_date+0)."
    )
    assert violations[0].feature == "parent_at_index"
    assert "future_input" in violations[0].inputs_implicated


def test_validate_chain_pre_index_input_is_not_a_violation():
    """The fix to rank_of must not flip directionality: an input that is
    earlier than its parent (negative offset / earlier reference) is fine.
    """
    from src.data.feature_contract import (
        FeatureContract,
        KnowableAt,
        validate_contract_chain,
    )

    parent = FeatureContract(
        name="parent_at_index",
        knowable_at=KnowableAt(reference="index_date", offset_days=0),
        source="derived",
        derivation_inputs=["earlier_input"],
    )
    earlier_input = FeatureContract(
        name="earlier_input",
        knowable_at=KnowableAt(reference="index_date", offset_days=-180),
        source="derived",
    )
    violations = validate_contract_chain(
        {"parent_at_index": parent, "earlier_input": earlier_input}
    )
    assert violations == [], (
        f"Earlier input (negative offset) must NOT be a chain violation; got {violations}"
    )


def test_allow_unwindowed_for_test_escape_hatch_requires_post_index():
    """The ``_allow_unwindowed_for_test`` escape hatch must reject contracts
    that claim ``knowable_at=index_date`` (or any pre-or-at-index claim).
    Codex audit Bug 2: the hatch was a footgun — a future author could
    construct ``knowable_at=index_date, aggregation=count, window_days=None,
    _allow_unwindowed_for_test=True`` silently, defeating the windowing
    enforcement that Layer 1 is built to provide.
    """
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    # The legit use: declare honestly post-index unwindowed aggregation
    legit = FeatureContract(
        name="cumulative_post_index_count",
        knowable_at=KnowableAt(reference="post_index"),
        source="medication_events",
        derivation_inputs=["medication_date"],
        aggregation="count",
        window_days=None,
        _allow_unwindowed_for_test=True,
    )
    assert legit.knowable_at.reference == "post_index"

    # The footgun: claim knowable_at=index_date while bypassing the window
    with pytest.raises(ContractViolation, match="_allow_unwindowed_for_test"):
        FeatureContract(
            name="footgun",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=None,
            _allow_unwindowed_for_test=True,
        )

    # And claiming enrollment-time
    with pytest.raises(ContractViolation, match="_allow_unwindowed_for_test"):
        FeatureContract(
            name="footgun_enrollment",
            knowable_at=KnowableAt(reference="enrollment"),
            source="medication_events",
            derivation_inputs=["medication_date"],
            aggregation="count",
            window_days=None,
            _allow_unwindowed_for_test=True,
        )


def test_window_days_without_aggregation_is_rejected():
    """A contract with ``window_days`` set but ``aggregation=None`` has no
    defined semantics — windows describe the temporal scope of an aggregation,
    so a window without one is meaningless. Codex audit design gap: the old
    validator silently accepted this combination, letting a contract claim a
    180-day window over a feature that never aggregates anything.
    """
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="window_days.*aggregation"):
        FeatureContract(
            name="meaningless_window",
            knowable_at=KnowableAt(reference="index_date"),
            source="demo",
            window_days=180,
            aggregation=None,
        )

    # Sanity: window_days WITH aggregation is still valid (regression check)
    legit = FeatureContract(
        name="legit_windowed",
        knowable_at=KnowableAt(reference="index_date"),
        source="medication_events",
        derivation_inputs=["medication_date"],
        aggregation="count",
        window_days=180,
    )
    assert legit.window_days == 180
    assert legit.aggregation == "count"


# ---------------------------------------------------------------------------
# PR-A: kg_entity_codes field (Phase 2.9 Stage 2)
# ---------------------------------------------------------------------------


def test_feature_contract_default_kg_entity_codes_is_empty_tuple():
    """Existing manifests without kg_entity_codes get the default ()."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="age",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("age",),
    )
    assert fc.kg_entity_codes == ()


def test_feature_contract_accepts_single_kg_entity_code():
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="primary_diagnosis_code",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("diagcode",),
        kg_entity_codes=(("ICD10CM", "L20.9"),),
    )
    assert fc.kg_entity_codes == (("ICD10CM", "L20.9"),)


def test_feature_contract_accepts_multiple_kg_entity_codes():
    """A feature can carry several entity codes (cross-walks)."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="has_atopic_dermatitis",
        knowable_at=KnowableAt(reference="index_date"),
        source="diagnosis_events",
        derivation_inputs=("admit_date", "diag1"),
        aggregation="max",
        window_days=180,
        kg_entity_codes=(
            ("ICD10CM", "L20.9"),
            ("UMLS", "C0011615"),
        ),
    )
    assert len(fc.kg_entity_codes) == 2


def test_feature_contract_rejects_empty_code_string():
    """Validation: every (system, code) tuple needs a non-empty code."""
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="kg_entity_codes"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("ICD10CM", ""),),
        )


def test_feature_contract_rejects_unknown_code_system():
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="kg_entity_codes"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("NOT_A_VOCAB", "L20.9"),),
        )


def test_feature_contract_normalizes_kg_entity_codes_to_tuple_of_tuples():
    """Caller may pass list-of-lists; stored as tuple-of-tuples (frozen)."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="x",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("a",),
        kg_entity_codes=[["ICD10CM", "L20.9"], ["UMLS", "C0011615"]],
    )
    assert isinstance(fc.kg_entity_codes, tuple)
    assert all(isinstance(t, tuple) for t in fc.kg_entity_codes)


def test_feature_contract_rejects_malformed_tuple_shape():
    """Each entry must be a 2-tuple."""
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="2-tuples"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("ICD10CM",),),  # missing code
        )


def test_feature_contract_empty_list_normalized_to_tuple():
    """Codex H1: an empty list ``[]`` for kg_entity_codes was passing the
    falsy guard and storing a mutable list inside a frozen dataclass.
    Verify normalization happens unconditionally and the field is always
    a tuple, never a list."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="x",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("a",),
        kg_entity_codes=[],  # empty list — must normalize to ()
    )
    assert isinstance(fc.kg_entity_codes, tuple)
    assert fc.kg_entity_codes == ()


def test_feature_contract_rejects_whitespace_only_code():
    """Codex M1: a whitespace-only code (e.g., `"   "`) was passing the
    `not code` check because non-empty strings are truthy. EntityLinker
    would then receive whitespace as a real code and confuse downstream
    diagnostics. Strip + reject at construction time."""
    from src.data.feature_contract import (
        ContractViolation,
        FeatureContract,
        KnowableAt,
    )

    with pytest.raises(ContractViolation, match="non-whitespace"):
        FeatureContract(
            name="x",
            knowable_at=KnowableAt(reference="enrollment"),
            source="demo",
            derivation_inputs=("a",),
            kg_entity_codes=(("ICD10CM", "   "),),
        )


def test_kg_known_systems_derived_from_canonical_codesystem_literal():
    """Codex M2: _KG_KNOWN_SYSTEMS must stay in sync with the canonical
    CodeSystem literal at src/data/kg/types.py. The derivation via
    typing.get_args + a {"UMLS"} addition prevents drift when types.py
    adds a new vocabulary."""
    from typing import get_args

    from src.data.feature_contract import _KG_KNOWN_SYSTEMS
    from src.data.kg.types import CodeSystem

    canonical = set(get_args(CodeSystem))
    expected = canonical | {"UMLS"}
    assert _KG_KNOWN_SYSTEMS == expected, (
        f"_KG_KNOWN_SYSTEMS ({_KG_KNOWN_SYSTEMS}) drifted from CodeSystem | {{UMLS}} ({expected})"
    )


def test_feature_contract_accepts_umls_system():
    """UMLS is a meta-system not in CodeSystem (which lists source
    vocabularies). _KG_KNOWN_SYSTEMS adds it back for manifests that
    declare a UMLS CUI directly."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="x",
        knowable_at=KnowableAt(reference="enrollment"),
        source="demo",
        derivation_inputs=("a",),
        kg_entity_codes=(("UMLS", "C0011615"),),
    )
    assert fc.kg_entity_codes == (("UMLS", "C0011615"),)


# ---------------------------------------------------------------------------
# Issue #501 — CausalStructureAttestation edge-list field (plan §8.3).
# ---------------------------------------------------------------------------


def test_feature_contract_causal_structure_defaults_none():
    """An un-attested contract has ``causal_structure is None`` and all existing
    construction paths are unaffected (backward-compatible default)."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    fc = FeatureContract(
        name="age_at_index",
        knowable_at=KnowableAt(reference="index_date"),
        source="demo",
        derivation_inputs=("birth_date", "index_date"),
    )
    assert fc.causal_structure is None


def test_feature_contract_causal_structure_roundtrips_edge_list():
    """An authored edge list builds a valid nx.DiGraph fragment and the extended
    ``extract_role`` over it returns the M-structure collider role.

    Encodes the REAL DAG ``T → V ← U → Y`` (independent second parent U), the
    anti-mocking-faithful shape for cases 1,2,7,8,9.
    """
    import networkx as nx

    from src.data.feature_contract import (
        CausalStructureAttestation,
        FeatureContract,
        KnowableAt,
    )
    from src.ml.causal_role_dgp.extractor import extract_role

    attestation = CausalStructureAttestation(
        treatment_node="T",
        outcome_node="Y",
        feature_node="V",
        edges=(("T", "V"), ("U", "V"), ("U", "Y")),
    )
    fc = FeatureContract(
        name="on_treatment_at_12m_flag",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        derivation_inputs=("treatment_status",),
        causal_role="collider",
        causal_structure=attestation,
    )
    assert fc.causal_structure is attestation

    graph = nx.DiGraph(list(fc.causal_structure.edges))
    role = extract_role(
        fc.causal_structure.feature_node,
        fc.causal_structure.treatment_node,
        fc.causal_structure.outcome_node,
        graph,
    )
    assert role == "collider"


def test_causal_structure_attestation_is_frozen_and_hashable():
    """The attestation is frozen + hashable (edges normalized to tuple-of-tuple),
    so a FeatureContract carrying it stays hashable."""
    from src.data.feature_contract import CausalStructureAttestation

    # Pass JSON-friendly list-of-lists; __post_init__ normalizes to tuples.
    attestation = CausalStructureAttestation(
        treatment_node="T",
        outcome_node="Y",
        feature_node="V",
        edges=[["T", "V"], ["U", "V"], ["U", "Y"]],  # type: ignore[arg-type]
    )
    assert attestation.edges == (("T", "V"), ("U", "V"), ("U", "Y"))
    # hashable (frozen + tuple edges)
    assert isinstance(hash(attestation), int)
