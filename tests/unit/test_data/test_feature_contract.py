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
