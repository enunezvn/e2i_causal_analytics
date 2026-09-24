"""Contract tests for ``merge_blocking_issues`` (#2283).

The graph-level regressions live in ``test_blocking_issues_channel_2283.py``;
this file pins the four properties the helper exists to provide, including the
two that rule out the alternative fixes considered in the issue:

* an ``operator.add`` reducer (duplicates on re-entry — #2238 / PR #2251), and
* a plain ``incoming + own`` merge (duplicates AND strands resolved issues,
  which would silently defeat the ``qc_remediation`` retry loop).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.blocking_issues import (
    KIND_GE_VALIDATION,
    KIND_LEAKAGE,
    KIND_QUALITY_CHECK,
    merge_blocking_issues,
    tag_blocking_issue,
)
from src.agents.ml_foundation.data_preparer.nodes.leakage_detector import detect_leakage
from src.agents.ml_foundation.data_preparer.nodes.leakage_remediation import (
    review_and_remediate_leakage,
)

_ANALYZE_LEAKAGE_LLM = (
    "src.agents.ml_foundation.data_preparer.nodes.leakage_remediation._analyze_leakage_with_llm"
)


def test_preserves_other_producers_entries() -> None:
    incoming = [
        "Schema validation failed: 6 error(s)",
        "sampling_frame_drift: max_drift_score=0.91 > 0.3000",
    ]
    result = merge_blocking_issues(incoming, ["score below bar"], kind=KIND_QUALITY_CHECK)
    assert result == incoming + [tag_blocking_issue(KIND_QUALITY_CHECK, "score below bar")]


def test_none_channel_is_treated_as_empty_and_never_returned() -> None:
    """The channel's initial value is ``None``; the helper must absorb it and
    must never hand ``None`` back, which is how ``ge_validator`` wiped it."""
    assert merge_blocking_issues(None, [], kind=KIND_GE_VALIDATION) == []
    assert merge_blocking_issues(None, ["boom"], kind=KIND_GE_VALIDATION) == [
        tag_blocking_issue(KIND_GE_VALIDATION, "boom")
    ]


def test_re_entry_replaces_own_entries_rather_than_appending() -> None:
    first = merge_blocking_issues(
        ["Schema validation failed: 6 error(s)"],
        ["missing values in column: age"],
        kind=KIND_QUALITY_CHECK,
    )
    second = merge_blocking_issues(
        first, ["missing values in column: age"], kind=KIND_QUALITY_CHECK
    )
    assert second == first, "own entry duplicated on the second pass"


def test_resolved_issue_is_retracted_on_re_entry() -> None:
    """Without this, ``qc_remediation``'s retry loop could never clear the gate:
    the entry that remediation fixed would sit in the channel forever."""
    first = merge_blocking_issues(
        ["Schema validation failed: 6 error(s)"],
        ["missing values in column: age"],
        kind=KIND_QUALITY_CHECK,
    )
    second = merge_blocking_issues(first, [], kind=KIND_QUALITY_CHECK)
    assert second == ["Schema validation failed: 6 error(s)"]


def test_one_kind_does_not_evict_another() -> None:
    after_qc = merge_blocking_issues(None, ["score below bar"], kind=KIND_QUALITY_CHECK)
    after_ge = merge_blocking_issues(after_qc, ["failed for train"], kind=KIND_GE_VALIDATION)
    assert after_ge == [
        tag_blocking_issue(KIND_QUALITY_CHECK, "score below bar"),
        tag_blocking_issue(KIND_GE_VALIDATION, "failed for train"),
    ]
    # ...and a later quality_check pass leaves the GE entry alone.
    after_qc_again = merge_blocking_issues(after_ge, [], kind=KIND_QUALITY_CHECK)
    assert after_qc_again == [tag_blocking_issue(KIND_GE_VALIDATION, "failed for train")]


@pytest.mark.asyncio
async def test_leakage_remediation_cannot_evict_a_foreign_entry() -> None:
    """Drives the REAL ``review_and_remediate_leakage`` node.

    Its prune used to drop ANY entry containing a leaked feature's name. A
    ``sampling_frame_drift:`` entry names its drifting columns, and a column can
    be both drifting and leaked — so remediating leakage silently retracted an
    unrelated, still-unresolved gate reason. That was the last surviving
    justification for the ``finalize_output`` re-promotion this PR deletes, so
    it is asserted against the node itself rather than a copy of its predicate.
    """
    sampling_entry = (
        "sampling_frame_drift: max_drift_score=0.91 > 0.3000 "
        "(worst column: 'age', columns_with_drift=['age'])"
    )
    schema_entry = "Schema validation failed: 6 error(s)"
    leakage_entry = tag_blocking_issue(
        KIND_LEAKAGE, "[CRITICAL] target_leakage: age correlates with target"
    )
    # A leakage entry about something ELSE. The leaked feature is ``age`` and
    # the kind prefix ``"leakage: "`` itself contains ``"age"``, so matching
    # the tagged string retracted this too (codex r2 HIGH) — it must survive.
    # Deliberately chosen so BOTH collision classes would retract it: the kind
    # prefix "leakage: " contains "age", and so does the message's own word
    # "leakage". Only identifier-boundary matching on the untagged message
    # keeps it. (codex r2 + r3 HIGHs.)
    unrelated_leakage_entry = tag_blocking_issue(
        KIND_LEAKAGE, "[HIGH] Temporal leakage: event_date precedes the label window"
    )

    rng = np.random.default_rng(2283)
    n = 120
    target = rng.integers(0, 2, n)
    train_df = pd.DataFrame(
        {
            "age": target * 1.0,  # perfectly correlated -> the leaked feature
            # The node requires >= 2 surviving clean features to call the
            # remediation viable, so give it three.
            "clean_a": rng.standard_normal(n),
            "clean_b": rng.standard_normal(n),
            "clean_c": rng.standard_normal(n),
            "target": target,
        }
    )
    state: dict = {
        "experiment_id": "exp-2283-leakage-prune",
        "leakage_severity": "critical",
        "leakage_remediation_attempts": 0,
        "leaked_features": ["age"],
        "leakage_findings": [
            {"feature": "age", "severity": "critical", "check_name": "target_leakage"}
        ],
        "blocking_issues": [
            leakage_entry,
            unrelated_leakage_entry,
            sampling_entry,
            schema_entry,
        ],
        "train_df": train_df,
        "validation_df": None,
        "test_df": None,
        "holdout_df": None,
        "scope_spec": {"prediction_target": "target"},
    }
    analysis = {
        "leakage_classifications": {"age": "target_leakage"},
        "features_to_drop": ["age"],
        "replacement_candidates": [],
        "recommended_feature_set": ["clean_a", "clean_b", "clean_c"],
        "reasoning": "test-injected",
    }

    with patch(_ANALYZE_LEAKAGE_LLM, new=AsyncMock(return_value=analysis)):
        result = await review_and_remediate_leakage(state)  # type: ignore[arg-type]

    assert result["leakage_remediation_status"] == "applied", (
        f"fixture did not reach the prune branch: {result!r}"
    )
    assert result["blocking_issues"] == [
        unrelated_leakage_entry,
        sampling_entry,
        schema_entry,
    ], (
        "the prune must retract only its OWN-kind entries that name a remediated "
        "feature: it either evicted a foreign entry or over-matched on the kind prefix"
    )


@pytest.mark.asyncio
async def test_leakage_detection_error_reaches_the_gate() -> None:
    """A crashed leakage audit must fail the gate CLOSED.

    ``detect_leakage``'s exception path recorded ``leakage_severity="critical"``
    and an ``error``, but no ``blocking_issues`` entry — and ``finalize_output``
    reads neither of those, so "assume worst case" was recorded nowhere the
    gate looks. The graph-level gate could therefore pass on a run whose
    leakage audit had crashed (codex r2 HIGH on #2283). The agent wrapper does
    raise on ``error``, but the gate contract must hold on its own.
    """
    state: dict = {
        "experiment_id": "exp-2283-leakage-error",
        "blocking_issues": ["Schema validation failed: 6 error(s)"],
        # A non-DataFrame train_df drives the node into its except block.
        "train_df": object(),
        "scope_spec": {"prediction_target": "target"},
    }

    result = await detect_leakage(state)  # type: ignore[arg-type]

    assert result["error_type"] == "leakage_detection_error"
    blocking = result["blocking_issues"]
    assert any(i.startswith(f"{KIND_LEAKAGE}: detection error") for i in blocking), (
        f"a crashed leakage audit left no blocker for the gate: {blocking!r}"
    )
    # ...and the upstream schema entry is still there.
    assert "Schema validation failed: 6 error(s)" in blocking
