"""Contract tests for ``merge_blocking_issues`` (#2283).

The graph-level regressions live in ``test_blocking_issues_channel_2283.py``;
this file pins the four properties the helper exists to provide, including the
two that rule out the alternative fixes considered in the issue:

* an ``operator.add`` reducer (duplicates on re-entry — #2238 / PR #2251), and
* a plain ``incoming + own`` merge (duplicates AND strands resolved issues,
  which would silently defeat the ``qc_remediation`` retry loop).
"""

from __future__ import annotations

from src.agents.ml_foundation.data_preparer.blocking_issues import (
    KIND_GE_VALIDATION,
    KIND_QUALITY_CHECK,
    merge_blocking_issues,
    tag_blocking_issue,
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
