"""Unit tests for audit_candidate_formatter (Plan
.claude/plans/layer4_evaluator_audit_consumer.md)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _make_event(feature: str = "f1", *, missed=("temporal_filter",), notes="thin rationale"):
    from src.data.audit_sidecar_reader import DisagreementEvent

    return DisagreementEvent(
        experiment_id="exp-1",
        written_at=datetime(2026, 5, 15, 10, 0, tzinfo=timezone.utc),
        source_path=Path("/tmp/exp-1/adaptive_verdicts_X.json"),
        feature=feature,
        worker_severity="moderate",
        worker_remediation="keep_with_caveat",
        rationale_complete=False,
        missed_considerations=missed,
        notes=notes,
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )


def test_format_markdown_report_includes_required_sections():
    from src.data.audit_candidate_formatter import format_markdown_report

    events = [_make_event("ondansetron_fills_180d")]
    md = format_markdown_report(
        events, generated_at=datetime(2026, 5, 15, 11, 0, tzinfo=timezone.utc)
    )

    assert "# Compile-set candidates" in md
    assert "ondansetron_fills_180d" in md
    assert "temporal_filter" in md
    assert "thin rationale" in md
    # Per acceptance criterion #4, an accept/reject checkbox must appear.
    assert "[ ] accept" in md
    assert "[ ] reject" in md
    # Source attribution.
    assert "exp-1" in md
    # Generated-at header.
    assert "2026-05-15" in md


def test_format_markdown_report_lists_all_required_fill_ins():
    """Codex review MED-8: the markdown must mention all 4 fields the
    reviewer must populate before accepting (derivation_pseudocode +
    dataset_context, not just the two expected_* fields)."""
    from src.data.audit_candidate_formatter import format_markdown_report

    md = format_markdown_report(
        [_make_event("f1")],
        generated_at=datetime(2026, 5, 15, 11, 0, tzinfo=timezone.utc),
    )
    for field in (
        "expected_causal_role",
        "expected_remediation",
        "derivation_pseudocode",
        "dataset_context",
    ):
        assert field in md, f"markdown missing required fill-in: {field}"
    assert "Required fill-ins before accepting" in md


def test_format_markdown_report_empty_input_is_handled():
    from src.data.audit_candidate_formatter import format_markdown_report

    md = format_markdown_report([], generated_at=datetime(2026, 5, 15, 11, 0, tzinfo=timezone.utc))
    assert "No candidates" in md
    assert "# Compile-set candidates" in md


def test_format_json_manifest_shape():
    from src.data.audit_candidate_formatter import format_json_manifest

    events = [_make_event("f1"), _make_event("f2", missed=(), notes="")]
    manifest = format_json_manifest(
        events, generated_at=datetime(2026, 5, 15, 11, 0, tzinfo=timezone.utc)
    )

    assert manifest["generated_at"] == "2026-05-15T11:00:00+00:00"
    # Codex review MED-8: top-level required_fill_ins enumerates the 4
    # fields downstream consumers must validate before merging.
    assert manifest["required_fill_ins"] == [
        "expected_causal_role",
        "expected_remediation",
        "derivation_pseudocode",
        "dataset_context",
    ]
    assert len(manifest["candidates"]) == 2
    c0 = manifest["candidates"][0]
    # Per acceptance criterion #5, expected_* keys are present and null
    # (human fills them in at review).
    assert c0["feature_name"] == "f1"
    assert c0["expected_causal_role"] is None
    assert c0["expected_remediation"] is None
    # The derivation/context pair is also null (codex review MED-8) and
    # listed in required_fill_ins above.
    assert c0["derivation_pseudocode"] is None
    assert c0["dataset_context"] is None
    # Evaluator audit echoed for context.
    assert c0["evaluator_audit"]["satisfied"] is False
    assert c0["evaluator_audit"]["missed_considerations"] == ["temporal_filter"]
    assert c0["evaluator_audit"]["notes"] == "thin rationale"
    # Source attribution.
    assert c0["source_run_id"] == "exp-1"


def test_json_manifest_is_round_trippable():
    """The manifest is consumed by a human; it must round-trip through
    json.dumps/json.loads cleanly."""
    from src.data.audit_candidate_formatter import format_json_manifest

    manifest = format_json_manifest(
        [_make_event()], generated_at=datetime(2026, 5, 15, 11, 0, tzinfo=timezone.utc)
    )
    serialised = json.dumps(manifest, indent=2)
    parsed = json.loads(serialised)
    assert parsed == manifest
