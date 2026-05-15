"""Unit tests for SidecarReader (Plan
.claude/plans/layer4_evaluator_audit_consumer.md)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _write_sidecar(directory: Path, experiment_id: str, written_at: str,
                   verdicts: list[dict]) -> Path:
    sub = directory / experiment_id
    sub.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "data_source": "synthetic",
        "written_at": written_at,
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [v["feature"] for v in verdicts],
        "adaptive_verdicts": verdicts,
    }
    out = sub / f"adaptive_verdicts_{written_at.replace(':', '')}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def test_reader_loads_all_sidecars_in_directory(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(tmp_path, "exp-a", "2026-05-15T10:00:00Z", [
        {"feature": "f1", "layer": "4", "severity": "moderate",
         "evaluator_satisfied": False,
         "evaluator_missed_considerations": ["temporal_filter"],
         "evaluator_notes": "thin rationale", "evaluator_model": "haiku",
         "evaluator_rationale_complete": False}
    ])
    _write_sidecar(tmp_path, "exp-b", "2026-05-15T11:00:00Z", [
        {"feature": "f2", "layer": "4", "severity": "moderate",
         "evaluator_satisfied": True,
         "evaluator_missed_considerations": [],
         "evaluator_notes": "", "evaluator_model": "haiku",
         "evaluator_rationale_complete": True}
    ])

    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 2
    assert {r.feature for r in records} == {"f1", "f2"}
    assert {r.experiment_id for r in records} == {"exp-a", "exp-b"}


def test_reader_time_window_filter(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(tmp_path, "exp-old", "2026-04-01T10:00:00Z",
                   [{"feature": "old", "layer": "4",
                     "evaluator_satisfied": False,
                     "evaluator_missed_considerations": [],
                     "evaluator_notes": "", "evaluator_model": "haiku",
                     "evaluator_rationale_complete": False}])
    _write_sidecar(tmp_path, "exp-new", "2026-05-10T10:00:00Z",
                   [{"feature": "new", "layer": "4",
                     "evaluator_satisfied": False,
                     "evaluator_missed_considerations": [],
                     "evaluator_notes": "", "evaluator_model": "haiku",
                     "evaluator_rationale_complete": False}])

    reader = SidecarReader(
        artifacts_dir=tmp_path,
        since=datetime(2026, 5, 1, tzinfo=timezone.utc),
        until=datetime(2026, 5, 31, tzinfo=timezone.utc),
    )
    features = [r.feature for r in reader.iter_verdict_records()]
    assert features == ["new"]


def test_reader_tolerates_pre_evaluator_schema_sidecar(tmp_path):
    """Sidecars from before 2026-05-15 lack the 5 evaluator keys.
    The reader must accept them and surface evaluator fields as None."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(tmp_path, "exp-pre", "2026-04-15T10:00:00Z",
                   [{"feature": "f1", "layer": "4", "severity": "moderate"}])
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    r = records[0]
    assert r.feature == "f1"
    assert r.evaluator_satisfied is None
    assert r.evaluator_missed_considerations is None


def test_reader_tolerates_malformed_json(tmp_path, caplog):
    from src.data.audit_sidecar_reader import SidecarReader

    sub = tmp_path / "exp-bad"
    sub.mkdir(parents=True)
    (sub / "adaptive_verdicts_BAD.json").write_text("{this is: not valid json")
    _write_sidecar(tmp_path, "exp-good", "2026-05-15T10:00:00Z",
                   [{"feature": "g", "layer": "4",
                     "evaluator_satisfied": False,
                     "evaluator_missed_considerations": [],
                     "evaluator_notes": "", "evaluator_model": "haiku",
                     "evaluator_rationale_complete": False}])

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert [r.feature for r in records] == ["g"]
    assert any("malformed" in rec.message.lower()
               or "decode" in rec.message.lower()
               or "skip" in rec.message.lower()
               for rec in caplog.records)


def test_reader_empty_directory_returns_empty(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader
    reader = SidecarReader(artifacts_dir=tmp_path)
    assert list(reader.iter_verdict_records()) == []


def test_reader_missing_directory_returns_empty(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader
    nonexistent = tmp_path / "does-not-exist"
    reader = SidecarReader(artifacts_dir=nonexistent)
    assert list(reader.iter_verdict_records()) == []


def test_reader_logs_warning_on_string_bool_drift(tmp_path, caplog):
    """Codex Gate-2 MED-1: a future producer that accidentally writes
    `"evaluator_satisfied": "false"` (string) instead of a real bool must
    log a WARNING — not silently coerce to None and drop the disagreement.
    The reader still returns the record with evaluator_satisfied=None
    (so extract_disagreements skips it), but operators see the drift in
    logs."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(tmp_path, "exp-drift", "2026-05-15T10:00:00Z", [
        {"feature": "f-drift", "layer": "4",
         "evaluator_satisfied": "false",  # WRONG: string, not bool
         "evaluator_rationale_complete": False,
         "evaluator_missed_considerations": [],
         "evaluator_notes": "drift", "evaluator_model": "haiku"},
    ])
    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].evaluator_satisfied is None
    assert any(
        "non-bool" in rec.message.lower() and "drift" in rec.message.lower()
        for rec in caplog.records
    )


def test_reader_compact_format_timestamp_from_producer(tmp_path):
    """Codex Gate-2 MED-3: the actual producer writes `written_at` using
    `%Y%m%dT%H%M%SZ` (e.g. `20260515T103000Z`), not the ISO-extended form
    with hyphens/colons. Pin reader compatibility with the producer's
    real format."""
    from src.data.audit_sidecar_reader import SidecarReader
    _write_sidecar(tmp_path, "exp-compact", "20260515T103000Z", [
        {"feature": "f-c", "layer": "4",
         "evaluator_satisfied": False,
         "evaluator_rationale_complete": False,
         "evaluator_missed_considerations": [],
         "evaluator_notes": "", "evaluator_model": "haiku"},
    ])
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].feature == "f-c"
    # Compact format parses to 2026-05-15 10:30:00 UTC.
    assert records[0].written_at.year == 2026
    assert records[0].written_at.month == 5
    assert records[0].written_at.day == 15


def test_extract_disagreements_filters_to_satisfied_false():
    from src.data.audit_sidecar_reader import (
        VerdictRecord,
        extract_disagreements,
    )

    sat = VerdictRecord(
        experiment_id="e", written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"), feature="f-sat", layer="4",
        severity="moderate", remediation="keep_with_caveat", evidence=None,
        z_score=2.0, p_value=0.05, delta_auc=0.04,
        evaluator_satisfied=True, evaluator_rationale_complete=True,
        evaluator_missed_considerations=[], evaluator_notes="",
        evaluator_model="haiku", raw_verdict={},
    )
    unsat = VerdictRecord(
        experiment_id="e", written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"), feature="f-unsat", layer="4",
        severity="moderate", remediation="keep_with_caveat", evidence=None,
        z_score=2.0, p_value=0.05, delta_auc=0.04,
        evaluator_satisfied=False, evaluator_rationale_complete=False,
        evaluator_missed_considerations=["temporal_filter"],
        evaluator_notes="thin", evaluator_model="haiku", raw_verdict={},
    )
    no_eval = VerdictRecord(
        experiment_id="e", written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"), feature="f-no-eval", layer="4",
        severity="moderate", remediation="keep_with_caveat", evidence=None,
        z_score=2.0, p_value=0.05, delta_auc=0.04,
        evaluator_satisfied=None, evaluator_rationale_complete=None,
        evaluator_missed_considerations=None, evaluator_notes=None,
        evaluator_model=None, raw_verdict={},
    )

    events = list(extract_disagreements([sat, unsat, no_eval]))
    assert len(events) == 1
    assert events[0].feature == "f-unsat"
    assert events[0].missed_considerations == ("temporal_filter",)
