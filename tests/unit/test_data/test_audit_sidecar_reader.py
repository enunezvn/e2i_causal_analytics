"""Unit tests for SidecarReader (Plan
.claude/plans/layer4_evaluator_audit_consumer.md)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _write_sidecar(
    directory: Path, experiment_id: str, written_at: str, verdicts: list[dict]
) -> Path:
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

    _write_sidecar(
        tmp_path,
        "exp-a",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f1",
                "layer": "4",
                "severity": "moderate",
                "evaluator_satisfied": False,
                "evaluator_missed_considerations": ["temporal_filter"],
                "evaluator_notes": "thin rationale",
                "evaluator_model": "haiku",
                "evaluator_rationale_complete": False,
            }
        ],
    )
    _write_sidecar(
        tmp_path,
        "exp-b",
        "2026-05-15T11:00:00Z",
        [
            {
                "feature": "f2",
                "layer": "4",
                "severity": "moderate",
                "evaluator_satisfied": True,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
                "evaluator_rationale_complete": True,
            }
        ],
    )

    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 2
    assert {r.feature for r in records} == {"f1", "f2"}
    assert {r.experiment_id for r in records} == {"exp-a", "exp-b"}


def test_reader_time_window_filter(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-old",
        "2026-04-01T10:00:00Z",
        [
            {
                "feature": "old",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
                "evaluator_rationale_complete": False,
            }
        ],
    )
    _write_sidecar(
        tmp_path,
        "exp-new",
        "2026-05-10T10:00:00Z",
        [
            {
                "feature": "new",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
                "evaluator_rationale_complete": False,
            }
        ],
    )

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

    _write_sidecar(
        tmp_path,
        "exp-pre",
        "2026-04-15T10:00:00Z",
        [{"feature": "f1", "layer": "4", "severity": "moderate"}],
    )
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
    _write_sidecar(
        tmp_path,
        "exp-good",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "g",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
                "evaluator_rationale_complete": False,
            }
        ],
    )

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert [r.feature for r in records] == ["g"]
    assert any(
        "malformed" in rec.message.lower()
        or "decode" in rec.message.lower()
        or "skip" in rec.message.lower()
        for rec in caplog.records
    )


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

    _write_sidecar(
        tmp_path,
        "exp-drift",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f-drift",
                "layer": "4",
                "evaluator_satisfied": "false",  # WRONG: string, not bool
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "drift",
                "evaluator_model": "haiku",
            },
        ],
    )
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

    _write_sidecar(
        tmp_path,
        "exp-compact",
        "20260515T103000Z",
        [
            {
                "feature": "f-c",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
            },
        ],
    )
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].feature == "f-c"
    # Compact format parses to 2026-05-15 10:30:00 UTC.
    assert records[0].written_at.year == 2026
    assert records[0].written_at.month == 5
    assert records[0].written_at.day == 15


def test_reader_opt_float_rejects_bool_typed_value(tmp_path):
    """Codex review LOW-3 (2026-05-15): isinstance(True, (int, float))
    is True in Python; a producer-side bug routing a bool into a float
    field must surface as None on the reader side, not as 1.0/0.0."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-bool-float",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f-bool",
                "layer": "4",
                "z_score": True,  # WRONG: bool, not float
                "delta_auc": False,
                "evaluator_satisfied": False,
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
            }
        ],
    )
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].z_score is None
    assert records[0].delta_auc is None


def test_reader_time_window_boundary_equality(tmp_path):
    """Codex review LOW-4 (2026-05-15): time-window filter is documented
    as a CLOSED interval [since, until]; pin the equality boundaries."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-since",
        "2026-05-01T00:00:00Z",
        [
            {
                "feature": "at-since",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
            }
        ],
    )
    _write_sidecar(
        tmp_path,
        "exp-until",
        "2026-05-31T23:59:59Z",
        [
            {
                "feature": "at-until",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
            }
        ],
    )
    reader = SidecarReader(
        artifacts_dir=tmp_path,
        since=datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
        until=datetime(2026, 5, 31, 23, 59, 59, tzinfo=timezone.utc),
    )
    features = sorted(r.feature for r in reader.iter_verdict_records())
    assert features == ["at-since", "at-until"]


def test_extract_disagreements_filters_to_satisfied_false():
    from src.data.audit_sidecar_reader import (
        VerdictRecord,
        extract_disagreements,
    )

    sat = VerdictRecord(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f-sat",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence=None,
        z_score=2.0,
        p_value=0.05,
        delta_auc=0.04,
        evaluator_satisfied=True,
        evaluator_rationale_complete=True,
        evaluator_missed_considerations=[],
        evaluator_notes="",
        evaluator_model="haiku",
        raw_verdict={},
    )
    unsat = VerdictRecord(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f-unsat",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence=None,
        z_score=2.0,
        p_value=0.05,
        delta_auc=0.04,
        evaluator_satisfied=False,
        evaluator_rationale_complete=False,
        evaluator_missed_considerations=["temporal_filter"],
        evaluator_notes="thin",
        evaluator_model="haiku",
        raw_verdict={},
    )
    no_eval = VerdictRecord(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f-no-eval",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence=None,
        z_score=2.0,
        p_value=0.05,
        delta_auc=0.04,
        evaluator_satisfied=None,
        evaluator_rationale_complete=None,
        evaluator_missed_considerations=None,
        evaluator_notes=None,
        evaluator_model=None,
        raw_verdict={},
    )

    events = list(extract_disagreements([sat, unsat, no_eval]))
    assert len(events) == 1
    assert events[0].feature == "f-unsat"
    assert events[0].missed_considerations == ("temporal_filter",)


def test_dedup_disagreements_collapses_by_feature_name():
    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    e1 = DisagreementEvent(
        experiment_id="exp1",
        written_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        source_path=Path("/dev/null"),
        feature="f1",
        worker_severity="moderate",
        worker_remediation="keep_with_caveat",
        rationale_complete=False,
        missed_considerations=("temporal",),
        notes="first",
        evaluator_model="haiku",
    )
    e2 = DisagreementEvent(
        experiment_id="exp2",
        written_at=datetime(2026, 5, 10, tzinfo=timezone.utc),  # later
        source_path=Path("/dev/null"),
        feature="f1",  # same feature
        worker_severity="moderate",
        worker_remediation="keep_with_caveat",
        rationale_complete=False,
        missed_considerations=("pearl_arrows",),
        notes="second",
        evaluator_model="haiku",
    )
    e3 = DisagreementEvent(
        experiment_id="exp3",
        written_at=datetime(2026, 5, 5, tzinfo=timezone.utc),
        source_path=Path("/dev/null"),
        feature="f2",
        worker_severity="moderate",
        worker_remediation="keep_with_caveat",
        rationale_complete=False,
        missed_considerations=(),
        notes="other",
        evaluator_model="haiku",
    )

    deduped = list(dedup_disagreements([e1, e2, e3]))
    assert len(deduped) == 2
    by_feature = {e.feature: e for e in deduped}
    # For f1, keep the LATEST occurrence (e2) so the curated example
    # reflects the most recent rationale critique.
    assert by_feature["f1"].notes == "second"
    assert by_feature["f1"].experiment_id == "exp2"
    assert by_feature["f2"].feature == "f2"


def test_dedup_disagreements_is_deterministic_under_repeat():
    """Re-running over the same input must produce byte-identical output.
    Pinned because the JSON manifest in Task 7 is consumed by humans
    who compare runs across days; nondeterministic ordering would create
    spurious diffs."""
    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    events = [
        DisagreementEvent(
            experiment_id="e",
            written_at=datetime(2026, 5, i, tzinfo=timezone.utc),
            source_path=Path("/dev/null"),
            feature=f"f{i}",
            worker_severity="m",
            worker_remediation="k",
            rationale_complete=False,
            missed_considerations=(),
            notes="",
            evaluator_model="haiku",
        )
        for i in [3, 1, 2]  # intentionally unordered input
    ]
    run1 = [e.feature for e in dedup_disagreements(events)]
    run2 = [e.feature for e in dedup_disagreements(events)]
    assert run1 == run2
    # Deterministic order is by feature name ascending (a stable lex sort).
    assert run1 == ["f1", "f2", "f3"]


def test_dedup_equal_timestamp_uses_composite_tiebreaker():
    """Issue #234: when two events for the same feature share
    written_at AND their source_paths do not point at real files on disk
    (so mtime fallback is unavailable), dedup falls back to the documented
    stable lex tiebreaker on ``source_path`` then ``experiment_id``.

    Documented as the *last-resort* tiebreaker, not the primary one — see
    ``test_dedup_tiebreaker_prefers_newer_written_at_when_paths_differ``
    and ``test_dedup_tiebreaker_falls_back_to_path_lex_when_written_at_ties``
    for the recency-then-lex contract."""
    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    same_ts = datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
    e_low = DisagreementEvent(
        experiment_id="exp-a",
        written_at=same_ts,
        source_path=Path("/artifacts/exp-a/x.json"),
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("a",),
        notes="from-a",
        evaluator_model="haiku",
    )
    e_high = DisagreementEvent(
        experiment_id="exp-b",
        written_at=same_ts,
        source_path=Path("/artifacts/exp-b/x.json"),  # sorts > exp-a
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("b",),
        notes="from-b",
        evaluator_model="haiku",
    )
    # Insertion order should not matter — composite key sort wins.
    winner1 = list(dedup_disagreements([e_low, e_high]))[0]
    winner2 = list(dedup_disagreements([e_high, e_low]))[0]
    assert winner1.notes == winner2.notes == "from-b"
    assert winner1.experiment_id == "exp-b"


# -----------------------------------------------------------------------------
# Issue #234 — dedup tiebreaker by recency, not lex.
# -----------------------------------------------------------------------------


def test_dedup_tiebreaker_prefers_newer_written_at_when_paths_differ(tmp_path):
    """Issue #234: when two events for the same feature have *different*
    written_at, the newer one wins regardless of lex order of source_path.
    Anchor: paths like ``/artifacts/exp-1/x.json`` and
    ``/artifacts/exp-10/x.json`` — lex sort orders ``exp-10`` after
    ``exp-1`` but BEFORE ``exp-2``; the dedup contract is *recency*, so
    the newer written_at must win even when its path lex-sorts earlier."""
    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    # exp-10 (lex-mid: "exp-1" < "exp-10" < "exp-2") but newest written_at.
    e_old = DisagreementEvent(
        experiment_id="exp-2",
        written_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        source_path=tmp_path / "exp-2" / "x.json",
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("old",),
        notes="from-exp-2",
        evaluator_model="haiku",
    )
    e_new = DisagreementEvent(
        experiment_id="exp-10",
        written_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
        source_path=tmp_path / "exp-10" / "x.json",
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("new",),
        notes="from-exp-10",
        evaluator_model="haiku",
    )
    deduped = list(dedup_disagreements([e_old, e_new]))
    assert len(deduped) == 1
    assert deduped[0].notes == "from-exp-10"
    assert deduped[0].experiment_id == "exp-10"


def test_dedup_tiebreaker_falls_back_to_path_lex_when_written_at_ties(tmp_path):
    """Issue #234 A5: when ``written_at`` ties exactly AND mtime tiebreaker
    is unavailable (paths do not exist on disk OR mtimes are equal), the
    fallback is stable lex sort on ``source_path`` then ``experiment_id``.
    Documented as deterministic-only, not semantically meaningful."""
    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    same_ts = datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
    # Both source_paths point at nonexistent files: mtime fallback skipped.
    e_low = DisagreementEvent(
        experiment_id="exp-a",
        written_at=same_ts,
        source_path=Path("/nonexistent/exp-a/x.json"),
        feature="g",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("a",),
        notes="from-a",
        evaluator_model="haiku",
    )
    e_high = DisagreementEvent(
        experiment_id="exp-b",
        written_at=same_ts,
        source_path=Path("/nonexistent/exp-b/x.json"),
        feature="g",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("b",),
        notes="from-b",
        evaluator_model="haiku",
    )
    deduped = list(dedup_disagreements([e_low, e_high]))
    assert len(deduped) == 1
    assert deduped[0].experiment_id == "exp-b"  # lex-higher path wins


def test_dedup_tiebreaker_prefers_newer_mtime_when_written_at_ties(tmp_path):
    """Issue #234: when ``written_at`` ties to the second AND both source
    files exist on disk, the file with the *newer mtime* wins. mtime is
    the recency proxy when the payload timestamp lacks sub-second
    granularity (producer uses ``%Y%m%dT%H%M%SZ``)."""
    import os
    import time

    from src.data.audit_sidecar_reader import (
        DisagreementEvent,
        dedup_disagreements,
    )

    same_ts = datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
    older_path = tmp_path / "exp-z" / "older.json"
    newer_path = tmp_path / "exp-a" / "newer.json"
    older_path.parent.mkdir(parents=True)
    newer_path.parent.mkdir(parents=True)
    older_path.write_text("{}")
    # Force mtime of older to be in the past; newer is "now".
    past = time.time() - 3600
    os.utime(older_path, (past, past))
    newer_path.write_text("{}")
    # Sanity: lex order of strs puts exp-a < exp-z, so a naive lex
    # tiebreaker would pick exp-z (the older, lex-higher one).
    assert str(older_path) > str(newer_path)

    e_older = DisagreementEvent(
        experiment_id="exp-z",
        written_at=same_ts,
        source_path=older_path,
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("old",),
        notes="from-older",
        evaluator_model="haiku",
    )
    e_newer = DisagreementEvent(
        experiment_id="exp-a",
        written_at=same_ts,
        source_path=newer_path,
        feature="f",
        worker_severity="m",
        worker_remediation="k",
        rationale_complete=False,
        missed_considerations=("new",),
        notes="from-newer",
        evaluator_model="haiku",
    )
    deduped = list(dedup_disagreements([e_older, e_newer]))
    assert len(deduped) == 1
    assert deduped[0].notes == "from-newer", (
        "newer mtime must win over lex-higher path when written_at ties"
    )


# -----------------------------------------------------------------------------
# Issue #235 — producer schema_version + reader unknown-key warning.
# -----------------------------------------------------------------------------


def test_sidecar_payload_includes_schema_version_v1(tmp_path, monkeypatch):
    """Issue #235 A1: a fresh producer run writes ``schema_version`` at
    the top level of the payload, pinned to the current canonical
    major.minor.

    Bumped to ``"1.1"`` by Phase 1 of Issue #237 (additive
    ``role_attributions`` key), then to ``"1.2"`` by Issue #240 Stage 1
    (additive shadow promotion keys), then to ``"1.3"`` by Issue #240
    Stage 3 (additive soft-gate keys), then to ``"1.4"`` by Issue #501 /
    #240 (additive leakage × role cross-check key). Reader still pins
    MAJOR=1, so the forward-compat contract is unchanged."""
    from src.agents.ml_foundation.data_preparer.graph import (
        write_adaptive_verdicts_sidecar,
    )

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    state = {
        "experiment_id": "exp-schema",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [
            {
                "feature": "f",
                "layer": "4",
                "severity": "moderate",
                "evaluator_satisfied": False,
            }
        ],
    }
    path = write_adaptive_verdicts_sidecar(state)
    assert path is not None
    payload = json.loads(Path(path).read_text())
    assert payload.get("schema_version") == "1.4", (
        f"producer must emit top-level schema_version='1.4'; got {payload.get('schema_version')!r}"
    )


def test_reader_warns_on_missing_schema_version_legacy_fallback(tmp_path, caplog):
    """Issue #235 A2a: a sidecar with no ``schema_version`` (legacy v0
    sidecars from runs before 2026-05-15 + this fix) must be parsed
    successfully, with a WARN noting the legacy fallback."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-legacy",
        "2026-04-15T10:00:00Z",
        [
            {
                "feature": "f-legacy",
                "layer": "4",
                "evaluator_satisfied": False,
                "evaluator_rationale_complete": False,
                "evaluator_missed_considerations": [],
                "evaluator_notes": "",
                "evaluator_model": "haiku",
            }
        ],
    )
    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].feature == "f-legacy"
    assert any(
        "schema_version" in rec.message and "legacy" in rec.message.lower()
        for rec in caplog.records
    ), f"expected legacy-fallback WARN; got: {[r.message for r in caplog.records]}"


def test_reader_warns_on_unknown_schema_version_major(tmp_path, caplog):
    """Issue #235 A2b: a sidecar carrying a ``schema_version`` whose major
    bumps past the reader's expected major must WARN with both versions
    surfaced. Reader still parses the known keys."""
    from src.data.audit_sidecar_reader import SidecarReader

    sub = tmp_path / "exp-future"
    sub.mkdir(parents=True)
    payload = {
        "experiment_id": "exp-future",
        "schema_version": "2.0",  # future major
        "data_source": "synthetic",
        "written_at": "2026-05-15T10:00:00Z",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [
            {
                "feature": "f-fut",
                "layer": "4",
                "evaluator_satisfied": False,
            }
        ],
    }
    out = sub / "adaptive_verdicts_20260515T100000Z.json"
    out.write_text(json.dumps(payload))

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert len(records) == 1
    assert records[0].feature == "f-fut"
    # codex MED-2 (2026-05-15): pin BOTH versions in the WARN — the payload's
    # ``"2.0"`` AND the reader's expected current version. Without this assertion
    # the test would still pass if the WARN stopped naming the reader's
    # expected version, which is the actionable half of the message.
    # Issue #501 / #240: reader's current version bumped to "1.4"
    # (still MAJOR=1).
    matches = [
        rec
        for rec in caplog.records
        if "schema_version" in rec.message and "2.0" in rec.message and "1.4" in rec.message
    ]
    assert matches, (
        "expected unknown-major WARN naming both '2.0' (payload) and '1.4' (reader); "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_reader_warns_on_unknown_keys_once_per_file(tmp_path, caplog):
    """Issue #235 A3: when a verdict carries keys the reader doesn't
    recognize, log a WARN *once per file*, not per-record. Bounds log
    spam when a future producer adds a new field across N verdicts."""
    from src.data.audit_sidecar_reader import SidecarReader

    sub = tmp_path / "exp-fwd"
    sub.mkdir(parents=True)
    payload = {
        "experiment_id": "exp-fwd",
        "schema_version": "1.0",
        "data_source": "synthetic",
        "written_at": "2026-05-15T10:00:00Z",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [
            {
                "feature": "f1",
                "layer": "4",
                "future_field": 42,
                "another_new_field": "blah",
                "evaluator_satisfied": False,
            },
            {
                "feature": "f2",
                "layer": "4",
                "future_field": 99,  # SAME unknown key on a 2nd record
                "evaluator_satisfied": False,
            },
        ],
    }
    out = sub / "adaptive_verdicts_20260515T100000Z.json"
    out.write_text(json.dumps(payload))

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert len(records) == 2
    unknown_warns = [
        rec
        for rec in caplog.records
        if "unknown" in rec.message.lower() and "future_field" in rec.message
    ]
    # Exactly ONE warn for this file, even though future_field appears in 2 records.
    assert len(unknown_warns) == 1, (
        f"expected exactly one unknown-key WARN per file; got {len(unknown_warns)}: "
        f"{[r.message for r in unknown_warns]}"
    )
    # codex LOW-1 (2026-05-15): both unknown keys must surface in the single
    # WARN; without this assertion an implementation that warns only the
    # first unknown key would pass.
    assert "another_new_field" in unknown_warns[0].message, (
        f"second unknown key 'another_new_field' missing from WARN: {unknown_warns[0].message}"
    )


def test_reader_unknown_key_warn_fires_on_lazy_consumption(tmp_path, caplog):
    """codex MED-1 (2026-05-15): the unknown-key WARN must be emitted at
    parse time (pre-scan), not after the generator's yield loop. A caller
    that breaks out early or calls ``next()`` once must still see the WARN.
    Without the pre-scan, a yield-suffix WARN would silently miss this
    case in production."""
    from src.data.audit_sidecar_reader import SidecarReader

    sub = tmp_path / "exp-lazy"
    sub.mkdir(parents=True)
    payload = {
        "experiment_id": "exp-lazy",
        "schema_version": "1.0",
        "data_source": "synthetic",
        "written_at": "2026-05-15T10:00:00Z",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [
            {
                "feature": "f1",
                "layer": "4",
                "lazy_unknown_field": "x",
                "evaluator_satisfied": False,
            },
            {
                "feature": "f2",
                "layer": "4",
                "evaluator_satisfied": False,
            },
        ],
    }
    out = sub / "adaptive_verdicts_20260515T100000Z.json"
    out.write_text(json.dumps(payload))

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        # Pull exactly ONE record and stop — generator never reaches the
        # second verdict or any post-loop logic.
        first = next(reader.iter_verdict_records())
    assert first.feature == "f1"
    matches = [
        rec
        for rec in caplog.records
        if "unknown" in rec.message.lower() and "lazy_unknown_field" in rec.message
    ]
    assert matches, (
        "unknown-key WARN must fire on lazy consumption (pre-scan); "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_reader_tolerates_non_list_adaptive_verdicts(tmp_path, caplog):
    """codex pass-2 MED-1 (2026-05-15): a malformed or forward-drifted
    sidecar where ``adaptive_verdicts`` is ``null`` / a scalar / a dict
    must NOT crash ``iter_verdict_records``. Reader normalizes to ``[]``
    after a WARN so one bad file does not take down the curation CLI."""
    from src.data.audit_sidecar_reader import SidecarReader

    sub = tmp_path / "exp-broken"
    sub.mkdir(parents=True)
    payload = {
        "experiment_id": "exp-broken",
        "schema_version": "1.0",
        "data_source": "synthetic",
        "written_at": "2026-05-15T10:00:00Z",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": None,  # WRONG: producer would emit [] but a
        # forward-drift or hand-edit can land here. Reader must not crash.
    }
    out = sub / "adaptive_verdicts_20260515T100000Z.json"
    out.write_text(json.dumps(payload))

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        # If iter_verdict_records crashed, this line would raise TypeError.
        records = list(reader.iter_verdict_records())
    assert records == []
    matches = [
        rec
        for rec in caplog.records
        if "non-list" in rec.message.lower() and "adaptive_verdicts" in rec.message
    ]
    assert matches, f"expected non-list WARN; got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Issue #240 Stage 1 — shadow-column surfacing on VerdictRecord.
#
# The producer (``_ensemble_to_legacy_dict``) emits three nullable shadow
# keys per verdict. The mirror's dedicated typed columns can only be
# populated if the reader surfaces those keys on ``VerdictRecord`` (the
# mirror reads ``r.would_promote_severity`` etc.). These tests pin that
# surfacing + the schema-tolerant absence path. Design ref:
# ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 1.
# ---------------------------------------------------------------------------


def test_reader_surfaces_shadow_columns_when_present(tmp_path):
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-shadow",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f_shadow",
                "layer": "4",
                "severity": "moderate",
                "evaluator_satisfied": False,
                "evaluator_missed_considerations": ["temporal_filter", "pearl_arrows"],
                "evaluator_rationale_complete": False,
                "evaluator_model": "haiku",
                # Issue #240 Stage-1 shadow keys.
                "would_promote_severity": "high",
                "would_flag_for_review": True,
                "rationale_incomplete_flag": True,
            }
        ],
    )

    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    r = records[0]
    assert r.would_promote_severity == "high"
    assert r.would_flag_for_review is True
    assert r.rationale_incomplete_flag is True


def test_reader_shadow_columns_none_when_absent(tmp_path):
    """Pre-#240 sidecars carry no shadow keys → all three surface as None
    (the same schema-tolerant pattern as the evaluator-audit fields)."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-pre240",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f_legacy",
                "layer": "4",
                "severity": "moderate",
                "evaluator_satisfied": True,
                "evaluator_model": "haiku",
            }
        ],
    )

    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    r = records[0]
    assert r.would_promote_severity is None
    assert r.would_flag_for_review is None
    assert r.rationale_incomplete_flag is None


def test_reader_does_not_warn_on_shadow_keys_as_unknown(tmp_path, caplog):
    """The three shadow keys are registered in ``_KNOWN_VERDICT_KEYS`` so
    they do not trip the per-file 'unknown verdict keys' WARN."""
    from src.data.audit_sidecar_reader import SidecarReader

    _write_sidecar(
        tmp_path,
        "exp-known",
        "2026-05-15T10:00:00Z",
        [
            {
                "feature": "f_known",
                "severity": "moderate",
                "would_promote_severity": "high",
                "would_flag_for_review": True,
                "rationale_incomplete_flag": True,
            }
        ],
    )

    reader = SidecarReader(artifacts_dir=tmp_path)
    with caplog.at_level("WARNING"):
        records = list(reader.iter_verdict_records())
    assert len(records) == 1
    unknown_warns = [
        rec
        for rec in caplog.records
        if "unknown" in rec.message.lower()
        and (
            "would_promote_severity" in rec.message
            or "would_flag_for_review" in rec.message
            or "rationale_incomplete_flag" in rec.message
        )
    ]
    assert not unknown_warns, (
        f"shadow keys must be registered as known; got unknown-key WARNs: "
        f"{[r.message for r in unknown_warns]}"
    )


# ---------------------------------------------------------------------------
# Issue #240 Stage 2 — surface the shadow ``would_promote_severity`` + the
# driving R1 input signals onto ``DisagreementEvent`` so the curation flow
# (markdown report + CLI filter) can present promotion candidates. Design ref:
# ``docs/plans/240-audit-evaluator-gate-promotion.md`` §3 Stage 2 Mechanism.
# ---------------------------------------------------------------------------


def test_extract_disagreements_surfaces_promotion_fields():
    """A satisfied=False record carrying the Stage-1 ``would_promote_severity``
    shadow field must surface that field AND the R1 driving signals
    (worker_severity, evaluator_satisfied, missed-considerations count) on the
    emitted ``DisagreementEvent``."""
    from src.data.audit_sidecar_reader import (
        VerdictRecord,
        extract_disagreements,
    )

    rec = VerdictRecord(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f-promote",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence=None,
        z_score=2.0,
        p_value=0.05,
        delta_auc=0.04,
        evaluator_satisfied=False,
        evaluator_rationale_complete=False,
        evaluator_missed_considerations=["temporal_filter", "pearl_arrows"],
        evaluator_notes="thin",
        evaluator_model="haiku",
        raw_verdict={},
        would_promote_severity="high",
    )

    events = list(extract_disagreements([rec]))
    assert len(events) == 1
    ev = events[0]
    assert ev.would_promote_severity == "high"
    # Driving R1 signals are surfaced for the curation reviewer.
    assert ev.worker_severity == "moderate"
    assert ev.evaluator_satisfied is False
    assert ev.missed_considerations == ("temporal_filter", "pearl_arrows")


def test_extract_disagreements_promotion_none_when_rule_did_not_fire():
    """A satisfied=False record where R1 did NOT fire (no shadow field) still
    yields a DisagreementEvent, but with ``would_promote_severity=None``."""
    from src.data.audit_sidecar_reader import (
        VerdictRecord,
        extract_disagreements,
    )

    rec = VerdictRecord(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f-no-promote",
        layer="4",
        severity="moderate",
        remediation="keep_with_caveat",
        evidence=None,
        z_score=2.0,
        p_value=0.05,
        delta_auc=0.04,
        evaluator_satisfied=False,
        evaluator_rationale_complete=False,
        evaluator_missed_considerations=[],
        evaluator_notes="thin",
        evaluator_model="haiku",
        raw_verdict={},
        # would_promote_severity defaults to None (R1 did not fire).
    )

    events = list(extract_disagreements([rec]))
    assert len(events) == 1
    ev = events[0]
    assert ev.would_promote_severity is None
    assert ev.evaluator_satisfied is False


def test_disagreement_event_promotion_fields_default_to_none():
    """The new promotion fields are additive/nullable with defaults so
    existing keyword-only DisagreementEvent constructions keep working."""
    from src.data.audit_sidecar_reader import DisagreementEvent

    ev = DisagreementEvent(
        experiment_id="e",
        written_at=datetime.now(timezone.utc),
        source_path=Path("/dev/null"),
        feature="f",
        worker_severity="moderate",
        worker_remediation="keep_with_caveat",
        rationale_complete=False,
        missed_considerations=("temporal_filter",),
        notes="n",
        evaluator_model="haiku",
    )
    assert ev.would_promote_severity is None
    assert ev.evaluator_satisfied is None
