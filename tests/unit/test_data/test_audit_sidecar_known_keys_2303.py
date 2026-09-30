"""#2303: the unknown-verdict-key WARN must not fire on keys the reader parses.

The four evaluator telemetry keys (#241) are VerdictRecord fields that
``_build_record`` reads from the raw verdict, yet they were missing from
``_KNOWN_VERDICT_KEYS`` — so every real sidecar (the writer emits them on
every verdict) logged a false 'unknown verdict key(s)' WARNING.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from src.data.audit_sidecar_reader import SidecarReader, VerdictRecord

# VerdictRecord fields that are NOT read from the per-verdict dict: they come
# from the sidecar envelope, the file path, or the file-level role map.
_RUN_LEVEL_FIELDS = {
    "experiment_id",
    "written_at",
    "source_path",
    "raw_verdict",
    "role_attribution",
    "audit_workflow_id",
}

_TELEMETRY = {
    "evaluator_latency_ms": 812.5,
    "evaluator_input_tokens": 1234,
    "evaluator_output_tokens": 210,
    "evaluator_cost_usd": 0.0041,
}


def _write_sidecar(directory: Path, verdicts: list[dict]) -> Path:
    sub = directory / "exp-2303"
    sub.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.10",
        "experiment_id": "exp-2303",
        "data_source": "synthetic",
        "written_at": "2026-09-24T00:52:02Z",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [v["feature"] for v in verdicts],
        "adaptive_verdicts": verdicts,
    }
    out = sub / "adaptive_verdicts_20260924T005202Z.json"
    out.write_text(json.dumps(payload))
    return out


def _unknown_key_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "unknown verdict key" in r.getMessage()]


def test_evaluator_telemetry_keys_do_not_warn_and_are_parsed(tmp_path, caplog):
    _write_sidecar(
        tmp_path,
        [{"feature": "f1", "severity": "moderate", "evaluator_satisfied": True, **_TELEMETRY}],
    )
    with caplog.at_level("WARNING"):
        records = list(SidecarReader(artifacts_dir=tmp_path).iter_verdict_records())

    assert _unknown_key_warnings(caplog) == []
    (r,) = records
    assert r.evaluator_latency_ms == 812.5
    assert r.evaluator_input_tokens == 1234
    assert r.evaluator_output_tokens == 210
    assert r.evaluator_cost_usd == 0.0041


def test_truly_unknown_key_still_warns_naming_only_that_key(tmp_path, caplog):
    _write_sidecar(
        tmp_path,
        [{"feature": "f1", "severity": "moderate", "brand_new_key": 1, **_TELEMETRY}],
    )
    with caplog.at_level("WARNING"):
        list(SidecarReader(artifacts_dir=tmp_path).iter_verdict_records())

    warns = _unknown_key_warnings(caplog)
    assert len(warns) == 1
    assert "['brand_new_key']" in warns[0]


def test_every_verdict_field_parsed_from_raw_is_a_known_key(tmp_path, caplog):
    """Pin: the allow-list cannot drift behind VerdictRecord again. A verdict
    carrying every raw-sourced VerdictRecord field must produce no WARN."""
    raw_fields = {f.name for f in fields(VerdictRecord)} - _RUN_LEVEL_FIELDS
    verdict: dict[str, object] = dict.fromkeys(raw_fields)
    verdict["feature"] = "f1"
    _write_sidecar(tmp_path, [verdict])
    with caplog.at_level("WARNING"):
        list(SidecarReader(artifacts_dir=tmp_path).iter_verdict_records())

    assert _unknown_key_warnings(caplog) == []
