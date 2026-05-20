"""Tests for the offline PHI/PII audit harness CLI (#391 security box 4).

Script under test: ``scripts/audit_phi_in_crystal_narratives.py``

The harness loads CrystalDigest rows and LLMCrystalNarrativeAudit rows
from Postgres (via ``psycopg``), runs :func:`src.security.phi_scanner.scan_text`
on ``key_finding`` (crystal narrative) + the audit's prompt text fields,
emits a JSON report to stdout, and exits non-zero if any PHI matches
are found.

These unit tests exercise the harness through its module entrypoint
(``audit_records`` and ``main``) with in-memory record lists so we don't
require a live Postgres instance. A separate integration test would
wire it against a real DB if needed.

The script lives under ``scripts/`` (not ``src/``) so we add ``scripts/``
to ``sys.path`` at test-time. This mirrors the existing
``tests/unit/test_scripts/`` convention.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Import the script-under-test as a module
# ---------------------------------------------------------------------------


def _load_audit_module() -> ModuleType:
    """Load ``scripts/audit_phi_in_crystal_narratives.py`` as a module.

    Done via ``importlib.util`` instead of a plain import so the script
    can keep its filename (with ``.py`` suffix and module-flavored
    name) without forcing the test to be inside ``scripts/``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "audit_phi_in_crystal_narratives.py"
    spec = importlib.util.spec_from_file_location("audit_phi_in_crystal_narratives", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_phi_in_crystal_narratives"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# audit_records: pure-function unit tests (no DB)
# ---------------------------------------------------------------------------


def _record_clean(insight_id: str = "i-1") -> Dict[str, Any]:
    """Build a clean (PHI-free) crystal+audit record."""
    return {
        "insight_id": insight_id,
        "key_finding": "Strong positive effect with adequate cohort size.",
        "narrative": "Clean narrative without identifiers.",
        "audit_input_prompt": "Summarize the finding for the executive dashboard.",
    }


def _record_with_ssn(insight_id: str = "i-bad-1") -> Dict[str, Any]:
    return {
        "insight_id": insight_id,
        "key_finding": "Patient ssn 555-12-3456 enrolled and showed response.",
        "narrative": "Cohort-level summary.",
        "audit_input_prompt": "Generate a key finding.",
    }


def _record_with_dob_in_prompt(insight_id: str = "i-bad-2") -> Dict[str, Any]:
    return {
        "insight_id": insight_id,
        "key_finding": "Clean output.",
        "narrative": "Clean narrative.",
        "audit_input_prompt": "Subject born 03/15/1987 cohort follow-up needed.",
    }


# ---------------------------------------------------------------------------


def test_audit_records_clean_dataset_returns_empty() -> None:
    """A dataset with no PHI matches returns an empty findings list."""
    module = _load_audit_module()
    records: List[Dict[str, Any]] = [_record_clean(), _record_clean("i-2")]
    report = module.audit_records(records)
    assert report["findings"] == []
    assert report["records_scanned"] == 2
    assert report["phi_match_count"] == 0


def test_audit_records_phi_in_key_finding_is_surfaced() -> None:
    """A PHI hit in ``key_finding`` is reported with the right field."""
    module = _load_audit_module()
    records = [_record_with_ssn()]
    report = module.audit_records(records)
    assert report["records_scanned"] == 1
    assert report["phi_match_count"] >= 1
    finding = report["findings"][0]
    assert finding["insight_id"] == "i-bad-1"
    assert finding["field"] == "key_finding"
    # The match details preserve pattern_name + the matched substring
    assert any(m["pattern_name"] == "ssn" for m in finding["matches"])


def test_audit_records_phi_in_audit_prompt_is_surfaced() -> None:
    """A PHI hit in ``audit_input_prompt`` is reported with field=prompt."""
    module = _load_audit_module()
    records = [_record_with_dob_in_prompt()]
    report = module.audit_records(records)
    assert report["records_scanned"] == 1
    assert report["phi_match_count"] >= 1
    finding = report["findings"][0]
    assert finding["insight_id"] == "i-bad-2"
    assert finding["field"] == "audit_input_prompt"
    assert any(m["pattern_name"] == "dob" for m in finding["matches"])


def test_audit_records_multiple_fields_in_one_record() -> None:
    """One record with PHI in multiple fields surfaces all hits."""
    module = _load_audit_module()
    record = {
        "insight_id": "i-bad-multi",
        "key_finding": "ssn 555-12-3456 detected",
        "narrative": "DOB 03/15/1987 noted",
        "audit_input_prompt": "phone (415) 555-1212",
    }
    report = module.audit_records([record])
    fields = sorted({f["field"] for f in report["findings"]})
    assert "key_finding" in fields
    assert "narrative" in fields
    assert "audit_input_prompt" in fields


# ---------------------------------------------------------------------------
# main / exit code contract
# ---------------------------------------------------------------------------


def test_main_exits_zero_on_clean_dataset(monkeypatch, capsys) -> None:
    """``main(records=...)`` with a clean dataset returns exit code 0 and
    emits valid JSON to stdout."""
    module = _load_audit_module()
    records = [_record_clean()]
    rc = module.main(records=records)
    assert rc == 0
    captured = capsys.readouterr()
    # JSON output must parse cleanly
    payload = json.loads(captured.out)
    assert payload["records_scanned"] == 1
    assert payload["phi_match_count"] == 0
    assert payload["findings"] == []


def test_main_exits_nonzero_when_phi_present(monkeypatch, capsys) -> None:
    """``main(records=...)`` with PHI returns exit code != 0 and JSON output."""
    module = _load_audit_module()
    records = [_record_with_ssn()]
    rc = module.main(records=records)
    assert rc != 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["records_scanned"] == 1
    assert payload["phi_match_count"] >= 1
    assert payload["findings"][0]["insight_id"] == "i-bad-1"


def test_main_output_is_valid_json_with_required_keys() -> None:
    """JSON shape contract: ``records_scanned``, ``phi_match_count``,
    ``findings``."""
    module = _load_audit_module()
    report = module.audit_records([_record_clean()])
    # Must serialize via json.dumps without errors
    serialized = json.dumps(report)
    re_parsed = json.loads(serialized)
    assert {"records_scanned", "phi_match_count", "findings"}.issubset(set(re_parsed.keys()))
