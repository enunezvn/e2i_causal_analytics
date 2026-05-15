"""End-to-end integration test for scripts/curate_compile_set_candidates.py
(Plan .claude/plans/layer4_evaluator_audit_consumer.md)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_sidecar(artifacts_dir: Path, experiment_id: str,
                   written_at: str, verdicts: list[dict]) -> Path:
    sub = artifacts_dir / experiment_id
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


def test_cli_end_to_end_produces_markdown_and_manifest(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    output_dir = tmp_path / "candidates"

    # Two disagreements (same feature → dedup keeps latest), one agreement,
    # one record from the pre-evaluator schema (no eval keys at all).
    _write_sidecar(artifacts_dir, "exp-1", "2026-05-10T10:00:00Z", [
        {"feature": "ondansetron_fills_180d", "layer": "4",
         "severity": "moderate", "remediation": "keep_with_caveat",
         "evaluator_satisfied": False,
         "evaluator_rationale_complete": False,
         "evaluator_missed_considerations": ["temporal_filter"],
         "evaluator_notes": "first critique",
         "evaluator_model": "anthropic/claude-haiku-4-5-20251001"},
    ])
    _write_sidecar(artifacts_dir, "exp-2", "2026-05-12T10:00:00Z", [
        {"feature": "ondansetron_fills_180d", "layer": "4",
         "severity": "moderate", "remediation": "keep_with_caveat",
         "evaluator_satisfied": False,
         "evaluator_rationale_complete": False,
         "evaluator_missed_considerations": ["pearl_arrows"],
         "evaluator_notes": "second critique",
         "evaluator_model": "anthropic/claude-haiku-4-5-20251001"},
        {"feature": "metformin_fills_90d", "layer": "4",
         "severity": "moderate", "remediation": "keep_with_caveat",
         "evaluator_satisfied": True,
         "evaluator_rationale_complete": True,
         "evaluator_missed_considerations": [],
         "evaluator_notes": "",
         "evaluator_model": "anthropic/claude-haiku-4-5-20251001"},
    ])
    _write_sidecar(artifacts_dir, "exp-old", "2026-04-01T10:00:00Z", [
        {"feature": "old_feature", "layer": "4", "severity": "moderate"},
    ])

    cli = Path(__file__).resolve().parents[2] / "scripts" / \
        "curate_compile_set_candidates.py"
    result = subprocess.run(
        [sys.executable, str(cli),
         "--artifacts-dir", str(artifacts_dir),
         "--output-dir", str(output_dir)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, \
        f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r}"

    md_files = list(output_dir.glob("compile_set_candidates_*.md"))
    json_files = list(output_dir.glob("compile_set_candidates_*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1

    md = md_files[0].read_text()
    # Only the dedup-winner (latest critique, "second critique") appears.
    assert "ondansetron_fills_180d" in md
    assert "second critique" in md
    # The earlier critique was deduped out — must not appear.
    assert "first critique" not in md
    # The agreement and the pre-evaluator-schema row must not appear.
    assert "metformin_fills_90d" not in md
    assert "old_feature" not in md
    # Accept/reject checkboxes present.
    assert "[ ] accept" in md
    assert "[ ] reject" in md

    manifest = json.loads(json_files[0].read_text())
    assert len(manifest["candidates"]) == 1
    c = manifest["candidates"][0]
    assert c["feature_name"] == "ondansetron_fills_180d"
    assert c["expected_causal_role"] is None
    assert c["expected_remediation"] is None
    assert c["evaluator_audit"]["satisfied"] is False
    assert c["evaluator_audit"]["missed_considerations"] == ["pearl_arrows"]
    assert c["source_run_id"] == "exp-2"


def test_cli_dedup_is_deterministic_across_repeated_runs(tmp_path):
    """Acceptance criterion #6: re-running over identical input must
    produce byte-identical manifest (modulo --output-dir timestamp)."""
    artifacts_dir = tmp_path / "artifacts"
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    for i in [3, 1, 2]:  # unordered features
        _write_sidecar(artifacts_dir, f"exp-{i}", f"2026-05-1{i}T10:00:00Z", [
            {"feature": f"f{i}", "layer": "4",
             "severity": "moderate", "remediation": "keep_with_caveat",
             "evaluator_satisfied": False,
             "evaluator_rationale_complete": False,
             "evaluator_missed_considerations": [],
             "evaluator_notes": "x",
             "evaluator_model": "haiku"},
        ])

    cli = Path(__file__).resolve().parents[2] / "scripts" / \
        "curate_compile_set_candidates.py"

    def _run(out: Path) -> dict:
        subprocess.run(
            [sys.executable, str(cli),
             "--artifacts-dir", str(artifacts_dir),
             "--output-dir", str(out)],
            check=True, capture_output=True, text=True,
        )
        json_file = next(out.glob("compile_set_candidates_*.json"))
        return json.loads(json_file.read_text())

    m_a = _run(out_a)
    m_b = _run(out_b)
    # Generated-at timestamp will differ; strip it before comparison.
    m_a.pop("generated_at")
    m_b.pop("generated_at")
    assert m_a == m_b
    # And the order is ascending feature name (deterministic).
    assert [c["feature_name"] for c in m_a["candidates"]] == ["f1", "f2", "f3"]
