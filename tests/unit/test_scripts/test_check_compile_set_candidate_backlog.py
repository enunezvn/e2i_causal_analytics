"""Tests for ``scripts/check_compile_set_candidate_backlog.py``.

Phase 4.5 of plan ``.claude/plans/layer4_evaluator_audit_consumer.md``:
auto-trigger surface for compile_causal_role_classifier (issue #236).

Stakeholder choices (from issue #236 body):
  - Threshold N = 5 (matches "≥5" forcing function in issue acceptance).
  - "Accepted" candidate = JSON manifest entry where every key in
    ``_REQUIRED_FILL_INS`` (``expected_causal_role``,
    ``expected_remediation``, ``derivation_pseudocode``,
    ``dataset_context``) is non-null.
  - Acceptance gate: backlog counted only over candidates produced
    AFTER the compiled artifact's mtime (rationale: changes already
    folded into ``build_compile_set()`` and recompiled SHOULD NOT
    re-count against the next backlog).

Test surface:
  * Pure-logic counting (``count_accepted_candidates``):
      - No candidates dir -> 0
      - Empty manifests -> 0
      - All 4 fill-ins null -> 0 (proposal, not accepted)
      - All 4 fill-ins non-null -> counted
      - Mixed accepted + unaccepted -> only accepted counted
      - Candidate manifest modified BEFORE artifact mtime -> excluded
      - Candidate manifest modified AFTER artifact mtime -> included
      - Missing artifact (no prior compile) -> count all accepted
      - Malformed manifest -> skipped with warning, no crash
      - Multiple manifests with same feature -> double-counted across
        files (we do NOT de-dup at this layer — the curator can rerun
        ``curate_compile_set_candidates.py`` and ship a fresh window
        that supersedes the old).
  * CLI exit codes:
      - Default: always exit 0 (informational).
      - ``--strict`` + backlog < threshold: exit 0, no error
      - ``--strict`` + backlog >= threshold: exit 0 + READY signal on stdout
      - Both modes print backlog count + threshold to stdout.
  * Threshold default = 5 (DEFAULT_THRESHOLD constant).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_compile_set_candidate_backlog as backlog  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(
    path: Path,
    candidates: list[dict],
    generated_at: str = "2026-05-15T12:00:00+00:00",
) -> Path:
    """Write a curate_compile_set_candidates-style JSON manifest."""
    payload = {
        "generated_at": generated_at,
        "schema_version": 1,
        "required_fill_ins": list(backlog.REQUIRED_FILL_INS),
        "candidates": candidates,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _accepted_candidate(feature: str) -> dict:
    """A candidate row with every required fill-in completed."""
    return {
        "feature_name": feature,
        "expected_causal_role": "confounder",
        "expected_remediation": "keep_with_caveat",
        "derivation_pseudocode": "select x from y where z",
        "dataset_context": "CSU target ON_180",
        "evaluator_audit": {
            "satisfied": False,
            "rationale_complete": False,
            "missed_considerations": ["temporal"],
            "notes": "n",
            "model": "anthropic/claude-haiku-4-5-20251001",
        },
        "source_run_id": "exp-1",
        "source_written_at": "2026-05-10T00:00:00+00:00",
        "source_path": "/path/sidecar.json",
    }


def _unaccepted_candidate(feature: str) -> dict:
    """A candidate row straight out of curate_compile_set_candidates."""
    return {
        "feature_name": feature,
        "expected_causal_role": None,
        "expected_remediation": None,
        "derivation_pseudocode": None,
        "dataset_context": None,
        "evaluator_audit": {
            "satisfied": False,
            "rationale_complete": False,
            "missed_considerations": ["temporal"],
            "notes": "n",
            "model": "anthropic/claude-haiku-4-5-20251001",
        },
        "source_run_id": "exp-1",
        "source_written_at": "2026-05-10T00:00:00+00:00",
        "source_path": "/path/sidecar.json",
    }


# ---------------------------------------------------------------------------
# count_accepted_candidates — pure logic
# ---------------------------------------------------------------------------


def test_missing_candidates_dir_returns_zero(tmp_path):
    """A nonexistent candidates dir is a normal pre-curation state."""
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    result = backlog.count_accepted_candidates(
        candidates_dir=tmp_path / "candidates_missing",
        compiled_artifact_path=artifact,
    )
    assert result.count == 0
    assert result.accepted_features == []


def test_empty_candidates_dir_returns_zero(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 0


def test_only_unaccepted_candidates_returns_zero(tmp_path):
    """Manifests fresh out of curate_compile_set_candidates have all 4
    fill-ins null -> nothing accepted yet."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_20260515T120000000000Z.json",
        [_unaccepted_candidate("f1"), _unaccepted_candidate("f2")],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    # Force artifact older than manifest
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 0


def test_fully_accepted_candidate_counted(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate("ondansetron_fills_180d")],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 1
    assert result.accepted_features == ["ondansetron_fills_180d"]


def test_mixed_accepted_and_unaccepted_only_accepted_counted(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [
            _accepted_candidate("f_ok_1"),
            _unaccepted_candidate("f_pending"),
            _accepted_candidate("f_ok_2"),
        ],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 2
    assert sorted(result.accepted_features) == ["f_ok_1", "f_ok_2"]


def test_partial_fillin_not_counted(tmp_path):
    """If derivation_pseudocode is null but the 3 ``expected_*`` are
    filled, the candidate is malformed (caught at compile-time anyway —
    formatter docstring) and MUST NOT count as backlog."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    partial = _accepted_candidate("f_partial")
    partial["derivation_pseudocode"] = None
    _write_manifest(cdir / "compile_set_candidates_a.json", [partial])
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 0


def test_manifest_modified_before_artifact_is_excluded(tmp_path):
    """A manifest stamped from before the most recent compile must NOT
    be re-counted — those accepted candidates were already folded in."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    stale = cdir / "stale.json"
    _write_manifest(stale, [_accepted_candidate("f_already_merged")])
    # Set stale mtime well in the past
    stale_mtime = stale.stat().st_mtime - 3600
    os.utime(stale, (stale.stat().st_atime, stale_mtime))

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    # Artifact is fresher than stale manifest
    os.utime(artifact, (artifact.stat().st_atime, stale_mtime + 1800))

    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 0


def test_manifest_modified_after_artifact_is_included(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    artifact_mtime = artifact.stat().st_mtime

    fresh = cdir / "fresh.json"
    _write_manifest(fresh, [_accepted_candidate("f_new")])
    os.utime(fresh, (fresh.stat().st_atime, artifact_mtime + 1800))

    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 1


def test_no_compiled_artifact_counts_all_accepted(tmp_path):
    """If the artifact doesn't exist yet (cold start), every accepted
    candidate is backlog."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate("f_ok_1"), _accepted_candidate("f_ok_2")],
    )
    nonexistent_artifact = tmp_path / "does_not_exist" / "artifact.json"
    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=nonexistent_artifact,
    )
    assert result.count == 2


def test_malformed_manifest_skipped_no_crash(tmp_path):
    """A corrupt JSON file in the candidates dir must NOT take the CLI
    down — it's the operator's audit trail, not a control channel."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    (cdir / "garbage.json").write_text("{not valid json")
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate("f_ok_1")],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 1
    # The bad path is recorded for surfacing
    assert any("garbage.json" in str(p) for p in result.malformed_paths)


def test_non_json_files_ignored(tmp_path):
    """The candidates dir also has markdown reports — those must be
    ignored, not parsed as JSON."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    (cdir / "compile_set_candidates_a.md").write_text("# random markdown")
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate("f_ok_1")],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    result = backlog.count_accepted_candidates(
        candidates_dir=cdir,
        compiled_artifact_path=artifact,
    )
    assert result.count == 1
    assert result.malformed_paths == []


# ---------------------------------------------------------------------------
# Threshold constant
# ---------------------------------------------------------------------------


def test_default_threshold_is_five():
    """Issue #236 acceptance: 'backlog crosses threshold' default is 5."""
    assert backlog.DEFAULT_THRESHOLD == 5


def test_required_fill_ins_matches_formatter():
    """REQUIRED_FILL_INS must stay in lockstep with the formatter's
    single-source-of-truth tuple. If the formatter adds a 5th required
    field, this test trips so backlog counting stays consistent."""
    from src.data.audit_candidate_formatter import _REQUIRED_FILL_INS

    assert backlog.REQUIRED_FILL_INS == tuple(_REQUIRED_FILL_INS)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


CLI = Path(__file__).resolve().parents[3] / "scripts" / "check_compile_set_candidate_backlog.py"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
    )


def test_cli_below_threshold_exits_zero_and_prints_backlog(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate(f"f_{i}") for i in range(3)],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "backlog=3" in proc.stdout.lower() or "backlog: 3" in proc.stdout.lower()


def test_cli_at_threshold_emits_ready_signal(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate(f"f_{i}") for i in range(5)],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    # Issue #236: "print a clear ready-to-compile signal when backlog ≥ 5"
    assert "ready" in proc.stdout.lower()


def test_cli_above_threshold_emits_ready_signal(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate(f"f_{i}") for i in range(7)],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "ready" in proc.stdout.lower()
    assert "7" in proc.stdout


def test_cli_strict_mode_no_backlog_exits_one(tmp_path):
    """Strict-mode is what the COMPILE pre-flight wires up: refuse the
    compile when there's no new accepted backlog (forcing function — no
    silent re-compile of the same artifact)."""
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_unaccepted_candidate("f_pending")],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
            "--strict",
        ]
    )
    assert proc.returncode != 0
    # Stderr surfaces the gap
    assert "backlog" in (proc.stderr + proc.stdout).lower()


def test_cli_strict_mode_with_backlog_exits_zero(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate(f"f_{i}") for i in range(2)],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
            "--strict",
        ]
    )
    # >=1 accepted -> strict pre-flight passes
    assert proc.returncode == 0, proc.stderr


def test_cli_custom_threshold_via_flag(tmp_path):
    cdir = tmp_path / "candidates"
    cdir.mkdir()
    _write_manifest(
        cdir / "compile_set_candidates_a.json",
        [_accepted_candidate(f"f_{i}") for i in range(3)],
    )
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    # threshold=2 -> 3 accepted SHOULD emit READY
    proc = _run_cli(
        [
            "--candidates-dir",
            str(cdir),
            "--artifact",
            str(artifact),
            "--threshold",
            "2",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "ready" in proc.stdout.lower()
