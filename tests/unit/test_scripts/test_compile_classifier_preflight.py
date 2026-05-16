"""Pre-flight hook for ``scripts/compile_causal_role_classifier.py``.

Phase 4.5 (issue #236): refuse to recompile when no new accepted
candidates have landed since the last artifact, unless ``--force`` is
passed. Forces operators to either curate-and-merge backlog OR
explicitly acknowledge they're recompiling without any new evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_compile_set_candidate_backlog as backlog  # noqa: E402


def _accepted(feature: str) -> dict:
    return {
        "feature_name": feature,
        "expected_causal_role": "confounder",
        "expected_remediation": "keep_with_caveat",
        "derivation_pseudocode": "select x from y",
        "dataset_context": "CSU target ON_180",
        "evaluator_audit": {
            "satisfied": False,
            "rationale_complete": False,
            "missed_considerations": [],
            "notes": "",
            "model": "anthropic/claude-haiku-4-5-20251001",
        },
        "source_run_id": "exp-1",
        "source_written_at": "2026-05-10T00:00:00+00:00",
        "source_path": "/path/sidecar.json",
    }


def _unaccepted(feature: str) -> dict:
    d = _accepted(feature)
    d.update(
        expected_causal_role=None,
        expected_remediation=None,
        derivation_pseudocode=None,
        dataset_context=None,
    )
    return d


def _write_manifest(path: Path, candidates: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-05-15T12:00:00+00:00",
                "schema_version": 1,
                "required_fill_ins": list(backlog.REQUIRED_FILL_INS),
                "candidates": candidates,
            },
            indent=2,
        )
    )


def test_preflight_helper_passes_when_artifact_missing(tmp_path):
    """Cold-start: artifact doesn't exist -> the script should not be
    blocked (operator is bootstrapping)."""
    from scripts.compile_causal_role_classifier import preflight_candidate_check

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    artifact = tmp_path / "missing" / "artifact.json"

    ok, message = preflight_candidate_check(
        candidates_dir=candidates_dir,
        compiled_artifact_path=artifact,
        force=False,
    )
    assert ok is True
    assert "bootstrap" in message.lower() or "no prior artifact" in message.lower()


def test_preflight_refuses_zero_backlog_no_force(tmp_path):
    """Standard guard: artifact exists, no new accepted candidates ->
    refuse the compile."""
    from scripts.compile_causal_role_classifier import preflight_candidate_check

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _write_manifest(
        candidates_dir / "compile_set_candidates_a.json",
        [_unaccepted("f1")],
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    ok, message = preflight_candidate_check(
        candidates_dir=candidates_dir,
        compiled_artifact_path=artifact,
        force=False,
    )
    assert ok is False
    assert "backlog" in message.lower()


def test_preflight_force_overrides_zero_backlog(tmp_path):
    """``--force`` is the operator's acknowledgement that they know
    they're recompiling without new evidence."""
    from scripts.compile_causal_role_classifier import preflight_candidate_check

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _write_manifest(
        candidates_dir / "compile_set_candidates_a.json",
        [_unaccepted("f1")],
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    ok, message = preflight_candidate_check(
        candidates_dir=candidates_dir,
        compiled_artifact_path=artifact,
        force=True,
    )
    assert ok is True
    assert "force" in message.lower()


def test_preflight_passes_with_nonzero_backlog(tmp_path):
    from scripts.compile_causal_role_classifier import preflight_candidate_check

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _write_manifest(
        candidates_dir / "compile_set_candidates_a.json",
        [_accepted("f_ok_1"), _accepted("f_ok_2")],
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    ok, message = preflight_candidate_check(
        candidates_dir=candidates_dir,
        compiled_artifact_path=artifact,
        force=False,
    )
    assert ok is True
    assert "2" in message or "backlog" in message.lower()


def test_compile_cli_blocked_when_backlog_zero(tmp_path):
    """End-to-end CLI integration: zero backlog + no ``--force`` -> the
    compile_causal_role_classifier.py script exits non-zero BEFORE it
    even configures the LM. Wired via ``--candidates-dir`` flag.

    The compile script defers DSPy imports past the pre-flight, so this
    test exercises the refusal path even when dspy isn't installed.
    """
    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _write_manifest(
        candidates_dir / "compile_set_candidates_a.json",
        [_unaccepted("f1")],
    )

    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    os.utime(artifact, (artifact.stat().st_atime, artifact.stat().st_mtime - 3600))

    cli = Path(__file__).resolve().parents[3] / "scripts" / "compile_causal_role_classifier.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--no-lm",
            "--out",
            str(artifact),
            "--candidates-dir",
            str(candidates_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, (
        f"compile CLI should have refused zero-backlog: stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
    # Unique "REFUSED:" prefix is emitted ONLY when the pre-flight
    # refusal path fires — distinguishes a gate trip from any downstream
    # import failure (e.g. dspy ModuleNotFoundError) for falsifiability.
    assert "REFUSED:" in proc.stderr, f"refusal banner missing from stderr: {proc.stderr!r}"
    assert "backlog is zero" in proc.stderr.lower()
    # Refusal message must point operator at the escape hatch.
    assert "--force" in proc.stderr


def test_compile_cli_force_proceeds_even_with_zero_backlog(tmp_path):
    """``--force`` bypasses the pre-flight (and the underlying compile
    runs in --no-lm path, which doesn't need credentials).

    Skipped when dspy isn't installed — the `--no-lm` path still
    constructs ``CausalRoleClassifier`` (and persists it) and that
    requires dspy. Pre-flight refusal test above covers the
    dspy-not-installed path.
    """
    pytest.importorskip("dspy")

    candidates_dir = tmp_path / "candidates"
    candidates_dir.mkdir()
    _write_manifest(
        candidates_dir / "compile_set_candidates_a.json",
        [_unaccepted("f1")],
    )

    out_artifact = tmp_path / "out_artifact.json"

    cli = Path(__file__).resolve().parents[3] / "scripts" / "compile_causal_role_classifier.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(cli),
            "--no-lm",
            "--out",
            str(out_artifact),
            "--candidates-dir",
            str(candidates_dir),
            "--force",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"--force should have proceeded: {proc.stderr}"
    assert out_artifact.exists()


def test_compile_cli_default_candidates_dir_is_repo_root_candidates(tmp_path):
    """When ``--candidates-dir`` is NOT passed, the script falls back to
    ``./candidates`` (matching ``make curate-candidates`` output dir)."""
    from scripts.compile_causal_role_classifier import DEFAULT_CANDIDATES_DIR

    # The constant points at <repo_root>/candidates (relative to PROJECT_ROOT
    # resolved in the compile module).
    assert DEFAULT_CANDIDATES_DIR.name == "candidates"
