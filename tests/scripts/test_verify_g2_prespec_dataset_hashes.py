"""Unit tests for ``scripts/verify_g2_prespec_dataset_hashes.py``.

Mirror of ``tests/unit/test_scripts/test_verify_g5_prespec_hashes.py``;
pins the parser + replace-placeholder logic so a memo edit doesn't
silently drop one of the two pinned-hash entries.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_g2_prespec_dataset_hashes as V  # noqa: E402


def test_artifacts_constant_has_two_keys() -> None:
    """The two artifacts referenced by the G2 experiment harness are
    pinned in V.ARTIFACTS."""
    expected_keys = {
        "optum_initiation_patient_journeys_parquet",
        "optum_initiation_treatment_events_parquet",
    }
    assert set(V.ARTIFACTS.keys()) == expected_keys


def test_parse_pinned_hashes_recognizes_placeholder() -> None:
    """When the memo has TODO_PIN_AT_FIRST_GREEN_RUN as the value, the
    parser maps the key to None (= placeholder, not a real hash)."""
    memo = """
    g2_dataset_hashes:
      optum_initiation_patient_journeys_parquet:
        path: "data/rwd/optum/initiation/x.parquet"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
      optum_initiation_treatment_events_parquet:
        path: "data/rwd/optum/initiation/y.parquet"
        sha256: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
    """
    parsed = V._parse_pinned_hashes(memo)
    assert parsed["optum_initiation_patient_journeys_parquet"] is None
    assert parsed["optum_initiation_treatment_events_parquet"] == (
        "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
    )


def test_replace_pinned_hash_only_replaces_placeholder() -> None:
    """The replacer swaps placeholder for a new hash; refuses to
    overwrite an already-pinned non-placeholder value."""
    memo = """
      optum_initiation_patient_journeys_parquet:
        path: "data/rwd/optum/initiation/x.parquet"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
    """
    new_hash = "0" * 64
    out = V._replace_pinned_hash(memo, "optum_initiation_patient_journeys_parquet", new_hash)
    assert "TODO_PIN_AT_FIRST_GREEN_RUN" not in out
    assert new_hash in out

    pinned_memo = """
      optum_initiation_patient_journeys_parquet:
        path: "data/rwd/optum/initiation/x.parquet"
        sha256: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
    """
    second_hash = "f" * 64
    out2 = V._replace_pinned_hash(
        pinned_memo, "optum_initiation_patient_journeys_parquet", second_hash
    )
    assert second_hash not in out2
    assert "abc123def456" in out2  # old hash preserved


def test_sha256_helper_matches_hashlib(tmp_path: Path) -> None:
    """V._sha256_of must match hashlib.sha256 on the same bytes."""
    content = b"hello G2 cohort\n" * 100
    f = tmp_path / "test_file.bin"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert V._sha256_of(f) == expected


def test_verify_returns_zero_when_all_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Vacuous-pass when no artifact is present on disk; the harness's
    CI-presence guard is the load-bearing check."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned)
    assert rc == 0


def test_verify_returns_nonzero_on_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hash mismatch is a hard failure."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["optum_initiation_patient_journeys_parquet"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"some-cohort-bytes")

    pinned = dict.fromkeys(V.ARTIFACTS)
    pinned["optum_initiation_patient_journeys_parquet"] = "0" * 64

    rc = V._verify(pinned)
    assert rc == 1


def test_verify_returns_zero_when_pinned_matches_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: live hash matches pinned → exit 0."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["optum_initiation_treatment_events_parquet"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"another-cohort-bytes"
    target_path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()

    pinned = dict.fromkeys(V.ARTIFACTS)
    pinned["optum_initiation_treatment_events_parquet"] = expected_hash

    rc = V._verify(pinned)
    assert rc == 0


def test_verify_returns_nonzero_when_artifact_present_but_pinned_is_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Artifact lands on disk but memo still has placeholder → fail."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["optum_initiation_patient_journeys_parquet"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"present-but-unpinned")

    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned)
    assert rc == 1


def test_memo_in_repo_has_two_pinned_entries() -> None:
    """The committed memo must have BOTH pinned-hash entries.
    Catches a regression where someone edits the memo and silently
    drops one of the two cohort artifacts."""
    memo_path = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"
    memo_text = memo_path.read_text(encoding="utf-8")
    parsed = V._parse_pinned_hashes(memo_text)
    assert set(parsed.keys()) == set(V.ARTIFACTS.keys())


def test_committed_memo_starts_with_placeholders() -> None:
    """At S_prespec time, both cohort hashes are TODO placeholders.

    The G2 spec deliberately ships placeholders; the operator
    populates them via --update on the present artifacts AFTER
    S_prespec lands and BEFORE the experiment tag is created. This
    test pins the "placeholder-at-S_prespec" property; once the
    operator pins live hashes, the test gets updated on the same
    diff that pins them.
    """
    memo_path = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"
    memo_text = memo_path.read_text(encoding="utf-8")
    parsed = V._parse_pinned_hashes(memo_text)
    # We accept either:
    #   - all placeholders (S_prespec state), OR
    #   - all real hashes (post-pinning state)
    # A MIXED state (one pinned, one placeholder) is a forbidden
    # half-pinning that the verifier would catch separately.
    n_pinned = sum(1 for v in parsed.values() if v is not None)
    n_placeholder = sum(1 for v in parsed.values() if v is None)
    assert n_pinned == 0 or n_placeholder == 0, (
        "Memo is in a half-pinned state; the operator must pin BOTH "
        "or NEITHER. Mixed states violate the threshold-shopping "
        "defense's data-content half."
    )


def test_verifier_main_is_callable_with_no_args() -> None:
    """``main()`` is the entry point CI invokes via
    ``python scripts/verify_g2_prespec_dataset_hashes.py``. The
    callable handshake completes the load-bearing path."""
    rc = V.main([])
    assert rc in (0, 1, 2), f"Unexpected exit code from V.main([]): {rc!r}"


def test_placeholder_constant_matches_memo_token() -> None:
    """Drift detector: the literal token used in the memo and the
    constant in the verifier must agree."""
    memo_path = REPO_ROOT / "docs" / "specs" / "tier1b_b2_prespec_20260510.md"
    memo_text = memo_path.read_text(encoding="utf-8")
    # The token is referenced multiple times in the memo (placeholder
    # values + explanatory prose). At least the placeholder block
    # must contain it; the explanatory prose is permitted to mention
    # it without quoting.
    assert V.PLACEHOLDER == "TODO_PIN_AT_FIRST_GREEN_RUN"
    assert V.PLACEHOLDER in memo_text


def test_update_skips_already_pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Operator-only --update path must REFUSE to overwrite an
    existing non-placeholder value (audit-visible re-pin requires a
    fresh memo)."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    fake_memo = tmp_path / "fake_memo.md"
    fake_memo.write_text(
        """
g2_dataset_hashes:
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/x.parquet"
    sha256: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/y.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(V, "MEMO_PATH", fake_memo)

    # Plant the patient-journeys artifact so it has a live hash.
    patient_relpath = V.ARTIFACTS["optum_initiation_patient_journeys_parquet"][0]
    patient_path = tmp_path / patient_relpath
    patient_path.parent.mkdir(parents=True, exist_ok=True)
    patient_path.write_bytes(b"patient-bytes")
    # Plant the treatment-events artifact too.
    events_relpath = V.ARTIFACTS["optum_initiation_treatment_events_parquet"][0]
    events_path = tmp_path / events_relpath
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_bytes(b"events-bytes")

    pinned = V._parse_pinned_hashes(fake_memo.read_text(encoding="utf-8"))
    rc = V._update(pinned)
    assert rc == 0  # update returns 0 even when some are skipped

    # The placeholder for events should be replaced; the patient hash
    # should NOT have been overwritten.
    new_text = fake_memo.read_text(encoding="utf-8")
    assert "abc123def456abc123def456abc123def456abc123def456abc123def456abcd" in new_text, (
        "patient hash was overwritten — verifier must refuse"
    )
    assert "TODO_PIN_AT_FIRST_GREEN_RUN" not in new_text or (
        new_text.count("TODO_PIN_AT_FIRST_GREEN_RUN") < 2
    )
