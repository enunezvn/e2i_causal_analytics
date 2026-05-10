"""Unit tests for ``scripts/verify_g5_prespec_hashes.py``.

Pins the parser + replace-placeholder logic so a memo edit doesn't
silently drop one of the four pinned-hash entries.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_g5_prespec_hashes as V  # noqa: E402


def test_artifacts_constant_has_all_four_keys() -> None:
    """The four artifacts referenced by the integration test are
    pinned in V.ARTIFACTS."""
    expected_keys = {
        "optum_initiation_patient_journeys_parquet",
        "csu_patient_journeys_json",
        "optum_initiation_treatment_events_parquet",
        "csu_treatment_events_json",
    }
    assert set(V.ARTIFACTS.keys()) == expected_keys


def test_parse_pinned_hashes_recognizes_placeholder() -> None:
    """When the memo has TODO_PIN_AT_FIRST_GREEN_RUN as the value, the
    parser maps the key to None (= placeholder, not a real hash)."""
    memo = """
    g5_dataset_hashes:
      optum_initiation_patient_journeys_parquet:
        path: "data/rwd/optum/initiation/x.parquet"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
      csu_patient_journeys_json:
        path: "data/rwd/csu/y.json"
        sha256: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
      optum_initiation_treatment_events_parquet:
        path: "data/rwd/optum/initiation/z.parquet"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
      csu_treatment_events_json:
        path: "data/rwd/csu/w.json"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
    """
    parsed = V._parse_pinned_hashes(memo)
    assert parsed["optum_initiation_patient_journeys_parquet"] is None
    assert parsed["csu_patient_journeys_json"] == (
        "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
    )
    assert parsed["optum_initiation_treatment_events_parquet"] is None
    assert parsed["csu_treatment_events_json"] is None


def test_replace_pinned_hash_only_replaces_placeholder() -> None:
    """When the memo has a placeholder, _replace_pinned_hash swaps
    in the new hash. When the memo has a real hash, the function
    leaves the existing value untouched (operator must explicitly
    re-pin via a fresh memo)."""
    memo = """
      csu_patient_journeys_json:
        path: "data/rwd/csu/y.json"
        sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
    """
    new_hash = "0" * 64
    out = V._replace_pinned_hash(memo, "csu_patient_journeys_json", new_hash)
    assert "TODO_PIN_AT_FIRST_GREEN_RUN" not in out
    assert new_hash in out

    # Now feed a memo where the value is already pinned (not a placeholder)
    # — the function MUST NOT replace it.
    pinned_memo = """
      csu_patient_journeys_json:
        path: "data/rwd/csu/y.json"
        sha256: "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
    """
    second_hash = "f" * 64
    out2 = V._replace_pinned_hash(pinned_memo, "csu_patient_journeys_json", second_hash)
    assert second_hash not in out2
    assert "abc123def456" in out2  # old hash preserved


def test_sha256_helper_matches_hashlib(tmp_path: Path) -> None:
    """V._sha256_of must match hashlib.sha256 on the same bytes."""
    content = b"hello world\n" * 100
    f = tmp_path / "test_file.bin"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    assert V._sha256_of(f) == expected


def test_verify_returns_zero_when_all_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If every pinned artifact is missing on disk, verify returns 0
    (vacuous pass) — the integration test's M2 fixture is the load-
    bearing CI gate. _verify prints a warning, not a failure."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned)
    assert rc == 0


def test_verify_returns_nonzero_on_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If a pinned artifact exists on disk but its sha256 differs from
    the memo, _verify returns 1 (mismatch is a hard failure)."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    # Plant ONE artifact so its hash can be computed.
    target_relpath = V.ARTIFACTS["csu_patient_journeys_json"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"some-cohort-bytes")

    # Pin a deliberately-wrong sha256 for that artifact; leave the
    # other three as None (missing → skipped).
    pinned = dict.fromkeys(V.ARTIFACTS)
    pinned["csu_patient_journeys_json"] = "0" * 64

    rc = V._verify(pinned)
    assert rc == 1


def test_verify_returns_zero_when_pinned_matches_live(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: live hash matches the pinned value → _verify
    returns 0."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["csu_patient_journeys_json"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"another-cohort-bytes"
    target_path.write_bytes(payload)
    expected_hash = hashlib.sha256(payload).hexdigest()

    pinned = dict.fromkeys(V.ARTIFACTS)
    pinned["csu_patient_journeys_json"] = expected_hash

    rc = V._verify(pinned)
    assert rc == 0


def test_verify_returns_nonzero_when_artifact_present_but_pinned_is_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the artifact lands on disk but the memo still has the
    TODO_PIN_AT_FIRST_GREEN_RUN placeholder, the operator forgot to
    run --update. This must FAIL (the cohort would otherwise run
    against an unpinned hash, defeating the protocol)."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["csu_patient_journeys_json"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"present-but-unpinned")

    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned)
    assert rc == 1


def test_memo_in_repo_has_four_pinned_entries() -> None:
    """The committed memo must have all four pinned-hash entries.
    Catches a regression where someone edits the memo and silently
    drops one of the four cohort artifacts."""
    memo_path = REPO_ROOT / "docs" / "specs" / "g5_coefficient_sensitivity_prespec_20260510.md"
    memo_text = memo_path.read_text(encoding="utf-8")
    parsed = V._parse_pinned_hashes(memo_text)
    # All four keys must be parseable (None or a real hash; the test
    # itself doesn't care which — that's _verify's job).
    assert set(parsed.keys()) == set(V.ARTIFACTS.keys())


def test_committed_memo_has_no_placeholder_sha256_values() -> None:
    """G5 codex pass-2 NEW MED 1 closure: the committed memo MUST have
    real sha256 values pinned for all four cohort artifacts (no
    ``TODO_PIN_AT_FIRST_GREEN_RUN`` placeholders in main).

    The verifier script is the load-bearing CI gate; if any of the four
    keys still parses to ``None`` (meaning the memo still has a
    placeholder), the threshold-shopping defense is incomplete and the
    cohort artifacts could drift between memo-lock and run without
    detection.
    """
    memo_path = REPO_ROOT / "docs" / "specs" / "g5_coefficient_sensitivity_prespec_20260510.md"
    memo_text = memo_path.read_text(encoding="utf-8")
    parsed = V._parse_pinned_hashes(memo_text)
    for key, value in parsed.items():
        assert value is not None, (
            f"Memo key {key!r} still has a TODO_PIN_AT_FIRST_GREEN_RUN "
            f"placeholder. M1 closure requires all four cohort artifact "
            f"hashes to be pinned. Run "
            f"scripts/verify_g5_prespec_hashes.py --update on a fresh "
            f"cohort and commit the result via a NEW pre-spec memo."
        )
        # Sanity: the value should be a 64-char lowercase hex string.
        assert len(value) == 64, (
            f"Memo key {key!r} has a malformed sha256 value of length "
            f"{len(value)} (expected 64): {value!r}"
        )
        assert all(c in "0123456789abcdef" for c in value), (
            f"Memo key {key!r} sha256 value contains non-hex chars: {value!r}"
        )


def test_verifier_main_is_callable_with_no_args() -> None:
    """G5 codex pass-2 NEW MED 1 closure: the verifier script's
    ``main()`` is the entry point CI invokes via
    ``python scripts/verify_g5_prespec_hashes.py``. This test ensures
    the function is importable + callable, completing the
    "load-bearing" handshake (an unimported script is dead code; an
    uncalled main is unreachable).

    We invoke main([]) (no args = verify mode) and accept any exit code
    (0 if hashes match, 1 if mismatch — both prove main runs end-to-end).
    """
    rc = V.main([])
    assert rc in (0, 1, 2), f"Unexpected exit code from V.main([]): {rc!r}"
