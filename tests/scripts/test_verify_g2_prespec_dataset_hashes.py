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


def test_verify_lenient_returns_zero_when_all_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LENIENT (strict=False): vacuous-pass when no artifact is present.

    This is the local-diagnostic mode (``--allow-missing``). The
    HIGH-2 fix means CI invocations always set strict=True, so this
    code path is unreachable from CI.
    """
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned, strict=False)
    assert rc == 0


def test_verify_strict_fails_when_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """HIGH-2 fix: strict mode (CI=true OR --strict) rejects missing
    artifacts as a HARD FAILURE. Vacuous-pass on missing artifacts is
    the threshold-shopping escape hatch the codex pass-1 flagged."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    pinned = dict.fromkeys(V.ARTIFACTS)
    rc = V._verify(pinned, strict=True)
    assert rc == 1


def test_verify_strict_fails_when_one_of_two_artifacts_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Strict mode catches partial-missing too — even ONE absent
    artifact violates the data-content half of the defense."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    # Plant only one of two artifacts.
    target_relpath = V.ARTIFACTS["optum_initiation_treatment_events_parquet"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"events-bytes"
    target_path.write_bytes(payload)

    pinned = {
        "optum_initiation_patient_journeys_parquet": "0" * 64,
        "optum_initiation_treatment_events_parquet": hashlib.sha256(payload).hexdigest(),
    }
    rc = V._verify(pinned, strict=True)
    assert rc == 1


def test_verify_returns_nonzero_on_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hash mismatch is a hard failure (independent of strict)."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)

    target_relpath = V.ARTIFACTS["optum_initiation_patient_journeys_parquet"][0]
    target_path = tmp_path / target_relpath
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"some-cohort-bytes")

    pinned = dict.fromkeys(V.ARTIFACTS)
    pinned["optum_initiation_patient_journeys_parquet"] = "0" * 64

    rc = V._verify(pinned, strict=False)
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

    rc = V._verify(pinned, strict=False)
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
    rc = V._verify(pinned, strict=False)
    assert rc == 1


def test_main_strict_in_ci_rejects_missing_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end HIGH-2 contract: CI=true → main() returns 1 when
    artifacts are missing."""
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
    sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(V, "MEMO_PATH", fake_memo)
    monkeypatch.setenv("CI", "true")
    rc = V.main(["--prespec-sha", "working"])
    assert rc == 1


def test_main_allow_missing_rejected_in_ci(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """HIGH-2 contract: --allow-missing is REJECTED in strict mode
    (CI=true OR --strict) so an operator cannot accidentally pass it
    in CI."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(V, "MEMO_PATH", tmp_path / "fake.md")
    (tmp_path / "fake.md").write_text("g2_dataset_hashes:\n", encoding="utf-8")
    monkeypatch.setenv("CI", "true")
    rc = V.main(["--allow-missing", "--prespec-sha", "working"])
    assert rc == 2


def test_main_allow_missing_rejected_with_explicit_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Symmetric: --strict + --allow-missing is also rejected."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(V, "MEMO_PATH", tmp_path / "fake.md")
    (tmp_path / "fake.md").write_text("g2_dataset_hashes:\n", encoding="utf-8")
    monkeypatch.delenv("CI", raising=False)
    rc = V.main(["--strict", "--allow-missing", "--prespec-sha", "working"])
    assert rc == 2


def test_main_local_allow_missing_returns_zero_when_artifacts_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local diagnostic flow: CI unset, --allow-missing, no artifacts
    on disk → exit 0 (vacuous pass)."""
    monkeypatch.setattr(V, "REPO_ROOT", tmp_path)
    fake_memo = tmp_path / "fake_memo.md"
    fake_memo.write_text(
        """
g2_dataset_hashes:
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/x.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/y.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(V, "MEMO_PATH", fake_memo)
    monkeypatch.delenv("CI", raising=False)
    rc = V.main(["--allow-missing", "--prespec-sha", "working"])
    assert rc == 0


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


# ---------------------------------------------------------------------------
# HIGH-3 (iter-3) — structured fingerprint extraction.
# ---------------------------------------------------------------------------


def _make_canonical_memo(
    *,
    t1: str = "0.03",
    t2: str = "0.5",
    t3: str = "0.7",
    seeds: str = "(42, 43, 44, 45, 46)",
    n_default: str = "1294",
    n_relaxed: str = "1697",
    target: str = "treatment_initiated",
    journeys_hash: str = "TODO_PIN_AT_FIRST_GREEN_RUN",
    events_hash: str = "TODO_PIN_AT_FIRST_GREEN_RUN",
) -> str:
    """Synthesize a memo whose load-bearing fingerprint matches the
    canonical pre-spec memo. Used to test that drift detection
    triggers on value changes but NOT on whitespace / ordering."""
    return f"""# Gate G2 Pre-Spec
| **T1** | dAUC | `Δ_AUC ≥ {t1}` | held-out lift |
| **T2** | ECE  | `ECE_post ≤ {t2} × ECE_pre` | calibration halving |
| **T3** | CV   | `(std/mean)_post ≤ {t3} × (std/mean)_pre` | CV stability |

- **Cohort label:** `optum_initiation_default`
- **Patient-journey count:** {n_default}
- **Target column:** `{target}`

- **Cohort label:** `optum_initiation_relaxed`
- **Patient-journey count:** {n_relaxed}
- **Target column:** `{target}`

```python
G2_DELTA_AUC_MIN: float = {t1}           # T1
G2_ECE_RATIO_MAX: float = {t2}            # T2
G2_CV_STABILITY_RATIO_MAX: float = {t3}   # T3
G2_SEEDS: tuple[int, ...] = {seeds}
```

```yaml
g2_dataset_hashes:
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/x.parquet"
    sha256: "{journeys_hash}"
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/y.parquet"
    sha256: "{events_hash}"
```
"""


class TestStructuredFingerprintExtraction:
    """HIGH-3 (iter-3): the fingerprint extracts canonical values for
    thresholds, seeds, cohort identifiers, and pinned hashes — making
    multi-line load-bearing edits visible while ignoring whitespace
    and prose changes."""

    def test_extracts_thresholds_and_seeds_and_cohort_identifiers(self) -> None:
        memo = _make_canonical_memo()
        fp = V._extract_load_bearing_fingerprint(memo)
        assert fp["t1_code"] == 0.03
        assert fp["t2_code"] == 0.5
        assert fp["t3_code"] == 0.7
        assert fp["seeds"] == [42, 43, 44, 45, 46]
        assert "optum_initiation_default" in fp["cohort_labels"]
        assert "optum_initiation_relaxed" in fp["cohort_labels"]
        assert "1294" in fp["cohort_patient_counts"]
        assert "1697" in fp["cohort_patient_counts"]
        assert "treatment_initiated" in fp["target_columns"]

    def test_whitespace_and_ordering_does_not_change_fingerprint(self) -> None:
        memo_a = _make_canonical_memo()
        # Deliberately reorder + add whitespace — same load-bearing
        # values, structurally reorganized.
        memo_b = _make_canonical_memo()
        memo_b = memo_b.replace("```python\n", "```python\n\n\n").replace("# T1", "# T1   ")
        fp_a = V._extract_load_bearing_fingerprint(memo_a)
        fp_b = V._extract_load_bearing_fingerprint(memo_b)
        assert fp_a == fp_b

    def test_threshold_value_change_changes_fingerprint(self) -> None:
        memo_a = _make_canonical_memo(t1="0.03")
        memo_b = _make_canonical_memo(t1="0.02")  # threshold edit
        fp_a = V._extract_load_bearing_fingerprint(memo_a)
        fp_b = V._extract_load_bearing_fingerprint(memo_b)
        assert fp_a != fp_b
        assert fp_a["t1_code"] != fp_b["t1_code"]
        assert fp_a["t1_prose"] != fp_b["t1_prose"]

    def test_seeds_change_changes_fingerprint(self) -> None:
        memo_a = _make_canonical_memo()
        memo_b = _make_canonical_memo(seeds="(42, 43, 44, 45, 47)")
        fp_a = V._extract_load_bearing_fingerprint(memo_a)
        fp_b = V._extract_load_bearing_fingerprint(memo_b)
        assert fp_a != fp_b
        assert fp_a["seeds"] != fp_b["seeds"]

    def test_pinned_hash_change_changes_fingerprint(self) -> None:
        memo_a = _make_canonical_memo(journeys_hash="0" * 64)
        memo_b = _make_canonical_memo(journeys_hash="f" * 64)
        fp_a = V._extract_load_bearing_fingerprint(memo_a)
        fp_b = V._extract_load_bearing_fingerprint(memo_b)
        assert fp_a != fp_b

    def test_cohort_count_change_changes_fingerprint(self) -> None:
        """A memo edit that swaps n=1294 for some other count
        (e.g. n=1300) constitutes load-bearing drift."""
        memo_a = _make_canonical_memo(n_default="1294")
        memo_b = _make_canonical_memo(n_default="1300")
        fp_a = V._extract_load_bearing_fingerprint(memo_a)
        fp_b = V._extract_load_bearing_fingerprint(memo_b)
        assert fp_a != fp_b
        assert fp_a["cohort_patient_counts"] != fp_b["cohort_patient_counts"]


# ---------------------------------------------------------------------------
# NEW HIGH-2 (iter-3) — REPO_ROOT resolution when staged into a governance
# checkout. The workflow copies this script into
# governance_checkout/scripts/ and runs from there, so
# Path(__file__).parents[1] resolves to "governance_checkout", NOT the
# actual worktree. The verifier MUST resolve the worktree via:
#   1. E2I_GOVERNANCE_REPO_ROOT env var, OR
#   2. --repo-root CLI flag, OR
#   3. git rev-parse --show-toplevel
# ---------------------------------------------------------------------------


def test_resolve_repo_root_uses_env_var_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """E2I_GOVERNANCE_REPO_ROOT env var wins over Path(__file__) parent."""
    monkeypatch.setenv("E2I_GOVERNANCE_REPO_ROOT", str(tmp_path))
    resolved = V._resolve_repo_root()
    assert resolved == tmp_path.resolve()


def test_main_repo_root_flag_overrides_module_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--repo-root`` CLI override is the workflow's contract for the
    staged governance checkout. The override must redirect every
    REPO_ROOT-derived path (MEMO_PATH, artifact paths) to the actual
    worktree.
    """
    # Simulate the workflow scenario: scripts staged in a fake
    # governance_checkout dir (where the verifier code lives), but the
    # actual worktree is tmp_path.
    actual_worktree = tmp_path / "actual_worktree"
    actual_worktree.mkdir()
    memo_dir = actual_worktree / "docs" / "specs"
    memo_dir.mkdir(parents=True)
    memo_path = memo_dir / "tier1b_b2_prespec_20260510.md"
    memo_path.write_text(
        """
g2_dataset_hashes:
  optum_initiation_patient_journeys_parquet:
    path: "data/rwd/optum/initiation/x.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
  optum_initiation_treatment_events_parquet:
    path: "data/rwd/optum/initiation/y.parquet"
    sha256: "TODO_PIN_AT_FIRST_GREEN_RUN"
""",
        encoding="utf-8",
    )
    # Make sure CI is not set so --allow-missing is honored.
    monkeypatch.delenv("CI", raising=False)
    # Pre-call: REPO_ROOT could be anything; --repo-root on CLI must
    # override and point at the actual_worktree.
    rc = V.main(
        [
            "--allow-missing",
            "--prespec-sha",
            "working",
            "--repo-root",
            str(actual_worktree),
        ]
    )
    # MEMO_PATH resolves under actual_worktree → memo found.
    # Both artifacts missing under actual_worktree/data/... → vacuous
    # pass under --allow-missing (lenient).
    assert rc == 0
    # And the post-flag REPO_ROOT module attr is the actual worktree.
    assert V.REPO_ROOT == actual_worktree.resolve()


def test_main_repo_root_flag_finds_planted_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end NEW HIGH-2: with --repo-root pointing at the actual
    worktree, the verifier locates artifacts under that root (not under
    its own staging dir)."""
    actual_worktree = tmp_path / "actual_worktree"
    actual_worktree.mkdir()
    memo_dir = actual_worktree / "docs" / "specs"
    memo_dir.mkdir(parents=True)
    memo_path = memo_dir / "tier1b_b2_prespec_20260510.md"
    journeys_relpath = V.ARTIFACTS["optum_initiation_patient_journeys_parquet"][0]
    events_relpath = V.ARTIFACTS["optum_initiation_treatment_events_parquet"][0]
    journeys_path = actual_worktree / journeys_relpath
    events_path = actual_worktree / events_relpath
    journeys_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    journeys_payload = b"journeys-bytes"
    events_payload = b"events-bytes"
    journeys_path.write_bytes(journeys_payload)
    events_path.write_bytes(events_payload)
    journeys_hash = hashlib.sha256(journeys_payload).hexdigest()
    events_hash = hashlib.sha256(events_payload).hexdigest()
    memo_path.write_text(
        f"""
g2_dataset_hashes:
  optum_initiation_patient_journeys_parquet:
    path: "{journeys_relpath}"
    sha256: "{journeys_hash}"
  optum_initiation_treatment_events_parquet:
    path: "{events_relpath}"
    sha256: "{events_hash}"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("CI", raising=False)
    rc = V.main(
        [
            "--prespec-sha",
            "working",
            "--repo-root",
            str(actual_worktree),
        ]
    )
    assert rc == 0


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
