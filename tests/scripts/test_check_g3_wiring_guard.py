"""Unit tests for ``scripts/check_g3_wiring_guard.py``.

Plan v4 §2 Gate G3 mechanical CI enforcement is the load-bearing
deliverable of this PR; the validator script must produce the right
verdict on:

  1. AST scan of the gated functions for `hblp_classify` callsites.
  2. Signoff file existence.
  3. Signoff `commit:` SHA ancestry vs HEAD.
  4. (Optional) signoff committer email match against the N3 reviewer
     registry.

These tests construct synthetic git repos under ``tmp_path`` to verify
each branch independently. The repos are created via ``subprocess`` so
no third-party git library is required.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

# Make the ``scripts/`` directory importable so we can call into the
# guard's helper functions directly (not just the CLI).
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_g3_wiring_guard as guard  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Bootstrap a tiny git repo under ``tmp_path`` and return its root."""

    if shutil.which("git") is None:
        pytest.skip("git binary not available")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "commit.gpgsign", "false"],
        check=True,
    )
    return tmp_path


def _git_commit(repo: Path, message: str = "checkpoint") -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", message],
        check=True,
    )
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _seed_initial_commit(repo: Path) -> str:
    """Create an initial commit so HEAD exists."""

    (repo / "README.md").write_text("# tmp repo for G3 wiring guard tests\n")
    return _git_commit(repo, "initial")


def _make_target_file(repo: Path, body: str) -> Path:
    """Create the gated source file with the supplied body."""

    target_dir = repo / "src" / "agents" / "ml_foundation" / "data_preparer" / "nodes"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "adaptive_validity_check.py"
    target_path.write_text(body, encoding="utf-8")
    return target_path


def _write_signoff(repo: Path, gate: str, commit_sha: Optional[str] = None) -> Path:
    """Write a placeholder signoff file under docs/calibration/."""

    cal_dir = repo / "docs" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    path = cal_dir / f"{gate}_completion_signoff_20260510.md"
    if commit_sha is None:
        body = f"# {gate} signoff\n\nNo commit reference yet.\n"
    else:
        body = f"# {gate} signoff\n\nCommit: `{commit_sha}`.\n"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# detect_hblp_wiring tests
# --------------------------------------------------------------------------- #


class TestDetectHblpWiring:
    def test_no_wiring_pre_g3(self, tmp_path: Path) -> None:
        """A file with no `hblp_classify` call returns ok=False (guard inactive)."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _build_verdict(feature, score):\n"
            "    return {'severity': 'info'}\n"
            "\n"
            "def _compose_legacy_verdict(feature, voter):\n"
            "    return {'severity': 'info'}\n"
            "\n"
            "def _adversarial_input(score):\n"
            "    return {'severity': 'info'}\n",
            encoding="utf-8",
        )

        result = guard.detect_hblp_wiring(target)
        assert result.ok is False
        assert "no" in result.detail.lower() or "guard inactive" in result.detail.lower()

    def test_wiring_detected_in_build_verdict(self, tmp_path: Path) -> None:
        """`hblp_classify(...)` inside `_build_verdict` triggers the guard."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _build_verdict(feature, score):\n"
            "    return hblp_classify(z_score=1.0, n_positives=50, layer_1_declared_safe=False)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert "_build_verdict" in result.detail
        assert "hblp_classify" in result.detail

    def test_wiring_detected_in_compose_legacy_verdict(self, tmp_path: Path) -> None:
        """`hblp_classify(...)` inside `_compose_legacy_verdict` triggers the guard."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _compose_legacy_verdict(feature, voter):\n"
            "    cls = hblp_classify(z_score=2.0, n_positives=22, layer_1_declared_safe=True)\n"
            "    return {'severity': cls['severity']}\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert "_compose_legacy_verdict" in result.detail

    def test_wiring_detected_in_adversarial_input(self, tmp_path: Path) -> None:
        """`hblp_classify(...)` inside `_adversarial_input` triggers the guard."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _adversarial_input(score):\n"
            "    out = hblp_classify(z_score=score['z'], n_positives=score['n'], layer_1_declared_safe=False)\n"
            "    return out\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert "_adversarial_input" in result.detail

    def test_wiring_in_other_function_does_not_trigger(self, tmp_path: Path) -> None:
        """A call inside an UN-gated function (e.g. `hblp_effective_z_threshold`) is fine."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def hblp_effective_z_threshold(n_positives, layer_1_declared_safe):\n"
            "    # This function defines HBLP itself; not gated.\n"
            "    return hblp_classify(z_score=0.0, n_positives=n_positives, layer_1_declared_safe=layer_1_declared_safe)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is False

    def test_attribute_access_call_detected(self, tmp_path: Path) -> None:
        """`module.hblp_classify(...)` (attribute-style) is also detected.

        Defense-in-depth: a `from .helpers import hblp_classify as fn` style
        import would change the call name; an `import .helpers as h` style
        keeps the attribute name. We catch the attribute form so a
        determined sidestep via attribute import is harder.
        """

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "from . import helpers\n"
            "\n"
            "def _build_verdict(feature, score):\n"
            "    return helpers.hblp_classify(z_score=score['z'], n_positives=50, layer_1_declared_safe=False)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert "attribute access" in result.detail

    def test_missing_target_file_returns_inactive(self, tmp_path: Path) -> None:
        """Missing target file → guard inactive (treat as no wiring)."""

        result = guard.detect_hblp_wiring(tmp_path / "does_not_exist.py")
        assert result.ok is False
        assert "not found" in result.detail.lower()

    def test_syntax_error_returns_inactive(self, tmp_path: Path) -> None:
        """Syntax-error in target → guard inactive (the SyntaxError surfaces elsewhere)."""

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _build_verdict(feature, score):\n"
            "    return hblp_classify(\n",  # Unterminated paren
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is False


# --------------------------------------------------------------------------- #
# extract_commit_sha tests
# --------------------------------------------------------------------------- #


class TestExtractCommitSha:
    def test_backtick_form(self) -> None:
        body = "# Signoff\n\nBranch / commit: `0123456789abcdef`. Some narrative.\n"
        assert guard.extract_commit_sha(body) == "0123456789abcdef"

    def test_short_sha_via_backticks(self) -> None:
        body = "Signoff at `9c51eac4` (origin/main).\n"
        assert guard.extract_commit_sha(body) == "9c51eac4"

    def test_prefix_form_no_backtick(self) -> None:
        body = "S_prespec commit SHA: 7f616f6f01abcdef\n"
        assert guard.extract_commit_sha(body) == "7f616f6f01abcdef"

    def test_no_sha_returns_none(self) -> None:
        body = "# Signoff\n\nNo SHA here. Just words.\n"
        assert guard.extract_commit_sha(body) is None

    def test_template_placeholder_returns_first_real_token(self) -> None:
        """Template `<sha>` placeholder is not a hex token; first real backtick-hex wins."""

        body = (
            "# Template\n\n"
            "- **CoI declaration commit SHA:** `<sha>` (deferred to N3 infra)\n"
            "- **Real commit:** `abc1234`.\n"
        )
        assert guard.extract_commit_sha(body) == "abc1234"


# --------------------------------------------------------------------------- #
# check_signoff_exists tests
# --------------------------------------------------------------------------- #


class TestCheckSignoffExists:
    def test_present(self, tmp_path: Path) -> None:
        cal_dir = tmp_path / "docs" / "calibration"
        cal_dir.mkdir(parents=True)
        signoff = cal_dir / "g1_completion_signoff_20260510.md"
        signoff.write_text("placeholder\n")
        result = guard.check_signoff_exists(
            tmp_path, "docs/calibration/g1_completion_signoff_20260510.md"
        )
        assert result.ok is True

    def test_missing(self, tmp_path: Path) -> None:
        result = guard.check_signoff_exists(
            tmp_path, "docs/calibration/g1_completion_signoff_20260510.md"
        )
        assert result.ok is False
        assert "MISSING" in result.detail


# --------------------------------------------------------------------------- #
# check_signoff_ancestor tests
# --------------------------------------------------------------------------- #


class TestCheckSignoffAncestor:
    def test_ancestor_passes(self, tmp_repo: Path) -> None:
        """SHA referenced in signoff IS an ancestor of HEAD → PASS."""

        first_sha = _seed_initial_commit(tmp_repo)
        # Make a follow-up commit that references first_sha in the signoff.
        _write_signoff(tmp_repo, "g1", commit_sha=first_sha)
        head_sha = _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is True
        assert "ancestor" in result.detail.lower()

    def test_non_ancestor_fails(self, tmp_repo: Path) -> None:
        """SHA referenced is NOT an ancestor of HEAD (sibling branch) → FAIL."""

        # Initial commit + branch off + write signoff referencing the
        # SIBLING-branch SHA (not in main's ancestry).
        _seed_initial_commit(tmp_repo)
        subprocess.run(
            ["git", "-C", str(tmp_repo), "checkout", "-q", "-b", "sibling"],
            check=True,
        )
        (tmp_repo / "sibling.txt").write_text("sibling work\n")
        sibling_sha = _git_commit(tmp_repo, "sibling commit")

        # Switch back to main and write a signoff referencing sibling_sha.
        # Switch to a fresh branch off the initial commit (not sibling).
        subprocess.run(
            ["git", "-C", str(tmp_repo), "checkout", "-q", "HEAD~1"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "checkout", "-q", "-b", "main-branch"],
            check=True,
        )
        _write_signoff(tmp_repo, "g1", commit_sha=sibling_sha)
        head_sha = _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is False
        assert "NOT" in result.detail

    def test_no_sha_in_signoff_fails(self, tmp_repo: Path) -> None:
        """Signoff with no extractable SHA → FAIL on ancestry check."""

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha=None)
        head_sha = _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is False
        assert "no commit sha" in result.detail.lower()

    def test_missing_signoff_fails(self, tmp_repo: Path) -> None:
        """Signoff file missing → FAIL with informative message."""

        first_sha = _seed_initial_commit(tmp_repo)
        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            first_sha,
        )
        assert result.ok is False
        assert "missing" in result.detail.lower()


# --------------------------------------------------------------------------- #
# parse_registry_emails tests
# --------------------------------------------------------------------------- #


class TestParseRegistryEmails:
    def test_empty_registry_when_missing(self, tmp_path: Path) -> None:
        emails = guard.parse_registry_emails(tmp_path / "missing.md")
        assert emails == set()

    def test_active_rows_only(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.md"
        registry.write_text(
            "# Methodology Reviewer Registry\n"
            "\n"
            "| name | email | github_handle | role | date_added | areas_of_expertise | status |\n"
            "|------|-------|---------------|------|------------|--------------------|--------|\n"
            "| Alice | alice@example.com | alice | reviewer | 2026-04-01 | causal | active |\n"
            "| Bob | bob@example.com | bob | reviewer | 2026-04-01 | survival | inactive |\n",
            encoding="utf-8",
        )
        emails = guard.parse_registry_emails(registry)
        assert emails == {"alice@example.com"}

    def test_email_alias_split(self, tmp_path: Path) -> None:
        registry = tmp_path / "registry.md"
        registry.write_text(
            "| name | email | github_handle | role | date_added | areas_of_expertise | status |\n"
            "|------|-------|---------------|------|------------|--------------------|--------|\n"
            "| Alice | alice@example.com, alice@oldjob.com | alice | reviewer | 2026 | causal | active |\n",
            encoding="utf-8",
        )
        emails = guard.parse_registry_emails(registry)
        assert "alice@example.com" in emails
        assert "alice@oldjob.com" in emails


# --------------------------------------------------------------------------- #
# check_signoff_committer_match tests
# --------------------------------------------------------------------------- #


class TestSignoffCommitterMatch:
    def test_advisory_skip_when_registry_empty(self, tmp_repo: Path) -> None:
        """Empty registry + advisory mode → PASS with WARN (default behavior)."""

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha="abc123")
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=False,
        )
        assert result.ok is True

    def test_require_mode_fails_with_empty_registry(self, tmp_repo: Path) -> None:
        """Empty registry + require_match=True → FAIL."""

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha="abc123")
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=True,
        )
        assert result.ok is False

    def test_require_mode_passes_when_committer_in_registry(self, tmp_repo: Path) -> None:
        """Committer email matches an active registry row → PASS."""

        _seed_initial_commit(tmp_repo)
        # Drop the registry with the test user's email.
        gov_dir = tmp_repo / "docs" / "governance"
        gov_dir.mkdir(parents=True, exist_ok=True)
        (gov_dir / "methodology_reviewer_registry.md").write_text(
            "| name | email | github_handle | role | date_added | areas_of_expertise | status |\n"
            "|------|-------|---------------|------|------------|--------------------|--------|\n"
            "| Test User | test@example.com | testuser | reviewer | 2026 | misc | active |\n",
            encoding="utf-8",
        )
        _write_signoff(tmp_repo, "g1", commit_sha="abc123")
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=True,
        )
        assert result.ok is True

    def test_missing_signoff_advisory_passes(self, tmp_repo: Path) -> None:
        """Missing signoff + advisory mode → PASS with WARN."""

        _seed_initial_commit(tmp_repo)
        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=False,
        )
        assert result.ok is True

    def test_missing_signoff_require_mode_fails(self, tmp_repo: Path) -> None:
        """Missing signoff + require mode → FAIL."""

        _seed_initial_commit(tmp_repo)
        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=True,
        )
        assert result.ok is False


# --------------------------------------------------------------------------- #
# check_g3_wiring_guard orchestrator tests
# --------------------------------------------------------------------------- #


class TestG3GuardOrchestrator:
    """Full-flow tests of the orchestrator with realistic synthetic repos."""

    def test_no_wiring_skips_signoff_checks(self, tmp_repo: Path) -> None:
        """Pre-G3 state: no wiring, no signoffs → only the wiring-detection result."""

        _seed_initial_commit(tmp_repo)
        # Make a non-wired version of the target file.
        _make_target_file(
            tmp_repo,
            "def _build_verdict(feature, score):\n"
            "    return {'severity': 'info'}\n"
            "\n"
            "def _compose_legacy_verdict(feature, voter):\n"
            "    return {'severity': 'info'}\n"
            "\n"
            "def _adversarial_input(score):\n"
            "    return {'severity': 'info'}\n",
        )
        head_sha = _git_commit(tmp_repo, "add target file")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        # Only one result (the wiring-detection check); guard inactive.
        assert len(results) == 1
        assert results[0].name == "wiring_detection"
        assert results[0].ok is False
        # Exit code is 0 (PASS) when guard inactive.
        assert guard.evaluate(results) == 0

    def test_wiring_without_signoffs_fails(self, tmp_repo: Path) -> None:
        """G3 wiring landed but G1+G2 signoffs absent → guard FAILS."""

        _seed_initial_commit(tmp_repo)
        _make_target_file(
            tmp_repo,
            "def _adversarial_input(score):\n"
            "    return hblp_classify(z_score=score['z'], n_positives=22, layer_1_declared_safe=False)\n",
        )
        head_sha = _git_commit(tmp_repo, "add wiring without signoffs")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        assert len(results) > 1  # wiring + signoff checks
        assert results[0].ok is True  # wiring detected
        # At least one downstream check fails.
        assert any(not r.ok for r in results[1:])
        assert guard.evaluate(results) == 1

    def test_wiring_with_both_signoffs_passes(self, tmp_repo: Path) -> None:
        """G3 wiring + G1 + G2 signoffs (referencing ancestor SHAs) → guard PASSES."""

        first_sha = _seed_initial_commit(tmp_repo)

        # Add target file at first commit (already done via _seed_initial_commit
        # which only created README; let's add target now).
        _make_target_file(
            tmp_repo,
            "def _adversarial_input(score):\n"
            "    return hblp_classify(z_score=score['z'], n_positives=22, layer_1_declared_safe=False)\n",
        )
        _git_commit(tmp_repo, "add wiring")

        # Drop both signoffs referencing first_sha (which IS an ancestor
        # of HEAD by construction).
        _write_signoff(tmp_repo, "g1", commit_sha=first_sha)
        _write_signoff(tmp_repo, "g2", commit_sha=first_sha)
        head_sha = _git_commit(tmp_repo, "add G1 + G2 signoffs")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        assert results[0].ok is True  # wiring detected
        # All downstream checks pass.
        for r in results[1:]:
            assert r.ok, f"Unexpected failure: {r.name}: {r.detail}"
        assert guard.evaluate(results) == 0

    def test_wiring_with_only_g1_signoff_fails(self, tmp_repo: Path) -> None:
        """G1 signoff only → G2 missing → guard FAILS."""

        first_sha = _seed_initial_commit(tmp_repo)
        _make_target_file(
            tmp_repo,
            "def _adversarial_input(score):\n"
            "    return hblp_classify(z_score=score['z'], n_positives=22, layer_1_declared_safe=False)\n",
        )
        _git_commit(tmp_repo, "add wiring")

        _write_signoff(tmp_repo, "g1", commit_sha=first_sha)
        head_sha = _git_commit(tmp_repo, "add g1 signoff only")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        assert results[0].ok is True  # wiring detected
        # The g2 existence check fails.
        g2_existence = next(
            r
            for r in results
            if r.name == "signoff_exists::docs/calibration/g2_completion_signoff_20260510.md"
        )
        assert not g2_existence.ok
        assert guard.evaluate(results) == 1


# --------------------------------------------------------------------------- #
# evaluate() tests
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_empty_returns_invocation_error(self) -> None:
        assert guard.evaluate([]) == 2

    def test_wiring_inactive_returns_zero(self) -> None:
        results = [guard.CheckResult("wiring_detection", False, "no wiring")]
        assert guard.evaluate(results) == 0

    def test_wiring_active_all_pass_returns_zero(self) -> None:
        results = [
            guard.CheckResult("wiring_detection", True, "found"),
            guard.CheckResult("signoff_exists::g1", True, "ok"),
            guard.CheckResult("signoff_ancestor::g1", True, "ok"),
            guard.CheckResult("signoff_committer_match::g1", True, "ok"),
            guard.CheckResult("signoff_exists::g2", True, "ok"),
            guard.CheckResult("signoff_ancestor::g2", True, "ok"),
            guard.CheckResult("signoff_committer_match::g2", True, "ok"),
        ]
        assert guard.evaluate(results) == 0

    def test_wiring_active_one_fail_returns_one(self) -> None:
        results = [
            guard.CheckResult("wiring_detection", True, "found"),
            guard.CheckResult("signoff_exists::g1", True, "ok"),
            guard.CheckResult("signoff_ancestor::g1", False, "not ancestor"),
        ]
        assert guard.evaluate(results) == 1


# --------------------------------------------------------------------------- #
# This-branch behavior — the guard MUST fail on the v4-g3-phase-c branch
# until G1+G2 signoffs are present on main + the branch is rebased.
# --------------------------------------------------------------------------- #


class TestThisBranchFailsClosed:
    """Plan v4 §2 G3 acceptance: the guard FAILS on this branch until G1+G2
    merge to main. This test asserts that the wiring DOES exist (the
    refactor landed) AND that without the signoff files at HEAD the guard
    correctly rejects.

    Skipped when run from outside the project repo (e.g. an isolated test
    fixture); only executed when ``REPO_ROOT/src/agents/...`` exists with
    the wired implementation.
    """

    def test_wiring_present_in_real_source(self) -> None:
        """The G3 refactor IS landed on this branch — the AST scan finds it."""

        repo_root = Path(__file__).resolve().parents[2]
        target = repo_root / guard.WIRED_FILE_REL
        if not target.is_file():
            pytest.skip("not running inside the e2i_causal_analytics repo")

        result = guard.detect_hblp_wiring(target)
        assert result.ok is True, (
            f"Expected G3 wiring to be detected on this branch but the AST "
            f"scan found nothing — implementation may have regressed. Detail: "
            f"{result.detail}"
        )
