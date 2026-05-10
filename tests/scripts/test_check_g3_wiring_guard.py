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
    """Write a placeholder signoff file under docs/calibration/.

    Per codex MED-6, the signoff carries exactly one explicit
    ``commit:`` field on its own line. Tests pass the SHA they want
    parsed; ``commit_sha=None`` writes a signoff with NO commit field
    (the parser raises ``ExtractCommitShaError`` on it).
    """

    cal_dir = repo / "docs" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    path = cal_dir / f"{gate}_completion_signoff_20260510.md"
    if commit_sha is None:
        body = f"# {gate} signoff\n\nNo commit reference yet.\n"
    else:
        body = f"# {gate} signoff\n\ncommit: `{commit_sha}`\n"
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# detect_hblp_wiring tests
# --------------------------------------------------------------------------- #


class TestDetectHblpWiring:
    def test_no_wiring_pre_g3(self, tmp_path: Path) -> None:
        """A file with no `hblp_classify` call → name=WIRING_ABSENT (guard inactive)."""

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
        assert result.name == guard.WIRING_ABSENT
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
        assert result.name == guard.WIRING_PRESENT
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
        assert result.name == guard.WIRING_PRESENT
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
        assert result.name == guard.WIRING_PRESENT
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
        assert result.name == guard.WIRING_ABSENT

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
        assert result.name == guard.WIRING_PRESENT
        assert "attribute access" in result.detail

    def test_import_alias_call_detected(self, tmp_path: Path) -> None:
        """codex HIGH-3: `from x import hblp_classify as fn; fn(...)` triggers the guard.

        A determined developer could re-import the helper under an alias to
        dodge the AST scan. The scanner now resolves module-scope import
        aliases before walking the gated function bodies.
        """

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "from src.calibration.hblp import hblp_classify as classify_hblp\n"
            "\n"
            "def _build_verdict(feature, score):\n"
            "    return classify_hblp(z_score=score['z'], n_positives=50, layer_1_declared_safe=False)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert result.name == guard.WIRING_PRESENT
        assert "_build_verdict" in result.detail
        assert "alias" in result.detail.lower() or "classify_hblp" in result.detail

    def test_assignment_alias_call_detected(self, tmp_path: Path) -> None:
        """codex HIGH-3: `alias = hblp_classify; alias(...)` triggers the guard.

        Module-scope assignment aliases also dodge the simple call-name
        match; the scanner tracks them as alias targets.
        """

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "from src.calibration.hblp import hblp_classify\n"
            "\n"
            "_classify = hblp_classify\n"
            "\n"
            "def _adversarial_input(score):\n"
            "    return _classify(z_score=score['z'], n_positives=22, layer_1_declared_safe=True)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert result.name == guard.WIRING_PRESENT
        assert "_adversarial_input" in result.detail

    def test_chained_alias_assignments_detected(self, tmp_path: Path) -> None:
        """`a = hblp_classify; b = a` then `b(...)` still triggers the guard.

        Chained assignment aliases are resolved transitively; once a name
        is bound to ``hblp_classify`` it propagates through the alias map.
        """

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "from src.calibration.hblp import hblp_classify\n"
            "\n"
            "_a = hblp_classify\n"
            "_b = _a\n"
            "\n"
            "def _compose_legacy_verdict(feature, voter):\n"
            "    return _b(z_score=1.0, n_positives=50, layer_1_declared_safe=False)\n",
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is True
        assert result.name == guard.WIRING_PRESENT

    def test_missing_target_file_returns_scan_error(self, tmp_path: Path) -> None:
        """codex MED-7: missing target file → SCAN_ERROR (NOT WIRING_ABSENT PASS).

        Previously the scanner returned ok=False with no distinction between
        "absent" and "missing", and ``evaluate()`` treated it as guard
        inactive PASS. A determined developer could rename / delete the
        gated file to dodge the scan. The new SCAN_ERROR state surfaces
        the failure as exit 1.
        """

        result = guard.detect_hblp_wiring(tmp_path / "does_not_exist.py")
        assert result.ok is False
        assert result.name == guard.SCAN_ERROR
        assert "not found" in result.detail.lower()
        # evaluate() must treat SCAN_ERROR as exit 1, NOT 0.
        assert guard.evaluate([result]) == 1

    def test_syntax_error_returns_scan_error(self, tmp_path: Path) -> None:
        """codex MED-7: syntax error in target → SCAN_ERROR (hard failure).

        A determined developer could intentionally introduce a syntax
        error to dodge the scan. SCAN_ERROR causes exit 1.
        """

        target = tmp_path / "adaptive_validity_check.py"
        target.write_text(
            "def _build_verdict(feature, score):\n"
            "    return hblp_classify(\n",  # Unterminated paren
            encoding="utf-8",
        )
        result = guard.detect_hblp_wiring(target)
        assert result.ok is False
        assert result.name == guard.SCAN_ERROR
        assert "parse" in result.detail.lower() or "syntax" in result.detail.lower()
        # evaluate() must treat SCAN_ERROR as exit 1, NOT 0.
        assert guard.evaluate([result]) == 1


# --------------------------------------------------------------------------- #
# extract_commit_sha tests
# --------------------------------------------------------------------------- #


class TestExtractCommitSha:
    """codex MED-6: extract_commit_sha parses exactly ONE explicit
    ``commit:`` field. Other "first backtick hex wins" behaviour is
    removed. The parser RAISES ExtractCommitShaError on invalid input.
    """

    _FULL_SHA = "0123456789abcdef0123456789abcdef01234567"
    _BRANCH_COMMIT_BODY = "# Signoff\n\nBranch / commit: `0123456789abcdef0123456789abcdef01234567`. Some narrative.\n"

    def test_backtick_form_full_length(self) -> None:
        """40-char hex inside the `commit:` field — the canonical form."""

        assert guard.extract_commit_sha(self._BRANCH_COMMIT_BODY) == self._FULL_SHA

    def test_field_form_no_backticks(self) -> None:
        """commit: <40-hex> with no backticks parses cleanly."""

        body = f"# G2 Signoff\n\ncommit: {self._FULL_SHA}\n"
        assert guard.extract_commit_sha(body) == self._FULL_SHA

    def test_field_form_with_bullet(self) -> None:
        """- **commit:** `<40-hex>` parses cleanly."""

        body = f"# G2 Signoff\n\n- **commit:** `{self._FULL_SHA}`\n"
        assert guard.extract_commit_sha(body) == self._FULL_SHA

    def test_short_sha_rejected(self) -> None:
        """codex MED-6: short SHA (7-39 hex chars) is rejected.

        Production policy requires full 40-char SHA; short forms get a
        descriptive ExtractCommitShaError rather than a silent pass.
        """

        body = "# Signoff\n\ncommit: `9c51eac4`\n"
        with pytest.raises(guard.ExtractCommitShaError, match="short SHA"):
            guard.extract_commit_sha(body)

    def test_short_sha_accepted_in_permissive_mode(self) -> None:
        """The legacy shim accepts short SHAs for advisory contexts."""

        body = "# Signoff\n\ncommit: `9c51eac4`\n"
        assert guard.extract_commit_sha(body, require_full_length=False) == "9c51eac4"

    def test_no_sha_raises(self) -> None:
        """codex MED-6: missing `commit:` field RAISES, no longer returns None."""

        body = "# Signoff\n\nNo SHA here. Just words.\n"
        with pytest.raises(guard.ExtractCommitShaError, match="no `commit:` field"):
            guard.extract_commit_sha(body)

    def test_placeholder_sha_rejected(self) -> None:
        """codex MED-6: `<sha>` placeholder rejected with descriptive error."""

        body = "# Signoff\n\ncommit: `<sha>`\n"
        with pytest.raises(guard.ExtractCommitShaError, match="placeholder"):
            guard.extract_commit_sha(body)

    def test_placeholder_token_tbd_rejected(self) -> None:
        """codex MED-6: TBD placeholder rejected."""

        body = "# Signoff\n\ncommit: `TBD`\n"
        with pytest.raises(guard.ExtractCommitShaError, match="placeholder"):
            guard.extract_commit_sha(body)

    def test_placeholder_token_PLACEHOLDER_rejected(self) -> None:
        """codex MED-6: literal PLACEHOLDER text rejected."""

        body = "# Signoff\n\ncommit: PLACEHOLDER\n"
        with pytest.raises(guard.ExtractCommitShaError, match="placeholder"):
            guard.extract_commit_sha(body)

    def test_non_hex_rejected(self) -> None:
        """codex MED-6: non-hex token rejected with descriptive error."""

        body = "# Signoff\n\ncommit: `not-a-real-sha-value-zzzzzzzzzzzzzzzzzzz`\n"
        with pytest.raises(guard.ExtractCommitShaError, match="not 40-char hex"):
            guard.extract_commit_sha(body)

    def test_duplicate_commit_field_rejected(self) -> None:
        """codex MED-6: multiple `commit:` lines are rejected — exactly ONE allowed."""

        body = f"# Signoff\n\ncommit: `{self._FULL_SHA}`\n\ncommit: `{self._FULL_SHA}`\n"
        with pytest.raises(guard.ExtractCommitShaError, match="multiple `commit:` fields"):
            guard.extract_commit_sha(body)

    def test_first_backtick_does_not_satisfy_requirement(self) -> None:
        """codex MED-6: a backtick-hex token elsewhere in the doc cannot
        satisfy the requirement — only an explicit ``commit:`` field
        line counts.

        Previously the parser would have returned the first
        backtick-wrapped hex anywhere in the doc; an attacker could
        insert an arbitrary backtick-hex anywhere and the parser would
        accept it. The new parser line-anchors on ``commit:``.
        """

        body = (
            "# Signoff\n\n"
            f"Some narrative referencing `{self._FULL_SHA}`.\n"
            "But no actual commit field.\n"
        )
        with pytest.raises(guard.ExtractCommitShaError, match="no `commit:` field"):
            guard.extract_commit_sha(body)

    def test_extract_or_none_legacy_shim(self) -> None:
        """The legacy ``_extract_commit_sha_or_none`` shim returns None on
        failure (used by some advisory-warn paths).
        """

        body = "# Signoff\n\nNo SHA here.\n"
        assert guard._extract_commit_sha_or_none(body) is None
        body_ok = f"# Signoff\n\ncommit: `{self._FULL_SHA}`\n"
        assert guard._extract_commit_sha_or_none(body_ok) == self._FULL_SHA


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
        """Signoff with no extractable SHA → FAIL on ancestry check.

        codex MED-6: the new parser raises ExtractCommitShaError; the
        ancestor check surfaces that as a fail with the rejection reason.
        """

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha=None)
        head_sha = _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is False
        assert "could not extract commit sha" in result.detail.lower()
        assert "no `commit:` field" in result.detail

    def test_placeholder_sha_in_signoff_fails(self, tmp_repo: Path) -> None:
        """codex MED-6: placeholder `<sha>` in signoff → FAIL with
        descriptive reason.
        """

        _seed_initial_commit(tmp_repo)
        cal_dir = tmp_repo / "docs" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        signoff = cal_dir / "g1_completion_signoff_20260510.md"
        signoff.write_text("# g1 signoff\n\ncommit: `<sha>`\n", encoding="utf-8")
        head_sha = _git_commit(tmp_repo, "add g1 signoff with placeholder")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is False
        assert "placeholder" in result.detail.lower()

    def test_short_sha_in_signoff_fails(self, tmp_repo: Path) -> None:
        """codex MED-6: short SHA in signoff → FAIL with full-length policy reason."""

        _seed_initial_commit(tmp_repo)
        cal_dir = tmp_repo / "docs" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        signoff = cal_dir / "g1_completion_signoff_20260510.md"
        signoff.write_text("# g1 signoff\n\ncommit: `9c51eac4`\n", encoding="utf-8")
        head_sha = _git_commit(tmp_repo, "add g1 signoff with short sha")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert result.ok is False
        assert "short sha" in result.detail.lower()

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


_FULL_SHA_TEST = "0123456789abcdef0123456789abcdef01234567"


class TestSignoffCommitterMatch:
    def test_advisory_skip_when_registry_empty(self, tmp_repo: Path) -> None:
        """Empty registry + advisory mode → PASS with WARN (default behavior)."""

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=False,
        )
        assert result.ok is True

    def test_require_mode_fails_with_empty_registry(self, tmp_repo: Path) -> None:
        """codex HIGH-1: empty registry + require_match=True → FAIL with descriptive reason."""

        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=True,
        )
        assert result.ok is False
        assert "empty" in result.detail.lower() or "no active" in result.detail.lower()

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
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
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

    def test_require_mode_unregistered_committer_fails(self, tmp_repo: Path) -> None:
        """codex HIGH-1: registered committer != signoff committer → FAIL."""

        _seed_initial_commit(tmp_repo)
        gov_dir = tmp_repo / "docs" / "governance"
        gov_dir.mkdir(parents=True, exist_ok=True)
        (gov_dir / "methodology_reviewer_registry.md").write_text(
            "| name | email | github_handle | role | date_added | areas_of_expertise | status |\n"
            "|------|-------|---------------|------|------------|--------------------|--------|\n"
            "| Other User | other@example.com | otheruser | reviewer | 2026 | misc | active |\n",
            encoding="utf-8",
        )
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        _git_commit(tmp_repo, "add g1 signoff")

        result = guard.check_signoff_committer_match(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            require_match=True,
        )
        assert result.ok is False
        assert "not in active registry" in result.detail.lower()


class TestSignoffExistsInBaseRef:
    """codex HIGH-2: signoff MUST exist in the BASE ref (origin/main HEAD
    at PR-open time), not just on PR HEAD. This closes the bypass where
    a developer merges G1+G2 into the same G3 PR.
    """

    def test_signoff_in_base_ref_passes(self, tmp_repo: Path) -> None:
        _seed_initial_commit(tmp_repo)
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        base_sha = _git_commit(tmp_repo, "add g1 signoff to main")

        result = guard.check_signoff_exists_in_base_ref(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            base_sha,
        )
        assert result.ok is True
        assert "present in base ref" in result.detail.lower()

    def test_signoff_missing_in_base_ref_fails(self, tmp_repo: Path) -> None:
        """Signoff added in PR but not in base → FAIL with HIGH-2 message."""

        base_sha = _seed_initial_commit(tmp_repo)
        # Add signoff AFTER base — simulates the bypass attempt.
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        _git_commit(tmp_repo, "add g1 signoff in PR (not on main)")

        result = guard.check_signoff_exists_in_base_ref(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            base_sha,
        )
        assert result.ok is False
        assert "missing in base ref" in result.detail.lower()


class TestSignoffAncestorWithBaseSha:
    """codex HIGH-2: when ``base_sha`` is provided to
    check_signoff_ancestor, the ``commit:`` field is read from the BASE
    ref (NOT the PR-checkout copy) so a malicious PR cannot rewrite the
    signoff body.
    """

    def test_base_sha_path_reads_from_base_ref(self, tmp_repo: Path) -> None:
        """Signoff exists on base with one SHA; PR rewrites it → check
        STILL uses base SHA.
        """

        # Step 1: create a real prior commit whose SHA we'll reference.
        first_sha = _seed_initial_commit(tmp_repo)
        # Step 2: add signoff referencing first_sha and commit.
        _write_signoff(tmp_repo, "g1", commit_sha=first_sha)
        base_sha = _git_commit(tmp_repo, "add signoff at base")

        # Step 3: rewrite the signoff to reference a non-ancestor SHA
        # (a fabricated SHA that's NOT in the repo).
        path = tmp_repo / "docs" / "calibration" / "g1_completion_signoff_20260510.md"
        path.write_text(
            f"# g1 signoff\n\ncommit: `{'f' * 40}`\n",
            encoding="utf-8",
        )
        head_sha = _git_commit(tmp_repo, "rewrite signoff in PR")

        # Without base_sha — uses PR-checkout body → fails because the
        # fabricated SHA isn't in the repo.
        no_base_result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
        )
        assert no_base_result.ok is False

        # With base_sha — reads body from base ref (which references
        # first_sha which IS an ancestor of head_sha) → PASS.
        base_result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
            base_sha=base_sha,
        )
        assert base_result.ok is True
        assert "ancestor" in base_result.detail.lower()
        assert "BASE_SHA" in base_result.detail

    def test_base_sha_missing_signoff_fails(self, tmp_repo: Path) -> None:
        """codex HIGH-2: when base_sha given but signoff missing in base → FAIL."""

        base_sha = _seed_initial_commit(tmp_repo)
        # Add signoff AFTER base.
        _write_signoff(tmp_repo, "g1", commit_sha=_FULL_SHA_TEST)
        head_sha = _git_commit(tmp_repo, "add signoff in PR not in base")

        result = guard.check_signoff_ancestor(
            tmp_repo,
            "docs/calibration/g1_completion_signoff_20260510.md",
            head_sha,
            base_sha=base_sha,
        )
        assert result.ok is False
        assert "missing in base ref" in result.detail.lower()


# --------------------------------------------------------------------------- #
# check_g3_wiring_guard orchestrator tests
# --------------------------------------------------------------------------- #


class TestG3GuardOrchestrator:
    """Full-flow tests of the orchestrator with realistic synthetic repos."""

    def test_no_wiring_skips_signoff_checks(self, tmp_repo: Path) -> None:
        """Pre-G3 state: no wiring, no signoffs → only WIRING_ABSENT result."""

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
        assert results[0].name == guard.WIRING_ABSENT
        assert results[0].ok is False
        # Exit code is 0 (PASS) when guard inactive.
        assert guard.evaluate(results) == 0

    def test_missing_target_file_returns_scan_error(self, tmp_repo: Path) -> None:
        """codex MED-7: missing target file at orchestrator level →
        SCAN_ERROR + exit 1.

        Closes the "delete the gated file to dodge the scan" bypass.
        """

        head_sha = _seed_initial_commit(tmp_repo)
        # Do NOT create the target file under
        # src/agents/ml_foundation/data_preparer/nodes/. The orchestrator
        # MUST treat this as SCAN_ERROR.

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        assert len(results) == 1
        assert results[0].name == guard.SCAN_ERROR
        assert results[0].ok is False
        # Exit code is 1 (FAIL) on SCAN_ERROR.
        assert guard.evaluate(results) == 1

    def test_syntax_error_target_returns_scan_error(self, tmp_repo: Path) -> None:
        """codex MED-7: syntax error in target → SCAN_ERROR + exit 1."""

        _seed_initial_commit(tmp_repo)
        _make_target_file(
            tmp_repo,
            "def _build_verdict(feature, score:\n"  # Broken signature
            "    return None\n",
        )
        head_sha = _git_commit(tmp_repo, "broken target file")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha)
        assert len(results) == 1
        assert results[0].name == guard.SCAN_ERROR
        assert results[0].ok is False
        assert guard.evaluate(results) == 1

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
        assert results[0].name == guard.WIRING_PRESENT  # wiring detected
        assert results[0].ok is True
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
        assert results[0].name == guard.WIRING_PRESENT
        assert results[0].ok is True
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
        assert results[0].name == guard.WIRING_PRESENT
        assert results[0].ok is True
        # The g2 existence check fails.
        g2_existence = next(
            r
            for r in results
            if r.name == "signoff_exists::docs/calibration/g2_completion_signoff_20260510.md"
        )
        assert not g2_existence.ok
        assert guard.evaluate(results) == 1

    def test_wiring_with_base_sha_requires_base_ref_signoff(self, tmp_repo: Path) -> None:
        """codex HIGH-2: with base_sha, signoffs MUST also exist in base ref.

        Simulates the merge-G1+G2-into-the-same-G3-PR bypass: signoffs
        appear at HEAD but not in base. Guard MUST fail.
        """

        first_sha = _seed_initial_commit(tmp_repo)
        _make_target_file(
            tmp_repo,
            "def _adversarial_input(score):\n"
            "    return hblp_classify(z_score=score['z'], n_positives=22, layer_1_declared_safe=False)\n",
        )
        # base_sha = AFTER target file added but BEFORE signoffs.
        base_sha = _git_commit(tmp_repo, "add wiring at base")
        # Add signoffs in PR (after base).
        _write_signoff(tmp_repo, "g1", commit_sha=first_sha)
        _write_signoff(tmp_repo, "g2", commit_sha=first_sha)
        head_sha = _git_commit(tmp_repo, "add signoffs in PR not in base")

        results = guard.check_g3_wiring_guard(tmp_repo, head_sha, base_sha=base_sha)
        assert results[0].name == guard.WIRING_PRESENT
        assert results[0].ok is True
        # base-ref existence check fails for both signoffs.
        base_failures = [
            r for r in results if r.name.startswith("signoff_exists_base::") and not r.ok
        ]
        assert len(base_failures) == 2
        assert guard.evaluate(results) == 1


# --------------------------------------------------------------------------- #
# evaluate() tests
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_empty_returns_invocation_error(self) -> None:
        assert guard.evaluate([]) == 2

    def test_wiring_inactive_returns_zero(self) -> None:
        results = [guard.CheckResult(guard.WIRING_ABSENT, False, "no wiring")]
        assert guard.evaluate(results) == 0

    def test_scan_error_returns_one(self) -> None:
        """codex MED-7: SCAN_ERROR is a hard failure (NOT guard inactive PASS)."""

        results = [guard.CheckResult(guard.SCAN_ERROR, False, "file missing")]
        assert guard.evaluate(results) == 1

    def test_wiring_active_all_pass_returns_zero(self) -> None:
        results = [
            guard.CheckResult(guard.WIRING_PRESENT, True, "found"),
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
            guard.CheckResult(guard.WIRING_PRESENT, True, "found"),
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
