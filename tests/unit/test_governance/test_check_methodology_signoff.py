"""Unit tests for ``scripts/check_methodology_signoff.py`` (Gate N3 scaffolding).

These tests exercise the validator on synthetic registry / CoI / sign-off
fixtures inside a temporary git repository, so the selection-rule check that
shells out to ``git log`` operates against a known commit graph.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

# --------------------------------------------------------------------------- #
# Module loading — scripts/ is not a Python package, so we import by path.
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_methodology_signoff.py"


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_methodology_signoff", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_methodology_signoff"] = module
    spec.loader.exec_module(module)
    return module


cms = _load_check_module()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _git(
    *args: str, cwd: Path, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a git command in ``cwd`` and return the completed process.

    ``env_overrides`` lets callers set GIT_COMMITTER_DATE alongside the
    ``--date`` flag for git commit; without that override git stamps the
    commit date as "now", which would defeat ``git log --since/--until``
    filtering in selection-rule tests.
    """

    import os

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )


VALID_REGISTRY = """\
# Methodology Reviewer Registry

Some prose.

| name | email | github_handle | role | date_added | areas_of_expertise | status | fingerprint |
|---|---|---|---|---|---|---|---|
| Alice Eligible | alice@example.com | alice | clinician | 2026-05-10 | methodology | active | `<TBD>` |
| Bob Conflicted | bob@example.com | bob | biostat | 2026-05-10 | methodology | active | `<TBD>` |
| Carol Inactive | carol@example.com | carol | advisor | 2026-05-10 | methodology | inactive | `<TBD>` |
"""


def _signoff_doc(
    *,
    handle: str = "alice",
    coi_sha: str = "abc1234567890def",
    coi_path: str = "docs/governance/coi_declarations/alice_20260510.md",
    include_signature: bool = True,
    omit_section: str | None = None,
) -> str:
    parts = [
        "# Optum Methodology Sign-off — 2026-05-10\n",
        "**Decision:** APPROVE.\n",
        "**Date:** 2026-05-10\n",
        "## Reviewer\n",
        f"- **GitHub handle:** @{handle}\n",
        "## Conflict-of-interest declaration\n",
        f"- **CoI document:** {coi_path}\n",
        f"- **CoI declaration commit SHA:** {coi_sha}\n",
        "## Methodology decision\n",
        "Approve relaxed window.\n",
    ]
    if include_signature:
        parts.append("## Cryptographic signature\n")
        parts.append("-----BEGIN PGP SIGNATURE-----\n")
        parts.append("FAKE-SIGNATURE-DATA-NOT-A-PLACEHOLDER-AAAAAAAAAA\n")
        parts.append("-----END PGP SIGNATURE-----\n")
    else:
        parts.append("## Cryptographic signature\n")
        parts.append("(missing)\n")
    text = "".join(parts)
    if omit_section is not None:
        # Drop everything from the heading onward to the next heading.
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        skipping = False
        for line in lines:
            if line.startswith(omit_section):
                skipping = True
                continue
            if skipping:
                if line.startswith("## ") and not line.startswith(omit_section):
                    skipping = False
                else:
                    continue
            out.append(line)
        text = "".join(out)
    return text


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Iterator[Path]:
    """Build a tmp git repo with the registry, a CoI for alice, and convert script."""

    repo = tmp_path / "repo"
    repo.mkdir()

    _git("init", "--initial-branch=main", "-q", cwd=repo)
    _git("config", "user.email", "ci@example.com", cwd=repo)
    _git("config", "user.name", "CI", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)

    # Layout: docs/governance/{registry, coi_declarations}, docs/results,
    # scripts/convert_optum_rwd.py.
    (repo / "docs" / "governance" / "coi_declarations").mkdir(parents=True)
    (repo / "docs" / "results").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)

    (repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        VALID_REGISTRY, encoding="utf-8"
    )
    # Seed a convert script so commits against it are meaningful.
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# stub\n", encoding="utf-8")

    _git("add", "-A", cwd=repo)
    # Initial commit BEFORE the named period — does NOT trigger selection rule.
    # Both author-date (--date) and committer-date (env override) must land
    # before 2026-04-15 because `git log --since/--until` filters on commit
    # date by default.
    _git(
        "-c",
        "user.email=alice@example.com",
        "-c",
        "user.name=Alice",
        "commit",
        "--date=2026-04-01T12:00:00",
        "-m",
        "initial",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-01T12:00:00"},
    )

    # H4: the CoI declaration must be added in a SEPARATE commit so the
    # first-add SHA derivation works. We add it as the second commit so
    # tests that need a resolvable SHA can read it via _coi_sha().
    (repo / "docs" / "governance" / "coi_declarations" / "alice_20260510.md").write_text(
        "# CoI Alice\n\nzero touches\n", encoding="utf-8"
    )
    _git("add", "docs/governance/coi_declarations/alice_20260510.md", cwd=repo)
    _git(
        "-c",
        "user.email=alice@example.com",
        "-c",
        "user.name=Alice",
        "commit",
        "--date=2026-04-02T12:00:00",
        "-m",
        "add CoI alice",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-02T12:00:00"},
    )

    yield repo


def _coi_sha(repo: Path, path: str = "docs/governance/coi_declarations/alice_20260510.md") -> str:
    """Return the first-add commit SHA for ``path`` in the fixture repo."""

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "log",
            "--diff-filter=A",
            "--follow",
            "--reverse",
            "--format=%H",
            "--",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    shas = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    assert shas, f"no first-add SHA for {path}"
    return shas[0]


@pytest.fixture
def fixture_repo_with_bob_conflict(fixture_repo: Path) -> Path:
    """Add a commit by ``bob@example.com`` to the convert script INSIDE the
    named period so the selection-rule check should fail for bob."""

    repo = fixture_repo
    convert = repo / "scripts" / "convert_optum_rwd.py"
    convert.write_text("# stub\n# bob touched\n", encoding="utf-8")
    _git("add", "scripts/convert_optum_rwd.py", cwd=repo)
    _git(
        "-c",
        "user.email=bob@example.com",
        "-c",
        "user.name=Bob",
        "commit",
        "--date=2026-04-20T12:00:00",
        "-m",
        "bob touched convert",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-20T12:00:00"},
    )
    return repo


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_filename_pattern_rejects_template():
    result = cms.check_filename(Path("/x/optum_methodology_signoff_template.md"))
    assert result.ok is False
    assert "template" in result.detail


def test_filename_pattern_accepts_dated_signoff():
    result = cms.check_filename(Path("/x/optum_methodology_signoff_20260510.md"))
    assert result.ok is True


def test_filename_pattern_accepts_dated_rejection():
    result = cms.check_filename(Path("/x/optum_methodology_rejection_20260510.md"))
    assert result.ok is True


def test_filename_pattern_rejects_random():
    result = cms.check_filename(Path("/x/random.md"))
    assert result.ok is False


def test_required_sections_signoff_complete():
    text = _signoff_doc()
    result = cms.check_required_sections(text, "signoff")
    assert result.ok is True


def test_required_sections_signoff_missing_methodology():
    text = _signoff_doc(omit_section="## Methodology decision")
    result = cms.check_required_sections(text, "signoff")
    assert result.ok is False
    assert "Methodology decision" in result.detail


def test_required_sections_signoff_missing_signature_section():
    text = _signoff_doc(omit_section="## Cryptographic signature")
    result = cms.check_required_sections(text, "signoff")
    assert result.ok is False


# --------------------------------------------------------------------------- #
# M4: PGP block render-paste taint + structural parse via gpg --list-packets.
# --------------------------------------------------------------------------- #


def test_signature_present_rejects_html_entity_taint():
    """A PGP block containing &amp; / &lt; / etc. is render-paste contaminated."""

    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "AAAA &amp; more data\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
    assert result.ok is False
    assert "tainted" in result.detail


def test_signature_present_rejects_jats_tag_taint():
    """A PGP block containing JATS tags is render-paste contaminated."""

    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "<jats:p>signature</jats:p>\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
    assert result.ok is False
    assert "tainted" in result.detail


def test_signature_present_rejects_html_p_tag_taint():
    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "<p>signature</p>\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
    assert result.ok is False


def test_pgp_block_taint_returns_first_token():
    block = "-----BEGIN PGP SIGNATURE-----\n&amp;\n-----END PGP SIGNATURE-----\n"
    assert cms._pgp_block_taint(block) == "&amp;"


def test_pgp_block_taint_returns_none_for_clean():
    block = "-----BEGIN PGP SIGNATURE-----\n\nfresh ascii armor\n-----END PGP SIGNATURE-----\n"
    assert cms._pgp_block_taint(block) is None


def test_signature_present_rejects_random_ascii_block():
    """M4: even a clean-looking armor with random non-base64 content fails
    the gpg --list-packets structural parse."""

    if shutil.which("gpg") is None:
        pytest.skip("gpg not on PATH; structural parse degrades to WARN")

    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "AAAARandomLooKingDataNotRealPGP\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
    assert result.ok is False
    assert "structural parse" in result.detail


def test_signature_present_template_placeholder_rejected():
    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "<signature blob>\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
    assert result.ok is False
    assert "placeholder" in result.detail


def test_signature_present_real_block_accepted(gpg_keyring: tuple[Path, str]):
    """M4: only a real, gpg-parseable PGP armor block passes presence check.

    The pre-M4 implementation accepted any string containing the BEGIN+END
    armor markers; this test now requires a real signed block.
    """

    home, _ = gpg_keyring
    payload = "demo payload\n"
    doc = _make_signed_signoff_doc(payload, home)
    result = cms.check_signature_present(doc)
    assert result.ok is True


def test_extract_handle_strips_at_sign():
    text = "- **GitHub handle:** @alice"
    assert cms.extract_handle(text) == "alice"


def test_extract_coi_sha_returns_value():
    text = "- **CoI declaration commit SHA:** abcdef1234567890"
    assert cms.extract_coi_sha(text) == "abcdef1234567890"


def test_extract_coi_sha_rejects_placeholder():
    # The extractor returns the placeholder verbatim; check_coi_referenced is
    # responsible for treating <sha> as invalid. Verify both behaviors.
    placeholder_text = "- **CoI declaration commit SHA:** `<sha>`"
    assert cms.extract_coi_sha(placeholder_text) == "<sha>"
    result = cms.check_coi_referenced(_signoff_doc(coi_sha="<sha>"))
    assert result.ok is False


def test_parse_registry_returns_three_rows(fixture_repo: Path):
    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    assert len(rows) == 3
    handles = {r.handle for r in rows}
    assert handles == {"alice", "bob", "carol"}
    statuses = {r.handle: r.status for r in rows}
    assert statuses["carol"] == "inactive"


def test_check_reviewer_registered_unknown_handle(fixture_repo: Path):
    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="zoe")
    result = cms.check_reviewer_registered(text, rows)
    assert result.ok is False
    assert "zoe" in result.detail


def test_check_reviewer_registered_inactive_handle(fixture_repo: Path):
    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="carol")
    result = cms.check_reviewer_registered(text, rows)
    assert result.ok is False
    assert "inactive" in result.detail


def test_check_reviewer_registered_active_handle(fixture_repo: Path):
    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="alice")
    result = cms.check_reviewer_registered(text, rows)
    assert result.ok is True


def test_selection_rule_alice_zero_touches(fixture_repo: Path):
    """Alice has no commits in [2026-04-15, 2026-05-10] → selection rule passes."""

    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, fixture_repo, rows)
    assert result.ok is True
    assert "alice@example.com" in result.detail


def test_selection_rule_bob_with_commit_in_window(
    fixture_repo_with_bob_conflict: Path,
):
    """Bob has a commit dated 2026-04-20 against convert_optum_rwd.py → selection rule fails."""

    rows = cms.parse_registry(
        fixture_repo_with_bob_conflict / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="bob")
    result = cms.check_selection_rule(text, fixture_repo_with_bob_conflict, rows)
    assert result.ok is False
    assert "convert_optum_rwd.py" in result.detail


def test_render_report_indicates_pass_and_fail():
    results = [
        cms.CheckResult("a", True, "ok"),
        cms.CheckResult("b", False, "boom"),
    ]
    text = cms.render_report(results)
    assert "[PASS] a" in text
    assert "[FAIL] b" in text


# --------------------------------------------------------------------------- #
# End-to-end test: a valid sign-off doc passes ALL checks.
# --------------------------------------------------------------------------- #


def test_check_signoff_full_success(
    fixture_repo: Path, gpg_keyring: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
):
    """Compose a valid registry + signoff and assert the full check passes.

    M4: requires a real PGP-signed payload because the signature_present
    check now invokes `gpg --list-packets` to confirm structural validity.
    """

    home, _ = gpg_keyring
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    # H4: use the real first-add SHA for alice's CoI declaration so the
    # coi_referenced sub-checks (path resolves + first-add commit) pass.
    real_sha = _coi_sha(fixture_repo)
    payload = _signoff_doc(handle="alice", coi_sha=real_sha, include_signature=False)
    # Replace the placeholder "(missing)" signature section with a real
    # PGP-signed armor block. The _make_signed_signoff_doc helper appends
    # a fresh "## Cryptographic signature" + armor; strip the "(missing)"
    # placeholder section first.
    payload_no_sig = payload.replace("## Cryptographic signature\n(missing)\n", "")
    doc_text = _make_signed_signoff_doc(payload_no_sig, home)
    signoff_path.write_text(doc_text, encoding="utf-8")

    # Prevent --require-signature path failure when toolchain absent.
    # Pin today so the iter-3 future-date check (NEW MED) doesn't reject a
    # doc dated 2026-05-20 against the system's current date.
    results = cms.check_signoff(
        signoff_path, fixture_repo, require_signature=False, today="2026-05-20"
    )
    failed = [r for r in results if not r.ok]
    assert failed == [], f"unexpected failures: {[(r.name, r.detail) for r in failed]}"


def test_check_signoff_pass4_low1_full_success_with_strict_signature_and_pinning(
    fixture_repo: Path, gpg_keyring: tuple[Path, str]
):
    """Codex pass-4 LOW-1 fix: end-to-end orchestrator test exercising
    the FULL strict-signature + pinning path INCLUDING CoI body sig.

    The pre-existing test_check_signoff_full_success passes
    require_signature=False AND no keyring_dir, so the new H1+H4
    pinning checks are essentially bypassed (advisory-mode passes).
    This test exercises the full chain:
      - Generate a real GPG keypair
      - Populate the registry with the resulting fingerprint
      - Sign the sign-off doc with that key (inline armor)
      - Sign the CoI body with that key (sibling .asc — H4 path)
      - Pass require_signature=True + keyring_dir
      - Assert ALL checks pass AND no signature_check_skipped fires

    This catches composition bugs that unit tests can't see.
    """

    import os as _os

    home, fingerprint = gpg_keyring
    # Replace the registry with one that has alice's fingerprint pinned.
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| Alice | alice@example.com | alice | clinician | 2026-05-10 | "
        f"methodology | active | {fingerprint} |\n"
    )
    (fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )

    # Sign the CoI body via sibling-asc detached signature — exercises
    # the H4 sibling-asc code path AND ensures the pinning check sees
    # signing_fingerprint from BOTH verify checks (sign-off + CoI body).
    coi_path = fixture_repo / "docs" / "governance" / "coi_declarations" / "alice_20260510.md"
    env = _os.environ.copy()
    env["GNUPGHOME"] = str(home)
    sib_sign = subprocess.run(
        [
            "gpg",
            "--batch",
            "--detach-sign",
            "--armor",
            "--output",
            str(coi_path) + ".asc",
            str(coi_path),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if sib_sign.returncode != 0:
        pytest.skip(f"gpg --detach-sign on CoI failed: {sib_sign.stderr}")

    real_sha = _coi_sha(fixture_repo)
    payload = _signoff_doc(handle="alice", coi_sha=real_sha, include_signature=False)
    payload_no_sig = payload.replace("## Cryptographic signature\n(missing)\n", "")
    doc_text = _make_signed_signoff_doc(payload_no_sig, home)
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(doc_text, encoding="utf-8")

    results = cms.check_signoff(
        signoff_path,
        fixture_repo,
        require_signature=True,
        keyring_dir=home,
        today="2026-05-20",
    )
    failed = [r for r in results if not r.ok]
    assert failed == [], f"unexpected failures: {[(r.name, r.detail) for r in failed]}"

    # Pass-4 LOW-1: NO signature_check_skipped should fire either.
    skipped = [r for r in results if r.signature_check_skipped]
    assert skipped == [], f"unexpected signature skips: {[(r.name, r.detail) for r in skipped]}"

    # Specific assertions: signature_verifies + pinning both PASSED with
    # the expected signing fingerprint.
    sig_result = next(r for r in results if r.name == "signature_verifies")
    assert sig_result.signing_fingerprint == fingerprint
    coi_sig_result = next(r for r in results if r.name == "coi_body_signature_verifies")
    assert coi_sig_result.signing_fingerprint == fingerprint
    pinning_result = next(r for r in results if r.name == "signing_fingerprint_matches_registry")
    assert pinning_result.ok is True
    assert "match" in pinning_result.detail


def test_check_signoff_pass4_low1_full_failure_when_signing_key_not_pinned(
    fixture_repo: Path, gpg_keyring: tuple[Path, str]
):
    """Pass-4 LOW-1 sibling: end-to-end FAIL when signing fp NOT pinned.

    Same setup but the registry pins a DIFFERENT fingerprint than the
    one that signed. The pinning check MUST fail with ok=False
    (NOT advisory pass).
    """

    home, fingerprint = gpg_keyring
    # Pin a WRONG fingerprint in the registry (not the one used to sign).
    wrong_fp = "DEADBEEF" + ("0" * 32)
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        f"| Alice | alice@example.com | alice | clinician | 2026-05-10 | "
        f"methodology | active | {wrong_fp} |\n"
    )
    (fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )

    real_sha = _coi_sha(fixture_repo)
    payload = _signoff_doc(handle="alice", coi_sha=real_sha, include_signature=False)
    payload_no_sig = payload.replace("## Cryptographic signature\n(missing)\n", "")
    doc_text = _make_signed_signoff_doc(payload_no_sig, home)
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(doc_text, encoding="utf-8")

    results = cms.check_signoff(
        signoff_path,
        fixture_repo,
        require_signature=True,
        keyring_dir=home,
        today="2026-05-20",
    )
    pinning_result = next(r for r in results if r.name == "signing_fingerprint_matches_registry")
    # Pinning MUST fail; the wrong fingerprint can't satisfy the
    # registered one.
    assert pinning_result.ok is False
    assert "do not match" in pinning_result.detail
    assert wrong_fp in pinning_result.detail
    assert fingerprint in pinning_result.detail


def test_check_signoff_missing_signature_section(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(
        _signoff_doc(handle="alice", omit_section="## Cryptographic signature"),
        encoding="utf-8",
    )
    results = cms.check_signoff(
        signoff_path, fixture_repo, require_signature=False, today="2026-05-20"
    )
    assert any(r.name == "required_sections" and not r.ok for r in results)


def test_check_signoff_unregistered_reviewer(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="zoe"), encoding="utf-8")
    results = cms.check_signoff(
        signoff_path, fixture_repo, require_signature=False, today="2026-05-20"
    )
    assert any(r.name == "reviewer_registered" and not r.ok for r in results)


def test_check_signoff_selection_rule_violation(
    fixture_repo_with_bob_conflict: Path,
):
    repo = fixture_repo_with_bob_conflict
    signoff_path = repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="bob"), encoding="utf-8")
    results = cms.check_signoff(signoff_path, repo, require_signature=False, today="2026-05-20")
    assert any(r.name == "selection_rule" and not r.ok for r in results)


def test_check_signoff_missing_coi_sha(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="alice", coi_sha="<sha>"), encoding="utf-8")
    results = cms.check_signoff(
        signoff_path, fixture_repo, require_signature=False, today="2026-05-20"
    )
    assert any(r.name == "coi_referenced" and not r.ok for r in results)


# --------------------------------------------------------------------------- #
# CI workflow YAML must parse.
# --------------------------------------------------------------------------- #


def test_workflow_yaml_parses():
    """The methodology_signoff_guard.yml workflow must be valid YAML.

    Issue #192 H3: the workflow architecture changed from a single
    inline-validator job to a thin caller (`identify`) + delegate
    (`validate`) that calls the reusable `methodology-signoff-validator.yml`
    workflow. The reusable workflow loads the validator script from the
    protected `main` ref. This test pins the new caller layout; the
    reusable workflow's contract is pinned by `TestReusableValidatorWorkflow`.
    """

    yaml = pytest.importorskip("yaml")
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"
    assert workflow_path.is_file(), f"workflow missing: {workflow_path}"
    parsed = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    # Top-level YAML on/jobs keys must be present. (PyYAML interprets bare
    # `on:` as the boolean True, so we accept either the string or the bool.)
    assert "on" in parsed or True in parsed
    assert "jobs" in parsed
    # H3: caller now has `identify` + `validate` jobs (validate delegates
    # to the reusable workflow).
    assert "identify" in parsed["jobs"], (
        "H3: caller must have an `identify` job that enumerates touched artifacts"
    )
    assert "validate" in parsed["jobs"], (
        "H3: caller must have a `validate` job that delegates to the reusable workflow"
    )
    # The validate job must `uses:` the reusable validator workflow, NOT
    # run python3 inline (that was the pre-H3 threat).
    validate_job = parsed["jobs"]["validate"]
    assert "uses" in validate_job, (
        "H3: validate job must use the reusable workflow (no inline python3)"
    )
    assert "methodology-signoff-validator.yml" in validate_job["uses"]


def test_workflow_has_has_files_boolean(tmp_path: Path):
    """M3: the workflow must write a `has_files` boolean output and gate on it."""

    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"
    text = workflow_path.read_text(encoding="utf-8")
    assert "has_files=true" in text, "workflow must write has_files=true"
    assert "has_files=false" in text, "workflow must write has_files=false"
    # Downstream gating uses needs.identify.outputs.has_files == 'true'
    # (the H3 split moved this from a step `if:` to a job `if:` since
    # the validate job now lives in a separate workflow).
    assert "needs.identify.outputs.has_files == 'true'" in text, (
        "H3: validate job must gate on identify.outputs.has_files"
    )
    # Whitespace-only lines must be stripped from the candidate file list.
    assert "grep -v '^[[:space:]]*$'" in text


def test_workflow_uses_base_ref_pinned_validator():
    """H3: the workflow architecture must pin the validator to a protected ref.

    Issue #192 H3 update: the historical mitigation was a `git show
    <base_sha>:scripts/check_methodology_signoff.py` fetch inside the
    same workflow. That has been replaced by a reusable-workflow split
    that does its own `actions/checkout@v4` of `main` for the validator
    script. This test pins the caller side; `TestReusableValidatorWorkflow`
    pins the reusable side.
    """

    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"
    text = workflow_path.read_text(encoding="utf-8")
    # H3 (post-#192): caller delegates to the reusable workflow which
    # loads the validator script from the protected ref.
    assert "uses: ./.github/workflows/methodology-signoff-validator.yml" in text, (
        "H3: caller must `uses:` the reusable validator workflow"
    )
    # Codex pass-2 LOW-1: validator_ref is now HARDCODED inside the
    # reusable workflow (env.VALIDATOR_PROTECTED_REF: 'main') rather
    # than passed as a caller input. Caller MUST NOT pass validator_ref.
    assert "validator_ref:" not in text, (
        "Codex pass-2 LOW-1: caller must not pass validator_ref — the "
        "ref is now hardcoded inside the reusable workflow to eliminate "
        "the caller-controlled-ref footgun"
    )


def test_script_exit_code_on_template():
    """End-to-end CLI test: the script returns non-zero on a template file."""

    if shutil.which("python3") is None:
        pytest.skip("python3 not on PATH in this test env")
    template = PROJECT_ROOT / "docs" / "results" / "optum_methodology_signoff_template.md"
    if not template.is_file():
        pytest.skip("template not present in this checkout")
    completed = subprocess.run(
        ["python3", str(SCRIPT_PATH), str(template)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "template" in completed.stdout


# --------------------------------------------------------------------------- #
# H1: signature verification — actual cryptographic verification with fixture
# GPG key. Tests gated on ``gpg`` being on PATH (skip otherwise — Ubuntu CI
# images carry it by default but local devboxes may not).
# --------------------------------------------------------------------------- #


@pytest.fixture
def gpg_keyring(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Generate a throwaway GPG keypair in an isolated GNUPGHOME.

    Yields ``(home_dir, fingerprint)``. Skips the test if gpg is unavailable
    or key generation fails (e.g. CI image lacks an entropy source).
    """

    if shutil.which("gpg") is None:
        pytest.skip("gpg not available on PATH")

    home = tmp_path / "gpghome"
    home.mkdir(mode=0o700)

    batch_input = (
        "%no-protection\n"
        "Key-Type: RSA\n"
        "Key-Length: 2048\n"
        "Name-Real: Test Reviewer\n"
        "Name-Email: test@example.com\n"
        "Expire-Date: 0\n"
        "%commit\n"
    )

    import os

    env = os.environ.copy()
    env["GNUPGHOME"] = str(home)
    gen = subprocess.run(
        ["gpg", "--batch", "--gen-key"],
        input=batch_input,
        capture_output=True,
        text=True,
        env=env,
    )
    if gen.returncode != 0:
        pytest.skip(f"gpg --gen-key failed in test env: {gen.stderr}")

    list_keys = subprocess.run(
        ["gpg", "--list-keys", "--with-colons", "--keyid-format=long"],
        capture_output=True,
        text=True,
        env=env,
    )
    fingerprint = ""
    for line in list_keys.stdout.splitlines():
        if line.startswith("fpr:"):
            fingerprint = line.split(":")[9]
            break
    if not fingerprint:
        pytest.skip("could not extract fingerprint from gpg --list-keys")

    yield home, fingerprint


def _make_signed_signoff_doc(
    payload: str,
    keyring_dir: Path,
) -> str:
    """Sign ``payload`` with the fixture key and embed the armored signature.

    Returns the full doc text: payload + ``## Cryptographic signature`` +
    armored sig. ``payload`` MUST end with a trailing newline so gpg's
    canonical-text-mode signing matches what the validator extracts.
    """

    import os

    env = os.environ.copy()
    env["GNUPGHOME"] = str(keyring_dir)

    sign = subprocess.run(
        ["gpg", "--batch", "--detach-sign", "--armor", "--output", "-"],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
    )
    if sign.returncode != 0:
        raise RuntimeError(f"gpg sign failed: {sign.stderr}")
    armor = sign.stdout

    return payload + "## Cryptographic signature\n\n" + armor


def test_signature_verifies_passes_for_valid_pgp(fixture_repo: Path, gpg_keyring: tuple[Path, str]):
    """A document signed by the fixture key under --require-signature passes."""

    home, _ = gpg_keyring
    payload = (
        "# Optum Methodology Sign-off — 2026-05-10\n"
        "## Reviewer\n- **GitHub handle:** @alice\n"
        "## Conflict-of-interest declaration\n"
        "- **CoI document:** docs/governance/coi_declarations/alice_20260510.md\n"
        "- **CoI declaration commit SHA:** abc1234567890def\n"
        "## Methodology decision\nApprove.\n"
    )
    doc_text = _make_signed_signoff_doc(payload, home)
    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(doc_text, encoding="utf-8")

    result = cms.check_signature_verifies(doc_path, require_signature=True, keyring_dir=home)
    assert result.ok is True, f"unexpected verify failure: {result.detail}"


def test_signature_verifies_fails_for_tampered_payload(
    fixture_repo: Path, gpg_keyring: tuple[Path, str]
):
    """A document whose payload is mutated after signing FAILS verification."""

    home, _ = gpg_keyring
    payload = (
        "# Optum Methodology Sign-off — 2026-05-10\n"
        "## Reviewer\n- **GitHub handle:** @alice\n"
        "## Conflict-of-interest declaration\n"
        "- **CoI document:** docs/governance/coi_declarations/alice_20260510.md\n"
        "- **CoI declaration commit SHA:** abc1234567890def\n"
        "## Methodology decision\nApprove.\n"
    )
    doc_text = _make_signed_signoff_doc(payload, home)
    # Tamper the payload AFTER the signature was generated.
    tampered = doc_text.replace("Approve.", "REJECT.")

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(tampered, encoding="utf-8")

    result = cms.check_signature_verifies(doc_path, require_signature=True, keyring_dir=home)
    assert result.ok is False
    assert "FAILED" in result.detail or "BAD" in result.detail.upper()


def test_signature_verifies_fails_when_signed_by_wrong_key(
    fixture_repo: Path, tmp_path: Path, gpg_keyring: tuple[Path, str]
):
    """A signature from a key NOT in the keyring fails verification."""

    home, _ = gpg_keyring

    # Create a SECOND, separate keyring with a different key. The doc is
    # signed by this OTHER key but verified against the FIRST keyring.
    other_home = tmp_path / "other_gpghome"
    other_home.mkdir(mode=0o700)
    import os

    env_other = os.environ.copy()
    env_other["GNUPGHOME"] = str(other_home)
    batch = (
        "%no-protection\n"
        "Key-Type: RSA\n"
        "Key-Length: 2048\n"
        "Name-Real: Other Reviewer\n"
        "Name-Email: other@example.com\n"
        "Expire-Date: 0\n"
        "%commit\n"
    )
    gen = subprocess.run(
        ["gpg", "--batch", "--gen-key"],
        input=batch,
        capture_output=True,
        text=True,
        env=env_other,
    )
    if gen.returncode != 0:
        pytest.skip(f"second gpg --gen-key failed: {gen.stderr}")

    payload = (
        "# Optum Methodology Sign-off — 2026-05-10\n"
        "## Reviewer\n- **GitHub handle:** @alice\n"
        "## Methodology decision\nApprove.\n"
    )
    doc_text = _make_signed_signoff_doc(payload, other_home)

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(doc_text, encoding="utf-8")

    # Verify against ``home`` (first keyring) which does NOT contain the
    # signing key.
    result = cms.check_signature_verifies(doc_path, require_signature=True, keyring_dir=home)
    assert result.ok is False
    assert "FAILED" in result.detail or "no public key" in result.detail.lower()


def test_signature_verifies_require_signature_fails_with_no_block(fixture_repo: Path):
    """No PGP / sigstore block + --require-signature → FAIL even if gpg present."""

    if shutil.which("gpg") is None:
        pytest.skip("gpg not on PATH; require_signature path only meaningful with toolchain")

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(
        "# Doc with no signature block\n## Reviewer\n@alice\n",
        encoding="utf-8",
    )
    result = cms.check_signature_verifies(doc_path, require_signature=True)
    assert result.ok is False


def test_signature_verifies_advisory_pass_when_keyring_missing(fixture_repo: Path, tmp_path: Path):
    """Codex pass-2 HIGH-1 fix: when --keyring-dir is set but the keyring
    is empty/missing, --require-signature returns ADVISORY pass + sig-skip.

    Without this, the workflow's ``strict_gpg: '0'`` rollout escape hatch
    is not reliable: the validator would still hit the gpg --verify path
    with an unprovisioned keyring, fail, and route through main()'s
    generic exit-1 branch — losing the routing signal AND breaking
    the documented advisory-mode contract during the operator-handoff
    window. The fix routes the missing-keyring case through the same
    signature_check_skipped flag the H4 advisory-pass branch uses.
    """

    if shutil.which("gpg") is None:
        pytest.skip("gpg not on PATH")

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(
        "# Sign-off\n"
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "FAKE\n"
        "-----END PGP SIGNATURE-----\n",
        encoding="utf-8",
    )
    # --keyring-dir set but EMPTY keyring (zero pubkeys imported).
    empty_keyring = tmp_path / "empty_keyring"
    empty_keyring.mkdir(mode=0o700)

    result = cms.check_signature_verifies(
        doc_path, require_signature=True, keyring_dir=empty_keyring
    )
    # Pass-2 HIGH-1: advisory pass + sig-skip (NOT generic fail).
    assert result.ok is True, f"expected advisory pass; got: {result.detail}"
    assert result.signature_check_skipped is True
    assert "missing/empty/unreadable" in result.detail
    assert "STRICT_GPG=1" in result.detail or "advisory" in result.detail.lower()


def test_signature_verifies_advisory_pass_when_keyring_dir_does_not_exist(
    fixture_repo: Path, tmp_path: Path
):
    """Codex pass-2 HIGH-1 fix sibling: nonexistent keyring dir = advisory."""

    if shutil.which("gpg") is None:
        pytest.skip("gpg not on PATH")

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    doc_path.write_text(
        "# Sign-off\n"
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "FAKE\n"
        "-----END PGP SIGNATURE-----\n",
        encoding="utf-8",
    )
    nonexistent = tmp_path / "does_not_exist"
    result = cms.check_signature_verifies(doc_path, require_signature=True, keyring_dir=nonexistent)
    assert result.ok is True
    assert result.signature_check_skipped is True


def test_signature_verifies_real_failure_when_keyring_present(
    fixture_repo: Path, gpg_keyring: tuple[Path, str]
):
    """Codex pass-2 HIGH-1: when keyring IS provisioned, real verify failures
    still produce ok=False (NOT advisory pass).

    Pin: the HIGH-1 fix MUST NOT silently swallow legitimate verify
    failures. The fix only kicks in for the "keyring not provisioned"
    case (operator-handoff window).
    """

    home, _ = gpg_keyring  # populated keyring
    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    # Doc with a malformed PGP block — verify will fail.
    doc_path.write_text(
        "# Sign-off\n"
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "GARBAGE\n"
        "-----END PGP SIGNATURE-----\n",
        encoding="utf-8",
    )
    result = cms.check_signature_verifies(doc_path, require_signature=True, keyring_dir=home)
    # Populated keyring → real verify failure → ok=False (NOT advisory).
    assert result.ok is False
    assert result.signature_check_skipped is False


def test_signature_verifies_pass3_med1_no_pgp_block_with_empty_keyring_no_advisory(
    fixture_repo: Path, tmp_path: Path
):
    """Codex pass-3 MED-1 fix pin: doc has NO PGP block, keyring is empty.

    Pre-fix: the keyring-advisory preflight ran BEFORE doc parsing, so
    a doc with no PGP block (e.g. a sigstore-only doc OR a doc with no
    signature at all) would advisory-pass-via-keyring-skip even though
    the real failure is "no signature block in the doc". Post-fix: the
    advisory only fires when the PGP path is actually being taken.
    """

    if shutil.which("gpg") is None:
        pytest.skip("gpg not on PATH")
    empty_keyring = tmp_path / "empty_keyring"
    empty_keyring.mkdir(mode=0o700)

    doc_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    # Doc with NO PGP block.
    doc_path.write_text("# Sign-off\nNo signature here.\n", encoding="utf-8")

    result = cms.check_signature_verifies(
        doc_path, require_signature=True, keyring_dir=empty_keyring
    )
    # Pass-3 MED-1: with no PGP block, the keyring advisory MUST NOT fire.
    # The legitimate "no signature block" failure must surface.
    assert result.ok is False
    assert result.signature_check_skipped is False
    # The detail must reference the missing block, not the empty keyring.
    assert "no extractable PGP armor block" in result.detail or "no PGP" in result.detail


def test_extract_pgp_armor_block_returns_full_block():
    text = "before\n-----BEGIN PGP SIGNATURE-----\nAAAA\n-----END PGP SIGNATURE-----\nafter\n"
    block = cms._extract_pgp_armor_block(text)
    assert block is not None
    assert block.startswith("-----BEGIN PGP SIGNATURE-----")
    assert block.endswith("-----END PGP SIGNATURE-----")
    assert "AAAA" in block


# --------------------------------------------------------------------------- #
# iter-3 NEW HIGH: sigstore verify-blob misuse — the verifier must pass
# the payload (not the bundle file) as the artifact arg to cosign.
# --------------------------------------------------------------------------- #


def test_verify_sigstore_bundle_no_payload_warns_about_known_broken():
    """iter-3 NEW HIGH: invoking _verify_sigstore_bundle without payload
    triggers the legacy degraded path that verifies the bundle as its
    own artifact — and the function must explicitly warn about it in
    the detail string.

    The test only asserts the warn-text presence (not the exit code,
    which depends on whether cosign is available on the test runner).
    """

    if shutil.which("cosign") is None:
        pytest.skip("cosign not on PATH; degraded-path warn test only meaningful with toolchain")

    bundle_json = (
        '{"base64Signature": "AAAA", "cert": "BBBB", '
        '"rekorBundle": {"Payload": {"body": "CCCC"}}, "signatures": ["AAAA"]}'
    )
    ok, detail = cms._verify_sigstore_bundle(bundle_json, payload=None)
    # When cosign is on PATH but no real signing materials are wired, ok
    # is False, but the WARN string must be present regardless.
    assert "KNOWN BROKEN" in detail or "WARN" in detail


def test_verify_sigstore_bundle_with_payload_separates_artifact_from_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """iter-3 NEW HIGH: with a payload supplied, cosign is invoked with
    the artifact path DIFFERENT from the bundle path.

    We monkeypatch subprocess.run to capture the actual cosign argv and
    assert the artifact arg points to a different file than --bundle.
    """

    if shutil.which("cosign") is None:
        # Monkeypatch shutil.which so the cosign path is taken without a
        # real cosign binary; we intercept subprocess.run before it
        # actually executes.
        monkeypatch.setattr(
            cms.shutil, "which", lambda name: "/fake/cosign" if name == "cosign" else None
        )

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)

        # Return a dummy CompletedProcess-like object.
        class _CP:
            returncode = 0
            stdout = ""
            stderr = "fake-cosign-ok"

        return _CP()

    monkeypatch.setattr(cms.subprocess, "run", fake_run)

    bundle_json = '{"signatures": ["AAAA"], "cert": "BBBB"}'
    payload = "## Reviewer\n@alice\n## Methodology decision\nApprove.\n"
    ok, _ = cms._verify_sigstore_bundle(bundle_json, payload=payload)
    assert ok is True

    cmd = captured["cmd"]
    # cmd shape: ["cosign", "verify-blob", "--bundle", <bundle_path>,
    #            "--insecure-ignore-tlog", <artifact_path>]
    assert cmd[0] == "cosign"
    assert cmd[1] == "verify-blob"
    bundle_idx = cmd.index("--bundle") + 1
    bundle_path = cmd[bundle_idx]
    # Artifact path is the trailing positional.
    artifact_path = cmd[-1]
    assert bundle_path != artifact_path, (
        "iter-3 NEW HIGH: cosign artifact path must NOT be the bundle file "
        f"(got bundle={bundle_path!r} artifact={artifact_path!r})"
    )


# --------------------------------------------------------------------------- #
# H2: selection-rule expansion — gh PR query, CoI declared-PR parse,
# warnings-vs-violations distinction.
# --------------------------------------------------------------------------- #


def test_parse_coi_declared_prs_extracts_json_array():
    coi_text = (
        "## Evidence\n\nSome prose.\n\n"
        "```json\n"
        '[{"number": 131, "title": "x", "files": [{"path": "scripts/foo.py"}]}]\n'
        "```\n"
        "More prose.\n"
    )
    parsed = cms._parse_coi_declared_prs(coi_text)
    assert len(parsed) == 1
    assert parsed[0]["number"] == 131


def test_parse_coi_declared_prs_returns_empty_on_no_array():
    coi_text = "## Evidence\n\nNo JSON here.\n"
    assert cms._parse_coi_declared_prs(coi_text) == []


def test_parse_coi_declared_prs_returns_empty_on_malformed_json():
    coi_text = "## Evidence\n```json\n[not valid json,]\n```\n"
    assert cms._parse_coi_declared_prs(coi_text) == []


def test_selection_rule_coi_self_declared_overlap_fails(fixture_repo: Path):
    """Reviewer's own CoI declares a PR that touches the subject file -> FAIL."""

    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="alice")
    coi_text = (
        "## Evidence\n\n"
        "```json\n"
        '[{"number": 999, "title": "alice touched convert", '
        '"files": [{"path": "scripts/convert_optum_rwd.py"}]}]\n'
        "```\n"
    )
    result = cms.check_selection_rule(text, fixture_repo, rows, coi_text=coi_text)
    assert result.ok is False
    assert "coi-self-declared" in result.detail
    assert "999" in result.detail


def test_selection_rule_coi_unrelated_overlap_passes(fixture_repo: Path):
    """CoI declares a PR but it touches an unrelated file -> does NOT fail."""

    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="alice")
    coi_text = (
        "## Evidence\n\n"
        "```json\n"
        '[{"number": 999, "title": "unrelated", '
        '"files": [{"path": "src/something/else.py"}]}]\n'
        "```\n"
    )
    result = cms.check_selection_rule(text, fixture_repo, rows, coi_text=coi_text)
    assert result.ok is True


# --------------------------------------------------------------------------- #
# H4: CoI SHA resolution + first-add commit + filename handle match.
# --------------------------------------------------------------------------- #


def test_coi_referenced_sha_must_resolve_in_repo(fixture_repo: Path):
    """A CoI SHA that does not exist as an object in the repo FAILS."""

    text = _signoff_doc(handle="alice", coi_sha="0000000000000000000000000000000000000000")
    result = cms.check_coi_referenced(text, repo_root=fixture_repo)
    assert result.ok is False
    assert "do not resolve" in result.detail or "git cat-file" in result.detail


def test_coi_referenced_sha_must_be_first_add(fixture_repo: Path):
    """A SHA that resolves but is NOT the first-add commit FAILS."""

    # Make a second commit that modifies the CoI declaration.
    coi_path = fixture_repo / "docs" / "governance" / "coi_declarations" / "alice_20260510.md"
    coi_path.write_text("# CoI Alice (modified)\n\nzero touches\n", encoding="utf-8")
    _git("add", "docs/governance/coi_declarations/alice_20260510.md", cwd=fixture_repo)
    _git(
        "-c",
        "user.email=alice@example.com",
        "-c",
        "user.name=Alice",
        "commit",
        "--date=2026-04-03T12:00:00",
        "-m",
        "modify CoI",
        cwd=fixture_repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-03T12:00:00"},
    )
    # HEAD is now the modify commit; reading HEAD gives the modify SHA.
    head_sha = subprocess.run(
        ["git", "-C", str(fixture_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    text = _signoff_doc(handle="alice", coi_sha=head_sha)
    result = cms.check_coi_referenced(text, repo_root=fixture_repo)
    assert result.ok is False
    assert "first-add" in result.detail


def test_coi_referenced_filename_handle_mismatch(fixture_repo: Path):
    """If the CoI filename is e.g. bob_20260510.md but reviewer is alice → FAIL."""

    text = _signoff_doc(
        handle="alice",
        coi_sha=_coi_sha(fixture_repo),
        coi_path="docs/governance/coi_declarations/bob_20260510.md",
    )
    result = cms.check_coi_referenced(text, repo_root=fixture_repo)
    assert result.ok is False
    assert "filename" in result.detail or "handle" in result.detail


def test_coi_referenced_filename_must_be_dated(fixture_repo: Path):
    """CoI filename without _<YYYYMMDD>.md suffix FAILS."""

    text = _signoff_doc(
        handle="alice",
        coi_sha=_coi_sha(fixture_repo),
        coi_path="docs/governance/coi_declarations/alice.md",
    )
    result = cms.check_coi_referenced(text, repo_root=fixture_repo)
    assert result.ok is False
    assert "<handle>_<YYYYMMDD>.md" in result.detail or "filename" in result.detail


def test_coi_referenced_resolves_with_real_sha(fixture_repo: Path):
    """A CoI declaration that resolves AND is the first-add commit PASSES."""

    text = _signoff_doc(handle="alice", coi_sha=_coi_sha(fixture_repo))
    result = cms.check_coi_referenced(text, repo_root=fixture_repo)
    assert result.ok is True
    assert "WARN" not in result.detail


def test_coi_referenced_no_repo_root_warns_only(fixture_repo: Path):
    """Without repo_root, the SHA-resolves and first-add subchecks WARN, not FAIL."""

    text = _signoff_doc(handle="alice", coi_sha="abc1234567890def")
    # No repo_root passed — git checks are skipped with a WARN.
    result = cms.check_coi_referenced(text)
    assert result.ok is True
    assert "WARN" in result.detail


# --------------------------------------------------------------------------- #
# M1: registry email-alias support — selection rule checks every alias.
# --------------------------------------------------------------------------- #


def test_parse_registry_supports_email_aliases(tmp_path: Path):
    """A registry email cell with comma-separated addresses produces
    a `ReviewerInfo` whose `emails` tuple contains every alias."""

    registry_text = (
        "# Reviewers\n\n"
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| Alice | alice@example.com, alice@oldjob.com | alice | "
        "clinician | 2026-05-10 | methodology | active | `<TBD>` |\n"
    )
    reg = tmp_path / "registry.md"
    reg.write_text(registry_text, encoding="utf-8")

    rows = cms.parse_registry(reg)
    assert len(rows) == 1
    assert rows[0].email == "alice@example.com"
    assert rows[0].emails == ("alice@example.com", "alice@oldjob.com")


def test_pass5_med1_parse_registry_with_warnings_surfaces_malformed_rows(
    tmp_path: Path,
):
    """Codex pass-5 MED-1 fix: parse_registry_with_warnings emits warnings
    for pipe-delimited rows with wrong column counts.

    Pre-fix: malformed rows were silently skipped, which after pass-4
    HIGH-1 row aggregation in check_selection_rule could DELETE
    disqualifying-evidence rows from matching → CoI bypass. Post-fix:
    malformed rows are tracked + surfaced so the orchestrator can
    fail-closed.
    """

    # 2 well-formed rows + 1 malformed (extra | in areas_of_expertise).
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| Alice | alice@example.com | alice | clinician | 2026-05-10 | "
        "methodology | active | `<TBD>` |\n"
        "| Bob | bob@example.com | bob | biostat | 2026-05-10 | "
        "methodology|extra-pipe | active | `<TBD>` |\n"
        "| Carol | carol@example.com | carol | advisor | 2026-05-10 | "
        "methodology | inactive | `<TBD>` |\n"
    )
    reg = tmp_path / "r.md"
    reg.write_text(registry_text, encoding="utf-8")
    rows, warnings = cms.parse_registry_with_warnings(reg)
    # Bob's row has 9 cells (extra |); Alice + Carol parse cleanly.
    assert len(rows) == 2
    assert {r.handle for r in rows} == {"alice", "carol"}
    # Bob's malformed row produces a warning.
    assert len(warnings) == 1
    assert "9 cells" in warnings[0] or "expected 8" in warnings[0]
    assert "bob" in warnings[0].lower() or "Bob" in warnings[0]


def test_pass5_med1_orchestrator_fails_on_malformed_registry(tmp_path: Path):
    """Pass-5 MED-1: orchestrator must FAIL registry_loaded on parser warnings.

    Without this, the malformed-row → silent-skip → CoI-bypass path
    survives in the full check_signoff orchestrator.
    """

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "docs" / "results").mkdir(parents=True)
    (repo / "docs" / "governance").mkdir(parents=True)
    # Registry with a malformed row.
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| A | a@ex.com | alice | clinician | 2026-05-10 | "
        "methodology|extra | active | `<TBD>` |\n"
    )
    (repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )
    # Minimal valid sign-off doc.
    signoff_path = repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="alice"), encoding="utf-8")

    results = cms.check_signoff(signoff_path, repo, today="2026-05-20")
    # registry_loaded MUST be False; downstream checks not run.
    reg_check = next(r for r in results if r.name == "registry_loaded")
    assert reg_check.ok is False
    assert "malformed" in reg_check.detail


# --------------------------------------------------------------------------- #
# M2: sign-off age limit — reject artifacts older than max_age_days vs today.
# --------------------------------------------------------------------------- #


def test_signoff_age_passes_within_window():
    """Doc dated 2026-05-01 vs today 2026-05-10 is 9 days old → PASS at 30d."""

    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_20260501.md"),
        today="2026-05-10",
        max_age_days=30,
    )
    assert result.ok is True


def test_signoff_age_fails_outside_window():
    """Doc dated 2026-04-01 vs today 2026-05-10 is 39 days old → FAIL at 30d."""

    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_20260401.md"),
        today="2026-05-10",
        max_age_days=30,
    )
    assert result.ok is False
    assert "older than today" in result.detail


def test_signoff_age_rejects_future_dated_beyond_tolerance():
    """iter-3 NEW MED: doc dated 22 days in the future → FAIL.

    Future-dated artifacts must be rejected; reviewers cannot pre-date
    sign-offs to evade the max-age window or claim review of work that
    has not yet happened. A small 1-day tolerance covers timezone-skew
    at the day boundary.
    """

    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_20260601.md"),
        today="2026-05-10",
        max_age_days=30,
    )
    assert result.ok is False
    assert "future" in result.detail


def test_signoff_age_tolerates_one_day_future_skew():
    """iter-3 NEW MED: a doc dated 1 day ahead is tolerated for TZ-skew."""

    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_20260511.md"),
        today="2026-05-10",
        max_age_days=30,
    )
    assert result.ok is True


def test_signoff_age_rejects_two_day_future_skew():
    """iter-3 NEW MED: a doc dated 2 days ahead is OUTSIDE the 1-day tolerance."""

    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_20260512.md"),
        today="2026-05-10",
        max_age_days=30,
    )
    assert result.ok is False
    assert "future" in result.detail


def test_signoff_age_rejects_unparseable_filename():
    result = cms.check_signoff_age(
        Path("/x/optum_methodology_signoff_99999999.md"),
        today="2026-05-10",
    )
    assert result.ok is False


def test_signoff_age_rejects_random_filename():
    result = cms.check_signoff_age(
        Path("/x/random.md"),
        today="2026-05-10",
    )
    assert result.ok is False


def test_check_signoff_includes_age_check_failure(fixture_repo: Path):
    """A stale-dated sign-off artifact (40 days old) FAILS the age check
    even when every other field is valid."""

    real_sha = _coi_sha(fixture_repo)
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260301.md"
    signoff_path.write_text(_signoff_doc(handle="alice", coi_sha=real_sha), encoding="utf-8")
    results = cms.check_signoff(
        signoff_path,
        fixture_repo,
        require_signature=False,
        today="2026-05-10",
        max_age_days=30,
    )
    age_failures = [r for r in results if r.name == "signoff_age" and not r.ok]
    assert age_failures, f"expected signoff_age failure, got {[r.name for r in results]}"


def test_selection_rule_catches_alias_commit(tmp_path: Path):
    """A reviewer with an alias declared in the registry whose alias
    authored a commit in-window is caught by the selection rule."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--initial-branch=main", "-q", cwd=repo)
    _git("config", "user.email", "ci@example.com", cwd=repo)
    _git("config", "user.name", "CI", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# stub\n", encoding="utf-8")
    (repo / "docs" / "governance").mkdir(parents=True)

    # Registry with TWO email aliases for alice.
    registry_text = (
        "# Reviewers\n\n"
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| Alice | alice@example.com; alice@oldjob.com | alice | "
        "clinician | 2026-05-10 | methodology | active | `<TBD>` |\n"
    )
    (repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )
    _git("add", "-A", cwd=repo)
    _git(
        "-c",
        "user.email=ci@example.com",
        "-c",
        "user.name=CI",
        "commit",
        "--date=2026-04-01T12:00:00",
        "-m",
        "initial",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-01T12:00:00"},
    )

    # Alice commits under her ALIAS (alice@oldjob.com), NOT the primary.
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# touched\n", encoding="utf-8")
    _git("add", "scripts/convert_optum_rwd.py", cwd=repo)
    _git(
        "-c",
        "user.email=alice@oldjob.com",
        "-c",
        "user.name=Alice",
        "commit",
        "--date=2026-04-20T12:00:00",
        "-m",
        "alice touched via alias",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-20T12:00:00"},
    )

    rows = cms.parse_registry(repo / "docs" / "governance" / "methodology_reviewer_registry.md")
    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, repo, rows)
    assert result.ok is False
    assert "alice@oldjob.com" in result.detail


def test_selection_rule_aggregates_duplicate_active_rows(tmp_path: Path):
    """Codex pass-2 MED-1: production registry has 2 rows for handle
    `enunezvn`. The selection-rule check MUST union the email aliases
    from BOTH rows so a commit authored under EITHER identity is caught.

    Pre-fix: ``matching[0]`` only consulted row 0's emails — a commit
    authored under row 1's no-reply email would NOT be caught.
    """

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--initial-branch=main", "-q", cwd=repo)
    _git("config", "user.email", "ci@example.com", cwd=repo)
    _git("config", "user.name", "CI", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# stub\n", encoding="utf-8")
    (repo / "docs" / "governance").mkdir(parents=True)

    # TWO active rows with DIFFERENT emails for the same handle.
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| Reviewer Canonical | alice@example.com | alice | clinician | "
        "2026-05-10 | methodology | active | `<TBD>` |\n"
        "| Reviewer No-Reply | 12345+alice@users.noreply.github.com | alice | "
        "clinician | 2026-05-10 | methodology | active | `<TBD>` |\n"
    )
    (repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )
    _git("add", "-A", cwd=repo)
    _git(
        "-c",
        "user.email=ci@example.com",
        "-c",
        "user.name=CI",
        "commit",
        "--date=2026-04-01T12:00:00",
        "-m",
        "initial",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-01T12:00:00"},
    )

    # Commit authored under ROW-2's no-reply email, NOT the canonical one.
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# touched\n", encoding="utf-8")
    _git("add", "scripts/convert_optum_rwd.py", cwd=repo)
    _git(
        "-c",
        "user.email=12345+alice@users.noreply.github.com",
        "-c",
        "user.name=Alice",
        "commit",
        "--date=2026-04-20T12:00:00",
        "-m",
        "alice touched via no-reply identity",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-20T12:00:00"},
    )

    rows = cms.parse_registry(repo / "docs" / "governance" / "methodology_reviewer_registry.md")
    assert len(rows) == 2
    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, repo, rows)
    # Pass-2 MED-1: with row aggregation, the no-reply email is checked.
    assert result.ok is False, (
        "pass-2 MED-1: aggregating across rows must catch commits authored "
        "under any active row's email"
    )
    assert "12345+alice@users.noreply.github.com" in result.detail


def test_selection_rule_pass4_h1_recused_row_email_still_catches_conflict(tmp_path: Path):
    """Codex pass-4 HIGH-1 fix pin: selection rule MUST aggregate across
    ALL rows (including recused/inactive) for CoI evidence.

    The pass-3 active-only filter introduced a CoI bypass: a reviewer
    with [recused row carrying disqualifying email + active row
    carrying clean email] could slip past. Pass-4 HIGH-1 reverted to
    full-row aggregation for selection-rule (active-only filter
    retained ONLY for fingerprint pinning + reviewer-registration).

    Scenario: alice has 2 rows — 1 active (clean email), 1 recused
    (email that authored the disqualifying commit). The selection
    rule MUST catch the recused-email commit.
    """

    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "--initial-branch=main", "-q", cwd=repo)
    _git("config", "user.email", "ci@example.com", cwd=repo)
    _git("config", "user.name", "CI", cwd=repo)
    _git("config", "commit.gpgsign", "false", cwd=repo)
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# stub\n", encoding="utf-8")
    (repo / "docs" / "governance").mkdir(parents=True)

    # ACTIVE row with clean email + RECUSED row with disqualifying email.
    registry_text = (
        "| name | email | github_handle | role | date_added | "
        "areas_of_expertise | status | fingerprint |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| Alice (current) | alice-current@example.com | alice | clinician | "
        "2026-05-10 | methodology | active | `<TBD>` |\n"
        "| Alice (recused) | alice-old@example.com | alice | clinician | "
        "2026-04-01 | methodology | recused | `<TBD>` |\n"
    )
    (repo / "docs" / "governance" / "methodology_reviewer_registry.md").write_text(
        registry_text, encoding="utf-8"
    )
    _git("add", "-A", cwd=repo)
    _git(
        "-c",
        "user.email=ci@example.com",
        "-c",
        "user.name=CI",
        "commit",
        "--date=2026-04-01T12:00:00",
        "-m",
        "initial",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-01T12:00:00"},
    )

    # The DISQUALIFYING commit is authored by the RECUSED row's email
    # IN-WINDOW.
    (repo / "scripts" / "convert_optum_rwd.py").write_text("# touched\n", encoding="utf-8")
    _git("add", "scripts/convert_optum_rwd.py", cwd=repo)
    _git(
        "-c",
        "user.email=alice-old@example.com",
        "-c",
        "user.name=Alice (recused)",
        "commit",
        "--date=2026-04-20T12:00:00",
        "-m",
        "alice (recused) touched convert in-window",
        cwd=repo,
        env_overrides={"GIT_COMMITTER_DATE": "2026-04-20T12:00:00"},
    )

    rows = cms.parse_registry(repo / "docs" / "governance" / "methodology_reviewer_registry.md")
    assert len(rows) == 2
    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, repo, rows)
    # Pass-4 HIGH-1: recused-row email MUST be caught.
    assert result.ok is False, (
        "pass-4 HIGH-1: selection-rule MUST aggregate across recused/inactive "
        "rows for CoI evidence; got pass when commit by recused email "
        "in-window should fail"
    )
    assert "alice-old@example.com" in result.detail


def test_reviewer_registered_pass4_med1_inactive_first_then_active_passes(
    tmp_path: Path,
):
    """Codex pass-4 MED-1 fix pin: check_reviewer_registered must walk
    ALL matching rows and PASS if any is active.

    Pre-fix: returned on the FIRST matching row's status. So a registry
    with [inactive row, active row] (in that order) for the same handle
    would FAIL. The registry is documented as append-only with status
    transitions encoded as new rows, so this scenario IS expected.
    """

    registry = [
        cms.ReviewerInfo(
            handle="alice",
            email="alice@example.com",
            status="inactive",  # ← appears FIRST
            emails=("alice@example.com",),
        ),
        cms.ReviewerInfo(
            handle="alice",
            email="alice@example.com",
            status="active",  # ← active row appears SECOND
            emails=("alice@example.com",),
        ),
    ]
    text = _signoff_doc(handle="alice")
    result = cms.check_reviewer_registered(text, registry)
    # Pass-4 MED-1: must walk all rows; an active row exists → PASS.
    assert result.ok is True, (
        f"pass-4 MED-1: must PASS when at least one active row exists; got {result.detail}"
    )
    assert "active" in result.detail
    # Helpful detail: total / active row counts surfaced.
    assert "1/2 rows active" in result.detail or "active" in result.detail


def test_reviewer_registered_pass4_med1_only_inactive_rows_fails(tmp_path: Path):
    """Pass-4 MED-1: matches exist but ZERO active → FAIL with clear detail."""

    registry = [
        cms.ReviewerInfo(
            handle="alice",
            email="alice@example.com",
            status="inactive",
            emails=("alice@example.com",),
        ),
        cms.ReviewerInfo(
            handle="alice",
            email="alice@example.com",
            status="recused",
            emails=("alice@example.com",),
        ),
    ]
    text = _signoff_doc(handle="alice")
    result = cms.check_reviewer_registered(text, registry)
    assert result.ok is False
    assert "no active rows" in result.detail
    # Both statuses surfaced for operator diagnostics.
    assert "inactive" in result.detail and "recused" in result.detail


def test_selection_rule_gh_missing_emits_warning(fixture_repo: Path, monkeypatch):
    """When gh is unavailable, selection rule still PASSES on git-log but:

    iter-3 M1:
      * a CRITICAL warning is in the detail string,
      * ``provenance_check_skipped=True`` is set on the CheckResult so
        the caller (CI) can decide to fail-closed.
    """

    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    # Force gh to be reported missing.
    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "gh":
            return None
        return real_which(name)

    monkeypatch.setattr(cms.shutil, "which", fake_which)

    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, fixture_repo, rows)
    assert result.ok is True
    # iter-3 M1: CRITICAL warning surfaces, and the explicit skip flag is set.
    assert "CRITICAL" in result.detail
    assert result.provenance_check_skipped is True
    assert "gh-author" in result.detail or "gh not on PATH" in result.detail


def test_selection_rule_gh_present_does_not_set_skip_flag(fixture_repo: Path):
    """iter-3 M1: when gh signal succeeds (or is empty), no skip flag is set.

    In this fixture-repo test, gh may or may not be on PATH. If it IS on
    PATH but unauthenticated, _gh_pr_touches returns None too, which
    legitimately sets the skip flag. We only assert on the case where
    gh is genuinely unavailable: provenance_check_skipped tracks the
    None-result count from _gh_pr_touches.
    """

    rows = cms.parse_registry(
        fixture_repo / "docs" / "governance" / "methodology_reviewer_registry.md"
    )
    text = _signoff_doc(handle="alice")
    result = cms.check_selection_rule(text, fixture_repo, rows)
    # ok must be True regardless (canonical git-log signal is clean).
    assert result.ok is True
    # The CheckResult's provenance_check_skipped attribute exists and is a bool.
    assert isinstance(result.provenance_check_skipped, bool)


# --------------------------------------------------------------------------- #
# Issue #192 H2/M1 — strict-gh policy resolution + main() exit-code contract.
# --------------------------------------------------------------------------- #


class TestStrictGhResolution:
    """``_resolve_strict_gh`` consults the CLI flag first, then env var.

    Issue #192 H2/M1: CI workflows export STRICT_GH=1 to elevate
    provenance_check_skipped from a logged warning to a hard exit (3).
    Local devs without the env var nor the CLI flag retain back-compat.
    """

    def test_default_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STRICT_GH", raising=False)
        assert cms._resolve_strict_gh(None) is False
        assert cms._resolve_strict_gh(False) is False

    def test_cli_flag_overrides_missing_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STRICT_GH", raising=False)
        assert cms._resolve_strict_gh(True) is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
    def test_env_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("STRICT_GH", value)
        assert cms._resolve_strict_gh(None) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "garbage"])
    def test_env_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("STRICT_GH", value)
        assert cms._resolve_strict_gh(None) is False

    def test_cli_flag_true_wins_over_env_falsy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("STRICT_GH", "0")
        # CLI flag True is the explicit operator intent; honor it.
        assert cms._resolve_strict_gh(True) is True

    def test_pass5_low1_cli_flag_false_overrides_env_truthy(self, monkeypatch: pytest.MonkeyPatch):
        """Codex pass-5 LOW-1 fix: explicit `--no-strict-gh` (cli_flag=False)
        overrides STRICT_GH=1 in env.

        Pre-fix: cli_flag=False fell through to env check (not
        intentional — there was no `--no-strict-gh` form). Post-fix:
        BooleanOptionalAction adds the negative form; resolver honors it.
        """

        monkeypatch.setenv("STRICT_GH", "1")
        # Explicit --no-strict-gh wins over env=1.
        assert cms._resolve_strict_gh(False) is False


class TestStrictGhMainExitCode:
    """``main()`` returns exit 3 when STRICT_GH is set AND provenance was skipped.

    Issue #192 H2/M1: this is the load-bearing wiring that escalates the
    M1 advisory warning to a fail-closed CI block. Tests both the strict
    fail path (exit 3) and the back-compat warn path (exit 0 — the
    canonical git-log signal still PASSES on its own).

    NOTE on exit-code priority: a generic validation failure (exit 1)
    takes precedence over the strict-gh failure (exit 3). The intent is
    that exit 3 specifically means "everything else passed but
    provenance was unverified" so log scrapers can distinguish the two.

    Implementation note: these tests monkeypatch ``cms.check_signoff`` to
    return a controlled list of CheckResults so the exit-code path is
    isolated from upstream concerns (GPG keyring, CoI SHA resolution,
    signature toolchain). The strict-gh wiring lives entirely in
    ``main()`` AFTER ``check_signoff`` returns, so this is a faithful
    test of the load-bearing logic.
    """

    @staticmethod
    def _make_results(all_pass: bool = True, provenance_skipped: bool = False) -> list:
        """Build a CheckResult list with controlled ok/skip flags."""

        results = [
            cms.CheckResult("filename", True, "ok"),
            cms.CheckResult("signoff_age", True, "ok"),
            cms.CheckResult("required_sections", True, "ok"),
            cms.CheckResult("signature_present", True, "ok"),
            cms.CheckResult("coi_referenced", True, "ok"),
            cms.CheckResult("registry_loaded", True, "ok"),
            cms.CheckResult("reviewer_registered", True, "ok"),
            cms.CheckResult(
                "selection_rule",
                all_pass,
                "ok | CRITICAL: gh provenance query SKIPPED" if provenance_skipped else "ok",
                provenance_check_skipped=provenance_skipped,
            ),
            cms.CheckResult("signature_verifies", True, "ok"),
        ]
        return results

    def _patch_check_signoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        all_pass: bool = True,
        provenance_skipped: bool = False,
    ) -> None:
        results = self._make_results(all_pass=all_pass, provenance_skipped=provenance_skipped)

        def fake_check_signoff(*args, **kwargs):  # noqa: ANN001 ANN002 ANN003
            return results

        monkeypatch.setattr(cms, "check_signoff", fake_check_signoff)

    @staticmethod
    def _make_doc_path(tmp_path: Path) -> Path:
        # main() requires the doc_path.is_file() check to pass; the
        # patched check_signoff above is what actually runs the validation.
        # Note: the file must be named with the dated pattern so
        # check_filename would have passed (we don't strictly need this
        # since check_signoff is patched, but keep it for realism).
        path = tmp_path / "optum_methodology_signoff_20260520.md"
        path.write_text("# placeholder\n", encoding="utf-8")
        return path

    def test_strict_gh_fail_when_gh_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """STRICT_GH=1 + provenance_check_skipped=True → exit 3 + clear message."""

        self._patch_check_signoff(monkeypatch, provenance_skipped=True)
        monkeypatch.setenv("STRICT_GH", "1")
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 3, (
            f"expected exit 3 (strict-gh policy violation), got {rc}. Captured: {captured}"
        )
        # Combined stdout + stderr — the report goes to stdout, the FAIL
        # message to stderr; both should be present.
        combined = captured.out + captured.err
        assert "FAIL" in combined
        assert "--strict-gh" in combined or "STRICT_GH=1" in combined
        assert "selection_rule" in combined

    def test_default_warn_when_gh_unavailable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """No STRICT_GH + no --strict-gh + provenance_skipped=True → exit 0."""

        self._patch_check_signoff(monkeypatch, provenance_skipped=True)
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0, f"expected exit 0 (back-compat warn-only), got {rc}. Captured: {captured}"
        # The CRITICAL warning should still surface on stdout (in the report
        # detail), confirming the warn behavior is preserved.
        combined = captured.out + captured.err
        assert "CRITICAL" in combined

    def test_strict_gh_cli_flag_overrides_missing_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """``--strict-gh`` on its own (no env var) also escalates to exit 3."""

        self._patch_check_signoff(monkeypatch, provenance_skipped=True)
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path), "--strict-gh"])
        assert rc == 3, f"expected exit 3 (CLI --strict-gh), got {rc}"

    def test_validation_failure_takes_precedence_over_strict_gh(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Exit 1 (generic validation fail) wins over exit 3 (strict-gh).

        Reserves exit 3 for the specific "everything passed except gh
        provenance" case so log scrapers can distinguish failure modes.
        """

        # all_pass=False (selection_rule.ok=False) AND
        # provenance_skipped=True — exit 1 must win.
        self._patch_check_signoff(monkeypatch, all_pass=False, provenance_skipped=True)
        monkeypatch.setenv("STRICT_GH", "1")
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        assert rc == 1, f"expected exit 1 (generic validation precedence), got {rc}"

    def test_strict_gh_pass_when_no_provenance_skip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """STRICT_GH=1 + all checks pass + no provenance skip → exit 0.

        Pin: the strict-gh policy doesn't false-positive when gh DID run
        successfully. Without this assertion the strict-gh failure path
        could silently expand to "always exit 3" on regressions.
        """

        self._patch_check_signoff(monkeypatch, provenance_skipped=False)
        monkeypatch.setenv("STRICT_GH", "1")
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        assert rc == 0, f"expected exit 0 (strict-gh satisfied), got {rc}"


# --------------------------------------------------------------------------- #
# Issue #192 H3 — workflow architecture pins for the reusable-workflow split.
# --------------------------------------------------------------------------- #


class TestReusableValidatorWorkflow:
    """The H3 mitigation requires:

    1. A reusable workflow at .github/workflows/methodology-signoff-validator.yml
       with `on: workflow_call:` that loads the validator script from a
       protected ref (default 'main') in a SEPARATE checkout.
    2. The caller `methodology_signoff_guard.yml` must delegate to the
       reusable workflow rather than running the validator inline.
    3. The reusable workflow must provision GH_TOKEN AND export STRICT_GH=1
       (issue #192 H2/M1 fail-closed).
    4. The reusable workflow must declare least-privilege permissions
       (contents:read + pull-requests:read).
    """

    REUSABLE_PATH = PROJECT_ROOT / ".github" / "workflows" / "methodology-signoff-validator.yml"
    CALLER_PATH = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"

    def _parse(self, path: Path):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_reusable_workflow_exists(self):
        assert self.REUSABLE_PATH.is_file(), (
            f"H3: reusable validator workflow missing: {self.REUSABLE_PATH}"
        )

    def test_reusable_workflow_has_workflow_call_trigger(self):
        parsed = self._parse(self.REUSABLE_PATH)
        # PyYAML interprets bare `on:` keys as boolean True; accept either.
        on_block = parsed.get("on") or parsed.get(True)
        assert on_block is not None, "reusable workflow missing on: block"
        assert "workflow_call" in on_block, "reusable workflow must use on: workflow_call: trigger"

    def test_reusable_workflow_declares_required_inputs(self):
        parsed = self._parse(self.REUSABLE_PATH)
        on_block = parsed.get("on") or parsed.get(True)
        inputs = on_block["workflow_call"]["inputs"]
        # touched_files is mandatory; without it the caller can't pass the
        # PR diff to the validator.
        assert "touched_files" in inputs
        assert inputs["touched_files"]["required"] is True
        # strict_gh is optional with sane default '1'.
        assert "strict_gh" in inputs
        # Codex pass-2 LOW-1: validator_ref MUST NOT be a caller input —
        # it is hardcoded inside the reusable workflow as
        # env.VALIDATOR_PROTECTED_REF: 'main'. Exposing it as an input
        # is a footgun (a future caller could accidentally pass a
        # non-protected ref).
        assert "validator_ref" not in inputs, (
            "Codex pass-2 LOW-1: validator_ref must not be a caller-"
            "controlled input — hardcode it inside the reusable workflow"
        )

    def test_reusable_workflow_checks_out_protected_ref(self):
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        parsed = self._parse(self.REUSABLE_PATH)
        # Codex pass-2 LOW-1: protected ref is hardcoded at workflow
        # level via env.VALIDATOR_PROTECTED_REF.
        env_block = parsed.get("env", {})
        assert env_block.get("VALIDATOR_PROTECTED_REF") == "main", (
            "Codex pass-2 LOW-1: VALIDATOR_PROTECTED_REF must be "
            "hardcoded to 'main' at workflow level"
        )
        # The checkout step must reference the env var (NOT the input).
        assert "ref: ${{ env.VALIDATOR_PROTECTED_REF }}" in text, (
            "H3: reusable workflow must checkout from env.VALIDATOR_PROTECTED_REF"
        )
        # And it must NOT reference inputs.validator_ref anywhere.
        assert "inputs.validator_ref" not in text, (
            "Codex pass-2 LOW-1: inputs.validator_ref must not appear in the reusable workflow"
        )
        # Two checkouts are required: PR head (artifacts) AND protected
        # ref (validator script source). The H3 defense rests on the
        # validator coming from the protected ref.
        assert "path: validator-source" in text, (
            "H3: validator source must live in a separate workspace"
        )
        assert "path: pr-checkout" in text, (
            "H3: PR-head artifacts must live in a separate workspace"
        )

    def test_reusable_workflow_provisions_gh_token(self):
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # H2/M1: GITHUB_TOKEN must be wired through to the validator's
        # gh CLI invocations.
        assert "GH_TOKEN: ${{ github.token }}" in text, (
            "H2/M1: reusable workflow must provision GH_TOKEN for gh CLI"
        )

    def test_reusable_workflow_exports_strict_gh(self):
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # H2/M1: STRICT_GH must be exported so the validator's main() hits
        # the fail-closed exit-3 path on any provenance_check_skipped=True.
        assert "STRICT_GH: ${{ inputs.strict_gh }}" in text, (
            "H2/M1: reusable workflow must export STRICT_GH for fail-closed"
        )

    def test_reusable_workflow_has_least_privilege_permissions(self):
        parsed = self._parse(self.REUSABLE_PATH)
        perms = parsed.get("permissions", {})
        # Least-privilege: read-only on contents + PRs; no write anywhere.
        assert perms.get("contents") == "read"
        assert perms.get("pull-requests") == "read"
        # Defensive: ensure no write permissions creep in.
        for key, value in perms.items():
            assert value == "read", f"H2/M1: permissions must be read-only; got {key}={value!r}"

    def test_caller_delegates_to_reusable_workflow(self):
        text = self.CALLER_PATH.read_text(encoding="utf-8")
        # The caller invokes the reusable workflow via `uses:` rather
        # than running python3 inline. Path-pinned same-repo invocation
        # is the baseline; future migration to cross-repo SHA-pinned
        # invocation is documented in the workflow header.
        assert "uses: ./.github/workflows/methodology-signoff-validator.yml" in text, (
            "H3: caller must delegate to the reusable validator workflow"
        )
        # Caller passes strict_gh: '1' so production CI hits fail-closed.
        assert "strict_gh: '1'" in text
        # Codex pass-2 LOW-1: validator_ref is no longer a caller input
        # (hardcoded inside the reusable workflow). Caller MUST NOT pass it.
        assert "validator_ref:" not in text, (
            "Codex pass-2 LOW-1: caller must not pass validator_ref"
        )

    def test_caller_no_longer_runs_validator_inline(self):
        text = self.CALLER_PATH.read_text(encoding="utf-8")
        # H3: the caller must NOT invoke check_methodology_signoff.py
        # directly — that's the threat the H3 split closes. The validator
        # is now invoked from inside the reusable workflow which loads
        # the script from the protected ref.
        assert "python3 /tmp/governance/check_methodology_signoff.py" not in text, (
            "H3: caller must delegate to reusable workflow, not invoke "
            "validator inline (the inline invocation was the H3 threat)"
        )

    def test_no_direct_input_interpolation_in_run_blocks(self):
        """Codex pass-1 HIGH-2 regression pin: inputs must route through env.

        Direct ``${{ inputs.X }}`` interpolation inside ``run:`` blocks
        is a shell-injection sink because GitHub Actions expression
        substitution happens before bash parses the line. Inputs must
        be exposed via ``env:`` so bash treats their values as data,
        never as code. This test forbids the dangerous pattern in the
        body of any run-script.

        We allow ``${{ inputs.X }}`` only inside ``env:`` blocks (where
        it becomes a shell variable assignment) and inside ``with:``
        blocks (parameters to actions). Anywhere else inside a
        ``run: |`` body is forbidden.
        """

        # Ensure pyyaml is importable; self._parse() uses it.
        pytest.importorskip("yaml")
        parsed = self._parse(self.REUSABLE_PATH)
        # Walk the YAML and find every `run:` value; assert the dangerous
        # patterns don't appear inside.
        offenders: list[tuple[str, str]] = []

        def scan_steps(steps: list, job_name: str) -> None:
            for step in steps:
                run_block = step.get("run")
                if not isinstance(run_block, str):
                    continue
                step_name = step.get("name", "<unnamed>")
                # Forbid `${{ inputs.<name> }}` inside the run block.
                if "${{ inputs." in run_block:
                    offenders.append((f"{job_name}::{step_name}", run_block[:200]))

        for job_name, job in parsed.get("jobs", {}).items():
            steps = job.get("steps", [])
            scan_steps(steps, job_name)

        assert not offenders, (
            "Codex pass-1 HIGH-2: ${{ inputs.X }} interpolation inside "
            "run: blocks is a shell-injection sink. Route the input "
            "through env: instead. Offenders:\n"
            + "\n".join(f"  {n}: {body!r}" for n, body in offenders)
        )

    def test_touched_files_routed_through_env(self):
        """HIGH-2 specific pin: touched_files must be in env: not direct interp.

        ``touched_files`` carries PR-influenced data (file paths from
        the diff) that matches a broad workflow path glob. A PR can add
        a file with a name like
        ``optum_methodology_signoff_$(curl evil).md`` — if that string
        is interpolated directly via ``${{ inputs.touched_files }}``
        into a here-string, command-substitution executes before bash
        sees the line. Routing via ``env: TOUCHED_FILES: ...`` and
        reading ``$TOUCHED_FILES`` in the script is the canonical fix.
        """

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Positive assertion: TOUCHED_FILES env var must be defined.
        assert "TOUCHED_FILES: ${{ inputs.touched_files }}" in text, (
            "HIGH-2: touched_files must be exposed via env: TOUCHED_FILES"
        )
        # Negative assertion: the dangerous interpolation must not appear
        # inside any here-string. We allow it inside an env: block.
        # Forbid the specific bash sink pattern.
        assert '<<< "${{ inputs.touched_files }}"' not in text, (
            'HIGH-2: <<< "${{ inputs.touched_files }}" is a shell-'
            'injection sink. Use <<< "$TOUCHED_FILES" instead.'
        )

    def test_validator_exit_code_3_preserved_by_workflow(self):
        """Codex pass-1 LOW-4 regression pin: workflow must preserve exit 3.

        The Python validator reserves exit 3 for STRICT_GH provenance
        gaps. The workflow's ``if ! python3 ...; then STATUS=1`` pattern
        collapses every non-zero into 1, undercutting the contract for
        log scrapers. The fix uses ``${PIPESTATUS[0]}`` to capture the
        validator's actual exit code and a priority rule that preserves
        3 unless 1 has occurred (mirrors the Python main()'s precedence).
        """

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Positive: PIPESTATUS must be used to capture the validator rc.
        assert "PIPESTATUS[0]" in text, (
            "LOW-4: workflow must capture validator exit code via "
            "${PIPESTATUS[0]} to preserve exit 3"
        )
        # Positive: priority logic must be present (1 > 3 > 0).
        assert 'STATUS"' in text and "ARTIFACT_RC" in text, (
            "LOW-4: workflow must implement exit-code priority "
            "(rc=1 > rc=3 > rc=0) so generic-validation failures still "
            "win over strict-gh failures"
        )
        # Negative: parse the YAML and inspect each `run:` block (not
        # comments) for the `if ! python3 ` pattern. The fix replaces
        # that with `python3 ...; ARTIFACT_RC=${PIPESTATUS[0]}`.
        parsed = self._parse(self.REUSABLE_PATH)
        bad_steps: list[str] = []
        for job_name, job in parsed.get("jobs", {}).items():
            for step in job.get("steps", []):
                run_block = step.get("run")
                if not isinstance(run_block, str):
                    continue
                # Strip comment lines so the check is on actual shell code.
                code_lines = [
                    line for line in run_block.splitlines() if not line.lstrip().startswith("#")
                ]
                code_only = "\n".join(code_lines)
                if "if ! python3 " in code_only:
                    bad_steps.append(f"{job_name}::{step.get('name', '<unnamed>')}")
        assert not bad_steps, (
            "LOW-4: the `if ! python3 ...` pattern collapses every "
            "validator non-zero into STATUS=1, masking exit 3. Use "
            "PIPESTATUS-based capture instead. Offenders: " + ", ".join(bad_steps)
        )

    def test_validator_exit_code_3_preserved_under_set_e_pipefail(self):
        """Codex pass-2 LOW-2: model `set -euo pipefail` semantics.

        The pass-1 LOW-4 fix used PIPESTATUS[0] but did NOT bracket the
        pipeline with `set +e` / `set -e`. Under `set -euo pipefail`,
        the pipeline exits the shell on any non-zero validator rc
        BEFORE the next line can read PIPESTATUS. Pass-2 MED-1 added
        the `set +e` / `set -e` bracketing. This test extracts the run
        block and exercises it as actual bash to confirm the rc path
        survives.
        """

        if shutil.which("bash") is None:
            pytest.skip("bash unavailable")

        # Extract the exact bash idiom from the workflow's run block so
        # this test exercises the production pattern.
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Sanity: the bracketing comments must mention the pass-2 MED-1 fix.
        assert "set +e" in text, "MED-1: workflow must bracket pipeline with set +e"
        assert "ARTIFACT_RC=${PIPESTATUS[0]}" in text, (
            "MED-1: workflow must capture PIPESTATUS[0] inside the set +e bracket"
        )
        assert "set -e" in text, "MED-1: workflow must restore set -e after the bracket"

        # Concrete shell harness: simulate a validator that exits 3
        # under set -euo pipefail. Without the set+e bracket, the
        # pipeline would terminate the shell BEFORE the rc capture.
        script = """
        set -euo pipefail
        STATUS=0
        set +e
        (exit 3) | tee /dev/null
        ARTIFACT_RC=${PIPESTATUS[0]}
        set -e
        if [ "$ARTIFACT_RC" -eq 1 ]; then
          STATUS=1
        elif [ "$ARTIFACT_RC" -eq 3 ] && [ "$STATUS" -eq 0 ]; then
          STATUS=3
        elif [ "$ARTIFACT_RC" -ne 0 ] && [ "$STATUS" -eq 0 ]; then
          STATUS="$ARTIFACT_RC"
        fi
        echo "FINAL_STATUS=$STATUS"
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        assert "FINAL_STATUS=3" in result.stdout, (
            f"MED-1: rc=3 must propagate; got {result.stdout!r}"
        )

    def test_validator_exit_code_priority_one_wins_over_three(self):
        """Codex pass-2 MED-1 + LOW-2: explicit precedence test (rc=1 > rc=3).

        The pass-1 LOW-4 fix used `-gt` for the priority comparison,
        which is wrong: `3 -gt 1` is true, so a later rc=1 would NOT
        overwrite an earlier rc=3 even though we want rc=1 to win.
        Pass-2 MED-1 fixed this with explicit `-eq` comparisons.
        """

        if shutil.which("bash") is None:
            pytest.skip("bash unavailable")

        # Simulate two artifacts: first exits 3 (strict-gh), second
        # exits 1 (generic validation fail). The final STATUS must be 1.
        script = """
        set -euo pipefail
        STATUS=0

        # Artifact 1: rc=3 (strict-gh provenance gap).
        set +e
        (exit 3) | tee /dev/null
        ARTIFACT_RC=${PIPESTATUS[0]}
        set -e
        if [ "$ARTIFACT_RC" -eq 1 ]; then
          STATUS=1
        elif [ "$ARTIFACT_RC" -eq 3 ] && [ "$STATUS" -eq 0 ]; then
          STATUS=3
        fi

        # Artifact 2: rc=1 (generic validation fail). MUST overwrite rc=3.
        set +e
        (exit 1) | tee /dev/null
        ARTIFACT_RC=${PIPESTATUS[0]}
        set -e
        if [ "$ARTIFACT_RC" -eq 1 ]; then
          STATUS=1
        elif [ "$ARTIFACT_RC" -eq 3 ] && [ "$STATUS" -eq 0 ]; then
          STATUS=3
        fi

        echo "FINAL_STATUS=$STATUS"
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        assert "FINAL_STATUS=1" in result.stdout, (
            f"MED-1: rc=1 must win over earlier rc=3; got {result.stdout!r}"
        )

    def test_threat_model_documented_honestly(self):
        """Codex pass-1 HIGH-1 pin: H3 status text must reflect partial close.

        Initial framing claimed H3 was fully RESOLVED by the
        reusable-workflow split. Codex pass-1 HIGH-1 corrected this:
        same-repo path-pinned `uses: ./...` loads the called workflow
        YAML from caller's ref, so a malicious PR can edit BOTH workflow
        files in one commit and disable the `ref: main` checkout. The
        validator script pin IS load-bearing; the workflow YAML pinning
        is NOT. This test pins the honest documentation.
        """

        n3_doc = PROJECT_ROOT / "docs" / "governance" / "n3_known_limitations_20260510.md"
        text = n3_doc.read_text(encoding="utf-8")
        # The H3 section must mark status as PARTIALLY RESOLVED, not RESOLVED.
        # The summary table must reflect the partial close.
        assert "PARTIALLY RESOLVED" in text, (
            "HIGH-1: n3_known_limitations doc must mark H3 status as "
            "PARTIALLY RESOLVED, not RESOLVED, until cross-repo "
            "SHA-pinned migration lands"
        )
        # The doc must reference the codex pass-1 HIGH-1 honesty correction.
        assert "codex pass-1 HIGH-1" in text or "HIGH-1" in text, (
            "HIGH-1: doc must explicitly reference the codex finding "
            "that prompted the honesty correction"
        )
        # The caller workflow header must also reflect the same nuance.
        caller_text = self.CALLER_PATH.read_text(encoding="utf-8")
        assert "PARTIALLY RESOLVED" in caller_text or "partial" in caller_text.lower(), (
            "HIGH-1: caller workflow header must reflect the partial-close status of H3"
        )


# --------------------------------------------------------------------------- #
# Issue #226 H1+H4 — fingerprint normalization, keyring-presence check,
# CoI body signature verification, STRICT_GPG resolution, exit code 4.
# --------------------------------------------------------------------------- #


class TestFingerprintNormalization:
    """``_normalize_fingerprint`` returns a canonical 40-char uppercase hex
    string OR the empty string for placeholders / unparseable input.

    Issue #226 H1: registry rows can land with operator-fillable
    placeholders (`<TBD ...>`); those parse to "" so the keyring check
    downstream knows the row hasn't been operator-populated yet. Real
    fingerprints round-trip through the normalizer regardless of
    surrounding markdown emphasis, internal whitespace, or 0x prefix.
    """

    def test_empty_input_returns_empty(self):
        assert cms._normalize_fingerprint("") == ""

    def test_whitespace_only_returns_empty(self):
        assert cms._normalize_fingerprint("   ") == ""

    @pytest.mark.parametrize(
        "placeholder",
        [
            "<TBD>",
            "<TBD — populated by operator>",
            "<placeholder>",
            "<populated by ops>",
            "TBD operator",
            "N/A",
            "none",
        ],
    )
    def test_placeholder_returns_empty(self, placeholder: str):
        assert cms._normalize_fingerprint(placeholder) == ""

    def test_backticks_are_stripped(self):
        assert cms._normalize_fingerprint("`<TBD>`") == ""
        # Backticked real fingerprint still parses.
        fp = "abcdef0123456789abcdef0123456789abcdef01"
        assert cms._normalize_fingerprint(f"`{fp}`") == fp.upper()

    def test_real_fingerprint_lowercase_uppercases(self):
        fp = "abcdef0123456789abcdef0123456789abcdef01"
        assert cms._normalize_fingerprint(fp) == fp.upper()

    def test_real_fingerprint_with_internal_spaces(self):
        # Convention: GPG renders with space-every-4-chars + double-
        # space at the midpoint. Operators sometimes paste verbatim.
        spaced = "ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01"
        assert cms._normalize_fingerprint(spaced) == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"

    def test_real_fingerprint_with_0x_prefix(self):
        assert (
            cms._normalize_fingerprint("0xABCDEF0123456789ABCDEF0123456789ABCDEF01")
            == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        )

    def test_garbage_returns_empty(self):
        # Not a fingerprint shape — return empty (not raise).
        assert cms._normalize_fingerprint("not-a-fingerprint") == ""
        assert cms._normalize_fingerprint("0123") == ""  # too short
        assert cms._normalize_fingerprint("z" * 40) == ""  # not hex

    def test_leading_bom_is_stripped(self):
        """Codex pass-2 LOW-1 fix: a BOM-prefixed real fingerprint
        must round-trip through normalization.

        Operators sometimes paste fingerprints from terminals that
        prepend U+FEFF; the pre-fix normalizer rejected those as
        garbage → STRICT_GPG=1 false-fail.
        """

        bom = "﻿"
        fp = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        assert cms._normalize_fingerprint(bom + fp) == fp
        # BOM + spaced + backticks combo also handled.
        spaced = "ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01"
        assert cms._normalize_fingerprint(bom + "`" + spaced + "`") == fp


class TestRegistryFingerprintColumn:
    """Registry parser populates ``ReviewerInfo.fingerprint`` from the 8th cell.

    Issue #226 H1: the registry schema migrated from 7→8 columns. The
    new column is ``fingerprint`` (after ``status``). Placeholder values
    parse to ""; real fingerprints round-trip through normalization.
    """

    def test_placeholder_fingerprint_parses_empty(self, tmp_path: Path):
        registry_text = (
            "| name | email | github_handle | role | date_added | "
            "areas_of_expertise | status | fingerprint |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| A | a@ex.com | a | clinician | 2026-05-10 | methodology | "
            "active | `<TBD — populated by operator>` |\n"
        )
        reg = tmp_path / "r.md"
        reg.write_text(registry_text, encoding="utf-8")
        rows = cms.parse_registry(reg)
        assert len(rows) == 1
        assert rows[0].fingerprint == ""

    def test_real_fingerprint_parses_canonical(self, tmp_path: Path):
        fp_spaced = "ABCD EF01 2345 6789 ABCD  EF01 2345 6789 ABCD EF01"
        registry_text = (
            "| name | email | github_handle | role | date_added | "
            "areas_of_expertise | status | fingerprint |\n"
            "|---|---|---|---|---|---|---|---|\n"
            "| A | a@ex.com | a | clinician | 2026-05-10 | methodology | "
            f"active | {fp_spaced} |\n"
        )
        reg = tmp_path / "r.md"
        reg.write_text(registry_text, encoding="utf-8")
        rows = cms.parse_registry(reg)
        assert len(rows) == 1
        assert rows[0].fingerprint == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"

    def test_legacy_7col_registry_rejected(self, tmp_path: Path):
        """A 7-column registry (no fingerprint) yields zero parsed rows.

        Forcing a header-equality match means operators can't accidentally
        run the new validator against the old schema; the registry
        migration must happen in lockstep with the workflow change.
        """

        registry_text = (
            "| name | email | github_handle | role | date_added | "
            "areas_of_expertise | status |\n"
            "|---|---|---|---|---|---|\n"
            "| A | a@ex.com | a | clinician | 2026-05-10 | methodology | active |\n"
        )
        reg = tmp_path / "r.md"
        reg.write_text(registry_text, encoding="utf-8")
        rows = cms.parse_registry(reg)
        assert rows == [], "legacy 7-col registry must NOT parse against new 8-col schema"

    def test_production_registry_has_fingerprint_column(self):
        """The shipped registry md uses the 8-col schema."""

        path = PROJECT_ROOT / "docs" / "governance" / "methodology_reviewer_registry.md"
        text = path.read_text(encoding="utf-8")
        # Header row must include the fingerprint column.
        assert "| fingerprint |" in text, (
            "issue #226 H1: production registry must include fingerprint column"
        )

    def test_reviewer_info_fingerprint_default_empty(self):
        """``ReviewerInfo()`` constructed without fingerprint defaults to ''."""

        info = cms.ReviewerInfo(handle="x", email="x@e.com", status="active")
        assert info.fingerprint == ""


class TestKeyringPresentCheck:
    """``check_keyring_present`` PASSES with WARN when keyring_dir is None;
    PASSES with key count when populated; FAILS when missing/empty.

    Issue #226 H1.
    """

    def test_keyring_dir_none_advisory_pass_with_skip_flag(self):
        """Codex pass-1 MED-1: missing keyring is ADVISORY pass + sig-skip.

        STRICT_GPG=1 callers escalate to exit 4 via the sig-skip path
        (NOT exit 1 via the generic ``not all(r.ok)`` branch — that
        would lose the routing signal for log scrapers).
        """

        result = cms.check_keyring_present(None)
        assert result.ok is True
        assert "WARN" in result.detail
        assert "issue #226" in result.detail
        # Codex pass-1 MED-1: this MUST flag for STRICT_GPG escalation.
        assert result.signature_check_skipped is True

    def test_keyring_dir_missing_advisory_pass_with_skip_flag(self, tmp_path: Path):
        """Codex pass-1 MED-1: missing-dir is now advisory pass + sig-skip."""

        nonexistent = tmp_path / "nope"
        result = cms.check_keyring_present(nonexistent)
        assert result.ok is True
        assert "missing" in result.detail or "does not exist" in result.detail
        assert result.signature_check_skipped is True

    def test_keyring_dir_empty_advisory_pass_with_skip_flag(self, tmp_path: Path):
        """Codex pass-1 MED-1: empty keyring is now advisory pass + sig-skip."""

        if shutil.which("gpg") is None:
            pytest.skip("gpg not on PATH")
        empty = tmp_path / "empty_gpghome"
        empty.mkdir(mode=0o700)
        result = cms.check_keyring_present(empty)
        assert result.ok is True
        # gpg --list-keys against an empty homedir reports zero keys.
        assert "zero public keys" in result.detail or "missing" in result.detail
        assert result.signature_check_skipped is True

    def test_keyring_dir_with_imported_key_passes(self, gpg_keyring: tuple[Path, str]):
        home, _ = gpg_keyring
        result = cms.check_keyring_present(home)
        assert result.ok is True, f"unexpected fail: {result.detail}"
        assert "1 public key" in result.detail or "key(s)" in result.detail
        # Populated keyring → pinning is possible; MUST NOT flag for skip.
        assert result.signature_check_skipped is False


class TestCoIBodySignatureVerifies:
    """``check_coi_body_signature_verifies`` honors the H4 contract.

    Issue #226 H4: validates the CoI declaration body (NOT just the
    sign-off doc) against an inline armor block OR a sibling
    ``<coi>.asc`` detached signature.
    """

    def _make_doc_pointing_at_coi(self, coi_path_str: str) -> str:
        """Minimal sign-off doc text with the CoI document field set."""

        return (
            "# Sign-off\n"
            "## Conflict-of-interest declaration\n"
            f"- **CoI document:** {coi_path_str}\n"
            f"- **CoI declaration commit SHA:** abc1234567890def\n"
        )

    def test_missing_coi_path_field_fails(self, tmp_path: Path):
        doc_text = "# Sign-off (no CoI fields)\n"
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=None
        )
        assert result.ok is False
        assert "missing or placeholder" in result.detail

    def test_placeholder_coi_path_fails(self, tmp_path: Path):
        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/<github_handle>_<YYYYMMDD>.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=None
        )
        assert result.ok is False
        assert "placeholder" in result.detail

    def test_coi_path_not_resolvable_fails(self, tmp_path: Path):
        doc_text = self._make_doc_pointing_at_coi("docs/governance/coi_declarations/nope.md")
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=None
        )
        assert result.ok is False
        assert "not found" in result.detail

    def test_coi_body_no_signature_advisory_pass(self, tmp_path: Path):
        """No inline armor + no sibling .asc → ADVISORY pass, sig-skip flag set."""

        coi_dir = tmp_path / "docs" / "governance" / "coi_declarations"
        coi_dir.mkdir(parents=True)
        coi_path = coi_dir / "alice_20260514.md"
        coi_path.write_text("# CoI alice\nzero touches\n", encoding="utf-8")

        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/alice_20260514.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=None
        )
        assert result.ok is True
        assert "WARN" in result.detail
        assert result.signature_check_skipped is True

    def test_coi_body_inline_signature_verifies(
        self, tmp_path: Path, gpg_keyring: tuple[Path, str]
    ):
        """An inline ASCII-armor signature in the CoI body verifies under keyring."""

        home, _ = gpg_keyring
        coi_dir = tmp_path / "docs" / "governance" / "coi_declarations"
        coi_dir.mkdir(parents=True)
        coi_path = coi_dir / "alice_20260514.md"

        body = "# CoI alice\nzero touches in named period\n"

        # Sign body with the fixture key, then write body + inline armor.
        import os

        env = os.environ.copy()
        env["GNUPGHOME"] = str(home)
        sign = subprocess.run(
            ["gpg", "--batch", "--detach-sign", "--armor", "--output", "-"],
            input=body,
            capture_output=True,
            text=True,
            env=env,
        )
        if sign.returncode != 0:
            pytest.skip(f"gpg --detach-sign failed: {sign.stderr}")

        coi_path.write_text(body + sign.stdout, encoding="utf-8")

        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/alice_20260514.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=home
        )
        assert result.ok is True, f"verify failed: {result.detail}"
        assert "OK" in result.detail
        assert result.signature_check_skipped is False

    def test_coi_body_inline_signature_tampered_fails(
        self, tmp_path: Path, gpg_keyring: tuple[Path, str]
    ):
        """Tampered CoI body + valid signature → FAIL with diagnostic."""

        home, _ = gpg_keyring
        coi_dir = tmp_path / "docs" / "governance" / "coi_declarations"
        coi_dir.mkdir(parents=True)
        coi_path = coi_dir / "alice_20260514.md"

        body = "# CoI alice\nzero touches in named period\n"
        import os

        env = os.environ.copy()
        env["GNUPGHOME"] = str(home)
        sign = subprocess.run(
            ["gpg", "--batch", "--detach-sign", "--armor", "--output", "-"],
            input=body,
            capture_output=True,
            text=True,
            env=env,
        )
        if sign.returncode != 0:
            pytest.skip(f"gpg --detach-sign failed: {sign.stderr}")

        # Tamper body BEFORE writing.
        tampered_body = body.replace("zero", "ONE")
        coi_path.write_text(tampered_body + sign.stdout, encoding="utf-8")

        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/alice_20260514.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=home
        )
        assert result.ok is False
        assert "FAILED" in result.detail or "BAD" in result.detail.upper()

    def test_coi_body_sibling_asc_signature_verifies(
        self, tmp_path: Path, gpg_keyring: tuple[Path, str]
    ):
        """A sibling <coi>.asc detached signature verifies."""

        home, _ = gpg_keyring
        coi_dir = tmp_path / "docs" / "governance" / "coi_declarations"
        coi_dir.mkdir(parents=True)
        coi_path = coi_dir / "alice_20260514.md"

        body = "# CoI alice (sibling-asc variant)\nzero touches\n"
        coi_path.write_text(body, encoding="utf-8")

        # Generate sibling .asc detached signature against the file.
        import os

        env = os.environ.copy()
        env["GNUPGHOME"] = str(home)
        sign = subprocess.run(
            [
                "gpg",
                "--batch",
                "--detach-sign",
                "--armor",
                "--output",
                str(coi_path) + ".asc",
                str(coi_path),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        if sign.returncode != 0:
            pytest.skip(f"gpg --detach-sign sibling failed: {sign.stderr}")

        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/alice_20260514.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=home
        )
        assert result.ok is True, f"sibling-asc verify failed: {result.detail}"
        assert "sibling-asc" in result.detail

    def test_pass6_med1_signed_coi_with_empty_keyring_advisory_pass(
        self, tmp_path: Path, gpg_keyring: tuple[Path, str]
    ):
        """Codex pass-6 MED-1 fix: signed CoI + unprovisioned keyring →
        advisory PASS + sig-skip (NOT generic FAIL).

        Pre-fix: check_coi_body_signature_verifies bypassed the keyring
        preflight that pass-2 added to check_signature_verifies — so a
        signed CoI with an empty keyring would route through main()'s
        generic exit-1 branch instead of the reserved exit-4 STRICT_GPG
        path. This broke the documented `strict_gpg: '0'` rollout
        escape hatch for already-signed CoIs.

        Fix mirrors the sign-off keyring preflight: signed CoI + empty
        keyring → advisory PASS + signature_check_skipped=True.
        """

        home, _ = gpg_keyring  # populated keyring (used to sign)

        coi_dir = tmp_path / "docs" / "governance" / "coi_declarations"
        coi_dir.mkdir(parents=True)
        coi_path = coi_dir / "alice_20260514.md"
        body = "# CoI alice\nzero touches\n"

        # Sign the CoI body using the populated keyring (the operator's
        # real keyring) BUT then verify against an EMPTY keyring (the
        # CI-runner-not-yet-provisioned scenario).
        import os as _os

        env = _os.environ.copy()
        env["GNUPGHOME"] = str(home)
        sib_sign = subprocess.run(
            [
                "gpg",
                "--batch",
                "--detach-sign",
                "--armor",
                "--output",
                str(coi_path) + ".asc",
                str(coi_path),
            ],
            input=body,
            capture_output=True,
            text=True,
            env=env,
        )
        coi_path.write_text(body, encoding="utf-8")
        if sib_sign.returncode != 0:
            # Above invocation actually requires the file to exist; redo.
            coi_path.write_text(body, encoding="utf-8")
            sib_sign = subprocess.run(
                [
                    "gpg",
                    "--batch",
                    "--detach-sign",
                    "--armor",
                    "--output",
                    str(coi_path) + ".asc",
                    str(coi_path),
                ],
                capture_output=True,
                text=True,
                env=env,
            )
            if sib_sign.returncode != 0:
                pytest.skip(f"gpg sign failed: {sib_sign.stderr}")

        # EMPTY keyring (operator hasn't provisioned the secret yet).
        empty_keyring = tmp_path / "empty_keyring"
        empty_keyring.mkdir(mode=0o700)

        doc_text = self._make_doc_pointing_at_coi(
            "docs/governance/coi_declarations/alice_20260514.md"
        )
        result = cms.check_coi_body_signature_verifies(
            doc_text, repo_root=tmp_path, keyring_dir=empty_keyring
        )
        # Pass-6 MED-1: advisory pass + sig-skip (NOT FAIL).
        assert result.ok is True, (
            f"pass-6 MED-1: signed CoI + empty keyring must be advisory "
            f"pass; got ok=False with {result.detail}"
        )
        assert result.signature_check_skipped is True
        assert (
            "signature present but keyring" in result.detail
            or "missing/empty/unreadable" in result.detail
        )


class TestExtractValidsigFingerprint:
    """``_extract_validsig_fingerprint`` parses GNUPG status-fd output.

    Issue #226 codex pass-1 HIGH-1: the fingerprint binding requires
    the validator to extract the signing fingerprint from gpg's
    ``--status-fd=1`` stream. The ``[GNUPG:] VALIDSIG <fpr> <date> ...``
    line is emitted only when verification succeeds AND the signing key
    is in the keyring.
    """

    def test_validsig_line_extracted(self):
        output = (
            "[GNUPG:] NEWSIG\n"
            "[GNUPG:] KEY_CONSIDERED ABCDEF0123456789ABCDEF0123456789ABCDEF01 0\n"
            "[GNUPG:] SIG_ID ID 2026-05-15 1700000000\n"
            "[GNUPG:] GOODSIG ABCDEF01 Test\n"
            "[GNUPG:] VALIDSIG ABCDEF0123456789ABCDEF0123456789ABCDEF01 2026-05-15 "
            "1700000000 0 4 0 1 8 01 ABCDEF0123456789ABCDEF0123456789ABCDEF01\n"
            "[GNUPG:] TRUST_UNDEFINED 0 pgp\n"
        )
        assert (
            cms._extract_validsig_fingerprint(output) == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        )

    def test_no_validsig_returns_none(self):
        output = "[GNUPG:] NEWSIG\n[GNUPG:] BADSIG ABCD 'Test'\n"
        assert cms._extract_validsig_fingerprint(output) is None

    def test_empty_input_returns_none(self):
        assert cms._extract_validsig_fingerprint("") is None

    def test_validsig_lowercase_uppercased(self):
        """Codex pass-2 LOW-1 fix: case-insensitive match + uppercase output.

        Modern gpg always emits uppercase but a future / patched version
        may emit lowercase; the validator must accept both shapes and
        emit canonical uppercase so downstream comparisons work.
        """

        output = (
            "[GNUPG:] VALIDSIG abcdef0123456789abcdef0123456789abcdef01 2026-05-15 "
            "1700000000 0 4 0 1 8 01 abcdef0123456789abcdef0123456789abcdef01\n"
        )
        # Pass-2 LOW-1: lowercase input now parses + uppercases.
        assert (
            cms._extract_validsig_fingerprint(output) == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        )

    def test_validsig_among_other_lines(self):
        output = (
            "Random non-status output\n"
            "Multiple lines\n"
            "[GNUPG:] VALIDSIG ABCDEF0123456789ABCDEF0123456789ABCDEF01 ...\n"
            "More noise\n"
        )
        assert (
            cms._extract_validsig_fingerprint(output) == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        )


class TestCheckSigningFingerprintMatchesRegistry:
    """``check_signing_fingerprint_matches_registry`` binds the verified
    signature to a registered reviewer fingerprint.

    Issue #226 codex pass-1 HIGH-1: without this binding, any key in
    the keyring can verify any reviewer's sign-off — the keyring +
    registry combo gives reviewer-identity binding only when the
    fingerprint comparison is enforced.
    """

    @staticmethod
    def _make_registry(handle: str, fingerprint: str = "") -> list:
        return [
            cms.ReviewerInfo(
                handle=handle,
                email=f"{handle}@example.com",
                status="active",
                emails=(f"{handle}@example.com",),
                fingerprint=fingerprint,
            ),
        ]

    @staticmethod
    def _make_doc_text(handle: str = "alice") -> str:
        return f"# Sign-off\n## Reviewer\n- **GitHub handle:** @{handle}\n"

    def test_no_signing_fingerprints_advisory_pass(self):
        """Both verify checks failed/skipped → ADVISORY pass + sig-skip."""

        registry = self._make_registry("alice", "ABCDEF0123456789ABCDEF0123456789ABCDEF01")
        # Verify checks failed → no signing_fingerprint populated.
        signature_results = [
            cms.CheckResult("signature_verifies", False, "FAILED"),
            cms.CheckResult("coi_body_signature_verifies", False, "FAILED"),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is True
        assert result.signature_check_skipped is True
        # Pass-2 HIGH-2: detail surfaces the no-successful-verify case.
        assert "no successful signature verification" in result.detail

    def test_registered_fingerprint_empty_advisory_pass(self):
        """Reviewer registered but fingerprint cell empty → advisory pass + sig-skip.

        The operator hasn't completed the fingerprint-population step
        yet. Don't break PRs but flag for STRICT_GPG=1 escalation.
        """

        registry = self._make_registry("alice", "")  # placeholder
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "ok",
                signing_fingerprint="ABCDEF0123456789ABCDEF0123456789ABCDEF01",
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is True
        assert result.signature_check_skipped is True
        assert "registered fingerprint" in result.detail.lower()
        assert "empty" in result.detail.lower() or "placeholder" in result.detail.lower()

    def test_signing_fingerprint_matches_passes(self):
        """Registered fingerprint matches the signing fingerprint → PASS."""

        fp = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        registry = self._make_registry("alice", fp)
        signature_results = [
            cms.CheckResult("signature_verifies", True, "ok", signing_fingerprint=fp),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is True
        assert result.signature_check_skipped is False
        assert "match" in result.detail

    def test_signing_fingerprint_mismatch_fails(self):
        """Wrong key signed it → FAIL with explicit detail.

        Load-bearing pin: this is the WHOLE POINT of the fingerprint-
        binding check. Without it, reviewer A's key would verify
        reviewer B's sign-off because both keys live in the same
        $KEYRING_DIR.
        """

        registered = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        attacker = "1111111111111111111111111111111111111111"
        registry = self._make_registry("alice", registered)
        signature_results = [
            cms.CheckResult("signature_verifies", True, "ok", signing_fingerprint=attacker),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is False
        assert "do not match" in result.detail
        assert registered in result.detail
        assert attacker in result.detail

    def test_coi_body_signing_fingerprint_also_pinned(self):
        """The CoI body sig fingerprint must ALSO match — pinning covers BOTH checks."""

        registered = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        attacker = "2222222222222222222222222222222222222222"
        registry = self._make_registry("alice", registered)
        signature_results = [
            cms.CheckResult("signature_verifies", True, "ok", signing_fingerprint=registered),
            # CoI body signed by a DIFFERENT key (e.g. a stale key the
            # reviewer rotated away from but didn't update the registry).
            cms.CheckResult(
                "coi_body_signature_verifies",
                True,
                "ok",
                signing_fingerprint=attacker,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is False
        assert "coi_body_signature_verifies" in result.detail

    def test_unknown_handle_fails(self):
        registry = self._make_registry("alice", "ABCD" * 10)
        signature_results = [
            cms.CheckResult("signature_verifies", True, "ok", signing_fingerprint="ABCD" * 10),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("bob"), registry, signature_results
        )
        assert result.ok is False
        assert "not in registry" in result.detail

    def test_codex_pass2_h2_unpinned_verify_flagged_as_pinning_gap(self):
        """Codex pass-2 HIGH-2: a successful verify with signing_fingerprint=None
        is a PINNING GAP, NOT a "no fingerprints to evaluate" PASS.

        Pre-fix bug: if `signature_verifies` succeeded via the SIGSTORE
        code path (no GPG VALIDSIG produced) AND the CoI body sig
        verified+pinned, the aggregate pinning check passed even though
        the SIGN-OFF artifact was never bound to the reviewer. This is
        the load-bearing security invariant — every successful verify
        MUST be pinned.
        """

        registered = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        registry = self._make_registry("alice", registered)
        # signature_verifies passed via sigstore path (no signing_fingerprint).
        # coi_body_signature_verifies passed via gpg path AND matches.
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "sigstore verify OK",
                signing_fingerprint=None,
            ),
            cms.CheckResult(
                "coi_body_signature_verifies",
                True,
                "ok",
                signing_fingerprint=registered,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        # ADVISORY pass + sig-skip: STRICT_GPG=1 escalates to exit 4.
        # Under default mode, the run continues.
        assert result.ok is True
        assert result.signature_check_skipped is True
        assert "unpinned" in result.detail.lower() or "pinning gap" in result.detail.lower()
        # The sigstore-emitting check name must be in the detail so
        # operators know which artifact wasn't bound.
        assert "signature_verifies" in result.detail

    def test_codex_pass2_med1_duplicate_active_rows_aggregated(self):
        """Codex pass-2 MED-1: production registry has 2 rows for handle
        `enunezvn` (canonical email + GitHub no-reply). The pinning check
        MUST consider BOTH rows' fingerprints — a key rotation history
        encoded as a second row should still satisfy pinning.
        """

        fp_canonical = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        fp_rotated = "1111222233334444555566667777888899990000"
        # Two active rows for the same handle.
        registry = [
            cms.ReviewerInfo(
                handle="enunezvn",
                email="canonical@example.com",
                status="active",
                emails=("canonical@example.com",),
                fingerprint=fp_canonical,
            ),
            cms.ReviewerInfo(
                handle="enunezvn",
                email="noreply@users.noreply.github.com",
                status="active",
                emails=("noreply@users.noreply.github.com",),
                fingerprint=fp_rotated,
            ),
        ]
        # Sign-off was signed by the rotated key (second row).
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "ok",
                signing_fingerprint=fp_rotated,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("enunezvn"), registry, signature_results
        )
        assert result.ok is True, (
            f"pass-2 MED-1: aggregation across active rows must accept either "
            f"fingerprint; got {result.detail}"
        )
        # Must NOT flag for skip — match is real.
        assert result.signature_check_skipped is False

    def test_codex_pass2_med1_signing_fingerprint_in_neither_row_fails(self):
        """Pass-2 MED-1: third-key signing (NOT in registry) still fails."""

        fp_canonical = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
        fp_rotated = "1111222233334444555566667777888899990000"
        fp_attacker = "BAD000000000000000000000000000000000BAD0"
        registry = [
            cms.ReviewerInfo(
                handle="enunezvn",
                email="a@example.com",
                status="active",
                emails=("a@example.com",),
                fingerprint=fp_canonical,
            ),
            cms.ReviewerInfo(
                handle="enunezvn",
                email="b@example.com",
                status="active",
                emails=("b@example.com",),
                fingerprint=fp_rotated,
            ),
        ]
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "ok",
                signing_fingerprint=fp_attacker,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("enunezvn"), registry, signature_results
        )
        assert result.ok is False
        assert "do not match" in result.detail
        # Both registered fingerprints surfaced.
        assert fp_canonical in result.detail
        assert fp_rotated in result.detail

    def test_codex_pass3_med2_recused_fingerprint_must_not_satisfy_pinning(self):
        """Codex pass-3 MED-2: a fingerprint on a RECUSED row MUST NOT
        satisfy pinning even when the handle has an active row.

        The active+recused split is how the registry encodes "this
        reviewer rotated their key — old key kept on inactive/recused
        row for historical reference, new key on active row." If we
        accepted recused fingerprints, the rotation semantic would be
        defeated AND a leaked old key could still verify.
        """

        fp_active = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        fp_recused = "DEAD000000000000000000000000000000000000"
        registry = [
            cms.ReviewerInfo(
                handle="alice",
                email="alice@example.com",
                status="active",
                emails=("alice@example.com",),
                fingerprint=fp_active,
            ),
            cms.ReviewerInfo(
                handle="alice",
                email="alice@example.com",
                status="recused",
                emails=("alice@example.com",),
                fingerprint=fp_recused,
            ),
        ]
        # Sign-off was signed by the RECUSED key.
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "ok",
                signing_fingerprint=fp_recused,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        # Pass-3 MED-2: recused-row fingerprint MUST NOT satisfy pinning.
        assert result.ok is False, (
            f"pass-3 MED-2: recused fingerprint must not satisfy pinning; got {result.detail}"
        )
        # Active fingerprint surfaced; recused NOT.
        assert fp_active in result.detail
        # The detail should NOT list the recused fp as registered.
        assert (
            f"registered=[{fp_recused!r}]" not in result.detail
            and f"registered={[fp_recused]}" not in result.detail
        )

    def test_codex_pass3_med2_no_active_rows_fails(self):
        """Pass-3 MED-2: handle exists in registry but ZERO active rows → FAIL."""

        registry = [
            cms.ReviewerInfo(
                handle="alice",
                email="alice@example.com",
                status="inactive",  # only inactive
                emails=("alice@example.com",),
                fingerprint="A" * 40,
            ),
        ]
        signature_results = [
            cms.CheckResult(
                "signature_verifies",
                True,
                "ok",
                signing_fingerprint="A" * 40,
            ),
        ]
        result = cms.check_signing_fingerprint_matches_registry(
            self._make_doc_text("alice"), registry, signature_results
        )
        assert result.ok is False
        assert "no active registry rows" in result.detail


class TestStrictGpgResolution:
    """``_resolve_strict_gpg`` mirrors ``_resolve_strict_gh`` semantics.

    Issue #226 H1+H4.
    """

    def test_default_off(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STRICT_GPG", raising=False)
        assert cms._resolve_strict_gpg(None) is False
        assert cms._resolve_strict_gpg(False) is False

    def test_cli_flag_overrides_missing_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("STRICT_GPG", raising=False)
        assert cms._resolve_strict_gpg(True) is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "On"])
    def test_env_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("STRICT_GPG", value)
        assert cms._resolve_strict_gpg(None) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "garbage"])
    def test_env_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str):
        monkeypatch.setenv("STRICT_GPG", value)
        assert cms._resolve_strict_gpg(None) is False

    def test_cli_flag_true_wins_over_env_falsy(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("STRICT_GPG", "0")
        assert cms._resolve_strict_gpg(True) is True

    def test_pass5_low1_cli_flag_false_overrides_env_truthy(self, monkeypatch: pytest.MonkeyPatch):
        """Codex pass-5 LOW-1 fix: explicit `--no-strict-gpg` overrides
        STRICT_GPG=1 in env.

        BooleanOptionalAction adds the negative form so operators have
        an explicit opt-out from the env-set strict mode.
        """

        monkeypatch.setenv("STRICT_GPG", "1")
        assert cms._resolve_strict_gpg(False) is False


class TestStrictGpgMainExitCode:
    """``main()`` returns exit 4 when STRICT_GPG is set AND signature was skipped.

    Issue #226 H1+H4: load-bearing wiring that escalates the H4 advisory
    warning to a fail-closed CI block. Tests both the strict fail path
    (exit 4) and the back-compat warn path (exit 0).

    Tests follow the same monkeypatch-check_signoff pattern as
    ``TestStrictGhMainExitCode`` so the exit-code logic is isolated from
    upstream concerns.
    """

    @staticmethod
    def _make_results(
        all_pass: bool = True,
        signature_skipped: bool = False,
    ) -> list:
        results = [
            cms.CheckResult("filename", True, "ok"),
            cms.CheckResult("signoff_age", True, "ok"),
            cms.CheckResult("required_sections", True, "ok"),
            cms.CheckResult("signature_present", True, "ok"),
            cms.CheckResult("coi_referenced", True, "ok"),
            cms.CheckResult("registry_loaded", True, "ok"),
            cms.CheckResult("reviewer_registered", True, "ok"),
            cms.CheckResult("selection_rule", all_pass, "ok"),
            cms.CheckResult("signature_verifies", True, "ok"),
            cms.CheckResult("keyring_present", True, "ok"),
            cms.CheckResult(
                "coi_body_signature_verifies",
                True,
                "WARN: no CoI body signature found" if signature_skipped else "ok",
                signature_check_skipped=signature_skipped,
            ),
        ]
        return results

    def _patch_check_signoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        all_pass: bool = True,
        signature_skipped: bool = False,
    ) -> None:
        results = self._make_results(all_pass=all_pass, signature_skipped=signature_skipped)

        def fake_check_signoff(*args, **kwargs):  # noqa: ANN001 ANN002 ANN003
            return results

        monkeypatch.setattr(cms, "check_signoff", fake_check_signoff)

    @staticmethod
    def _make_doc_path(tmp_path: Path) -> Path:
        path = tmp_path / "optum_methodology_signoff_20260520.md"
        path.write_text("# placeholder\n", encoding="utf-8")
        return path

    def test_strict_gpg_fail_when_sig_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """STRICT_GPG=1 + signature_check_skipped=True → exit 4 + clear message."""

        self._patch_check_signoff(monkeypatch, signature_skipped=True)
        monkeypatch.setenv("STRICT_GPG", "1")
        # Ensure STRICT_GH is not set so we don't hit exit 3 by accident.
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 4, (
            f"expected exit 4 (strict-gpg policy violation), got {rc}. Captured: {captured}"
        )
        combined = captured.out + captured.err
        assert "FAIL" in combined
        assert "--strict-gpg" in combined or "STRICT_GPG=1" in combined
        assert "coi_body_signature_verifies" in combined

    def test_default_warn_when_sig_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """No STRICT_GPG + signature_skipped=True → exit 0 (advisory back-compat)."""

        self._patch_check_signoff(monkeypatch, signature_skipped=True)
        monkeypatch.delenv("STRICT_GPG", raising=False)
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 0, f"expected exit 0 (advisory back-compat), got {rc}. Captured: {captured}"
        # The WARN message should still surface in the report.
        combined = captured.out + captured.err
        assert "WARN" in combined

    def test_strict_gpg_cli_flag_overrides_missing_env(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """``--strict-gpg`` on its own (no env var) also escalates to exit 4."""

        self._patch_check_signoff(monkeypatch, signature_skipped=True)
        monkeypatch.delenv("STRICT_GPG", raising=False)
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path), "--strict-gpg"])
        assert rc == 4, f"expected exit 4 (CLI --strict-gpg), got {rc}"

    def test_validation_failure_takes_precedence_over_strict_gpg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Exit 1 (generic validation fail) wins over exit 4 (strict-gpg)."""

        self._patch_check_signoff(monkeypatch, all_pass=False, signature_skipped=True)
        monkeypatch.setenv("STRICT_GPG", "1")
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        assert rc == 1, f"expected exit 1 (generic validation precedence), got {rc}"

    def test_strict_gh_takes_precedence_over_strict_gpg(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Exit 3 (strict-gh) wins over exit 4 (strict-gpg).

        Pin: when both strict modes would trigger, the gh-provenance gap
        (rc=3) is reported first because main() evaluates the
        ``provenance_check_skipped`` gate before the
        ``signature_check_skipped`` gate. This keeps the exit-code
        priority deterministic for log scrapers.
        """

        # Build a results list with BOTH skip flags set. Use a custom
        # patcher because the helper only handles signature_skipped.
        results = [
            cms.CheckResult("filename", True, "ok"),
            cms.CheckResult(
                "selection_rule",
                True,
                "ok",
                provenance_check_skipped=True,
            ),
            cms.CheckResult(
                "coi_body_signature_verifies",
                True,
                "WARN",
                signature_check_skipped=True,
            ),
        ]

        def fake_check_signoff(*args, **kwargs):  # noqa: ANN001 ANN002 ANN003
            return results

        monkeypatch.setattr(cms, "check_signoff", fake_check_signoff)
        monkeypatch.setenv("STRICT_GH", "1")
        monkeypatch.setenv("STRICT_GPG", "1")
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        assert rc == 3, (
            f"expected exit 3 (strict-gh wins over strict-gpg by source-order), got {rc}"
        )

    def test_strict_gpg_pass_when_no_sig_skip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """STRICT_GPG=1 + all checks pass + no sig skip → exit 0.

        Pin: strict-gpg doesn't false-positive when the keyring + CoI
        sigs are present and verifying.
        """

        self._patch_check_signoff(monkeypatch, signature_skipped=False)
        monkeypatch.setenv("STRICT_GPG", "1")
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        assert rc == 0, f"expected exit 0 (strict-gpg satisfied), got {rc}"

    def test_codex_pass3_low1_exit4_message_distinguishes_subclasses(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ):
        """Codex pass-3 LOW-1 fix: exit-4 stderr distinguishes the 3 subclasses.

        The exit code 4 is reserved for "STRICT_GPG=1 + any sig-skip"
        but the underlying failure can be (a) keyring not provisioned,
        (b) CoI body sig missing, OR (c) fingerprint pinning gap. The
        stderr message must enumerate which subclass(es) fired so log
        scrapers can route the three distinct cases under the shared
        exit code.
        """

        # Build a results list where ONLY the pinning check skipped.
        # This is the (c) subclass that the pre-fix message would have
        # blamed on (a) keyring-missing.
        results = [
            cms.CheckResult("filename", True, "ok"),
            cms.CheckResult("signature_verifies", True, "sigstore verify OK"),
            cms.CheckResult("keyring_present", True, "ok"),
            cms.CheckResult("coi_body_signature_verifies", True, "ok"),
            cms.CheckResult(
                "signing_fingerprint_matches_registry",
                True,
                "WARN: pinning gap",
                signature_check_skipped=True,
            ),
        ]

        def fake_check_signoff(*args, **kwargs):  # noqa: ANN001 ANN002 ANN003
            return results

        monkeypatch.setattr(cms, "check_signoff", fake_check_signoff)
        monkeypatch.setenv("STRICT_GPG", "1")
        monkeypatch.delenv("STRICT_GH", raising=False)
        doc = self._make_doc_path(tmp_path)
        rc = cms.main([str(doc), "--repo-root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 4, f"expected exit 4 (pinning-gap subclass), got {rc}"
        # Pass-3 LOW-1: stderr must surface the PINNING GAP subclass label.
        combined = captured.out + captured.err
        assert "FINGERPRINT PINNING GAP" in combined, (
            "pass-3 LOW-1: exit-4 stderr must distinguish pinning-gap from "
            f"keyring/sig-missing subclasses; got: {combined!r}"
        )
        assert "signing_fingerprint_matches_registry" in combined


# --------------------------------------------------------------------------- #
# Issue #226 H1+H4 — workflow YAML pins for the keyring-import step.
# --------------------------------------------------------------------------- #


class TestKeyringImportWorkflow:
    """The reusable workflow must:

    1. Declare a ``strict_gpg`` workflow_call input.
    2. Have a "Provision GPG keyring" step BEFORE the validator step
       that imports from the GPG_REVIEWER_KEYS_ARMOR_BASE64 secret via
       env (NOT direct interpolation — same shell-injection lesson as
       PR #225 HIGH-2).
    3. Pass --keyring-dir to the validator only when KEYRING_DIR is
       set (no empty --keyring-dir "" arg).
    4. Reserve workflow exit code 4 for keyring-missing-under-strict.

    The caller workflow must:

    5. Pass strict_gpg: '1' to the reusable workflow.
    6. Pass ONLY the GPG_REVIEWER_KEYS_ARMOR_BASE64 secret explicitly
       (least privilege) so the reusable workflow can see it — NOT
       ``secrets: inherit`` (which leaks all repo secrets and trips
       Semgrep yaml.github-actions.security.secrets-inherit). The
       reusable workflow must DECLARE that secret under
       on.workflow_call.secrets so the explicit pass resolves.
    """

    REUSABLE_PATH = PROJECT_ROOT / ".github" / "workflows" / "methodology-signoff-validator.yml"
    CALLER_PATH = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"

    def _parse(self, path: Path):
        yaml = pytest.importorskip("yaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_reusable_declares_strict_gpg_input(self):
        parsed = self._parse(self.REUSABLE_PATH)
        on_block = parsed.get("on") or parsed.get(True)
        inputs = on_block["workflow_call"]["inputs"]
        assert "strict_gpg" in inputs, "issue #226 H1+H4: reusable must declare strict_gpg input"
        # Default must be '1' so production CI hits fail-closed.
        assert inputs["strict_gpg"].get("default") == "1"

    def test_reusable_has_keyring_import_step(self):
        parsed = self._parse(self.REUSABLE_PATH)
        steps = parsed["jobs"]["validate"]["steps"]
        keyring_step = next(
            (s for s in steps if isinstance(s.get("name"), str) and "GPG keyring" in s["name"]),
            None,
        )
        assert keyring_step is not None, (
            "issue #226 H1+H4: reusable workflow must have a 'Provision GPG keyring' step"
        )

    def test_keyring_secret_routed_through_env(self):
        """Secret must be exposed via env: not direct ${{ secrets.X }} interp.

        Mirrors PR #225 HIGH-2 (TOUCHED_FILES) — secret-store values
        passed through ``env:`` are bash-data; direct interpolation is
        a shell-injection sink even for secret values (a malicious
        actor with secret-write access could craft a payload).
        """

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Positive: must define an env var carrying the secret.
        assert (
            "GPG_REVIEWER_KEYS_ARMOR_BASE64: ${{ secrets.GPG_REVIEWER_KEYS_ARMOR_BASE64 }}" in text
        ), "issue #226 H1: secret must be routed through env:"
        # Negative: must NOT directly interpolate the secret into a run block.
        # Use the same scan as test_no_direct_input_interpolation_in_run_blocks.
        parsed = self._parse(self.REUSABLE_PATH)
        offenders: list[tuple[str, str]] = []
        for job_name, job in parsed.get("jobs", {}).items():
            for step in job.get("steps", []):
                run_block = step.get("run")
                if not isinstance(run_block, str):
                    continue
                step_name = step.get("name", "<unnamed>")
                if "${{ secrets." in run_block:
                    offenders.append((f"{job_name}::{step_name}", run_block[:200]))
        assert not offenders, (
            "issue #226 H1: ${{ secrets.X }} interpolation inside run: "
            "blocks is a shell-injection sink. Route through env:. "
            "Offenders:\n" + "\n".join(f"  {n}: {body!r}" for n, body in offenders)
        )

    def test_keyring_import_step_runs_before_validator(self):
        """Step-order pin: keyring must be provisioned BEFORE the validator.

        Without this ordering the validator's --keyring-dir would point
        at an empty/missing dir on first invocation.
        """

        parsed = self._parse(self.REUSABLE_PATH)
        steps = parsed["jobs"]["validate"]["steps"]
        keyring_idx = None
        validator_idx = None
        for idx, step in enumerate(steps):
            name = step.get("name") or ""
            if "GPG keyring" in name:
                keyring_idx = idx
            if "Run validator" in name:
                validator_idx = idx
        assert keyring_idx is not None, "keyring step missing"
        assert validator_idx is not None, "validator step missing"
        assert keyring_idx < validator_idx, (
            f"keyring step (idx={keyring_idx}) must precede validator step (idx={validator_idx})"
        )

    def test_validator_exit_code_4_reserved(self):
        """Workflow must propagate validator exit 4 with explicit precedence."""

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Positive: STATUS=4 branch must be present in the priority logic.
        assert "STATUS=4" in text, "issue #226 H1+H4: workflow must propagate validator exit 4"
        # The keyring-import step must also reserve exit 4 for the
        # missing-secret-under-strict path.
        assert "exit 4" in text, "keyring-import step must `exit 4` under STRICT_GPG=1"

    def test_validator_invocation_passes_keyring_dir_when_set(self):
        """The validator command line must include --keyring-dir conditionally."""

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # The exact array-construction pattern. The conditional is what
        # protects against passing --keyring-dir "" (which gpg would
        # interpret as cwd-relative).
        assert "--keyring-dir" in text
        assert 'if [ -n "${KEYRING_DIR:-}"' in text, (
            "issue #226: validator invocation must guard --keyring-dir on KEYRING_DIR being set"
        )

    def test_reusable_step_passes_strict_gpg_env(self):
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        assert "STRICT_GPG: ${{ inputs.strict_gpg }}" in text, (
            "issue #226: reusable workflow must export STRICT_GPG to validator"
        )

    def test_caller_passes_strict_gpg_one(self):
        text = self.CALLER_PATH.read_text(encoding="utf-8")
        assert "strict_gpg: '1'" in text, "issue #226: caller must pass strict_gpg: '1'"

    def test_caller_passes_gpg_secret_explicitly(self):
        """Least privilege (supersedes the old issue #226 H1 ``secrets: inherit``):

        the caller must pass ONLY the GPG_REVIEWER_KEYS_ARMOR_BASE64 secret to the
        reusable workflow explicitly, NOT ``secrets: inherit`` (which would forward all
        repo secrets — Semgrep yaml.github-actions.security.secrets-inherit). The reusable
        workflow must DECLARE that secret under on.workflow_call.secrets so the explicit
        pass resolves. The keyring step still reads it via env: (test above), so the GPG
        verification behaviour is unchanged.
        """

        # Least privilege: must NOT forward all secrets.
        caller_text = self.CALLER_PATH.read_text(encoding="utf-8")
        assert "secrets: inherit" not in caller_text, (
            "least privilege: caller must not use `secrets: inherit`; pass only the "
            "GPG_REVIEWER_KEYS_ARMOR_BASE64 secret explicitly"
        )

        # Caller must pass the GPG secret explicitly so the reusable workflow can read it.
        caller = self._parse(self.CALLER_PATH)
        validate_secrets = caller["jobs"]["validate"].get("secrets") or {}
        assert validate_secrets.get("GPG_REVIEWER_KEYS_ARMOR_BASE64") == (
            "${{ secrets.GPG_REVIEWER_KEYS_ARMOR_BASE64 }}"
        ), (
            "issue #226 H1: caller must pass GPG_REVIEWER_KEYS_ARMOR_BASE64 explicitly to "
            "the reusable workflow"
        )

        # Reusable workflow must DECLARE the secret so the explicit pass resolves.
        reusable = self._parse(self.REUSABLE_PATH)
        on_block = reusable.get("on") or reusable.get(True)
        declared = on_block["workflow_call"].get("secrets") or {}
        assert "GPG_REVIEWER_KEYS_ARMOR_BASE64" in declared, (
            "issue #226 H1: reusable workflow must declare GPG_REVIEWER_KEYS_ARMOR_BASE64 "
            "under on.workflow_call.secrets for the least-privilege explicit pass to resolve"
        )

    def test_workflow_documents_secret_name(self):
        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        assert "GPG_REVIEWER_KEYS_ARMOR_BASE64" in text, (
            "issue #226: reusable workflow must document the secret name"
        )

    def test_keyring_import_captures_full_pipestatus(self):
        """Codex pass-1 MED-2 fix pin: workflow must capture EVERY pipeline rc.

        The pre-fix pattern (single-index PIPESTATUS capture) caught
        only ``gpg --import`` and silently swallowed ``base64 -d``
        failures. If base64 emits partial decoded output and exits
        non-zero, gpg can still import partial key material and return
        0, so a malformed secret would silently become a populated
        keyring under STRICT_GPG=1. The fix snapshots PIPESTATUS into
        an array and asserts EACH stage succeeded.
        """

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Positive: array snapshot must be present.
        assert 'PIPE_RC=("${PIPESTATUS[@]}")' in text, (
            "MED-2: workflow must snapshot PIPESTATUS into an array"
        )
        # Each pipeline stage RC must be checked.
        assert 'BASE64_RC="${PIPE_RC[1]}"' in text, "MED-2: must capture base64 rc"
        assert 'IMPORT_RC="${PIPE_RC[2]}"' in text, "MED-2: must capture gpg rc"
        assert 'PRINTF_RC="${PIPE_RC[0]}"' in text, "MED-2: must capture printf rc"
        # Negative: the pre-fix single-index assignment pattern must NOT
        # remain in actual SHELL CODE. We accept the pattern inside YAML
        # comments (which describe the bug being fixed).
        # Walk the YAML run blocks; strip leading-`#` comment lines;
        # then assert the bad pattern is absent from the residue.
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(text)
        offenders: list[str] = []
        for job_name, job in parsed.get("jobs", {}).items():
            for step in job.get("steps", []):
                run_block = step.get("run")
                if not isinstance(run_block, str):
                    continue
                code_lines = [
                    line for line in run_block.splitlines() if not line.lstrip().startswith("#")
                ]
                code_only = "\n".join(code_lines)
                if "IMPORT_RC=${PIPESTATUS[2]}" in code_only:
                    offenders.append(f"{job_name}::{step.get('name', '<unnamed>')}")
        assert not offenders, (
            "MED-2: the pre-fix `IMPORT_RC=${PIPESTATUS[2]}` pattern must "
            f"be removed from actual shell code. Offenders: {offenders}"
        )

    def test_keyring_import_pipestatus_captured_under_set_e_pipefail(self):
        """Codex pass-1 MED-2 sibling pin: model `set -euo pipefail` semantics.

        Same lesson as the LOW-2 / MED-1 fix in PR #225's exit-code
        bracketing: under ``set -euo pipefail``, the keyring-import
        pipeline would exit the shell on any non-zero rc BEFORE
        PIPESTATUS could be read. The fix brackets with ``set +e`` /
        ``set -e``. Concrete bash harness exercises the production
        pattern with a synthetic 4-stage pipeline whose middle stage
        deliberately fails (rc=1).
        """

        if shutil.which("bash") is None:
            pytest.skip("bash unavailable")

        # Use 4 stages mirroring printf | base64 | gpg | tee. We
        # synthesize each stage with `true` / `false` so the test does
        # not depend on base64 / gpg availability AND so output is
        # deterministic ASCII (avoids UnicodeDecodeError).
        script = """
        set -euo pipefail
        STATUS=0
        set +e
        true | false | true | true
        PIPE_RC=("${PIPESTATUS[@]}")
        set -e
        PRINTF_RC="${PIPE_RC[0]}"
        BASE64_RC="${PIPE_RC[1]}"
        IMPORT_RC="${PIPE_RC[2]}"
        if [ "$PRINTF_RC" -ne 0 ] || [ "$BASE64_RC" -ne 0 ] || [ "$IMPORT_RC" -ne 0 ]; then
          STATUS=4
        fi
        echo "FINAL_STATUS=$STATUS PRINTF_RC=$PRINTF_RC BASE64_RC=$BASE64_RC IMPORT_RC=$IMPORT_RC"
        """
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
        # Load-bearing: even though tee (final stage) returned 0, the
        # middle-stage rc=1 is captured and STATUS escalates to 4.
        # Without the set+e bracket, the pipeline's rc=1 would have
        # exited the shell before PIPE_RC was read.
        assert "FINAL_STATUS=4" in result.stdout
        assert "BASE64_RC=1" in result.stdout
        assert "PRINTF_RC=0" in result.stdout
        assert "IMPORT_RC=0" in result.stdout

    def test_keyring_import_step_emits_exit_4_under_strict(self):
        """Issue #226 H1: keyring-import step exits 4 under STRICT_GPG=1.

        Both the missing-secret branch AND the import-pipeline-failure
        branch must `exit 4` under STRICT_GPG=1. Pin both code paths.
        """

        text = self.REUSABLE_PATH.read_text(encoding="utf-8")
        # Count occurrences of `exit 4` within the keyring step's run
        # block. Should be present in (a) the gpg-not-on-PATH branch,
        # (b) the missing-secret branch, (c) the import-pipeline-fail
        # branch.
        # The simplest assertion: at least 3 `exit 4` appear in the file.
        exit_4_count = text.count("exit 4")
        assert exit_4_count >= 3, (
            f"issue #226 H1: expected at least 3 `exit 4` (gpg-missing, "
            f"secret-missing, import-fail under STRICT_GPG=1); got {exit_4_count}"
        )


# --------------------------------------------------------------------------- #
# Issue #226 H1+H4 — n3_known_limitations + operator handoff doc pins.
# --------------------------------------------------------------------------- #


class TestN3LimitationsAndOperatorDoc:
    """The known-limitations doc and the new operator handoff doc must reflect
    the issue #226 partial-resolved state.
    """

    N3_DOC = PROJECT_ROOT / "docs" / "governance" / "n3_known_limitations_20260510.md"
    OPERATOR_DOC = PROJECT_ROOT / "docs" / "governance" / "operator_gpg_keyring_setup.md"

    def test_n3_doc_marks_h1_partially_resolved(self):
        text = self.N3_DOC.read_text(encoding="utf-8")
        # H1 (item 1) must reflect issue #226 partial resolution.
        # Look for both "H1" anchor AND the PARTIALLY RESOLVED marker
        # in proximity. Since PARTIALLY RESOLVED is ALSO used for H3,
        # we require a line that mentions issue #226.
        assert "issue #226" in text, "n3 doc must reference issue #226"
        # H1 section header should be updated (was "H1 PARTIAL → ACCEPTED-RISK").
        assert "H1 PARTIAL → PARTIALLY RESOLVED" in text or (
            "H1 PARTIAL → ACCEPTED-RISK" not in text and "H1" in text
        ), "n3 doc must update H1 status from ACCEPTED-RISK to PARTIALLY RESOLVED"

    def test_n3_doc_marks_h4_partially_resolved(self):
        text = self.N3_DOC.read_text(encoding="utf-8")
        # H4 (item 4) must similarly flip.
        assert "H4 PARTIAL → PARTIALLY RESOLVED" in text or (
            "H4 PARTIAL → ACCEPTED-RISK" not in text and "H4" in text
        ), "n3 doc must update H4 status from ACCEPTED-RISK to PARTIALLY RESOLVED"

    def test_operator_doc_exists(self):
        assert self.OPERATOR_DOC.is_file(), (
            f"issue #226: operator handoff doc must exist at {self.OPERATOR_DOC}"
        )

    def test_operator_doc_covers_required_steps(self):
        text = self.OPERATOR_DOC.read_text(encoding="utf-8")
        # Required content (loose pins so doc can rephrase but must
        # cover the same operational ground):
        for marker in (
            "GPG_REVIEWER_KEYS_ARMOR_BASE64",
            "fingerprint",
            "STRICT_GPG",
            "base64",
        ):
            assert marker in text, f"issue #226: operator handoff doc must cover '{marker}'"

    def test_operator_doc_warns_about_strict_default(self):
        """Codex pass-1 LOW-1 fix: doc must NOT mislead operators that
        pre-secret CI runs are advisory.

        The caller workflow defaults `strict_gpg: '1'` which means CI
        exits 4 immediately after merge if the secret/fingerprints are
        not in place. The doc must surface this clearly so operators
        know to either (a) provision the secret BEFORE merging this
        change, or (b) temporarily flip the caller to `strict_gpg: '0'`
        as a controlled rollout.
        """

        text = self.OPERATOR_DOC.read_text(encoding="utf-8")
        # Doc must reference the strict-default + opt-out option.
        assert "fail" in text.lower() and "closed" in text.lower(), (
            "LOW-1: operator doc must explain CI fails closed by default"
        )
        # Must reference the opt-out path (`strict_gpg: '0'`).
        assert "strict_gpg: '0'" in text, (
            "LOW-1: operator doc must document the temporary advisory-mode opt-out"
        )
