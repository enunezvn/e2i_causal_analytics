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

| name | email | github_handle | role | date_added | areas_of_expertise | status |
|---|---|---|---|---|---|---|
| Alice Eligible | alice@example.com | alice | clinician | 2026-05-10 | methodology | active |
| Bob Conflicted | bob@example.com | bob | biostat | 2026-05-10 | methodology | active |
| Carol Inactive | carol@example.com | carol | advisor | 2026-05-10 | methodology | inactive |
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
    (repo / "docs" / "governance" / "coi_declarations" / "alice_20260510.md").write_text(
        "# CoI Alice\n\nzero touches\n", encoding="utf-8"
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

    yield repo


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


def test_signature_present_real_block_accepted():
    text = (
        "## Cryptographic signature\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "AAAARealLooKingDataBLOB\n"
        "-----END PGP SIGNATURE-----\n"
    )
    result = cms.check_signature_present(text)
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


def test_check_signoff_full_success(fixture_repo: Path, monkeypatch: pytest.MonkeyPatch):
    """Compose a valid registry + signoff and assert the full check passes."""

    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="alice"), encoding="utf-8")

    # Prevent --require-signature path failure when toolchain absent.
    results = cms.check_signoff(signoff_path, fixture_repo, require_signature=False)
    failed = [r for r in results if not r.ok]
    assert failed == [], f"unexpected failures: {[(r.name, r.detail) for r in failed]}"


def test_check_signoff_missing_signature_section(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(
        _signoff_doc(handle="alice", omit_section="## Cryptographic signature"),
        encoding="utf-8",
    )
    results = cms.check_signoff(signoff_path, fixture_repo, require_signature=False)
    assert any(r.name == "required_sections" and not r.ok for r in results)


def test_check_signoff_unregistered_reviewer(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="zoe"), encoding="utf-8")
    results = cms.check_signoff(signoff_path, fixture_repo, require_signature=False)
    assert any(r.name == "reviewer_registered" and not r.ok for r in results)


def test_check_signoff_selection_rule_violation(
    fixture_repo_with_bob_conflict: Path,
):
    repo = fixture_repo_with_bob_conflict
    signoff_path = repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="bob"), encoding="utf-8")
    results = cms.check_signoff(signoff_path, repo, require_signature=False)
    assert any(r.name == "selection_rule" and not r.ok for r in results)


def test_check_signoff_missing_coi_sha(fixture_repo: Path):
    signoff_path = fixture_repo / "docs" / "results" / "optum_methodology_signoff_20260520.md"
    signoff_path.write_text(_signoff_doc(handle="alice", coi_sha="<sha>"), encoding="utf-8")
    results = cms.check_signoff(signoff_path, fixture_repo, require_signature=False)
    assert any(r.name == "coi_referenced" and not r.ok for r in results)


# --------------------------------------------------------------------------- #
# CI workflow YAML must parse.
# --------------------------------------------------------------------------- #


def test_workflow_yaml_parses():
    """The methodology_signoff_guard.yml workflow must be valid YAML."""

    yaml = pytest.importorskip("yaml")
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "methodology_signoff_guard.yml"
    assert workflow_path.is_file(), f"workflow missing: {workflow_path}"
    parsed = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    # Top-level YAML on/jobs keys must be present. (PyYAML interprets bare
    # `on:` as the boolean True, so we accept either the string or the bool.)
    assert "on" in parsed or True in parsed
    assert "jobs" in parsed
    assert "validate-signoff" in parsed["jobs"]
    steps = parsed["jobs"]["validate-signoff"]["steps"]
    # Find the step that invokes the python script.
    invokes = [s for s in steps if "check_methodology_signoff.py" in s.get("run", "")]
    assert invokes, "workflow does not invoke check_methodology_signoff.py"


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


def test_extract_pgp_armor_block_returns_full_block():
    text = "before\n-----BEGIN PGP SIGNATURE-----\nAAAA\n-----END PGP SIGNATURE-----\nafter\n"
    block = cms._extract_pgp_armor_block(text)
    assert block is not None
    assert block.startswith("-----BEGIN PGP SIGNATURE-----")
    assert block.endswith("-----END PGP SIGNATURE-----")
    assert "AAAA" in block


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


def test_selection_rule_gh_missing_emits_warning(fixture_repo: Path, monkeypatch):
    """When gh is unavailable, selection rule still PASSES (with WARN in detail)."""

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
    # Warning surfaces, but does not fail the check.
    assert "WARN" in result.detail
    assert "gh-author" in result.detail or "gh not on PATH" in result.detail
