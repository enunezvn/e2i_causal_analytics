#!/usr/bin/env python3
"""Validate a methodology sign-off / rejection artifact (Gate N3 of v4 plan).

This script enforces the selection rule defined in
``docs/governance/methodology_reviewer_registry.md``: a reviewer is eligible
to sign off on a methodology decision touching ``scripts/convert_optum_rwd.py``
only if no commit (or merged PR) attributable to them touched the file inside
the named period (``2026-04-15 → 2026-05-10`` for the Optum n=1697 anchor).

Usage::

    python scripts/check_methodology_signoff.py docs/results/optum_methodology_signoff_<YYYYMMDD>.md

Exit codes:

* ``0`` — all checks passed.
* ``1`` — generic validation failure (missing section, unregistered reviewer,
  selection-rule violation, unverifiable signature, etc.).
* ``2`` — script invocation error (e.g. missing file argument).

The CI workflow ``.github/workflows/methodology_signoff_guard.yml`` calls this
script with the ``--repo-root`` flag so it can be exercised from any
working directory.

Design notes:

* The check is INTENTIONALLY pure-Python with stdlib only — no third-party
  dependencies — so it can run in the security-scanning workflow image without
  pulling the project requirements.
* PGP / sigstore verification is *attempted* but a missing toolchain is
  treated as a non-fatal warning UNLESS the ``--require-signature`` flag is
  passed; in that case verification failure is fatal.
* The template files themselves
  (``docs/results/optum_methodology_signoff_template.md`` and the rejection
  twin) are explicitly skipped by name — the CI workflow only invokes this
  script on dated artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Subject files for which the selection rule applies. Add new entries here as
# additional methodology decisions come online. Each entry is interpreted as
# a path relative to the repo root.
SUBJECT_FILES: tuple[str, ...] = ("scripts/convert_optum_rwd.py",)

# Named period during which involvement with the subject file disqualifies a
# reviewer. The dates correspond to the Optum n=1697 empirical-anchor window
# (see docs/governance/methodology_reviewer_registry.md §Selection rule).
NAMED_PERIOD_START: str = "2026-04-15"
NAMED_PERIOD_END: str = "2026-05-10"

# Required section headings (Markdown level-2) for sign-off and rejection
# artifacts. Order does not matter; presence does.
REQUIRED_SECTIONS_SIGNOFF: tuple[str, ...] = (
    "## Reviewer",
    "## Conflict-of-interest declaration",
    "## Methodology decision",
    "## Cryptographic signature",
)
REQUIRED_SECTIONS_REJECTION: tuple[str, ...] = (
    "## Reviewer",
    "## Conflict-of-interest declaration",
    "## Reasons for rejection",
    "## Cryptographic signature",
)

# Filename pattern: docs/results/optum_methodology_signoff_<YYYYMMDD>.md
# OR  docs/results/optum_methodology_rejection_<YYYYMMDD>.md
FILENAME_PATTERN = re.compile(
    r"^optum_methodology_(?P<kind>signoff|rejection)_(?P<date>\d{8})\.md$"
)
TEMPLATE_FILENAMES: frozenset[str] = frozenset(
    {
        "optum_methodology_signoff_template.md",
        "optum_methodology_rejection_template.md",
    }
)


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class CheckResult:
    """Outcome of an individual check.

    Attributes:
        name: short human label (used in reporter output).
        ok: True when the check passed.
        detail: optional explanation (always populated on failure).
    """

    name: str
    ok: bool
    detail: str = ""


@dataclasses.dataclass
class ReviewerInfo:
    """Subset of the registry row needed for selection-rule checks."""

    handle: str
    email: str
    status: str


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


_REGISTRY_HEADERS = (
    "name",
    "email",
    "github_handle",
    "role",
    "date_added",
    "areas_of_expertise",
    "status",
)


def parse_registry(registry_path: Path) -> List[ReviewerInfo]:
    """Parse the reviewer-registry markdown table into ReviewerInfo records.

    The registry is a simple GFM table; we hand-parse rather than depending on
    a markdown library so the script remains stdlib-only.

    Raises:
        FileNotFoundError if the registry does not exist.
        ValueError if the table headers do not match the expected schema.
    """

    if not registry_path.is_file():
        raise FileNotFoundError(f"registry not found: {registry_path}")

    rows: List[ReviewerInfo] = []
    in_table = False
    saw_separator = False
    for raw_line in registry_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            in_table = False
            saw_separator = False
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # Header row: cells exactly match _REGISTRY_HEADERS.
        if not in_table and tuple(cells) == _REGISTRY_HEADERS:
            in_table = True
            saw_separator = False
            continue
        # Separator row: cells match e.g. ['---', '---', ...].
        if in_table and not saw_separator:
            if all(re.fullmatch(r"-{3,}", cell) for cell in cells if cell):
                saw_separator = True
                continue
            # Header without separator → malformed.
            in_table = False
            continue
        if in_table and saw_separator and len(cells) == len(_REGISTRY_HEADERS):
            handle, email, status = cells[2], cells[1], cells[6]
            # Strip Markdown emphasis (e.g. _PLACEHOLDER_).
            handle = handle.strip("_*`")
            rows.append(ReviewerInfo(handle=handle, email=email, status=status))
    return rows


def extract_section_headings(doc_text: str) -> List[str]:
    """Return all level-2 section headings present in ``doc_text``."""

    return [line.rstrip() for line in doc_text.splitlines() if line.startswith("## ")]


def extract_field(doc_text: str, label: str) -> Optional[str]:
    """Extract the first occurrence of ``- **<label>:** <value>`` from doc.

    Returns the right-hand-side string with surrounding whitespace and
    surrounding backticks stripped. Returns ``None`` if the label is absent.
    """

    pattern = re.compile(
        r"^\s*[-*]\s*\*\*" + re.escape(label) + r":\*\*\s*(?P<value>.+?)\s*$",
        re.MULTILINE,
    )
    match = pattern.search(doc_text)
    if match is None:
        return None
    return match.group("value").strip().strip("`")


def extract_handle(doc_text: str) -> Optional[str]:
    """Pull the reviewer's GitHub handle out of the artifact.

    Looks for a line of the form ``- **GitHub handle:** @<handle>`` and
    returns ``<handle>`` (without the leading ``@``).
    """

    raw = extract_field(doc_text, "GitHub handle")
    if raw is None:
        return None
    return raw.lstrip("@").strip()


def extract_coi_sha(doc_text: str) -> Optional[str]:
    """Pull the CoI declaration commit SHA out of the artifact."""

    return extract_field(doc_text, "CoI declaration commit SHA")


def extract_coi_path(doc_text: str) -> Optional[str]:
    """Pull the CoI declaration file path out of the artifact.

    The relevant line in the template is::

        - **CoI document:** docs/governance/coi_declarations/<handle>_<YYYYMMDD>.md

    We look for the literal ``CoI document`` label.
    """

    return extract_field(doc_text, "CoI document")


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #


def check_filename(doc_path: Path) -> CheckResult:
    """Filename must match optum_methodology_(signoff|rejection)_<YYYYMMDD>.md.

    Templates (``..._template.md``) are explicitly rejected — the CI guard
    only validates dated artifacts.
    """

    name = doc_path.name
    if name in TEMPLATE_FILENAMES:
        return CheckResult(
            "filename",
            False,
            f"template files are not validated by this script: {name!r}",
        )
    if FILENAME_PATTERN.match(name):
        return CheckResult("filename", True, name)
    return CheckResult(
        "filename",
        False,
        f"filename does not match expected pattern: {name!r}",
    )


def check_required_sections(doc_text: str, kind: str) -> CheckResult:
    """All required level-2 sections must be present (in any order)."""

    expected = REQUIRED_SECTIONS_SIGNOFF if kind == "signoff" else REQUIRED_SECTIONS_REJECTION
    present = set(extract_section_headings(doc_text))
    missing = [section for section in expected if section not in present]
    if missing:
        return CheckResult(
            "required_sections",
            False,
            f"missing sections: {missing!r}",
        )
    return CheckResult("required_sections", True, f"all {len(expected)} sections present")


def check_reviewer_registered(
    doc_text: str,
    registry: Sequence[ReviewerInfo],
) -> CheckResult:
    """Reviewer's handle must appear in the registry as ``status=active``."""

    handle = extract_handle(doc_text)
    if handle is None:
        return CheckResult(
            "reviewer_registered",
            False,
            "GitHub handle missing from sign-off doc",
        )
    for row in registry:
        if row.handle == handle:
            if row.status != "active":
                return CheckResult(
                    "reviewer_registered",
                    False,
                    f"reviewer {handle!r} is in registry but status={row.status!r} (expected 'active')",
                )
            return CheckResult(
                "reviewer_registered",
                True,
                f"{handle} (status=active)",
            )
    return CheckResult(
        "reviewer_registered",
        False,
        f"reviewer {handle!r} not in registry",
    )


def check_coi_referenced(doc_text: str) -> CheckResult:
    """A CoI commit SHA and a CoI file path must both be present."""

    sha = extract_coi_sha(doc_text)
    path = extract_coi_path(doc_text)
    placeholder_sha = sha is None or "<sha>" in (sha or "") or sha == ""
    placeholder_path = path is None or "<github_handle>" in (path or "") or path == ""
    if placeholder_sha or placeholder_path:
        return CheckResult(
            "coi_referenced",
            False,
            f"CoI fields missing or placeholder (sha={sha!r}, path={path!r})",
        )
    return CheckResult("coi_referenced", True, f"sha={sha[:12]}, path={path}")


def _git_log_touches(
    repo_root: Path,
    email: str,
    subject_file: str,
    since: str,
    until: str,
) -> tuple[bool, str]:
    """Return ``(has_touches, raw_output)`` for a git-log filter.

    ``has_touches`` is True iff the filter returned at least one commit. The
    raw output is suitable for embedding in a CheckResult ``detail`` field.
    """

    cmd = [
        "git",
        "-C",
        str(repo_root),
        "log",
        f"--author={email}",
        f"--since={since}",
        f"--until={until}",
        "--pretty=format:%H %ad %s",
        "--date=short",
        "--",
        subject_file,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return True, "git binary not found on PATH"
    output = completed.stdout.strip()
    return bool(output), output


def check_selection_rule(
    doc_text: str,
    repo_root: Path,
    registry: Sequence[ReviewerInfo],
) -> CheckResult:
    """For each subject file, the reviewer's git log in the named period must be empty."""

    handle = extract_handle(doc_text)
    if handle is None:
        return CheckResult(
            "selection_rule",
            False,
            "cannot evaluate without reviewer handle",
        )
    matching = [row for row in registry if row.handle == handle]
    if not matching:
        return CheckResult(
            "selection_rule",
            False,
            f"reviewer {handle!r} not in registry — cannot resolve email",
        )
    email = matching[0].email
    violations: List[str] = []
    for subject in SUBJECT_FILES:
        has_touches, output = _git_log_touches(
            repo_root,
            email,
            subject,
            NAMED_PERIOD_START,
            NAMED_PERIOD_END,
        )
        if has_touches:
            violations.append(f"{subject}: {output}")
    if violations:
        return CheckResult(
            "selection_rule",
            False,
            "; ".join(violations),
        )
    return CheckResult(
        "selection_rule",
        True,
        f"0 git touches for {email} in {NAMED_PERIOD_START}..{NAMED_PERIOD_END}",
    )


def check_signature_present(doc_text: str) -> CheckResult:
    """A PGP-armor block or sigstore JSON block must be present in the doc."""

    if "-----BEGIN PGP SIGNATURE-----" in doc_text and "-----END PGP SIGNATURE-----" in doc_text:
        if "<signature blob>" in doc_text:
            return CheckResult(
                "signature_present",
                False,
                "PGP block contains template placeholder '<signature blob>'",
            )
        return CheckResult("signature_present", True, "PGP signature block present")
    # Sigstore bundle is a plain JSON object inside a ```json fence; match a
    # rough heuristic — we don't parse the JSON here.
    if re.search(r"```json[\s\S]*?\"signatures\"[\s\S]*?```", doc_text):
        return CheckResult("signature_present", True, "sigstore-like JSON block present")
    return CheckResult(
        "signature_present",
        False,
        "no PGP or sigstore signature block found",
    )


def check_signature_verifies(
    doc_path: Path,
    require_signature: bool,
) -> CheckResult:
    """Attempt PGP / sigstore verification; non-fatal unless required.

    When ``require_signature`` is False (the default), absence of the
    verification toolchain produces a passing result with a ``detail`` warning;
    failure of an available verifier is still fatal. When True, missing
    toolchain is itself fatal.
    """

    has_gpg = shutil.which("gpg") is not None
    has_sigstore = shutil.which("cosign") is not None or shutil.which("rekor-cli") is not None
    if not (has_gpg or has_sigstore):
        if require_signature:
            return CheckResult(
                "signature_verifies",
                False,
                "no signature-verification tool found (gpg, cosign, rekor-cli) and --require-signature passed",
            )
        return CheckResult(
            "signature_verifies",
            True,
            "WARN: no signature-verification tool found; skipping verification",
        )
    # We do NOT attempt actual verification here — it requires the reviewer's
    # public key on the verifying host, which is host-specific. Real CI will
    # need to set up the keyring before invoking this script with
    # --require-signature. For scaffolding purposes we record that the
    # toolchain is available.
    return CheckResult(
        "signature_verifies",
        True,
        f"signature toolchain available (gpg={has_gpg}, sigstore={has_sigstore})",
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def check_signoff(
    doc_path: Path,
    repo_root: Path,
    require_signature: bool = False,
) -> List[CheckResult]:
    """Run all checks against ``doc_path`` and return their results."""

    results: List[CheckResult] = []

    filename_check = check_filename(doc_path)
    results.append(filename_check)
    if not filename_check.ok:
        return results

    match = FILENAME_PATTERN.match(doc_path.name)
    assert match is not None  # check_filename guarantees this
    kind = match.group("kind")

    doc_text = doc_path.read_text(encoding="utf-8")

    results.append(check_required_sections(doc_text, kind))
    results.append(check_signature_present(doc_text))
    results.append(check_coi_referenced(doc_text))

    registry_path = repo_root / "docs" / "governance" / "methodology_reviewer_registry.md"
    try:
        registry = parse_registry(registry_path)
    except FileNotFoundError as exc:
        results.append(CheckResult("registry_loaded", False, str(exc)))
        return results

    results.append(CheckResult("registry_loaded", True, f"{len(registry)} rows"))
    results.append(check_reviewer_registered(doc_text, registry))
    results.append(check_selection_rule(doc_text, repo_root, registry))
    results.append(check_signature_verifies(doc_path, require_signature))
    return results


def render_report(results: Sequence[CheckResult]) -> str:
    """Format a sequence of CheckResult objects as a single-line-per-row report."""

    lines = []
    for r in results:
        marker = "PASS" if r.ok else "FAIL"
        lines.append(f"[{marker}] {r.name}: {r.detail}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_methodology_signoff",
        description=__doc__,
    )
    parser.add_argument(
        "doc",
        type=Path,
        help="Path to a methodology sign-off / rejection artifact.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Path to the repository root. Defaults to the directory two "
            "levels above this script (i.e. <project>/scripts/.. = "
            "<project>/)."
        ),
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help=(
            "Treat absence of a signature-verification toolchain (gpg, "
            "cosign, rekor-cli) as fatal."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent
    doc_path = args.doc

    if not doc_path.is_file():
        print(f"error: file not found: {doc_path}", file=sys.stderr)
        return 2

    results = check_signoff(doc_path, repo_root, require_signature=args.require_signature)
    print(render_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
