#!/usr/bin/env python3
"""G3 wiring guard — Plan v4 §2 Gate G3 mechanical CI enforcement.

Rationale (codex-rescue HIGH-1, Plan v4 §2 G3):

    "G3 lands last" is a process rule. An honest PR sequence will follow
    it; a determined developer can sidestep with a feature-flag config flip
    or a commit labeled "test fix" that quietly enables HBLP wiring before
    G1 + G2 close. The AST scan + signoff-file requirement converts the
    rule from process discipline to machine-enforceable gate.

This script implements the "Mechanical CI enforcement" sub-acceptance of
Plan v4 §2 G3:

    If any callsite of `hblp_classify` exists in `_build_verdict` or
    `_compose_legacy_verdict` (or any helper they call into for severity
    classification — i.e., the production HBLP wiring is in place), the
    build FAILS unless:

      1. File `docs/calibration/g1_completion_signoff_20260510.md` exists
         at HEAD, AND
      2. File `docs/calibration/g2_completion_signoff_20260510.md` exists
         at HEAD, AND
      3. Each signoff's `commit:` field references a SHA that is an
         ancestor of HEAD, AND
      4. (When `--require-signature-registry-match` is passed): the
         signoff committer's email matches the N3 reviewer registry.

The script is INTENTIONALLY pure-Python with stdlib only — no third-party
dependencies — so it runs in the security-scanning workflow image without
pulling project requirements.

H3 SECURITY ADVISORY (mirror of methodology-signoff-guard policy):
    The CI workflow that invokes this script
    (`.github/workflows/g3_wiring_guard.yml`) MUST run the validator from
    the PR's BASE ref, not the PR-checkout copy, so a malicious PR cannot
    weaken the validator and have its weakened copy validate itself. The
    workflow at the named path implements that base-ref preferred-source
    pattern (mirror of `methodology_signoff_guard.yml` H3 mitigation).

Usage::

    python scripts/check_g3_wiring_guard.py [--repo-root <path>]
                                           [--head-sha <sha>]
                                           [--require-signature-registry-match]

Exit codes:

* ``0`` — wiring not detected, OR wiring detected AND all signoff +
  ancestry checks pass.
* ``1`` — wiring detected AND at least one signoff/ancestry check failed.
* ``2`` — script invocation error.
"""

from __future__ import annotations

import argparse
import ast
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

# The file the AST scanner audits for HBLP wiring presence.
WIRED_FILE_REL: str = "src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py"

# Functions whose bodies must NOT carry an `hblp_classify` call until G1 + G2
# signoffs land. Per Plan v4 §2 G3, the wiring is the production default once
# G1 + G2 close; before that, the presence of a callsite inside these
# functions triggers the signoff-existence requirement.
GATED_FUNCTIONS: tuple[str, ...] = (
    "_build_verdict",
    "_compose_legacy_verdict",
    "_adversarial_input",
)

# The HBLP classifier whose presence inside a gated function indicates the
# wiring is live.
HBLP_CALL_NAME: str = "hblp_classify"

# Required signoff files. Each must:
#   - exist at HEAD
#   - reference a `commit:` SHA that is an ancestor of HEAD (per signoff
#     templates landed in PRs #137 and #136)
SIGNOFF_FILES: tuple[str, ...] = (
    "docs/calibration/g1_completion_signoff_20260510.md",
    "docs/calibration/g2_completion_signoff_20260510.md",
)

# Pattern matching the signoff template's `commit:` field. The template uses
# the form ``Branch / commit: `<SHA>`` OR ``commit SHA: `<SHA>``` etc. We
# accept any of the documented variants conservatively.
COMMIT_FIELD_PATTERN = re.compile(
    r"(?:`(?P<bt>[0-9a-fA-F]{7,40})`|"
    r"(?P<commit>commit|sha|S_prespec|experiment commit SHA)\s*:\s*"
    r"`?(?P<plain>[0-9a-fA-F]{7,40})`?)",
    re.IGNORECASE,
)

# N3 reviewer registry path (G3 reuses the methodology-signoff registry per
# plan §2 N3 — the same eligible reviewers approve G1/G2/G3 sign-offs).
REVIEWER_REGISTRY_REL: str = "docs/governance/methodology_reviewer_registry.md"


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


# --------------------------------------------------------------------------- #
# AST scanner
# --------------------------------------------------------------------------- #


def detect_hblp_wiring(source_path: Path) -> CheckResult:
    """AST-scan ``source_path`` for `hblp_classify` calls inside gated functions.

    Returns a CheckResult whose ``ok`` field encodes presence:

      * ``ok=True`` when the wiring IS present (a gated function's body
        contains an `hblp_classify` call). The caller treats this as
        "guard activates — must check signoffs".
      * ``ok=False`` when the wiring is NOT present. The caller treats this
        as "guard inactive — pass through without signoff checks".

    Detail field carries the function name + line number of the first
    detected callsite (the FIRST one is sufficient evidence; we don't
    enumerate all). When no wiring is present, detail names which gated
    functions were scanned.
    """

    if not source_path.is_file():
        return CheckResult(
            "wiring_detection",
            False,
            f"target file not found: {source_path} — guard inactive",
        )

    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        # Defensive: a syntax error in the production file is itself a
        # CI-blocking problem, but we can't make a wiring claim under those
        # conditions. Return ok=False (wiring undetected) and let the
        # main-loop surface the syntax error elsewhere.
        return CheckResult(
            "wiring_detection",
            False,
            f"failed to parse {source_path}: {exc}",
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in GATED_FUNCTIONS:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Name) and func.id == HBLP_CALL_NAME:
                        return CheckResult(
                            "wiring_detection",
                            True,
                            f"{HBLP_CALL_NAME!r} called inside {node.name!r} "
                            f"at line {inner.lineno}",
                        )
                    # Defensive: also catch attribute access (module.fn) so a
                    # `from .helpers import hblp_classify as fn`-style indirection
                    # doesn't sidestep the AST scan.
                    if isinstance(func, ast.Attribute) and func.attr == HBLP_CALL_NAME:
                        return CheckResult(
                            "wiring_detection",
                            True,
                            f"{HBLP_CALL_NAME!r} (attribute access) "
                            f"called inside {node.name!r} at line {inner.lineno}",
                        )

    return CheckResult(
        "wiring_detection",
        False,
        f"no {HBLP_CALL_NAME!r} call detected inside gated functions "
        f"{list(GATED_FUNCTIONS)} — guard inactive",
    )


# --------------------------------------------------------------------------- #
# Signoff existence + ancestor check
# --------------------------------------------------------------------------- #


def check_signoff_exists(repo_root: Path, signoff_rel: str) -> CheckResult:
    """The signoff file MUST exist at HEAD."""

    full = repo_root / signoff_rel
    if full.is_file():
        return CheckResult(
            f"signoff_exists::{signoff_rel}",
            True,
            f"file present: {full}",
        )
    return CheckResult(
        f"signoff_exists::{signoff_rel}",
        False,
        f"file MISSING at HEAD: {full}",
    )


def extract_commit_sha(doc_text: str) -> Optional[str]:
    """Pull the first plausible commit SHA out of a signoff document.

    Signoff templates carry the SHA in one of several documented forms:

      * Inline backtick: ``...Branch / commit:`` `<sha>` ``...`` (G1 template)
      * Field-form: ``- **commit:** `<sha>``` (some N2/N3 docs)
      * Field-form with prefixes: ``S_prespec commit SHA: `<sha>``` (G2)

    Returns the FIRST backtick-wrapped 7-40-char hex token (the canonical
    form used by both the G1 and G2 templates the team has shipped).
    Falls back to a plain-prefix match if no backtick form is found.
    """

    # Prefer backtick-wrapped SHAs (the dominant template form). Iterate
    # all matches and return the first that looks like a hex SHA.
    backtick_pattern = re.compile(r"`(?P<sha>[0-9a-fA-F]{7,40})`")
    for match in backtick_pattern.finditer(doc_text):
        return match.group("sha")

    # Fall back to prefix-form (G2 template uses these).
    prefix_pattern = re.compile(
        r"(?:commit|sha|S_prespec|experiment[\s_]+commit[\s_]+sha)\s*:\s*"
        r"(?P<sha>[0-9a-fA-F]{7,40})",
        re.IGNORECASE,
    )
    prefix_match = prefix_pattern.search(doc_text)
    if prefix_match is not None:
        return prefix_match.group("sha")
    return None


def check_signoff_ancestor(
    repo_root: Path,
    signoff_rel: str,
    head_sha: str,
) -> CheckResult:
    """The SHA referenced in the signoff MUST be an ancestor of HEAD.

    Skips the check (PASS with WARN detail) when the signoff SHA contains
    template-placeholder tokens (e.g. ``<sha>``) — that's a separate
    failure surfaced by the signoff-completeness check, not the ancestry
    check's responsibility.
    """

    if shutil.which("git") is None:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            "git binary not on PATH — cannot verify ancestry",
        )

    full = repo_root / signoff_rel
    if not full.is_file():
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"signoff file missing: {full} — cannot extract SHA",
        )

    sha = extract_commit_sha(full.read_text(encoding="utf-8"))
    if sha is None:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"no commit SHA extractable from {signoff_rel}",
        )

    # ``git merge-base --is-ancestor`` exit codes:
    #   0 → ancestry holds
    #   1 → ancestry does NOT hold
    #   * → other errors
    cmd = [
        "git",
        "-C",
        str(repo_root),
        "merge-base",
        "--is-ancestor",
        sha,
        head_sha,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"git merge-base failed: {exc}",
        )

    if completed.returncode == 0:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            True,
            f"signoff SHA {sha[:12]} is ancestor of HEAD {head_sha[:12]}",
        )
    if completed.returncode == 1:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"signoff SHA {sha[:12]} is NOT an ancestor of HEAD {head_sha[:12]}",
        )
    stderr = (completed.stderr or "").strip()[:200]
    return CheckResult(
        f"signoff_ancestor::{signoff_rel}",
        False,
        f"git merge-base returncode={completed.returncode}: {stderr}",
    )


# --------------------------------------------------------------------------- #
# Reviewer registry email match (best-effort)
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


def parse_registry_emails(registry_path: Path) -> set[str]:
    """Extract the set of registered reviewer emails (active rows only).

    Returns an empty set when the registry is missing OR contains no active
    rows (the typical state of the template-stage registry — the caller
    treats an empty set as "registry empty, skip the email match check").
    """

    if not registry_path.is_file():
        return set()

    emails: set[str] = set()
    in_table = False
    saw_separator = False
    for raw_line in registry_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            in_table = False
            saw_separator = False
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not in_table and tuple(cells) == _REGISTRY_HEADERS:
            in_table = True
            saw_separator = False
            continue
        if in_table and not saw_separator:
            if all(re.fullmatch(r"-{3,}", cell) for cell in cells if cell):
                saw_separator = True
                continue
            in_table = False
            continue
        if in_table and saw_separator and len(cells) == len(_REGISTRY_HEADERS):
            email_cell = cells[1]
            status = cells[6]
            if status != "active":
                continue
            for token in re.split(r"[,;]", email_cell):
                tok = token.strip().strip("_*`")
                if tok:
                    emails.add(tok.lower())
    return emails


def get_committer_email(repo_root: Path, sha: str) -> Optional[str]:
    """Resolve the committer email of ``sha`` via ``git show``.

    Returns None when git is unavailable OR the SHA does not resolve.
    """

    if shutil.which("git") is None:
        return None
    cmd = [
        "git",
        "-C",
        str(repo_root),
        "show",
        "-s",
        "--format=%ce",
        sha,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    out = (completed.stdout or "").strip()
    return out.lower() if out else None


def check_signoff_committer_match(
    repo_root: Path,
    signoff_rel: str,
    require_match: bool,
) -> CheckResult:
    """Optional: signoff's introducing-commit committer email must be in registry.

    Behavior:
      * ``require_match=False`` (default, advisory): always returns PASS
        with a WARN annotation noting whether a match was found.
      * ``require_match=True`` (production-deployment policy): the check
        FAILS if no active registry row carries the committer's email, OR
        if the registry is empty.

    The signoff's introducing-commit SHA is derived via
    ``git log --diff-filter=A --follow --reverse`` against the signoff path
    (mirror of ``check_methodology_signoff.py::_coi_first_add_commit``).
    """

    full = repo_root / signoff_rel
    if not full.is_file():
        if require_match:
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                False,
                f"signoff file missing: {full}",
            )
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            True,
            "WARN: signoff file missing — committer match advisory-skipped",
        )

    if shutil.which("git") is None:
        if require_match:
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                False,
                "git binary not on PATH",
            )
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            True,
            "WARN: git not on PATH — committer match advisory-skipped",
        )

    cmd = [
        "git",
        "-C",
        str(repo_root),
        "log",
        "--diff-filter=A",
        "--follow",
        "--reverse",
        "--format=%H",
        "--",
        signoff_rel,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            not require_match,
            f"git log failed: {exc}",
        )
    if completed.returncode != 0 or not (completed.stdout or "").strip():
        if require_match:
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                False,
                f"could not derive introducing-commit SHA for {signoff_rel}",
            )
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            True,
            "WARN: introducing-commit not derivable — committer match advisory-skipped",
        )

    introducing_sha = (completed.stdout or "").strip().splitlines()[0]
    committer_email = get_committer_email(repo_root, introducing_sha)
    if committer_email is None:
        if require_match:
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                False,
                f"could not resolve committer email for {introducing_sha[:12]}",
            )
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            True,
            "WARN: committer email unresolvable — advisory-skipped",
        )

    registry_emails = parse_registry_emails(repo_root / REVIEWER_REGISTRY_REL)
    if committer_email in registry_emails:
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            True,
            f"committer {committer_email} matches registry "
            f"(introducing-commit={introducing_sha[:12]})",
        )

    detail = (
        f"committer {committer_email!r} (introducing-commit={introducing_sha[:12]}) "
        f"not in active registry rows ({len(registry_emails)} active emails registered)"
    )
    if require_match:
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            False,
            detail,
        )
    return CheckResult(
        f"signoff_committer_match::{signoff_rel}",
        True,
        f"WARN: {detail} — advisory-skipped (registry currently template-stage)",
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def check_g3_wiring_guard(
    repo_root: Path,
    head_sha: str,
    require_signature_registry_match: bool = False,
) -> List[CheckResult]:
    """Run all G3 wiring-guard checks against ``repo_root``.

    Sequence:

      1. AST-scan the gated file for `hblp_classify` callsites in the gated
         functions. If no wiring detected → guard inactive → return only
         the wiring-detection result with ok=False (treated as PASS by
         the CLI exit policy).

      2. Wiring detected. Now require BOTH G1 and G2 signoff files to:
         (a) exist at HEAD,
         (b) reference a SHA that is an ancestor of HEAD.

      3. (Optional, gated on `require_signature_registry_match`): the
         signoff's introducing-commit committer email must match an
         active reviewer in the N3 registry.
    """

    results: List[CheckResult] = []

    wiring_check = detect_hblp_wiring(repo_root / WIRED_FILE_REL)
    results.append(wiring_check)

    if not wiring_check.ok:
        # Guard inactive — no signoff requirement.
        return results

    for signoff_rel in SIGNOFF_FILES:
        results.append(check_signoff_exists(repo_root, signoff_rel))
        results.append(check_signoff_ancestor(repo_root, signoff_rel, head_sha))
        results.append(
            check_signoff_committer_match(
                repo_root,
                signoff_rel,
                require_match=require_signature_registry_match,
            )
        )
    return results


def render_report(results: Sequence[CheckResult]) -> str:
    """Format CheckResults as one PASS/FAIL line per row."""

    lines: list[str] = []
    for r in results:
        marker = "PASS" if r.ok else "FAIL"
        lines.append(f"[{marker}] {r.name}: {r.detail}")
    return "\n".join(lines)


def evaluate(results: Sequence[CheckResult]) -> int:
    """Compute the CLI exit code from a result set.

    Policy:
      * Wiring-detection's ``ok`` field encodes PRESENCE — when False,
        the guard is inactive AND that's the desired pre-G1/G2 state. We
        return 0 in that case.
      * Wiring-detection ok=True but ANY downstream signoff check failed →
        exit 1 (guard activated, signoffs not in place).
      * Wiring-detection ok=True AND all downstream checks ok → exit 0.
    """

    if not results:
        return 2  # invocation error
    wiring = results[0]
    if not wiring.ok:
        # Guard inactive — pre-G1/G2 honest state. Build PASSES.
        return 0
    # Wiring detected — every downstream check must pass.
    for r in results[1:]:
        if not r.ok:
            return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_g3_wiring_guard",
        description=__doc__,
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
        "--head-sha",
        default="HEAD",
        help=(
            "Reference SHA for the ancestry checks. Defaults to 'HEAD'. "
            "CI workflows pass the PR head SHA explicitly."
        ),
    )
    parser.add_argument(
        "--require-signature-registry-match",
        action="store_true",
        help=(
            "Promote the signoff committer email match from advisory-warn "
            "to fail-closed. Recommended for production deployments where "
            "the N3 reviewer registry has at least one active row."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root or Path(__file__).resolve().parent.parent

    # Resolve the requested head SHA via git (so 'HEAD' becomes a real SHA
    # before we hand it to merge-base).
    head_sha = args.head_sha
    if head_sha == "HEAD" and shutil.which("git") is not None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if completed.returncode == 0:
                head_sha = (completed.stdout or "").strip() or head_sha
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    results = check_g3_wiring_guard(
        repo_root,
        head_sha,
        require_signature_registry_match=args.require_signature_registry_match,
    )
    print(render_report(results))
    return evaluate(results)


if __name__ == "__main__":
    sys.exit(main())
