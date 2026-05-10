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


# Detection states for ``detect_hblp_wiring``. Per codex MED-7, missing /
# unparseable target files are a HARD failure (NOT "guard inactive PASS"):
# a determined developer could rename or delete the gated file to dodge
# the AST scan. We surface ``scan_error`` as a distinct state and the
# orchestrator treats it as a downstream-blocking failure.
WIRING_ABSENT: str = "wiring_absent"
WIRING_PRESENT: str = "wiring_present"
SCAN_ERROR: str = "scan_error"


def _resolve_alias_targets(tree: ast.Module) -> set[str]:
    """Collect every module-scope name that points at ``hblp_classify``.

    Tracks two alias paths a determined developer could use to dodge the
    AST scan (codex HIGH-3):

    * ``ImportFrom`` aliases:
      ``from src.X import hblp_classify as classify_hblp``
      → ``classify_hblp`` in returned set.

    * Module-scope assignments:
      ``classify_hblp = hblp_classify``
      → ``classify_hblp`` in returned set.

    The canonical name ``hblp_classify`` is always included so callers
    that match against this set automatically catch the un-aliased form.
    """

    aliases: set[str] = {HBLP_CALL_NAME}
    for node in ast.iter_child_nodes(tree):
        # Path 1: ImportFrom aliases. The module path doesn't matter for
        # the detection — we only need the local binding name. Re-imports
        # like ``from .helpers import hblp_classify as fn`` and
        # ``from src.x.y import hblp_classify`` (no alias) both bind a
        # local name that calls the helper.
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == HBLP_CALL_NAME:
                    aliases.add(alias.asname or alias.name)
        # Path 2: module-scope ``alias = hblp_classify`` assignments.
        elif isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Name) and value.id in aliases:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            if (
                isinstance(value, ast.Name)
                and value.id in aliases
                and isinstance(node.target, ast.Name)
            ):
                aliases.add(node.target.id)
    return aliases


def detect_hblp_wiring(source_path: Path) -> CheckResult:
    """AST-scan ``source_path`` for `hblp_classify` calls inside gated functions.

    Returns a CheckResult whose ``name`` encodes the detection state:

      * ``WIRING_PRESENT`` when a gated function's body contains a call
        to ``hblp_classify`` (or a tracked alias). ``ok=True``.
      * ``WIRING_ABSENT`` when the file parses cleanly but contains no
        gated callsite. ``ok=False`` — this is the pre-G3 honest state
        and the orchestrator treats it as "guard inactive PASS".
      * ``SCAN_ERROR`` when the target file is missing OR has a syntax
        error. ``ok=False`` AND the orchestrator treats this as a hard
        downstream-blocking failure (codex MED-7) — a determined
        developer could rename / delete / break the file to side-step
        the scan, so we MUST fail closed.

    Per codex HIGH-3, the scanner now resolves module-scope import
    aliases (``from x import hblp_classify as fn``) and assignment
    aliases (``fn = hblp_classify``) before walking the function bodies,
    so call-name re-bindings can't dodge the gate.
    """

    if not source_path.is_file():
        # codex MED-7: missing target file is NOT "guard inactive PASS";
        # an attacker could rename / delete the file to side-step the
        # scan. Return SCAN_ERROR so the orchestrator fails closed.
        return CheckResult(
            SCAN_ERROR,
            False,
            f"target file not found: {source_path} — scan_error (hard failure)",
        )

    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        # codex MED-7: syntax errors are also a SCAN_ERROR (not absent).
        # If the file's broken we can't make a wiring claim and a
        # determined developer could intentionally introduce a syntax
        # error to dodge the scan.
        return CheckResult(
            SCAN_ERROR,
            False,
            f"failed to parse {source_path}: {exc} — scan_error (hard failure)",
        )

    # codex HIGH-3: collect every name the file binds to ``hblp_classify``
    # at module scope (canonical + import-aliases + assignment-aliases).
    alias_names = _resolve_alias_targets(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in GATED_FUNCTIONS:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    if isinstance(func, ast.Name) and func.id in alias_names:
                        alias_note = (
                            f" (alias of {HBLP_CALL_NAME!r})" if func.id != HBLP_CALL_NAME else ""
                        )
                        return CheckResult(
                            WIRING_PRESENT,
                            True,
                            f"{func.id!r}{alias_note} called inside {node.name!r} "
                            f"at line {inner.lineno}",
                        )
                    # Defensive: also catch attribute access (module.fn) so a
                    # ``import .helpers as h; h.hblp_classify(...)``-style
                    # indirection doesn't sidestep the scan.
                    if isinstance(func, ast.Attribute) and func.attr == HBLP_CALL_NAME:
                        return CheckResult(
                            WIRING_PRESENT,
                            True,
                            f"{HBLP_CALL_NAME!r} (attribute access) "
                            f"called inside {node.name!r} at line {inner.lineno}",
                        )

    return CheckResult(
        WIRING_ABSENT,
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


# Per codex MED-6 — exactly ONE ``commit:`` (or ``Branch / commit:``)
# field. The pattern accepts both ``commit:`` and ``Branch / commit:``
# variants used by G1/G2 templates, but NOT bare-backtick SHA tokens
# (which previously meant the first backtick-wrapped hex anywhere in the
# doc could be misread as the commit reference).
#
# Captures the value:
#   * single token in backticks: ``commit: `<sha>` ``
#   * single bare token:        ``commit: <sha>``
#   * field-bullet form:        ``- **commit:** `<sha>` ``
#
# We deliberately do NOT match other field names like ``S_prespec`` or
# ``experiment commit SHA`` here — the signoff schema for G3 calls for
# the canonical ``commit:`` / ``Branch / commit:`` field. Other fields
# may appear in the signoff but only this one carries authoritative
# commit-reference semantics for the wiring-guard check.
COMMIT_FIELD_PATTERN = re.compile(
    # Optional bullet (``- `` or ``* ``), optional bold (``**``), optional
    # ``Branch / `` prefix, the literal token ``commit``, optional inner
    # colon + bold close, the canonical colon + value.
    r"^\s*(?:[-*]\s+)?(?:\*\*)?(?:Branch\s*/\s*)?commit\s*:?\s*(?:\*\*)?\s*:?\s*"
    # Value: backtick-quoted token (any chars except backtick) OR
    # bare token (no whitespace / list separators / period).
    r"(?:`(?P<sha_q>[^`]+)`|(?P<sha_p>[^\s,;.]+))",
    re.IGNORECASE | re.MULTILINE,
)

# A commit SHA the policy accepts: 7-40 hex chars (per git's short-SHA
# convention; full-length is 40). codex MED-6 also requires we reject
# placeholders like "<sha>", "PLACEHOLDER", "TBD", and any token that
# isn't full-length hex. We surface that reason explicitly via
# ``ExtractCommitShaError``.
_FULL_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHORT_HEX_PATTERN = re.compile(r"^[0-9a-fA-F]{7,39}$")
_PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "<sha>",
    "<commit>",
    "placeholder",
    "tbd",
    "tba",
    "to be filled",
    "not yet filled",
    "n/a",
)


class ExtractCommitShaError(Exception):
    """Raised by ``extract_commit_sha`` for any non-conforming input.

    The error's ``args[0]`` carries a human-readable reason (used in
    ``CheckResult.detail``).
    """


def extract_commit_sha(doc_text: str, *, require_full_length: bool = True) -> str:
    """Pull the explicit ``commit:`` field SHA out of a signoff document.

    Per codex MED-6 the parser:

      * Matches exactly one explicit ``commit:`` (or ``Branch / commit:``)
        field — line-anchored, so backtick-hex tokens elsewhere in the
        document cannot satisfy the requirement.
      * RAISES ``ExtractCommitShaError`` on:
          - missing field,
          - duplicated field (multiple ``commit:`` lines),
          - placeholder values (``<sha>``, ``PLACEHOLDER``, ``TBD``, etc.),
          - non-hex tokens,
          - short hex (when ``require_full_length=True``, the default —
            full-length 40-char SHA is the policy for G3).

    The previous "first backtick SHA wins" behaviour is removed because
    it could be bypassed by inserting a backtick-hex token anywhere in
    the document (codex MED-6).
    """

    matches = list(COMMIT_FIELD_PATTERN.finditer(doc_text))
    if not matches:
        raise ExtractCommitShaError(
            "no `commit:` field found — signoff template must carry exactly "
            "one explicit `commit: <sha>` line"
        )
    if len(matches) > 1:
        raise ExtractCommitShaError(
            f"multiple `commit:` fields found ({len(matches)} matches); "
            "signoff template must carry exactly one"
        )

    sha_q = matches[0].group("sha_q")
    sha_p = matches[0].group("sha_p")
    raw = (sha_q or sha_p or "").strip().strip("`'\"")
    if not raw:
        raise ExtractCommitShaError("`commit:` field is empty")

    lower = raw.lower()
    for placeholder in _PLACEHOLDER_TOKENS:
        if placeholder in lower:
            raise ExtractCommitShaError(
                f"`commit:` field is a placeholder ({raw!r}); "
                f"signoff must reference a real commit SHA"
            )

    if require_full_length:
        if not _FULL_HEX_PATTERN.match(raw):
            if _SHORT_HEX_PATTERN.match(raw):
                raise ExtractCommitShaError(
                    f"`commit:` field is short SHA ({raw!r}); policy requires full 40-char hex"
                )
            raise ExtractCommitShaError(f"`commit:` field is not 40-char hex ({raw!r})")
    else:
        # Permissive variant for legacy callers / tests.
        if not _FULL_HEX_PATTERN.match(raw) and not _SHORT_HEX_PATTERN.match(raw):
            raise ExtractCommitShaError(f"`commit:` field is not hex ({raw!r})")
    return raw


def _extract_commit_sha_or_none(doc_text: str) -> Optional[str]:
    """Backward-compat shim that returns ``None`` instead of raising.

    Used by callers that want optional SHA extraction (e.g. the
    advisory-warn paths). Production callers SHOULD use
    ``extract_commit_sha`` directly so the failure reason surfaces in
    the report.
    """

    try:
        return extract_commit_sha(doc_text, require_full_length=False)
    except ExtractCommitShaError:
        return None


def _git_show_blob(repo_root: Path, sha: str, path: str) -> Optional[str]:
    """Return the blob contents of ``path`` at ``sha`` (None if absent)."""

    if shutil.which("git") is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{sha}:{path}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def check_signoff_exists_in_base_ref(
    repo_root: Path,
    signoff_rel: str,
    base_sha: str,
) -> CheckResult:
    """The signoff file MUST exist in the BASE ref (codex HIGH-2).

    A determined developer could otherwise merge fake / early G1+G2
    signoffs into the same G3 PR, making the signoff blobs ancestors of
    PR HEAD. Requiring base-ref presence prevents that bypass:

      * PR HEAD has the wiring change (G3),
      * BASE ref (origin/main HEAD at PR-open time) MUST already carry
        the signoff files — i.e. G1/G2 must have ALREADY merged to main
        before the G3 PR can pass the guard.
    """

    if shutil.which("git") is None:
        return CheckResult(
            f"signoff_exists_base::{signoff_rel}",
            False,
            "git binary not on PATH — cannot verify base-ref existence",
        )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{base_sha}:{signoff_rel}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return CheckResult(
            f"signoff_exists_base::{signoff_rel}",
            False,
            f"git cat-file failed: {exc}",
        )
    if completed.returncode == 0:
        return CheckResult(
            f"signoff_exists_base::{signoff_rel}",
            True,
            f"signoff present in base ref ({base_sha[:12]})",
        )
    return CheckResult(
        f"signoff_exists_base::{signoff_rel}",
        False,
        f"signoff MISSING in base ref ({base_sha[:12]}) — codex HIGH-2: "
        f"G1/G2 must merge to main BEFORE G3 wiring lands",
    )


def check_signoff_ancestor(
    repo_root: Path,
    signoff_rel: str,
    head_sha: str,
    *,
    base_sha: Optional[str] = None,
) -> CheckResult:
    """The SHA referenced in the signoff MUST be an ancestor.

    Per codex HIGH-2:
      * When ``base_sha`` is provided, the signoff's ``commit:`` field is
        read from the BASE-ref copy (NOT the PR-checkout copy) so a
        malicious PR cannot rewrite the signoff to reference an
        attacker-controlled SHA.
      * Ancestry is checked against ``head_sha`` (which the orchestrator
        passes as ``base_sha or head_sha``) so the bypass-via-merge-magic
        path (merging G1+G2 into the same G3 PR) cannot satisfy the
        check.

    Per codex MED-6:
      * The ``commit:`` field is parsed via ``extract_commit_sha``, which
        rejects missing / duplicated / placeholder / non-hex / short SHA
        tokens. A failed extraction surfaces with the rejection reason.
    """

    if shutil.which("git") is None:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            "git binary not on PATH — cannot verify ancestry",
        )

    # codex HIGH-2: read the signoff body from the BASE ref so the PR
    # cannot rewrite the commit field. Fall back to the on-disk copy
    # when no base_sha was provided.
    if base_sha is not None:
        body = _git_show_blob(repo_root, base_sha, signoff_rel)
        if body is None:
            return CheckResult(
                f"signoff_ancestor::{signoff_rel}",
                False,
                f"signoff missing in base ref ({base_sha[:12]}) — cannot extract SHA",
            )
    else:
        full = repo_root / signoff_rel
        if not full.is_file():
            return CheckResult(
                f"signoff_ancestor::{signoff_rel}",
                False,
                f"signoff file missing: {full} — cannot extract SHA",
            )
        body = full.read_text(encoding="utf-8")

    try:
        sha = extract_commit_sha(body)
    except ExtractCommitShaError as exc:
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"could not extract commit SHA from {signoff_rel}: {exc}",
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
        ref_label = "BASE_SHA" if base_sha is not None else "HEAD"
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            True,
            f"signoff SHA {sha[:12]} is ancestor of {ref_label} {head_sha[:12]}",
        )
    if completed.returncode == 1:
        ref_label = "BASE_SHA" if base_sha is not None else "HEAD"
        return CheckResult(
            f"signoff_ancestor::{signoff_rel}",
            False,
            f"signoff SHA {sha[:12]} is NOT an ancestor of {ref_label} {head_sha[:12]}",
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
    *,
    base_sha: Optional[str] = None,
) -> CheckResult:
    """The signoff's introducing-commit committer email must be in registry.

    Behavior (codex HIGH-1):
      * ``require_match=False`` (advisory): returns PASS with WARN
        annotation noting whether a match was found.
      * ``require_match=True`` (fail-closed, the production CI mode):
        the check FAILS if:
          - the registry is missing OR has zero active rows,
          - the committer email cannot be resolved,
          - the committer email is NOT in the active registry.
        These are the codex HIGH-1 failure-closed criteria; the
        ``--require-signature-registry-match`` flag promotes the check
        from advisory-warn to hard-fail.

    Per codex HIGH-2: when ``base_sha`` is provided, the introducing-
    commit lookup runs against ``base_sha`` (NOT against the PR-tip)
    so a PR-introduced signoff can't auto-register a malicious
    committer. The lookup also reads the signoff blob from the base ref
    to confirm the signoff exists there (mirror of
    ``check_signoff_exists_in_base_ref``).
    """

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

    # codex HIGH-2: when a base_sha is provided, look up the signoff in
    # the base ref. We do NOT fall back to PR-checkout when require_match
    # is True; in fail-closed mode the signoff MUST exist in base.
    if base_sha is not None:
        # Confirm the signoff exists in base.
        try:
            cat = subprocess.run(
                ["git", "-C", str(repo_root), "cat-file", "-e", f"{base_sha}:{signoff_rel}"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                not require_match,
                f"git cat-file failed: {exc}",
            )
        if cat.returncode != 0:
            if require_match:
                return CheckResult(
                    f"signoff_committer_match::{signoff_rel}",
                    False,
                    f"signoff missing in base ref ({base_sha[:12]})",
                )
            return CheckResult(
                f"signoff_committer_match::{signoff_rel}",
                True,
                f"WARN: signoff missing in base ref ({base_sha[:12]}) — "
                f"committer match advisory-skipped",
            )
        # log range bounded by base_sha so we only consider commits up to
        # base; this rejects PR-introduced "register myself" attacks.
        log_range = base_sha
    else:
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
        log_range = "HEAD"

    cmd = [
        "git",
        "-C",
        str(repo_root),
        "log",
        log_range,
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
    # codex HIGH-1: in fail-closed mode the registry MUST be non-empty.
    # An empty registry under --require-signature-registry-match means
    # the gate is a no-op (no email can match the empty set), so we
    # surface that as the failure reason.
    if require_match and not registry_emails:
        return CheckResult(
            f"signoff_committer_match::{signoff_rel}",
            False,
            "active reviewer registry is EMPTY — codex HIGH-1: fail-closed "
            "policy requires at least one active row in "
            f"{REVIEWER_REGISTRY_REL}",
        )
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
    base_sha: Optional[str] = None,
) -> List[CheckResult]:
    """Run all G3 wiring-guard checks against ``repo_root``.

    Sequence:

      1. AST-scan the gated file for `hblp_classify` callsites in the gated
         functions. Three outcomes (codex MED-7):

         * ``WIRING_PRESENT`` (ok=True) — guard activates; downstream
           signoff checks must all pass.
         * ``WIRING_ABSENT`` (ok=False) — pre-G3 honest state; guard
           inactive; build PASSES.
         * ``SCAN_ERROR`` (ok=False) — file missing / unparseable;
           HARD failure (orchestrator surfaces this distinctly so
           ``evaluate()`` returns 1, NOT 0).

      2. Wiring detected. Require BOTH G1 and G2 signoff files to:
         (a) exist at HEAD AND in the BASE ref (or merge-base) so
             a determined developer can't merge fake / early signoff
             files into the same G3 PR (codex HIGH-2),
         (b) extract a single explicit ``commit:`` field SHA from the
             BASE-ref copy (codex MED-6),
         (c) the SHA is an ancestor of ``base_sha`` (NOT just HEAD —
             codex HIGH-2: PR HEAD ancestry is bypassable via merge
             magic if the signoff was merged into the same PR).

      3. (Optional, gated on `require_signature_registry_match`): the
         signoff's introducing-commit committer email must match an
         active reviewer in the N3 registry. When ``base_sha`` is
         provided, the introducing-commit lookup runs against the BASE
         ref so PR-introduced signoffs don't auto-register a malicious
         committer.
    """

    results: List[CheckResult] = []

    wiring_check = detect_hblp_wiring(repo_root / WIRED_FILE_REL)
    results.append(wiring_check)

    if wiring_check.name == WIRING_ABSENT:
        # Guard inactive — no signoff requirement.
        return results

    if wiring_check.name == SCAN_ERROR:
        # codex MED-7: file missing OR syntax error → hard failure.
        # We do NOT run downstream signoff checks because we can't
        # claim wiring presence; but we also do NOT return early-PASS
        # because the orchestrator treats SCAN_ERROR as exit 1.
        return results

    # Wiring present — enforce signoff existence + ancestry against both
    # HEAD and BASE_SHA (codex HIGH-2).
    ancestor_target = base_sha or head_sha
    for signoff_rel in SIGNOFF_FILES:
        results.append(check_signoff_exists(repo_root, signoff_rel))
        # codex HIGH-2: signoff must exist on base ref / merge-base too,
        # not only on PR HEAD. A base-ref existence failure prevents the
        # "merge G1+G2 into the same PR" bypass.
        if base_sha is not None:
            results.append(check_signoff_exists_in_base_ref(repo_root, signoff_rel, base_sha))
        results.append(
            check_signoff_ancestor(
                repo_root,
                signoff_rel,
                ancestor_target,
                base_sha=base_sha,
            )
        )
        results.append(
            check_signoff_committer_match(
                repo_root,
                signoff_rel,
                require_match=require_signature_registry_match,
                base_sha=base_sha,
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

    Policy (codex MED-7 split):
      * Wiring-detection's ``name`` field carries the detection state:
        - ``WIRING_ABSENT`` (ok=False) — pre-G3 honest state. Build
          PASSES (exit 0).
        - ``SCAN_ERROR`` (ok=False) — file missing / syntax error.
          HARD failure (exit 1) — a determined developer must NOT be
          able to dodge the scan by deleting / breaking the gated file.
        - ``WIRING_PRESENT`` (ok=True) — every downstream signoff
          check must pass for exit 0.
    """

    if not results:
        return 2  # invocation error
    wiring = results[0]
    if wiring.name == WIRING_ABSENT:
        # Guard inactive — pre-G1/G2 honest state. Build PASSES.
        return 0
    if wiring.name == SCAN_ERROR:
        # codex MED-7: hard failure on missing / unparseable file.
        return 1
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
        "--base-sha",
        default=None,
        help=(
            "Base ref SHA (e.g. origin/main HEAD at PR-open time). When "
            "provided (codex HIGH-2): "
            "(a) the signoff existence check is run against the BASE ref, "
            "(b) the signoff `commit:` field is parsed from the BASE-ref "
            "    blob (NOT the PR-checkout copy), "
            "(c) the SHA must be an ancestor of BASE_SHA (NOT just HEAD), "
            "    closing the merge-G1+G2-into-the-same-G3-PR bypass."
        ),
    )
    parser.add_argument(
        "--require-signature-registry-match",
        action="store_true",
        help=(
            "Promote the signoff committer email match from advisory-warn "
            "to fail-closed (codex HIGH-1). In fail-closed mode the "
            "active reviewer registry MUST be non-empty AND the "
            "signoff-introducing-commit committer email MUST appear in "
            "an active registry row. Recommended for ALL production CI "
            "deployments — the workflow at "
            "`.github/workflows/g3_wiring_guard.yml` passes this flag."
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

    base_sha = args.base_sha
    if base_sha is not None and shutil.which("git") is not None:
        # Resolve the base SHA via rev-parse so a symbolic ref like
        # ``origin/main`` becomes a real SHA before we hand it to
        # ``cat-file``.
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", base_sha],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if completed.returncode == 0:
                base_sha = (completed.stdout or "").strip() or base_sha
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    results = check_g3_wiring_guard(
        repo_root,
        head_sha,
        require_signature_registry_match=args.require_signature_registry_match,
        base_sha=base_sha,
    )
    print(render_report(results))
    return evaluate(results)


if __name__ == "__main__":
    sys.exit(main())
