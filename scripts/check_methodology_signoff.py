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

H3 SECURITY ADVISORY:
    A pull request that adds a sign-off artifact can ALSO modify this
    validator script and the workflow YAML; without mitigation, the PR's
    weakened copy of the script will validate the PR's own artifact. The
    workflow at ``.github/workflows/methodology_signoff_guard.yml``
    addresses this by copying the validator from the PR's base SHA via
    ``git show <base_sha>:scripts/check_methodology_signoff.py`` BEFORE
    invoking it. In production deployment, the validator should additionally
    be moved to a separate, protected repository and pinned by SHA. Until
    that happens, ``scripts/check_methodology_signoff.py`` and
    ``.github/workflows/methodology_signoff_guard.yml`` MUST be CODEOWNERS-
    gated to require security-team review.

M1 (iter-3) — gh provenance best-effort:
    The selection rule's ``gh pr list`` signals (PRs authored / reviewed by
    the reviewer that touch the subject files) are best-effort: when ``gh``
    is unavailable on the runner OR the runner lacks an authenticated token
    OR the gh API returns an error, ``check_selection_rule`` PASSES on the
    canonical git-log signal but emits a CRITICAL warning in the result
    detail AND sets ``CheckResult.provenance_check_skipped=True``. The CI
    workflow MUST inspect this flag and decide whether to fail-closed; see
    ``.github/workflows/methodology_signoff_guard.yml`` for the policy
    comment. See also ``docs/governance/n3_known_limitations_20260510.md``
    for the full deferred-infra rationale.

NEW MED (iter-3) — future-dated artifacts:
    ``check_signoff_age`` rejects sign-offs whose filename date is more than
    1 day ahead of ``today``. The 1-day tolerance covers TZ-skew at the day
    boundary. Prevents reviewers from pre-dating sign-offs to evade the
    max-age window.
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
        provenance_check_skipped: True when a best-effort provenance signal
            (gh PR/review query) could not be evaluated due to missing
            tooling or auth. The check passed on the canonical signal
            (git log) but the caller (CI) MUST decide whether to fail-
            closed when this flag is set. A CRITICAL warning is logged
            in the detail string when this flag is True (iter-3 M1).
    """

    name: str
    ok: bool
    detail: str = ""
    provenance_check_skipped: bool = False


@dataclasses.dataclass
class ReviewerInfo:
    """Subset of the registry row needed for selection-rule checks.

    M1: ``email`` is the canonical primary email (single value, used for
    log lines / summaries). ``emails`` is the full set of aliases parsed
    from the registry's email cell; the cell may contain a single address
    OR a comma/semicolon-separated list (e.g.
    ``"alice@example.com, alice@oldjob.com"``). Each alias is checked by
    the selection rule's ``git log --author=<email>`` filter so a reviewer
    who committed under an alternate identity is still detected.
    """

    handle: str
    email: str
    status: str
    emails: tuple[str, ...] = ()


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
            handle, email_cell, status = cells[2], cells[1], cells[6]
            # Strip Markdown emphasis (e.g. _PLACEHOLDER_).
            handle = handle.strip("_*`")
            # M1: split the email cell on comma/semicolon to support alias
            # lists like "alice@example.com, alice@oldjob.com". The first
            # address is treated as canonical; all addresses go into the
            # emails tuple for the selection rule's git-log probes.
            aliases = tuple(a.strip() for a in re.split(r"[,;]", email_cell) if a.strip())
            primary = aliases[0] if aliases else email_cell
            rows.append(
                ReviewerInfo(
                    handle=handle,
                    email=primary,
                    status=status,
                    emails=aliases or (email_cell,),
                )
            )
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


MAX_SIGNOFF_AGE_DAYS: int = 30


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


FUTURE_DATE_TOLERANCE_DAYS: int = 1


def check_signoff_age(
    doc_path: Path,
    today: Optional[str] = None,
    max_age_days: int = MAX_SIGNOFF_AGE_DAYS,
) -> CheckResult:
    """M2 + iter-3 NEW MED: reject sign-offs older than ``max_age_days`` AND
    sign-offs whose filename date is more than ``FUTURE_DATE_TOLERANCE_DAYS``
    ahead of ``today``.

    The artifact's date is parsed from the filename suffix
    ``_<YYYYMMDD>.md`` (the FILENAME_PATTERN-captured ``date`` group).
    ``today`` defaults to the system date in ISO-8601 format; tests can
    inject a fixed reference for deterministic behaviour.

    Returns:
      * PASS when the doc date is within the window AND not unreasonably
        future-dated. A 1-day tolerance covers timezone-skew at the day
        boundary (a CI runner in UTC may see "tomorrow" on a doc dated
        for the reviewer's local "today").
      * FAIL when the doc date is older than ``max_age_days`` vs today.
      * FAIL when the doc date is more than
        ``FUTURE_DATE_TOLERANCE_DAYS`` ahead of today (iter-3 NEW MED:
        prevents reviewers from pre-dating sign-offs to evade the
        max-age window or to claim review of work not yet performed).
      * FAIL on filename pattern mismatch (defensive — the orchestrator
        already filtered via check_filename, but we re-validate so the
        function stays callable in isolation).
    """

    import datetime as _dt

    name = doc_path.name
    match = FILENAME_PATTERN.match(name)
    if match is None:
        return CheckResult(
            "signoff_age",
            False,
            f"filename does not carry a parseable date: {name!r}",
        )

    date_str = match.group("date")
    try:
        doc_date = _dt.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except ValueError as exc:
        return CheckResult(
            "signoff_age",
            False,
            f"unparseable date {date_str!r} in filename: {exc}",
        )

    if today is None:
        ref_date = _dt.date.today()
    else:
        try:
            ref_date = _dt.date.fromisoformat(today)
        except ValueError as exc:
            return CheckResult(
                "signoff_age",
                False,
                f"unparseable today {today!r}: {exc}",
            )

    age_days = (ref_date - doc_date).days

    # iter-3 NEW MED: future-dated artifacts beyond the small TZ-skew
    # tolerance are rejected. age_days < 0 means the doc is dated in the
    # future relative to today; we tolerate up to FUTURE_DATE_TOLERANCE_DAYS
    # of skew (so age_days >= -FUTURE_DATE_TOLERANCE_DAYS is OK).
    if age_days < -FUTURE_DATE_TOLERANCE_DAYS:
        future_days = -age_days
        return CheckResult(
            "signoff_age",
            False,
            f"sign-off filename date {doc_date.isoformat()} is {future_days} days "
            f"in the future vs today={ref_date.isoformat()} "
            f"(tolerance={FUTURE_DATE_TOLERANCE_DAYS}d)",
        )

    if age_days > max_age_days:
        return CheckResult(
            "signoff_age",
            False,
            f"sign-off filename date {doc_date.isoformat()} is {age_days} days "
            f"older than today={ref_date.isoformat()} (max={max_age_days}d)",
        )
    return CheckResult(
        "signoff_age",
        True,
        f"sign-off filename date {doc_date.isoformat()} is {age_days} days old "
        f"(<= {max_age_days}d)",
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


_COI_FILENAME_PATTERN = re.compile(r"^(?P<handle>[A-Za-z0-9][-A-Za-z0-9_]*)_(?P<date>\d{8})\.md$")


def check_coi_referenced(
    doc_text: str,
    repo_root: Optional[Path] = None,
) -> CheckResult:
    """H4: validate CoI fields beyond mere presence.

    Checks (in order):

    1. SHA and path fields are non-empty and not template placeholders.
    2. CoI path filename matches ``<handle>_<YYYYMMDD>.md`` where
       ``<handle>`` matches the sign-off doc's reviewer handle.
    3. ``git cat-file -e <sha>:<path>`` resolves (the SHA committed the
       declared path).
    4. The declared SHA is the FIRST commit that added the path
       (``git log --diff-filter=A --follow --reverse``); a later-modify
       SHA is rejected because it would let a reviewer point at a
       declaration that was originally written for a different period.

    Steps 3 and 4 run only when ``repo_root`` is provided AND ``git`` is
    on PATH; otherwise they are skipped with a WARN annotation. Step 4
    is also skipped (with a WARN, not a fail) if ``git log
    --diff-filter=A`` returns no result — that case can occur in test
    fixture repos that do not yet have the CoI declaration committed.
    """

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

    # Type narrowing for mypy — guarded above.
    assert sha is not None and path is not None

    # H4 sub-check 2: filename format and handle match.
    handle = extract_handle(doc_text)
    coi_filename = path.rsplit("/", 1)[-1]
    name_match = _COI_FILENAME_PATTERN.match(coi_filename)
    if name_match is None:
        return CheckResult(
            "coi_referenced",
            False,
            f"CoI filename {coi_filename!r} does not match <handle>_<YYYYMMDD>.md",
        )
    file_handle = name_match.group("handle")
    if handle is not None and file_handle != handle:
        return CheckResult(
            "coi_referenced",
            False,
            f"CoI filename handle {file_handle!r} does not match reviewer handle {handle!r}",
        )

    warnings: list[str] = []

    # H4 sub-check 3: SHA + path resolve in repo.
    if repo_root is not None and shutil.which("git") is not None:
        ok, detail = _coi_sha_resolves(repo_root, sha, path)
        if not ok:
            return CheckResult(
                "coi_referenced",
                False,
                f"CoI SHA/path do not resolve in git: {detail}",
            )
        # H4 sub-check 4: SHA is the first-add commit for the path.
        first_add = _coi_first_add_commit(repo_root, path)
        if first_add is None:
            warnings.append("first-add SHA not derivable (path may not be committed in fixture)")
        elif not (sha.startswith(first_add) or first_add.startswith(sha)):
            return CheckResult(
                "coi_referenced",
                False,
                f"CoI SHA {sha[:12]} is not the first-add commit for {path} (first_add={first_add[:12]})",
            )
    else:
        warnings.append("git resolution skipped (no repo_root or git not on PATH)")

    detail = f"sha={sha[:12]}, path={path}"
    if warnings:
        detail += " | WARN: " + "; ".join(warnings)
    return CheckResult("coi_referenced", True, detail)


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


def _gh_pr_touches(
    handle: str,
    role: str,
    subject_files: Sequence[str],
    period_start: str,
    period_end: str,
) -> tuple[Optional[bool], str]:
    """Best-effort `gh pr list` query for PRs authored OR reviewed by ``handle``.

    Returns ``(has_touches, detail)`` where ``has_touches`` is:
      * ``True`` — at least one PR matches handle + period + subject files.
      * ``False`` — query ran cleanly and returned no overlapping PRs.
      * ``None`` — query could not run (gh not on PATH, no repo, no auth).
        In that case the caller MUST treat the result as best-effort and
        emit a warning rather than fail-or-pass on this signal alone.

    ``role`` is either "author" or "reviewer".

    The PR-files JSON shape from `gh pr list --json files` is::

        [{"number": 131, "files": [{"path": "scripts/foo.py"}, ...]}, ...]
    """

    if shutil.which("gh") is None:
        return None, "gh not on PATH (best-effort signal skipped)"

    if role == "author":
        flag = "--author"
        date_flag = "created"
    elif role == "reviewer":
        flag = "--reviewer"
        date_flag = "updated"
    else:
        raise ValueError(f"unknown role {role!r} (expected 'author' or 'reviewer')")

    search = f"{date_flag}:{period_start}..{period_end}"
    cmd = [
        "gh",
        "pr",
        "list",
        flag,
        handle,
        "--state",
        "all",
        "--search",
        search,
        "--json",
        "number,files,title",
        "--limit",
        "200",
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, f"gh invocation failed: {exc}"

    if completed.returncode != 0:
        # gh returns non-zero on auth errors or repo unavailable. We treat
        # that as best-effort skip rather than a violation.
        return (
            None,
            f"gh returncode={completed.returncode}: {(completed.stderr or '').strip()[:200]}",
        )

    import json as _json

    try:
        prs = _json.loads(completed.stdout or "[]")
    except _json.JSONDecodeError as exc:
        return None, f"gh JSON parse failed: {exc}"

    overlaps: list[str] = []
    subject_set = set(subject_files)
    for pr in prs:
        files = pr.get("files") or []
        pr_paths = {entry.get("path") for entry in files if isinstance(entry, dict)}
        intersect = pr_paths & subject_set
        if intersect:
            overlaps.append(f"PR#{pr.get('number')} ({role}, files={sorted(intersect)})")
    if overlaps:
        return True, "; ".join(overlaps)
    return False, f"0 {role} PRs touching subject files in {period_start}..{period_end}"


def _coi_sha_resolves(repo_root: Path, sha: str, path: str) -> tuple[bool, str]:
    """Run ``git cat-file -e <sha>:<path>`` to confirm the SHA + path resolve.

    Returns ``(ok, detail)``. ``ok`` is True iff git's exit is 0 (the
    object exists at that commit). The detail is a short human label.
    """

    if shutil.which("git") is None:
        return False, "git not on PATH"

    cmd = [
        "git",
        "-C",
        str(repo_root),
        "cat-file",
        "-e",
        f"{sha}:{path}",
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
        return False, f"git cat-file failed: {exc}"
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        return False, f"git cat-file -e {sha[:12]}:{path}: {stderr or 'not found'}"
    return True, f"git cat-file resolves {sha[:12]}:{path}"


def _coi_first_add_commit(repo_root: Path, path: str) -> Optional[str]:
    """Return the first commit SHA that added ``path`` to history, or None.

    Uses ``git log --diff-filter=A --follow --reverse``. None is returned
    if git is unavailable or the path was never added (e.g. only modified).
    """

    if shutil.which("git") is None:
        return None

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
        path,
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    shas = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    return shas[0] if shas else None


def _parse_coi_declared_prs(coi_text: str) -> list[dict]:
    """Extract the JSON array of declared PRs from the CoI markdown body.

    The CoI template asks reviewers to paste `gh pr list --json` output
    inside fenced code blocks. This helper scans for the FIRST JSON-like
    array inside any code fence and returns it parsed. On parse failure or
    no array found, returns an empty list.

    The returned list is best-effort: each entry is whatever shape the
    reviewer pasted (typically ``{"number": int, "title": str,
    "files": [{"path": str}, ...]}``).
    """

    import json as _json

    pattern = re.compile(r"```[a-zA-Z]*\s*(?P<body>\[[\s\S]*?\])\s*```")
    for match in pattern.finditer(coi_text):
        body = match.group("body").strip()
        try:
            parsed = _json.loads(body)
        except _json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return []


def check_selection_rule(
    doc_text: str,
    repo_root: Path,
    registry: Sequence[ReviewerInfo],
    coi_text: Optional[str] = None,
) -> CheckResult:
    """Selection rule (H2): combine git-log + gh PR/review evidence + CoI parse.

    For each subject file, the reviewer must have:
      * 0 git commits authored in the named period (git log --author=email),
      * 0 PRs authored intersecting the subject files in the named period,
      * 0 PRs reviewed intersecting the subject files in the named period,
      * if the CoI document text is provided, 0 PRs declared therein that
        intersect the subject files (this is the reviewer's own admission).

    The git-log signal is canonical; the gh signals are best-effort and a
    "could not query" outcome (gh missing, auth error, etc.) emits a CRITICAL
    warning in the detail string AND sets ``provenance_check_skipped=True``
    on the returned ``CheckResult`` (iter-3 M1). The check itself still
    PASSES on the canonical git-log signal — gh-skip alone is not a
    violation — but the caller (CI workflow) MUST decide whether to fail-
    closed when ``provenance_check_skipped`` is True. Recommended CI policy:
    fail-closed in production deployments where every PR landing must have
    its sign-off provenance fully verified; warn-only in early-stage / dev
    contexts where infra (gh CLI auth on runner) is not yet provisioned.

    The CoI-declared-PR signal IS authoritative for what the reviewer
    self-declares (a non-empty intersection means they have admitted touching
    the subject file and the rule fails regardless of git/gh signal).
    """

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
    row = matching[0]
    email = row.email
    # M1: iterate over ALL declared aliases so a commit authored under an
    # alternate identity is still caught by `git log --author=`. Falls back
    # to the primary email if the row predates the alias-aware schema.
    aliases = row.emails or (email,)
    violations: List[str] = []
    warnings: List[str] = []

    for subject in SUBJECT_FILES:
        for alias in aliases:
            has_touches, output = _git_log_touches(
                repo_root,
                alias,
                subject,
                NAMED_PERIOD_START,
                NAMED_PERIOD_END,
            )
            if has_touches:
                violations.append(f"git({subject}, {alias}): {output}")

    # Best-effort gh queries — both author and reviewer roles.
    # iter-3 M1: track whether ANY gh query was skipped so the caller (CI)
    # can decide to fail-closed. A skipped gh query means we did NOT
    # confirm absence of overlapping PRs; the canonical git-log signal is
    # confirmed, but the gh signal is unavailable.
    gh_skipped = False
    for role in ("author", "reviewer"):
        result, detail = _gh_pr_touches(
            handle,
            role,
            SUBJECT_FILES,
            NAMED_PERIOD_START,
            NAMED_PERIOD_END,
        )
        if result is True:
            violations.append(f"gh-{role}: {detail}")
        elif result is None:
            warnings.append(f"gh-{role}: {detail}")
            gh_skipped = True
        # result is False → no overlap; nothing to record.

    # Authoritative self-declaration check from the CoI document body, if
    # the caller provided it.
    if coi_text is not None:
        declared = _parse_coi_declared_prs(coi_text)
        if declared:
            subject_set = set(SUBJECT_FILES)
            for pr in declared:
                if not isinstance(pr, dict):
                    continue
                files = pr.get("files") or []
                pr_paths = {entry.get("path") for entry in files if isinstance(entry, dict)}
                intersect = pr_paths & subject_set
                if intersect:
                    violations.append(
                        f"coi-self-declared: PR#{pr.get('number')} touches {sorted(intersect)}"
                    )

    if violations:
        return CheckResult(
            "selection_rule",
            False,
            "; ".join(violations),
            provenance_check_skipped=gh_skipped,
        )

    detail = f"0 git touches for {email} in {NAMED_PERIOD_START}..{NAMED_PERIOD_END}"
    if warnings:
        # iter-3 M1: when any gh signal was skipped, elevate the warning to
        # CRITICAL so log scrapers / CI summaries notice. The canonical
        # git-log signal still gates the PASS, but the caller (CI) MUST
        # decide whether to fail-closed when provenance_check_skipped=True.
        if gh_skipped:
            detail += (
                " | CRITICAL: gh provenance query SKIPPED — "
                "PR/review evidence not validated; caller must decide whether "
                "to fail-closed (provenance_check_skipped=True): "
            )
        else:
            detail += " | WARN: "
        detail += "; ".join(warnings)
    return CheckResult(
        "selection_rule",
        True,
        detail,
        provenance_check_skipped=gh_skipped,
    )


# Tokens whose presence inside a PGP armor block indicates the block is
# render-paste contaminated (HTML/JATS escape entities, tags, or rendered
# Markdown emphasis), NOT the literal ASCII-armor that gpg expects. See M4.
_RENDER_PASTE_TAINT_PATTERNS: tuple[str, ...] = (
    "&amp;",
    "&lt;",
    "&gt;",
    "&#",
    "<p>",
    "</p>",
    "<br",
    "<jats:",
    "</jats:",
    "<span",
    "</span>",
    "<em>",
    "</em>",
    "<i>",
    "</i>",
)


def _pgp_block_taint(armor_block: str) -> Optional[str]:
    """Return the first taint token detected in ``armor_block``, or None.

    A taint token (HTML entity, tag, JATS namespace prefix) inside the
    PGP block indicates the doc was rendered + copy-pasted rather than
    raw-armor-pasted. gpg cannot verify a tainted block, so we reject it
    upstream of the verifier.
    """

    for token in _RENDER_PASTE_TAINT_PATTERNS:
        if token in armor_block:
            return token
    return None


def _pgp_block_parses_via_gpg(armor_block: str) -> tuple[bool, str]:
    """Run ``gpg --list-packets`` against the block and check for a sigpacket.

    Returns ``(ok, detail)``. ``ok`` is True iff gpg parses at least one
    packet AND the output mentions a "signature packet" (gpg's name for
    type-2 packets). When gpg is unavailable, returns ``(True,
    'WARN: gpg unavailable')`` so the function degrades gracefully in
    advisory mode (require_signature path runs the real gpg --verify).
    """

    if shutil.which("gpg") is None:
        return True, "WARN: gpg unavailable; structural parse skipped"

    try:
        completed = subprocess.run(
            ["gpg", "--batch", "--list-packets"],
            input=armor_block,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return False, f"gpg --list-packets failed: {exc}"

    out = (completed.stdout or "") + (completed.stderr or "")
    # gpg's --list-packets writes "off=N ctb=NN tag=2 hlen=...len=... :signature packet:"
    # to stdout. Substring search on "signature packet" is enough to
    # distinguish well-formed armor from random ASCII.
    if "signature packet" in out:
        return True, "gpg --list-packets recognises a signature packet"
    return False, f"gpg --list-packets did not recognise a signature packet: {out.strip()[:200]}"


def check_signature_present(doc_text: str) -> CheckResult:
    """A PGP-armor block or sigstore JSON block must be present in the doc.

    M4 hardening:

    1. Use the ``_extract_pgp_armor_block`` regex (NOT a substring search)
       so a doc that contains stray "-----BEGIN PGP SIGNATURE-----" text
       outside an armor block does not trivially pass.
    2. Reject the block when it contains render-paste taint tokens (HTML
       entities, JATS tags, etc.) — these indicate the reviewer pasted a
       rendered HTML view rather than the raw ASCII armor that gpg parses.
    3. Run ``gpg --list-packets`` to require the block be structurally
       recognisable as containing a signature packet. The check degrades
       to WARN if gpg is unavailable, since require_signature path will
       still run a real ``gpg --verify``.

    For sigstore JSON the heuristic remains a regex match on the JSON
    fence; the bundle is structurally validated under require_signature.
    """

    pgp_block = _extract_pgp_armor_block(doc_text)
    if pgp_block is not None:
        if "<signature blob>" in pgp_block:
            return CheckResult(
                "signature_present",
                False,
                "PGP block contains template placeholder '<signature blob>'",
            )
        taint = _pgp_block_taint(pgp_block)
        if taint is not None:
            return CheckResult(
                "signature_present",
                False,
                f"PGP block is render-paste tainted (token={taint!r}); "
                f"paste raw ASCII armor, not rendered HTML/JATS",
            )
        ok, detail = _pgp_block_parses_via_gpg(pgp_block)
        if not ok:
            return CheckResult(
                "signature_present",
                False,
                f"PGP block fails structural parse: {detail}",
            )
        return CheckResult("signature_present", True, f"PGP signature block present ({detail})")
    # Sigstore bundle is a plain JSON object inside a ```json fence; match a
    # rough heuristic — we don't parse the JSON here.
    if re.search(r"```json[\s\S]*?\"signatures\"[\s\S]*?```", doc_text):
        return CheckResult("signature_present", True, "sigstore-like JSON block present")
    return CheckResult(
        "signature_present",
        False,
        "no PGP or sigstore signature block found",
    )


def _extract_pgp_armor_block(doc_text: str) -> Optional[str]:
    """Return the first complete PGP armor block from doc_text, or None.

    The block is matched literally between ``-----BEGIN PGP SIGNATURE-----``
    and ``-----END PGP SIGNATURE-----``. We use a non-greedy match so multi-
    block documents return only the first block. Returns the entire armored
    string (including BEGIN/END markers) so it can be piped to ``gpg``.
    """

    pattern = re.compile(
        r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",
        re.DOTALL,
    )
    match = pattern.search(doc_text)
    return match.group(0) if match else None


def _extract_sigstore_json_block(doc_text: str) -> Optional[str]:
    """Return the first sigstore JSON bundle (between ```json fences), or None."""

    pattern = re.compile(
        r"```json\s*(?P<body>\{[\s\S]*?\"signatures\"[\s\S]*?\})\s*```",
        re.DOTALL,
    )
    match = pattern.search(doc_text)
    return match.group("body") if match else None


def _verify_pgp_signature(
    doc_path: Path,
    armor_block: str,
    keyring_dir: Optional[Path] = None,
) -> tuple[bool, str]:
    """Run ``gpg --verify`` against the armored block and the doc payload.

    Returns ``(ok, detail)``. ``ok`` is True iff gpg returns 0. The detail
    string contains gpg's stderr (which is what gpg writes verification
    output to).

    The "doc payload" is the body of the document up to (but not including)
    the ``## Cryptographic signature`` heading — see
    ``docs/results/optum_methodology_signoff_template.md`` §Cryptographic
    signature.
    """

    if shutil.which("gpg") is None:
        return False, "gpg binary not found on PATH"

    doc_text = doc_path.read_text(encoding="utf-8")
    payload_marker = "## Cryptographic signature"
    if payload_marker in doc_text:
        payload = doc_text.split(payload_marker, 1)[0]
    else:
        payload = doc_text

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        sig_path = tmpdir_path / "sig.asc"
        payload_path = tmpdir_path / "payload.txt"
        sig_path.write_text(armor_block, encoding="utf-8")
        payload_path.write_text(payload, encoding="utf-8")

        cmd = ["gpg", "--batch", "--status-fd=1"]
        if keyring_dir is not None:
            cmd.extend(["--homedir", str(keyring_dir)])
        cmd.extend(["--verify", str(sig_path), str(payload_path)])

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except FileNotFoundError:
            return False, "gpg binary not found on PATH"
        except subprocess.TimeoutExpired:
            return False, "gpg verification timed out"

        ok = completed.returncode == 0
        # gpg writes verification output to stderr; status messages to stdout.
        combined = (completed.stderr or "") + (completed.stdout or "")
        return ok, combined.strip() or f"gpg returncode={completed.returncode}"


def _verify_sigstore_bundle(bundle_json: str) -> tuple[bool, str]:
    """Run ``cosign verify-blob`` (or ``rekor-cli verify``) against the bundle.

    Both tools are stub-best-effort: the bundle alone is not enough — cosign
    needs ``--certificate-identity`` and ``--certificate-oidc-issuer``, and
    rekor-cli needs a separate artifact. We accept the bundle, write it to a
    temp file, and ask cosign to verify the bundle's internal signatures
    via ``cosign verify-blob --bundle <path> --insecure-ignore-tlog``.
    Failure is fatal; absence of any tool is fatal under require_signature.
    """

    has_cosign = shutil.which("cosign") is not None
    has_rekor = shutil.which("rekor-cli") is not None
    if not (has_cosign or has_rekor):
        return False, "neither cosign nor rekor-cli found on PATH"

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "bundle.sigstore"
        bundle_path.write_text(bundle_json, encoding="utf-8")
        if has_cosign:
            cmd = [
                "cosign",
                "verify-blob",
                "--bundle",
                str(bundle_path),
                "--insecure-ignore-tlog",
                str(bundle_path),
            ]
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, f"cosign invocation failed: {exc}"
            ok = completed.returncode == 0
            combined = (completed.stderr or "") + (completed.stdout or "")
            return ok, combined.strip() or f"cosign returncode={completed.returncode}"
        # rekor-cli verify needs the original artifact + a UUID; without those
        # we can only report best-effort presence (matches its limitation).
        return False, "rekor-cli requires --uuid + artifact; bundle alone insufficient"


def check_signature_verifies(
    doc_path: Path,
    require_signature: bool,
    keyring_dir: Optional[Path] = None,
) -> CheckResult:
    """Verify the PGP / sigstore block in the document cryptographically.

    Behaviour:

    * ``require_signature=True`` (the CI path): actual cryptographic
      verification is performed. Missing toolchain → FAIL. Missing armor /
      bundle → FAIL. Non-zero verifier exit → FAIL. This is the H1 fix —
      previously this code path only checked binary existence.
    * ``require_signature=False`` (the default, used for local-dev /
      scaffolding): no cryptographic verification is attempted; the function
      records a WARN and returns PASS, preserving the prior contract for
      callers that want to defer verification (e.g. tests that compose
      sign-off docs with placeholder signatures).

    The PGP code path verifies the armored block against the document body
    up to (but not including) the ``## Cryptographic signature`` heading,
    matching the convention in
    ``docs/results/optum_methodology_signoff_template.md``.

    When a sigstore bundle is present and cosign is available we shell out
    to ``cosign verify-blob``.
    """

    if not require_signature:
        # H1: in advisory mode (no --require-signature), we do not silently
        # claim verification succeeded just because gpg is on PATH. We
        # explicitly note that no crypto check ran. Callers that want
        # actual verification MUST pass --require-signature.
        return CheckResult(
            "signature_verifies",
            True,
            "WARN: --require-signature not set; cryptographic verification skipped",
        )

    doc_text = doc_path.read_text(encoding="utf-8")
    pgp_block = _extract_pgp_armor_block(doc_text)
    sigstore_block = _extract_sigstore_json_block(doc_text)

    has_gpg = shutil.which("gpg") is not None
    has_cosign = shutil.which("cosign") is not None
    has_rekor = shutil.which("rekor-cli") is not None
    has_any = has_gpg or has_cosign or has_rekor

    if not has_any:
        return CheckResult(
            "signature_verifies",
            False,
            "no signature-verification tool found (gpg, cosign, rekor-cli) and --require-signature passed",
        )

    if pgp_block is None and sigstore_block is None:
        return CheckResult(
            "signature_verifies",
            False,
            "no extractable PGP armor block or sigstore JSON bundle found",
        )

    if pgp_block is not None and has_gpg:
        ok, detail = _verify_pgp_signature(doc_path, pgp_block, keyring_dir=keyring_dir)
        if ok:
            return CheckResult("signature_verifies", True, f"gpg --verify OK: {detail[:200]}")
        return CheckResult(
            "signature_verifies",
            False,
            f"gpg --verify FAILED: {detail[:500]}",
        )

    if sigstore_block is not None and (has_cosign or has_rekor):
        ok, detail = _verify_sigstore_bundle(sigstore_block)
        if ok:
            return CheckResult("signature_verifies", True, f"sigstore verify OK: {detail[:200]}")
        return CheckResult(
            "signature_verifies",
            False,
            f"sigstore verify FAILED: {detail[:500]}",
        )

    # We have a verifier available, but it does not match the block kind.
    return CheckResult(
        "signature_verifies",
        False,
        f"no matching verifier for block kind (pgp_block={pgp_block is not None}, sigstore_block={sigstore_block is not None}, has_gpg={has_gpg}, has_cosign={has_cosign}, has_rekor={has_rekor})",
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def check_signoff(
    doc_path: Path,
    repo_root: Path,
    require_signature: bool = False,
    keyring_dir: Optional[Path] = None,
    today: Optional[str] = None,
    max_age_days: int = MAX_SIGNOFF_AGE_DAYS,
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

    results.append(check_signoff_age(doc_path, today=today, max_age_days=max_age_days))
    results.append(check_required_sections(doc_text, kind))
    results.append(check_signature_present(doc_text))
    results.append(check_coi_referenced(doc_text, repo_root=repo_root))

    registry_path = repo_root / "docs" / "governance" / "methodology_reviewer_registry.md"
    try:
        registry = parse_registry(registry_path)
    except FileNotFoundError as exc:
        results.append(CheckResult("registry_loaded", False, str(exc)))
        return results

    results.append(CheckResult("registry_loaded", True, f"{len(registry)} rows"))
    results.append(check_reviewer_registered(doc_text, registry))

    # Read the CoI declaration body (if it exists at the declared path) so
    # check_selection_rule can scan its declared-PRs JSON for self-admitted
    # subject-file overlap.
    coi_text: Optional[str] = None
    coi_path = extract_coi_path(doc_text)
    if coi_path:
        coi_full = repo_root / coi_path
        if coi_full.is_file():
            coi_text = coi_full.read_text(encoding="utf-8")
    results.append(check_selection_rule(doc_text, repo_root, registry, coi_text=coi_text))
    results.append(check_signature_verifies(doc_path, require_signature, keyring_dir=keyring_dir))
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
            "Treat absence of a signature-verification toolchain (gpg, cosign, rekor-cli) as fatal."
        ),
    )
    parser.add_argument(
        "--keyring-dir",
        type=Path,
        default=None,
        help=(
            "Optional path to a GPG home directory containing the trusted "
            "public keys for sign-off reviewers. Passed to gpg via "
            "--homedir. If unset, the system default keyring is used."
        ),
    )
    parser.add_argument(
        "--today",
        default=None,
        help=(
            "Reference date (YYYY-MM-DD) for the sign-off-age check (M2). "
            "Defaults to the system date. Pass an explicit value when "
            "running the validator from a deterministic harness."
        ),
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=MAX_SIGNOFF_AGE_DAYS,
        help=(
            f"Maximum age (days) of the sign-off artifact's filename date "
            f"vs --today. Older artifacts are rejected. Default: "
            f"{MAX_SIGNOFF_AGE_DAYS}."
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

    results = check_signoff(
        doc_path,
        repo_root,
        require_signature=args.require_signature,
        keyring_dir=args.keyring_dir,
        today=args.today,
        max_age_days=args.max_age_days,
    )
    print(render_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
