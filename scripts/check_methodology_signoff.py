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
* ``3`` — strict-gh policy violation: ``--strict-gh`` (or ``STRICT_GH=1`` in
  the environment) was set AND at least one CheckResult reports
  ``provenance_check_skipped=True`` (gh CLI unavailable / unauthenticated).
  This exit is the issue #192 H2/M1 fail-closed policy: in CI deployments
  where the methodology-signoff-guard workflow provisions ``GH_TOKEN`` and
  exports ``STRICT_GH=1``, a skipped gh provenance query MUST hard-fail
  rather than warn — otherwise a reviewer can self-declare clean while
  authoring/reviewing PRs via the GitHub web UI (no git-attributable commit).
  Local devs without ``--strict-gh`` retain the warn-only back-compat path.
* ``4`` — strict-gpg policy violation (issue #226 H1+H4): ``--strict-gpg``
  (or ``STRICT_GPG=1`` in the environment) was set AND the keyring
  pre-check failed (keyring directory missing, empty, or contains zero
  imported public keys). Reserved exit code so log scrapers can distinguish
  "keyring not provisioned on this runner" from generic validation failures.
  The CI workflow ``.github/workflows/methodology-signoff-validator.yml``
  provisions the keyring from the ``GPG_REVIEWER_KEYS_ARMOR_BASE64`` repo
  secret BEFORE invoking the validator; this exit code fires when the
  secret is unset or the import produced no usable keys.

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

    Issue #192 H2/M1 fail-closed escalation:
        The ``--strict-gh`` CLI flag (or ``STRICT_GH=1`` in the environment)
        promotes ``provenance_check_skipped=True`` from a logged warning to
        a hard exit (code 3). The methodology-signoff-guard workflow now
        provisions ``GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}`` AND exports
        ``STRICT_GH=1`` so CI runs hit the fail-closed path. Local devs who
        run the validator without GH_TOKEN retain the warn-only back-compat
        behavior because they neither pass ``--strict-gh`` nor have the env
        var set.

NEW MED (iter-3) — future-dated artifacts:
    ``check_signoff_age`` rejects sign-offs whose filename date is more than
    1 day ahead of ``today``. The 1-day tolerance covers TZ-skew at the day
    boundary. Prevents reviewers from pre-dating sign-offs to evade the
    max-age window.

Issue #226 H1+H4 — GPG keyring bridge code (2026-05-14):
    Adds the code-side infrastructure for the registry-pinned GPG keyring
    (H1) AND CoI body signature verification (H4). After this PR ships,
    operator's residual job is one secret addition + populating the
    fingerprint column in the reviewer registry markdown.

    Three new validator-side surfaces:

    1. ``ReviewerInfo.fingerprint`` (parsed from the new
       ``fingerprint`` registry column). Stripped of internal whitespace
       so the raw 40-char hex fingerprint can be passed to gpg's ``--with-
       fingerprint`` filter at validation time.
    2. ``check_keyring_present`` (called when ``--keyring-dir`` is set):
       PASSES when the keyring directory exists AND ``gpg --list-keys``
       reports at least one key. WARNs in default mode + FAILs under
       ``--strict-gpg`` (exit code 4).
    3. ``check_coi_body_signature_verifies`` (H4): verifies the CoI
       declaration markdown body against an embedded ASCII-armor block
       (or a sibling ``<coi_path>.asc`` detached signature) using
       ``gpg --homedir <keyring_dir> --verify``. PASSES when no signature
       is present in default mode + FAILs under ``--strict-gpg``.

    Strategy for keyring distribution (defendable to codex): secret-encoded
    ASCII-armor bundle. Operator generates reviewer pubkeys offline,
    concatenates ASCII-armor exports into a single multi-key blob,
    base64-encodes it, and adds as repo secret
    ``GPG_REVIEWER_KEYS_ARMOR_BASE64``. CI workflow imports that into a
    per-job ``$KEYRING_DIR`` and passes ``--keyring-dir`` to the
    validator. See ``docs/governance/operator_gpg_keyring_setup.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
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
        signature_check_skipped: Issue #226 H4 — True when a cryptographic
            signature check passed in advisory mode (no signature found
            but the operator has not opted into STRICT_GPG=1). Mirrors
            ``provenance_check_skipped`` semantically: the check passed
            on the available evidence, but the caller (CI) MUST decide
            whether to fail-closed when this flag is set. Distinct from
            ``provenance_check_skipped`` so log scrapers / strict-mode
            policy logic can distinguish "gh provenance not confirmed"
            from "no CoI body signature pinned". A WARN is logged in the
            detail string when this flag is True.
        signing_fingerprint: Issue #226 H1 (codex pass-1 HIGH-1) — the
            40-char hex fingerprint extracted from gpg's
            ``[GNUPG:] VALIDSIG`` status line when a signature
            verification check succeeded. Populated by
            ``check_signature_verifies`` (sign-off doc) and
            ``check_coi_body_signature_verifies`` (CoI body); consumed
            by ``check_signing_fingerprint_matches_registry`` to bind
            the signature to a registered reviewer. None when the
            corresponding verify check did NOT pass.
    """

    name: str
    ok: bool
    detail: str = ""
    provenance_check_skipped: bool = False
    signature_check_skipped: bool = False
    signing_fingerprint: Optional[str] = None


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

    Issue #226 H1: ``fingerprint`` is the registered GPG key fingerprint
    (40-char hex, no spaces — internal whitespace from the registry cell
    is stripped at parse time). Empty string when the registry row carries
    a placeholder (``<TBD ...>``) or omits the cell. Consumed by the H1
    sign-off-key-binding check and by ``check_keyring_present`` when the
    workflow exports STRICT_GPG=1.
    """

    handle: str
    email: str
    status: str
    emails: tuple[str, ...] = ()
    fingerprint: str = ""


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
    "fingerprint",
)

# Tokens that indicate the fingerprint cell is an operator-fillable
# placeholder (NOT a real GPG fingerprint). When the parsed value matches
# any of these (case-insensitive substring), ReviewerInfo.fingerprint is
# set to the empty string so the H1 keyring check downstream knows to
# treat the row as "no fingerprint pinned yet".
_FINGERPRINT_PLACEHOLDER_TOKENS: tuple[str, ...] = (
    "<tbd",
    "<placeholder",
    "<populated",
    "tbd ",
    "n/a",
    "none",
)


def _normalize_fingerprint(raw: str) -> str:
    """Return the canonical 40-char hex fingerprint, or '' for placeholders.

    Strips:
      * Surrounding markdown emphasis / backticks (``_`` ``*`` `` ` ``).
      * All internal whitespace (operators sometimes paste fingerprints
        with the conventional space-every-4-chars formatting).
      * The ``0x`` prefix if present.

    Returns:
      * The uppercased hex string when the result is exactly 40 hex chars.
      * The empty string when the cell matches a placeholder pattern OR
        cannot be parsed as a fingerprint. The caller (the keyring check)
        treats empty as "not pinned" rather than as a hard failure so
        registry rows can be added incrementally.
    """

    # Codex pass-2 LOW-1: strip a leading BOM (operators sometimes paste
    # via terminals that prepend U+FEFF) along with surrounding markdown
    # emphasis and whitespace. Without this, a BOM-prefixed real
    # fingerprint normalizes to "" → STRICT_GPG=1 false-fail.
    cleaned = raw.lstrip("﻿").strip().strip("_*`").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for token in _FINGERPRINT_PLACEHOLDER_TOKENS:
        if token in lowered:
            return ""
    # Strip whitespace and a leading 0x; uppercase.
    no_ws = re.sub(r"\s+", "", cleaned)
    if no_ws.lower().startswith("0x"):
        no_ws = no_ws[2:]
    if re.fullmatch(r"[0-9A-Fa-f]{40}", no_ws):
        return no_ws.upper()
    # Any other shape is not a usable fingerprint — return empty so the
    # downstream check treats the row as "no fingerprint pinned yet"
    # rather than failing the whole registry parse.
    return ""


def parse_registry(registry_path: Path) -> List[ReviewerInfo]:
    """Parse the reviewer-registry markdown table into ReviewerInfo records.

    The registry is a simple GFM table; we hand-parse rather than depending on
    a markdown library so the script remains stdlib-only.

    Raises:
        FileNotFoundError if the registry does not exist.
        ValueError if the table headers do not match the expected schema.

    Issue #226 H1 schema migration: the registry now includes a
    ``fingerprint`` column (8th cell). Rows whose fingerprint cell is a
    placeholder (``<TBD ...>``, ``<populated by operator>``, etc.) are
    parsed with ``ReviewerInfo.fingerprint=""``; the H1 keyring-binding
    check treats empty as "not pinned" (warns in default mode, FAILs
    under STRICT_GPG=1 only when ``check_keyring_present`` is also
    failing — i.e. the operator hasn't completed the handoff).

    Back-compat: legacy registries (without the fingerprint column) are
    still parsed best-effort. The header-equality check fails on the
    8-column header, so a 7-column registry is rejected — that is the
    intended behavior so operators are forced to migrate the schema in
    lockstep with the workflow change.
    """

    rows, _warnings = parse_registry_with_warnings(registry_path)
    return rows


def parse_registry_with_warnings(
    registry_path: Path,
) -> tuple[List[ReviewerInfo], List[str]]:
    """Like :func:`parse_registry` but ALSO returns parser warnings.

    Codex pass-5 MED-1: malformed registry rows (e.g. extra ``|`` in
    a free-text cell breaking column count) were previously silently
    skipped — which after pass-4's all-row aggregation in
    ``check_selection_rule`` could DELETE disqualifying-evidence
    rows from the matching set. This function returns:

    * The list of successfully parsed rows.
    * A list of human-readable warnings about rows that LOOKED like
      table-body rows (in_table+saw_separator was True AND the line
      started with ``|``) but had wrong column counts. The
      orchestrator surfaces these as a non-fatal but visible WARN
      so operators see the rows they thought they added but that
      didn't parse.
    """

    if not registry_path.is_file():
        raise FileNotFoundError(f"registry not found: {registry_path}")

    rows: List[ReviewerInfo] = []
    warnings: List[str] = []
    in_table = False
    saw_separator = False
    for line_no, raw_line in enumerate(
        registry_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
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
        if in_table and saw_separator:
            if len(cells) != len(_REGISTRY_HEADERS):
                # Codex pass-5 MED-1: emit a warning for rows that look
                # like table-body rows (in_table + post-separator + |
                # delimited) but have wrong column counts. Pre-fix this
                # was silently skipped, hiding disqualifying-evidence rows
                # from selection-rule aggregation.
                warnings.append(
                    f"line {line_no}: pipe-delimited row has "
                    f"{len(cells)} cells but expected {len(_REGISTRY_HEADERS)} "
                    f"({_REGISTRY_HEADERS!r}); row skipped — was: {raw_line!r}"
                )
                continue
            handle, email_cell, status = cells[2], cells[1], cells[6]
            # Strip Markdown emphasis (e.g. _PLACEHOLDER_).
            handle = handle.strip("_*`")
            # M1: split the email cell on comma/semicolon to support alias
            # lists like "alice@example.com, alice@oldjob.com". The first
            # address is treated as canonical; all addresses go into the
            # emails tuple for the selection rule's git-log probes.
            aliases = tuple(a.strip() for a in re.split(r"[,;]", email_cell) if a.strip())
            primary = aliases[0] if aliases else email_cell
            # Issue #226 H1: parse the fingerprint cell (8th column).
            # Placeholder values normalize to "" so the keyring check
            # downstream knows the row hasn't been operator-populated yet.
            fingerprint = _normalize_fingerprint(cells[7])
            rows.append(
                ReviewerInfo(
                    handle=handle,
                    email=primary,
                    status=status,
                    emails=aliases or (email_cell,),
                    fingerprint=fingerprint,
                )
            )
    return rows, warnings


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
    """Reviewer's handle must appear in the registry as ``status=active``.

    Codex pass-4 MED-1 fix: walk ALL matching rows and PASS if at
    least one is active. The pre-fix returned on the FIRST matching
    row's status, so an inactive/recused historical row appearing
    BEFORE a later active row would falsely reject the reviewer. The
    registry is documented as append-only ("do not edit historical
    rows; to deactivate, set status to inactive AND add a new row")
    so duplicate handles with mixed statuses are an EXPECTED state.
    """

    handle = extract_handle(doc_text)
    if handle is None:
        return CheckResult(
            "reviewer_registered",
            False,
            "GitHub handle missing from sign-off doc",
        )
    matching = [row for row in registry if row.handle == handle]
    if not matching:
        return CheckResult(
            "reviewer_registered",
            False,
            f"reviewer {handle!r} not in registry",
        )
    active_rows = [r for r in matching if r.status == "active"]
    if active_rows:
        n_active = len(active_rows)
        n_total = len(matching)
        suffix = f" ({n_active}/{n_total} rows active)" if n_total > 1 else ""
        return CheckResult(
            "reviewer_registered",
            True,
            f"{handle} (status=active){suffix}",
        )
    # No active rows but matches exist → reviewer is registered as
    # historical / recused / inactive only. FAIL with clear detail
    # listing the statuses found so the operator knows the row is
    # historical, not missing.
    statuses_found = sorted({r.status for r in matching})
    return CheckResult(
        "reviewer_registered",
        False,
        f"reviewer {handle!r} is in registry but no active rows "
        f"(found {len(matching)} row(s) with statuses {statuses_found!r})",
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
    # Codex pass-4 HIGH-1 fix: REVERT the pass-3 active-only filter
    # for selection-rule. Selection-rule evidence is about reviewer
    # INVOLVEMENT during the named period — historical / recused rows'
    # emails are STILL valid selection evidence (a reviewer who
    # touched the subject file in-window then got recused is not
    # magically un-conflicted). Active-only filtering here would
    # create a CoI bypass: a reviewer with [historical/recused row
    # carrying the disqualifying email + active row carrying a clean
    # email] would slip past the git-log probe. Active-row filtering
    # is retained ONLY for fingerprint pinning (where it gates
    # CURRENT signing eligibility) and reviewer-registration
    # (CURRENT review eligibility). Pass-2 MED-1 row aggregation
    # across ALL matching rows is the correct semantic for the
    # selection rule's CoI evidence.
    row = matching[0]
    email = row.email
    # M1 + pass-2 MED-1 + pass-4 HIGH-1: iterate over ALL declared
    # aliases (across ALL registry rows for this handle, regardless
    # of status) so a commit authored under an alternate identity —
    # including historical / recused rows — is still caught by
    # `git log --author=`. Falls back to the primary email if the
    # row predates the alias-aware schema.
    aliases_set: list[str] = []
    seen_aliases: set[str] = set()
    for r in matching:
        for alias in r.emails or (r.email,):
            if alias not in seen_aliases:
                seen_aliases.add(alias)
                aliases_set.append(alias)
    aliases = tuple(aliases_set) or (email,)
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


# Codex pass-2 LOW-1: case-insensitive match. Modern gpg always emits
# uppercase for VALIDSIG fingerprints, but be defensive against future /
# patched versions that emit lowercase. The captured group is upper-cased
# at extraction time so downstream comparisons are deterministic.
_VALIDSIG_PATTERN = re.compile(
    r"^\[GNUPG:\]\s+VALIDSIG\s+(?P<fpr>[0-9A-Fa-f]{40})\b",
    re.MULTILINE,
)


def _extract_validsig_fingerprint(gpg_status_output: str) -> Optional[str]:
    """Return the 40-char hex fingerprint from a ``[GNUPG:] VALIDSIG`` line.

    Issue #226 codex pass-1 HIGH-1: the validator must bind the verified
    signature to a registered reviewer fingerprint. ``gpg --status-fd=1``
    emits a ``[GNUPG:] VALIDSIG <fingerprint> <date> ...`` line ONLY when
    the signature verifies cryptographically AND the signing key is
    available in the keyring (i.e. equivalent to gpg returning 0).

    Returns the uppercased 40-char hex string, or None when no
    VALIDSIG line is present (verification failed OR gpg version too
    old to emit the line in this format — in either case fingerprint
    pinning cannot be evaluated and the caller treats this as a
    fail/skip per the strict-gpg policy).
    """

    match = _VALIDSIG_PATTERN.search(gpg_status_output)
    if match is None:
        return None
    return match.group("fpr").upper()


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
) -> tuple[bool, str, Optional[str]]:
    """Run ``gpg --verify`` against the armored block and the doc payload.

    Returns ``(ok, detail, signing_fingerprint)``. ``ok`` is True iff gpg
    returns 0. The detail string contains gpg's stderr+stdout (gpg writes
    verification output to stderr, status messages to stdout).
    ``signing_fingerprint`` is the 40-char hex VALIDSIG fingerprint when
    verification succeeded, else None.

    Issue #226 codex pass-1 HIGH-1: callers use the returned fingerprint
    to bind the signature to a registry-pinned reviewer (separate
    ``check_signing_fingerprint_matches_registry`` check). Returning
    None on verification failure is intentional — pinning is
    impossible without a known-good fingerprint.

    The "doc payload" is the body of the document up to (but not including)
    the ``## Cryptographic signature`` heading — see
    ``docs/results/optum_methodology_signoff_template.md`` §Cryptographic
    signature.
    """

    if shutil.which("gpg") is None:
        return False, "gpg binary not found on PATH", None

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
            return False, "gpg binary not found on PATH", None
        except subprocess.TimeoutExpired:
            return False, "gpg verification timed out", None

        ok = completed.returncode == 0
        # gpg writes verification output to stderr; status messages to stdout.
        # Extract VALIDSIG fingerprint from --status-fd=1 stdout for issue
        # #226 H1 fingerprint-pinning binding.
        status_fd_output = completed.stdout or ""
        signing_fpr = _extract_validsig_fingerprint(status_fd_output) if ok else None
        combined = (completed.stderr or "") + status_fd_output
        detail = combined.strip() or f"gpg returncode={completed.returncode}"
        return ok, detail, signing_fpr


def _verify_sigstore_bundle(
    bundle_json: str,
    payload: Optional[str] = None,
) -> tuple[bool, str]:
    """Run ``cosign verify-blob`` against the bundle and a payload.

    iter-3 NEW HIGH (sigstore misuse):

    The pre-iter-3 implementation invoked
    ``cosign verify-blob --bundle <bundle> <bundle>`` — i.e. it asked cosign
    to verify the BUNDLE FILE as if it were the signed artifact, which is
    structurally wrong. cosign verify-blob's positional arg is the
    ARTIFACT whose signature is in the bundle. Verifying the bundle as
    its own artifact silently passes for any well-formed bundle whose
    payload-hash field matches the bundle file hash, OR fails for
    spurious reasons unrelated to the actual sign-off doc.

    This iter-3 fix accepts the original ``payload`` (the sign-off doc
    body up to but not including the ``## Cryptographic signature``
    heading) and writes it to a temp file before invoking cosign so the
    correct artifact is verified.

    KNOWN LIMITATION (deferred — see
    ``docs/governance/n3_known_limitations_20260510.md``): cosign also
    requires ``--certificate-identity`` and ``--certificate-oidc-issuer``
    for the keyless flow OR ``--key`` for the long-lived-key flow.
    Without those, an attacker who can run a sigstore-signing-capable
    OIDC identity can produce a verifiable signature regardless of who
    they are. Production deployments MUST set those flags via env-var
    or config; this validator does not enforce identity binding.

    Args:
        bundle_json: the JSON text inside the sign-off doc's
            `````json ... ````` fence.
        payload: the doc body up to the ``## Cryptographic signature``
            heading (must match what was signed). When omitted, the
            verification falls back to verifying the bundle JSON as the
            artifact — KNOWN BROKEN, retained only for transitional
            compatibility and emits a warning in the detail string.

    Returns:
        ``(ok, detail)``. ``ok`` is True iff cosign returns 0 against
        the supplied payload. Failure is fatal under require_signature.
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
            warnings: list[str] = []
            if payload is None:
                # Pre-iter-3 broken behaviour retained only for legacy
                # callers; new callers MUST pass payload. Emit a warning
                # in the detail so operators see this is a degraded path.
                artifact_path = bundle_path
                warnings.append(
                    "payload is None — verifying bundle file as its own artifact "
                    "(KNOWN BROKEN; pass payload to fix)"
                )
            else:
                artifact_path = Path(tmpdir) / "payload.txt"
                artifact_path.write_text(payload, encoding="utf-8")
            cmd = [
                "cosign",
                "verify-blob",
                "--bundle",
                str(bundle_path),
                "--insecure-ignore-tlog",
                str(artifact_path),
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
            detail = combined.strip() or f"cosign returncode={completed.returncode}"
            if warnings:
                detail = "WARN: " + "; ".join(warnings) + " | " + detail
            return ok, detail
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

    # Codex pass-3 MED-1 fix: parse the doc + extract block kinds BEFORE
    # the keyring-advisory preflight. This guarantees:
    #   (a) sigstore-only docs are NOT preempted by an empty GPG keyring
    #       (the GPG keyring is irrelevant to cosign verification);
    #   (b) the gpg-not-on-PATH case is surfaced as
    #       "no signature-verification tool found" rather than collapsed
    #       into "keyring missing advisory" (which would falsely tell
    #       operators their keyring is the problem).
    doc_text = doc_path.read_text(encoding="utf-8")
    pgp_block = _extract_pgp_armor_block(doc_text)
    sigstore_block = _extract_sigstore_json_block(doc_text)

    # Issue #226 codex pass-2 HIGH-1 + pass-3 MED-1: only apply the
    # keyring-advisory preflight to the PGP code path. Sigstore-only
    # docs proceed to cosign verification regardless of GPG keyring
    # state. Distinguishes "operator hasn't provisioned the keyring
    # yet" (advisory-pass per H1) from "no verifier toolchain at all"
    # (which is the existing "no signature-verification tool found"
    # FAIL branch downstream).
    if pgp_block is not None and keyring_dir is not None and shutil.which("gpg") is not None:
        kr_ok, _kr_n, kr_detail = _gpg_list_keys_in_keyring(keyring_dir)
        if not kr_ok:
            return CheckResult(
                "signature_verifies",
                True,
                f"WARN: keyring at {keyring_dir} is missing/empty/unreadable "
                f"({kr_detail}); cryptographic verification skipped (issue #226 "
                "H1 advisory-pass; STRICT_GPG=1 escalates via signature_check_skipped)",
                signature_check_skipped=True,
            )

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
        ok, detail, signing_fpr = _verify_pgp_signature(
            doc_path, pgp_block, keyring_dir=keyring_dir
        )
        if ok:
            fpr_suffix = f" [signing_fpr={signing_fpr}]" if signing_fpr else ""
            return CheckResult(
                "signature_verifies",
                True,
                f"gpg --verify OK: {detail[:200]}{fpr_suffix}",
                signing_fingerprint=signing_fpr,
            )
        return CheckResult(
            "signature_verifies",
            False,
            f"gpg --verify FAILED: {detail[:500]}",
        )

    if sigstore_block is not None and (has_cosign or has_rekor):
        # iter-3 NEW HIGH: extract the doc body up to the cryptographic-
        # signature heading and pass it as the payload — cosign verify-
        # blob's positional arg must be the ARTIFACT, not the bundle.
        payload_marker = "## Cryptographic signature"
        if payload_marker in doc_text:
            sigstore_payload = doc_text.split(payload_marker, 1)[0]
        else:
            sigstore_payload = doc_text
        ok, detail = _verify_sigstore_bundle(sigstore_block, payload=sigstore_payload)
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
# Issue #226 H1+H4 — GPG keyring presence + CoI body signature verification.
# --------------------------------------------------------------------------- #


def _gpg_list_keys_in_keyring(keyring_dir: Path) -> tuple[bool, int, str]:
    """Return ``(ok, n_keys, detail)`` for ``gpg --homedir <dir> --list-keys``.

    ``ok`` is True iff gpg returned 0 AND parsed at least one ``pub:``
    record. ``n_keys`` is the count of ``pub:`` records (each public-key
    primary). ``detail`` is the truncated combined stderr+stdout for
    inclusion in CheckResult details.

    When gpg is unavailable or the keyring path does not exist, returns
    ``(False, 0, "<reason>")`` so the caller can decide whether to FAIL
    (under STRICT_GPG=1) or WARN (default).
    """

    if shutil.which("gpg") is None:
        return False, 0, "gpg binary not found on PATH"
    if not keyring_dir.is_dir():
        return False, 0, f"keyring directory does not exist: {keyring_dir}"

    cmd = [
        "gpg",
        "--batch",
        "--homedir",
        str(keyring_dir),
        "--list-keys",
        "--with-colons",
        "--keyid-format=long",
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
        return False, 0, f"gpg --list-keys failed: {exc}"

    if completed.returncode != 0:
        # gpg returns non-zero when the homedir is unreadable or the
        # trustdb cannot be initialized; surface that as fail-with-detail.
        stderr = (completed.stderr or "").strip()
        return False, 0, f"gpg --list-keys returncode={completed.returncode}: {stderr[:200]}"

    n_pub = sum(1 for line in (completed.stdout or "").splitlines() if line.startswith("pub:"))
    if n_pub == 0:
        return False, 0, "gpg --list-keys reports zero public keys in keyring"
    return True, n_pub, f"{n_pub} public key(s) in keyring"


def check_keyring_present(keyring_dir: Optional[Path]) -> CheckResult:
    """H1: keyring directory exists AND contains at least one usable pubkey.

    Behavior matrix (codex pass-1 MED-1 fix: missing/empty keyring is
    ADVISORY PASS with ``signature_check_skipped=True`` so the
    STRICT_GPG=1 escalation path returns the reserved exit code 4 via
    the same code path as the H4 sig-skip — NOT exit 1 via the generic
    ``not all(r.ok)`` branch which would lose the routing signal):

    * ``keyring_dir is None``  → ADVISORY PASS with sig-skip flag.
      Default-mode validators (no STRICT_GPG) treat this as "the
      operator is running locally without keyring infra — preserve
      back-compat". CI workflows that provision the keyring MUST pass
      ``--keyring-dir <path>`` so this branch only fires when the
      operator handoff secret has not yet been provisioned.
    * ``keyring_dir`` set BUT directory missing OR empty (zero keys) →
      ADVISORY PASS with sig-skip flag. STRICT_GPG=1 escalates to
      exit 4 in main(); STRICT_GPG=0 yields a logged WARN but the
      orchestrator continues so a partial rollout (keyring secret not
      yet uploaded) doesn't break every PR.
    * ``keyring_dir`` set AND populated → PASS with key count, no
      sig-skip flag.

    The CheckResult name is ``keyring_present`` so log scrapers can
    distinguish from ``signature_verifies`` (which is about a specific
    artifact's signature). Codex pass-1 MED-1 (2026-05-15): converted
    fail-mode to advisory-skip so exit 4 is the deterministic outcome
    under STRICT_GPG=1 instead of the generic exit 1.
    """

    if keyring_dir is None:
        return CheckResult(
            "keyring_present",
            True,
            "WARN: --keyring-dir not set; keyring binding skipped (issue #226 H1)",
            signature_check_skipped=True,
        )

    ok, _n_keys, detail = _gpg_list_keys_in_keyring(keyring_dir)
    if not ok:
        return CheckResult(
            "keyring_present",
            True,
            f"WARN: keyring at {keyring_dir} is missing/empty/unreadable: "
            f"{detail}; H1 advisory mode",
            signature_check_skipped=True,
        )
    return CheckResult(
        "keyring_present",
        True,
        f"keyring at {keyring_dir}: {detail}",
    )


_COI_INLINE_SIG_PATTERN = re.compile(
    r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----",
    re.DOTALL,
)


def _verify_coi_body_signature(
    coi_path: Path,
    keyring_dir: Optional[Path],
) -> tuple[Optional[bool], str, Optional[str]]:
    """Run ``gpg --verify`` against the CoI declaration body.

    Returns ``(ok, detail, signing_fingerprint)`` where ``ok`` is:

    * ``True``  — gpg verified the CoI body successfully.
    * ``False`` — gpg ran AND verification failed.
    * ``None``  — there is no signature to verify (no inline armor block,
                  no sibling ``<coi_path>.asc`` detached signature). The
                  caller (the H4 check) treats None as "advisory pass" in
                  default mode and FAIL under STRICT_GPG=1.

    ``signing_fingerprint`` is the 40-char hex VALIDSIG fingerprint when
    verification succeeded (used by H4 fingerprint pinning), else None.

    Two signature-discovery paths:

    1. **Inline ASCII armor**: an embedded
       ``-----BEGIN PGP SIGNATURE-----...-----END PGP SIGNATURE-----``
       block somewhere in the CoI markdown body. Verified against the
       body content UP TO (but not including) the first signature block
       (mirrors the sign-off doc's payload-extraction convention).
    2. **Sibling detached sig**: a file at ``<coi_path>.asc`` containing
       a detached signature for the CoI body. Verified against the FULL
       CoI body (no payload truncation).

    Inline takes precedence when both are present (the inline-armor case
    is the operator-friendly default; the sibling-file case is for
    reviewers who prefer git-attestation-style detached sigs).
    """

    if not coi_path.is_file():
        return None, f"CoI body not found at {coi_path}", None

    if shutil.which("gpg") is None:
        return None, "gpg binary not found on PATH", None

    coi_text = coi_path.read_text(encoding="utf-8")
    inline_match = _COI_INLINE_SIG_PATTERN.search(coi_text)

    import tempfile

    if inline_match is not None:
        armor_block = inline_match.group(0)
        # Payload = body text up to the first armor block.
        payload = coi_text[: inline_match.start()]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            sig_path = tmpdir_path / "coi.sig.asc"
            payload_path = tmpdir_path / "coi.payload.txt"
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
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                return False, f"gpg --verify (inline) failed: {exc}", None

            ok = completed.returncode == 0
            status_fd_output = completed.stdout or ""
            signing_fpr = _extract_validsig_fingerprint(status_fd_output) if ok else None
            combined = (completed.stderr or "") + status_fd_output
            detail = combined.strip() or f"rc={completed.returncode}"
            return ok, "inline-armor: " + detail, signing_fpr

    # Inline not found — check for sibling .asc.
    sibling = coi_path.with_suffix(coi_path.suffix + ".asc")
    if sibling.is_file():
        cmd = ["gpg", "--batch", "--status-fd=1"]
        if keyring_dir is not None:
            cmd.extend(["--homedir", str(keyring_dir)])
        cmd.extend(["--verify", str(sibling), str(coi_path)])

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return False, f"gpg --verify (sibling .asc) failed: {exc}", None

        ok = completed.returncode == 0
        status_fd_output = completed.stdout or ""
        signing_fpr = _extract_validsig_fingerprint(status_fd_output) if ok else None
        combined = (completed.stderr or "") + status_fd_output
        detail = combined.strip() or f"rc={completed.returncode}"
        return ok, "sibling-asc: " + detail, signing_fpr

    return None, "no inline PGP block AND no sibling <coi>.asc detached signature", None


def check_coi_body_signature_verifies(
    doc_text: str,
    repo_root: Path,
    keyring_dir: Optional[Path] = None,
) -> CheckResult:
    """H4: cryptographically verify the CoI declaration body.

    The sign-off doc points at a CoI declaration via the
    ``- **CoI document:** <path>`` field (extracted by ``extract_coi_path``).
    This check resolves that path under ``repo_root`` and runs
    ``gpg --verify`` against either:

    * an inline ASCII-armor signature block embedded in the CoI body
      (the operator-friendly default), OR
    * a sibling ``<coi>.asc`` detached signature file.

    Behavior matrix (matches ``check_keyring_present`` semantics):

    * Path missing / unreadable → FAIL with detail (the H4 sub-check 3 in
      ``check_coi_referenced`` already covers SHA resolution; this check
      additionally requires the body to be readable from the working
      tree). Failing the existing ``coi_referenced`` check already blocks
      the PR; this is defense-in-depth.
    * Path readable but no signature found → ADVISORY PASS in default
      mode (mirrors the sign-off doc's ``--require-signature=False`` path
      so existing CoI declarations without sigs still validate locally).
      The caller (main()) escalates to FAIL under STRICT_GPG=1.
    * Path readable AND signature present AND verifies → PASS.
    * Path readable AND signature present AND does NOT verify → FAIL.

    The ``signature_check_skipped`` flag is set when the advisory-pass
    branch fires (no signature found): STRICT_GPG=1 callers fail-closed
    on missing CoI signatures the same way STRICT_GH=1 fails on missing
    gh provenance. Distinct from ``provenance_check_skipped`` so log
    scrapers can distinguish the two failure classes.
    """

    coi_path_str = extract_coi_path(doc_text)
    if coi_path_str is None or "<github_handle>" in coi_path_str or coi_path_str == "":
        return CheckResult(
            "coi_body_signature_verifies",
            False,
            f"CoI path field missing or placeholder: {coi_path_str!r}",
        )

    coi_full = repo_root / coi_path_str
    if not coi_full.is_file():
        return CheckResult(
            "coi_body_signature_verifies",
            False,
            f"CoI body not found at resolved path: {coi_full}",
        )

    # Codex pass-6 MED-1 fix: mirror the keyring-advisory preflight from
    # check_signature_verifies for the CoI path. If a CoI signature is
    # present BUT the explicit --keyring-dir is missing/empty/unreadable,
    # return advisory PASS + signature_check_skipped=True so STRICT_GPG=1
    # routes to exit code 4 (NOT generic exit 1) AND the documented
    # `strict_gpg: '0'` rollout escape hatch works for already-signed
    # CoIs during the operator-handoff window.
    #
    # The preflight ONLY runs when (a) keyring_dir was explicitly set
    # (None means local-dev advisory) AND (b) gpg is on PATH AND (c)
    # the CoI body / sibling-asc HAS a signature to verify (no point
    # advisory-passing on an unsigned CoI — that's a different sig-skip
    # surface handled by the `ok is None` branch below).
    has_coi_signature = (
        _COI_INLINE_SIG_PATTERN.search(coi_full.read_text(encoding="utf-8")) is not None
        or coi_full.with_suffix(coi_full.suffix + ".asc").is_file()
    )
    if keyring_dir is not None and shutil.which("gpg") is not None and has_coi_signature:
        kr_ok, _kr_n, kr_detail = _gpg_list_keys_in_keyring(keyring_dir)
        if not kr_ok:
            return CheckResult(
                "coi_body_signature_verifies",
                True,
                f"WARN: CoI signature present but keyring at {keyring_dir} "
                f"is missing/empty/unreadable ({kr_detail}); H4 advisory mode "
                "(STRICT_GPG=1 escalates via signature_check_skipped)",
                signature_check_skipped=True,
            )

    ok, detail, signing_fpr = _verify_coi_body_signature(coi_full, keyring_dir=keyring_dir)
    if ok is True:
        fpr_suffix = f" [signing_fpr={signing_fpr}]" if signing_fpr else ""
        return CheckResult(
            "coi_body_signature_verifies",
            True,
            f"gpg --verify OK on CoI body: {detail[:200]}{fpr_suffix}",
            signing_fingerprint=signing_fpr,
        )
    if ok is False:
        return CheckResult(
            "coi_body_signature_verifies",
            False,
            f"gpg --verify FAILED on CoI body: {detail[:500]}",
        )
    # ok is None → no signature found; advisory PASS but flag it so
    # STRICT_GPG callers fail-closed.
    return CheckResult(
        "coi_body_signature_verifies",
        True,
        f"WARN: no CoI body signature found ({detail[:200]}); H4 advisory mode",
        signature_check_skipped=True,
    )


def check_signing_fingerprint_matches_registry(
    doc_text: str,
    registry: Sequence[ReviewerInfo],
    signature_results: Sequence[CheckResult],
) -> CheckResult:
    """Issue #226 codex pass-1 HIGH-1: bind verified signatures to a
    registry-pinned reviewer fingerprint.

    Ensures that every successful signature verification (sign-off doc
    AND CoI body) was made by the GPG key whose fingerprint is the
    registered fingerprint for the sign-off's reviewer handle. Without
    this binding, ``check_signature_verifies`` only proves the
    signature was made by SOME key in ``$KEYRING_DIR`` — but the
    keyring contains every reviewer's pubkey, so reviewer A's key
    would verify a sign-off from reviewer B and the validator would
    accept it.

    Behavior matrix:

    * No verify check produced a signing fingerprint (both either
      failed or skipped) → ADVISORY PASS with
      ``signature_check_skipped=True``. Pinning has nothing to evaluate.
      STRICT_GPG=1 callers fail-closed downstream via the existing
      sig-skip escalation.
    * Reviewer handle missing OR not in registry → FAIL (the
      orchestrator already evaluates ``reviewer_registered`` upstream;
      this is defense-in-depth).
    * Reviewer registered but ``fingerprint`` cell is empty (not yet
      operator-populated; placeholder normalized to "") → ADVISORY
      PASS with ``signature_check_skipped=True``. Same operator-
      progressive-rollout semantics as missing keyring.
    * Reviewer registered AND fingerprint pinned AND every verify-
      check signing fingerprint matches → PASS.
    * Any verify-check signing fingerprint does NOT match registered
      fingerprint → FAIL with explicit detail listing which check
      mismatched.

    The check is intentionally evaluated AFTER the verify checks so it
    has access to their signing-fingerprint outputs via
    ``signature_results``. The orchestrator passes the full results
    list; we filter to verify checks by name.
    """

    handle = extract_handle(doc_text)
    if handle is None:
        return CheckResult(
            "signing_fingerprint_matches_registry",
            False,
            "cannot evaluate fingerprint pinning without reviewer handle",
        )

    matching = [row for row in registry if row.handle == handle]
    if not matching:
        return CheckResult(
            "signing_fingerprint_matches_registry",
            False,
            f"reviewer {handle!r} not in registry — cannot resolve pinned fingerprint",
        )
    # Codex pass-3 MED-2 fix: filter to ACTIVE rows only. A stale
    # fingerprint left on an inactive/recused row would otherwise
    # satisfy pinning even after the reviewer rotated keys — this
    # weakens recusal AND key-rotation semantics. Per the registry
    # contract ("only `active` rows are eligible"), inactive/recused
    # rows MUST NOT contribute to the pinning fingerprint set.
    active_matching = [r for r in matching if r.status == "active"]
    if not active_matching:
        return CheckResult(
            "signing_fingerprint_matches_registry",
            False,
            f"reviewer {handle!r} has no active registry rows — pinning unsatisfiable",
        )
    # Codex pass-2 MED-1 + pass-3 MED-2 fix: aggregate fingerprints
    # across ALL ACTIVE registry rows for the handle (not just
    # matching[0], not all rows). The H1 pinning model is "any ACTIVE
    # row's registered fingerprint MAY match" so reviewers can rotate
    # keys without deleting the prior row — but rotated-AWAY keys
    # belong on inactive rows, NOT active ones, and inactive rows
    # don't contribute. Empty fingerprints are filtered out; at
    # least one non-empty registered fingerprint on an active row
    # is required for pinning to be evaluable.
    registered_fprs: list[str] = []
    for r in active_matching:
        if r.fingerprint and r.fingerprint not in registered_fprs:
            registered_fprs.append(r.fingerprint)
    # Single canonical fingerprint for log/detail messages: prefer the
    # first non-empty one, else empty string.
    registered_fpr = registered_fprs[0] if registered_fprs else ""

    # Codex pass-2 HIGH-2 fix: collect ALL successful verify-check
    # results (ok=True) regardless of whether they populated
    # signing_fingerprint. A verify-OK result with signing_fingerprint=None
    # (e.g. the sigstore code path which doesn't produce a GPG
    # fingerprint, OR a future gpg version that drops VALIDSIG) is a
    # PINNING GAP — we cannot bind that successful verification to a
    # registered reviewer. Without this, a sigstore-verified sign-off
    # PLUS a gpg-verified-and-pinned CoI body would falsely satisfy
    # the check even though the SIGN-OFF artifact was never bound to
    # the reviewer's identity.
    verify_check_names = ("signature_verifies", "coi_body_signature_verifies")
    successful_verify_results = [
        r for r in signature_results if r.name in verify_check_names and r.ok
    ]

    if not successful_verify_results:
        # No verify check passed (both either failed OR were skipped).
        # Pinning has nothing to evaluate; flag for STRICT_GPG=1
        # escalation via the existing sig-skip path.
        return CheckResult(
            "signing_fingerprint_matches_registry",
            True,
            f"WARN: no successful signature verification for {handle} "
            "(both verify checks failed/skipped); pinning skipped",
            signature_check_skipped=True,
        )

    # Separate verify-OK results into pinned (signing_fingerprint set)
    # and unpinned (signing_fingerprint=None). Codex pass-2 HIGH-2:
    # any unpinned-but-successful verify is a pinning gap that
    # STRICT_GPG=1 MUST escalate.
    pinned_results = [r for r in successful_verify_results if r.signing_fingerprint is not None]
    unpinned_results = [r for r in successful_verify_results if r.signing_fingerprint is None]

    if not registered_fpr:
        # Reviewer is registered but the fingerprint cell is empty
        # (operator hasn't completed the H1 handoff). ADVISORY PASS
        # with sig-skip flag so STRICT_GPG=1 fails closed.
        seen_fprs = [r.signing_fingerprint for r in pinned_results]
        return CheckResult(
            "signing_fingerprint_matches_registry",
            True,
            f"WARN: registered fingerprint for {handle} is empty (placeholder); "
            f"signing fingerprints seen: {seen_fprs} — "
            "operator must populate the fingerprint column",
            signature_check_skipped=True,
        )

    # Pinned results MUST match ONE OF the registered fingerprints
    # (codex pass-2 MED-1: a handle may have multiple active rows
    # encoding key-rotation history; ANY active row's fingerprint is
    # acceptable). The set is canonical-uppercase already (parsed via
    # _normalize_fingerprint).
    registered_fpr_set = set(registered_fprs)
    mismatches = [
        (r.name, r.signing_fingerprint)
        for r in pinned_results
        if r.signing_fingerprint not in registered_fpr_set
    ]
    if mismatches:
        mismatch_strs = [f"{name}: signed_by={fpr}" for name, fpr in mismatches]
        return CheckResult(
            "signing_fingerprint_matches_registry",
            False,
            f"signing fingerprint(s) do not match any registered fingerprint for "
            f"{handle} (registered={sorted(registered_fpr_set)}). Mismatches: "
            f"{'; '.join(mismatch_strs)}",
        )

    # Codex pass-2 HIGH-2: any unpinned-but-successful verify is a
    # pinning gap. ADVISORY PASS with sig-skip; STRICT_GPG=1 escalates.
    if unpinned_results:
        unpinned_names = [r.name for r in unpinned_results]
        return CheckResult(
            "signing_fingerprint_matches_registry",
            True,
            f"WARN: {len(pinned_results)} verify-check(s) pinned to "
            f"{registered_fpr} but {len(unpinned_results)} unpinned successful "
            f"verify(s) cannot be bound to a registered fingerprint: "
            f"{unpinned_names}. Likely cause: sigstore code path doesn't emit "
            "GNUPG VALIDSIG. Pinning gap; STRICT_GPG=1 escalates.",
            signature_check_skipped=True,
        )

    return CheckResult(
        "signing_fingerprint_matches_registry",
        True,
        f"all {len(pinned_results)} verify-check signing fingerprint(s) match "
        f"registered fingerprint for {handle} ({registered_fpr})",
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
        registry, parser_warnings = parse_registry_with_warnings(registry_path)
    except FileNotFoundError as exc:
        results.append(CheckResult("registry_loaded", False, str(exc)))
        return results

    # Codex pass-5 MED-1 fix: surface registry parser warnings as a
    # registry_loaded FAILURE, NOT silent skip. Malformed rows that
    # look like table-body rows but have wrong column counts could
    # silently DROP disqualifying-evidence rows from selection-rule
    # aggregation, defeating the pass-4 HIGH-1 fix. Treat parser
    # warnings as a hard fail so operators MUST fix the registry
    # syntax before the validator accepts the sign-off.
    if parser_warnings:
        results.append(
            CheckResult(
                "registry_loaded",
                False,
                f"registry has {len(parser_warnings)} malformed row(s) that "
                f"would silently drop selection-rule evidence: "
                f"{'; '.join(parser_warnings)[:600]}",
            )
        )
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
    # Issue #226 H1: keyring presence pre-check. Always runs so the report
    # surfaces keyring state even in advisory mode; STRICT_GPG=1 escalates
    # a missing keyring to exit code 4 in main() via signature_check_skipped.
    results.append(check_keyring_present(keyring_dir))
    # Issue #226 H4: CoI body signature verification. Runs unconditionally
    # so its result is in the report; advisory pass when no sig is found
    # (signature_check_skipped=True triggers fail-closed under STRICT_GPG=1).
    results.append(
        check_coi_body_signature_verifies(
            doc_text,
            repo_root=repo_root,
            keyring_dir=keyring_dir,
        )
    )
    # Issue #226 H1 codex pass-1 HIGH-1: bind verified signatures to the
    # registered reviewer fingerprint. MUST run AFTER the verify checks so
    # signing_fingerprint values are populated. Without this, a verify-OK
    # only proves "some key in the keyring signed it" — pinning is what
    # makes the keyring + registry into reviewer-identity binding.
    results.append(
        check_signing_fingerprint_matches_registry(
            doc_text,
            registry,
            results,
        )
    )
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
            "Path to a GPG home directory containing the trusted public "
            "keys for sign-off reviewers. Passed to gpg via --homedir for "
            "verification AND consulted by check_keyring_present + the H1 "
            "fingerprint-pinning check. If UNSET, the keyring_present "
            "check returns advisory pass with signature_check_skipped=True "
            "(STRICT_GPG=1 escalates to exit 4); the system default "
            "keyring is NOT consulted (a deliberate change vs pre-#226 — "
            "the H1 model requires explicit registry-pinned keyring "
            "binding, not implicit reliance on whatever keys happen to be "
            "in the user's $GNUPGHOME). Local devs running ad-hoc without "
            "STRICT_GPG=1 retain advisory back-compat."
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
    parser.add_argument(
        "--strict-gh",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fail-closed when any CheckResult has provenance_check_skipped=True "
            "(i.e. gh CLI was unavailable / unauthenticated, so PR/review "
            "provenance was NOT confirmed). The validator returns exit code 3 "
            "in that case. Defaults to OFF for back-compat with local dev "
            "runs; CI workflows that provision GH_TOKEN MUST set this flag "
            "(or export STRICT_GH=1, which the CLI also honors). Use "
            "--no-strict-gh to override STRICT_GH=1 from the env. Issue "
            "#192 H2/M1 fail-closed enforcement."
        ),
    )
    parser.add_argument(
        "--strict-gpg",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fail-closed when (a) the keyring directory passed via "
            "--keyring-dir is missing/empty/unreadable OR (b) any check "
            "has signature_check_skipped=True (no CoI body signature "
            "found OR pinning gap). The validator returns exit code 4 in "
            "that case. Defaults to OFF for back-compat with local dev "
            "runs; CI workflows that provision the keyring MUST set this "
            "flag (or export STRICT_GPG=1, which the CLI also honors). "
            "Use --no-strict-gpg to override STRICT_GPG=1 from the env. "
            "Issue #226 H1+H4 fail-closed enforcement."
        ),
    )
    return parser


def _resolve_strict_gh(cli_flag: Optional[bool]) -> bool:
    """Resolve the strict-gh policy: CLI flag wins; falls back to env var.

    Issue #192 H2/M1: when neither the CLI flag nor the env var is set, the
    validator preserves the historical warn-only behavior (back-compat for
    local devs running the script ad-hoc). CI workflows that provision
    GH_TOKEN MUST set ``--strict-gh`` (or export ``STRICT_GH=1``) so a
    skipped gh provenance query becomes a hard PR block (exit code 3)
    rather than a logged warning.

    Codex pass-5 LOW-1: CLI flag now uses ``BooleanOptionalAction`` so
    ``--strict-gh`` → True, ``--no-strict-gh`` → False, omitted → None.
    An explicit ``--no-strict-gh`` overrides ``STRICT_GH=1`` in env.
    Truthy env values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else (including absent / empty) defaults to OFF.
    """

    if cli_flag is True:
        return True
    if cli_flag is False:
        # Pass-5 LOW-1: explicit `--no-strict-gh` overrides STRICT_GH=1.
        return False
    raw = os.environ.get("STRICT_GH", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _resolve_strict_gpg(cli_flag: Optional[bool]) -> bool:
    """Resolve the strict-gpg policy: CLI flag wins; falls back to env var.

    Issue #226 H1+H4: mirrors ``_resolve_strict_gh`` semantics. When
    neither the CLI flag nor the env var is set, the validator preserves
    the historical advisory behavior (keyring-missing and CoI-sig-missing
    are PASS-with-WARN). CI workflows that provision the keyring via the
    ``GPG_REVIEWER_KEYS_ARMOR_BASE64`` secret MUST set ``--strict-gpg``
    (or export ``STRICT_GPG=1``) so missing keyring / missing CoI body
    signature become hard PR blocks (exit code 4) rather than warnings.

    Codex pass-5 LOW-1: CLI flag now uses ``BooleanOptionalAction`` so
    ``--strict-gpg`` → True, ``--no-strict-gpg`` → False, omitted → None.
    An explicit ``--no-strict-gpg`` overrides ``STRICT_GPG=1`` in env.
    Truthy env values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else (including absent / empty) defaults to OFF.
    """

    if cli_flag is True:
        return True
    if cli_flag is False:
        # Pass-5 LOW-1: explicit `--no-strict-gpg` overrides STRICT_GPG=1.
        return False
    raw = os.environ.get("STRICT_GPG", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


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

    # Generic validation failure path takes precedence over strict-gh AND
    # strict-gpg — a selection-rule violation (ok=False) is exit 1 even if
    # STRICT_GH=1 / STRICT_GPG=1 also would have triggered. This preserves
    # the existing exit-code contract for the dominant failure mode and
    # reserves exits 3 + 4 specifically for "everything else passed but
    # provenance / keyring was not confirmed under strict mode" so log
    # scrapers can distinguish the failure classes.
    if not all(r.ok for r in results):
        return 1

    strict_gh = _resolve_strict_gh(args.strict_gh)
    if strict_gh and any(r.provenance_check_skipped for r in results):
        skipped = [r.name for r in results if r.provenance_check_skipped]
        print(
            "FAIL: --strict-gh policy is in effect (or STRICT_GH=1 in env) "
            "AND at least one check has provenance_check_skipped=True. "
            f"Skipped checks: {', '.join(skipped)}. "
            "Provision GH_TOKEN with pull-requests:read on the runner to "
            "satisfy gh PR/review provenance queries. "
            "See docs/governance/n3_known_limitations_20260510.md item 2.",
            file=sys.stderr,
        )
        return 3

    # Issue #226 H1+H4 strict-gpg policy: exit 4 when either (a) the
    # keyring pre-check would have failed but for advisory mode (we ran
    # the full check above so a missing keyring already shows as ok=False
    # and would have been caught by the exit-1 branch — but check
    # ``signature_check_skipped`` for the H4 advisory-pass branch), OR
    # (b) the CoI body sig check fired the advisory-pass branch
    # (signature_check_skipped=True). Exit 4 is reserved to distinguish
    # "keyring/CoI sig infra not provisioned" from generic validation
    # failures so log scrapers can route the two failure classes.
    strict_gpg = _resolve_strict_gpg(args.strict_gpg)
    if strict_gpg and any(r.signature_check_skipped for r in results):
        skipped = [r.name for r in results if r.signature_check_skipped]
        # Codex pass-3 LOW-1 fix: enumerate the failure subclasses so
        # operators / log scrapers can route the three distinct cases
        # under the same exit code:
        #   (a) keyring_present skipped → keyring not provisioned
        #   (b) signature_verifies skipped → keyring missing for sign-off
        #       OR coi_body_signature_verifies skipped → no CoI sig
        #   (c) signing_fingerprint_matches_registry skipped → PINNING
        #       GAP (verify succeeded BUT cannot bind to a registered
        #       reviewer; e.g. sigstore code path or empty fingerprint
        #       column placeholder)
        # Exit code 4 is reserved for "STRICT_GPG=1 + any sig-skip"
        # generally; the routing distinction is the check name list.
        keyring_skips = [n for n in skipped if n in ("keyring_present", "signature_verifies")]
        coi_skips = [n for n in skipped if n == "coi_body_signature_verifies"]
        pinning_skips = [n for n in skipped if n == "signing_fingerprint_matches_registry"]
        subclass_msgs = []
        if keyring_skips:
            subclass_msgs.append(
                f"KEYRING/SIG NOT PROVISIONED ({', '.join(keyring_skips)}) — "
                "provision GPG_REVIEWER_KEYS_ARMOR_BASE64 secret"
            )
        if coi_skips:
            subclass_msgs.append(
                f"COI BODY SIGNATURE MISSING ({', '.join(coi_skips)}) — "
                "ensure CoI carries inline armor OR sibling <coi>.asc"
            )
        if pinning_skips:
            subclass_msgs.append(
                f"FINGERPRINT PINNING GAP ({', '.join(pinning_skips)}) — "
                "verify succeeded BUT cannot bind to a registered reviewer; "
                "populate fingerprint column OR disable sigstore for methodology sign-offs"
            )
        print(
            "FAIL: --strict-gpg policy is in effect (or STRICT_GPG=1 in env) "
            "AND at least one check has signature_check_skipped=True. "
            f"Skipped checks: {', '.join(skipped)}. "
            f"Subclasses: {' | '.join(subclass_msgs) if subclass_msgs else 'unknown'}. "
            "See docs/governance/operator_gpg_keyring_setup.md and "
            "docs/governance/n3_known_limitations_20260510.md items 1+4.",
            file=sys.stderr,
        )
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())
