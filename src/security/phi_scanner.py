"""Deterministic PHI/PII scanner for crystal narratives + LLM-prompt audit.

Issue #391 security box 4: audit that no PHI/PII leaks into crystal
``key_finding`` text via LLM summarization. The scanner is the building
block of ``scripts/audit_phi_in_crystal_narratives.py``.

Design choices
--------------
* Pure regex — no LLM / ML. Reproducibility is the security contract:
  the same input MUST produce the same matches across runs, OS, and
  Python versions. ML-based PHI detection (Presidio, etc.) is rejected
  here because non-determinism makes audit reports unreviewable.
* Each pattern has a short stable ``pattern_name`` used as the key in
  audit reports. Renaming a pattern is a breaking change.
* Patterns are tuned to minimize false positives (see negative test
  fixtures in ``tests/unit/test_security/test_phi_scanner.py``). When a
  pattern would over-match (e.g. a bare 7-digit number) we require
  labeling context (e.g. ``MRN:``) so the audit doesn't drown in
  inventory IDs.
* PHI != PII in the strict HIPAA sense — this scanner deliberately
  catches BOTH because the audit harness's job is to surface ANY
  identifier shape that could leak through the LLM narrator. False
  positives on author emails in narrative prose are an acceptable
  trade-off vs missing a patient email.

Patterns (per #391 brief)
-------------------------
* ``ssn``: ``\\b\\d{3}-\\d{2}-\\d{4}\\b``
* ``us_phone``: ``\\(\\d{3}\\)\\s*\\d{3}-\\d{4}`` OR ``\\b\\d{3}-\\d{3}-\\d{4}\\b``
* ``dob``: ``\\b(0[1-9]|1[0-2])/(0[1-9]|[12]\\d|3[01])/(19|20)\\d{2}\\b``
* ``email``: ``\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b`` (patient-context-scoped — see below)
* ``mrn``: ``\\b(?:MRN|Medical Record Number)\\s*[:#]?\\s*\\d{6,12}\\b``

Email patient-context scoping (codex iter-2 M3)
-----------------------------------------------
The email regex matches `local@domain.tld`. Without scoping it would
flag every email-shaped string in a narrative — operational addresses
(`support@example.com`, `noreply@internal`), author signatures, doc
links. That produces false-positive deploy blocks: the audit harness
exits non-zero on any match, so a single ops email in a narrative kills
the whole pre-deploy gate.

The codex iter-2 M3 fix scopes email matches to a "patient context
window": an email only emits a ``PhiMatch`` if a patient-shape token
(``patient``, ``mrn``, ``dob``, ``subject``, ``name``, ``ssn``) appears
within :data:`_EMAIL_PATIENT_CONTEXT_WINDOW` characters BEFORE or AFTER
the match span. This is a deterministic heuristic — same input, same
verdict. The window value (default 30) is small enough to be a meaningful
locality constraint (the patient-shape token has to be in the same
clause / sentence as the email) and large enough to tolerate normal
prose ("Patient John alice@example.com").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple


@dataclass(frozen=True)
class PhiMatch:
    """One hit emitted by :func:`scan_text`.

    Attributes:
        pattern_name: Stable short name of the matching pattern (e.g.
            ``"ssn"``, ``"us_phone"``). Used as a stable key in audit
            reports — renaming is a breaking change.
        match: The substring that matched.
        start: 0-based inclusive byte offset into the source text.
        end: 0-based exclusive byte offset; ``text[start:end] == match``.
    """

    pattern_name: str
    match: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Email patient-context scoping (codex iter-2 M3 closure)
# ---------------------------------------------------------------------------
# Window (in characters) on either side of a candidate email match. If
# any of the patient-shape tokens below appear within this window, the
# match is emitted as a ``email`` PhiMatch. Otherwise it is suppressed
# (operational / non-patient email).
_EMAIL_PATIENT_CONTEXT_WINDOW = 30

# Patient-shape tokens that promote a generic email match to a patient
# email. Matched case-INSENSITIVE in the surrounding window. The set is
# small + audit-reviewable; expand only when a real false negative
# surfaces.
_PATIENT_CONTEXT_TOKENS = frozenset(
    {
        "patient",
        "mrn",
        "dob",
        "subject",  # clinical-trial vocabulary for an enrolled person
        "name",  # "Patient name: ..." or "Name: ..."
        "ssn",
    }
)


def _in_patient_context(text: str, match_start: int, match_end: int) -> bool:
    """Return True iff a patient-shape token appears within
    :data:`_EMAIL_PATIENT_CONTEXT_WINDOW` characters BEFORE or AFTER
    the ``[match_start:match_end]`` span.

    Deterministic: same input + span → same verdict. No state, no
    randomness, no order-dependence. The window is fixed at module
    scope.

    Edge cases:
    * If the window extends past either end of ``text``, only the
      in-bounds portion is checked (no IndexError).
    * Case-insensitive — ``Patient``, ``patient``, ``PATIENT`` all
      qualify.
    * Whole-word matching: ``namespace`` does NOT activate ``name``.
      We surround tokens with non-alphanumeric boundaries to avoid
      false positives on prose like ``namespace`` / ``patiently``.
    """
    if not text:
        return False
    win_start = max(0, match_start - _EMAIL_PATIENT_CONTEXT_WINDOW)
    win_end = min(len(text), match_end + _EMAIL_PATIENT_CONTEXT_WINDOW)
    window = text[win_start:win_end].lower()
    # Whole-word check: scan for each token surrounded by non-word
    # boundaries. ``re.search(r"\bpatient\b", window)`` is the canonical
    # form; we precompile per-call (cheap — _PATIENT_CONTEXT_TOKENS is
    # 6 items) for clarity.
    for token in _PATIENT_CONTEXT_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", window):
            return True
    return False


# Order matters for documentation and report stability, NOT for correctness:
# each pattern is independently applied to the full input. Ordering this list
# determines the order matches appear in a report when multiple patterns
# match the same input.
_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    # SSN: NNN-NN-NNNN. \b anchors avoid bleeding into longer digit runs
    # like ID 1234-5678-9012.
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # US phone — two accepted forms: (NNN) NNN-NNNN OR NNN-NNN-NNNN.
    # The alternation is one combined pattern so a single match emits
    # one ``us_phone`` PhiMatch (not two).
    (
        "us_phone",
        re.compile(r"\(\d{3}\)\s*\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b"),
    ),
    # DOB: MM/DD/YYYY with month 01-12, day 01-31, year 19xx or 20xx.
    # The regex is intentionally lenient on day-vs-month combinations
    # (does not enforce February 29 etc.) — the audit harness needs to
    # surface anything that LOOKS like a DOB; calendar validation would
    # miss real DOBs in the LLM output.
    (
        "dob",
        re.compile(r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(19|20)\d{2}\b"),
    ),
    # Email: standard local@domain.tld. The character class for the
    # local part includes ``+`` for tagged addresses; the domain part
    # requires at least one dot.
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # MRN: requires the literal label ``MRN`` or ``Medical Record
    # Number`` to avoid flagging every 6-12 digit number in a narrative
    # (cohort sizes, study IDs, etc). 6-12 digit range covers typical
    # EHR MRN lengths. ``[:#]?`` accepts ``MRN: 123456``, ``MRN #123456``,
    # ``MRN 123456`` — common shapes seen in clinical text.
    (
        "mrn",
        re.compile(
            r"\b(?:MRN|Medical Record Number)\s*[:#]?\s*\d{6,12}\b",
            re.IGNORECASE,
        ),
    ),
]


def scan_text(text: str) -> List[PhiMatch]:
    """Scan ``text`` for PHI/PII patterns. Deterministic.

    Returns matches in pattern-order (first by the order in
    :data:`_PATTERNS`, then by start offset within a pattern). Two calls
    with the same input return the same list — this is asserted in
    ``test_scan_text_deterministic_repeatable``.

    Edge cases:
    * Empty / whitespace-only / non-string-with-no-matches input returns ``[]``.
    * Unicode whitespace and non-ASCII characters in the input do NOT
      crash; they simply don't match any of the ASCII-anchored regexes.

    Email patient-context scoping (codex iter-2 M3):
    * ``email`` candidates are only emitted as ``PhiMatch`` if a
      patient-shape token (``patient``, ``mrn``, ``dob``, ``subject``,
      ``name``, ``ssn``) appears within
      :data:`_EMAIL_PATIENT_CONTEXT_WINDOW` chars before/after the
      match span. Otherwise the email is treated as operational and
      suppressed. This prevents the audit harness from blocking
      deploys on every ``support@example.com`` in a narrative.

    Args:
        text: The body to scan. Typically a crystal ``key_finding`` or
            an LLM input prompt.

    Returns:
        List of :class:`PhiMatch` records (possibly empty).
    """
    if not text:
        return []
    matches: List[PhiMatch] = []
    for pattern_name, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            # Email-only: suppress matches outside the patient context
            # window. Other patterns (SSN, phone, DOB, MRN) emit on
            # any hit because they're already context-bounded by their
            # own regex (MRN requires the label, SSN requires the
            # exact dash shape, etc.).
            if pattern_name == "email" and not _in_patient_context(text, m.start(), m.end()):
                continue
            matches.append(
                PhiMatch(
                    pattern_name=pattern_name,
                    match=m.group(0),
                    start=m.start(),
                    end=m.end(),
                )
            )
    return matches
