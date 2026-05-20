"""PHI/PII scanner unit tests (#391 security box 4).

The scanner is deterministic regex-based — no LLM / ML — so the same
input always produces the same matches. This is the cornerstone of the
audit harness at ``scripts/audit_phi_in_crystal_narratives.py``.

Patterns covered (per #391 brief):
* SSN: ``\\b\\d{3}-\\d{2}-\\d{4}\\b``
* US phone: ``\\(\\d{3}\\)\\s*\\d{3}-\\d{4}`` or ``\\d{3}-\\d{3}-\\d{4}``
* DOB: ``\\b(0[1-9]|1[0-2])/(0[1-9]|[12]\\d|3[01])/(19|20)\\d{2}\\b``
* email: ``\\b[\\w.+-]+@[\\w-]+\\.[\\w.-]+\\b``
* MRN context: ``\\b(?:MRN|Medical Record Number)\\s*[:#]?\\s*\\d{6,12}\\b``

Test classes mirror the pattern names so a failure isolates which pattern
broke. Each pattern has BOTH a true-positive and a true-negative fixture
to guard against over-matching.
"""

from __future__ import annotations

import pytest

from src.security.phi_scanner import PhiMatch, scan_text

# ---------------------------------------------------------------------------
# Top-level API shape
# ---------------------------------------------------------------------------


def test_scan_text_empty_string_returns_empty_list() -> None:
    """Edge case: empty input must not crash and must return []."""
    assert scan_text("") == []


def test_scan_text_no_phi_returns_empty_list() -> None:
    """Plain text with no patterns matched returns []."""
    text = "The patient responded well to treatment over the observation window."
    assert scan_text(text) == []


def test_scan_text_returns_phimatch_dataclass() -> None:
    """Each hit is a :class:`PhiMatch` with the four expected fields."""
    text = "ssn=123-45-6789"
    matches = scan_text(text)
    assert len(matches) == 1
    m = matches[0]
    assert isinstance(m, PhiMatch)
    assert m.pattern_name == "ssn"
    assert m.match == "123-45-6789"
    assert m.start == text.index("123-45-6789")
    assert m.end == m.start + len("123-45-6789")


def test_scan_text_unicode_does_not_crash() -> None:
    """Unicode whitespace + non-ASCII chars must not raise."""
    text = "patient record — no PHI here"
    assert scan_text(text) == []


def test_scan_text_whitespace_only_returns_empty_list() -> None:
    """Whitespace-only input returns []."""
    assert scan_text("   \n\t ") == []


# ---------------------------------------------------------------------------
# SSN
# ---------------------------------------------------------------------------


class TestSSN:
    def test_ssn_positive(self) -> None:
        """Canonical SSN format ``NNN-NN-NNNN`` is detected."""
        matches = scan_text("subject ssn 555-12-3456 was redacted")
        names = [m.pattern_name for m in matches]
        assert "ssn" in names
        ssn_hit = next(m for m in matches if m.pattern_name == "ssn")
        assert ssn_hit.match == "555-12-3456"

    def test_ssn_negative_part_number(self) -> None:
        """Random 9-digit number without SSN dash shape is NOT flagged as SSN."""
        # 999999999 has no dashes; should not match the SSN regex
        matches = scan_text("part number 999999999 inventory entry")
        names = [m.pattern_name for m in matches]
        assert "ssn" not in names

    def test_ssn_negative_wrong_grouping(self) -> None:
        """``4-4-4`` grouping is NOT an SSN."""
        matches = scan_text("ID 1234-5678-9012 inventory")
        names = [m.pattern_name for m in matches]
        assert "ssn" not in names


# ---------------------------------------------------------------------------
# US phone
# ---------------------------------------------------------------------------


class TestUSPhone:
    def test_phone_dash_format_positive(self) -> None:
        """``NNN-NNN-NNNN`` is detected as phone."""
        matches = scan_text("call 415-555-1212 for details")
        names = [m.pattern_name for m in matches]
        assert "us_phone" in names

    def test_phone_paren_format_positive(self) -> None:
        """``(NNN) NNN-NNNN`` is detected as phone."""
        matches = scan_text("contact (415) 555-1212 ASAP")
        names = [m.pattern_name for m in matches]
        assert "us_phone" in names

    def test_phone_negative_short_number(self) -> None:
        """Short numbers (e.g. an extension) are NOT flagged."""
        matches = scan_text("extension 1212 only")
        names = [m.pattern_name for m in matches]
        assert "us_phone" not in names


# ---------------------------------------------------------------------------
# DOB
# ---------------------------------------------------------------------------


class TestDOB:
    def test_dob_positive(self) -> None:
        """``MM/DD/YYYY`` in 19xx/20xx range is detected as DOB."""
        matches = scan_text("DOB 03/15/1987 enrolled")
        names = [m.pattern_name for m in matches]
        assert "dob" in names

    def test_dob_positive_2000s(self) -> None:
        """``MM/DD/YYYY`` in 20xx range also detected."""
        matches = scan_text("born 12/31/2005 in cohort")
        names = [m.pattern_name for m in matches]
        assert "dob" in names

    def test_dob_negative_invalid_month(self) -> None:
        """Invalid month (13/) is NOT a DOB."""
        matches = scan_text("code 13/15/1987 sequence")
        names = [m.pattern_name for m in matches]
        assert "dob" not in names

    def test_dob_negative_outside_year_range(self) -> None:
        """Year < 1900 is NOT flagged as DOB."""
        matches = scan_text("entry 03/15/1850 archival")
        names = [m.pattern_name for m in matches]
        assert "dob" not in names


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------


class TestEmail:
    def test_email_positive(self) -> None:
        """Standard email format is detected."""
        matches = scan_text("contact alice@example.com please")
        names = [m.pattern_name for m in matches]
        assert "email" in names
        email_hit = next(m for m in matches if m.pattern_name == "email")
        assert email_hit.match == "alice@example.com"

    def test_email_positive_with_plus(self) -> None:
        """Email with ``+`` tag is detected."""
        matches = scan_text("alias jane+filter@org.example.com active")
        names = [m.pattern_name for m in matches]
        assert "email" in names

    def test_email_negative_no_at_sign(self) -> None:
        """``alice.example.com`` (no @) is NOT an email."""
        matches = scan_text("url alice.example.com here")
        names = [m.pattern_name for m in matches]
        assert "email" not in names


# ---------------------------------------------------------------------------
# MRN
# ---------------------------------------------------------------------------


class TestMRN:
    def test_mrn_positive_with_label(self) -> None:
        """``MRN: NNNNNN`` is detected."""
        matches = scan_text("patient MRN: 123456 admitted")
        names = [m.pattern_name for m in matches]
        assert "mrn" in names

    def test_mrn_positive_with_hash(self) -> None:
        """``MRN #NNNNNNN`` is detected."""
        matches = scan_text("MRN #7654321 transferred")
        names = [m.pattern_name for m in matches]
        assert "mrn" in names

    def test_mrn_positive_full_label(self) -> None:
        """``Medical Record Number 1234567`` is detected."""
        matches = scan_text("Medical Record Number 1234567 referenced")
        names = [m.pattern_name for m in matches]
        assert "mrn" in names

    def test_mrn_negative_no_label(self) -> None:
        """A bare 7-digit number without MRN context is NOT flagged as MRN.

        (May still be flagged by a different pattern but never as MRN
        without the labeling context — that prevents over-matching.)
        """
        matches = scan_text("inventory count 1234567 last week")
        names = [m.pattern_name for m in matches]
        assert "mrn" not in names


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class TestComposition:
    def test_multiple_patterns_in_one_text(self) -> None:
        """All five patterns in one passage are all detected."""
        text = (
            "Subject: SSN 123-45-6789, phone (415) 555-1212, DOB 03/15/1987, "
            "MRN: 9876543, contact alice@example.com"
        )
        matches = scan_text(text)
        names = sorted({m.pattern_name for m in matches})
        assert "ssn" in names
        assert "us_phone" in names
        assert "dob" in names
        assert "mrn" in names
        assert "email" in names

    def test_deterministic_repeatable(self) -> None:
        """Two runs on the same input produce the same matches."""
        text = "DOB 03/15/1987 SSN 555-12-3456 email alice@example.com"
        matches_1 = scan_text(text)
        matches_2 = scan_text(text)
        assert [(m.pattern_name, m.match, m.start, m.end) for m in matches_1] == [
            (m.pattern_name, m.match, m.start, m.end) for m in matches_2
        ]


# ---------------------------------------------------------------------------
# Boundary contract: PhiMatch dataclass has the expected fields
# ---------------------------------------------------------------------------


def test_phimatch_has_required_fields() -> None:
    """PhiMatch has ``pattern_name``, ``match``, ``start``, ``end``."""
    m = PhiMatch(pattern_name="ssn", match="123-45-6789", start=0, end=11)
    assert m.pattern_name == "ssn"
    assert m.match == "123-45-6789"
    assert m.start == 0
    assert m.end == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
