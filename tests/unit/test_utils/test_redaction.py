"""Tests for redact_query — the SSOT query-text log/telemetry redactor (#1367).

~20 call sites hand-rolled ``query[:N]`` slices to keep free-text pharma-KPI
queries out of logs; one site logged the full query untruncated, and a second
logged it untruncated at INFO. redact_query centralizes the truncation length
(and any future PII scrubbing) into one knob. These tests pin the None-safety,
the no-marker-when-short rule, the truncation marker, and byte-for-byte
compatibility with the ``query[:N] + "..."`` idiom the call sites replaced.
"""

from src.utils.redaction import redact_query


class TestRedactQuery:
    def test_none_returns_empty(self):
        assert redact_query(None) == ""

    def test_empty_returns_empty(self):
        assert redact_query("") == ""

    def test_short_query_unchanged_no_marker(self):
        assert redact_query("what is TRx for brand X") == "what is TRx for brand X"

    def test_exactly_max_len_unchanged_no_marker(self):
        s = "a" * 50
        assert redact_query(s) == s

    def test_one_over_max_len_truncated_with_marker(self):
        s = "a" * 51
        assert redact_query(s) == "a" * 50 + "..."

    def test_long_query_truncated_with_marker(self):
        s = "a" * 500
        assert redact_query(s) == "a" * 50 + "..."

    def test_max_len_override_truncates_at_override(self):
        assert redact_query("a" * 120, max_len=100) == "a" * 100 + "..."

    def test_max_len_override_short_query_unchanged(self):
        assert redact_query("hello", max_len=80) == "hello"

    def test_whitespace_under_cap_preserved(self):
        assert redact_query("   ") == "   "

    def test_matches_legacy_slice_idiom_default(self):
        # The idiom these sites replaced on the truncation branch: query[:50] + "..."
        q = "brand x hcp engagement impact on patient conversions " * 5
        assert redact_query(q) == q[:50] + "..."

    def test_matches_legacy_slice_idiom_max_len_100(self):
        q = "brand x hcp engagement impact on patient conversions " * 5
        assert redact_query(q, max_len=100) == q[:100] + "..."

    def test_returns_str_type(self):
        assert isinstance(redact_query(None), str)
        assert isinstance(redact_query("a" * 100), str)
