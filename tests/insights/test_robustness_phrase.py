"""#1868: the shared gate-verdict phrase must be a function of the per-test
outcomes, not the gate band alone.

"Survived all robustness checks" was emitted for EVERY proceed gate — but
PROCEED deliberately tolerates a critical test in WARNING (and a non-critical
test can even be FAILED under a proceed gate), so the phrase over-claimed.
Observed live: Kisqali trigger-grain acceptance_status -> conversion_flag said
"survived all robustness checks" while the same payload carried
tests_passed: 2/3 (E-value CI bound 1.51 = warning).

The helper lives in a dspy-free module so route modules can import it eagerly
(clinical_narrative pulls dspy at import time and stays function-local).
"""

import pytest

from src.insights.robustness_phrase import gate_verdict_phrase

pytestmark = pytest.mark.unit


def _t(name, status=None, passed=None, details=None):
    d = {"test_name": name}
    if status is not None:
        d["status"] = status
    if passed is not None:
        d["passed"] = passed
    if details is not None:
        d["details"] = details
    return d


class TestProceedGate:
    def test_all_passed_says_survived_all_with_count(self):
        tests = [
            _t("placebo_treatment", status="passed", passed=True),
            _t("random_common_cause", status="passed", passed=True),
            _t("unobserved_common_cause", status="passed", passed=True),
        ]
        assert gate_verdict_phrase("proceed", tests) == "survived all 3 robustness checks"

    def test_warning_is_named_and_survived_all_is_not_claimed(self):
        tests = [
            _t("placebo_treatment", status="passed", passed=True),
            _t("random_common_cause", status="passed", passed=True),
            _t(
                "unobserved_common_cause",
                status="warning",
                passed=False,
                details="E-value (CI bound) 1.51 suggests moderate sensitivity to confounding",
            ),
        ]
        phrase = gate_verdict_phrase("proceed", tests)
        assert "survived all" not in phrase
        assert "2 of 3" in phrase
        assert "unmeasured-confounding sensitivity" in phrase
        assert "raised a warning" in phrase

    def test_noncritical_failure_is_named_as_failed_but_not_gating(self):
        tests = [
            _t("placebo_treatment", status="passed", passed=True),
            _t("random_common_cause", status="passed", passed=True),
            _t("unobserved_common_cause", status="passed", passed=True),
            _t("data_subset", status="failed", passed=False),
        ]
        phrase = gate_verdict_phrase("proceed", tests)
        assert "survived all" not in phrase
        assert "3 of 4" in phrase
        assert "data-subset" in phrase
        assert "does not gate" in phrase

    def test_warning_and_failure_are_both_reported(self):
        tests = [
            _t("placebo_treatment", status="passed", passed=True),
            _t("unobserved_common_cause", status="warning", passed=False),
            _t("bootstrap", status="failed", passed=False),
        ]
        phrase = gate_verdict_phrase("proceed", tests)
        assert "raised a warning" in phrase
        assert "does not gate" in phrase

    def test_legacy_two_state_not_passed_is_reported_by_count_only(self):
        """A legacy payload (no status) cannot distinguish warning from
        non-critical failure — the phrase must claim neither."""
        tests = [
            _t("placebo_treatment", passed=True),
            _t("random_common_cause", passed=True),
            _t("unobserved_common_cause", passed=False),
        ]
        phrase = gate_verdict_phrase("proceed", tests)
        assert "survived all" not in phrase
        assert "2 of 3" in phrase
        assert "warning" not in phrase
        assert "does not gate" not in phrase

    def test_no_test_data_says_passed_the_gate_without_survived_all(self):
        assert gate_verdict_phrase("proceed", None) == "passed the robustness gate"
        assert gate_verdict_phrase("proceed", []) == "passed the robustness gate"


class TestOtherGates:
    def test_review_and_block_phrases_unchanged(self):
        assert gate_verdict_phrase("review", None) == "needs review (mixed robustness)"
        assert gate_verdict_phrase("block", None) == "failed robustness checks"

    def test_review_block_ignore_tests(self):
        tests = [_t("placebo_treatment", status="passed", passed=True)]
        assert gate_verdict_phrase("review", tests) == "needs review (mixed robustness)"
        assert gate_verdict_phrase("block", tests) == "failed robustness checks"

    def test_unmapped_or_absent_gate_returns_none(self):
        # Call sites keep their raw-report / "robustness unknown" handling.
        assert gate_verdict_phrase(None, None) is None
        assert gate_verdict_phrase("pass", None) is None
        assert gate_verdict_phrase("PROCEED", None) == "passed the robustness gate"
