"""Tests for QC gate checker node."""

import pytest
from src.agents.ml_foundation.model_trainer.nodes.qc_gate_checker import check_qc_gate


@pytest.mark.asyncio
class TestCheckQCGate:
    """Test QC gate validation."""

    async def test_qc_gate_passes_when_qc_passed_true(self):
        """QC gate should pass when qc_passed is True."""
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.92,
                "qc_errors": [],
                "qc_warnings": [],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is True
        assert "PASSED" in result["qc_gate_message"]
        assert "0.92" in result["qc_gate_message"]
        assert "error" not in result

    async def test_qc_gate_blocks_when_qc_passed_false(self):
        """QC gate should block when qc_passed is False."""
        state = {
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.45,
                "qc_errors": ["Missing values > 10%", "Duplicate rows detected"],
                "qc_warnings": [],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is False
        assert "BLOCKED" in result["qc_gate_message"]
        assert "0.45" in result["qc_gate_message"]
        assert result["error"] is not None
        assert result["error_type"] == "qc_gate_blocked_error"

    async def test_includes_first_3_errors_in_message(self):
        """Should include first 3 errors in gate message."""
        state = {
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.30,
                "qc_errors": [
                    "Error 1",
                    "Error 2",
                    "Error 3",
                    "Error 4",
                    "Error 5",
                ],
            }
        }

        result = await check_qc_gate(state)

        # Should include first 3 errors
        assert "Error 1" in result["qc_gate_message"]
        assert "Error 2" in result["qc_gate_message"]
        assert "Error 3" in result["qc_gate_message"]
        # Should not include errors 4 and 5
        assert "Error 4" not in result["qc_gate_message"]
        assert "Error 5" not in result["qc_gate_message"]

    async def test_includes_warnings_when_present(self):
        """Should include warnings in gate message when QC passes with warnings."""
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.88,
                "qc_errors": [],
                "qc_warnings": ["Low sample count in some segments", "Slight imbalance"],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is True
        assert "warnings present" in result["qc_gate_message"].lower()

    async def test_no_warnings_message_when_none_present(self):
        """Should indicate no warnings when none present."""
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.95,
                "qc_errors": [],
                "qc_warnings": [],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is True
        assert "No QC warnings" in result["qc_gate_message"]

    async def test_handles_missing_qc_report(self):
        """Should handle missing qc_report gracefully."""
        state = {}

        result = await check_qc_gate(state)

        # Default behavior: qc_passed defaults to False
        assert result["qc_gate_passed"] is False
        assert result["error"] is not None

    async def test_handles_missing_qc_passed_field(self):
        """Should handle missing qc_passed field."""
        state = {
            "qc_report": {
                "overall_score": 0.75,
            }
        }

        result = await check_qc_gate(state)

        # qc_passed defaults to False
        assert result["qc_gate_passed"] is False
        assert result["error"] is not None

    async def test_handles_empty_qc_errors(self):
        """Should handle empty qc_errors list."""
        state = {
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.50,
                "qc_errors": [],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is False
        # Should not crash when joining empty errors list
        assert "qc_gate_message" in result

    async def test_perfect_qc_score(self):
        """Should pass with perfect QC score."""
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 1.0,
                "qc_errors": [],
                "qc_warnings": [],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is True
        assert "1.0" in result["qc_gate_message"] or "1" in result["qc_gate_message"]

    async def test_borderline_qc_score(self):
        """Should respect exact pass/fail threshold."""
        # Assuming threshold is somewhere around 0.7
        # This test checks behavior near boundary
        state_pass = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.71,
                "qc_errors": [],
                "qc_warnings": [],
            }
        }

        state_fail = {
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.69,
                "qc_errors": ["Borderline quality"],
                "qc_warnings": [],
            }
        }

        result_pass = await check_qc_gate(state_pass)
        result_fail = await check_qc_gate(state_fail)

        assert result_pass["qc_gate_passed"] is True
        assert result_fail["qc_gate_passed"] is False

    async def test_warning_count_in_message(self):
        """Should include first 2 warnings in message."""
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.85,
                "qc_errors": [],
                "qc_warnings": ["Warning 1", "Warning 2", "Warning 3", "Warning 4"],
            }
        }

        result = await check_qc_gate(state)

        assert result["qc_gate_passed"] is True
        # Should include first 2 warnings
        assert "Warning 1" in result["qc_gate_message"]
        assert "Warning 2" in result["qc_gate_message"]
