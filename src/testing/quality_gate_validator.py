"""Quality Gate Validator for Agent Outputs.

Validates agent outputs against per-agent quality gates that go beyond
simple contract validation. Checks for:
- Meaningful output (not just structure)
- Agent-specific success criteria
- Error detection in output content
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.testing.agent_quality_gates import (
    AGENT_QUALITY_GATES,
    AgentQualityGate,
    DataQualityCheck,
)


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""

    check_name: str
    field_name: str
    passed: bool
    message: str
    expected: str | None = None
    actual: str | None = None


@dataclass
class QualityGateResult:
    """Result of quality gate validation for an agent."""

    agent_name: str
    passed: bool
    failed_checks: list[QualityCheckResult] = field(default_factory=list)
    passed_checks: list[QualityCheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Summary metrics
    total_checks: int = 0
    checks_passed: int = 0
    checks_failed: int = 0

    # Field presence tracking
    required_output_fields_present: list[str] = field(default_factory=list)
    required_output_fields_missing: list[str] = field(default_factory=list)

    # Status-based failure
    status_failure: bool = False
    status_value: str | None = None

    @property
    def summary(self) -> str:
        """Get a summary string of the quality gate result."""
        if self.passed:
            if self.warnings:
                return f"PASS ({self.checks_passed}/{self.total_checks} checks, {len(self.warnings)} warnings)"
            return f"PASS ({self.checks_passed}/{self.total_checks} checks)"
        return f"FAIL ({self.checks_failed} failed: {', '.join(c.check_name for c in self.failed_checks[:3])})"


class QualityGateValidator:
    """Validates agent outputs against per-agent quality gates.

    Unlike ContractValidator which checks TypedDict structure,
    QualityGateValidator checks for meaningful output quality:
    - Are critical fields present?
    - Do values indicate success vs failure?
    - Are there error indicators in the content?

    Usage:
        validator = QualityGateValidator()
        result = validator.validate("tool_composer", agent_output)
        if result.passed:
            print("Quality gate passed")
        else:
            print(f"Failed checks: {result.failed_checks}")
    """

    def __init__(self, quality_gates: dict[str, AgentQualityGate] | None = None):
        """Initialize validator.

        Args:
            quality_gates: Custom quality gates dict. If None, uses AGENT_QUALITY_GATES.
        """
        self.quality_gates = quality_gates or AGENT_QUALITY_GATES

    def validate(
        self,
        agent_name: str,
        output: dict[str, Any],
        contract_required_fields_pct: float | None = None,
        contract_required_total: int = -1,
    ) -> QualityGateResult:
        """Validate agent output against its quality gate.

        Args:
            agent_name: Name of the agent
            output: Agent output dictionary
            contract_required_fields_pct: Percentage of contract required fields present
                                         (from ContractValidator). Used for min_required_fields_pct check.
            contract_required_total: Total number of required fields in the contract.
                                    If 0 (e.g., total=False TypedDict), the percentage check is skipped.

        Returns:
            QualityGateResult with validation details
        """
        result = QualityGateResult(agent_name=agent_name, passed=True)

        # Get quality gate config
        gate = self.quality_gates.get(agent_name)
        if gate is None:
            result.warnings.append(f"No quality gate configured for {agent_name}")
            return result

        # Check required output fields
        self._check_required_output_fields(output, gate, result)

        # Check min required fields percentage (from contract validation)
        if contract_required_fields_pct is not None:
            self._check_min_required_fields_pct(
                contract_required_fields_pct, gate, result, contract_required_total
            )

        # Run data quality checks
        self._run_data_quality_checks(output, gate, result)

        # Check for status-based failure
        self._check_status_failure(output, gate, result)

        # Compute summary
        result.total_checks = len(result.passed_checks) + len(result.failed_checks)
        result.checks_passed = len(result.passed_checks)
        result.checks_failed = len(result.failed_checks)

        # Overall pass/fail
        result.passed = len(result.failed_checks) == 0

        return result

    def _check_required_output_fields(
        self,
        output: dict[str, Any],
        gate: AgentQualityGate,
        result: QualityGateResult,
    ) -> None:
        """Check that required output fields are present."""
        required_fields = gate.get("required_output_fields", [])

        for field_name in required_fields:
            if field_name in output and output[field_name] is not None:
                result.required_output_fields_present.append(field_name)
                result.passed_checks.append(QualityCheckResult(
                    check_name="required_output_field",
                    field_name=field_name,
                    passed=True,
                    message=f"Required field '{field_name}' is present",
                ))
            else:
                result.required_output_fields_missing.append(field_name)
                result.failed_checks.append(QualityCheckResult(
                    check_name="required_output_field",
                    field_name=field_name,
                    passed=False,
                    message=f"Required field '{field_name}' is missing or null",
                    expected="present and non-null",
                    actual="missing" if field_name not in output else "null",
                ))

    def _check_min_required_fields_pct(
        self,
        actual_pct: float,
        gate: AgentQualityGate,
        result: QualityGateResult,
        required_total: int = -1,
    ) -> None:
        """Check minimum required fields percentage.

        If required_total is 0 (e.g., TypedDict with total=False), this check
        automatically passes since there are no required fields to validate.
        """
        min_pct = gate.get("min_required_fields_pct", 0.0)

        # If there are no required fields (total=False TypedDict), auto-pass
        if required_total == 0:
            result.passed_checks.append(QualityCheckResult(
                check_name="min_required_fields_pct",
                field_name="(contract)",
                passed=True,
                message="No required fields in contract (total=False) - check skipped",
            ))
            return

        if actual_pct >= min_pct:
            result.passed_checks.append(QualityCheckResult(
                check_name="min_required_fields_pct",
                field_name="(contract)",
                passed=True,
                message=f"Required fields percentage ({actual_pct:.1%}) >= minimum ({min_pct:.1%})",
            ))
        else:
            result.failed_checks.append(QualityCheckResult(
                check_name="min_required_fields_pct",
                field_name="(contract)",
                passed=False,
                message=f"Required fields percentage ({actual_pct:.1%}) < minimum ({min_pct:.1%})",
                expected=f">= {min_pct:.1%}",
                actual=f"{actual_pct:.1%}",
            ))

    def _run_data_quality_checks(
        self,
        output: dict[str, Any],
        gate: AgentQualityGate,
        result: QualityGateResult,
    ) -> None:
        """Run data quality checks on output fields."""
        checks = gate.get("data_quality_checks", {})

        for field_name, check in checks.items():
            value = output.get(field_name)

            # Check not_null
            if check.get("not_null", False):
                if value is None:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="not_null",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' is null but not_null=True",
                        expected="non-null value",
                        actual="null",
                    ))
                    continue  # Skip other checks if null

            # Skip remaining checks if value is None
            if value is None:
                continue

            # Check type
            expected_type = check.get("type")
            if expected_type:
                check_result = self._check_value_type(field_name, value, expected_type)
                if check_result.passed:
                    result.passed_checks.append(check_result)
                else:
                    result.failed_checks.append(check_result)

            # Check must_be
            if "must_be" in check:
                expected = check["must_be"]
                if value == expected:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="must_be",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' equals expected value",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="must_be",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' must be {expected!r} but is {value!r}",
                        expected=repr(expected),
                        actual=repr(value),
                    ))

            # Check must_not_be
            if "must_not_be" in check:
                forbidden = check["must_not_be"]
                if value != forbidden:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="must_not_be",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' does not equal forbidden value",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="must_not_be",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' must not be {forbidden!r}",
                        expected=f"not {forbidden!r}",
                        actual=repr(value),
                    ))

            # Check in_set
            if "in_set" in check:
                allowed = check["in_set"]
                if value in allowed:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="in_set",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' is in allowed set",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="in_set",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' value {value!r} not in allowed set {allowed}",
                        expected=f"one of {allowed}",
                        actual=repr(value),
                    ))

            # Check not_contains (for strings)
            if "not_contains" in check and isinstance(value, str):
                forbidden_substrings = check["not_contains"]
                found = [s for s in forbidden_substrings if s in value]
                if not found:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="not_contains",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' does not contain forbidden substrings",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="not_contains",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' contains error indicators: {found}",
                        expected=f"no substrings from {forbidden_substrings}",
                        actual=f"contains {found}",
                    ))

            # Check min_value
            if "min_value" in check and isinstance(value, (int, float)):
                min_val = check["min_value"]
                if value >= min_val:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="min_value",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' >= {min_val}",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="min_value",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' is {value} but min is {min_val}",
                        expected=f">= {min_val}",
                        actual=str(value),
                    ))

            # Check max_value
            if "max_value" in check and isinstance(value, (int, float)):
                max_val = check["max_value"]
                if value <= max_val:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="max_value",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' <= {max_val}",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="max_value",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' is {value} but max is {max_val}",
                        expected=f"<= {max_val}",
                        actual=str(value),
                    ))

            # Check min_length
            if "min_length" in check and hasattr(value, "__len__"):
                min_len = check["min_length"]
                actual_len = len(value)
                if actual_len >= min_len:
                    result.passed_checks.append(QualityCheckResult(
                        check_name="min_length",
                        field_name=field_name,
                        passed=True,
                        message=f"Field '{field_name}' length ({actual_len}) >= {min_len}",
                    ))
                else:
                    result.failed_checks.append(QualityCheckResult(
                        check_name="min_length",
                        field_name=field_name,
                        passed=False,
                        message=f"Field '{field_name}' length ({actual_len}) < min ({min_len})",
                        expected=f">= {min_len}",
                        actual=str(actual_len),
                    ))

    def _check_value_type(
        self,
        field_name: str,
        value: Any,
        expected_type: str,
    ) -> QualityCheckResult:
        """Check if value matches expected type."""
        type_map = {
            "str": str,
            "int": int,
            "float": (int, float),  # Allow int for float
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return QualityCheckResult(
                check_name="type",
                field_name=field_name,
                passed=True,
                message=f"Unknown type '{expected_type}', skipping check",
            )

        if isinstance(value, expected):
            return QualityCheckResult(
                check_name="type",
                field_name=field_name,
                passed=True,
                message=f"Field '{field_name}' is {expected_type}",
            )
        else:
            return QualityCheckResult(
                check_name="type",
                field_name=field_name,
                passed=False,
                message=f"Field '{field_name}' expected {expected_type}, got {type(value).__name__}",
                expected=expected_type,
                actual=type(value).__name__,
            )

    def _check_status_failure(
        self,
        output: dict[str, Any],
        gate: AgentQualityGate,
        result: QualityGateResult,
    ) -> None:
        """Check if output status indicates failure."""
        fail_statuses = gate.get("fail_on_status", [])
        if not fail_statuses:
            return

        # Check common status fields
        status_fields = ["status", "success", "state"]
        for field_name in status_fields:
            value = output.get(field_name)
            if value is None:
                continue

            # String status check
            if isinstance(value, str):
                status_lower = value.lower()
                for fail_status in fail_statuses:
                    if fail_status.lower() == status_lower:
                        result.status_failure = True
                        result.status_value = value
                        result.failed_checks.append(QualityCheckResult(
                            check_name="status_failure",
                            field_name=field_name,
                            passed=False,
                            message=f"Agent returned failure status: {value}",
                            expected=f"not in {fail_statuses}",
                            actual=value,
                        ))
                        return

            # Boolean success check
            elif isinstance(value, bool) and field_name == "success":
                if not value and "failed" in fail_statuses:
                    result.status_failure = True
                    result.status_value = "success=False"
                    result.failed_checks.append(QualityCheckResult(
                        check_name="status_failure",
                        field_name=field_name,
                        passed=False,
                        message="Agent returned success=False",
                        expected="success=True",
                        actual="success=False",
                    ))
                    return

        # No status failure detected
        result.passed_checks.append(QualityCheckResult(
            check_name="status_failure",
            field_name="(status fields)",
            passed=True,
            message="No failure status detected",
        ))
