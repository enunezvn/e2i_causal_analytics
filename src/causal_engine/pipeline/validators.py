"""Stage validators for inter-library compatibility.

Validates that outputs from one library stage are compatible
with inputs expected by the next stage in the pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .router import CausalLibrary
from .state import PipelineState

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.suggestions = suggestions or []

    def __bool__(self) -> bool:
        return self.is_valid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class StageValidator(ABC):
    """Abstract base class for stage validators."""

    @property
    @abstractmethod
    def source_library(self) -> CausalLibrary:
        """Library whose output is being validated."""
        pass

    @property
    @abstractmethod
    def target_library(self) -> CausalLibrary:
        """Library that will receive the output."""
        pass

    @abstractmethod
    def validate(self, state: PipelineState) -> ValidationResult:
        """Validate state for inter-library compatibility.

        Args:
            state: Current pipeline state after source library execution

        Returns:
            ValidationResult with validity status and any issues found
        """
        pass


class NetworkXToDoWhyValidator(StageValidator):
    """Validates NetworkX output for DoWhy input."""

    @property
    def source_library(self) -> CausalLibrary:
        return CausalLibrary.NETWORKX

    @property
    def target_library(self) -> CausalLibrary:
        return CausalLibrary.DOWHY

    def validate(self, state: PipelineState) -> ValidationResult:
        """Validate that graph structure is suitable for DoWhy."""
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # Check if NetworkX produced a graph
        if not state.get("causal_graph"):
            errors.append("No causal graph available from NetworkX")
            return ValidationResult(False, errors, warnings, suggestions)

        graph = state["causal_graph"]

        # Check for required nodes
        nodes = graph.get("nodes", [])  # type: ignore[union-attr]
        edges = graph.get("edges", [])  # type: ignore[union-attr]

        if not nodes:
            errors.append("Causal graph has no nodes")

        if not edges:
            warnings.append("Causal graph has no edges - DoWhy may have limited utility")

        # Check treatment and outcome are in graph
        treatment = state.get("treatment_var")
        outcome = state.get("outcome_var")

        if treatment and treatment not in nodes:
            warnings.append(f"Treatment variable '{treatment}' not in graph nodes")
            suggestions.append(f"Add {treatment} to causal graph nodes")

        if outcome and outcome not in nodes:
            warnings.append(f"Outcome variable '{outcome}' not in graph nodes")
            suggestions.append(f"Add {outcome} to causal graph nodes")

        # Check for path from treatment to outcome
        if treatment and outcome and edges:
            has_path = any(e.get("from") == treatment and e.get("to") == outcome for e in edges)
            if not has_path:
                suggestions.append(f"Consider adding direct edge from {treatment} to {outcome}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)


class DoWhyToEconMLValidator(StageValidator):
    """Validates DoWhy output for EconML input."""

    @property
    def source_library(self) -> CausalLibrary:
        return CausalLibrary.DOWHY

    @property
    def target_library(self) -> CausalLibrary:
        return CausalLibrary.ECONML

    def validate(self, state: PipelineState) -> ValidationResult:
        """Validate that DoWhy results are suitable for EconML CATE."""
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # Check if DoWhy produced results
        if not state.get("dowhy_result"):
            errors.append("No DoWhy results available")
            return ValidationResult(False, errors, warnings, suggestions)

        dowhy_result = state["dowhy_result"]

        # Check for successful DoWhy execution
        if not dowhy_result.get("success"):  # type: ignore[union-attr]
            error_msg = dowhy_result.get("error", "Unknown error")  # type: ignore[union-attr]
            errors.append(f"DoWhy execution failed: {error_msg}")
            return ValidationResult(False, errors, warnings, suggestions)

        # Check identification method
        identification = state.get("identification_method")
        if not identification:
            warnings.append("No identification method specified from DoWhy")
            suggestions.append("EconML will use default backdoor identification")

        # Check causal effect estimate
        causal_effect = state.get("causal_effect")
        if causal_effect is None:
            warnings.append("No causal effect estimate from DoWhy")
            suggestions.append("EconML will estimate ATE independently")

        # Check refutation results
        refutation = state.get("refutation_results")
        if refutation:
            # Check for any failed refutation tests
            for test_name, test_result in refutation.items():
                if isinstance(test_result, dict) and not test_result.get("passed", True):
                    warnings.append(f"Refutation test '{test_name}' raised concerns")
                    suggestions.append(
                        "Review refutation results before trusting EconML CATE estimates"
                    )

        # Check for effect modifiers (needed for heterogeneity)
        if not state.get("effect_modifiers"):
            warnings.append("No effect modifiers specified")
            suggestions.append("Specify effect modifiers for meaningful CATE estimation")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)


class EconMLToCausalMLValidator(StageValidator):
    """Validates EconML output for CausalML input."""

    @property
    def source_library(self) -> CausalLibrary:
        return CausalLibrary.ECONML

    @property
    def target_library(self) -> CausalLibrary:
        return CausalLibrary.CAUSALML

    def validate(self, state: PipelineState) -> ValidationResult:
        """Validate that EconML results are suitable for CausalML uplift."""
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # Check if EconML produced results
        if not state.get("econml_result"):
            errors.append("No EconML results available")
            return ValidationResult(False, errors, warnings, suggestions)

        econml_result = state["econml_result"]

        # Check for successful EconML execution
        if not econml_result.get("success"):  # type: ignore[union-attr]
            error_msg = econml_result.get("error", "Unknown error")  # type: ignore[union-attr]
            errors.append(f"EconML execution failed: {error_msg}")
            return ValidationResult(False, errors, warnings, suggestions)

        # Check heterogeneity score
        het_score = state.get("heterogeneity_score")
        if het_score is not None and het_score < 0.1:
            warnings.append(f"Low heterogeneity score ({het_score:.3f}) suggests uniform effects")
            suggestions.append("CausalML uplift modeling may provide limited additional value")

        # Check CATE by segment
        cate_by_segment = state.get("cate_by_segment")
        if cate_by_segment:
            # Check for any segments with negative CATE
            negative_segments = [
                seg
                for seg, cate in cate_by_segment.items()
                if isinstance(cate, (int, float)) and cate < 0
            ]
            if negative_segments:
                warnings.append(f"Negative CATE found in segments: {negative_segments}")
                suggestions.append(
                    "CausalML can help identify optimal targeting to avoid negative effects"
                )

        # Check ATE
        ate = state.get("overall_ate")
        if ate is None:
            warnings.append("No overall ATE from EconML")
        elif ate == 0:
            warnings.append("ATE is zero - treatment may have no effect on average")
            suggestions.append("CausalML may still find segments with positive uplift")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, suggestions)


class PipelineValidator:
    """Orchestrates validation across pipeline stages."""

    def __init__(self):
        self._validators: Dict[tuple[CausalLibrary, CausalLibrary], StageValidator] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """Register the default stage validators."""
        validators = [
            NetworkXToDoWhyValidator(),
            DoWhyToEconMLValidator(),
            EconMLToCausalMLValidator(),
        ]
        for v in validators:
            self._validators[(v.source_library, v.target_library)] = v

    def register_validator(self, validator: StageValidator):
        """Register a custom validator.

        Args:
            validator: StageValidator implementation
        """
        key = (validator.source_library, validator.target_library)
        self._validators[key] = validator

    def validate_transition(
        self,
        state: PipelineState,
        from_library: CausalLibrary,
        to_library: CausalLibrary,
    ) -> ValidationResult:
        """Validate transition between two libraries.

        Args:
            state: Current pipeline state
            from_library: Source library
            to_library: Target library

        Returns:
            ValidationResult for the transition
        """
        key = (from_library, to_library)
        validator = self._validators.get(key)

        if not validator:
            # No specific validator - return valid with warning
            return ValidationResult(
                True,
                warnings=[f"No validator for {from_library.value} → {to_library.value}"],
            )

        return validator.validate(state)

    def validate_full_pipeline(
        self,
        state: PipelineState,
        libraries: List[CausalLibrary],
    ) -> Dict[str, ValidationResult]:
        """Validate all transitions in a pipeline.

        Args:
            state: Current pipeline state
            libraries: Ordered list of libraries in the pipeline

        Returns:
            Dict mapping transition names to ValidationResults
        """
        results = {}

        for i in range(len(libraries) - 1):
            from_lib = libraries[i]
            to_lib = libraries[i + 1]
            transition_name = f"{from_lib.value}_to_{to_lib.value}"

            results[transition_name] = self.validate_transition(state, from_lib, to_lib)

        return results

    def get_all_warnings(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Extract all warnings from validation results.

        Args:
            validation_results: Dict of validation results

        Returns:
            List of all warning messages
        """
        warnings = []
        for transition, result in validation_results.items():
            for warning in result.warnings:
                warnings.append(f"[{transition}] {warning}")
        return warnings

    def get_all_suggestions(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Extract all suggestions from validation results.

        Args:
            validation_results: Dict of validation results

        Returns:
            List of all suggestion messages
        """
        suggestions = []
        for transition, result in validation_results.items():
            for suggestion in result.suggestions:
                suggestions.append(f"[{transition}] {suggestion}")
        return suggestions


def validate_pipeline_state(
    state: PipelineState,
    from_library: CausalLibrary,
    to_library: CausalLibrary,
) -> ValidationResult:
    """Convenience function to validate a single transition.

    Args:
        state: Current pipeline state
        from_library: Source library
        to_library: Target library

    Returns:
        ValidationResult for the transition
    """
    validator = PipelineValidator()
    return validator.validate_transition(state, from_library, to_library)
