#!/usr/bin/env python3
"""GEPA Integration Test Script - End-to-End Validation.

This script validates the complete GEPA migration by testing:
1. Module imports and initialization
2. Metric class instantiation for all agent types
3. Optimizer factory functionality
4. Versioning and rollback capabilities
5. A/B testing infrastructure
6. MLOps integrations (MLflow, Opik, RAGAS)
7. Tool definitions and registry
8. Domain vocabulary validation

Usage:
    # Run all integration tests
    python scripts/gepa_integration_test.py

    # Run specific test category
    python scripts/gepa_integration_test.py --category imports

    # Verbose output
    python scripts/gepa_integration_test.py --verbose

Author: E2I Causal Analytics Team
Version: 4.2.0
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    category: str
    passed: bool
    duration_ms: float
    error: str | None = None


class GEPAIntegrationTests:
    """GEPA integration test suite."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize test suite.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results: list[TestResult] = []

    def run_test(
        self,
        name: str,
        category: str,
        test_fn: Callable[[], bool],
    ) -> TestResult:
        """Run a single test.

        Args:
            name: Test name
            category: Test category
            test_fn: Test function returning True if passed

        Returns:
            TestResult
        """
        start = datetime.now()
        try:
            passed = test_fn()
            error = None
        except Exception as e:
            passed = False
            error = str(e)

        duration = (datetime.now() - start).total_seconds() * 1000

        result = TestResult(
            name=name,
            category=category,
            passed=passed,
            duration_ms=duration,
            error=error,
        )
        self.results.append(result)

        status = "PASS" if passed else "FAIL"
        if self.verbose or not passed:
            logger.info(f"[{status}] {category}/{name} ({duration:.1f}ms)")
            if error:
                logger.error(f"       Error: {error}")

        return result

    # =========================================================================
    # Category: Imports
    # =========================================================================

    def test_core_imports(self) -> bool:
        """Test core GEPA module imports."""
        from src.optimization.gepa import (
            CausalImpactGEPAMetric,
            E2IGEPAMetric,
            ExperimentDesignerGEPAMetric,
            FeedbackLearnerGEPAMetric,
            StandardAgentGEPAMetric,
            create_gepa_optimizer,
            get_metric_for_agent,
        )

        return True

    def test_optimizer_imports(self) -> bool:
        """Test optimizer-related imports."""
        from src.optimization.gepa import (
            AGENT_BUDGETS,
            BUDGET_PRESETS,
            create_optimizer_for_agent,
            optimize_causal_impact_agent,
            optimize_experiment_designer_agent,
            optimize_feedback_learner_agent,
            optimize_standard_agent,
        )

        return True

    def test_versioning_imports(self) -> bool:
        """Test versioning module imports."""
        from src.optimization.gepa import (
            compare_versions,
            compute_instruction_hash,
            generate_version_id,
            list_versions,
            load_optimized_module,
            rollback_to_version,
            save_optimized_module,
        )

        return True

    def test_ab_test_imports(self) -> bool:
        """Test A/B testing imports."""
        from src.optimization.gepa import (
            ABTestObservation,
            ABTestResults,
            ABTestVariant,
            GEPAABTest,
        )

        return True

    def test_tools_imports(self) -> bool:
        """Test tool definitions imports."""
        from src.optimization.gepa.tools import (
            CAUSAL_TOOLS,
            CAUSAL_FOREST,
            DIFFERENCE_IN_DIFFERENCES,
            DOUBLE_ROBUST_LEARNER,
            GEPATool,
            INSTRUMENTAL_VARIABLE,
            LINEAR_DML,
            METALEARNER_S,
            METALEARNER_T,
            REGRESSION_DISCONTINUITY,
            SPARSE_LINEAR_DML,
            TOOL_REGISTRY,
            get_tool_by_name,
            get_tools_by_category,
            get_tools_for_agent,
        )

        return True

    def test_integration_imports(self) -> bool:
        """Test MLOps integration imports."""
        from src.optimization.gepa.integration import (
            GEPAMLflowCallback,
            GEPAOpikTracer,
            RAGASFeedbackProvider,
            create_ragas_metric,
            log_optimization_run,
            trace_optimization,
        )

        return True

    # =========================================================================
    # Category: Metrics
    # =========================================================================

    def test_metric_factory(self) -> bool:
        """Test metric factory function."""
        from src.optimization.gepa import get_metric_for_agent

        # Test all agent types
        agents = [
            "causal_impact",
            "experiment_designer",
            "feedback_learner",
            "gap_analyzer",
            "orchestrator",
        ]

        for agent in agents:
            metric = get_metric_for_agent(agent)
            if metric is None:
                return False

        return True

    def test_causal_impact_metric(self) -> bool:
        """Test CausalImpactGEPAMetric instantiation."""
        from src.optimization.gepa import CausalImpactGEPAMetric

        metric = CausalImpactGEPAMetric()
        return hasattr(metric, "__call__") or hasattr(metric, "evaluate")

    def test_experiment_designer_metric(self) -> bool:
        """Test ExperimentDesignerGEPAMetric instantiation."""
        from src.optimization.gepa import ExperimentDesignerGEPAMetric

        metric = ExperimentDesignerGEPAMetric()
        return hasattr(metric, "__call__") or hasattr(metric, "evaluate")

    def test_feedback_learner_metric(self) -> bool:
        """Test FeedbackLearnerGEPAMetric instantiation."""
        from src.optimization.gepa import FeedbackLearnerGEPAMetric

        metric = FeedbackLearnerGEPAMetric()
        return hasattr(metric, "__call__") or hasattr(metric, "evaluate")

    def test_standard_agent_metric(self) -> bool:
        """Test StandardAgentGEPAMetric instantiation."""
        from src.optimization.gepa import StandardAgentGEPAMetric

        metric = StandardAgentGEPAMetric(name="gap_analyzer")
        return hasattr(metric, "__call__") or hasattr(metric, "evaluate")

    # =========================================================================
    # Category: Optimizer
    # =========================================================================

    def test_budget_presets(self) -> bool:
        """Test budget presets configuration."""
        from src.optimization.gepa import BUDGET_PRESETS

        required_budgets = ["light", "medium", "heavy"]
        for budget in required_budgets:
            if budget not in BUDGET_PRESETS:
                return False
            config = BUDGET_PRESETS[budget]
            if "max_metric_calls" not in config:
                return False

        return True

    def test_agent_budgets(self) -> bool:
        """Test agent-specific budget configuration."""
        from src.optimization.gepa import AGENT_BUDGETS

        # Check that key agents have budgets defined
        key_agents = ["causal_impact", "experiment_designer", "feedback_learner"]
        for agent in key_agents:
            if agent not in AGENT_BUDGETS:
                return False

        return True

    # =========================================================================
    # Category: Tools
    # =========================================================================

    def test_causal_tools_registry(self) -> bool:
        """Test causal tools registry."""
        from src.optimization.gepa.tools import CAUSAL_TOOLS, TOOL_REGISTRY

        if len(CAUSAL_TOOLS) != 9:
            return False

        if len(TOOL_REGISTRY) != 9:
            return False

        return True

    def test_tool_lookup_by_name(self) -> bool:
        """Test tool lookup by name."""
        from src.optimization.gepa.tools import get_tool_by_name

        tool = get_tool_by_name("causal_forest")
        if tool is None:
            return False

        return tool.name == "causal_forest"

    def test_tools_for_agent(self) -> bool:
        """Test getting tools for specific agents."""
        from src.optimization.gepa.tools import get_tools_for_agent

        causal_tools = get_tools_for_agent("causal_impact")
        if len(causal_tools) != 9:
            return False

        exp_tools = get_tools_for_agent("experiment_designer")
        if len(exp_tools) != 2:
            return False

        other_tools = get_tools_for_agent("gap_analyzer")
        if len(other_tools) != 9:  # causal agents get all tools
            return False

        return True

    def test_tools_by_category(self) -> bool:
        """Test getting tools by category."""
        from src.optimization.gepa.tools import get_tools_by_category

        dml_tools = get_tools_by_category("dml")
        if len(dml_tools) != 2:
            return False

        metalearner_tools = get_tools_by_category("metalearner")
        if len(metalearner_tools) != 2:
            return False

        return True

    # =========================================================================
    # Category: Integrations
    # =========================================================================

    def test_mlflow_callback(self) -> bool:
        """Test MLflow callback instantiation."""
        from src.optimization.gepa.integration import GEPAMLflowCallback

        callback = GEPAMLflowCallback(
            experiment_name="test_experiment",
            run_name="test_run",
        )
        return callback is not None

    def test_opik_tracer(self) -> bool:
        """Test Opik tracer instantiation."""
        from src.optimization.gepa.integration import GEPAOpikTracer

        tracer = GEPAOpikTracer(project_name="test_project")
        return tracer is not None

    def test_ragas_feedback_provider(self) -> bool:
        """Test RAGAS feedback provider."""
        from src.optimization.gepa.integration import RAGASFeedbackProvider

        provider = RAGASFeedbackProvider()
        return provider is not None

    def test_ragas_metric_factory(self) -> bool:
        """Test RAGAS metric factory."""
        from src.optimization.gepa.integration import create_ragas_metric

        metric = create_ragas_metric(agent_name="cognitive_rag")
        return callable(metric)

    # =========================================================================
    # Category: Versioning
    # =========================================================================

    def test_version_id_generation(self) -> bool:
        """Test version ID generation."""
        from src.optimization.gepa import generate_version_id

        version = generate_version_id("test_agent")

        # Should follow format: gepa_v1_{agent}_{timestamp}
        return version.startswith("gepa_v1_test_agent_") and len(version) > len("gepa_v1_test_agent_")

    def test_instruction_hash(self) -> bool:
        """Test instruction hashing."""
        from src.optimization.gepa import compute_instruction_hash

        hash1 = compute_instruction_hash("Test instruction 1")
        hash2 = compute_instruction_hash("Test instruction 2")
        hash3 = compute_instruction_hash("Test instruction 1")

        return hash1 != hash2 and hash1 == hash3

    # =========================================================================
    # Category: A/B Testing
    # =========================================================================

    def test_ab_test_variants(self) -> bool:
        """Test A/B test variant definition."""
        from src.optimization.gepa import ABTestVariant

        variant = ABTestVariant(
            variant_id="v1",
            name="test_variant",
        )
        return variant.name == "test_variant"

    def test_ab_test_initialization(self) -> bool:
        """Test GEPAABTest initialization."""
        from src.optimization.gepa import GEPAABTest

        ab_test = GEPAABTest(
            test_name="test_ab",
            agent_name="causal_impact",
            baseline_instruction_id="v0",
            treatment_instruction_id="v1",
        )
        return ab_test.test_name == "test_ab"

    # =========================================================================
    # Category: Vocabulary
    # =========================================================================

    def test_vocabulary_file_exists(self) -> bool:
        """Test domain vocabulary file exists."""
        vocab_path = project_root / "config" / "domain_vocabulary.yaml"
        return vocab_path.exists()

    def test_vocabulary_gepa_section(self) -> bool:
        """Test GEPA section in vocabulary."""
        import yaml

        vocab_path = project_root / "config" / "domain_vocabulary.yaml"
        with open(vocab_path) as f:
            vocab = yaml.safe_load(f)

        # Check for GEPA-specific keys
        required_keys = [
            "optimizer_types",
            "gepa_budget_presets",
            "ab_test_variants",
            "gepa_metric_components",
        ]

        for key in required_keys:
            if key not in vocab:
                return False

        return True

    def test_vocabulary_version(self) -> bool:
        """Test vocabulary version is 5.0.0."""
        import yaml

        vocab_path = project_root / "config" / "domain_vocabulary.yaml"
        with open(vocab_path) as f:
            vocab = yaml.safe_load(f)

        metadata = vocab.get("_metadata", {})
        return metadata.get("version") == "5.1.0"

    # =========================================================================
    # Run Tests
    # =========================================================================

    def run_category(self, category: str) -> list[TestResult]:
        """Run all tests in a category.

        Args:
            category: Category name

        Returns:
            List of test results
        """
        tests = {
            "imports": [
                ("core_imports", self.test_core_imports),
                ("optimizer_imports", self.test_optimizer_imports),
                ("versioning_imports", self.test_versioning_imports),
                ("ab_test_imports", self.test_ab_test_imports),
                ("tools_imports", self.test_tools_imports),
                ("integration_imports", self.test_integration_imports),
            ],
            "metrics": [
                ("metric_factory", self.test_metric_factory),
                ("causal_impact_metric", self.test_causal_impact_metric),
                ("experiment_designer_metric", self.test_experiment_designer_metric),
                ("feedback_learner_metric", self.test_feedback_learner_metric),
                ("standard_agent_metric", self.test_standard_agent_metric),
            ],
            "optimizer": [
                ("budget_presets", self.test_budget_presets),
                ("agent_budgets", self.test_agent_budgets),
            ],
            "tools": [
                ("causal_tools_registry", self.test_causal_tools_registry),
                ("tool_lookup_by_name", self.test_tool_lookup_by_name),
                ("tools_for_agent", self.test_tools_for_agent),
                ("tools_by_category", self.test_tools_by_category),
            ],
            "integrations": [
                ("mlflow_callback", self.test_mlflow_callback),
                ("opik_tracer", self.test_opik_tracer),
                ("ragas_feedback_provider", self.test_ragas_feedback_provider),
                ("ragas_metric_factory", self.test_ragas_metric_factory),
            ],
            "versioning": [
                ("version_id_generation", self.test_version_id_generation),
                ("instruction_hash", self.test_instruction_hash),
            ],
            "ab_testing": [
                ("ab_test_variants", self.test_ab_test_variants),
                ("ab_test_initialization", self.test_ab_test_initialization),
            ],
            "vocabulary": [
                ("vocabulary_file_exists", self.test_vocabulary_file_exists),
                ("vocabulary_gepa_section", self.test_vocabulary_gepa_section),
                ("vocabulary_version", self.test_vocabulary_version),
            ],
        }

        category_tests = tests.get(category, [])
        results = []
        for name, test_fn in category_tests:
            result = self.run_test(name, category, test_fn)
            results.append(result)

        return results

    def run_all(self) -> list[TestResult]:
        """Run all integration tests.

        Returns:
            List of all test results
        """
        categories = [
            "imports",
            "metrics",
            "optimizer",
            "tools",
            "integrations",
            "versioning",
            "ab_testing",
            "vocabulary",
        ]

        for category in categories:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {category} tests")
            logger.info("=" * 50)
            self.run_category(category)

        return self.results

    def summary(self) -> dict[str, Any]:
        """Generate test summary.

        Returns:
            Summary dict
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        by_category: dict[str, dict[str, int]] = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = {"passed": 0, "failed": 0}
            if result.passed:
                by_category[result.category]["passed"] += 1
            else:
                by_category[result.category]["failed"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "by_category": by_category,
            "failed_tests": [r.name for r in self.results if not r.passed],
        }


def main() -> None:
    """Main entry point for integration tests."""
    parser = argparse.ArgumentParser(
        description="GEPA Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--category",
        choices=[
            "imports",
            "metrics",
            "optimizer",
            "tools",
            "integrations",
            "versioning",
            "ab_testing",
            "vocabulary",
        ],
        help="Run specific test category only",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run tests
    suite = GEPAIntegrationTests(verbose=args.verbose)

    if args.category:
        suite.run_category(args.category)
    else:
        suite.run_all()

    # Print summary
    summary = suite.summary()

    print("\n" + "=" * 60)
    print("GEPA INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")

    if summary["failed_tests"]:
        print(f"\nFailed Tests:")
        for test in summary["failed_tests"]:
            print(f"  - {test}")

    print("\nBy Category:")
    for category, stats in summary["by_category"].items():
        status = "PASS" if stats["failed"] == 0 else "FAIL"
        print(f"  [{status}] {category}: {stats['passed']}/{stats['passed'] + stats['failed']}")

    print("=" * 60)

    # Exit with error if any failed
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
