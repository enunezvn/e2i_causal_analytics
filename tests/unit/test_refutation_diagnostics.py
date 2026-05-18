"""Phase 5 forcing tests — refutation diagnostic enrichment.

Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §5 (and v3 §5).

When the placebo refutation test FAILS, ``RefutationResult.details``
must include an ``offending_features`` list. Each entry is a dict of
shape::

    {
        "feature": str,
        "causal_role": Optional[str],
        "source": Optional[str],
        "evaluator_satisfied": Optional[bool],
    }

Resolution is keyed on ``feature``: when a matching
``RoleAttribution`` row exists in the ``role_attributions`` parameter
threaded through ``run_all_tests`` → ``_run_placebo_test``, the entry
inherits ``causal_role`` / ``source`` / ``evaluator_satisfied`` from
that row. When no match is found the three fields are ``None``.

Seam: input dict, no AsyncMock — the test patches the mock placebo
to force a FAILED status and supplies offending feature names via the
new ``offending_features`` kwarg on ``_run_placebo_test``.

Falsifiability: reverting the enrichment branch raises ``KeyError``
on Case 1 (no ``offending_features`` key in ``details``).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.causal_engine.refutation_runner import (
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)


@pytest.fixture
def runner() -> RefutationRunner:
    return RefutationRunner()


def _attribution(
    feature: str,
    causal_role: str,
    source: str = "llm",
    evaluator_satisfied: bool = True,
) -> Dict[str, Any]:
    """Build a ``RoleAttribution``-shaped dict for the test fixtures.

    Matches ``src.data.role_attribution.RoleAttribution`` TypedDict.
    """
    return {
        "feature": feature,
        "causal_role": causal_role,
        "source": source,
        "evaluator_satisfied": evaluator_satisfied,
        "evaluator_model": "test-model:v1",
    }


class TestPlaceboOffendingFeatureEnrichment:
    """Forcing tests for §5 acceptance."""

    def test_offending_feature_with_matching_role_attribution(
        self, runner: RefutationRunner
    ) -> None:
        """Case 1: placebo trips on a feature whose role_attribution row
        exists → entry populated with ``causal_role="collider"``.
        """
        role_attributions: List[Dict[str, Any]] = [
            _attribution("f1", "collider", source="llm", evaluator_satisfied=True),
            _attribution("f2", "confounder", source="manifest", evaluator_satisfied=True),
        ]

        with patch.object(runner, "_mock_placebo_test", return_value=(0.12, 0.02)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
                role_attributions=role_attributions,
                offending_features=["f1"],
            )

        assert result.status == RefutationStatus.FAILED
        assert result.test_name == RefutationTestType.PLACEBO_TREATMENT

        offending = result.details["offending_features"]
        assert isinstance(offending, list)
        assert len(offending) == 1

        entry = offending[0]
        assert entry["feature"] == "f1"
        assert entry["causal_role"] == "collider"
        assert entry["source"] == "llm"
        assert entry["evaluator_satisfied"] is True

    def test_offending_feature_absent_from_role_attributions(
        self, runner: RefutationRunner
    ) -> None:
        """Case 2: placebo trips on a feature with no matching row →
        entry has ``causal_role is None`` (plus None for source +
        evaluator_satisfied).
        """
        role_attributions: List[Dict[str, Any]] = [
            _attribution("known_feature", "ancestor"),
        ]

        with patch.object(runner, "_mock_placebo_test", return_value=(0.12, 0.02)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
                role_attributions=role_attributions,
                offending_features=["unknown_feature"],
            )

        assert result.status == RefutationStatus.FAILED

        offending = result.details["offending_features"]
        assert isinstance(offending, list)
        assert len(offending) == 1

        entry = offending[0]
        assert entry["feature"] == "unknown_feature"
        assert entry["causal_role"] is None
        assert entry["source"] is None
        assert entry["evaluator_satisfied"] is None

    def test_no_estimator_behavior_change_when_role_attributions_omitted(
        self, runner: RefutationRunner
    ) -> None:
        """Acceptance: 'no estimator behavior change'.

        Omitting ``role_attributions`` / ``offending_features`` keeps the
        original placebo behavior intact — status + p_value path is
        identical to pre-Phase 5.
        """
        with patch.object(runner, "_mock_placebo_test", return_value=(0.12, 0.02)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
            )

        assert result.status == RefutationStatus.FAILED
        assert result.original_effect == 0.15
        assert result.p_value == 0.02
        # ``offending_features`` may be absent or empty; either is fine
        # when no role_attributions context was provided.
        assert result.details.get("offending_features", []) == []

    def test_offending_features_only_emitted_on_failure(self, runner: RefutationRunner) -> None:
        """When the placebo PASSES, no offending_features list is needed.

        Enrichment is gated on FAILED status (placebo "trips" semantics).
        """
        role_attributions: List[Dict[str, Any]] = [
            _attribution("f1", "collider"),
        ]

        with patch.object(runner, "_mock_placebo_test", return_value=(0.01, 0.85)):
            result = runner._run_placebo_test(
                original_effect=0.15,
                causal_model=None,
                identified_estimand=None,
                estimate=None,
                use_dowhy=False,
                role_attributions=role_attributions,
                offending_features=["f1"],
            )

        assert result.status == RefutationStatus.PASSED
        # No enrichment on PASSED — keep details minimal.
        assert (
            "offending_features" not in result.details or result.details["offending_features"] == []
        )
