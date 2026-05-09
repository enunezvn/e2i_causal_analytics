"""Scenario A balanced — 50:50 prevalence derivative of scenario_a.

Per `.claude/plans/synthetic_cohort_growth_plan_20260509.md` Phase 3:

- Inherits scenario_a's HR+/HER2- early BC iDFS DGP (40 features, 6 clinical
  clusters + 1 noise cluster, locked correlation blocks, locked slope multiplier).
- Overrides ONLY ``target_prevalence`` (0.20 → 0.50). The intercept solver in
  ``api.py:204`` re-calibrates per call against the same standardized linear
  predictor, producing labels with empirical prevalence ~0.50 while preserving
  the feature ↔ target correlation structure that the post-hoc ``--imbalanced``
  flag (``run_tier0_test.py:1430-1452``) destroys by relabeling after generation.

Used by ``--regime scenario_a_balanced`` to empirically test whether a
balanced cohort at scale (n=20,000, prevalence=0.50) produces a deployable
model when scenario_a's DGP is held fixed but prevalence is shifted.

The ``target_auc_band`` is intentionally inherited from scenario_a even
though the AUC envelope at prevalence=0.50 will differ; downstream
``test_synthetic_cohort_growth.py`` empirically pins a separate
balanced-regime band (Phase 4).
"""

from __future__ import annotations

from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.scenarios.scenario_a import ScenarioABuilder


class ScenarioABalancedBuilder(ScenarioABuilder):
    @property
    def name(self) -> ScenarioName:
        return ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED

    @property
    def target_prevalence(self) -> float:
        return 0.50


SCENARIO_REGISTRY[ScenarioName.A_DIAGNOSTIC_BC_IDFS_BALANCED] = ScenarioABalancedBuilder
