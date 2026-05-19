"""S12a synthetic golden-set DGP — DAG-explicit role-labeled feature generator.

Plan: ``.claude/plans/s12_synthetic_golden_set_plan.md``.

This package builds a synthetic precision/recall baseline for the DSPy
``CausalRoleClassifier`` (compiled artifact at
``artifacts/dspy/causal_role_classifier.json``). It is the engineering-
doable half of issue #358 — the S12a path of the
``s12_golden_set_feasibility_20260519.md`` feasibility split.

The package coexists as a SIBLING of ``src/ml/synthetic_v2``: that package
is a single-layer logistic regression DGP (``y ~ Bernoulli(sigmoid(X·coef
+ b))``) whose ``FeatureManifest`` carries linear coefficients but no DAG
structure between features. This package is a DAG-explicit, role-labeled
fixture-generator whose only consumer is the precision/recall harness.

Modules:

- ``extractor``: mechanical role extraction from a DAG (Pearl-Lauritzen
  graph properties).
- ``scenarios``: 4 hand-authored DAG scenarios (A1-A4) covering the 6
  role classes.
- ``golden_set``: JSON schema regexes + builder that consumes scenarios
  and emits the fixture dict.
"""

from __future__ import annotations
