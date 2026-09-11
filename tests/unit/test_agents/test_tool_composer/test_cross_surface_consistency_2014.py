"""#2014 whole-diff review: surfaces that carry the same interval must agree.

1. ``causal_effect_estimator`` returns ``ci_lower`` / ``ci_upper`` = null when no
   sampling uncertainty exists, and the planner maps them into ``sensitivity_analyzer``
   by name. A consumer that validates against the declared input schema must accept
   that null, so every dependency that carries a nullable producer field declares a
   nullable consumer input.
2. ``refutation_runner``'s E-value test reads the interval the DoWhy executor builds
   from the same HC1 SE; it must use the same critical value as the estimator's CI
   (it used 1.96 against the tool's 1.959963984540054, so a z-ratio between the two
   read "null" on one surface and not on the other).

Real registry, real DoWhy refutation suite on a real deterministic frame.
"""

from __future__ import annotations

from typing import Union, get_args, get_origin

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.tool_registry import (
    DEPENDENCY_FIELD_MAPPINGS,
    create_default_tools,
)
from src.tool_registry import get_registry

Z_95 = 1.959963984540054


def _nullable(annotation) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _schema_allows_null(prop: dict) -> bool:
    return prop.get("type") == "null" or any(
        branch.get("type") == "null" for branch in prop.get("anyOf", [])
    )


def test_a_nullable_producer_field_maps_to_an_input_that_accepts_null() -> None:
    registry = get_registry()
    schemas = {tool.name: tool.input_schema for tool in create_default_tools()}
    rejecting = []
    for (consumer, producer), (output_field, input_field) in DEPENDENCY_FIELD_MAPPINGS.items():
        if consumer not in schemas or output_field is None and input_field is not None:
            continue
        fields = registry.get(producer).pydantic_output_model.model_fields
        properties = schemas[consumer]["properties"]
        pairs = (
            [(name, name) for name in set(fields) & set(properties)]
            if output_field is None
            else [(output_field, input_field)]
        )
        rejecting += [
            f"{producer}.{field} -> {consumer}.{param}"
            for field, param in pairs
            if _nullable(fields[field].annotation) and not _schema_allows_null(properties[param])
        ]
    assert not rejecting, rejecting


def test_refutation_reads_the_estimators_own_interval(monkeypatch) -> None:
    # The suite's legacy output flattens the E-value details to two-decimal prose, so
    # the interval is observed where the executor hands it over: a subclass that
    # records ``original_ci`` and runs the real suite unchanged.
    from src.causal_engine.pipeline.executors import dowhy as dowhy_executor
    from src.causal_engine.refutation_runner import RefutationRunner

    seen = []

    class RecordingRunner(RefutationRunner):
        def run_all_tests(self, **kwargs):  # type: ignore[no-untyped-def]
            seen.append((kwargs["original_effect"], kwargs["original_ci"]))
            return super().run_all_tests(**kwargs)

    monkeypatch.setattr(dowhy_executor, "RefutationRunner", RecordingRunner)
    rng = np.random.default_rng(2014)
    n = 600
    c = rng.normal(0.0, 1.0, n)
    t = (c + rng.normal(0.0, 1.0, n) > 0.3).astype(int)
    y = 0.4 * t + 0.8 * c + rng.normal(0.0, 1.0, n) * (1.0 + 2.0 * t)
    df = pd.DataFrame({"treatment": t, "outcome": y, "confounder_a": c})
    kwargs = {"treatment": "treatment", "outcome": "outcome", "confounders": ["confounder_a"]}

    estimate = tr.causal_effect_estimator(estimation_data=df, **kwargs)
    refutation = tr.refutation_runner(estimation_data=df, **kwargs)

    assert refutation["gate_decision"] is not None
    assert len(seen) == 1
    effect, (lower, upper) = seen[0]
    assert effect == pytest.approx(estimate.ate, rel=1e-12)
    assert lower == pytest.approx(estimate.ci_lower, rel=1e-12)
    assert upper == pytest.approx(estimate.ci_upper, rel=1e-12)
