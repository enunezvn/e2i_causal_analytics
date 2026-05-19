"""Golden-set JSON assembly + schema regexes (plan §4).

Consumes the 4 scenarios from :mod:`scenarios` and emits a dict matching
the §4 schema. Family B ((T, Y)-explicit) is generated here by mirroring
Family A with a ``; treatment=...; outcome=...`` suffix appended to
``dataset_context``.

Volatile metadata (timestamps, git commits) is NOT included in the
persisted fixture — only in the runtime ``--out`` artifact produced by
:mod:`scripts.evaluate_causal_role_precision`. This keeps the fixture
deterministic for the §6.1 ``test_golden_set_fixture_pin`` semantic
compare.
"""

from __future__ import annotations

from typing import Any

from src.ml.causal_role_dgp.scenarios import (
    SCENARIO_NAMES,
    FeatureEntry,
    SyntheticScenario,
    build_scenario,
)

GOLDEN_SET_VERSION = "1"

# §4 regex invariants. Tightened on codex iter-0 HIGH-1 (add
# `derivation_inputs`) + iter-0 MED-1 (allow `aggregation=None`).
# Live-verified against the production f-string at
# `adaptive_validity_check.py:879-885`.
DERIVATION_PSEUDOCODE_REGEX = (
    r"^source=[A-Za-z_][A-Za-z0-9_]*; "
    r"derivation_inputs=\[(?:'[^']*'(?:, '[^']*')*)?\]; "
    r"aggregation=([A-Za-z_]+|None); "
    r"window_days=(\d+|None); "
    r"knowable_at=[a-z_]+(?:[+-]\d+d)?$"
)

DATASET_CONTEXT_REGEX = (
    r"^cohort=[A-Za-z0-9_]+; "
    r"target=[a-z0-9_]+; "
    r"prediction_anchor=[a-z0-9_+-]+"
    r"(?:; treatment=[a-z0-9_]+; outcome=[a-z0-9_]+)?$"
)


def _entry_to_dict(scenario_name: str, entry: FeatureEntry) -> dict[str, Any]:
    return {
        "scenario": scenario_name,
        "feature_name": entry.feature_name,
        "derivation_pseudocode": entry.derivation_pseudocode,
        "dataset_context": entry.dataset_context,
        "ground_truth_role": entry.ground_truth_role,
        "rationale": entry.rationale,
        "treatment_explicit": entry.treatment_explicit,
    }


def _scenario_to_dict(scenario: SyntheticScenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "treatment_node": scenario.treatment_node,
        "outcome_node": scenario.outcome_node,
        "dag_edges": [list(e) for e in sorted(scenario.dag.edges())],
        "n_features": len(scenario.entries),
    }


def _build_family_b_entry(
    scenario: SyntheticScenario,
    entry_a: FeatureEntry,
) -> FeatureEntry:
    """Mirror a Family A entry with (T, Y)-explicit suffix in dataset_context."""
    new_context = (
        f"{entry_a.dataset_context}; "
        f"treatment={scenario.treatment_node}; "
        f"outcome={scenario.outcome_node}"
    )
    return FeatureEntry(
        node_name=entry_a.node_name,
        feature_name=entry_a.feature_name,
        derivation_pseudocode=entry_a.derivation_pseudocode,
        dataset_context=new_context,
        ground_truth_role=entry_a.ground_truth_role,
        rationale=entry_a.rationale,
        treatment_explicit=True,
    )


def build_golden_set() -> dict[str, Any]:
    """Assemble the golden-set dict from all 4 scenarios.

    Returns a dict matching the §4 schema:

    .. code-block:: python

        {
            "version": "1",
            "scenarios": [
                {"name": ..., "treatment_node": ..., "outcome_node": ...,
                 "dag_edges": [[u, v], ...], "n_features": int},
                ...,
            ],
            "entries": [
                {"scenario": ..., "feature_name": ..., "derivation_pseudocode": ...,
                 "dataset_context": ..., "ground_truth_role": ...,
                 "rationale": ..., "treatment_explicit": bool},
                ...,
            ],
        }

    Family A (cohort-only `dataset_context`) is emitted first, then
    Family B ((T, Y)-explicit re-emissions) for the same set of features
    and roles. Family B is informational-only at the harness layer (plan
    §0 + §5).
    """
    scenarios_list = [build_scenario(name) for name in SCENARIO_NAMES]

    scenarios_payload: list[dict[str, Any]] = [_scenario_to_dict(s) for s in scenarios_list]

    entries_payload: list[dict[str, Any]] = []
    # Family A
    for scenario in scenarios_list:
        for entry in scenario.entries:
            entries_payload.append(_entry_to_dict(scenario.name, entry))
    # Family B: re-emission of Family A with (T, Y)-explicit suffix
    for scenario in scenarios_list:
        for entry in scenario.entries:
            mirror = _build_family_b_entry(scenario, entry)
            entries_payload.append(_entry_to_dict(scenario.name, mirror))

    return {
        "version": GOLDEN_SET_VERSION,
        "scenarios": scenarios_payload,
        "entries": entries_payload,
    }
