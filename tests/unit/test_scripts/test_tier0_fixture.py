"""Contract tests for the committed Tier-0 cache fixture — GitHub issue #600.

Background (#600): the Tier 1-5 agent harness CI workflow
(``.github/workflows/tier1-5-test.yml``) only *executes the 13 agents* when a
Tier 0 cache exists at ``scripts/tier0_output_cache/latest.pkl``. That path was
gitignored (``.gitignore`` ``scripts/tier0_output_cache/``) so the cache was
never committed, and on every PR the harness took its graceful-skip branch —
making the check a no-op for its primary purpose (exercising agent + contract
correctness).

The maintainer-decided fix (option a): commit a small, sanitized Tier-0 cache
fixture and un-ignore that single file, so the harness actually runs the 13
agents on every relevant PR. The fixture is built deterministically by
``scripts/generate_tier0_fixture.py`` (the refresh / staleness mechanism).

These tests pin the fixture's contract:

1. it exists at the path CI resolves, is a REAL file (not a symlink), and loads;
2. it satisfies ``Tier0OutputMapper`` (the contract gate) and drives all 13
   ``map_to_*`` methods without raising;
3. it carries the structure the 13 mappers require (rich ``eligible_df``, a
   ``discontinuation_flag == 1`` row, ``feature_names``, etc.);
4. it is small + version-robust (no fitted preprocessor / encoder objects; the
   only model object is a tiny estimator); and
5. the generator can re-produce a valid fixture (so a refresh stays valid).
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = PROJECT_ROOT / "scripts" / "tier0_output_cache" / "latest.pkl"
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "generate_tier0_fixture.py"

# Top-level keys read across the 13 map_to_* methods (see issue #600
# investigation). experiment_id + eligible_df are contract-Required; the rest
# are read by >= 1 mapper.
_REQUIRED_DF_COLUMNS = {
    "discontinuation_flag",
    "hcp_visits",
    "prior_treatments",
    "days_on_therapy",
    "geographic_region",
    "age_group",
}
_THIRTEEN_MAPPINGS = [
    "map_to_orchestrator",
    "map_to_tool_composer",
    "map_to_causal_impact",
    "map_to_gap_analyzer",
    "map_to_heterogeneous_optimizer",
    "map_to_drift_monitor",
    "map_to_experiment_designer",
    "map_to_experiment_monitor",
    "map_to_health_score",
    "map_to_prediction_synthesizer",
    "map_to_resource_optimizer",
    "map_to_explainer",
    "map_to_feedback_learner",
]
# Heavy / pickle-fragile objects that the sanitized fixture must NOT embed
# (fitted sklearn transformers are version-coupled and bloat the cache).
_FORBIDDEN_FRAGILE_KEYS = {"fitted_preprocessor", "categorical_encoding"}


def _load_committed_fixture() -> dict:
    with open(FIXTURE_PATH, "rb") as fh:
        return pickle.load(fh)


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_tier0_fixture", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_fixture_exists_and_is_a_real_file():
    """#600 headline: the fixture must be committed at the CI-resolved path as a
    real file (a committed symlink would dangle across checkouts)."""
    assert FIXTURE_PATH.exists(), (
        f"Committed tier0 fixture missing at {FIXTURE_PATH.relative_to(PROJECT_ROOT)}. "
        "Without it the Tier 1-5 harness graceful-skips on every PR (issue #600)."
    )
    assert not FIXTURE_PATH.is_symlink(), (
        "Committed fixture must be a real file, not a symlink (symlinks to "
        "timestamped caches dangle on a fresh checkout)."
    )
    state = _load_committed_fixture()
    assert isinstance(state, dict), "Fixture must unpickle to a dict (tier0 state)."


def test_committed_fixture_passes_mapper_contract_and_all_13_mappings():
    """The fixture must satisfy the Tier0OutputMapper contract gate and drive
    every one of the 13 map_to_* methods without raising — that is exactly what
    the harness exercises on each PR."""
    from src.testing.tier0_output_mapper import Tier0OutputMapper

    state = _load_committed_fixture()
    mapper = Tier0OutputMapper(state)  # raises if contract gate fails
    for method_name in _THIRTEEN_MAPPINGS:
        out = getattr(mapper, method_name)()
        assert isinstance(out, dict) and out, f"{method_name} produced no mapping"


def test_committed_fixture_has_required_structure():
    """eligible_df must be rich enough for the mappers (numeric effect-modifiers
    for heterogeneous_optimizer, a discontinuation_flag==1 row for
    prediction_synthesizer) and the required scalar keys must be present."""
    state = _load_committed_fixture()

    assert isinstance(state.get("experiment_id"), str) and state["experiment_id"]
    eligible_df = state.get("eligible_df")
    assert isinstance(eligible_df, pd.DataFrame) and len(eligible_df) > 0
    missing = _REQUIRED_DF_COLUMNS - set(eligible_df.columns)
    assert not missing, f"eligible_df missing required columns: {sorted(missing)}"
    assert (eligible_df["discontinuation_flag"] == 1).any(), (
        "eligible_df needs at least one discontinuation_flag==1 row "
        "(prediction_synthesizer selects it)."
    )
    assert isinstance(state.get("feature_names"), list) and state["feature_names"]
    assert isinstance(state.get("validation_metrics"), dict) and state["validation_metrics"]
    # Block-4: split_assignments must be present so downstream consumers reuse
    # them instead of re-deriving splits.
    assert isinstance(state.get("split_assignments"), dict) and state["split_assignments"]


def test_committed_fixture_is_small_and_version_robust():
    """The fixture must stay small (no full real cache) and free of
    version-fragile fitted transformers; the only model object is a tiny
    estimator usable by Tier0ModelClient (numeric predict_proba)."""
    size_mb = FIXTURE_PATH.stat().st_size / (1024 * 1024)
    assert size_mb < 2.0, (
        f"Committed fixture is {size_mb:.2f} MB; keep it < 2 MB (sanitize/shrink)."
    )

    state = _load_committed_fixture()
    present_fragile = _FORBIDDEN_FRAGILE_KEYS & set(state)
    assert not present_fragile, (
        f"Fixture embeds version-fragile fitted objects {sorted(present_fragile)}; "
        "strip them (downstream reads metrics / model_uri, not these)."
    )
    model = state.get("trained_model")
    assert model is not None, "trained_model expected (gives prediction_synthesizer a real client)."
    assert hasattr(model, "predict_proba"), "trained_model must expose predict_proba."


def test_generator_reproduces_a_valid_fixture():
    """Staleness / refresh guard: the committed generator must re-produce a
    contract-valid fixture state (so a refresh never silently rots)."""
    from src.testing.tier0_output_mapper import Tier0OutputMapper

    module = _load_generator_module()
    assert hasattr(module, "build_fixture_state"), (
        "scripts/generate_tier0_fixture.py must expose build_fixture_state()."
    )
    state = module.build_fixture_state()
    assert isinstance(state, dict)
    mapper = Tier0OutputMapper(state)
    for method_name in _THIRTEEN_MAPPINGS:
        assert isinstance(getattr(mapper, method_name)(), dict)
    # The freshly built state must carry the same required structure.
    assert isinstance(state.get("eligible_df"), pd.DataFrame)
    assert not (_FORBIDDEN_FRAGILE_KEYS & set(state))
