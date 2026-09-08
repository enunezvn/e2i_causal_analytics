"""Tests for the Tier-0 runner's deployment-name construction (#1939, #1957).

``scripts/run_tier0_test.py`` used to build the Step-7 deployment name as an
unconditional literal::

    deployment_name = f"kisqali_discontinuation_{experiment_id[:8]}"

That name is not cosmetic — it is handed to ``ModelDeployerAgent`` and reaches
three persistent stores:

* the **MLflow model registry** (``mlflow.register_model(model_uri, name)``),
* **BentoML** (``service_name`` / ``bento_name``, and ``endpoint_name`` =
  ``f"{deployment_name}-{environment}"``),
* the **``ml_deployments`` table** (an ``MLDeployment`` row).

So a Fabhalta, Remibrutinib, CSU or competitor run registered an MLflow model
literally named ``kisqali_discontinuation_*``. The metrics were right; the
labels were wrong, and they outlive the run (#1939).

The name is now derived from ``CONFIG.brand`` / ``CONFIG.target_outcome``, both
of which are free-form ``--brand`` / ``--target`` strings. Free-form text is not
automatically a legal name downstream: BentoML tags must match
``^[a-z0-9]([-._a-z0-9]*[a-z0-9])?$`` and be at most 63 characters, so raw
interpolation of a brand like ``"Xolair (omalizumab)"`` would raise inside
``bentoml build``. These tests pin both halves: the label is right, *and* it
stays a legal name for every store it reaches.

#1957 removed the other half of the old literal. The ``experiment_id[:8]``
suffix looked per-run but was not: the caller builds the id as
``f"tier0_e2e_{uuid4().hex[:8]}"``, so the slice stopped before the UUID began
and the suffix was always the constant ``"tier0_e2"``. It was dropped rather
than made unique — a stable registered-model name is how the MLflow registry is
meant to be used (one model accumulating versions), whereas a genuinely
per-run suffix would register a NEW model per run. The name is now exactly
``<brand>_<outcome>``, and the builder takes no run-scoped argument at all, so
that stability is structural rather than incidental.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Callable

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_tier0_test.py"


@pytest.fixture(scope="module")
def runner_module():
    """Load ``run_tier0_test.py`` via importlib (~1.2 s, ~120 MiB peak).

    Mirrors the seam used by ``test_tier0_verdict.py``. An import failure is
    environment-specific, so we skip; a *missing helper* is a real regression,
    so that fails loudly in the tests below rather than skipping.
    """
    spec = importlib.util.spec_from_file_location("run_tier0_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.skip("Could not build import spec for run_tier0_test")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment-specific
        pytest.skip(f"Could not import run_tier0_test: {exc}")
    return module


@pytest.fixture(scope="module")
def build_deployment_name(runner_module) -> Callable[..., str]:
    """The name builder under test. Absence is a failure, never a skip."""
    assert hasattr(runner_module, "_build_deployment_name"), (
        "run_tier0_test must expose _build_deployment_name(brand, target_outcome); "
        "without it the Step-7 deployment name is a hard-coded "
        "'kisqali_discontinuation_*' literal (issue #1939)"
    )
    return runner_module._build_deployment_name


# ---------------------------------------------------------------------------
# The label is right
# ---------------------------------------------------------------------------


def test_default_config_name_uses_brand_and_target(build_deployment_name, runner_module):
    """The CONFIG defaults (Kisqali / discontinuation_flag) round-trip verbatim."""
    name = build_deployment_name(
        runner_module.CONFIG.brand,
        runner_module.CONFIG.target_outcome,
    )
    assert name == "kisqali_discontinuation_flag"


@pytest.mark.parametrize(
    "brand",
    [
        # Every non-Kisqali brand the runner can actually be given. The first
        # three are reachable with NO --brand flag at all: --regime scenario_b/c
        # auto-syncs CONFIG.brand via _SCENARIO_REGIME_TO_BRAND, and the file's
        # own usage docstring (:12) shows `--brand competitor`.
        "Fabhalta",
        "Remibrutinib",
        "competitor",
        "XOLAIR",
    ],
)
def test_non_kisqali_brand_is_never_labelled_kisqali(build_deployment_name, brand):
    """The #1939 regression: a non-Kisqali run must not register as Kisqali."""
    name = build_deployment_name(brand, "discontinuation_flag")
    assert "kisqali" not in name, (
        f"brand={brand!r} produced {name!r}; this name is written to the MLflow "
        "registry, BentoML and ml_deployments and outlives the run (#1939)"
    )
    assert brand.lower() in name


def test_scenario_regime_brands_all_reachable_without_brand_flag(
    build_deployment_name, runner_module
):
    """Auto-synced regime brands must each label their own deployment."""
    mapping = runner_module._SCENARIO_REGIME_TO_BRAND
    for regime, brand in mapping.items():
        name = build_deployment_name(brand, "discontinuation_flag")
        assert brand.lower() in name, f"regime {regime} (brand {brand}) → {name}"


def test_target_override_replaces_discontinuation_framing(build_deployment_name):
    """`--target treatment_initiated` must not still say 'discontinuation'."""
    name = build_deployment_name("competitor", "treatment_initiated")
    assert "treatment_initiated" in name
    assert "discontinuation" not in name


# ---------------------------------------------------------------------------
# The name is <brand>_<outcome> and nothing else (#1957)
# ---------------------------------------------------------------------------


def test_name_is_exactly_brand_and_outcome(build_deployment_name):
    """No suffix, no prefix, no run id — just the two slugged parts."""
    assert build_deployment_name("Fabhalta", "eskd_progression") == "fabhalta_eskd_progression"


def test_no_constant_experiment_suffix_survives(build_deployment_name, runner_module):
    """The dropped suffix was the constant ``"tier0_e2"`` for every run (#1957)."""
    name = build_deployment_name(
        runner_module.CONFIG.brand,
        runner_module.CONFIG.target_outcome,
    )
    assert "tier0_e2" not in name, (
        f"{name!r} still carries the constant experiment suffix; it discriminated "
        "no run and was dropped in #1957"
    )


def test_builder_takes_no_run_scoped_argument(build_deployment_name):
    """Stability is structural: nothing per-run can reach the name.

    An ``experiment_id`` parameter is what invites the "obvious" one-character
    fix (``[:8]`` → ``[-8:]``) that would register a new MLflow model per run
    and fragment the version history. There is no such parameter to slice.
    """
    params = list(inspect.signature(build_deployment_name).parameters)
    assert params == ["brand", "target_outcome"], (
        f"_build_deployment_name{tuple(params)} takes a run-scoped argument; the "
        "Step-7 name must depend only on brand and outcome (#1957)"
    )


def test_step_7_call_site_passes_only_brand_and_outcome(runner_module):
    """The single call site must match the signature.

    A stale third argument is a ``TypeError`` reachable only in a full Step-7
    run, which no unit test exercises — so pin it at the source.
    """
    source = inspect.getsource(runner_module.step_7_model_deployer)
    call = re.search(r"_build_deployment_name\((?P<args>[^)]*)\)", source)
    assert call is not None, "step_7_model_deployer no longer calls _build_deployment_name"
    args = [a.strip() for a in call.group("args").split(",") if a.strip()]
    assert args == ["CONFIG.brand", "CONFIG.target_outcome"], args


# ---------------------------------------------------------------------------
# The name stays legal for every store it reaches
# ---------------------------------------------------------------------------

# Realistic and hostile ``--brand`` / ``--target`` pairs. Raw interpolation of
# these (the fix originally proposed on #1939) produces an illegal BentoML tag
# for five of them.
_FREE_FORM_CASES = [
    ("Kisqali", "discontinuation_flag"),
    ("competitor", "treatment_initiated"),
    ("XOLAIR", "discontinuation_flag"),
    ("Xolair (omalizumab)", "adherence"),
    ("Kisqali/Ribociclib", "discontinuation_flag"),
    ("Kesimpta 20mg", "TRx Growth"),
    ("Cosentyx", "persistence@12mo"),
    ("  Fabhalta  ", "  eskd_progression  "),
    ("Scemblix®", "mmr_by_12mo"),
    ("", "discontinuation_flag"),
    ("Kisqali", ""),
    ("", ""),
    ("A" * 80, "B" * 80),
]


@pytest.mark.parametrize(("brand", "target"), _FREE_FORM_CASES)
def test_name_is_a_legal_bentoml_tag(build_deployment_name, brand, target):
    """BentoML is the binding constraint — it rejects spaces, '/', '@', caps.

    Uses the real validator (``bentoml._internal.tag`` costs ~0.3 s / ~28 MiB;
    it does not pull the full ``bentoml`` package) so an upstream change to the
    tag grammar surfaces here rather than in a production build.
    """
    tag_mod = pytest.importorskip("bentoml._internal.tag")
    name = build_deployment_name(brand, target)
    tag_mod.validate_tag_str(name)  # raises ValueError on an illegal tag


@pytest.mark.parametrize(("brand", "target"), _FREE_FORM_CASES)
def test_name_is_a_legal_mlflow_registered_model_name(build_deployment_name, brand, target):
    """MLflow rejects '/' and ':' and empty names (mlflow.utils.validation)."""
    name = build_deployment_name(brand, target)
    assert name.strip(), "MLflow rejects an empty registered-model name"
    assert "/" not in name and ":" not in name
    assert ".." not in name  # path_not_unique / bad_path_message


@pytest.mark.parametrize(("brand", "target"), _FREE_FORM_CASES)
def test_name_fits_bentoml_63_character_cap(build_deployment_name, brand, target):
    """A long brand must be trimmed, not passed through to a build failure."""
    name = build_deployment_name(brand, target)
    assert 0 < len(name) <= 63, f"{name!r} is {len(name)} chars"


@pytest.mark.parametrize(("brand", "target"), _FREE_FORM_CASES)
def test_endpoint_name_derived_from_it_is_also_legal(build_deployment_name, brand, target):
    """deployment_orchestrator builds ``f"{deployment_name}-{environment}"``.

    That derived string is what reaches the ``ml_deployments`` row, so it must
    survive the same grammar.
    """
    name = build_deployment_name(brand, target)
    endpoint_name = f"{name}-production"
    assert re.fullmatch(r"[a-z0-9]([-._a-z0-9]*[a-z0-9])?", endpoint_name), endpoint_name


def test_distinct_brands_produce_distinct_names(build_deployment_name):
    """Slugging must not collapse two different brands onto one registry name."""
    a = build_deployment_name("Fabhalta", "discontinuation_flag")
    b = build_deployment_name("Remibrutinib", "discontinuation_flag")
    assert a != b
