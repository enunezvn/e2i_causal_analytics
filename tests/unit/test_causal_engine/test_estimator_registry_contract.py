"""One estimator registry drives construction, API, agent, and refutation seams."""

from src.api.routes.causal.activity import _ESTIMATOR_REGISTRY
from src.api.schemas.causal import AGENT_FORCEABLE_ESTIMATORS
from src.causal_engine.energy_score.estimator_selector import (
    ESTIMATOR_WRAPPERS,
    EstimatorSelectorConfig,
)
from src.causal_engine.estimator_registry import (
    DEFAULT_ESTIMATOR_SPECS,
    DOWHY_METHOD_BY_LABEL,
    ESTIMATOR_SPECS,
    EstimatorType,
)


def test_every_spec_drives_a_wrapper_and_public_catalog_entry():
    public_names = {item.name for item in _ESTIMATOR_REGISTRY}
    assert {spec.estimator_type for spec in ESTIMATOR_SPECS} == set(ESTIMATOR_WRAPPERS)
    assert {
        spec.public_name or spec.estimator_type.value for spec in ESTIMATOR_SPECS
    } <= public_names


def test_default_chain_is_generated_from_registry_priorities():
    assert [item.estimator_type for item in EstimatorSelectorConfig().estimators] == [
        spec.estimator_type for spec in DEFAULT_ESTIMATOR_SPECS
    ]
    assert EstimatorType.DML_LEARNER not in {
        spec.estimator_type for spec in DEFAULT_ESTIMATOR_SPECS
    }


def test_forceable_and_refutation_contracts_are_generated_exactly():
    expected_forceable = {
        spec.forceable_alias for spec in ESTIMATOR_SPECS if spec.forceable_alias is not None
    }
    assert expected_forceable <= set(AGENT_FORCEABLE_ESTIMATORS)
    assert DOWHY_METHOD_BY_LABEL["dml_learner"] == "backdoor.econml.dml.DML"
    assert "DML" not in DOWHY_METHOD_BY_LABEL
