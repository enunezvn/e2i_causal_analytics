"""Integration tests for DataPreparerAgent."""

import pytest
import pandas as pd
import numpy as np
from src.agents.ml_foundation.data_preparer import DataPreparerAgent


@pytest.fixture
def mock_scope_spec():
    """Create a mock scope specification."""
    return {
        "experiment_id": "exp_integration_123",
        "problem_type": "binary_classification",
        "prediction_target": "target",
        "required_features": ["feature1", "feature2", "feature3"],
        "minimum_samples": 50,
    }


@pytest.fixture
def sample_data_source():
    """Mock data source name."""
    return "patient_journeys"


@pytest.mark.asyncio
async def test_data_preparer_agent_initialization():
    """Test that DataPreparerAgent initializes correctly."""
    agent = DataPreparerAgent()

    assert agent.tier == 0
    assert agent.tier_name == "ml_foundation"
    assert agent.agent_type == "standard"
    assert agent.sla_seconds == 60
    assert agent.graph is not None


@pytest.mark.asyncio
async def test_data_preparer_agent_missing_scope_spec(sample_data_source):
    """Test that agent raises error when scope_spec is missing."""
    agent = DataPreparerAgent()

    with pytest.raises(ValueError, match="scope_spec is required"):
        await agent.run({
            "data_source": sample_data_source,
            # Missing scope_spec
        })


@pytest.mark.asyncio
async def test_data_preparer_agent_missing_data_source(mock_scope_spec):
    """Test that agent raises error when data_source is missing."""
    agent = DataPreparerAgent()

    with pytest.raises(ValueError, match="data_source is required"):
        await agent.run({
            "scope_spec": mock_scope_spec,
            # Missing data_source
        })


@pytest.mark.asyncio
async def test_data_preparer_agent_missing_experiment_id(sample_data_source):
    """Test that agent raises error when experiment_id missing from scope_spec."""
    agent = DataPreparerAgent()

    with pytest.raises(ValueError, match="experiment_id"):
        await agent.run({
            "scope_spec": {"problem_type": "classification"},  # Missing experiment_id
            "data_source": sample_data_source,
        })


# NOTE: The following tests require actual data loading implementation
# They are marked as TODO for now, to be implemented after data loading is complete


@pytest.mark.skip(reason="Requires data loading implementation")
@pytest.mark.asyncio
async def test_data_preparer_agent_full_pipeline_success(
    mock_scope_spec, sample_data_source
):
    """Test full data preparation pipeline with passing QC.

    TODO: Implement after data loading is completed.
    """
    agent = DataPreparerAgent()

    output = await agent.run({
        "scope_spec": mock_scope_spec,
        "data_source": sample_data_source,
    })

    # Check output structure
    assert "qc_report" in output
    assert "baseline_metrics" in output
    assert "data_readiness" in output
    assert "gate_passed" in output

    # Check QC report
    qc_report = output["qc_report"]
    assert qc_report["experiment_id"] == "exp_integration_123"
    assert qc_report["status"] in ["passed", "warning", "failed", "skipped"]
    assert 0.0 <= qc_report["overall_score"] <= 1.0

    # Check baseline metrics
    baseline = output["baseline_metrics"]
    assert baseline["experiment_id"] == "exp_integration_123"
    assert baseline["split_type"] == "train"
    assert "feature_stats" in baseline

    # Check data readiness
    readiness = output["data_readiness"]
    assert readiness["experiment_id"] == "exp_integration_123"
    assert isinstance(readiness["is_ready"], bool)


@pytest.mark.skip(reason="Requires data loading implementation")
@pytest.mark.asyncio
async def test_data_preparer_agent_qc_gate_blocks_when_failed(
    mock_scope_spec, sample_data_source
):
    """Test that QC gate blocks when data quality fails.

    TODO: Implement after data loading is completed.
    This test should use intentionally bad data to trigger QC failure.
    """
    agent = DataPreparerAgent()

    # Use data source with bad data quality
    output = await agent.run({
        "scope_spec": mock_scope_spec,
        "data_source": "bad_quality_data",  # Hypothetical bad data
    })

    # QC gate should block
    assert output["gate_passed"] is False
    assert output["qc_report"]["status"] == "failed"
    assert len(output["qc_report"]["blocking_issues"]) > 0


@pytest.mark.skip(reason="Requires data loading implementation")
@pytest.mark.asyncio
async def test_data_preparer_agent_leakage_detection_blocks(
    mock_scope_spec, sample_data_source
):
    """Test that detected leakage blocks the QC gate.

    TODO: Implement after data loading is completed.
    This test should use data with known leakage issues.
    """
    agent = DataPreparerAgent()

    # Use data source with leakage
    output = await agent.run({
        "scope_spec": mock_scope_spec,
        "data_source": "data_with_leakage",  # Hypothetical leaky data
    })

    # Gate should be blocked due to leakage
    assert output["gate_passed"] is False
    assert len(output["data_readiness"]["blockers"]) > 0


@pytest.mark.skip(reason="Requires database implementation")
@pytest.mark.asyncio
async def test_data_preparer_agent_persists_to_database(
    mock_scope_spec, sample_data_source
):
    """Test that agent persists outputs to database.

    TODO: Implement after database persistence is added.
    """
    agent = DataPreparerAgent()

    output = await agent.run({
        "scope_spec": mock_scope_spec,
        "data_source": sample_data_source,
    })

    # TODO: Verify that QC report was written to ml_data_quality_reports table
    # TODO: Verify that baseline metrics were written
    # TODO: Verify that features were registered in Feast
    pass


@pytest.mark.skip(reason="Requires Feast integration")
@pytest.mark.asyncio
async def test_data_preparer_agent_registers_features_in_feast(
    mock_scope_spec, sample_data_source
):
    """Test that agent registers features in Feast feature store.

    TODO: Implement after Feast integration is completed.
    """
    agent = DataPreparerAgent()

    output = await agent.run({
        "scope_spec": mock_scope_spec,
        "data_source": sample_data_source,
    })

    # TODO: Verify features were registered in Feast
    # TODO: Verify feature definitions are correct
    pass
