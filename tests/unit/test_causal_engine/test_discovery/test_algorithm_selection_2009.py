"""LiNGAM-on-binary guard (#2009).

Premise (measured 2026-09-11, ``docs/demos/results/2026-09-11_pc_indep_test/``):
live discovery is PC on 41/41 runs, and every platform outcome is a 0/1 flag.
The LiNGAM wrappers (DirectLiNGAM, ICA-LiNGAM) assume linear relationships
with non-Gaussian CONTINUOUS errors; a Bernoulli column violates that model, so
a LiNGAM run on a live frame would fit and return a plausible-looking but
mis-specified DAG with ``converged=True``. The guard refuses such a frame with
``DiscoveryError`` naming the algorithm and the offending columns — never a
silent drop, never a fallback to PC.

Frames follow ``test_structural_recovery._make_frame`` (binary treatment and
outcome, continuous covariates) at a size where PC itself is cheap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.discovery.algorithms.lingam_wrapper import (
    DirectLiNGAMAlgorithm,
    ICALiNGAMAlgorithm,
    binary_columns,
)
from src.causal_engine.discovery.algorithms.pc_wrapper import PCAlgorithm
from src.causal_engine.discovery.base import (
    AlgorithmResult,
    DiscoveryAlgorithmType,
    DiscoveryConfig,
)
from src.causal_engine.discovery.runner import DiscoveryRunner
from src.causal_engine.errors import CausalEngineError, DiscoveryError

TREATMENT = "treatment_arm"
OUTCOME = "persistent_180d"
BINARY_COLUMNS = (TREATMENT, OUTCOME)
CONTINUOUS_COLUMNS = ("disease_severity", "prognostic_only")


def _mixed_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Binary treatment/outcome plus continuous covariates, the live shape."""
    rng = np.random.default_rng(seed)
    severity = rng.normal(0.0, 1.0, n)
    prognostic = rng.normal(0.0, 1.0, n)
    logit_t = -0.2 + 0.9 * severity
    treatment = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_t)), n).astype(float)
    logit_y = -0.3 + 0.8 * treatment + 0.8 * severity + 0.6 * prognostic
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_y)), n).astype(float)
    return pd.DataFrame(
        {
            TREATMENT: treatment,
            OUTCOME: outcome,
            "disease_severity": severity,
            "prognostic_only": prognostic,
        }
    )


def _continuous_frame(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Linear, non-Gaussian (uniform) errors — the shape LiNGAM is specified for."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.8 * x + rng.uniform(-1.0, 1.0, n)
    z = 0.5 * y + rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _lingam_config(algorithm: DiscoveryAlgorithmType) -> DiscoveryConfig:
    return DiscoveryConfig(algorithms=[algorithm], max_iter=100)


class TestLingamRefusesBinaryColumns:
    """(a)/(b): a binary column is refused by name, for both LiNGAM variants."""

    @pytest.mark.parametrize(
        ("algorithm", "algo_type"),
        [
            (DirectLiNGAMAlgorithm(), DiscoveryAlgorithmType.DIRECT_LINGAM),
            (ICALiNGAMAlgorithm(), DiscoveryAlgorithmType.ICA_LINGAM),
        ],
        ids=["direct_lingam", "ica_lingam"],
    )
    def test_binary_frame_raises_discovery_error_naming_algorithm_and_columns(
        self,
        algorithm: DirectLiNGAMAlgorithm | ICALiNGAMAlgorithm,
        algo_type: DiscoveryAlgorithmType,
    ) -> None:
        frame = _mixed_frame()
        with pytest.raises(DiscoveryError) as excinfo:
            algorithm.discover(frame, _lingam_config(algo_type))

        message = str(excinfo.value)
        assert algo_type.value in message
        for column in BINARY_COLUMNS:
            assert column in message
        for column in CONTINUOUS_COLUMNS:
            assert column not in message

        # Structured details for the agent error path (CausalEngineError family).
        assert isinstance(excinfo.value, CausalEngineError)
        assert excinfo.value.details["algorithm"] == algo_type.value
        assert excinfo.value.details["binary_columns"] == list(BINARY_COLUMNS)

    def test_single_binary_column_is_enough_to_refuse(self) -> None:
        frame = _continuous_frame()
        frame["flag"] = (frame["x"] > 0).astype(float)
        with pytest.raises(DiscoveryError, match="flag"):
            DirectLiNGAMAlgorithm().discover(
                frame, _lingam_config(DiscoveryAlgorithmType.DIRECT_LINGAM)
            )

    def test_constant_column_counts_as_binary(self) -> None:
        """<= 2 distinct non-null values: a constant column is refused too (it
        is degenerate for LiNGAM's regression-on-residuals in any case)."""
        frame = _continuous_frame()
        frame["always_one"] = 1.0
        assert binary_columns(frame) == ["always_one"]
        with pytest.raises(DiscoveryError, match="always_one"):
            ICALiNGAMAlgorithm().discover(frame, _lingam_config(DiscoveryAlgorithmType.ICA_LINGAM))


class TestGuardLeavesOtherPathsAlone:
    """(c)/(d): the guard is specific to LiNGAM on binary columns."""

    @pytest.mark.parametrize(
        ("algorithm", "algo_type"),
        [
            (DirectLiNGAMAlgorithm(), DiscoveryAlgorithmType.DIRECT_LINGAM),
            (ICALiNGAMAlgorithm(), DiscoveryAlgorithmType.ICA_LINGAM),
        ],
        ids=["direct_lingam", "ica_lingam"],
    )
    def test_continuous_frame_is_not_refused_by_the_guard(
        self,
        algorithm: DirectLiNGAMAlgorithm | ICALiNGAMAlgorithm,
        algo_type: DiscoveryAlgorithmType,
    ) -> None:
        """The wrapper may still run or fail for its own reasons (the ``lingam``
        package is not a project dependency, so today it returns
        ``converged=False`` with an install hint); the only assertion is that
        the guard did not fire."""
        frame = _continuous_frame()
        assert binary_columns(frame) == []
        result = algorithm.discover(frame, _lingam_config(algo_type))
        assert isinstance(result, AlgorithmResult)
        assert "binary" not in str(result.metadata.get("error", ""))

    def test_pc_on_a_binary_frame_is_unaffected(self) -> None:
        frame = _mixed_frame()
        result = PCAlgorithm().discover(
            frame, DiscoveryConfig(algorithms=[DiscoveryAlgorithmType.PC])
        )
        assert result.converged is True
        assert "error" not in result.metadata


class TestAllNullColumnIsNotBinary:
    """(e): zero distinct NON-NULL values is not binary. Pinned behaviour: the
    column finder ignores it, and ``discover`` refuses the frame one step
    earlier with the base class's missing-values ``ValueError`` — so an
    all-null column never reaches the LiNGAM guard, and never masks it."""

    def test_column_finder_ignores_an_all_null_column(self) -> None:
        frame = _mixed_frame()
        frame["unobserved"] = np.nan
        assert binary_columns(frame) == list(BINARY_COLUMNS)

    def test_all_null_column_fails_the_missing_values_check_first(self) -> None:
        frame = _continuous_frame()
        frame["unobserved"] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            DirectLiNGAMAlgorithm().discover(
                frame, _lingam_config(DiscoveryAlgorithmType.DIRECT_LINGAM)
            )


class TestRunnerSurfacesTheRefusal:
    """Through ``DiscoveryRunner`` the wrapper's exception becomes a FAILED
    algorithm result (the runner's #1978 policy: converged=False, the gate
    scores no evidence) whose error text still names the algorithm and the
    columns. Not a silent drop, and no PC fallback: the requested algorithm
    is the only one that ran."""

    async def test_refusal_is_recorded_on_the_failed_result(self) -> None:
        frame = _mixed_frame()
        result = await DiscoveryRunner().discover_dag(
            frame, _lingam_config(DiscoveryAlgorithmType.DIRECT_LINGAM)
        )
        assert [r.algorithm for r in result.algorithm_results] == [
            DiscoveryAlgorithmType.DIRECT_LINGAM
        ]
        failed = result.algorithm_results[0]
        assert failed.converged is False
        error = failed.metadata["error"]
        assert DiscoveryAlgorithmType.DIRECT_LINGAM.value in error
        for column in BINARY_COLUMNS:
            assert column in error
        assert result.edges == []
