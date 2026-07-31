"""Red-first tests for the two defects the #1351 live verify unmasked (PR #1390 follow-up).

1. Subgraph checkpoint inheritance: /chat/stream runs the chatbot graph with a
   Redis checkpointer; LangGraph propagates the parent config into any child
   ``graph.ainvoke`` issued inside a node, so a bare ``workflow.compile()``
   child graph inherits the parent's checkpointer. The causal_impact /
   heterogeneous_optimizer / gap_analyzer graphs carry pandas DataFrames in
   state (estimation frames / tier0 passthrough), and the checkpoint serde is
   ormsgpack: the live turn died with ``Type is not msgpack serializable:
   DataFrame`` in <1s. ``compile(checkpointer=False)`` is LangGraph's designed
   knob for "never checkpoint, even as a subgraph" — these tests pin it on the
   three DataFrame-carrying graphs.

2. Categorical confounders: the resolver binds real substrate driver columns
   (trigger_type, delivery_channel, ...) which are strings; every estimator
   failed with ``could not convert string to float`` and the run failed
   closed. The estimation design matrix must one-hot encode categorical
   covariates (standard design-matrix practice, real math) and fail closed on
   absurd-cardinality string columns (an id leak, not a confounder).
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.graph import create_causal_impact_graph
from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
from src.agents.heterogeneous_optimizer.graph import create_heterogeneous_optimizer_graph


class TestSubgraphCheckpointDetachment:
    """DataFrame-carrying agent graphs must never inherit a parent checkpointer."""

    def test_causal_impact_graph_detached(self):
        graph = create_causal_impact_graph(enable_checkpointing=False)
        assert graph.checkpointer is False, (
            "bare compile() inherits the chatbot graph's Redis checkpointer as a "
            "subgraph; state carries an estimation DataFrame -> ormsgpack TypeError"
        )

    def test_heterogeneous_optimizer_graph_detached(self):
        graph = create_heterogeneous_optimizer_graph()
        assert graph.checkpointer is False

    def test_gap_analyzer_graph_detached(self):
        graph = create_gap_analyzer_graph()
        assert graph.checkpointer is False


def _frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    channel = rng.choice(["mobile", "email", "call"], size=n)
    trigger = rng.choice(["crm", "rules"], size=n)
    score = rng.normal(0.5, 0.15, size=n)
    treated = (score + (channel == "mobile") * 0.2 + rng.normal(0, 0.1, n)) > 0.55
    outcome = (
        0.3 * treated.astype(float)
        + 0.2 * (trigger == "crm").astype(float)
        + score
        + rng.normal(0, 0.1, n)
    )
    return pd.DataFrame(
        {
            "accepted": treated.astype(int),
            "converted": outcome,
            "delivery_channel": channel,
            "trigger_type": trigger,
            "confidence_score": score,
        }
    )


class TestCategoricalCovariates:
    """String confounders must be one-hot encoded, not crash every estimator."""

    def test_categorical_confounders_estimate(self):
        node = EstimationNode()
        result, _selection, _latency = node._select_estimator_with_energy_score(
            data=_frame(),
            treatment="accepted",
            outcome="converted",
            adjustment_set=["delivery_channel", "trigger_type", "confidence_score"],
            explicit_method="ols",
        )
        ate = result["ate"] if isinstance(result, dict) else result.ate
        assert ate is not None
        assert np.isfinite(ate)

    def test_absurd_cardinality_fails_closed(self):
        df = _frame()
        df["hcp_id"] = [f"hcp-{i}" for i in range(len(df))]  # id leak, not a confounder
        node = EstimationNode()
        with pytest.raises(Exception, match="hcp_id"):
            node._select_estimator_with_energy_score(
                data=df,
                treatment="accepted",
                outcome="converted",
                adjustment_set=["delivery_channel", "hcp_id"],
                explicit_method="ols",
            )
