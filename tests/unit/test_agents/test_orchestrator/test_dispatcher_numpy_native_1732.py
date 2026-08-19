"""#1732: numpy scalars in an agent's result must not reach checkpointed state.

The orchestrator graph checkpoints ``agent_results`` through langgraph's
``JsonPlusSerializer`` (ormsgpack), which raises
``TypeError: Type is not msgpack serializable: numpy.float64`` on any numpy
scalar — aborting the WHOLE ``orchestrator.run()`` after the agent already
completed (live req ``dbbf8f5d``: het optimizer finished a 2-minute analysis,
then the run died at the dispatch→synthesize superstep and the chat answer
disclaimed the results). ``_normalize_agent_result`` is the single seam every
agent's raw output passes through before entering state, so coercion happens
there; ``_calculate_heterogeneity`` is the proven at-source leak and must
return a builtin float.
"""

import numpy as np
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.agents.heterogeneous_optimizer.nodes.cate_estimator import CATEEstimatorNode
from src.agents.orchestrator.nodes.dispatcher import _normalize_agent_result


def _het_shaped_raw_result() -> dict:
    """Raw output shaped like the live het result that killed req dbbf8f5d."""
    return {
        "status": "completed",
        "heterogeneity_score": np.float64(0.6065),
        "overall_ate": np.float64(0.1234),
        "n_segments": np.int64(3),
        "significant": np.bool_(True),
        "cate_individual_sample": np.array([0.1, 0.2, 0.3]),
        "cate_by_segment": [
            {
                "segment": "academic_hcp",
                "cate": np.float64(0.21),
                "n": np.int64(2879),
                "ci": (np.float64(0.05), np.float64(0.37)),
            }
        ],
        "feature_importance": {"disease_severity": np.float64(0.44)},
    }


class TestNormalizeAgentResultNumpyNative:
    def test_normalized_result_serializes_under_production_serde(self):
        """The exact production failure: ormsgpack must accept the result."""
        normalized = _normalize_agent_result(_het_shaped_raw_result())
        # Raises TypeError("Type is not msgpack serializable: numpy.float64")
        # on the pre-fix passthrough.
        JsonPlusSerializer().dumps_typed(normalized)

    def test_scalars_become_builtin_types_with_values_intact(self):
        normalized = _normalize_agent_result(_het_shaped_raw_result())

        assert type(normalized["heterogeneity_score"]) is float
        assert normalized["heterogeneity_score"] == 0.6065
        assert type(normalized["n_segments"]) is int
        assert normalized["n_segments"] == 3
        assert type(normalized["significant"]) is bool
        assert normalized["significant"] is True

        seg = normalized["cate_by_segment"][0]
        assert type(seg["cate"]) is float and seg["cate"] == 0.21
        assert type(seg["n"]) is int and seg["n"] == 2879
        assert all(type(v) is float for v in seg["ci"])
        assert tuple(seg["ci"]) == (0.05, 0.37)

        assert normalized["cate_individual_sample"] == [0.1, 0.2, 0.3]
        assert type(normalized["feature_importance"]["disease_severity"]) is float

    def test_dataclass_result_also_coerced(self):
        """The __dict__ flattening path must coerce too, not just the dict path."""

        class FakeOutput:
            def __init__(self):
                self.score = np.float64(0.5)
                self.count = np.int64(7)

        normalized = _normalize_agent_result(FakeOutput())
        assert type(normalized["score"]) is float
        assert type(normalized["count"]) is int
        JsonPlusSerializer().dumps_typed(normalized)

    def test_native_result_passes_through_unchanged(self):
        raw = {"status": "completed", "value": 1.5, "items": [{"k": "v"}]}
        assert _normalize_agent_result(raw) == raw

    def test_zero_dim_array_coerced_not_crashed(self):
        """0-d ndarray.tolist() returns a bare scalar, not a list — iterating
        it raised TypeError, and _dispatch_agent's broad try would turn the
        completed agent into a failed AgentResult (codex iter-1 finding 1)."""
        raw = {"status": "completed", "ate_scalar": np.array(0.5)}
        normalized = _normalize_agent_result(raw)
        assert type(normalized["ate_scalar"]) is float
        assert normalized["ate_scalar"] == 0.5
        JsonPlusSerializer().dumps_typed(normalized)

    def test_numpy_dict_keys_coerced(self):
        """Dict KEYS must be native too, not just values (codex iter-1
        finding 2) — a numpy-keyed dict still breaks the seam contract."""
        raw = {"by_segment": {np.int64(1): np.float64(0.2), np.str_("k"): 1.0}}
        normalized = _normalize_agent_result(raw)
        assert all(type(k) in (int, str) for k in normalized["by_segment"])
        assert normalized["by_segment"][1] == 0.2
        assert normalized["by_segment"]["k"] == 1.0
        JsonPlusSerializer().dumps_typed(normalized)

    def test_object_dtype_array_elements_coerced(self):
        """Guard for the ndarray branch: object-dtype arrays keep numpy
        scalars through tolist(), so recursion must coerce the elements."""
        raw = {"vals": np.array([np.float64(0.1), np.int64(2)], dtype=object)}
        normalized = _normalize_agent_result(raw)
        assert normalized["vals"] == [0.1, 2]
        assert type(normalized["vals"][0]) is float
        assert type(normalized["vals"][1]) is int
        JsonPlusSerializer().dumps_typed(normalized)


class TestHeterogeneityScoreAtSource:
    def test_calculate_heterogeneity_returns_builtin_float(self):
        """np.std/abs(ate) yields numpy.float64; min() preserves it — the
        cast(float, ...) is a typing no-op. The value must be a real float."""
        node = CATEEstimatorNode()
        score = node._calculate_heterogeneity(np.array([0.1, 0.5, 0.9]), 0.5)
        assert type(score) is float
        # Pinned: std([0.1,0.5,0.9]) = 0.32659..., cv = 0.65319..., /2 = 0.32659...
        assert score == float(min((np.std([0.1, 0.5, 0.9]) / 0.5) / 2, 1.0))
