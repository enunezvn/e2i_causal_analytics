"""Shard 09 Task 4: observability span slice (observability_connector leakage-test
substrate) + learning_signals feedback fuel (feedback_learner F15). Enum-exact
agent_name/agent_tier and signal_type/rated_agent; is_synthetic-tagged."""

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.feedback_generator import FeedbackGenerator
from src.ml.synthetic.generators.observability_generator import ObservabilityGenerator

_AGENTS = {
    "gap_analyzer",
    "causal_impact",
    "tool_composer",
    "drift_monitor",
    "experiment_monitor",
}
_TIERS = {
    "ml_foundation",
    "coordination",
    "causal_analytics",
    "monitoring",
    "ml_predictions",
    "self_improvement",
}


def test_spans_enum_safe_and_tagged():
    df = ObservabilityGenerator(GeneratorConfig(seed=3, n_records=40)).generate()
    assert len(df) == 40
    assert set(df["agent_name"]).issubset(_AGENTS)
    assert set(df["agent_tier"]).issubset(_TIERS)
    assert df["is_synthetic"].all()
    assert df["total_tokens"].ge(0).all()


def test_feedback_signals_enum_safe_training_examples():
    df = FeedbackGenerator(GeneratorConfig(seed=4, n_records=30)).generate()
    assert set(df["signal_type"]).issubset(
        {
            "thumbs_up",
            "thumbs_down",
            "correction",
            "rating",
            "implicit_positive",
            "implicit_negative",
        }
    )
    assert set(df["rated_agent"]).issubset(
        {"gap_analyzer", "causal_impact", "tool_composer", "heterogeneous_optimizer"}
    )
    assert df["is_training_example"].any()
    assert df["reward"].between(0.0, 1.0).all()  # GEPA reward>=0.5 fuel band
    assert df["dspy_metric_value"].notna().all()
    assert df["is_synthetic"].all()
