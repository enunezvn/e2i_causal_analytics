"""Shard 09: per-recipient optimizer — materialization preserves placeholders."""

from __future__ import annotations


def test_materialize_preserves_placeholders():
    from src.agents.feedback_learner.recipient_optimizer import materialize_template

    current = (
        "Describe Sample Ratio Mismatch for experiment '{experiment_name}'. "
        "Chi-squared: {chi_squared:.2f}, p-value: {p_value:.4f}."
    )
    improved_instruction = "Be concise and lead with the action a PM must take."
    out = materialize_template(current, improved_instruction)
    # Every placeholder field in the original MUST survive (recipient .format() needs them).
    for field in ["experiment_name", "chi_squared", "p_value"]:
        assert "{" + field in out
    # The improved guidance is incorporated.
    assert "action" in out.lower() or improved_instruction[:10].lower() in out.lower()


def test_materialize_rejects_dropping_placeholders():
    from src.agents.feedback_learner.recipient_optimizer import (
        extract_placeholders,
        validate_materialized,
    )

    original = "x {a} y {b:.2f}"
    assert extract_placeholders(original) == {"a", "b"}
    assert validate_materialized(original, "x {a} y {b:.2f} improved") is True
    assert validate_materialized(original, "x {a} only") is False  # dropped {b} -> invalid


def test_produce_bundle_from_current_templates(tmp_path, monkeypatch):
    from src.agents.experiment_monitor.dspy_integration import ExperimentMonitorPrompts
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle
    from src.agents.feedback_learner.recipient_optimizer import (
        produce_bundle_from_instructions,
    )

    monkeypatch.chdir(tmp_path)
    current = ExperimentMonitorPrompts().to_dict()
    instructions = {"srm_template": "Lead with the PM action.", "summary_template": "Be terse."}
    path = produce_bundle_from_instructions(
        "experiment_monitor", current_templates=current, instructions=instructions, score=0.8
    )
    assert path
    bundle = load_prompt_bundle("experiment_monitor")
    assert "{experiment_name" in bundle["templates"]["srm_template"]  # placeholder preserved
    # A field without an instruction is kept verbatim (not dropped).
    assert "alert_template" in bundle["templates"]


def test_produce_bundle_never_breaks_format(tmp_path, monkeypatch):
    """Every materialized template must still .format() with the recipient's kwargs."""
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle
    from src.agents.feedback_learner.recipient_optimizer import produce_bundle_from_instructions

    monkeypatch.chdir(tmp_path)
    current = {"srm_template": "Exp {experiment_name} chi={chi_squared:.2f} p={p_value:.4f}"}
    produce_bundle_from_instructions(
        "experiment_monitor",
        current_templates=current,
        instructions={"srm_template": "Lead with the action."},
        score=0.9,
    )
    bundle = load_prompt_bundle("experiment_monitor")
    # Must not raise KeyError -> placeholders all preserved.
    rendered = bundle["templates"]["srm_template"].format(
        experiment_name="E", chi_squared=1.2, p_value=0.03
    )
    assert "E" in rendered


def test_golden_seeds_have_inputs_set():
    """Golden seed examples must declare their input fields (with_inputs)."""
    import pytest

    pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_seeds import default_example_provider

    provider = default_example_provider("experiment_monitor")
    srm = provider("srm_template")
    assert len(srm) >= 2
    assert "experiment_name" in srm[0].inputs()


def test_scheduled_task_invokes_recipient_optimizer():
    """The Shard 08 task must call the Shard 09 recipient optimizer (guarded)."""
    import inspect

    from src.tasks import dspy_optimization_tasks as t

    src = inspect.getsource(t._run)
    assert "optimize_and_save_recipient" in src
    assert "install_all_prompt_bundles" in src


def test_wrap_metric_coerces_dict_to_prediction():
    """_wrap_metric must coerce a plain-dict metric to dspy.Prediction (prevents
    the int+dict crash in GEPA's valset Evaluate). Deterministic, no LM."""
    import pytest

    dspy = pytest.importorskip("dspy")
    from src.agents.feedback_learner.recipient_optimizer import _wrap_metric

    def dict_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return {"score": 0.75, "feedback": "ok"}

    def scalar_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 0.5

    wrapped_d = _wrap_metric(dict_metric)(dspy.Example(), dspy.Prediction())
    wrapped_s = _wrap_metric(scalar_metric)(dspy.Example(), dspy.Prediction())
    assert isinstance(wrapped_d, dspy.Prediction) and wrapped_d.score == 0.75
    assert isinstance(wrapped_s, dspy.Prediction) and wrapped_s.score == 0.5
    # The crash this prevents: summing the raw dict returns (0 + dict).
    with pytest.raises(TypeError):
        _ = 0 + dict_metric(None, None)
