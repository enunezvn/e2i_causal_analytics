"""Shard 07: durable PromptBundle install path for recipients (F2 install-half)."""

from __future__ import annotations


def test_save_and_load_bundle_roundtrip(tmp_path, monkeypatch):
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle, save_prompt_bundle

    monkeypatch.chdir(tmp_path)
    path = save_prompt_bundle(
        "experiment_monitor",
        templates={"srm_template": "OPTIMIZED srm for {experiment_name} p={p_value:.4f}"},
        score=0.91,
    )
    assert path
    bundle = load_prompt_bundle("experiment_monitor")
    assert bundle is not None
    assert bundle["templates"]["srm_template"].startswith("OPTIMIZED srm")
    assert bundle["score"] == 0.91


def test_load_missing_bundle_returns_none(tmp_path, monkeypatch):
    from src.agents.feedback_learner.prompt_bundles import load_prompt_bundle

    monkeypatch.chdir(tmp_path)
    assert load_prompt_bundle("explainer") is None


def test_install_updates_experiment_monitor_template(tmp_path, monkeypatch):
    from src.agents.experiment_monitor import dspy_integration as em
    from src.agents.feedback_learner.prompt_bundles import (
        install_prompt_bundle,
        save_prompt_bundle,
    )

    monkeypatch.chdir(tmp_path)
    # reset the recipient singleton to defaults for a clean assertion
    if hasattr(em, "reset_dspy_integration"):
        em.reset_dspy_integration()

    save_prompt_bundle(
        "experiment_monitor",
        templates={
            "srm_template": "OPT {experiment_name} chi={chi_squared:.2f} p={p_value:.4f} "
            "exp={expected_ratio} act={actual_counts}"
        },
        score=0.88,
    )
    ok = install_prompt_bundle("experiment_monitor")
    assert ok is True

    integ = em.get_experiment_monitor_dspy_integration()
    assert integ.prompts.srm_template.startswith("OPT ")
    assert integ.prompts.optimization_score == 0.88
    # The getter the alert_generator calls now serves the optimized template.
    msg = integ.get_srm_prompt(
        experiment_name="E1",
        chi_squared=12.3,
        p_value=0.0004,
        expected_ratio="50/50",
        actual_counts="600/400",
    )
    assert msg.startswith("OPT E1")
    # leave the singleton clean for other tests
    if hasattr(em, "reset_dspy_integration"):
        em.reset_dspy_integration()


def test_install_all_is_safe_with_no_bundles(tmp_path, monkeypatch):
    from src.agents.feedback_learner.prompt_bundles import install_all_prompt_bundles

    monkeypatch.chdir(tmp_path)
    results = install_all_prompt_bundles()
    # No bundles on disk -> every install is a no-op False, but never raises.
    assert set(results.keys()) == {
        "experiment_monitor",
        "resource_optimizer",
        "explainer",
        "health_score",
    }
    assert all(v is False for v in results.values())
