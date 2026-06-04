from __future__ import annotations

from pathlib import Path

import yaml


def _config_path() -> Path:
    # Walk up from this file until we find the directory containing config/
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "digital_twin_config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate config/digital_twin_config.yaml")


def test_config_has_no_orphaned_intervention_effects_and_has_thresholds():
    cfg = yaml.safe_load(_config_path().read_text())
    # The config is structured as sections at the top level; the simulation
    # section previously held the drifted intervention_effects block.
    dt = cfg["simulation"]
    # The drifted, never-read intervention_effects block is removed.
    assert "intervention_effects" not in dt
    # Calibrated effect-engine thresholds are present.
    eng = dt["effect_engine"]
    assert "min_effect_threshold" in eng
    assert "selected_learner" in eng
    assert eng["selected_learner"] in {"uplift_random_forest", "uplift_gradient_boosting"}
