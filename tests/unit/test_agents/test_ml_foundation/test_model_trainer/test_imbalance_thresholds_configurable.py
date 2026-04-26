"""Tests for configurable imbalance severity bands and strategy matrix.

Block 6A (Findings #9, #16): the imbalance strategy is loaded from
``config/imbalance_strategy.yaml`` rather than baked into Python. These
tests prove that swapping the YAML in a tmp_path:

1. Shifts severity-band classification accordingly.
2. Reroutes strategy lookup through the alternate matrix.

The default config (used everywhere else) must already match the legacy
behavior bit-for-bit; that's covered in `test_detect_class_imbalance.py`.
This file is purely about *override* mechanics.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance import (
    _calculate_imbalance_metrics,
    _load_imbalance_config,
    _lookup_strategy,
)

# ============================================================================
# Helpers
# ============================================================================


@pytest.fixture(autouse=True)
def clear_yaml_cache():
    """Clear the lru_cache on `_read_yaml` between tests so tmp_path
    YAML files load fresh.

    Without this, the first test's tmp YAML would shadow subsequent ones
    when path strings collide (extremely unlikely under tmp_path but
    cheap to guard).
    """
    # Import the module (not the re-exported function) by full dotted path
    # so we can reach the cached `_read_yaml` helper.
    import importlib

    mod = importlib.import_module(
        "src.agents.ml_foundation.model_trainer.nodes.detect_class_imbalance"
    )
    mod._read_yaml.cache_clear()
    yield
    mod._read_yaml.cache_clear()


def _write_yaml(path: Path, payload: dict) -> Path:
    """Write a dict as YAML to `path` and return the path."""
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ============================================================================
# Tests
# ============================================================================


class TestSeverityBandsConfigurable:
    """Severity band thresholds must come from the YAML, not module constants."""

    def test_alternate_bands_shift_classification(self, tmp_path: Path):
        """Bumping the `none` threshold to 0.50 reclassifies a 65/35 split
        from `none` to `moderate`.

        Default bands: minority_ratio >= 0.40 -> none.
        Override bands: minority_ratio >= 0.50 -> none. So a 35% minority
        share now lands in `moderate`.
        """
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {
                    "none": 0.50,
                    "moderate": 0.30,
                    "severe": 0.10,
                },
                "tree_models": ["XGBoost"],
                "strategy_matrix": {
                    "none": {
                        "default": {
                            "strategy": "none",
                            "rationale": "balanced",
                        }
                    },
                    "moderate": {
                        "tree": {
                            "strategy": "class_weight",
                            "rationale": "moderate-tree",
                        },
                        "non_tree": {
                            "strategy": "random_oversample",
                            "rationale": "moderate-non-tree",
                        },
                    },
                    "severe": {
                        "tree": {
                            "strategy": "class_weight",
                            "rationale": "severe-tree",
                        },
                        "non_tree": {
                            "strategy": "smote",
                            "rationale": "severe-non-tree",
                        },
                    },
                    "extreme": {
                        "tree": {
                            "strategy": "class_weight",
                            "rationale": "extreme-tree",
                        },
                        "non_tree": {
                            "strategy": "combined",
                            "rationale": "extreme-non-tree",
                        },
                    },
                },
            },
        )

        config = _load_imbalance_config(path=cfg_path)
        assert config["severity_bands"]["none"] == 0.50
        assert config["severity_bands"]["moderate"] == 0.30

        # 65/35 split: minority ratio = 0.35.
        # Default bands -> none; alternate bands (>=0.50) -> moderate.
        y = np.array([0] * 65 + [1] * 35)
        metrics = _calculate_imbalance_metrics(y, config=config)
        assert metrics["minority_ratio"] == 0.35
        assert metrics["severity"] == "moderate", (
            f"Expected 'moderate' under alternate bands, got {metrics['severity']!r}"
        )

    def test_default_bands_classify_45_55_as_none(self, tmp_path: Path):
        """Sanity: under the default config, a 55/45 split (minority=0.45,
        above the 0.40 `none` threshold) is `none`. The previous test
        asserts that an override below 0.50 reclassifies a 65/35 split
        — this is the corresponding default-side proof that the band is
        actually being read from YAML rather than being a hardcoded
        constant.
        """
        config = _load_imbalance_config()
        y = np.array([0] * 55 + [1] * 45)
        metrics = _calculate_imbalance_metrics(y, config=config)
        assert metrics["severity"] == "none"

    def test_band_override_reclassifies_severe_to_extreme(self, tmp_path: Path):
        """Tightening the `severe` band to 0.15 reclassifies a 90/10 split
        from `severe` to `extreme`."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {
                    "none": 0.40,
                    "moderate": 0.20,
                    "severe": 0.15,  # was 0.05 in default
                },
                "tree_models": ["XGBoost"],
                "strategy_matrix": {
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
                    "moderate": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        "non_tree": {"strategy": "random_oversample", "rationale": "x"},
                    },
                    "severe": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        "non_tree": {"strategy": "smote", "rationale": "x"},
                    },
                    "extreme": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        "non_tree": {"strategy": "combined", "rationale": "x"},
                    },
                },
            },
        )

        config = _load_imbalance_config(path=cfg_path)
        # 90/10 -> minority_ratio=0.10, below severe (0.15) -> extreme.
        y = np.array([0] * 90 + [1] * 10)
        metrics = _calculate_imbalance_metrics(y, config=config)
        assert metrics["severity"] == "extreme"


class TestStrategyMatrixConfigurable:
    """The strategy matrix must be authoritative — overriding it changes
    what `_lookup_strategy` returns."""

    def test_custom_strategy_for_extreme_tree(self, tmp_path: Path):
        """Overriding `extreme.tree.strategy` to 'smote_tomek' must be
        honored by the deterministic lookup."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
                "tree_models": ["XGBoost", "RandomForest"],
                "strategy_matrix": {
                    "none": {
                        "default": {"strategy": "none", "rationale": "balanced"}
                    },
                    "moderate": {
                        "tree": {
                            "strategy": "class_weight",
                            "rationale": "mod-tree",
                        },
                        "non_tree": {
                            "strategy": "random_oversample",
                            "rationale": "mod-non-tree",
                        },
                    },
                    "severe": {
                        "tree": {
                            "strategy": "class_weight",
                            "rationale": "sev-tree",
                        },
                        "non_tree": {
                            "strategy": "smote",
                            "rationale": "sev-non-tree",
                        },
                    },
                    "extreme": {
                        # Overridden — was class_weight, now smote_tomek.
                        "tree": {
                            "strategy": "smote_tomek",
                            "rationale": "custom override for extreme-tree",
                        },
                        "non_tree": {
                            "strategy": "combined",
                            "rationale": "ext-non-tree",
                        },
                    },
                },
            },
        )

        config = _load_imbalance_config(path=cfg_path)

        metrics = {
            "severity": "extreme",
            "minority_count": 15,
            "total_samples": 1000,
        }
        strategy, rationale = _lookup_strategy(
            metrics, "XGBoost", "binary_classification", config=config
        )
        assert strategy == "smote_tomek"
        assert rationale == "custom override for extreme-tree"

    def test_unsorted_min_minority_count_rules_still_work(self, tmp_path: Path):
        """The loader must defensively sort `non_tree` rule lists by
        `min_minority_count` descending so the lookup is correct
        regardless of YAML ordering."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
                "tree_models": ["XGBoost"],
                "strategy_matrix": {
                    "none": {
                        "default": {"strategy": "none", "rationale": "x"}
                    },
                    "moderate": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        "non_tree": {
                            "strategy": "random_oversample",
                            "rationale": "x",
                        },
                    },
                    "severe": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        "non_tree": {"strategy": "smote", "rationale": "x"},
                    },
                    "extreme": {
                        "tree": {"strategy": "class_weight", "rationale": "x"},
                        # Deliberately unsorted (ascending).
                        "non_tree": [
                            {
                                "min_minority_count": 0,
                                "strategy": "class_weight",
                                "rationale": "tiny",
                            },
                            {
                                "min_minority_count": 5,
                                "strategy": "random_oversample",
                                "rationale": "small",
                            },
                            {
                                "min_minority_count": 10,
                                "strategy": "combined",
                                "rationale": "enough",
                            },
                        ],
                    },
                },
            },
        )

        config = _load_imbalance_config(path=cfg_path)

        # mc=15 -> rule with min=10 wins -> combined
        s_high, _ = _lookup_strategy(
            {"severity": "extreme", "minority_count": 15, "total_samples": 100},
            "LogisticRegression",
            "binary_classification",
            config=config,
        )
        # mc=7 -> rule with min=5 wins -> random_oversample
        s_mid, _ = _lookup_strategy(
            {"severity": "extreme", "minority_count": 7, "total_samples": 100},
            "LogisticRegression",
            "binary_classification",
            config=config,
        )
        # mc=2 -> rule with min=0 wins -> class_weight
        s_low, _ = _lookup_strategy(
            {"severity": "extreme", "minority_count": 2, "total_samples": 100},
            "LogisticRegression",
            "binary_classification",
            config=config,
        )

        assert s_high == "combined"
        assert s_mid == "random_oversample"
        assert s_low == "class_weight"

    def test_loader_rejects_missing_keys(self, tmp_path: Path):
        """A YAML missing required top-level keys must raise ValueError."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                # Missing tree_models and strategy_matrix.
                "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
            },
        )

        with pytest.raises(ValueError, match="missing required key"):
            _load_imbalance_config(path=cfg_path)

    def test_loader_rejects_incomplete_severity_bands(self, tmp_path: Path):
        """A YAML missing severity bands must raise ValueError."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {"none": 0.40},  # missing moderate, severe
                "tree_models": [],
                "strategy_matrix": {},
            },
        )

        with pytest.raises(ValueError, match="severity_bands must define"):
            _load_imbalance_config(path=cfg_path)
