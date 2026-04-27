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
                    "none": {"default": {"strategy": "none", "rationale": "balanced"}},
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
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
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


# ============================================================================
# Loader hardening — load-time validation contracts (6A polish)
# ============================================================================


class TestLoaderValidatesSeverityBandOrdering:
    """severity_bands must satisfy none > moderate > severe > 0 at load time."""

    def test_rejects_inverted_ordering(self, tmp_path: Path):
        """An inverted YAML (none < moderate) is nonsense and must fail
        loudly rather than producing wrong classifications at runtime."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {
                    "none": 0.10,
                    "moderate": 0.40,
                    "severe": 0.05,
                },
                "tree_models": [],
                "strategy_matrix": {
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
                    "moderate": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                    "severe": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                    "extreme": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                },
            },
        )

        with pytest.raises(ValueError, match="none > moderate > severe > 0"):
            _load_imbalance_config(path=cfg_path)

    def test_rejects_zero_severe_band(self, tmp_path: Path):
        """severe must be strictly positive — a 0 collapses the extreme
        band into a degenerate range."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {
                    "none": 0.40,
                    "moderate": 0.20,
                    "severe": 0.0,
                },
                "tree_models": [],
                "strategy_matrix": {
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
                    "moderate": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                    "severe": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                    "extreme": {"tree": {"strategy": "class_weight", "rationale": "x"}},
                },
            },
        )

        with pytest.raises(ValueError, match="none > moderate > severe > 0"):
            _load_imbalance_config(path=cfg_path)


class TestLoaderValidatesNonTreeRules:
    """list-form non_tree branches must be valid before reaching the resolver."""

    def _base_payload(self) -> dict:
        """Helper: a minimum-valid YAML, ready to be perturbed per test."""
        return {
            "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
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
        }

    def test_rejects_empty_list_branch(self, tmp_path: Path):
        """An empty list-form branch must be caught at load time, not at
        the first runtime query (where it would IndexError)."""
        payload = self._base_payload()
        payload["strategy_matrix"]["extreme"]["non_tree"] = []
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(cfg_path, payload)

        with pytest.raises(ValueError, match="empty list"):
            _load_imbalance_config(path=cfg_path)

    def test_rejects_rule_missing_strategy_key(self, tmp_path: Path):
        """A list rule missing `strategy` must surface as a ValueError
        from the loader, not as a KeyError from the resolver."""
        payload = self._base_payload()
        payload["strategy_matrix"]["extreme"]["non_tree"] = [
            {
                "min_minority_count": 0,
                # Missing: strategy
                "rationale": "catch-all",
            },
        ]
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(cfg_path, payload)

        with pytest.raises(ValueError, match="missing required key"):
            _load_imbalance_config(path=cfg_path)

    def test_rejects_rule_missing_rationale_key(self, tmp_path: Path):
        """Symmetric to the above — `rationale` must also be required."""
        payload = self._base_payload()
        payload["strategy_matrix"]["extreme"]["non_tree"] = [
            {
                "min_minority_count": 0,
                "strategy": "class_weight",
                # Missing: rationale
            },
        ]
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(cfg_path, payload)

        with pytest.raises(ValueError, match="missing required key"):
            _load_imbalance_config(path=cfg_path)

    def test_rejects_branch_without_catch_all(self, tmp_path: Path):
        """A list branch with no `min_minority_count: 0` rule violates
        the catch-all contract — the loader must reject it."""
        payload = self._base_payload()
        # All rules require minority_count >= 5, so a count of 1 would
        # silently fall through to entry[-1] in the resolver.
        payload["strategy_matrix"]["extreme"]["non_tree"] = [
            {
                "min_minority_count": 10,
                "strategy": "combined",
                "rationale": "enough",
            },
            {
                "min_minority_count": 5,
                "strategy": "random_oversample",
                "rationale": "small",
            },
        ]
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(cfg_path, payload)

        with pytest.raises(ValueError, match="catch-all"):
            _load_imbalance_config(path=cfg_path)


class TestLookupStrategyValidation:
    """`_lookup_strategy` runtime guards (problem_type + VALID_STRATEGIES)."""

    def _valid_metrics(self) -> dict:
        return {"severity": "extreme", "minority_count": 15, "total_samples": 100}

    def test_rejects_non_binary_problem_type(self):
        """`_lookup_strategy` must refuse problem types it cannot serve.
        The public node short-circuits regression/continuous earlier; this
        is a defence-in-depth check."""
        config = _load_imbalance_config()
        with pytest.raises(
            ValueError,
            match="binary_classification",
        ):
            _lookup_strategy(
                self._valid_metrics(),
                "XGBoost",
                "regression",
                config=config,
            )

    def test_rejects_unknown_strategy_in_yaml(self, tmp_path: Path):
        """A YAML with an unrecognised strategy string must fail at lookup
        time with a message that enumerates the valid set."""
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
                "tree_models": ["XGBoost"],
                "strategy_matrix": {
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
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
                        "tree": {
                            # Not in VALID_STRATEGIES.
                            "strategy": "not_a_real_strategy",
                            "rationale": "x",
                        },
                        "non_tree": {"strategy": "combined", "rationale": "x"},
                    },
                },
            },
        )
        config = _load_imbalance_config(path=cfg_path)

        with pytest.raises(
            ValueError,
            match=r"VALID_STRATEGIES",
        ):
            _lookup_strategy(
                self._valid_metrics(),
                "XGBoost",
                "binary_classification",
                config=config,
            )


class TestReadYamlCacheReturnsNormalised:
    """`_read_yaml` caches normalised data — the loader must not mutate
    a cached dict in place across calls."""

    def test_cached_payload_is_pre_sorted(self, tmp_path: Path):
        """Two calls to `_load_imbalance_config` (the second of which
        hits the lru_cache) return a strategy_matrix whose list-form
        branches are already sorted by `min_minority_count` descending.
        """
        cfg_path = tmp_path / "imbalance_strategy.yaml"
        _write_yaml(
            cfg_path,
            {
                "severity_bands": {"none": 0.40, "moderate": 0.20, "severe": 0.05},
                "tree_models": ["XGBoost"],
                "strategy_matrix": {
                    "none": {"default": {"strategy": "none", "rationale": "x"}},
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
                        # Authored ascending — loader must sort defensively.
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

        first = _load_imbalance_config(path=cfg_path)
        second = _load_imbalance_config(path=cfg_path)

        first_rules = first["strategy_matrix"]["extreme"]["non_tree"]
        second_rules = second["strategy_matrix"]["extreme"]["non_tree"]

        # Both calls return data sorted by min_minority_count descending.
        first_counts = [int(r["min_minority_count"]) for r in first_rules]
        second_counts = [int(r["min_minority_count"]) for r in second_rules]
        assert first_counts == [10, 5, 0]
        assert second_counts == [10, 5, 0]
        # And the cached object is consistent across calls (no in-place
        # double-mutation that would have re-sorted an already-sorted list).
        assert first_counts == second_counts
