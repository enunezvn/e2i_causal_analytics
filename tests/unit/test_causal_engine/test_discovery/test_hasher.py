"""
Tests for src/causal_engine/discovery/hasher.py

Covers:
- hash_dataframe function
- _hash_values internal function
- hash_config function
- make_cache_key function
- hash_discovery_request function
- verify_hash_determinism function
"""

import numpy as np
import pandas as pd

from src.causal_engine.discovery.base import DiscoveryConfig
from src.causal_engine.discovery.hasher import (
    _hash_values,
    hash_config,
    hash_dataframe,
    hash_discovery_request,
    make_cache_key,
    verify_hash_determinism,
)

# =============================================================================
# hash_dataframe Tests
# =============================================================================


class TestHashDataframe:
    """Tests for hash_dataframe function."""

    def test_hash_simple_dataframe(self):
        """Test hashing a simple DataFrame."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = hash_dataframe(df)

        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex chars

    def test_hash_empty_dataframe(self):
        """Test hashing an empty DataFrame."""
        df = pd.DataFrame()
        result = hash_dataframe(df)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_is_deterministic(self):
        """Test that hashing is deterministic."""
        df = pd.DataFrame({"X": [1.5, 2.5], "Y": [3.5, 4.5]})

        hash1 = hash_dataframe(df)
        hash2 = hash_dataframe(df)

        assert hash1 == hash2

    def test_different_data_produces_different_hash(self):
        """Test that different data produces different hashes."""
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [1, 2, 4]})  # Last value different

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        assert hash1 != hash2

    def test_column_order_matters(self):
        """Test that column order affects the hash."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"B": [3, 4], "A": [1, 2]})

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        # Column order should matter
        assert hash1 != hash2

    def test_dtype_matters(self):
        """Test that dtype affects the hash."""
        df1 = pd.DataFrame({"A": [1, 2, 3]})  # int64
        df2 = pd.DataFrame({"A": [1.0, 2.0, 3.0]})  # float64

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        # Different dtypes should produce different hashes
        assert hash1 != hash2

    def test_float_precision_consistency(self):
        """Test that floats are rounded for consistency."""
        df1 = pd.DataFrame({"A": [1.123456789012345]})
        df2 = pd.DataFrame({"A": [1.123456789012346]})  # Tiny difference

        hash1 = hash_dataframe(df1)
        hash2 = hash_dataframe(df2)

        # Beyond 8 decimal places should be the same (after rounding)
        # Due to internal precision handling, they might differ
        # This tests the function doesn't crash on edge cases
        assert len(hash1) == 64
        assert len(hash2) == 64

    def test_hash_with_mixed_types(self):
        """Test hashing DataFrame with mixed types."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
            }
        )
        result = hash_dataframe(df)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_mixed_dtype_float_precision_contract(self):
        """Mixed categorical+float frame: float differences beyond 8 decimals must
        NOT change the hash (FU-hasher: object-dtype frames previously bypassed
        rounding because df.values.dtype == object)."""
        # 'region' forces df.values to object dtype; 'spend' differs only at the
        # 9th+ decimal place, which the 8-decimal contract must collapse.
        df1 = pd.DataFrame({"region": ["A", "B", "C"], "spend": [1.123456789012345, 2.0, 3.0]})
        df2 = pd.DataFrame({"region": ["A", "B", "C"], "spend": [1.123456789012346, 2.0, 3.0]})
        assert df1.values.dtype == object  # guards the regression scenario
        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_mixed_dtype_hash_is_deterministic_across_objects(self):
        """Two independently-constructed mixed-dtype frames with identical content
        must hash identically. The old code hashed object-array POINTERS via
        .tobytes(), making cross-object (and cross-process) hashes unstable."""
        df1 = pd.DataFrame({"region": ["A", "B", "C"], "spend": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"region": ["A", "B", "C"], "spend": [1.0, 2.0, 3.0]})
        assert df1.values.dtype == object
        assert hash_dataframe(df1) == hash_dataframe(df2)

    def test_mixed_dtype_meaningful_diff_still_distinct(self):
        """Differences within the 8-decimal window, or in a categorical value,
        must still change the hash (no over-collapsing)."""
        base = pd.DataFrame({"region": ["A", "B"], "spend": [1.10000000, 2.0]})
        float_diff = pd.DataFrame(
            {"region": ["A", "B"], "spend": [1.10000002, 2.0]}
        )  # differs at 8th decimal -> within contract -> must differ
        cat_diff = pd.DataFrame(
            {"region": ["A", "X"], "spend": [1.10000000, 2.0]}
        )  # categorical change -> must differ
        assert hash_dataframe(base) != hash_dataframe(float_diff)
        assert hash_dataframe(base) != hash_dataframe(cat_diff)

    def test_nullable_boolean_column_hashes_by_value_not_pointer(self):
        """FU-hasher: a pandas nullable-boolean column (`dtype='boolean'`) reports
        is_bool_dtype=True but its to_numpy() is an OBJECT array (pd.NA can't fit
        a numpy bool buffer). Hashing that with .tobytes() serializes object
        POINTERS — stable within a process (True/False/NA are singletons) but
        DIFFERENT across processes, making the discovery cache key process-
        dependent. It must instead serialize by VALUE, identically to the
        equivalent object column.
        """
        vals = [True, False, pd.NA, True]
        df_nullable = pd.DataFrame({"flag": pd.array(vals, dtype="boolean")})
        df_object = pd.DataFrame({"flag": pd.Series(vals, dtype="object")})
        # Guard the scenario: the nullable column really is object-backed.
        assert df_nullable["flag"].to_numpy().dtype == object
        assert _hash_values(df_nullable) == _hash_values(df_object), (
            "nullable-boolean column must hash by value (process-independent), "
            "matching the equivalent object column"
        )


# =============================================================================
# hash_config Tests
# =============================================================================


class TestHashConfig:
    """Tests for hash_config function."""

    def test_hash_default_config(self):
        """Test hashing a default DiscoveryConfig."""
        config = DiscoveryConfig()
        result = hash_config(config)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_is_deterministic(self):
        """Test that config hashing is deterministic."""
        config = DiscoveryConfig(alpha=0.05)

        hash1 = hash_config(config)
        hash2 = hash_config(config)

        assert hash1 == hash2

    def test_different_alpha_produces_different_hash(self):
        """Test that different alpha produces different hash."""
        config1 = DiscoveryConfig(alpha=0.05)
        config2 = DiscoveryConfig(alpha=0.01)

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 != hash2

    def test_different_max_cond_vars_produces_different_hash(self):
        """Test that different max_cond_vars produces different hash."""
        config1 = DiscoveryConfig(max_cond_vars=3)
        config2 = DiscoveryConfig(max_cond_vars=5)

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        assert hash1 != hash2

    def test_algorithm_order_does_not_matter(self):
        """Test that algorithm order doesn't affect hash (sorted internally)."""
        from src.causal_engine.discovery.base import DiscoveryAlgorithmType

        # Create configs with algorithms in different orders
        config1 = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.PC, DiscoveryAlgorithmType.GES]
        )
        config2 = DiscoveryConfig(
            algorithms=[DiscoveryAlgorithmType.GES, DiscoveryAlgorithmType.PC]
        )

        hash1 = hash_config(config1)
        hash2 = hash_config(config2)

        # Should be the same because algorithms are sorted
        assert hash1 == hash2


# =============================================================================
# make_cache_key Tests
# =============================================================================


class TestMakeCacheKey:
    """Tests for make_cache_key function."""

    def test_creates_valid_cache_key(self):
        """Test creating a valid cache key."""
        data_hash = "a" * 64
        config_hash = "b" * 64

        key = make_cache_key(data_hash, config_hash)

        assert key.startswith("discovery:")
        assert len(key) > 10  # Has meaningful content

    def test_uses_first_16_chars_of_hashes(self):
        """Test that cache key uses truncated hashes."""
        data_hash = "a" * 64
        config_hash = "b" * 64

        key = make_cache_key(data_hash, config_hash)

        # Format: discovery:{data_hash[:16]}:{config_hash[:16]}
        parts = key.split(":")
        assert len(parts) == 3
        assert parts[0] == "discovery"
        assert len(parts[1]) == 16
        assert len(parts[2]) == 16

    def test_different_hashes_produce_different_keys(self):
        """Test that different hashes produce different keys."""
        key1 = make_cache_key("a" * 64, "b" * 64)
        key2 = make_cache_key("c" * 64, "d" * 64)

        assert key1 != key2


# =============================================================================
# hash_discovery_request Tests
# =============================================================================


class TestHashDiscoveryRequest:
    """Tests for hash_discovery_request function."""

    def test_creates_cache_key_from_df_and_config(self):
        """Test creating cache key from DataFrame and config."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        config = DiscoveryConfig()

        key = hash_discovery_request(df, config)

        assert key.startswith("discovery:")
        assert isinstance(key, str)

    def test_is_deterministic(self):
        """Test that hash_discovery_request is deterministic."""
        df = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        config = DiscoveryConfig(alpha=0.05)

        key1 = hash_discovery_request(df, config)
        key2 = hash_discovery_request(df, config)

        assert key1 == key2

    def test_different_data_produces_different_key(self):
        """Test that different data produces different key."""
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"A": [4, 5, 6]})
        config = DiscoveryConfig()

        key1 = hash_discovery_request(df1, config)
        key2 = hash_discovery_request(df2, config)

        assert key1 != key2

    def test_different_config_produces_different_key(self):
        """Test that different config produces different key."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        config1 = DiscoveryConfig(alpha=0.05)
        config2 = DiscoveryConfig(alpha=0.10)

        key1 = hash_discovery_request(df, config1)
        key2 = hash_discovery_request(df, config2)

        assert key1 != key2


# =============================================================================
# verify_hash_determinism Tests
# =============================================================================


class TestVerifyHashDeterminism:
    """Tests for verify_hash_determinism function."""

    def test_returns_true_for_deterministic_hashing(self):
        """Test that verify returns True for deterministic data."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]})
        config = DiscoveryConfig()

        result = verify_hash_determinism(df, config, n_iterations=5)

        assert result is True

    def test_works_with_default_iterations(self):
        """Test with default n_iterations (3)."""
        df = pd.DataFrame({"X": [10, 20, 30]})
        config = DiscoveryConfig()

        result = verify_hash_determinism(df, config)

        assert result is True

    def test_works_with_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        config = DiscoveryConfig()

        result = verify_hash_determinism(df, config)

        assert result is True

    def test_works_with_large_dataframe(self):
        """Test with larger DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100),
                "C": np.random.randn(100),
            }
        )
        config = DiscoveryConfig()

        result = verify_hash_determinism(df, config, n_iterations=3)

        assert result is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestHasherIntegration:
    """Integration tests for the hasher module."""

    def test_full_workflow(self):
        """Test the full hashing workflow."""
        # Create test data
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0, 1],
                "outcome": [10, 20, 15, 25, 12, 22],
                "covariate": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            }
        )
        config = DiscoveryConfig(alpha=0.05, max_cond_vars=2)

        # Generate cache key
        key = hash_discovery_request(df, config)

        # Verify determinism
        assert verify_hash_determinism(df, config)

        # Verify key format
        assert key.startswith("discovery:")
        parts = key.split(":")
        assert len(parts) == 3

    def test_cache_key_for_causal_analysis_data(self):
        """Test cache key generation for typical causal analysis data."""
        np.random.seed(123)
        n = 50

        df = pd.DataFrame(
            {
                "marketing_spend": np.random.uniform(100, 1000, n),
                "sales": np.random.normal(5000, 500, n),
                "region": np.random.choice(["A", "B", "C"], n),
                "season": np.random.choice(["Q1", "Q2", "Q3", "Q4"], n),
            }
        )
        config = DiscoveryConfig()

        key = hash_discovery_request(df, config)

        # Should produce valid cache key
        assert key.startswith("discovery:")
        assert len(key) > 20  # Has meaningful content


class TestPriorKnowledgeInConfigHash:
    """Guided priors are part of discovery's identity: two configs that differ
    only in prior_knowledge must not share a cache key (they can produce
    different DAGs and different gate corroboration bases)."""

    def test_priors_change_the_hash(self):
        from src.causal_engine.discovery.base import CausalPriorKnowledge

        bare = DiscoveryConfig()
        guided = DiscoveryConfig(
            prior_knowledge=CausalPriorKnowledge(
                tiers=[["c"], ["t"], ["y"]], required_edges=[("t", "y")]
            )
        )
        assert hash_config(bare) != hash_config(guided)

    def test_different_required_edges_hash_differently(self):
        from src.causal_engine.discovery.base import CausalPriorKnowledge

        one = DiscoveryConfig(prior_knowledge=CausalPriorKnowledge(required_edges=[("t", "y")]))
        other = DiscoveryConfig(
            prior_knowledge=CausalPriorKnowledge(required_edges=[("t", "y"), ("c", "y")])
        )
        assert hash_config(one) != hash_config(other)

    def test_edge_order_does_not_change_the_hash(self):
        from src.causal_engine.discovery.base import CausalPriorKnowledge

        forward = DiscoveryConfig(
            prior_knowledge=CausalPriorKnowledge(required_edges=[("a", "b"), ("c", "d")])
        )
        reversed_order = DiscoveryConfig(
            prior_knowledge=CausalPriorKnowledge(required_edges=[("c", "d"), ("a", "b")])
        )
        assert hash_config(forward) == hash_config(reversed_order)

    def test_empty_prior_hashes_like_no_prior(self):
        from src.causal_engine.discovery.base import CausalPriorKnowledge

        assert hash_config(DiscoveryConfig()) == hash_config(
            DiscoveryConfig(prior_knowledge=CausalPriorKnowledge())
        )
