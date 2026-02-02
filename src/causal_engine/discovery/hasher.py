"""
E2I Causal Analytics - Discovery Result Hasher
===============================================

Utilities for hashing DataFrame content and DiscoveryConfig for cache keys.

Provides deterministic hashing for:
- DataFrame content (values, dtypes, column order)
- DiscoveryConfig (algorithm list, parameters)
- Combined cache keys

Author: E2I Causal Analytics Team
"""

import hashlib
import json

import numpy as np
import pandas as pd

from .base import DiscoveryConfig


def hash_dataframe(df: pd.DataFrame) -> str:
    """Generate SHA-256 hash of DataFrame content.

    Creates a deterministic hash based on:
    - Column names and order
    - Data types
    - Values (with fixed precision for floats)

    Args:
        df: Input DataFrame to hash

    Returns:
        64-character hexadecimal hash string

    Example:
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> h = hash_dataframe(df)
        >>> len(h)
        64
    """
    if df.empty:
        return hashlib.sha256(b"empty_dataframe").hexdigest()

    # Create a representation that captures structure and content
    components = []

    # 1. Column names (order matters)
    components.append(f"columns:{','.join(df.columns.tolist())}")

    # 2. Data types
    dtypes_str = ",".join(f"{col}:{dtype}" for col, dtype in df.dtypes.items())
    components.append(f"dtypes:{dtypes_str}")

    # 3. Shape
    components.append(f"shape:{df.shape[0]}x{df.shape[1]}")

    # 4. Values - use numpy tobytes with fixed precision for floats
    # Round floats to 8 decimal places for consistency
    values_hash = _hash_values(df)
    components.append(f"values:{values_hash}")

    # Combine components and hash
    combined = "|".join(components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _hash_values(df: pd.DataFrame) -> str:
    """Hash DataFrame values with consistent handling of floats.

    Args:
        df: DataFrame to hash values from

    Returns:
        Hash of the values
    """
    # Convert to numpy and handle float precision
    arr = df.values.copy()

    # Round floats to fixed precision
    if arr.dtype in [np.float64, np.float32]:
        arr = np.round(arr, decimals=8)

    # Convert to bytes and hash
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def hash_config(config: DiscoveryConfig) -> str:
    """Generate SHA-256 hash of DiscoveryConfig.

    Creates a deterministic hash based on:
    - Algorithm list (sorted for consistency)
    - Numeric parameters
    - Boolean flags

    Args:
        config: DiscoveryConfig to hash

    Returns:
        64-character hexadecimal hash string

    Example:
        >>> config = DiscoveryConfig(alpha=0.05)
        >>> h = hash_config(config)
        >>> len(h)
        64
    """
    # Create dictionary of relevant config values
    config_dict = {
        # Sort algorithm names for consistency
        "algorithms": sorted([alg.value for alg in config.algorithms]),
        "alpha": round(config.alpha, 8),
        "max_cond_vars": config.max_cond_vars,
        "ensemble_threshold": round(config.ensemble_threshold, 8),
        "max_iter": config.max_iter,
        "random_state": config.random_state,
        "score_func": config.score_func,
        "assume_linear": config.assume_linear,
        "assume_gaussian": config.assume_gaussian,
    }

    # Convert to deterministic JSON string
    json_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def make_cache_key(data_hash: str, config_hash: str) -> str:
    """Create cache key from data and config hashes.

    Args:
        data_hash: Hash of the DataFrame
        config_hash: Hash of the DiscoveryConfig

    Returns:
        Cache key in format "discovery:{data_hash[:16]}:{config_hash[:16]}"

    Example:
        >>> key = make_cache_key("abc123...", "def456...")
        >>> key.startswith("discovery:")
        True
    """
    # Use first 16 chars of each hash for shorter keys
    return f"discovery:{data_hash[:16]}:{config_hash[:16]}"


def hash_discovery_request(
    df: pd.DataFrame,
    config: DiscoveryConfig,
) -> str:
    """Generate cache key for a discovery request.

    Convenience function that combines hash_dataframe, hash_config,
    and make_cache_key.

    Args:
        df: Input DataFrame
        config: Discovery configuration

    Returns:
        Cache key string

    Example:
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> config = DiscoveryConfig()
        >>> key = hash_discovery_request(df, config)
        >>> key.startswith("discovery:")
        True
    """
    data_hash = hash_dataframe(df)
    config_hash = hash_config(config)
    return make_cache_key(data_hash, config_hash)


def verify_hash_determinism(
    df: pd.DataFrame,
    config: DiscoveryConfig,
    n_iterations: int = 3,
) -> bool:
    """Verify that hashing is deterministic.

    Useful for testing and debugging to ensure the same
    input always produces the same hash.

    Args:
        df: DataFrame to test
        config: Config to test
        n_iterations: Number of times to hash and compare

    Returns:
        True if all iterations produce identical hashes
    """
    hashes = set()
    for _ in range(n_iterations):
        key = hash_discovery_request(df, config)
        hashes.add(key)

    return len(hashes) == 1
