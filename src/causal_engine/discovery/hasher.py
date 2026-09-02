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
    # Hash column-by-column so heterogeneous (object/mixed-dtype) frames honor
    # the 8-decimal float precision contract. df.values on a mixed frame is
    # object-dtype: np.round is skipped AND .tobytes() serializes object
    # POINTERS (non-deterministic across processes). Per-column serialization
    # avoids both failure modes.
    hasher = hashlib.sha256()
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        if pd.api.types.is_float_dtype(dtype):
            # Round floats to fixed precision (8 decimals) then hash raw bytes.
            rounded = np.round(series.to_numpy(), decimals=8)
            hasher.update(rounded.tobytes())
        elif (
            pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_integer_dtype(dtype)
        ) and series.to_numpy().dtype != object:
            hasher.update(series.to_numpy().tobytes())
        else:
            # object / categorical / string / datetime AND pandas *nullable*
            # boolean (whose to_numpy() is object because pd.NA can't fit a numpy
            # bool buffer): serialize by VALUE, not by pointer. .tobytes() on an
            # object array would hash object POINTERS — non-deterministic across
            # processes — making the discovery cache key process-dependent.
            hasher.update(b"\x1f".join(repr(v).encode("utf-8") for v in series.tolist()))
        # Column delimiter so concatenations across columns can't collide.
        hasher.update(b"::")
    return hasher.hexdigest()[:16]


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
        "bootstrap_resamples": config.bootstrap_resamples,
        "latent_diagnostic": config.latent_diagnostic,
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
