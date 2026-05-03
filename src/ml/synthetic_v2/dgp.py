"""Shared DGP machinery (shard 02).

Will host:
- ``solve_intercept`` binary-search prevalence calibrator
- ``apply_block_correlation`` consuming ``builder.correlation_blocks``
- ``sample_features`` helpers shared across scenarios
- audit fingerprint (SHA-256) computation

Filled in by commit 03.
"""
