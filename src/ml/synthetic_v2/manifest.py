"""``FeatureManifest`` dataclass + helpers (shard 01 §C.5).

A frozen per-feature audit record carrying name, distribution, signed
coefficient (logit scale), monotone direction, and clinical justification.
Consumed by W2 day-4 monotone-LightGBM via ``metadata.monotone_vector``.

Filled in by commit 02.
"""
