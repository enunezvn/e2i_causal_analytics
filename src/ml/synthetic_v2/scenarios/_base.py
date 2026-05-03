"""``ScenarioBuilder`` ABC (shard 01 §B.4).

Per-scenario subclasses implement disease-specific feature surface
(name, target_prevalence, target_auc_band, n_features,
correlation_strength, slope_multiplier, feature_manifest, sample_features,
default_n_total, correlation_blocks). Concrete ``compute_logits`` default
provided by the ABC.

Filled in by commit 04.
"""
