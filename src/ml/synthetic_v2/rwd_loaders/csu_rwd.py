"""CSU/Remibrutinib RWD loader for Scenario C concurrent-validation (shard 07 §C).

Wraps ``scripts/convert_csu_rwd.py`` + ``data/synthetic/e2i_ml_v3_*.json``;
emits a contract surface compatible with ``test_scenario_c_rwd_concurrent.py``
(KS tests + AUC delta vs synthetic; tie-breaking per shard 05 §G.5
nearest-visit rule with equidistant → earlier visit).

Filled in by commit 13.
"""
