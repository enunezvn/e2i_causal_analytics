"""Commercial-KPI grain of the causal_paths substrate (2026-07-07).

Why this exists: the registry modeled patient/HCP/trigger outcomes only, so
"what drives TRx?" was a genuine substrate-coverage gap — the chat tool
honestly returned 0 chains for every commercial volume KPI. This grain seeds
curated synthetic driver chains for the most impactful commercial KPIs
(TRx / NRx / NBRx / TRx Share / ROI / intent-to-prescribe, per WS3-BI-005..008,
WS3-BI-010, BR-002).

Two contracts are load-bearing:

1. TOKEN MATCH — the chat read path (CausalPathRepository.search_paths_for_
   outcome) matches 6-char token prefixes as ILIKE substrings against
   start_node/end_node. A seeded chain only surfaces if its end_node carries
   the KPI token, so the node names are pinned against outcome_match_tokens
   for every phrasing the model plausibly sends.

2. DETERMINISM — path_ids AND effect values are content-addressed per
   (brand, start, end), independent of seed and n_records, so the targeted
   apply script is idempotent (upsert on path_id; re-runs are no-ops). This is
   the lesson of the PR #1105/#1106 reseed non-idempotency incident.
"""

import pandas as pd

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.causal_paths_generator import (
    N_COMMERCIAL_ROWS,
    CausalPathsGenerator,
    commercial_rows_for_upsert,
)
from src.repositories.causal_path import outcome_match_tokens

_BRANDS = {"Remibrutinib", "Kisqali", "Fabhalta"}

_COMMERCIAL_OUTCOMES = {
    "trx_volume",
    "nrx_volume",
    "nbrx_volume",
    "trx_market_share",
    "roi",
    "intent_to_prescribe",
}


def _commercial(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["grain"] == "commercial"]


def test_commercial_grain_covers_all_outcomes_per_brand():
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate()
    com = _commercial(df)
    assert len(com) == N_COMMERCIAL_ROWS
    assert set(com["end_node"]) == _COMMERCIAL_OUTCOMES
    # Every outcome is modeled for every brand (brand-scoped chat queries).
    cells = set(zip(com["brand"], com["end_node"], strict=True))
    for brand in _BRANDS:
        for outcome in _COMMERCIAL_OUTCOMES:
            assert (brand, outcome) in cells


def test_commercial_end_nodes_match_chat_kpi_tokens():
    """THE load-bearing contract: every phrasing the chat model plausibly sends
    for a covered KPI must token-match at least one commercial end_node, using
    the REAL matching function (6-char prefixes, ILIKE substring semantics)."""
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate()
    end_nodes = list(_commercial(df)["end_node"])
    for query in (
        "TRx",
        "NRx",
        "NBRx",
        "TRx Share",
        "market share",
        "ROI",
        "intent to prescribe",
    ):
        tokens = outcome_match_tokens(query)
        assert any(token in node for token in tokens for node in end_nodes), (
            f"chat query {query!r} (tokens {tokens}) matches no commercial end_node"
        )


def test_commercial_rows_respect_db_constraints():
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate()
    com = _commercial(df)
    for _, row in com.iterrows():
        # varchar(20) PK, namespaced so the apply script can upsert safely.
        assert row["path_id"].startswith("scp_c")
        assert len(row["path_id"]) <= 20
        # causal_paths_effect_decomp_chk (migration 049).
        assert abs(row["direct_effect"] + row["indirect_effect"] - row["causal_effect_size"]) < 1e-3
        # Chat tool default min_confidence=0.7 must pass.
        assert 0.70 <= row["confidence_level"] <= 0.95
        # numeric(5,3) effect precision.
        assert abs(row["causal_effect_size"]) < 1.0
        # Clean mediated 2-hop chain (generator invariant for non-trigger grains).
        assert len(row["mediators_identified"]) == 1
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == row["start_node"]
        assert nodes[-1] == row["end_node"]
        assert len(nodes) == len(set(nodes))
        assert row["is_synthetic"]
        assert row["data_split"] == "unassigned"
        assert row["validation_status"] == "validated"


def test_commercial_rows_are_content_addressed_and_deterministic():
    """Same rows regardless of seed AND n_records: path_id and every numeric
    value derive from (brand, start, end), so the targeted apply script can
    upsert idempotently and a later full reseed cannot silently rewrite them."""
    a = _commercial(CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate())
    b = _commercial(CausalPathsGenerator(GeneratorConfig(seed=99, n_records=27)).generate())
    cols = [
        "path_id",
        "start_node",
        "end_node",
        "brand",
        "causal_effect_size",
        "direct_effect",
        "indirect_effect",
        "confidence_level",
    ]
    pd.testing.assert_frame_equal(
        a[cols].sort_values("path_id").reset_index(drop=True),
        b[cols].sort_values("path_id").reset_index(drop=True),
    )
    assert a["path_id"].is_unique


def test_competitor_pressure_effects_are_negative():
    """competitor_activity chains model pressure, not uplift: negative effect,
    decomposition still consistent (both components share the sign)."""
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate()
    com = _commercial(df)
    comp = com[com["start_node"] == "competitor_activity"]
    assert len(comp) > 0
    for _, row in comp.iterrows():
        assert row["causal_effect_size"] < 0
        assert row["direct_effect"] <= 0
        assert row["indirect_effect"] <= 0


def test_commercial_rows_for_upsert_are_db_shaped():
    """The apply script's records: projected to the loader's causal_paths
    column list (no generator-only 'grain' column — the DB has no such column,
    a stray key would 400 the upsert), unique namespaced path_ids."""
    import json

    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    records = commercial_rows_for_upsert()
    assert len(records) == N_COMMERCIAL_ROWS
    allowed = set(TABLE_COLUMNS["causal_paths"])
    for rec in records:
        assert set(rec) <= allowed
        assert "grain" not in rec
        assert rec["path_id"].startswith("scp_c")
        assert rec["is_synthetic"] is True
    assert len({r["path_id"] for r in records}) == N_COMMERCIAL_ROWS
    # postgrest serializes with stdlib json: numpy scalars (np.int64/np.bool_)
    # left by a DataFrame round-trip would raise here and 500 the upsert
    # (the PR #1098 numpy-serialization lesson).
    json.dumps(records)


def test_commercial_grain_links_patient_journey_to_volume():
    """Coherence with the existing grains: persistence feeds TRx, initiation
    feeds NRx — so the registry tells one story across grains."""
    df = CausalPathsGenerator(GeneratorConfig(seed=7, n_records=9)).generate()
    com = _commercial(df)
    edges = set(zip(com["start_node"], com["end_node"], strict=True))
    assert ("persistent_180d", "trx_volume") in edges
    assert ("treatment_initiated", "nrx_volume") in edges
    # intent_to_prescribe is both an outcome (BR-002) and the mediator-story
    # start of the NRx chain.
    assert ("intent_to_prescribe", "nrx_volume") in edges
    assert ("rep_detailing_frequency", "intent_to_prescribe") in edges
