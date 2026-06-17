"""Shard 09 Task 5c (CM-003 / CM-005): minimal is_synthetic-tagged causal_paths rows
so causal_effect_size (CM-003) and mediators_identified (CM-005) are non-NULL.
data_split enum-exact; brand from brand_type; we do NOT touch the 50 stale real rows."""

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.causal_paths_generator import CausalPathsGenerator


def test_causal_paths_nonnull_effect_and_mediators_and_tagged():
    df = CausalPathsGenerator(GeneratorConfig(seed=8, n_records=12)).generate()
    assert len(df) == 12
    assert df["causal_effect_size"].notna().all()  # CM-003 non-NULL
    assert df["mediators_identified"].apply(lambda m: len(m) >= 1).all()  # CM-005
    assert df["is_synthetic"].all()
    # data_split enum-exact (faithful: train/validation/test/holdout/unassigned)
    assert set(df["data_split"]).issubset({"holdout", "test", "train", "unassigned", "validation"})
    assert set(df["brand"]).issubset({"Remibrutinib", "Kisqali", "Fabhalta"})
    # NOT-NULL columns on the faithful DB must be present + populated
    for col in ("path_id", "discovery_date", "causal_chain", "created_at", "confirmation_count"):
        assert df[col].notna().all()


def test_causal_paths_cover_all_three_gold_standard_cohort_outcomes():
    """The KG must carry persistence + discontinuation chains, not just
    initiation. The gold standard validates treatment_arm -> persistent_180d and
    -> discontinued_180d (src/mlops/gold_standard_eval/cohort_spec.py;
    scripts/validate_synthetic_causal.py), but causal_paths only ever emitted
    treatment_initiated, so "Discover chains in KG" returned nothing for the
    persistent_180d default outcome."""
    df = CausalPathsGenerator(GeneratorConfig(seed=3, n_records=30)).generate()
    end_nodes = set(df["end_node"])
    assert {"treatment_initiated", "persistent_180d", "discontinued_180d"} <= end_nodes
    # Every chain starts at the treatment arm and terminates at its end_node,
    # and causal_chain.nodes agrees with start/end (so the FalkorDB sync builds a
    # correct (:Variable treatment_arm)-[:CAUSES]->(:Variable <outcome>) path).
    assert (df["start_node"] == "treatment_arm").all()
    for _, row in df.iterrows():
        nodes = row["causal_chain"]["nodes"]
        assert nodes[0] == "treatment_arm"
        assert nodes[-1] == row["end_node"]
        # No repeated nodes (would be dropped by _clean_causal_chains).
        assert len(nodes) == len(set(nodes))
