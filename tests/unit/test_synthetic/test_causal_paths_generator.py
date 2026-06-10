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
