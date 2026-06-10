"""HCP adoption CAUSAL cohort: treatment arm + exogenous centrality -> adoption,
leak-safe, in-band, with a recoverable per-HCP CATE artifact."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ml.synthetic.generators.hcp_adoption_artifact import (
    ADOPTER_VALUE,
    generate_hcp_adoption_frame,
    write_per_hcp_cate_artifact,
)


def test_adoption_frame_shape_and_band():
    df = generate_hcp_adoption_frame(seed=3, n_hcps=4000, brand="Kisqali")
    assert {
        "hcp_id",
        "entity_type",
        "centrality_score",
        "influence_network_size",
        "treatment_arm",
        "hcp_segment",
        "adoption_category",
        "cate_estimate",
        "is_synthetic",
    } <= set(df.columns)
    assert (df["entity_type"] == "optum_hcp").all()
    assert df["adoption_category"].isin([ADOPTER_VALUE, "NON_ADOPTER"]).all()
    adopted = (df["adoption_category"] == ADOPTER_VALUE).astype(int)
    assert 0.05 <= adopted.mean() <= 0.60
    assert df["treatment_arm"].isin([0, 1]).all()
    assert df["is_synthetic"].all()


def test_treatment_arm_raises_adoption_recoverable():
    df = generate_hcp_adoption_frame(seed=5, n_hcps=8000, brand="Kisqali")
    adopted = (df["adoption_category"] == ADOPTER_VALUE).astype(int)
    diff = adopted[df["treatment_arm"] == 1].mean() - adopted[df["treatment_arm"] == 0].mean()
    assert diff > 0.05, f"treatment must raise adoption; got diff={diff}"
    assert df.groupby("hcp_segment")["cate_estimate"].mean().nunique() >= 2
    assert (df["cate_estimate"].abs() > 0).any()


def test_centrality_drives_adoption_not_reverse():
    df = generate_hcp_adoption_frame(seed=3, n_hcps=8000, brand="Kisqali")
    adopted = (df["adoption_category"] == ADOPTER_VALUE).astype(int)
    hi = adopted[df["centrality_score"] > df["centrality_score"].quantile(0.75)]
    lo = adopted[df["centrality_score"] < df["centrality_score"].quantile(0.25)]
    assert hi.mean() > lo.mean() + 0.05


def test_no_leaky_columns_present():
    df = generate_hcp_adoption_frame(seed=3, n_hcps=1000, brand="Fabhalta")
    for leak in ("days_to_first", "first_adoption_dt", "adopter_rank", "adoption_cumulative_share"):
        assert leak not in df.columns


def test_hcp_generator_emits_adoption_cohort_columns():
    from src.ml.synthetic.config import Brand
    from src.ml.synthetic.generators.base import GeneratorConfig
    from src.ml.synthetic.generators.hcp_generator import HCPGenerator

    df = HCPGenerator(GeneratorConfig(seed=4, n_records=2000, brand=Brand.KISQALI)).generate()
    for c in ("peer_influence_score", "influence_network_size", "adoption_category"):
        assert c in df.columns
    assert df["adoption_category"].isin([ADOPTER_VALUE, "NON_ADOPTER"]).all()
    rate = (df["adoption_category"] == ADOPTER_VALUE).mean()
    assert 0.05 <= rate <= 0.60, f"hcp_profiles adoption rate {rate} out of band"


def test_hcp_loader_registers_adoption_columns():
    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    for c in ("peer_influence_score", "influence_network_size", "adoption_category"):
        assert c in TABLE_COLUMNS["hcp_profiles"]


def test_per_hcp_cate_artifact_for_shard08(tmp_path: Path):
    df = generate_hcp_adoption_frame(seed=7, n_hcps=2000, brand="Kisqali")
    out = write_per_hcp_cate_artifact(df, brand="Kisqali", out_dir=tmp_path)
    assert out.name == "per_hcp_cate_hcp_adoption_Kisqali.parquet"
    art = pd.read_parquet(out)
    assert list(art.columns) == ["hcp_id", "cate_estimate", "is_synthetic"]
    assert len(art) == len(df)
    assert art["is_synthetic"].all()
    assert art["cate_estimate"].abs().sum() > 0
