"""The disaster-recovery load path carries the nbrx series (canonical TRx lane).

``generate_datasets`` is the full-reseed generation step of
``scripts/load_synthetic_data.py``; small sizes keep the whole dataset graph fast.
"""

import importlib

from src.ml.synthetic.config import DGPType

load_mod = importlib.import_module("scripts.load_synthetic_data")

_SMALL_SIZES = {
    "hcp": 50,
    "patient": 200,
    "treatment": 200,
    "prediction": 50,
    "trigger": 400,
    # 2 monthly dates x 60 brand/region/metric combos (+1 headroom per date).
    "business_metrics": 2 * 61,
    "feature_values": 50,
}


def test_generate_datasets_appends_nbrx_beside_the_business_metrics_stream():
    datasets = load_mod.generate_datasets(sizes=_SMALL_SIZES, dgp_type=DGPType.CONFOUNDED, seed=11)
    bm = datasets["business_metrics"]
    trx = bm[bm["metric_type"] == "trx"]
    nbrx = bm[bm["metric_type"] == "nbrx"]
    assert len(trx) == 24
    assert len(nbrx) == len(trx)
    assert nbrx["metric_id"].str.startswith("nbrx_").all()
    merged = nbrx.merge(trx, on=["metric_date", "brand", "region"], suffixes=("", "_trx"))
    assert len(merged) == len(nbrx)
    assert (merged["data_split"] == merged["data_split_trx"]).all()
    assert bm["is_synthetic"].all()
