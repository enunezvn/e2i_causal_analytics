import json
from pathlib import Path
import pandas as pd
from src.ml.synthetic.config import DGPType
from scripts.load_synthetic_data import SMALL_SIZES, generate_datasets, write_parquet_snapshots


def test_writes_one_parquet_per_table_plus_manifest(tmp_path):
    datasets = generate_datasets(SMALL_SIZES, DGPType.CONFOUNDED, seed=42)
    out = write_parquet_snapshots(datasets, tmp_path)
    for table in datasets:
        p = Path(out) / f"{table}.parquet"
        assert p.exists(), f"missing {p}"
        back = pd.read_parquet(p)
        assert "is_synthetic" in back.columns and bool(back["is_synthetic"].all())
    manifest = json.loads((Path(out) / "manifest.json").read_text())
    assert manifest["is_synthetic"] is True
    assert {m["table"] for m in manifest["tables"]} == set(datasets)
    assert all(m["rows"] > 0 for m in manifest["tables"])
