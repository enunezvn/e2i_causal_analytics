"""Shard 05 Task 4 — the load script must pass treatment_df into TriggerGenerator
AND merge the injected conversion prescriptions back into treatment_events, else the
designed lift is generated but never loaded (gate reads the degenerate ~0.002).

Uses the real builder entrypoint generate_datasets(sizes, dgp_type, ...) (hermetic,
no DB)."""
import importlib

from src.ml.synthetic.config import DGPType

load_mod = importlib.import_module("scripts.load_synthetic_data")


def test_injected_prescriptions_merged_into_treatment_events():
    datasets = load_mod.generate_datasets(
        sizes={
            "hcp": 50, "patient": 200, "treatment": 200,
            "prediction": 50, "trigger": 400, "business_metrics": 30,
            "feature_values": 50,
        },
        dgp_type=DGPType.CONFOUNDED,
        seed=11,
    )
    te = datasets["treatment_events"]
    # injected conversion prescriptions carry the 'trxc' id segment.
    injected = te[te["treatment_event_id"].astype(str).str.contains("trxc")]
    assert len(injected) > 0, "injected conversion prescriptions not merged"
    # Provenance: appended AFTER the central stamp -> must be is_synthetic=True.
    assert bool(injected["is_synthetic"].all())
    assert (injected["event_type"] == "prescription").all()
