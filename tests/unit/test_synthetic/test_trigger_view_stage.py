"""#1387 DGP trigger view-stage realism.

Pre-#1387 the generator never modeled the "HCP views the trigger" step:
``view_timestamp`` was emitted on 0 rows and 13,147 live accepted rows still
carried ``delivery_status='delivered'``, so the WS2-TR-009 funnel read
viewed < accepted (a funnel that goes up). Contract under test:

* MONOTONE STATUS: any explicit disposition (accepted / rejected / overridden)
  implies the trigger was seen -> ``delivery_status == 'viewed'``. A share of
  no-disposition (pending / expired acceptance) delivered triggers is also
  viewed; a nonzero delivered-never-viewed remainder survives.
* DENOMINATOR PRESERVATION: the acceptance / precision / override KPIs
  denominate on ``delivery_status IN ('delivered','viewed')`` (migrations
  090/092). The view stage only MOVES rows inside that set — it never promotes
  pending/failed rows into it and never demotes union rows out of it.
* VIEW TIMESTAMP: non-null exactly on viewed rows, bounded below by
  ``trigger_timestamp`` (the delivery moment on this substrate — delivery /
  acceptance timestamps are not modeled) and above by the anchor reference
  (the repo's ``event_timestamp <= NOW()`` discipline) when anchoring is on.
* STREAM PRESERVATION (the keyed-backfill enabler): the view stage draws from
  a dedicated seeded RNG, NEVER from the generator's main ``self._rng``
  stream, so every pre-#1387 column reproduces byte-identically under the
  same seed/anchor — a triggers-only upsert backfill cannot decohere the
  frozen substrate (acceptance draws stay welded to the injected ``trxc``
  conversion prescriptions already in treatment_events).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators import GeneratorConfig
from src.ml.synthetic.generators.trigger_generator import TriggerGenerator

_N_PATIENTS = 600
_ACTIVE = ("accepted", "rejected", "overridden")
_UNION = ("delivered", "viewed")
_ANCHOR_REF = date(2026, 7, 21)


def _patient_df(seed: int = 21, n: int = _N_PATIENTS) -> pd.DataFrame:
    """Minimal linked-path patient frame (mirrors test_trigger_outcome_decoupling)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "patient_id": [f"pt_{i:06d}" for i in range(n)],
            "hcp_id": [f"hcp_{i % 60:04d}" for i in range(n)],
            "journey_start_date": ["2025-09-01"] * n,
            "engagement_score": rng.normal(5.0, 2.0, n).clip(0, 10),
            "treatment_initiated": rng.integers(0, 2, n),
            "disease_severity": rng.normal(5.0, 2.0, n).clip(0, 10),
            "age_at_diagnosis": rng.integers(18, 85, n),
            "brand": [Brand.REMIBRUTINIB.value] * n,
            "trigger_accepted": rng.integers(0, 2, n),
        }
    )


def _generate(seed: int = 21, n_triggers: int = 1800, **cfg_kwargs):
    patients = _patient_df(seed)
    gen = TriggerGenerator(
        GeneratorConfig(seed=seed, n_records=n_triggers, brand=Brand.REMIBRUTINIB, **cfg_kwargs),
        patient_df=patients,
    )
    return patients, gen.generate(), gen


# ---------------------------------------------------------------------------
# Monotone status
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_accepted_implies_viewed_linked():
    _, triggers, _ = _generate()
    accepted = triggers[triggers["acceptance_status"] == "accepted"]
    offenders = accepted[accepted["delivery_status"] != "viewed"]
    assert offenders.empty, (
        f"{len(offenders)} accepted triggers not marked viewed — the WS2-TR-009 "
        "funnel reads viewed < accepted again"
    )


@pytest.mark.unit
def test_active_dispositions_imply_viewed_linked():
    # A rep cannot reject or override a recommendation they never opened —
    # the same incoherence class #1387 fixes for accepted.
    _, triggers, _ = _generate()
    active = triggers[triggers["acceptance_status"].isin(_ACTIVE)]
    offenders = active[active["delivery_status"] != "viewed"]
    assert offenders.empty, f"{len(offenders)} rejected/overridden triggers not marked viewed"


@pytest.mark.unit
def test_accepted_implies_viewed_standalone():
    gen = TriggerGenerator(GeneratorConfig(seed=7, n_records=2000))
    df = gen.generate()
    active = df[df["acceptance_status"].isin(_ACTIVE)]
    offenders = active[active["delivery_status"] != "viewed"]
    assert offenders.empty, (
        f"standalone path: {len(offenders)} disposition-bearing triggers not viewed"
    )


@pytest.mark.unit
def test_nonzero_viewed_without_disposition_and_delivered_remainder():
    _, triggers, _ = _generate()
    no_disposition = triggers[triggers["acceptance_status"].isin(["pending", "expired"])]
    in_union = no_disposition[no_disposition["delivery_status"].isin(_UNION)]
    n_viewed = (in_union["delivery_status"] == "viewed").sum()
    n_delivered = (in_union["delivery_status"] == "delivered").sum()
    assert n_viewed > 0, "no viewed-but-no-disposition triggers — funnel stage is degenerate"
    assert n_delivered > 0, (
        "no delivered-never-viewed remainder — the delivered stage collapsed into viewed"
    )


# ---------------------------------------------------------------------------
# KPI denominator preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pending_failed_delivery_rows_untouched():
    _, triggers, _ = _generate()
    outside = triggers[~triggers["delivery_status"].isin(_UNION)]
    assert set(outside["delivery_status"].unique()) <= {"pending", "failed"}
    # delivery still gates acceptance: nothing outside the union carries a disposition
    assert (outside["acceptance_status"] == "pending").all(), (
        "a pending/failed-delivery trigger carries a disposition — the view stage "
        "moved rows INTO the KPI denominator set"
    )


@pytest.mark.unit
def test_disposition_bearing_rows_stay_in_union():
    _, triggers, _ = _generate()
    disposed = triggers[triggers["acceptance_status"] != "pending"]
    assert disposed["delivery_status"].isin(_UNION).all(), (
        "a disposition-bearing trigger left the ('delivered','viewed') set — "
        "acceptance/precision/override KPI denominators lose rows"
    )


# ---------------------------------------------------------------------------
# view_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_view_timestamp_iff_viewed():
    _, triggers, _ = _generate()
    assert "view_timestamp" in triggers.columns
    viewed = triggers["delivery_status"] == "viewed"
    assert triggers.loc[viewed, "view_timestamp"].notna().all(), (
        "a viewed trigger has no view_timestamp"
    )
    assert triggers.loc[~viewed, "view_timestamp"].isna().all(), (
        "a non-viewed trigger carries a view_timestamp"
    )


@pytest.mark.unit
def test_view_timestamp_format_and_lower_bound():
    _, triggers, _ = _generate()
    viewed = triggers[triggers["delivery_status"] == "viewed"]
    assert viewed["view_timestamp"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()
    view_ts = pd.to_datetime(viewed["view_timestamp"])
    trig_ts = pd.to_datetime(viewed["trigger_timestamp"])
    assert (view_ts >= trig_ts).all(), "view precedes the trigger/delivery moment"
    assert (view_ts <= trig_ts + pd.Timedelta(hours=72)).all(), (
        "view delay exceeds the modeled 72h window"
    )


@pytest.mark.unit
def test_view_timestamp_respects_anchor_now_cap():
    # event_timestamp <= NOW() discipline: under anchoring no derived timestamp
    # may pass the rolling-window reference (same cap as trigger_timestamp).
    _, triggers, _ = _generate(anchor_to_now=True, anchor_reference=_ANCHOR_REF)
    viewed = triggers[triggers["delivery_status"] == "viewed"]
    assert not viewed.empty
    view_ts = pd.to_datetime(viewed["view_timestamp"])
    assert (view_ts <= pd.Timestamp(_ANCHOR_REF)).all(), (
        "view_timestamp passes the anchor reference — future event timestamp"
    )
    trig_ts = pd.to_datetime(viewed["trigger_timestamp"])
    assert (view_ts >= trig_ts).all()


# ---------------------------------------------------------------------------
# Determinism + stream preservation (keyed-backfill enabler)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_same_seed_reproduces_identical_frame():
    _, run_a, _ = _generate(anchor_to_now=True, anchor_reference=_ANCHOR_REF)
    _, run_b, _ = _generate(anchor_to_now=True, anchor_reference=_ANCHOR_REF)
    pd.testing.assert_frame_equal(run_a, run_b)


@pytest.mark.unit
def test_view_stage_does_not_consume_main_rng_stream(monkeypatch):
    """Every pre-#1387 column must reproduce byte-identically with the view
    stage disabled: the stage may only touch delivery_status/view_timestamp and
    must draw from its own keyed RNG, never self._rng. This is what makes the
    triggers-only upsert backfill safe against the frozen 2026-07-21 substrate."""
    _, with_stage, gen_a = _generate()

    monkeypatch.setattr(TriggerGenerator, "_apply_view_stage", lambda self, df: df)
    _, without_stage, gen_b = _generate()

    changed = {"delivery_status", "view_timestamp"}
    assert set(with_stage.columns) - set(without_stage.columns) <= changed
    shared = [c for c in with_stage.columns if c not in changed]
    pd.testing.assert_frame_equal(with_stage[shared], without_stage[shared])
    # the injected conversion prescriptions (the welded treatment_events rows)
    # must be identical too
    pd.testing.assert_frame_equal(gen_a.injected_prescriptions, gen_b.injected_prescriptions)
    # and no demotion: every originally-viewed row is still viewed
    originally_viewed = without_stage["delivery_status"] == "viewed"
    assert (with_stage.loc[originally_viewed, "delivery_status"] == "viewed").all()


@pytest.mark.unit
def test_view_stage_preserves_row_count_ids_and_synthetic_stamp():
    _, triggers, gen = _generate()
    assert len(triggers) == triggers["trigger_id"].nunique()
    # injected prescriptions keep their self-stamped provenance (#1389 trap:
    # is_synthetic must never default false on the synthetic path)
    if len(gen.injected_prescriptions) > 0:
        assert gen.injected_prescriptions["is_synthetic"].all()
