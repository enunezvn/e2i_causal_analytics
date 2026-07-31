"""COMM-ARMS Phase 4 trigger-side wiring: acceptance_status consistent with the
patient-level trigger_accepted arm, and outcome_value DECOUPLED from acceptance.

Pre-Phase-4 weld (trigger_generator.py): outcome_value was assigned only when
outcome_tracked AND acceptance_status == 'accepted', so two cells of the
acceptance x outcome table were structurally empty — precision was unfalsifiable
(level == P(tracked)) and acceptance uplift unmeasurable. Phase 4 semantics:

* outcome := real downstream prescription within the 30d window (injected
  conversion prescriptions OR baseline treatment_events), acceptance-independent;
* outcome_value is TRI-STATE: NULL iff NOT outcome_tracked; 0.0 when tracked and
  no Rx landed in-window; > 0 when tracked and one did (conversion_flag, the DB
  stored-generated outcome_value > 0, then means "tracked and converted");
* acceptance_status is driven by patient_df.trigger_accepted on the linked path:
  arm=1 patients carry >=1 accepted trigger, arm=0 patients carry none, and the
  arm-conditional distributions are chosen so the TRIGGER-level marginals
  (acceptance ~0.50, override ~0.14 among delivered/viewed) are preserved.
"""

import hashlib
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.config import Brand
from src.ml.synthetic.generators import GeneratorConfig
from src.ml.synthetic.generators.trigger_generator import TriggerGenerator

_N_PATIENTS = 600


def _patient_df(seed: int = 21, n: int = _N_PATIENTS) -> pd.DataFrame:
    """Minimal linked-path patient frame carrying the Phase 4 arm column."""
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


def _generate(seed: int = 21, n_triggers: int = 1800):
    patients = _patient_df(seed)
    gen = TriggerGenerator(
        GeneratorConfig(seed=seed, n_records=n_triggers, brand=Brand.REMIBRUTINIB),
        patient_df=patients,
    )
    return patients, gen.generate(), gen


@pytest.mark.unit
def test_arm_1_patients_have_at_least_one_accepted_trigger():
    patients, triggers, _ = _generate()
    accepted_by_patient = (
        triggers[triggers["acceptance_status"] == "accepted"].groupby("patient_id").size()
    )
    arm1 = patients.loc[patients["trigger_accepted"] == 1, "patient_id"]
    missing = set(arm1) - set(accepted_by_patient.index)
    assert not missing, (
        f"{len(missing)} arm=1 patients have NO accepted trigger — the patient-level "
        "arm and the triggers table disagree; the DIRECT estimate on the join would be "
        "diluted. First few: " + ", ".join(sorted(missing)[:5])
    )


@pytest.mark.unit
def test_arm_0_patients_have_no_accepted_trigger():
    patients, triggers, _ = _generate()
    arm0 = set(patients.loc[patients["trigger_accepted"] == 0, "patient_id"])
    offenders = triggers[
        (triggers["acceptance_status"] == "accepted") & (triggers["patient_id"].isin(arm0))
    ]
    assert offenders.empty, (
        f"{len(offenders)} accepted triggers belong to arm=0 patients — acceptance "
        "must be gated by the patient-level arm"
    )


@pytest.mark.unit
def test_trigger_level_acceptance_and_override_marginals_are_preserved():
    """The arm-conditional acceptance distributions must reproduce the pre-P4
    trigger-level marginals so WS2-TR-004 (~0.51) and WS2-TR-006 (~0.14) do not
    silently move. Bands are +/-0.05 around the pre-P4 planted masses."""
    _, triggers, _ = _generate()
    dv = triggers[triggers["delivery_status"].isin(["delivered", "viewed"])]
    acc_rate = (dv["acceptance_status"] == "accepted").mean()
    ovr_rate = (dv["acceptance_status"] == "overridden").mean()
    assert 0.45 <= acc_rate <= 0.55, f"acceptance {acc_rate:.3f} left the 0.50 band"
    assert 0.09 <= ovr_rate <= 0.19, f"override {ovr_rate:.3f} left the 0.14 band"


@pytest.mark.unit
def test_all_four_acceptance_x_outcome_cells_are_populated():
    """The decoupling deliverable: rejected-but-converted and accepted-but-failed
    exist. Pre-P4 both cells were structurally empty."""
    _, triggers, _ = _generate()
    tracked = triggers[triggers["outcome_tracked"].astype(bool)]
    accepted = tracked["acceptance_status"] == "accepted"
    converted = tracked["outcome_value"].fillna(0) > 0
    cells = {
        "accepted_converted": int((accepted & converted).sum()),
        "accepted_failed": int((accepted & ~converted).sum()),
        "other_converted": int((~accepted & converted).sum()),
        "other_failed": int((~accepted & ~converted).sum()),
    }
    empty = [k for k, v in cells.items() if v == 0]
    assert not empty, f"structurally empty acceptance x outcome cells remain: {empty} ({cells})"


@pytest.mark.unit
def test_outcome_value_is_tristate_on_tracking():
    """NULL iff not tracked; tracked rows carry a float (0.0 = tracked, no Rx
    in-window; >0 = tracked, converted)."""
    _, triggers, _ = _generate()
    tracked = triggers["outcome_tracked"].astype(bool)
    assert triggers.loc[~tracked, "outcome_value"].isna().all(), (
        "untracked triggers must carry NULL outcome_value"
    )
    assert triggers.loc[tracked, "outcome_value"].notna().all(), (
        "tracked triggers must carry a concrete outcome_value (0.0 or the realized "
        "magnitude) — NULL-when-tracked recreates the pre-P4 ambiguity"
    )
    vals = triggers.loc[tracked, "outcome_value"]
    assert (vals >= 0).all()
    assert (vals > 0).any() and (vals == 0).any()


@pytest.mark.unit
def test_conversion_matches_prescription_in_window():
    """outcome_value > 0 must be backed by a REAL prescription row: either an
    injected conversion prescription or a baseline treatment_events row inside
    [trigger_ts, trigger_ts + 30d]. No free-floating positive outcomes."""
    _, triggers, gen = _generate()
    injected = gen.injected_prescriptions
    assert not injected.empty, "injection channel must survive the decoupling"
    inj_by_patient: dict[str, list] = {}
    for pid, d in zip(injected["patient_id"], pd.to_datetime(injected["event_date"]), strict=False):
        inj_by_patient.setdefault(pid, []).append(d)
    tracked = triggers[triggers["outcome_tracked"].astype(bool)]
    pos = tracked[tracked["outcome_value"].fillna(0) > 0]
    ts = pd.to_datetime(pos["trigger_timestamp"])
    unbacked = 0
    for pid, t in zip(pos["patient_id"], ts, strict=False):
        dates = inj_by_patient.get(pid, [])
        if not any(t <= d <= t + pd.Timedelta(days=30) for d in dates):
            unbacked += 1
    # Without a treatment_df fixture the ONLY legitimate backing is the injected
    # channel, so every positive outcome must be injection-backed.
    assert unbacked == 0, f"{unbacked} positive outcomes with no Rx inside the 30d window"


@pytest.mark.unit
def test_false_positive_flag_only_on_tracked_unproductive_triggers():
    _, triggers, _ = _generate()
    fp = triggers[triggers["false_positive_flag"].astype(bool)]
    assert fp["outcome_tracked"].astype(bool).all()
    assert (fp["outcome_value"].fillna(0) <= 0).all()


# Phase-4-era pinned digest of TriggerGenerator's NON-acceptance/outcome fields
# (design-review M2): Phase 4 restructured _generate_trigger_record, which
# legitimately shifted the per-record RNG stream (the destructive reseed
# rewrites every trigger row, and no consumer pins trigger bytes — unlike the
# patient persistence digests, which held). THIS pin locks the post-Phase-4
# stream so any FUTURE refactor that silently adds/removes/reorders draws
# (shifting delivery_channel, confidence_score, control_group_flag, ... for
# every subsequent trigger) fails here instead of shipping unnoticed.
_STREAM_FIELDS = [
    "trigger_id",
    "patient_id",
    "trigger_type",
    "trigger_timestamp",
    "priority",
    "confidence_score",
    "delivery_channel",
    "delivery_status",
    "control_group_flag",
    "action_taken",
    "lead_time_days",
    "outcome_tracked",
]
# Re-pinned for #1387 (view-stage realism): delivery_status now advances to
# 'viewed' on disposition-bearing rows + a keyed share of no-disposition rows.
# The per-record RNG stream itself is UNCHANGED (the view stage draws from a
# dedicated keyed rng — pinned by test_trigger_view_stage.py's
# test_view_stage_does_not_consume_main_rng_stream) and the trigger count/ids
# are identical, so an upsert reseed leaves no stale tail rows.
_STREAM_DIGEST = "cfaeb428e72ec85e0f6f0a0994ce7c9da8f8b2e5164f764521cfb0c68bb62255"


def _stable_token(v) -> str:
    """Pandas/numpy-version-proof scalar serialization: ``to_csv`` default
    float/datetime formatting is an implementation detail that could re-format
    on a dependency bump and fire the pin without any stream shift.

    * floats (incl. np.floating — float32 does NOT subclass float) → bit-exact
      ``float.hex``; ints via ``int()`` so numpy scalar repr never leaks in;
    * datetimes → ``isoformat()`` regardless of container (datetime.datetime,
      datetime.date, pd.Timestamp — a Timestamp subclasses datetime), so the
      digest is invariant to object-vs-datetime64 dtype inference;
    * every token is length-prefixed so the \\x1f/\\n join is injective even
      if a future string value contains a separator."""
    if isinstance(v, (bool, np.bool_)):
        s = "1" if v else "0"
    elif isinstance(v, (float, np.floating)):
        s = float(v).hex()
    elif isinstance(v, (int, np.integer)):
        s = str(int(v))
    elif isinstance(v, (datetime, date)):
        s = v.isoformat()
    else:
        s = str(v)
    return f"{len(s)}:{s}"


@pytest.mark.unit
def test_non_outcome_trigger_fields_stream_is_pinned():
    _, df, _ = _generate()
    missing = [f for f in _STREAM_FIELDS if f not in df.columns]
    assert not missing, f"stream-pinned fields missing from the frame: {missing}"
    payload = "\n".join(
        "\x1f".join(_stable_token(v) for v in row)
        for row in df[_STREAM_FIELDS].itertuples(index=False, name=None)
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert digest == _STREAM_DIGEST, (
        "TriggerGenerator per-record RNG stream shifted (non-acceptance/outcome "
        "fields changed for a fixed seed). If the change is INTENTIONAL, re-pin "
        "the digest AND note that an upsert reseed leaves stale tail rows when "
        "the generated trigger count shrinks (delete the orphan id range)."
    )


def _patient_df_with_share(share: float, seed: int = 7, n: int = _N_PATIENTS) -> pd.DataFrame:
    """Fixture with a CONTROLLED arm share — the base ``_patient_df`` fixture's
    coin flip lands at 0.4983 (seed 21), which is BELOW p[accepted]=0.50 and so
    exercises the degenerate fallback, not the production mixture (codex pass-1
    finding: the healthy q1/q0 algebra was untested by fixture accident)."""
    df = _patient_df(seed, n)
    rng = np.random.default_rng(seed + 1)
    df["trigger_accepted"] = (rng.random(n) < share).astype(int)
    return df


@pytest.mark.unit
def test_mixture_healthy_branch_algebra_and_marginals():
    """Production share is 0.5786 (treatment_arm.py) — a fixture in that zone
    must take the smooth-mixture branch: q0[accepted]=0, zero clipping, and the
    share-weighted mixture reproduces ACCEPTANCE_STATUS_P exactly."""
    patients = _patient_df_with_share(0.60)
    gen = TriggerGenerator(
        GeneratorConfig(seed=7, n_records=1800, brand=Brand.REMIBRUTINIB),
        patient_df=patients,
    )
    q = gen._acceptance_mixture()
    assert q is not None
    share = float(patients["trigger_accepted"].mean())
    statuses = list(gen.ACCEPTANCE_STATUS_VALUES)
    p = np.asarray(gen.ACCEPTANCE_STATUS_P, dtype=float)
    acc = statuses.index("accepted")
    q1, q0 = np.asarray(q[1]), np.asarray(q[0])
    assert q1[acc] < 1.0, "fixture unexpectedly hit the degenerate branch"
    assert q0[acc] == 0.0
    np.testing.assert_allclose(share * q1 + (1 - share) * q0, p, atol=1e-9)
    # Frame level: marginals hold AND arm-1 within-patient heterogeneity
    # survives (arm=1 patients still carry non-accepted triggers).
    df = gen.generate()
    dv = df[df["delivery_status"].isin(["delivered", "viewed"])]
    assert 0.45 <= (dv["acceptance_status"] == "accepted").mean() <= 0.55
    arm1 = set(patients.loc[patients["trigger_accepted"] == 1, "patient_id"])
    arm1_triggers = df[df["patient_id"].isin(arm1)]
    assert (arm1_triggers["acceptance_status"] != "accepted").any(), (
        "healthy branch must NOT force every arm=1 trigger to 'accepted'"
    )


@pytest.mark.unit
def test_mixture_degenerate_branch_is_explicit_and_loud():
    """share <= p[accepted]: q1 collapses to one-hot accepted and the realized
    acceptance marginal degrades toward the share — LOUDLY below the 0.45 floor
    the marginal-preservation test enforces, never a silent negative q0."""
    patients = _patient_df_with_share(0.30)
    gen = TriggerGenerator(
        GeneratorConfig(seed=7, n_records=1800, brand=Brand.REMIBRUTINIB),
        patient_df=patients,
    )
    q = gen._acceptance_mixture()
    assert q is not None
    statuses = list(gen.ACCEPTANCE_STATUS_VALUES)
    acc = statuses.index("accepted")
    q1 = np.asarray(q[1])
    assert q1[acc] == 1.0 and q1.sum() == 1.0, "degenerate branch must be one-hot accepted"
    df = gen.generate()
    dv = df[df["delivery_status"].isin(["delivered", "viewed"])]
    assert (dv["acceptance_status"] == "accepted").mean() < 0.42, (
        "degraded marginal must be loud (well below the preserved-band floor)"
    )


@pytest.mark.unit
def test_legacy_path_without_arm_column_still_generates():
    """Backward compatibility: a patient_df WITHOUT trigger_accepted (and the
    standalone path) must keep generating — pre-P4 callers and tests survive."""
    patients = _patient_df().drop(columns=["trigger_accepted"])
    gen = TriggerGenerator(
        GeneratorConfig(seed=21, n_records=600, brand=Brand.REMIBRUTINIB),
        patient_df=patients,
    )
    df = gen.generate()
    assert len(df) > 0
    assert (df["acceptance_status"] == "accepted").any()
