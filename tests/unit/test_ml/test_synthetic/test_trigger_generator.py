"""Tests for TriggerGenerator brand_id emission (6B-infra-6).

Covers the audit-driven gap surfaced by the 2026-04-28 pre-apply blast-radius
audit: migration 033 promotes ``triggers.brand_id`` to NOT NULL with no DEFAULT,
so every producer of trigger rows must emit ``brand_id``. ``TriggerGenerator``
is the canonical synthetic-data producer (consumed by
``scripts/load_synthetic_data.py``); without ``brand_id`` in its output, fresh-DB
re-bootstrap fails on the ``triggers`` insert.

Sourcing strategy is Option A from the plan: pass-through of the existing
``brand`` value, since the post-migration column semantic IS the brand
identifier.
"""

import pandas as pd
import pytest

from src.ml.synthetic.config import BRANDS, Brand
from src.ml.synthetic.generators import GeneratorConfig, TriggerGenerator


def _make_patient_df(brands: list[str], n_per_brand: int = 5) -> pd.DataFrame:
    """Build a minimal patient_df satisfying TriggerGenerator's `.get()` lookups."""
    rows = []
    for brand in brands:
        for i in range(n_per_brand):
            rows.append(
                {
                    "patient_id": f"pt_{brand}_{i:03d}",
                    "hcp_id": f"hcp_{i:03d}",
                    "engagement_score": 5.0 + (i % 5),
                    "treatment_initiated": i % 2,
                    "journey_start_date": "2023-01-01",
                    "brand": brand,
                }
            )
    return pd.DataFrame(rows)


def _make_hcp_df(n: int = 5) -> pd.DataFrame:
    """Build a minimal hcp_df (TriggerGenerator only checks `is not None`)."""
    return pd.DataFrame({"hcp_id": [f"hcp_{i:03d}" for i in range(n)]})


class TestTriggerGeneratorBrandId:
    """6B-infra-6: brand_id column must be emitted by TriggerGenerator."""

    def test_linked_mode_emits_brand_id_column(self):
        """Linked-mode path (patient_df + hcp_df): brand_id is in output columns."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        assert "brand_id" in df.columns, "TriggerGenerator must emit brand_id"
        assert df["brand_id"].notna().all(), "brand_id must be populated for every row"

    def test_linked_mode_brand_id_matches_brand(self):
        """Linked-mode path: brand_id is the pass-through of the brand value."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        assert (df["brand_id"] == df["brand"]).all(), (
            "Option A sourcing: brand_id must equal brand on every row"
        )

    def test_linked_mode_brand_id_values_are_brand_enum(self):
        """Linked-mode path: brand_id values are members of the Brand enum."""
        patient_df = _make_patient_df(brands=BRANDS)
        hcp_df = _make_hcp_df()
        config = GeneratorConfig(seed=42, n_records=len(patient_df))
        gen = TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df)

        df = gen.generate()

        valid_values = {b.value for b in Brand}
        assert set(df["brand_id"].unique()).issubset(valid_values), (
            "brand_id values must be drawn from Brand enum"
        )

    def test_standalone_mode_emits_brand_id_column(self):
        """Standalone path (no patient_df / hcp_df): brand_id is in output columns."""
        config = GeneratorConfig(seed=42, n_records=50)
        gen = TriggerGenerator(config)

        df = gen.generate()

        assert "brand_id" in df.columns, "Standalone path must also emit brand_id"
        assert df["brand_id"].notna().all(), "brand_id must be populated for every row"
        assert (df["brand_id"] == df["brand"]).all(), (
            "Option A sourcing: brand_id must equal brand on every row"
        )

    @pytest.mark.parametrize("brand", list(Brand))
    def test_single_brand_propagates_to_brand_id(self, brand: Brand):
        """When config pins a single brand, brand_id mirrors it across all rows."""
        config = GeneratorConfig(seed=42, n_records=30, brand=brand)
        gen = TriggerGenerator(config)

        df = gen.generate()

        assert (df["brand_id"] == brand.value).all(), (
            f"All brand_id values must equal {brand.value} when config pins it"
        )


# ---------------------------------------------------------------------------
# #1118 (WS2-TR-005) — false_positive_flag must be populated by the active DGP
# ---------------------------------------------------------------------------


def _standalone_df(n: int = 4000, seed: int = 42):
    """Generate a standalone-mode triggers frame (no patient/hcp linkage)."""
    config = GeneratorConfig(seed=seed, n_records=n)
    return TriggerGenerator(config).generate()


def _linked_df(n_patients: int = 500, seed: int = 42):
    """Generate a linked-mode triggers frame off a synthetic patient_df."""
    patient_df = _make_patient_df(brands=BRANDS, n_per_brand=n_patients // len(BRANDS))
    config = GeneratorConfig(seed=seed, n_records=len(patient_df) * 4)
    return TriggerGenerator(config, patient_df=patient_df, hcp_df=_make_hcp_df()).generate()


class TestFalsePositiveFlag:
    """#1118: the active seeder must emit false_positive_flag, tied to the
    outcome story so WS2-TR-005 and WS2-TR-001 describe the same reality."""

    def test_column_emitted_standalone(self):
        df = _standalone_df(n=200)
        assert "false_positive_flag" in df.columns, (
            "#1118: TriggerGenerator must emit false_positive_flag or the "
            "column stays schema-default FALSE and TR-005 is vacuously GOOD"
        )
        assert df["false_positive_flag"].notna().all()

    def test_column_emitted_linked(self):
        df = _linked_df(n_patients=50)
        assert "false_positive_flag" in df.columns
        assert df["false_positive_flag"].notna().all()

    @pytest.mark.parametrize("mode", ["standalone", "linked"])
    def test_flag_implies_tracked_and_unproductive(self, mode: str):
        """A false alert can only be MARKED when the outcome was tracked and
        demonstrably did not materialize (outcome_value null or <= 0)."""
        df = _standalone_df() if mode == "standalone" else _linked_df()
        flagged = df[df["false_positive_flag"].astype(bool)]
        assert len(flagged) > 0, "some triggers must be flagged false-positive"
        assert flagged["outcome_tracked"].astype(bool).all(), (
            "flag must only be set on outcome-tracked triggers"
        )
        productive = flagged["outcome_value"].notna() & (flagged["outcome_value"] > 0)
        assert not productive.any(), (
            "flag must never be set on triggers with a positive tracked outcome "
            "(would contradict TR-001 precision)"
        )

    def test_realized_rate_in_warning_band(self):
        """Calibration: P(tracked)=0.40 x P(unproductive|tracked)~0.575 x
        P(marked|unproductive)=0.60 => ~0.14 false-alert rate over ALL triggers
        (TR-005 denominator) — WARNING band (0.10, 0.20], coherent with TR-001
        precision ~0.38-0.43 CRITICAL."""
        df = _standalone_df(n=4000)
        rate = df["false_positive_flag"].astype(bool).mean()
        assert 0.10 < rate <= 0.20, (
            f"realized false-alert rate {rate:.3f} must land in the WARNING band "
            "(0.10, 0.20] to stay coherent with CRITICAL trigger precision"
        )

    def test_linked_rate_in_warning_band(self):
        df = _linked_df(n_patients=500)
        rate = df["false_positive_flag"].astype(bool).mean()
        assert 0.08 < rate <= 0.22, f"linked-mode false-alert rate {rate:.3f} out of band"

    def test_deterministic_same_seed(self):
        """Reseed idempotency: same seed => identical flags and statuses."""
        df_a = _standalone_df(n=500, seed=123)
        df_b = _standalone_df(n=500, seed=123)
        pd.testing.assert_series_equal(df_a["false_positive_flag"], df_b["false_positive_flag"])
        pd.testing.assert_series_equal(df_a["acceptance_status"], df_b["acceptance_status"])
        pd.testing.assert_series_equal(df_a["trigger_id"], df_b["trigger_id"])


# ---------------------------------------------------------------------------
# #1119 (WS2-TR-006) — 'overridden' acceptance_status arm must exist
# ---------------------------------------------------------------------------


class TestOverriddenAcceptanceStatus:
    """#1119: 'overridden' must be a reachable acceptance_status so the
    Override Rate numerator is structurally possible."""

    def test_generator_constant_includes_overridden(self):
        assert "overridden" in TriggerGenerator.ACCEPTANCE_STATUS_VALUES

    def test_validation_schema_constant_includes_overridden(self):
        from src.ml.synthetic.validation.schemas import ACCEPTANCE_STATUS_VALUES

        assert "overridden" in ACCEPTANCE_STATUS_VALUES, (
            "#1119: pandera Check.isin would reject 'overridden' rows at load"
        )

    def test_pandera_accepts_overridden(self):
        from src.ml.synthetic.validation.schemas import TriggerSchema

        col = TriggerSchema.columns["acceptance_status"]
        df = pd.DataFrame(
            {"acceptance_status": ["pending", "accepted", "rejected", "expired", "overridden"]}
        )
        col.validate(df)  # must not raise SchemaError

    @pytest.mark.parametrize("mode", ["standalone", "linked"])
    def test_overridden_emitted(self, mode: str):
        df = _standalone_df() if mode == "standalone" else _linked_df()
        assert (df["acceptance_status"] == "overridden").any(), (
            "#1119: the DGP must actually emit 'overridden' rows"
        )

    @pytest.mark.parametrize("mode", ["standalone", "linked"])
    def test_overridden_only_on_delivered_or_viewed(self, mode: str):
        """Only delivered triggers can be overridden (delivery gates any
        acceptance disposition — trigger_generator delivery gate)."""
        df = _standalone_df() if mode == "standalone" else _linked_df()
        overridden = df[df["acceptance_status"] == "overridden"]
        assert overridden["delivery_status"].isin(["delivered", "viewed"]).all()

    def test_override_rate_of_delivered_in_low_to_mid_teens(self):
        """Calibration: P(overridden | delivered/viewed) = 0.14 — just under the
        TR-006 target 0.15 (GOOD, but honestly earned and non-degenerate)."""
        df = _standalone_df(n=4000)
        delivered = df[df["delivery_status"].isin(["delivered", "viewed"])]
        rate = (delivered["acceptance_status"] == "overridden").mean()
        assert 0.10 < rate < 0.18, (
            f"override rate of delivered {rate:.3f} must land in the low-to-mid teens"
        )

    def test_accepted_share_of_delivered_preserved(self):
        """The 'overridden' arm must be carved out of pending/rejected/expired —
        NOT out of 'accepted' — so TR-001 precision (~P(accepted)) and the
        designed trigger->prescription conversion lift substrate are unperturbed."""
        df = _standalone_df(n=4000)
        delivered = df[df["delivery_status"].isin(["delivered", "viewed"])]
        accepted_share = (delivered["acceptance_status"] == "accepted").mean()
        assert 0.45 < accepted_share < 0.55, (
            f"accepted share of delivered {accepted_share:.3f} drifted from the "
            "designed 0.50 — this perturbs TR-001/TR-004 and the conversion lift"
        )


# ---------------------------------------------------------------------------
# #1125 — generator/schema value-set drift tripwire
# ---------------------------------------------------------------------------


def _schema_isin_allowed(column_name: str) -> set:
    """The value set the pandera TriggerSchema actually ENFORCES, introspected
    from the live Check.isin object (not a parallel constant, so a stale
    schema cannot vouch for itself)."""
    from src.ml.synthetic.validation.schemas import TriggerSchema

    column = TriggerSchema.columns[column_name]
    for check in column.checks:
        if check.name == "isin":
            return set(check.statistics["allowed_values"])
    raise AssertionError(f"TriggerSchema.{column_name} has no Check.isin check")


class TestTriggerSchemaValueSetDrift:
    """#1125: TriggerSchema's Check.isin sets had drifted from the generator
    (rejecting emitted 'viewed'/'crm'/'mobile'/'rep_alert', accepting
    never-emitted 'sent'/'call'/'in_person'). Both sides now alias the shared
    constants in src/ml/synthetic/config.py; these tests assert the emitted
    value sets stay within the schema-enforced sets so any future divergence
    fails CI instead of becoming a latent load-path booby trap."""

    @pytest.mark.parametrize(
        ("generator_values", "column_name"),
        [
            (TriggerGenerator.DELIVERY_CHANNELS, "delivery_channel"),
            (TriggerGenerator.DELIVERY_STATUS_VALUES, "delivery_status"),
            (TriggerGenerator.ACCEPTANCE_STATUS_VALUES, "acceptance_status"),
            (TriggerGenerator.TRIGGER_TYPES, "trigger_type"),
        ],
    )
    def test_generator_value_sets_within_schema_isin(self, generator_values, column_name):
        allowed = _schema_isin_allowed(column_name)
        drifted = set(generator_values) - allowed
        assert not drifted, (
            f"#1125 drift: TriggerGenerator emits {sorted(drifted)} for "
            f"'{column_name}' but TriggerSchema Check.isin only allows "
            f"{sorted(allowed)} — reconcile via the shared constants in "
            "src/ml/synthetic/config.py"
        )

    @pytest.mark.parametrize("mode", ["standalone", "linked"])
    def test_emitted_values_within_schema_isin(self, mode: str):
        """End-to-end: values actually present in a generated frame must pass
        the schema's isin checks (catches emission paths that bypass the
        class constants)."""
        df = _standalone_df(n=2000) if mode == "standalone" else _linked_df()
        for column_name in ("delivery_channel", "delivery_status", "acceptance_status"):
            allowed = _schema_isin_allowed(column_name)
            emitted = set(df[column_name].astype(str))
            assert emitted <= allowed, (
                f"#1125 drift: generated '{column_name}' contains "
                f"{sorted(emitted - allowed)} which TriggerSchema would reject"
            )
