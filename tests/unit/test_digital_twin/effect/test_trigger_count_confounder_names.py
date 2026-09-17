"""The twin's volume confounder is the per-HCP TRIGGER total (migration 144).

Intent (9dc0a3468, 8f8d242c9, 2b66b5f0f): {market_share, total_rx_count} are the
pre-treatment controls because the gold-standard DGP generates every treatment
channel and the outcome baseline from exactly these observed columns; the rename
changes the name, not the causal role. The estimator must adjust on the renamed
column and fail closed on a frame that still carries only the legacy name.
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.effect import cohort_causal_estimator, cohort_loader, provider
from src.digital_twin.effect.errors import EffectCause, EffectDataUnavailable


def test_cohort_columns_and_confounders_use_the_honest_name():
    assert "triggers_total_count" in cohort_loader._COHORT_COLUMNS.split(",")
    assert "triggers_total_count" in cohort_loader._NUMERIC_COLUMNS
    assert provider.COHORT_CONFOUNDERS == ("market_share", "triggers_total_count")
    assert cohort_causal_estimator.DEFAULT_CONFOUNDERS == ("market_share", "triggers_total_count")
    assert cohort_causal_estimator._LOG_CONFOUNDERS == frozenset({"triggers_total_count"})


def _small_cohort(volume_column: str, n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regions = np.repeat(["northeast", "west", "south", "midwest"], n)
    market = rng.uniform(0.0, 1.0, len(regions))
    volume = np.expm1(np.abs(rng.normal(0.0, 1.0, len(regions))) * 2.0)
    engagement = 10.0 / (1.0 + np.exp(-(1.6 * (market - 0.5) + rng.normal(0.0, 0.5, len(regions)))))
    outcome = (
        0.5
        + 0.8 * market
        + 0.2 * (engagement > np.median(engagement))
        + rng.normal(0.0, 0.25, len(regions))
    )
    return pd.DataFrame(
        {
            "region": regions,
            "engagement_score": engagement,
            "market_share": market,
            volume_column: volume,
            "conversion_rate": np.clip(outcome, 0.0, None),
        }
    )


def test_a_frame_with_only_the_legacy_volume_column_fails_closed():
    """Never an under-adjusted estimate that LOOKS adjusted (8f8d242c9).

    Pins WHY it refuses: the renamed confounder is the missing column. Without
    the cause and the column name this assertion would pass on any unrelated
    frame-validation failure, and the partner test below proves the same frame
    with the renamed column estimates fine.
    """
    with pytest.raises(EffectDataUnavailable) as excinfo:
        cohort_causal_estimator.estimate_cohort_effect(
            _small_cohort("total_rx_count"), "engagement_score"
        )
    assert excinfo.value.cause is EffectCause.REQUIRED_COLUMN_MISSING
    assert "triggers_total_count" in str(excinfo.value)


@pytest.mark.slow
def test_the_same_frame_with_the_renamed_volume_column_estimates_fine():
    """Discriminating control for the fail-closed test above: identical builder,
    identical seed, only the volume column's NAME differs. The refusal above is
    therefore caused by the renamed confounder's absence and nothing else."""
    eff = cohort_causal_estimator.estimate_cohort_effect(
        _small_cohort("triggers_total_count"), "engagement_score"
    )
    assert eff.adjustment_set == ["region", "market_share", "triggers_total_count"]
    assert eff.ate_ci_lower < eff.ate < eff.ate_ci_upper
    assert eff.n == 1200


def test_the_engagement_backfill_reads_the_honest_name():
    import scripts.backfill_segment_engagement as backfill

    source = inspect.getsource(backfill)
    assert "triggers_total_count" in source
    assert not re.search(r"\b(trx_count|nrx_count|total_rx_count)\b", source)
