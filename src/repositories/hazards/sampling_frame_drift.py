"""Sampling-frame-drift hazard for the ml_patients cohort.

Plants a training distribution where the targeted segment is heavily
under-represented (default 5%) relative to the deployment-reference
distribution where the same segment is over-represented (default 40%).
The drift signal is then surfaced via two channels:

1. The returned DataFrame has the segment artificially down-sampled.
2. ``df.attrs["deployment_reference"]`` is populated with the deployment-
   reference distribution shape that the sampling-frame audit consumes
   (mirroring ``scope_spec["deployment_reference"]`` schema). Callers
   wire this onto ``scope_spec`` before invoking the audit.

Detection signal:
  ``state["sampling_frame_audit_report"]["max_drift_score"] > 0.3`` AND
  ``"sampling_frame_drift:" in str(state["blocking_issues"])`` (the
  blocking gate added in PR #35).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_sampling_frame_drift(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    drift_segment_col: str = "age_group",
    drift_segment_value: str = ">65",
    train_fraction: float = 0.05,
    deployment_fraction: float = 0.40,
) -> pd.DataFrame:
    """Down-sample a segment in train; attach deployment_reference to attrs.

    Args:
        df: ml_patients DataFrame (input is not mutated).
        seed: RNG seed for reproducibility (controls which rows in the
            targeted segment are kept after down-sampling).
        drift_segment_col: Column defining the segment.
        drift_segment_value: Value within ``drift_segment_col`` to target.
            Convenience alias: ``"elderly"`` maps to ``">65"`` when
            ``drift_segment_col == "age_group"``.
        train_fraction: Fraction of rows in the returned DataFrame that
            should belong to the targeted segment (default 5%).
        deployment_fraction: Fraction of the deployment-reference
            distribution belonging to the targeted segment (default 40%).
            Embedded into ``df.attrs['deployment_reference']`` so the
            sampling-frame audit can compare distributions.

    Returns:
        New DataFrame with the targeted segment down-sampled to
        ``train_fraction`` of total rows, and ``df.attrs[
        'deployment_reference']`` populated with a categorical_freq
        distribution that puts ``deployment_fraction`` mass on the
        targeted segment value.
    """
    if drift_segment_col not in df.columns:
        raise ValueError(
            f"drift_segment_col {drift_segment_col!r} not found in DataFrame; "
            f"available: {list(df.columns)}"
        )

    resolved_value = (
        ">65"
        if (drift_segment_col == "age_group" and drift_segment_value == "elderly")
        else drift_segment_value
    )
    rng = np.random.default_rng(seed)

    in_segment_mask = df[drift_segment_col].astype(str) == str(resolved_value)
    in_segment = df[in_segment_mask]
    out_segment = df[~in_segment_mask]

    n_out = len(out_segment)
    if n_out == 0:
        # Edge case: the entire cohort is in-segment, can't synthesise
        # under-representation. Return the input copy with attrs only.
        out = df.copy()
        out.attrs["deployment_reference"] = _build_reference(
            df, drift_segment_col, resolved_value, deployment_fraction
        )
        return out

    # We want target_in_segment such that
    #   target_in_segment / (target_in_segment + n_out) == train_fraction
    # -> target_in_segment = train_fraction * n_out / (1 - train_fraction)
    if not 0.0 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0, 1)")
    target_in_segment = int(round(train_fraction * n_out / max(1.0 - train_fraction, 1e-9)))
    target_in_segment = min(target_in_segment, len(in_segment))

    if target_in_segment == 0:
        sampled_in = in_segment.iloc[0:0]
    else:
        # rng.choice without replacement preserves row identity for the
        # downstream sampling-frame audit's value_counts.
        keep_idx = rng.choice(len(in_segment), size=target_in_segment, replace=False)
        sampled_in = in_segment.iloc[keep_idx]

    out = pd.concat([out_segment, sampled_in], ignore_index=True)
    out.attrs["deployment_reference"] = _build_reference(
        df, drift_segment_col, resolved_value, deployment_fraction
    )
    return out


def _build_reference(
    original_df: pd.DataFrame,
    segment_col: str,
    segment_value: str,
    deployment_fraction: float,
) -> dict:
    """Construct a deployment_reference dict shaped for sampling_frame_audit.

    Builds a categorical_freq distribution that:
      * puts ``deployment_fraction`` mass on ``segment_value``
      * spreads the remaining mass across other observed values in
        proportion to their original frequency
    """
    if not 0.0 <= deployment_fraction <= 1.0:
        raise ValueError("deployment_fraction must be in [0, 1]")

    other_values = (
        original_df[segment_col]
        .astype(str)
        .loc[lambda s: s != str(segment_value)]
        .value_counts(normalize=True)
    )
    remaining = 1.0 - deployment_fraction
    other_freq = (other_values * remaining).to_dict()
    other_freq[str(segment_value)] = deployment_fraction
    return {
        "distributions": {
            segment_col: {"categorical_freq": other_freq},
        },
        "n_reference_samples": int(len(original_df)),
    }
