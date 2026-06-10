"""WS2-TR-008 Change-Fail Rate (CFR) substrate stamp (Shard 09).

CFR = count(triggers WHERE change_failed) / count(triggers WHERE
previous_trigger_id IS NOT NULL), over the rolling window. The synthetic trigger
generator emits none of the change-tracking columns (previous_trigger_id /
change_type / change_failed / change_outcome_delta), so the denominator is 0 ->
NULLIF -> CFR reads N/A. This stamps a supersession chain onto a fraction of the
synthetic triggers (each "change" references an earlier trigger_id) with a
realistic minority of failures, so CFR is non-NULL and non-degenerate.

All four columns exist on triggers (faithful-DB verified, nullable); the loader
carries them via TABLE_COLUMNS["triggers"] (Task 1).
"""

import numpy as np
import pandas as pd

_CHANGE_TYPES = ["threshold_update", "model_swap", "rule_change", "config_change"]


def stamp_change_tracking(
    df: pd.DataFrame,
    seed: int = 0,
    change_fraction: float = 0.3,
    fail_fraction: float = 0.2,
) -> pd.DataFrame:
    """Return a copy of df with change-tracking columns on a fraction of triggers.

    change_fraction of rows become "changes" that supersede an earlier trigger
    (previous_trigger_id -> an actual trigger_id) with a change_type; fail_fraction
    of those changes are marked change_failed=True (CFR numerator).
    """
    out = df.copy()
    out["previous_trigger_id"] = None
    out["change_type"] = None
    out["change_failed"] = False
    out["change_outcome_delta"] = np.nan
    n = len(out)
    if n < 2 or "trigger_id" not in out.columns:
        return out

    rng = np.random.default_rng(seed)
    ids = out["trigger_id"].to_numpy()
    n_change = max(1, int(n * change_fraction))
    # Pick change rows from the second half so each can point back at an earlier id.
    candidate_pos = np.arange(1, n)
    change_pos = rng.choice(candidate_pos, size=min(n_change, len(candidate_pos)), replace=False)

    fails = rng.random(len(change_pos)) < fail_fraction
    # Guarantee at least one failure so the CFR numerator > 0.
    if len(change_pos) > 0 and not fails.any():
        fails[0] = True

    for k, pos in enumerate(change_pos):
        prev_pos = int(rng.integers(0, pos))  # an earlier trigger
        out.iat[pos, out.columns.get_loc("previous_trigger_id")] = ids[prev_pos]
        out.iat[pos, out.columns.get_loc("change_type")] = str(rng.choice(_CHANGE_TYPES))
        out.iat[pos, out.columns.get_loc("change_failed")] = bool(fails[k])
        out.iat[pos, out.columns.get_loc("change_outcome_delta")] = round(
            float(rng.normal(0.0, 0.15)), 4
        )
    return out
