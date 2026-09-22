"""Lane A real-DB probe (spec §3A gates): the loaded table's arm split and
per-outcome positives equal the causal export parquet.

Gate: ``E2I_DB_INTEGRATION=1``; skipped when the parquet is absent (data/ is
gitignored — it exists only on the droplet's main checkout) or when migration 148
has not been applied (the loader's live read returns None).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="E2I_DB_INTEGRATION!=1; set to 1 to run against the real Supabase DB.",
)

# data/ is gitignored: derive the default from THIS checkout's root (absent in a
# worktree unless E2I_OPTUM_CAUSAL_PARQUET points at the main checkout's copy).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARQUET = Path(
    os.getenv(
        "E2I_OPTUM_CAUSAL_PARQUET",
        str(
            _REPO_ROOT
            / "data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet"
        ),
    )
)


def test_live_arm_split_equals_the_export():
    if not _PARQUET.exists():
        pytest.skip(f"causal export not present at {_PARQUET}")
    from scripts.load_optum_causal_cohort import (
        _client,
        arm_split,
        fetch_live_split,
        load_frame,
        verify,
    )

    expected = arm_split(load_frame(_PARQUET))
    live = fetch_live_split(_client())
    if live is None:
        pytest.skip("optum_biologic_persistence_causal unreachable (migration 148 not applied?)")
    assert live["n"] > 0, "live table is empty — the owner-GO load has not run"
    for arm, n_arm in live["arms"].items():
        assert n_arm > 0, f"live table has no {arm} rows — no causal contrast is estimable"
    assert verify(expected, live) == []
