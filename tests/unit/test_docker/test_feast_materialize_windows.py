"""TTL-bounded Feast materialization windows (canonical TRx lane, owner decisions #1/#2 2026-09-15)."""

import importlib.util
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
MODULE = REPO / "docker" / "feast" / "materialize_windows.py"
ENTRYPOINT = REPO / "docker" / "feast" / "materializer-entrypoint.sh"
DOCKERFILE = REPO / "docker" / "Dockerfile.feast"
CONFIG = REPO / "config" / "feast_materialization.yaml"
HCP_FEATURES = REPO / "feature_repo" / "features" / "hcp_features.py"
UTC = timezone.utc
H = timedelta(hours=1)
D = timedelta(days=1)


def _mw():
    spec = importlib.util.spec_from_file_location("materialize_windows", MODULE)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: @dataclass resolves the defining module through sys.modules
    # (Python 3.12 raises AttributeError on an unregistered spec-loaded module).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_lookback_is_the_view_ttl():
    mw = _mw()
    assert mw.lookback_for(7 * D) == 7 * D
    assert mw.lookback_for(1 * D) == 1 * D


def test_lookback_without_a_ttl_or_above_the_cap_is_the_cap():
    mw = _mw()
    assert mw.lookback_for(None) == mw.MAX_LOOKBACK == 31 * D
    assert mw.lookback_for(timedelta(0)) == 31 * D
    assert mw.lookback_for(90 * D) == 31 * D


def test_a_late_weekly_batch_row_is_inside_the_first_cycle_after_the_rollup():
    """After Task 22A: Tuesday 2026-09-08's triggers arrive in the Monday 2026-09-14 03:00:55 batch,
    and the 03:15 beat writes their per-HCP row (event Tue 00:00). The first materializer cycle that
    starts after 03:15, at most one 6 h interval later, must load it under the 7-day TTL."""
    mw = _mw()
    event = datetime(2026, 9, 8, 0, 0, tzinfo=UTC)
    written = datetime(2026, 9, 14, 3, 15, tzinfo=UTC)
    for cycle_end in (written + timedelta(seconds=1), written + 6 * H):
        [window] = mw.compute_windows(cycle_end, {"hcp_conversion_features": 7 * D})
        assert window.start <= event <= window.end


def test_a_one_day_lookback_would_miss_that_row():
    [window] = _mw().compute_windows(datetime(2026, 9, 14, 9, 15, tzinfo=UTC), {"v": 1 * D})
    assert window.start > datetime(2026, 9, 8, 0, 0, tzinfo=UTC)


def test_the_7_day_ttl_covers_the_weekly_lateness_plus_one_cycle():
    """TTL(hcp_conversion_features) >= 6 days (a Tue..Mon batch) + the per-HCP beat time + one loop interval."""
    from src.workers.celery_app import celery_app

    beat = celery_app.conf.beat_schedule["business-metrics-per-hcp-rollup"]["schedule"]
    [hour], [minute] = sorted(beat.hour), sorted(beat.minute)
    interval = timedelta(
        hours=float(yaml.safe_load(CONFIG.read_text())["schedule"]["interval_hours"])
    )
    match = re.search(
        r'name="hcp_conversion_features",.*?ttl=timedelta\(days=(\d+)\)',
        HCP_FEATURES.read_text(),
        re.S,
    )
    assert match, "hcp_conversion_features ttl not found"
    lateness = 6 * D + timedelta(hours=hour, minutes=minute)
    assert _mw().lookback_for(timedelta(days=int(match.group(1)))) >= lateness + interval


def test_views_sharing_a_ttl_share_one_call_shortest_first():
    mw = _mw()
    end = datetime(2026, 9, 15, 9, 0, tzinfo=UTC)
    windows = mw.compute_windows(end, {"a": 7 * D, "b": 7 * D, "c": 1 * D})
    assert mw.group_windows(windows) == [
        (end - 1 * D, end, ("c",)),
        (end - 7 * D, end, ("a", "b")),
    ]


def test_a_naive_end_is_refused():
    with pytest.raises(ValueError):
        _mw().compute_windows(datetime(2026, 9, 15, 9, 0), {"a": 1 * D})


def test_the_loop_uses_bounded_windows_under_the_lock_and_the_image_ships_the_helper():
    body = "\n".join(
        line for line in ENTRYPOINT.read_text().splitlines() if not line.lstrip().startswith("#")
    )
    assert "materialize-incremental" not in body
    assert "/materialize_windows.py" in body
    assert 'flock "$LOCK" feast --chdir /feast materialize ' in body
    assert "feast/materialize_windows.py" in DOCKERFILE.read_text()
