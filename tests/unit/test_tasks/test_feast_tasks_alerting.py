"""#556 (H4): the staleness alert must fire on UNVERIFIABLE runs, not just on
explicitly-stale views.

After H1/H3, an all-unverifiable freshness run is ``fresh=False`` with an EMPTY
``stale_features`` list and the offending views in ``errors``. The alert helper
must emit for that case — returning early on empty ``stale_features`` would
re-suppress exactly the alert the fail-closed chain was built to surface.
"""

from __future__ import annotations

import logging

from src.tasks.feast_tasks import _send_staleness_alert


def test_alert_emitted_for_unverifiable_only_run(caplog):
    """fresh=False with stale_features=[] but errors=[...] must still alert."""
    result = {
        "status": "completed",
        "fresh": False,
        "stale_features": [],
        "errors": [{"feature_view": "hcp_engagement_features", "error": "No statistics available"}],
    }
    with caplog.at_level(logging.WARNING, logger="src.tasks.feast_tasks"):
        _send_staleness_alert(result)

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "ALERT" in messages, "an unverifiable-only freshness run must emit an alert (H4)"
    assert "hcp_engagement_features" in messages


def test_alert_emitted_for_stale_views(caplog):
    """Regression: explicitly-stale views still alert."""
    result = {
        "status": "completed",
        "fresh": False,
        "stale_features": [{"feature_view": "patient_journey_features", "age_hours": 99.0}],
        "errors": [],
    }
    with caplog.at_level(logging.WARNING, logger="src.tasks.feast_tasks"):
        _send_staleness_alert(result)

    assert "patient_journey_features" in " ".join(r.getMessage() for r in caplog.records)


def test_no_alert_when_all_fresh(caplog):
    """A genuinely fresh run (no stale, no errors) must NOT alert."""
    result = {"status": "completed", "fresh": True, "stale_features": [], "errors": []}
    with caplog.at_level(logging.WARNING, logger="src.tasks.feast_tasks"):
        _send_staleness_alert(result)

    assert "ALERT" not in " ".join(r.getMessage() for r in caplog.records)
