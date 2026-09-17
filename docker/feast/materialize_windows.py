"""Bounded-lookback materialization windows for the Feast sidecar (canonical TRx lane).

`feast materialize-incremental` starts each view at its last end time, so a row written LATE
(event time before the previous run's end) is never loaded. The per-HCP business_metrics rollup
writes a weekly trigger batch on Monday 03:15 with event times back to the previous Tuesday 00:00
(canonical TRx lane Task 22A). Each cycle instead materializes [end - lookback, end] per online view,
with lookback = the view's TTL (no TTL, or a TTL above MAX_LOOKBACK: MAX_LOOKBACK). Re-reading an
overlap is idempotent: Feast 0.43's Redis online store skips a write whose event time is <= the
stored one (feast/infra/online_stores/redis.py:313).

WHAT A TTL-SIZED LOOKBACK DOES NOT COVER
----------------------------------------
Owner decision #4 (2026-09-15): the lookback never exceeds a view's TTL. A 1-day TTL is the view's
own declaration that it serves fresh values only, so loading an older row into it would hand a
consumer stale data labelled fresh. Two consequences are known and accepted, both measured by the
lane dispatcher on 2026-09-17 and both SNAPSHOTS of the data as it was then, not invariants:

1. The four 1-day-TTL online views -- territory_performance_features, patient_adherence_features,
   hcp_engagement_features, trigger_response_features -- load only entities whose newest row is
   inside 24 h. Against a weekly source cadence that means "entities whose newest row is a Monday".
   None of the four has an online consumer today: they appear in src/ only inside the view->source
   metadata map at src/feature_store/feast_client.py:1015-1021. Do NOT "fix" the materializer to
   make them serve -- the mismatch is between a 1-day TTL and a weekly delivery cadence, and
   resolving it (daily delivery, or a longer declared TTL) is a product decision, not a
   materializer change. If a consumer ever appears, this is why the view is empty.

   territory_performance_features specifically: post-deploy lateness is 3 h 45 m -- every per-HCP
   write batch in the 56 days to 2026-09-17 carried a same-day metric_date (7 of 7, span 0), so
   Task 22B's arrival-based date selection gives the 03:45 territory rollup a same-day date to
   write -- which is INSIDE its 1-day TTL. BEWARE when re-measuring: prod still runs the pre-22B
   rollup until this lane deploys, and there the figure is 27 h 45 m (an exact constant, because
   the old selection always wrote exactly yesterday's date at 03:45). A measurement taken on prod
   before the deploy describes the OLD selection and says nothing about this one.

2. A trigger can arrive with an event timestamp older than any view's TTL. The arrival window in
   the rollups bounds how late an ARRIVAL may be, not how old its EVENT is, so a trigger arriving
   today with a 14-day-old trigger_timestamp creates a per-HCP row for a 14-day-old metric_date;
   if that date had no prior row its created_at is now, putting it outside even the 7-day views.
   Measured over the 56 days to 2026-09-17 (an upstream property that Tasks 22A/22B do not
   change): max arrival lateness 14 d 03:00:41, median 2 d 03:00:56; 20 of 849 arrivals (2.4%)
   later than 7 d, creating 8 distinct event dates. Expected inconsequential for online serving,
   which takes the newest row per entity: these are old dates, unlikely to be any entity's newest.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Mapping, Optional, Tuple

MAX_LOOKBACK = timedelta(days=31)
CONFIG_PATH = "/feast/feast_materialization.yaml"


@dataclass(frozen=True)
class Window:
    view: str
    start: datetime
    end: datetime


def lookback_for(ttl: Optional[timedelta], cap: timedelta = MAX_LOOKBACK) -> timedelta:
    if ttl is None or ttl.total_seconds() <= 0:
        return cap
    return min(ttl, cap)


def compute_windows(
    end: datetime,
    views: Mapping[str, Optional[timedelta]],
    cap: timedelta = MAX_LOOKBACK,
) -> List[Window]:
    if end.tzinfo is None:
        raise ValueError("end must be timezone-aware (UTC)")
    return [Window(name, end - lookback_for(ttl, cap), end) for name, ttl in sorted(views.items())]


def group_windows(windows: List[Window]) -> List[Tuple[datetime, datetime, Tuple[str, ...]]]:
    grouped: Dict[Tuple[datetime, datetime], List[str]] = {}
    for w in windows:
        grouped.setdefault((w.start, w.end), []).append(w.view)
    return sorted(
        ((start, end, tuple(sorted(names))) for (start, end), names in grouped.items()),
        key=lambda g: g[1] - g[0],
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2026-09-15T09:17:27")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--repo", default="/feast")
    args = parser.parse_args(argv)

    import yaml
    from feast import FeatureStore

    cfg = yaml.safe_load(open(args.config)) or {}
    enabled = {
        name
        for name, v in (cfg.get("feature_views") or {}).items()
        if (v or {}).get("enabled", True)
    }
    store = FeatureStore(repo_path=args.repo)
    views = {
        fv.name: fv.ttl for fv in store.list_feature_views() if fv.online and fv.name in enabled
    }
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    for start, stop, names in group_windows(compute_windows(end, views)):
        print(
            start.strftime("%Y-%m-%dT%H:%M:%S"), stop.strftime("%Y-%m-%dT%H:%M:%S"), " ".join(names)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
