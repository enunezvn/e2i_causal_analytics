"""FAITHFUL baseline: drive the REAL forecast_series, not score_model directly.

The first probe bypassed service.py:400, which applies MIN_BACKTEST_ORIGINS PER
MODEL and skips a model that cannot reach 4 origins. That gate changes which
cases can even produce a mismatched comparison, so the flip count from the
direct-score probe is not the number that matters.

Prints, per (brand, truncation length), what the production path actually does:
which models were scored, on how many origins each, and who won.
"""

import csv
import sys
from collections import defaultdict
from datetime import date

WT = "/home/enunez/Projects/e2i_causal_analytics/.claude/worktrees/2199-champion-origins"
sys.path.insert(0, WT)
import src  # noqa: E402

assert "2199-champion-origins" in src.__file__, src.__file__

from src.kpi.canonical_volume_series import (  # noqa: E402
    CanonicalVolumeSeries,
    MonthlyVolumePoint,
    month_end,
)
from src.kpi.forecast import service as svc  # noqa: E402

CSV = (
    "/tmp/claude-1000/-home-enunez-Projects-e2i-causal-analytics/"
    "934cfe33-5894-4dfe-9024-c3ac1924f984/scratchpad/l2199/series.csv"
)

raw = defaultdict(list)
with open(CSV) as fh:
    for brand, mo, v, _n in csv.reader(fh):
        raw[brand].append((date.fromisoformat(mo), float(v)))
for b in raw:
    raw[b].sort()


def build(brand, pairs):
    pts = tuple(MonthlyVolumePoint(m, v, 4) for m, v in pairs)
    return CanonicalVolumeSeries(
        metric="trx",
        brand=brand,
        region=None,
        points=pts,
        data_through=month_end(pts[-1].month),
        as_of=date(2026, 9, 20),
        dropped_in_progress=(),
        dropped_incomplete=(),
        query_id="probe_2199",
    )


print(f"MIN_BACKTEST_ORIGINS={svc.MIN_BACKTEST_ORIGINS}  MIN_OBSERVATIONS={svc.MIN_OBSERVATIONS}")
print()

mismatched = 0
total = 0
for brand in sorted(raw):
    for L in (30, 36, 42, 48, 54, 60, 164):
        pairs = raw[brand][-L:]
        try:
            fc = svc.forecast_series(
                build(brand, pairs), horizon=6, include_timesfm=False, cache=None
            )
        except Exception as exc:
            print(f"{brand:13s} L={L:3d}  REFUSED: {type(exc).__name__}: {exc}")
            continue
        counts = {s.name: s.n_origins for s in fc.scores}
        total += 1
        distinct = len(set(counts.values()))
        flag = ""
        if distinct > 1:
            mismatched += 1
            flag = "   <<< UNEQUAL ORIGIN SETS COMPARED"
        skipped = [n for n, _ in fc.skipped] if hasattr(fc, "skipped") else []
        print(f"{brand:13s} L={L:3d}  champion={fc.champion} origins_used={fc.origins_used}{flag}")
        print(f"                scored={counts}  skipped={skipped}")
print()
print(f"=== {mismatched} of {total} served cases compared UNEQUAL origin sets ===")
