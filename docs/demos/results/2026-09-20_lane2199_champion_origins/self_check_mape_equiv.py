"""Self-check the statistic, not the plumbing.

Two questions I have asserted but not directly proved:

Q1. Is ``mape_on_origins(score, all_of_its_own_cutoffs)`` identical to that score's
    ``monthly_mape``? If not, the fix silently changes the metric as well as the
    comparison set, and the "no-op when all origins are equal" claim is false.

Q2. When the champion changes, can the payload show a champion whose REPORTED MAPE is
    the worst of the field? That is legitimate under the new rule but would read as a
    bug, so I want to know whether it actually happens on real data.
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
from src.kpi.forecast import backtest as bt  # noqa: E402
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
        query_id="selfcheck_2199",
    )


print("Q1: mape_on_origins(score, its own cutoffs) == score.monthly_mape ?")
worst = 0.0
n = 0
for brand in sorted(raw):
    for L in (36, 42, 48, 60):
        fc = svc.forecast_series(
            build(brand, raw[brand][-L:]), horizon=6, include_timesfm=False, cache=None
        )
        for s in fc.scores:
            got = bt.mape_on_origins(s, s.origin_cutoffs)
            worst = max(worst, abs(got - s.monthly_mape))
            n += 1
print(f"  checked {n} scores, max |difference| = {worst:.3e}")
print("  VERDICT:", "IDENTICAL" if worst < 1e-9 else "*** DIFFERENT — claim is false ***")

print()
print("Q2: does the champion ever have the WORST reported MAPE?")
for brand in sorted(raw):
    for L in (36, 42, 48, 60, 164):
        fc = svc.forecast_series(
            build(brand, raw[brand][-L:]), horizon=6, include_timesfm=False, cache=None
        )
        ranked = sorted(fc.scores, key=lambda s: s.monthly_mape)
        pos = [s.name for s in ranked].index(fc.champion) + 1
        shared = fc.selection_origins
        note = ""
        if pos > 1:
            note = f"   <<< champion is #{pos} of {len(ranked)} by REPORTED mape"
        print(
            f"  {brand:13s} L={L:3d} champion={fc.champion:26s} "
            f"reported_rank={pos}/{len(ranked)} shared_origins={shared}{note}"
        )
