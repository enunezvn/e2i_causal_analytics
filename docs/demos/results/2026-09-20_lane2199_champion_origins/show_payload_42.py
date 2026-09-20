"""What a reader now sees at Remibrutinib L=42 — the case where the champion has the
worst reported MAPE. Before the ordering fix the champion was the LAST row."""

import csv
import json
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

pairs = raw["Remibrutinib"][-42:]
pts = tuple(MonthlyVolumePoint(m, v, 4) for m, v in pairs)
series = CanonicalVolumeSeries(
    metric="trx",
    brand="Remibrutinib",
    region=None,
    points=pts,
    data_through=month_end(pts[-1].month),
    as_of=date(2026, 9, 20),
    dropped_in_progress=(),
    dropped_incomplete=(),
    query_id="show_2199",
)
fc = svc.forecast_series(series, horizon=6, include_timesfm=False, cache=None)
bt_block = fc.to_payload()["backtest"]
print("champion:", fc.champion)
print("selection_origins:", bt_block["selection_origins"])
print()
print(f"{'#':<3}{'model':28s}{'own MAPE':>10s}{'on shared':>12s}{'origins':>9s}")
for i, r in enumerate(bt_block["models"], 1):
    mark = "  <-- CHAMPION" if r["model"] == fc.champion else ""
    print(
        f"{i:<3}{r['model']:28s}{r['monthly_mape_pct']:>10.2f}"
        f"{r['monthly_mape_on_shared_origins_pct']:>12.2f}{r['origins_scored']:>9d}{mark}"
    )
print()
print("selection_rule:", bt_block["selection_rule"])
print()
print("strict-JSON serialisable:", bool(json.dumps(fc.to_payload(), allow_nan=False, default=str)))
band_ok = all(p["lower"] <= p["value"] <= p["upper"] for p in fc.to_payload()["forecast"])
print("every band contains its point:", band_ok)
for p in fc.to_payload()["forecast"]:
    print(f"   {p['month']}  {p['value']:>12,.0f}  [{p['lower']:>12,.0f} .. {p['upper']:>12,.0f}]")
