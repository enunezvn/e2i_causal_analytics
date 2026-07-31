#!/usr/bin/env python3
"""Offline MIN_ACTIVE_CONFIDENCE retune proposals from labeled routing telemetry (#1341 Phase 3).

Reads labeled ``classification_logs`` rows (Phase-1 labeler output) and compiles
a *proposal* artifact: for each candidate active-mode floor, the engagement
flips vs the live floor and their judged accuracy. Proposals only — this script
NEVER mutates routing config (RouterNode.MIN_ACTIVE_CONFIDENCE stays whatever it
is). A human reviews the artifact and, if warranted, opens a PR. This is the
human-gated "iterate toward optimal" step; the stage-3-LLM DSPy recipient half
of Phase 3 is VOID (its #1337 Step-0 contingency failed — no stage-3 layer).

Why a MANUAL script (not the nightly task): proposals are a deliberate,
human-review-cadence action against ACCUMULATED labels — emitting them nightly
would be noise, and the nightly task must stay a lean, bounded labeler. Phase 2
(standing telemetry) rides the nightly task; Phase 3 (retune proposals) is run
on demand when someone is considering a threshold change.

Read-only; SELECT-only DB access. Run from repo root on the droplet with the
real .env::

    SUPABASE_URL=http://localhost:54321 .venv/bin/python \\
        scripts/routing/propose_threshold_retune.py \\
        [--lookback-days 30] [--candidates 0.4,0.45,0.55,0.6] \\
        [--min-evidence 20] [--out artifacts/routing/threshold_proposals.json]

Exits 0 always (advisory). Prints a human-readable summary + writes the JSON
artifact a reviewer can attach to a config-change PR.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.tasks.routing_metrics import (  # noqa: E402
    DEFAULT_ACTIVE_FLOOR,
    compute_run_metrics,
    compute_threshold_proposals,
)


async def _fetch_rows(lookback_days: int, limit: int) -> List[dict]:
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.classification_log import ClassificationLogRepository

    client = await get_async_supabase_client()
    if client is None:
        print("no supabase client — cannot fetch labels", file=sys.stderr)
        return []
    repo = ClassificationLogRepository(client)
    return await repo.fetch_for_metrics(lookback_days=lookback_days, limit=limit)


def _parse_candidates(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback-days", type=int, default=30)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--current-floor", type=float, default=DEFAULT_ACTIVE_FLOOR)
    ap.add_argument("--candidates", type=str, default=None, help="comma-separated floors")
    ap.add_argument("--min-evidence", type=int, default=20)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "routing" / "threshold_proposals.json",
    )
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    rows = asyncio.run(_fetch_rows(args.lookback_days, args.limit))

    metrics = compute_run_metrics(rows, active_floor=args.current_floor)
    proposals = compute_threshold_proposals(
        rows,
        current_floor=args.current_floor,
        candidates=_parse_candidates(args.candidates),
        min_evidence=args.min_evidence,
    )
    artifact = {
        "lookback_days": args.lookback_days,
        "rows_scanned": len(rows),
        "window_metrics": metrics,
        "proposals": proposals,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2))

    # Human-readable summary.
    print(f"\nRouting threshold-retune proposals  ({len(rows)} rows, {args.lookback_days}d window)")
    print(f"  labeled={metrics['labeled']}  overall_accuracy={metrics['overall_accuracy_pct']}%")
    print(
        f"  current floor={proposals['current_floor']}  "
        f"baseline engaged={proposals['baseline_engaged_n']} "
        f"@ {proposals['baseline_accuracy_pct']}%"
    )
    for p in proposals["candidates"]:
        print(
            f"  floor {p['candidate_floor']:.2f} ({p['direction']:5s}): "
            f"engaged={p['engaged_n']} @ {p['engaged_accuracy_pct']}% "
            f"(Δacc {p['accuracy_delta_pct']}), flips={p['labeled_flips']} "
            f"[{p['flips_judged_correct']}✓/{p['flips_judged_incorrect']}✗]"
        )
    rec = proposals["recommended_floor"]
    if rec is None:
        print("  → no recommendation (insufficient labeled evidence or no accuracy gain)")
    else:
        print(f"  → RECOMMENDED floor: {rec}  (PROPOSAL ONLY — human opens the PR)")
    print(f"\nartifact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
