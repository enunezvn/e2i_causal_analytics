"""Recompute ``layer_3.declared_safe_immunity_applied`` in a committed panel artifact.

Verifier MED-3 (PR #2226): the field used to read the node's POST-joint severity,
which the node never leaves at ``high`` on a declared-safe feature, so it could
never fire — the committed ``panel_layer4_fake/panel.json`` served "immunity
applied 0" against 9 records that were pre-joint ``high`` and declared safe.

This recomputes the field from what the run already recorded — the same rule the
panel now applies at build time (``src/causal_engine/feature_role_panel.py``):

    ran AND severity_pre_joint_check == "high" AND layer_1.declared_safe AND NOT leak_verdict

No layer is re-run (the fake-LM run is the measurement; ``built_at`` and every
other number are untouched); ``summary.md`` is re-rendered with the script's own
writer so the two files stay one artifact.

    python docs/demos/results/2026-09-22_lane_e_feature_role_voters/regenerate_immunity_field.py \
        docs/demos/results/2026-09-22_lane_e_feature_role_voters/panel_layer4_fake
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def recompute(payload: dict) -> tuple[int, int]:
    before = 0
    after = 0
    for rec in payload["records"].values():
        l3 = rec["layer_3"]
        before += int(bool(l3.get("declared_safe_immunity_applied")))
        applied = bool(
            l3.get("ran")
            and l3.get("severity_pre_joint_check") == "high"
            and rec["layer_1"].get("declared_safe")
            and not rec["leak_verdict"]
        )
        l3["declared_safe_immunity_applied"] = applied
        after += int(applied)
    payload["layer_activity"]["layer_3"]["declared_safe_immunity_applied"] = after
    return before, after


def main(argv: list[str]) -> int:
    out = Path(argv[1])
    panel_path = out / "panel.json"
    summary_path = out / "summary.md"
    payload = json.loads(panel_path.read_text())
    before, after = recompute(payload)
    panel_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # Re-render summary.md with the measurement script's writer, keeping the
    # run's own LM label and cost note (line 5 of the existing summary).
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.measure_feature_role_panel import _write_summary

    head = summary_path.read_text().splitlines()[4]
    m = re.fullmatch(r"Layer 4 LM: \*\*(.+?)\*\*\. (.*)", head)
    if m is None:
        raise SystemExit(f"unexpected summary header: {head!r}")
    _write_summary(summary_path, payload, lm_label=m.group(1), cost_note=m.group(2))
    print(
        f"declared_safe_immunity_applied: {before} -> {after} (records: {len(payload['records'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
