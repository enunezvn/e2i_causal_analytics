#!/usr/bin/env python
"""Build the SYNTHETIC backing of ``public.csu_escalation_causal`` (Lane C,
spec 2026-09-22 §3C.2): a planted-truth CSU escalation cohort in the table's
exact column contract, plus its ground-truth sidecar.

Remibrutinib is absent from every real claims drop, so until the post-launch
refresh the registry entry ``csu_escalation_causal`` is backed by these rows
(every one ``is_synthetic = true`` -- the real-mode provenance filter returns
none of them). The frame comes from
``src.ml.synthetic.generators.csu_escalation_causal`` with a FIXED seed, so the
parquet is reproducible and byte-stable; the treatment column is derived from
``index_biologic_brand`` through the mart converter's own
``select_csu_escalation_contrast`` and asserted equal to the DGP's arm, so the
synthetic rows and the future real export agree by construction.

Writes (parquet-only; loading the table is a separate, owner-GO prod write):
  <out-dir>/csu_escalation_causal_synthetic.parquet
  <out-dir>/ground_truth.json          (one PlantedTruth per outcome)
  <out-dir>/build_summary.json         (n, arm split, outcome means, attrition)

Usage:
  python -m scripts.build_csu_escalation_synthetic_cohort --out-dir data/rwd/synthetic_CSU/csu_escalation_causal
  python -m scripts.build_csu_escalation_synthetic_cohort --n 3000 --seed 20260922 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CSU_ESCALATION_TREATMENT_COL,
    select_csu_escalation_contrast,
)
from src.ml.synthetic.generators.csu_escalation_causal import (  # noqa: E402
    CONTRACT_COLUMNS,
    DEFAULT_N,
    DEFAULT_SEED,
    OUTCOMES,
    TREATMENT,
    generate_csu_escalation_cohort,
)

logger = logging.getLogger("build_csu_escalation_synthetic_cohort")

PARQUET_NAME = "csu_escalation_causal_synthetic.parquet"
GROUND_TRUTH_NAME = "ground_truth.json"
SUMMARY_NAME = "build_summary.json"


def build(n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate the cohort and cross-check the treatment through the mart
    converter's contrast selector. Returns ``(frame, summary)``."""
    frame, truths = generate_csu_escalation_cohort(n=n, seed=seed)
    assert list(frame.columns) == list(CONTRACT_COLUMNS)
    assert TREATMENT == CSU_ESCALATION_TREATMENT_COL

    # The converter derives the treatment from the brand label; it must equal
    # the DGP's arm on every row (the two paths agree by construction).
    derived, attrition = select_csu_escalation_contrast(frame.drop(columns=[TREATMENT]))
    if len(derived) != len(frame):
        raise AssertionError(
            f"contrast selector kept {len(derived)}/{len(frame)} rows; attrition={attrition}"
        )
    mismatch = int((derived[TREATMENT].to_numpy() != frame[TREATMENT].to_numpy()).sum())
    if mismatch:
        raise AssertionError(
            f"{mismatch} row(s) where the brand-derived treatment differs from the DGP arm"
        )
    if not bool(frame["is_synthetic"].all()):
        raise AssertionError("every synthetic backing row must carry is_synthetic=True")

    summary: dict[str, Any] = {
        "n": int(len(frame)),
        "seed": seed,
        "columns": len(frame.columns),
        "arm_split": {
            str(k): int(v) for k, v in frame[TREATMENT].value_counts().sort_index().items()
        },
        "brand_split": {
            str(k): int(v) for k, v in frame["index_biologic_brand"].value_counts().items()
        },
        "outcome_means_by_arm": {
            o: {
                str(arm): round(float(frame.loc[frame[TREATMENT] == arm, o].mean()), 4)
                for arm in (0, 1)
            }
            for o in OUTCOMES
        },
        "attrition": [[step, int(count)] for step, count in attrition],
        "ground_truth": {o: t.to_dict() for o, t in truths.items()},
    }
    return frame, summary


def write(frame: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / PARQUET_NAME
    frame.to_parquet(parquet, index=False)
    truth_path = out_dir / GROUND_TRUTH_NAME
    truth_path.write_text(json.dumps(summary["ground_truth"], indent=2, sort_keys=True))
    summary_path = out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return {"parquet": parquet, "ground_truth": truth_path, "summary": summary_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    frame, summary = build(n=args.n, seed=args.seed)
    paths = write(frame, summary, args.out_dir)
    logger.info(
        "wrote %s rows x %s cols -> %s (arms %s; primary true ATE %s, naive %s)",
        summary["n"],
        summary["columns"],
        paths["parquet"],
        summary["arm_split"],
        summary["ground_truth"][OUTCOMES[0]]["true_ate"],
        summary["ground_truth"][OUTCOMES[0]]["naive_diff"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
