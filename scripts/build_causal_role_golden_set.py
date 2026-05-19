"""Build the synthetic golden-set fixture (plan §3.5 CLI).

Invocation::

    python scripts/build_causal_role_golden_set.py \\
        [--out tests/fixtures/causal_role_golden_set_synthetic.json]

Idempotent: re-running with no scenario changes produces byte-identical
output (the fixture is sorted/deterministic by construction).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.causal_role_dgp.golden_set import build_golden_set  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set_synthetic.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output JSON path (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    golden = build_golden_set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")

    family_a = sum(1 for e in golden["entries"] if not e["treatment_explicit"])
    family_b = sum(1 for e in golden["entries"] if e["treatment_explicit"])
    logger.info(
        "wrote golden set: %s (Family A=%d cohort-only-gated, Family B=%d informational, total=%d)",
        args.out,
        family_a,
        family_b,
        len(golden["entries"]),
    )


if __name__ == "__main__":
    main()
