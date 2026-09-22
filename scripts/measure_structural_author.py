"""Author once on the blind golden briefs, score once against the golden labels.

Lane B of the real-data causal estimation program (spec §3 Lane B item 3):
run the structural author (``src.data.kg.structural_author``) on the 91
label-free briefs projected from ``tests/fixtures/causal_role_golden_set.json``
(the committed CSU blind briefs are that projection for one cohort), derive the
role with ``extract_role``, and score against ``ground_truth_role`` with
``src.data.kg.structural_author_scoring``: per-role precision / recall and the
missed-leak rate. **Gate: zero missed leaks** — else the report is the
deliverable (exit 2).

``--lm`` selects the language model:

* ``fake`` (default) — a ``DummyLM`` with scripted answers, no provider call.
  ``--fake-source replay`` replays the committed CSU blind authored edges
  (``tests/fixtures/causal_role_csu_blind_authored_edges.json``) through the
  full author pipeline (parse → extract_role → grade → stamp); briefs of the
  other cohorts get a plain confounder fragment, clearly a stand-in. Provider
  keys are blanked in-process so nothing paid can happen by accident.
* ``real`` — the paid run. Refuses unless ``--i-accept-cost`` is given and
  prints the estimate first (an OWNER decision, spec §3 Lane B item 3).

The Lane E feature-role panel is NOT an input here, by construction: the 91
golden briefs are literature-derived fixtures (ConcertAI CSU / PNH / BC
cohorts that exist as label sets, not as frames), so the four voters have no
data to run on and no panel can exist for them. The benchmark measures the
author on the label-free brief alone (``feature_role_panel`` = "no
feature-role panel record for this feature"), which is the brief every real
feature gets when its panel record is absent; the cohort runs
(``scripts/author_cohort_dag.py``) are where the panel is required.

Citations are verified through ``CitationResolver`` only on the real run
(``--resolver live``); the fake run uses an offline resolver that records every
citation as unverifiable, so every fake-run edge grades ``unsupported``.

Usage::

    python -m scripts.measure_structural_author --lm fake --out docs/demos/results/<dir>/measure_fake
    python -m scripts.measure_structural_author --lm real --i-accept-cost --resolver live \\
        --out docs/demos/results/<dir>/measure_real
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from src.ml.causal_role_dgp.extractor import extract_role  # noqa: E402

logger = logging.getLogger("measure_structural_author")

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set.json"
DEFAULT_CSU_EDGES = (
    PROJECT_ROOT / "tests" / "fixtures" / "causal_role_csu_blind_authored_edges.json"
)
CSU_COHORT = "CSU_remibrutinib"

_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

#: ASSUMED list prices (USD per million tokens) for the estimate the owner reads
#: before a real run. They are NOT the platform's contracted rates: pass
#: ``--usd-per-mtok-in`` / ``--usd-per-mtok-out`` with the current list price of
#: ``DSPY_LM_MODEL``. The token counts are measured from the real prompt.
DEFAULT_USD_PER_MTOK_IN = 2.0
DEFAULT_USD_PER_MTOK_OUT = 8.0
#: Output-token allowance per brief (reasoning + edges + rationales + names).
ASSUMED_OUTPUT_TOKENS_PER_BRIEF = 900
CHARS_PER_TOKEN = 4.0

# Treatment / outcome labels per golden cohort (the cohort metadata in the
# golden set names the brand and the target; the brief's dataset_context
# carries them verbatim — these are the short labels for the T / Y nodes).
COHORT_LABELS: dict[str, tuple[str, str]] = {
    CSU_COHORT: ("remibrutinib", "UAS7 reduction at 180 days"),
    "PNH_fabhalta": ("iptacopan", "hemoglobin response at 180 days"),
    "BC_kisqali": ("ribociclib", "progression-free survival / response"),
}


class OfflineResolver:
    """Records every citation as unverifiable: no network on a fake run."""

    def verify_citation(self, identifier, *, identifier_kind, subject_name, object_name):
        from src.data.kg.types import CitationVerdict

        return CitationVerdict(
            identifier=identifier,
            identifier_kind=identifier_kind,
            abstract_resolved=False,
            error="offline resolver (fake run): citation not checked",
        )


def _fake_answer_for(brief: dict[str, str], replay: dict[str, dict[str, Any]]) -> dict[str, Any]:
    feature = brief["feature_name"]
    entry = replay.get(feature) if brief["cohort"] == CSU_COHORT else None
    expected = "confounder"
    if entry is not None:
        edges = [list(e) for e in entry["edges"]]
        ambiguous = bool(entry.get("ambiguous", False))
        reasoning = "fake LM (replay of the committed CSU blind authored edges)"
        # The replayed author's expectation is what its own edges derive: a
        # replay must not fabricate a self-contradiction the author never made.
        try:
            expected = extract_role(
                entry["feature_node"],
                entry["treatment_node"],
                entry["outcome_node"],
                nx.DiGraph([tuple(e) for e in entry["edges"]]),
            )
        except ValueError:
            expected = "confounder"
    else:
        edges = [[feature, "T"], [feature, "Y"], ["T", "Y"]]
        ambiguous = False
        reasoning = "fake LM (stand-in confounder fragment; not an authored claim)"
    return {
        "reasoning": reasoning,
        "edges": json.dumps(edges),
        "edge_rationales": "[]",
        "entity_names": "{}",
        "expected_role": expected,
        "ambiguous": "true" if ambiguous else "false",
    }


def _load_replay(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {e["feature_name"]: e for e in payload["entries"]}


def estimate_cost(
    briefs: list[Any],
    *,
    usd_in: float,
    usd_out: float,
) -> dict[str, Any]:
    """Measure the real prompt for every brief (no LM call) and price it."""
    import dspy

    from src.data.kg.structural_author import StructuralAttestationSignature

    adapter = dspy.ChatAdapter()
    total_chars = 0
    per_brief: list[int] = []
    for brief in briefs:
        messages = adapter.format(
            StructuralAttestationSignature,
            demos=[],
            inputs={
                "feature_name": brief.feature_name,
                "derivation_pseudocode": brief.derivation_pseudocode,
                "dataset_context": brief.dataset_context,
                "treatment_label": brief.treatment_label,
                "outcome_label": brief.outcome_label,
                "feature_role_panel": brief.panel_text(),
            },
        )
        chars = sum(len(str(m.get("content", ""))) for m in messages)
        per_brief.append(chars)
        total_chars += chars
    in_tokens = int(total_chars / CHARS_PER_TOKEN)
    out_tokens = ASSUMED_OUTPUT_TOKENS_PER_BRIEF * len(briefs)
    return {
        "n_briefs": len(briefs),
        "prompt_chars_total": total_chars,
        "prompt_chars_per_brief_mean": (total_chars / len(briefs)) if briefs else 0,
        "prompt_tokens_estimate": in_tokens,
        "output_tokens_estimate": out_tokens,
        "chars_per_token_assumed": CHARS_PER_TOKEN,
        "output_tokens_per_brief_assumed": ASSUMED_OUTPUT_TOKENS_PER_BRIEF,
        "usd_per_mtok_in_assumed": usd_in,
        "usd_per_mtok_out_assumed": usd_out,
        "usd_estimate": in_tokens / 1e6 * usd_in + out_tokens / 1e6 * usd_out,
        "model": os.environ.get("DSPY_LM_MODEL") or "(get_default_dspy_model)",
        "note": (
            "token counts measured from the real ChatAdapter prompt at 4 chars/token; "
            "rates are ASSUMED unless passed explicitly"
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument(
        "--cohort", default="all", help="all | CSU_remibrutinib | PNH_fabhalta | BC_kisqali"
    )
    p.add_argument("--lm", choices=("fake", "real"), default="fake")
    p.add_argument("--fake-source", choices=("replay", "confounder"), default="replay")
    p.add_argument("--replay-edges", type=Path, default=DEFAULT_CSU_EDGES)
    p.add_argument("--resolver", choices=("offline", "live"), default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--i-accept-cost", action="store_true", help="Required with --lm real.")
    p.add_argument("--usd-per-mtok-in", type=float, default=DEFAULT_USD_PER_MTOK_IN)
    p.add_argument("--usd-per-mtok-out", type=float, default=DEFAULT_USD_PER_MTOK_OUT)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    from src.data.kg.structural_author import author_feature, build_brief, tree_identity
    from src.data.kg.structural_author_scoring import (
        golden_briefs,
        load_golden_entries,
        score_roles,
    )

    entries = load_golden_entries(args.golden)
    cohort = None if args.cohort == "all" else args.cohort
    if cohort is not None and cohort not in COHORT_LABELS:
        logger.error("unknown cohort %r (known: %s)", cohort, sorted(COHORT_LABELS))
        return 4
    entries = [e for e in entries if cohort is None or e["cohort"] == cohort]
    raw_briefs = golden_briefs(entries, cohort=cohort)
    briefs = [
        build_brief(
            b["feature_name"],
            derivation_pseudocode=b["derivation_pseudocode"],
            dataset_context=b["dataset_context"],
            treatment_label=COHORT_LABELS[b["cohort"]][0],
            outcome_label=COHORT_LABELS[b["cohort"]][1],
        )
        for b in raw_briefs
    ]
    logger.info("%d briefs (%s)", len(briefs), args.cohort)

    cost = estimate_cost(briefs, usd_in=args.usd_per_mtok_in, usd_out=args.usd_per_mtok_out)
    (args.out / "cost_estimate.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    logger.info(
        "cost estimate: %d prompt tokens + %d output tokens ≈ USD %.2f at assumed rates "
        "(%.2f in / %.2f out per Mtok)",
        cost["prompt_tokens_estimate"],
        cost["output_tokens_estimate"],
        cost["usd_estimate"],
        args.usd_per_mtok_in,
        args.usd_per_mtok_out,
    )

    resolver_mode = args.resolver or ("live" if args.lm == "real" else "offline")
    if args.lm == "real":
        if not args.i_accept_cost:
            logger.error(
                "--lm real is a paid run (owner decision). Re-run with --i-accept-cost "
                "after reading %s",
                args.out / "cost_estimate.json",
            )
            return 3
        from src.optimization.dspy_lm import ensure_dspy_configured

        if not ensure_dspy_configured():
            logger.error("no DSPy LM could be configured (missing provider key?)")
            return 3
        lm_label = os.environ.get("DSPY_LM_MODEL") or "configured"
        per_feature_lm = None
        replay: dict[str, dict[str, Any]] = {}
    else:
        for var in _PROVIDER_KEYS:
            os.environ.pop(var, None)
        lm_label = "fake"
        per_feature_lm = "dummy"
        replay = _load_replay(args.replay_edges) if args.fake_source == "replay" else {}
        if resolver_mode == "live":
            logger.error("--resolver live is only allowed with --lm real")
            return 4

    if resolver_mode == "live":
        from src.data.kg.citation_resolver import CitationResolver

        resolver: Any = CitationResolver()
    else:
        resolver = OfflineResolver()

    authored: list[dict[str, Any]] = []
    predicted: dict[str, Optional[str]] = {}
    for raw, brief in zip(raw_briefs, briefs, strict=True):
        if per_feature_lm == "dummy":
            from dspy.utils.dummies import DummyLM

            lm = DummyLM([_fake_answer_for(raw, replay)])
            rec = author_feature(brief, resolver=resolver, lm=lm)
        else:
            rec = author_feature(brief, resolver=resolver)
        authored.append({"cohort": raw["cohort"], **rec.to_dict()})
        predicted[f"{raw['cohort']}/{raw['feature_name']}"] = rec.derived_role

    report = score_roles(predicted, entries)
    meta = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "lm": lm_label,
        "fake_source": args.fake_source if args.lm == "fake" else None,
        "resolver": resolver_mode,
        "cohort": args.cohort,
        "golden": str(args.golden),
        "n_briefs": len(briefs),
        "tree": tree_identity(PROJECT_ROOT),
    }
    (args.out / "authored.json").write_text(
        json.dumps({"meta": meta, "records": authored}, indent=2), encoding="utf-8"
    )
    (args.out / "score.json").write_text(
        json.dumps({"meta": meta, **report.to_dict()}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Structural author measurement",
        "",
        f"- lm: `{lm_label}`"
        + (f" (fake source: {args.fake_source})" if args.lm == "fake" else ""),
        f"- resolver: `{resolver_mode}`; cohort: `{args.cohort}`; briefs: {len(briefs)}",
        f"- measured_at: {meta['measured_at']}",
        f"- tree: commit {meta['tree']['commit']} "
        f"(dirty src/scripts/tests: {meta['tree']['dirty_src_scripts_tests']})",
        "",
        "## Score",
        "",
        *[f"    {ln}" for ln in report.summary_lines()],
        "",
        "## Missed leaks",
        "",
        *(
            [
                f"- {m['cohort']}/{m['feature_name']}: golden {m['ground_truth_role']}, derived {m['derived_role']}"
                for m in report.missed_leaks
            ]
            or ["- none"]
        ),
        "",
        "## Routed to review",
        "",
        *(
            [
                f"- {r['cohort']}/{r['feature_name']} (golden {r['ground_truth_role']})"
                for r in report.review
            ]
            or ["- none"]
        ),
        "",
        "## Cost estimate for the real run",
        "",
        f"- prompt tokens {cost['prompt_tokens_estimate']}, output tokens {cost['output_tokens_estimate']} "
        f"→ USD {cost['usd_estimate']:.2f} at ASSUMED {args.usd_per_mtok_in}/{args.usd_per_mtok_out} per Mtok",
    ]
    if args.lm == "fake":
        lines += [
            "",
            "NOTE: a fake-LM run measures the PIPELINE (parse → extract_role → grade → score), "
            "not the author. The replayed CSU edges reproduce the committed validation record; "
            "the other cohorts' stand-in fragments are not authored claims.",
        ]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ln in report.summary_lines():
        logger.info(ln)
    return 0 if report.gate_passed else 2


if __name__ == "__main__":
    sys.exit(main())
