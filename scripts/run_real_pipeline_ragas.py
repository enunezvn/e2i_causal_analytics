#!/usr/bin/env python3
"""Real-pipeline RAGAS gate — judge recorded replays and gate the result (#1485).

What this measures, and why it is not the other RAGAS job
----------------------------------------------------------
``scripts/run_ragas_eval.py`` scores a STATIC FIXTURE: it calls
``run_evaluation()`` with no ``rag_pipeline``, so the judge sees the golden
set's hardcoded answers over ``retrieved_contexts`` byte-identical to the
reference ``contexts``. That job is a judge-drift sentinel on frozen input and
is named accordingly ("RAGAS Fixture Regression"); it cannot see production
quality. This script is the honest counterpart — it judges what the pipeline
actually generated over what it actually retrieved.

The two halves it wires together already existed; #1489 called it plumbing:

1. ``scripts/replay_golden_set.py --target cognitive --record-out PATH``
   replays the golden questions through ``POST /api/cognitive/rag`` and (since
   #1485) records each real answer and its really-retrieved contexts.
2. ``scripts/run_dspy_lane_ragas_judge.py`` scores exactly that record shape
   with the frozen gpt-4o judge via the production ``RAGASEvaluator``. It is
   invoked UNCHANGED — this driver never reimplements judging.

Metrics: faithfulness and answer_relevancy only. context_precision and
context_recall need a ground-truth reference the replay deliberately does not
fabricate, so they are dropped rather than reported as 1.0-by-construction —
mirroring how the DSPy-lane judge already omits them.

Cadence (#504, #1489 step 3): n≈10-15, on demand. NEVER per-PR — the CI
OpenAI key's throughput was the binding constraint that made the fixture eval
manual-only, and this path costs more (retrieval + generation + judging).

Usage::

    # 1. record real replays (~15-18s per question)
    .venv/bin/python scripts/replay_golden_set.py --limit 12 \\
        --record-out /tmp/goldset_records.json

    # 2. judge them inside the prod container (ragas + key live there)
    .venv/bin/python scripts/run_real_pipeline_ragas.py \\
        --records /tmp/goldset_records.json --fail-on-threshold

``--judge-mode local`` runs the same judge script in this interpreter instead,
for a box that has ragas and OPENAI_API_KEY locally.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.ragas_persistence import (  # noqa: E402
    RagasPersistenceError,
    judged_turns,
    persist_judged_turns,
)
from src.rag.real_pipeline_eval import (  # noqa: E402
    MIN_REAL_PIPELINE_SAMPLES,
    MIN_RETRIEVAL_HIT_RATE,
    REAL_PIPELINE_METRICS,
    REAL_PIPELINE_THRESHOLDS,
    build_samples_from_replay,
    check_run_gates,
    summarize_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_real_pipeline_ragas")

# The judge script is reused verbatim; this driver only feeds and reads it.
JUDGE_SCRIPT = REPO_ROOT / "scripts" / "run_dspy_lane_ragas_judge.py"
RESULTS_BEGIN = "RESULTS_JSON_BEGIN"
RESULTS_END = "RESULTS_JSON_END"

# Hardcoded in RAGASEvaluator._evaluate_with_ragas (src/rag/evaluation.py:1100).
# gpt-4o, not -mini: the mini judge produces spurious context-precision zeros
# on clearly-relevant contexts (#491).
JUDGE_MODEL = "gpt-4o"


class JudgeOutputError(RuntimeError):
    """The judge produced no parseable result block.

    Raised rather than returning an empty dict on purpose: a judge that
    crashed, timed out, or was killed mid-run must BLOCK. An empty block would
    otherwise flow into the gates and could read as "nothing failed".
    """


def parse_judge_output(stdout: str) -> Dict[str, Any]:
    """Extract the judge's RESULTS_JSON block, failing closed on anything else."""
    if RESULTS_BEGIN not in stdout:
        raise JudgeOutputError(
            f"judge emitted no {RESULTS_BEGIN} marker — it crashed or was killed "
            "before scoring; refusing to report a result"
        )
    body = stdout.split(RESULTS_BEGIN, 1)[1]
    if RESULTS_END not in body:
        raise JudgeOutputError(f"judge output truncated: {RESULTS_BEGIN} without {RESULTS_END}")
    payload = body.split(RESULTS_END, 1)[0].strip()
    try:
        block = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise JudgeOutputError(f"judge result block is not valid JSON: {exc}") from exc
    if not isinstance(block, dict):
        raise JudgeOutputError(f"judge result block is {type(block).__name__}, expected object")
    return block


def judge_env_failure(key_present: bool) -> Optional[str]:
    """Refuse to judge in an environment that would fake the numbers.

    With no OPENAI_API_KEY, ``RAGASEvaluator`` takes ``_evaluate_with_fallback``
    (src/rag/evaluation.py:1191) and returns word-overlap heuristics — ordinary
    looking floats that are NOT gpt-4o judgments. The judge's output shape
    cannot distinguish them from real scores, so the only safe place to catch
    this is before the run.
    """
    if not key_present:
        return (
            "no OPENAI_API_KEY in the judging environment — RAGASEvaluator would fall "
            "back to word-overlap heuristics (src/rag/evaluation.py:1191) and emit "
            "plausible-looking scores that are not gpt-4o judgments"
        )
    return None


def load_records(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load replay records, accepting the wrapper or a bare list."""
    with Path(path).open() as fh:
        raw = json.load(fh)
    if isinstance(raw, list):
        records, meta = raw, {}
    elif isinstance(raw, dict):
        records = raw.get("records") or []
        meta = {k: v for k, v in raw.items() if k != "records"}
    else:
        raise ValueError(f"unsupported records file shape: {type(raw).__name__}")
    if not records:
        raise ValueError(f"{path} contains no records")
    return records, meta


def build_report(
    block: Dict[str, Any],
    retrieval: Dict[str, Any],
    thresholds: Dict[str, float],
    passed: bool,
    failures: Sequence[str],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the run report. Only real-path metrics are ever reported."""
    return {
        "kind": "real_pipeline_ragas",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "replay": meta,
        "n_samples": block.get("n_samples"),
        "n_faithfulness": block.get("n_faithfulness"),
        "metrics": {m: block.get(m) for m in REAL_PIPELINE_METRICS},
        "thresholds": dict(thresholds),
        "retrieval": retrieval,
        "passed": passed,
        "failures": list(failures),
        "per_sample": block.get("per_sample"),
    }


def _openai_key_present(mode: str, container: str) -> bool:
    """Probe for a non-empty key WITHOUT ever reading its value."""
    if mode == "local":
        return bool(os.environ.get("OPENAI_API_KEY"))
    probe = subprocess.run(  # noqa: S603
        [
            "docker",
            "exec",
            container,
            "python",
            "-c",
            "import os,sys; sys.exit(0 if os.environ.get('OPENAI_API_KEY') else 1)",
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def run_judge(
    samples: List[Dict[str, Any]],
    mode: str,
    container: str,
    model_label: str,
    timeout: int,
) -> Dict[str, Any]:
    """Feed samples to the UNCHANGED judge script and parse its result block."""
    payload = json.dumps({"model": model_label, "samples": samples})
    source = JUDGE_SCRIPT.read_text()
    if mode == "local":
        cmd = [sys.executable, "-c", source]
        cwd: Optional[str] = str(REPO_ROOT)
    else:
        cmd = ["docker", "exec", "-i", container, "python", "-c", source]
        cwd = None

    logger.info("judging %d samples via %s (judge=%s)", len(samples), mode, JUDGE_SCRIPT.name)
    proc = subprocess.run(  # noqa: S603
        cmd, input=payload, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-800:]
        raise JudgeOutputError(f"judge exited {proc.returncode}; stderr tail:\n{tail}")
    return parse_judge_output(proc.stdout)


async def _build_writers(persist_signals: bool) -> Tuple[Any, Any]:
    """The #1487 writers, both of which shipped with no caller.

    One async seam rather than two sync factories because
    ``get_async_supabase_client`` is a COROUTINE and both writers share the
    client it returns. Imports are local: this script is also read by
    ``--dry-run``-style invocations that never persist, and the feedback
    learner package is heavy.

    Returns:
        ``(evaluation_results_repository, rubric_node_or_None)``. The rubric
        node is built only for ``--persist-signals`` — it JUDGES, one Anthropic
        call per sample, on top of the gpt-4o judging already done.
    """
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.evaluation_results import get_evaluation_results_repository

    client = await get_async_supabase_client()
    eval_repo = get_evaluation_results_repository(supabase_client=client)

    rubric_node = None
    if persist_signals:
        from src.agents.feedback_learner.nodes.rubric_node import RubricNode

        rubric_node = RubricNode(db_client=client)
    return eval_repo, rubric_node


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(report, fh, indent=2)


def persist_run(
    block: Dict[str, Any],
    records: Sequence[Dict[str, Any]],
    persist_signals: bool,
) -> Dict[str, Any]:
    """Write this run's judged samples into the self-improvement schema.

    The join is done BEFORE any writer is built, so a provenance failure
    (an unjoinable sample, a heuristic-contaminated one) costs no DB
    connection and writes nothing.

    Persistence deliberately survives a FAILING verdict — a regression has to
    reach the trend view or the view cannot show one. It does NOT survive a
    block that cannot be reconciled against its own rows; ``judged_turns``
    enforces that for every caller, not just this one.

    Raises:
        RagasPersistenceError: The block is inconsistent, or any turn could
            not be persisted faithfully.
    """
    turns = judged_turns(block, records, judge_model=JUDGE_MODEL)

    async def _run() -> Dict[str, Any]:
        eval_repo, rubric_node = await _build_writers(persist_signals)
        return await persist_judged_turns(turns, eval_repo=eval_repo, rubric_node=rubric_node)

    return asyncio.run(_run())


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--records",
        required=True,
        type=Path,
        help="Replay records JSON from replay_golden_set.py --record-out",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("container", "local"),
        default="container",
        help="Where to run the judge (default: container — ragas + key live there)",
    )
    parser.add_argument(
        "--container", default="e2i_api", help="Container for --judge-mode container"
    )
    parser.add_argument("--limit", type=int, default=None, help="Judge only the first N samples")
    parser.add_argument(
        "--faithfulness",
        type=float,
        default=REAL_PIPELINE_THRESHOLDS["faithfulness"],
        help=f"Faithfulness gate (default: {REAL_PIPELINE_THRESHOLDS['faithfulness']})",
    )
    parser.add_argument(
        "--answer-relevancy",
        type=float,
        default=REAL_PIPELINE_THRESHOLDS["answer_relevancy"],
        help=f"Answer-relevancy gate (default: {REAL_PIPELINE_THRESHOLDS['answer_relevancy']})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MIN_REAL_PIPELINE_SAMPLES,
        help=f"Minimum judged samples (default: {MIN_REAL_PIPELINE_SAMPLES})",
    )
    parser.add_argument(
        "--retrieval-floor",
        type=float,
        default=MIN_RETRIEVAL_HIT_RATE,
        help=(
            "Minimum share of replays that must retrieve any context "
            f"(default: {MIN_RETRIEVAL_HIT_RATE}). The metric gates cannot see a "
            "retrieval collapse on their own."
        ),
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Judge subprocess timeout (s)")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON report here")
    parser.add_argument(
        "--fail-on-threshold", action="store_true", help="Exit 1 when the gates block"
    )
    parser.add_argument(
        "--no-persist",
        dest="persist",
        action="store_false",
        help=(
            "Do NOT write the judged samples to evaluation_results. Persistence is "
            "on by default (#1489): the schema exists to hold exactly these numbers, "
            "and a default-off flag is how #1487's writers ended up with no callers."
        ),
    )
    parser.add_argument(
        "--persist-signals",
        action="store_true",
        help=(
            "Also write a learning_signals row per sample carrying ragas_scores + "
            "retrieved_chunks. Opt-in because it runs the RUBRIC judge (one "
            "Anthropic call per sample) on top of the gpt-4o judging already done."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    env_failure = judge_env_failure(_openai_key_present(args.judge_mode, args.container))
    if env_failure:
        logger.error("BLOCKED: %s", env_failure)
        return 1

    records, meta = load_records(args.records)
    if args.limit is not None:
        records = records[: args.limit]

    retrieval = summarize_retrieval(records)
    samples = build_samples_from_replay(records)
    if not samples:
        logger.error(
            "BLOCKED: every replay errored or produced an empty answer (%d records) — "
            "nothing judgeable",
            len(records),
        )
        return 1

    logger.info(
        "records=%d judgeable=%d retrieval_hit=%d/%d errors=%d",
        len(records),
        len(samples),
        retrieval["n_with_contexts"],
        retrieval["n_records"],
        retrieval["n_errors"],
    )

    block = run_judge(
        samples,
        mode=args.judge_mode,
        container=args.container,
        model_label=str(meta.get("target") or "real-pipeline"),
        timeout=args.timeout,
    )

    thresholds = {
        "faithfulness": args.faithfulness,
        "answer_relevancy": args.answer_relevancy,
    }
    passed, failures = check_run_gates(
        block,
        retrieval,
        thresholds=thresholds,
        min_samples=args.min_samples,
        retrieval_floor=args.retrieval_floor,
    )
    report = build_report(block, retrieval, thresholds, passed, failures, meta)

    # Write the report BEFORE touching the database. This file is the durable
    # record of several minutes of gpt-4o judging; the rows are derived from
    # it. A DB outage must cost the rows, never the run.
    if args.output:
        _write_report(args.output, report)
        logger.info("report -> %s", args.output)

    print("-" * 64)
    print("REAL-PIPELINE RAGAS (judge: %s)" % JUDGE_MODEL)
    print("-" * 64)
    print(f"  samples judged     {block.get('n_samples')}")
    print(
        f"  retrieval hit      {retrieval['n_with_contexts']}/{retrieval['n_records']} "
        f"({(retrieval['retrieval_hit_rate'] or 0):.3f})"
    )
    for metric in REAL_PIPELINE_METRICS:
        value = block.get(metric)
        shown = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
        extra = f"  (n={block.get('n_faithfulness')})" if metric == "faithfulness" else ""
        print(f"  {metric:<18} {shown}   threshold {thresholds[metric]}{extra}")
    print("-" * 64)
    if failures:
        print("GATES BLOCKED:")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("GATES PASSED")

    # #1489: land the judged samples in the schema #1487 built for them.
    #
    # Regardless of the VERDICT. Persisting only passing runs would make
    # v_ragas_performance_trends survivorship-biased — a "daily RAGAS metric
    # trends for monitoring" view that can only ever contain runs which
    # already cleared the thresholds is structurally incapable of showing a
    # decline, which is the one thing it exists to show. A faithfulness of
    # 0.12 IS the measurement; the gate's job is to say so loudly and set the
    # exit code, not to keep the number out of the table.
    #
    # Trustworthiness is enforced ROW-WISE instead, which is where it belongs:
    # judged_turns refuses heuristic-contaminated samples, unjoinable
    # provenance and malformed blocks, and a turn with no measured metric is
    # skipped and counted. A judge that crashed never reaches here at all
    # (JudgeOutputError, above).
    persistence_failed = False
    if args.persist:
        try:
            report["persistence"] = persist_run(block, records, args.persist_signals)
        except RagasPersistenceError as exc:
            persistence_failed = True
            report["persistence"] = {**exc.summary, "error": str(exc)}
            print(f"PERSISTENCE FAILED: {exc}")
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            persistence_failed = True
            report["persistence"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"PERSISTENCE FAILED: {type(exc).__name__}: {exc}")
        else:
            written = report["persistence"]
            print(
                f"persisted: {written['evaluation_results_written']} evaluation_results, "
                f"{written['learning_signals_written']} learning_signals"
                + (
                    f", {len(written['skipped_unscored'])} skipped (no measured metric)"
                    if written["skipped_unscored"]
                    else ""
                )
            )
        if args.output:
            _write_report(args.output, report)

    if persistence_failed:
        return 1
    if not passed and args.fail_on_threshold:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
