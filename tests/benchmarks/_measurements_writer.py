"""Per-benchmark measurements emission helper (issue #403).

Each performance benchmark calls ``write_measurements`` after computing
its median/p95 so the raw data is captured in a stable JSON artifact at
``test-results/measurements-<box>.json`` (and additively at
``test-results/measurements.json`` as a flat array of all boxes from the
same pytest invocation).

Why we need this — junit XML at ``test-results/benchmark-*.xml`` only
captures test-level wall-clock (setup + benchmark body + teardown); the
per-iteration ``timings_ms`` + computed ``median_ms`` / ``p95_ms`` are
printed to stderr and captured by pytest's default capture — they do NOT
appear in ``gh run view --log``. Issue #403 needs ≥3 CI runs of the
benchmarks with measurements extractable from artifacts so the baseline
can be re-blessed from the median across runs.

The writer is deliberately small and dependency-free (stdlib json + os
+ pathlib + datetime). It does not raise on filesystem failure — a
measurement failure must not mask the underlying benchmark result; we
print the error to stderr and continue.

File shape per box::

    {
      "box": "cascade_5hop_bfs",
      "test": "test_cascade_5hop_bfs_latency_against_baseline",
      "statistic": "median",                  # which scalar `value_ms` represents
      "value_ms": <float>,                    # the primary scalar for THIS box
      "runs": [<list of float timings_ms>],   # raw per-iteration timings
      "median_ms": <float>,                   # convenience: median of `runs`
      "p95_ms": <float | None>,               # convenience: p95 of `runs`
      "ci_run_id": "<env GITHUB_RUN_ID, or empty>",
      "ci_run_attempt": "<env GITHUB_RUN_ATTEMPT, or empty>",
      "ci_sha": "<env GITHUB_SHA, or empty>",
      "ci_ref": "<env GITHUB_REF, or empty>",
      "emitted_at": "<ISO-8601 UTC timestamp>"
    }

``statistic`` + ``value_ms`` are the consumer-friendly contract: the
re-bless script reads ``value_ms`` and uses ``statistic`` to label it.
``median_ms`` + ``p95_ms`` are retained as convenience aggregates over
``runs`` so the file remains self-describing (a consumer that wants to
re-derive any percentile can do so from ``runs[]`` alone). For a box
where the scalar IS the median, ``statistic="median"`` and
``value_ms == median_ms``; for a box where the scalar is the p95,
``statistic="p95"`` and ``value_ms == p95_ms`` — the two scalars are
NOT swapped silently (codex iter-2 M2 closure).

The combined ``measurements.json`` file is a JSON array of the per-box
records, appended in-process so a single ``scripts/run_benchmarks.sh``
invocation that runs all three boxes ends up with one combined artifact
covering all of them; this is convenient for the issue-#403 re-bless
script (single artifact per CI run rather than three separate files).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Resolve the test-results directory relative to the repo root. ``cwd``
# during pytest invocation is the repo root (see scripts/run_benchmarks.sh
# `cd "$REPO_ROOT"`), so a relative path lands in the artifact-upload
# surface the workflow YAML expects (`path: test-results/`).
_ARTIFACT_DIR = Path("test-results")
_COMBINED_FILE = _ARTIFACT_DIR / "measurements.json"


def _ci_env() -> Dict[str, str]:
    """Snapshot CI-relevant environment variables (empty string if absent).

    Captured so a downstream re-bless script can match an artifact to its
    workflow run without having to cross-reference filenames.
    """
    return {
        "ci_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "ci_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", ""),
        "ci_sha": os.getenv("GITHUB_SHA", ""),
        "ci_ref": os.getenv("GITHUB_REF", ""),
    }


def write_measurements(
    *,
    box: str,
    test: str,
    runs: Sequence[float],
    median_ms: float,
    p95_ms: Optional[float] = None,
    statistic: str = "median",
    value_ms: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a per-box measurement record to disk.

    Args:
        box: stable identifier for the baseline this box re-blesses
            (e.g. ``"cascade_5hop_bfs"``, ``"hybrid_retriever_search_p95"``).
        test: pytest test ID (including parametrization), useful for
            tracing back to the source test from a failure investigation.
        runs: raw per-iteration timings in ms; the consumer can re-derive
            any percentile from this list.
        median_ms: median of ``runs`` (convenience aggregate).
        p95_ms: p95 of ``runs`` (convenience aggregate, ``None`` if not
            computed by the caller).
        statistic: which scalar ``value_ms`` represents — "median" or
            "p95" are the conventional values. The re-bless script
            uses (box, statistic, value_ms) as its primary tuple.
        value_ms: the primary scalar for THIS box. If ``None``, defaults
            to ``median_ms`` (the common case). For a box whose scalar
            is the p95 (e.g. ``hybrid_retriever_search_p95``), pass
            ``statistic="p95", value_ms=p95_ms`` so the artifact
            honestly labels its primary scalar (codex iter-2 M2 closure
            — prevents silent p50-as-p95 mis-attribution).
        extra: optional per-box metadata (e.g. ``slice_n`` for BM25
            parametrized runs).

    Writes TWO files:
      1. ``test-results/measurements-<box>.json`` (single-record file —
         useful when the workflow runs a single box at a time).
      2. ``test-results/measurements.json`` (append-style array — every
         box from the same pytest invocation accumulates here).

    Both files are JSON; (1) is a single object and (2) is an array.

    Filesystem failures are logged to stderr but do NOT raise — a
    measurements emission failure must not mask the underlying benchmark
    result or break test isolation.
    """
    primary_scalar = float(value_ms) if value_ms is not None else float(median_ms)
    record: Dict[str, Any] = {
        "box": box,
        "test": test,
        "statistic": statistic,
        "value_ms": primary_scalar,
        "runs": [float(t) for t in runs],
        "median_ms": float(median_ms),
        "p95_ms": float(p95_ms) if p95_ms is not None else None,
        "emitted_at": datetime.now(timezone.utc).isoformat(),
    }
    record.update(_ci_env())
    if extra:
        record["extra"] = extra

    try:
        _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"[measurements_writer] could not create {_ARTIFACT_DIR}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return

    # Per-box single-record file. Last writer wins on box-key collision
    # (e.g., parametrized BM25 re-uses the writer per slice; each slice
    # gets its own box name so there is no collision in practice).
    single_file = _ARTIFACT_DIR / f"measurements-{box}.json"
    try:
        with single_file.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        print(
            f"[measurements_writer] could not write {single_file}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    # Combined append-style file. Reads + appends + rewrites; we accept
    # the read-modify-write cost because the dataset is tiny (one record
    # per box per pytest invocation, ≤6 boxes per invocation).
    existing: List[Dict[str, Any]] = []
    if _COMBINED_FILE.exists():
        try:
            with _COMBINED_FILE.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, list):
                existing = [r for r in loaded if isinstance(r, dict)]
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[measurements_writer] could not read existing "
                f"{_COMBINED_FILE} ({exc}); will overwrite with fresh array.",
                file=sys.stderr,
                flush=True,
            )
            existing = []

    existing.append(record)
    try:
        with _COMBINED_FILE.open("w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        print(
            f"[measurements_writer] could not write {_COMBINED_FILE}: {exc}",
            file=sys.stderr,
            flush=True,
        )
