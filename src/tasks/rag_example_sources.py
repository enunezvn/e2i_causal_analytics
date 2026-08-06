"""One example source for the #1486 RAG-prompt GEPA leg, two backings (#1489 d5).

What this is trying to do
-------------------------
#1486 shipped a nightly GEPA leg that tunes the cognitive-RAG synthesis prompt
against a RAGAS judge. Its feedstock was a single JSON file named by
``DSPY_RAG_RECORDS_PATH`` and produced by a MANUAL
``scripts/replay_golden_set.py --target cognitive --record-out`` run. So the
leg's documented steady state was a permanent skip: nothing refreshes that file
on a schedule, and #1489 deferral 2 asks for "a records-refresh cadence, so the
nightly GEPA cycle consumes fresh real records unattended".

Why a second source rather than a scheduled replay
--------------------------------------------------
Re-running the replay nightly would spend retrieval + generation + judge calls
on 30 golden questions every night, which is exactly what #504 concluded not to
automate. It is also unnecessary: live traffic ALREADY writes the judgeable
triple to ``learning_signals`` on every cognitive turn —

* ``training_input``  -> the user's query,
* ``training_output`` -> the answer that was served,
* ``retrieved_chunks``-> the evidence it was grounded in (#1489 deferral 1).

Measured on the live DB 2026-08-06: 3,523 of 3,959 rows already carry a real
query and answer, spanning 2026-06-10..2026-08-06. That is a continuously
refreshed feedstock that costs nothing to collect, so the cadence deferral 2
asks for is "read what live traffic already wrote", not a new schedule.

Which source is authoritative
-----------------------------
The file, when set. A records file is an operator's explicit, reproducible
choice — the golden set, comparable across runs, and the substrate #1485's gate
measures. The DB is the ambient one: real serving distribution, always fresh,
but uncurated. Explicit beats ambient, so a set ``DSPY_RAG_RECORDS_PATH`` wins
and the DB source is what runs when nobody has chosen.

Why the DB source is OFF by default
-----------------------------------
Enabling it changes the leg from "never runs" to "runs whenever live traffic has
supplied enough judgeable turns", which spends the ``DSPY_RAG_MAX_METRIC_CALLS``
judge budget. That is the intent, but it is an operator's cost decision and must
not arrive as a side effect of merging this module. ``DSPY_RAG_DB_FEEDSTOCK_ENABLED``
is the single switch, parsed fail-closed.

Provenance
----------
The DB read goes through :func:`apply_provenance_filter` like every other
read-path chokepoint. This is load-bearing rather than ceremonial here: 3,300 of
those 3,959 rows are ``is_synthetic`` and all 3,300 carry a query, so without
the predicate the prompt served to real users would be tuned predominantly on
showcase fixtures.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DB_ROW_CAP",
    "FEEDSTOCK_TABLE",
    "RAG_DB_FEEDSTOCK_ENV",
    "RAG_DB_LOOKBACK_DAYS_ENV",
    "RAG_RECORDS_PATH_ENV",
    "SOURCE_DB",
    "SOURCE_FILE",
    "RagExampleBatch",
    "RagExampleSourceUnavailable",
    "db_batch",
    "load_rag_examples",
    "load_rag_examples_from_records",
    "records_batch",
]

# Path to a replay records file produced by scripts/replay_golden_set.py
# (--record-out). Records are consumed as PURE JSON; this module imports no
# replay code. Forwarded to the containers via docker-compose x-common-env.
RAG_RECORDS_PATH_ENV = "DSPY_RAG_RECORDS_PATH"

# Opt-in for the live-traffic feedstock. See the module docstring for why this
# defaults OFF.
RAG_DB_FEEDSTOCK_ENV = "DSPY_RAG_DB_FEEDSTOCK_ENABLED"

# How far back the DB source looks. 30 days is long enough to accumulate turns
# at the measured ~3/10 retrieval-hit rate without training on prompts that
# predate the last few deploys.
RAG_DB_LOOKBACK_DAYS_ENV = "DSPY_RAG_DB_LOOKBACK_DAYS"
_DEFAULT_DB_LOOKBACK_DAYS = 30

FEEDSTOCK_TABLE = "learning_signals"

# Ceiling on rows pulled per beat. The window is newest-first, so a saturated
# read means recent traffic alone exceeded the cap — logged, never silent,
# because the usable rows could then be sitting just past the boundary.
DB_ROW_CAP = 500

SOURCE_FILE = "file"
SOURCE_DB = "db"

# Truthy spellings, matching src/repositories/provenance.py's idiom. Anything
# else is False: this gates real judge spend, so an ambiguous value must fail
# closed (bool("false") is True — the trap coerce_provenance_flag documents).
_TRUTHY = ("1", "true", "yes")

_REPLAY_REMEDY = (
    "Produce one with `.venv/bin/python scripts/replay_golden_set.py "
    "--target cognitive --record-out <path>` and set %s=<path>, or enable the "
    "live-traffic feedstock with %s=true."
) % (RAG_RECORDS_PATH_ENV, RAG_DB_FEEDSTOCK_ENV)


class RagExampleSourceUnavailable(RuntimeError):
    """The configured source could not be read AT ALL.

    Distinct from "read it, and there was not enough in it". Only the latter is
    a completed measurement, and the leg fingerprints completed measurements so
    they are not re-attempted; an unavailable source must stay retryable.
    """

    def __init__(self, reason: str, remedy: str = "") -> None:
        super().__init__(reason if not remedy else f"{reason}. {remedy}")
        self.reason = reason
        self.remedy = remedy


@dataclass(frozen=True)
class _Turn:
    """One judgeable turn, source-independent."""

    query: str
    answer: str
    contexts: Tuple[str, ...]
    intent: str = "UNKNOWN"


@dataclass(frozen=True)
class RagExampleBatch:
    """Examples plus the provenance the leg needs to log and dedup on."""

    examples: Tuple[Any, ...]
    total_records: int
    source: str
    origin: str
    fingerprint_material: bytes

    @property
    def record_noun(self) -> str:
        """What ``total_records`` counts, in words, for this source.

        Not cosmetic. The DB window is pre-narrowed to rows that carry evidence,
        so "0 records" there means "no recent turn carried evidence" — NOT "no
        traffic", which is what an unqualified zero reads as and which would
        send an operator to look for a dead pipeline instead of a missing
        producer. The file source really does count every record in the file.
        """
        return "candidate row" if self.source == SOURCE_DB else "record"

    def fingerprint(self, max_metric_calls: int) -> str:
        """Digest of the feedstock AND the budget, for run-once dedup.

        The budget is part of the key because a run can legitimately end without
        improving on the seed; keying on content alone would mark those records
        permanently done, so a later ``DSPY_RAG_MAX_METRIC_CALLS`` increase —
        the one action that could find an improvement — would be silently
        skipped. For the file source ``fingerprint_material`` is the file's raw
        bytes, so this reproduces the digests already persisted in
        ``.trigger_state.json`` on the production volume byte for byte.
        """
        digest = hashlib.sha256(self.fingerprint_material)
        digest.update(f"|max_metric_calls={max_metric_calls}".encode())
        return digest.hexdigest()


def _example_from_turn(turn: _Turn) -> Any:
    """The ONE dspy.Example shape, whichever source supplied the turn.

    ``retrieved_contexts`` is set alongside ``evidence_board`` on purpose: the
    signature wants an evidence string, while the RAGAS metric wants the passage
    list so context_precision is computed per passage rather than over one blob.
    """
    import dspy

    contexts = list(turn.contexts)
    return dspy.Example(
        user_query=turn.query,
        # Turns carry no investigation goal; the query stands in. This is an
        # INPUT substitution, not a score, but it does mean the prompt is tuned
        # against a slightly narrower input distribution than production sees.
        investigation_goal=turn.query,
        evidence_board=json.dumps(contexts),
        intent=turn.intent,
        retrieved_contexts=contexts,
        synthesis=turn.answer,
    ).with_inputs("user_query", "investigation_goal", "evidence_board", "intent")


def _examples(turns: Sequence[_Turn]) -> Tuple[Any, ...]:
    return tuple(_example_from_turn(t) for t in turns)


def _turn_digest(turn: _Turn) -> str:
    return hashlib.sha256(
        json.dumps(
            [turn.query, turn.answer, list(turn.contexts), turn.intent], sort_keys=True
        ).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# File source (the #1486 shape, unchanged)
# ---------------------------------------------------------------------------
def _turns_from_records(records: Iterable[Any]) -> List[_Turn]:
    """Keep only turns that can actually be judged.

    Filtering here rather than in the metric matters — the RAGAS metric REFUSES
    an unjudgeable example, and a refusal inside GEPA is silently converted to
    failure_score 0.0, which would fabricate a bad-quality signal.
    """
    turns: List[_Turn] = []
    for rec in records:
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        query = (rec.get("query") or "").strip()
        answer = (rec.get("response_text") or "").strip()
        contexts = [c for c in (rec.get("contexts") or []) if isinstance(c, str) and c.strip()]
        if not query or not answer or not contexts:
            continue
        turns.append(
            _Turn(
                query=query,
                answer=answer,
                contexts=tuple(contexts),
                intent=rec.get("detected_intent") or "UNKNOWN",
            )
        )
    return turns


def _records_of(raw: Any) -> List[Any]:
    """Accepts the wrapper the replay writes (``{"records": [...]}``) or a bare list."""
    records = raw.get("records", []) if isinstance(raw, dict) else raw
    return list(records) if isinstance(records, list) else []


def load_rag_examples_from_records(path: str) -> List[Any]:
    """Build dspy Examples from replay records (#1485 shape), as PURE JSON.

    Kept as the #1486 public surface; ``records_batch`` is the seam-aware form.
    """
    return list(records_batch(path).examples)


def records_batch(path: str) -> RagExampleBatch:
    """Read the replay records file into a batch."""
    file_path = Path(path)
    if not file_path.exists():
        raise RagExampleSourceUnavailable(
            f"records file not found: {path}",
            f"Regenerate it with `.venv/bin/python scripts/replay_golden_set.py "
            f"--target cognitive --record-out {path}` ({RAG_RECORDS_PATH_ENV}).",
        )
    try:
        raw_bytes = file_path.read_bytes()
        parsed = json.loads(raw_bytes)
    except (OSError, ValueError) as exc:
        # A file that exists but cannot be read or parsed is the SAME situation
        # as a missing one: the configured source could not be read at all. The
        # beat's blanket guard would otherwise turn a truncated replay write
        # into a leg FAILURE whose reason is a parser traceback, which tells an
        # operator nothing about which file to regenerate. (ValueError covers
        # JSONDecodeError; OSError covers permissions and a vanished file.)
        raise RagExampleSourceUnavailable(
            f"records file could not be read: {path} ({type(exc).__name__}: {exc})",
            f"Regenerate it with `.venv/bin/python scripts/replay_golden_set.py "
            f"--target cognitive --record-out {path}` ({RAG_RECORDS_PATH_ENV}).",
        ) from exc
    records = _records_of(parsed)
    return RagExampleBatch(
        examples=_examples(_turns_from_records(records)),
        total_records=len(records),
        source=SOURCE_FILE,
        origin=path,
        # Raw file bytes: reproduces the already-persisted #1486 digests.
        fingerprint_material=raw_bytes,
    )


# ---------------------------------------------------------------------------
# DB source (live traffic)
# ---------------------------------------------------------------------------
def _db_feedstock_enabled() -> bool:
    return os.environ.get(RAG_DB_FEEDSTOCK_ENV, "").strip().lower() in _TRUTHY


def _db_lookback_days() -> int:
    raw = os.environ.get(RAG_DB_LOOKBACK_DAYS_ENV, "").strip()
    if not raw:
        return _DEFAULT_DB_LOOKBACK_DAYS
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d",
            RAG_DB_LOOKBACK_DAYS_ENV,
            raw,
            _DEFAULT_DB_LOOKBACK_DAYS,
        )
        return _DEFAULT_DB_LOOKBACK_DAYS


def _context_texts(chunks: Any) -> List[str]:
    """Passage text out of ``retrieved_chunks``, tolerant of both writers' shapes.

    #1489's producer writes ``{"content": ...}`` dicts
    (``src/rag/retrieved_chunks.py``, whose docstring makes ``content`` the key
    every producer sets), but migration 022 put NO shape constraint on the JSONB
    column and the replay path's own contexts are bare strings. A reader that
    understood only one shape would silently drop every row the other wrote —
    and a silently short context list is judged as if retrieval had returned
    less than it did.
    """
    if not isinstance(chunks, list):
        return []
    texts: List[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            text = chunk
        elif isinstance(chunk, dict):
            content = chunk.get("content")
            text = content if isinstance(content, str) else ""
        else:
            # A number or null is not evidence. Dropping it is right; counting
            # it as an empty context would tell the judge retrieval returned a
            # blank passage.
            continue
        if text.strip():
            texts.append(text)
    return texts


def _turns_from_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[_Turn], Dict[str, int]]:
    """Judgeable turns, plus a COUNT of why each unusable row was dropped.

    The reasons are returned rather than inferred by the caller because they
    need different fixes in different files — blank ``training_input`` /
    ``training_output`` is a writer problem, evidence that yields no passage is
    a chunk-shape problem — and a diagnosis that guessed between them would send
    an operator to the wrong module.
    """
    turns: List[_Turn] = []
    drops: Dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for row in rows:
        if not isinstance(row, dict):
            _drop("row is not an object")
            continue
        query = str(row.get("training_input") or "").strip()
        answer = str(row.get("training_output") or "").strip()
        if not query or not answer:
            _drop("blank training_input/training_output")
            continue
        contexts = _context_texts(row.get("retrieved_chunks"))
        if not contexts:
            _drop("no readable evidence text in retrieved_chunks")
            continue
        turns.append(_Turn(query=query, answer=answer, contexts=tuple(contexts)))
    return turns, drops


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _resolve_client(client: Optional[Any]) -> Any:
    if client is not None:
        return client
    try:
        from src.memory.services.factories import get_supabase_client

        return await _maybe_await(get_supabase_client())
    except Exception as exc:  # noqa: BLE001 - an unreachable DB is a skip, not a crash
        logger.warning("No Supabase client for the RAG feedstock read: %s", exc)
        return None


async def db_batch(client: Optional[Any] = None) -> RagExampleBatch:
    """Read judgeable turns that live traffic already persisted.

    Read-only by construction: one ``select`` and nothing else.
    """
    from src.repositories.provenance import apply_provenance_filter

    resolved = await _resolve_client(client)
    if resolved is None:
        raise RagExampleSourceUnavailable(
            "no Supabase client for the live-traffic feedstock",
            "The next beat retries; nothing is fingerprinted.",
        )

    lookback = _db_lookback_days()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback)).isoformat()
    query = resolved.table(FEEDSTOCK_TABLE).select(
        "training_input,training_output,retrieved_chunks,created_at"
    )
    # Default-exclude synthetic. Skipped only on a showcase deployment that has
    # opted in via E2I_INCLUDE_SYNTHETIC, exactly like every other reader.
    query = apply_provenance_filter(query)
    query = query.eq("is_training_example", True)
    # Narrow to rows that could POSSIBLY be judgeable before the row cap bites.
    # Measured 2026-08-06: without this the newest-500 window over 30 days came
    # back 499/500 full of rating/thumbs/implicit_positive rows that never carry
    # evidence, so once the producer lands the judgeable turns would have sat
    # just past the boundary and the leg would have reported a confident
    # shortfall. The Python-side shape check below stays the authority; this is
    # only a window narrower. (PostgREST jsonb neq verified against the live DB:
    # signal_details neq '{}' matched, SQL ground truth agreeing at 3,959.)
    query = query.neq("retrieved_chunks", "[]")
    query = query.gte("created_at", cutoff)
    query = query.order("created_at", desc=True).limit(DB_ROW_CAP)

    result = await _maybe_await(query.execute())
    rows = list(getattr(result, "data", None) or [])
    turns, drops = _turns_from_rows(rows)

    if drops:
        # Reported whenever ANY candidate row was dropped, not only when the
        # batch is empty: one usable row among forty dropped ones would
        # otherwise look like a healthy feedstock while the leg trains on a
        # sliver of the traffic that actually carried evidence.
        #
        # Counts, never a guessed cause. Measured 2026-08-06: the only signal
        # that will carry retrieved_chunks is the cognitive Reflector's `agent`
        # signal, whose dict keys its content as query/response, while
        # SignalCollector reads signal["input"]/["output"] — so
        # training_input/training_output persist EMPTY on 133 of 356 agent rows
        # today. That is the cause we EXPECT to dominate, but it is not the only
        # one, so the breakdown is measured rather than asserted.
        #
        # Refusing such rows is deliberate: that signal's `query` is a synthetic
        # descriptor ("Intent: X, Evidence: N items"), not the user's question,
        # so reading it would hand GEPA a fabricated prompt to optimize against.
        logger.warning(
            "RAG feedstock: %d of %d candidate row(s) unusable (%s); %d judgeable "
            "turn(s) kept. Each reason has a different fix — blank text is a "
            "writer that is not persisting the turn, unreadable evidence is a "
            "chunk-shape mismatch.",
            sum(drops.values()),
            len(rows),
            "; ".join(f"{count} {reason}" for reason, count in sorted(drops.items())),
            len(turns),
        )

    if len(rows) >= DB_ROW_CAP:
        # Newest-first, so a full window means usable rows may sit just past the
        # boundary. Say so rather than reporting a confident shortfall.
        logger.warning(
            "RAG feedstock read saturated its %d-row window over the last %d day(s): "
            "%d usable turn(s) found, but older judgeable turns may be excluded. "
            "Narrow the window with %s.",
            DB_ROW_CAP,
            lookback,
            len(turns),
            RAG_DB_LOOKBACK_DAYS_ENV,
        )

    # Order-independent: postgrest ties on created_at could reorder identical
    # feedstock between beats, and a digest that moved on a reorder would
    # re-spend the entire judge budget on turns already optimized against.
    material = ("db:" + "".join(sorted(_turn_digest(t) for t in turns))).encode()
    return RagExampleBatch(
        examples=_examples(turns),
        total_records=len(rows),
        source=SOURCE_DB,
        # "candidate" rather than "row": the window is already narrowed to rows
        # carrying evidence, so a reader of this string is not misled into
        # thinking it counts all recent traffic.
        origin=f"{FEEDSTOCK_TABLE} (last {lookback}d, newest {len(rows)} candidate row(s))",
        fingerprint_material=material,
    )


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------
async def load_rag_examples(client: Optional[Any] = None) -> RagExampleBatch:
    """Resolve the configured feedstock. Explicit file beats ambient DB."""
    path = os.environ.get(RAG_RECORDS_PATH_ENV, "").strip()
    if path:
        return records_batch(path)
    if not _db_feedstock_enabled():
        raise RagExampleSourceUnavailable(
            f"no feedstock configured ({RAG_RECORDS_PATH_ENV} unset and "
            f"{RAG_DB_FEEDSTOCK_ENV} not enabled)",
            _REPLAY_REMEDY,
        )
    return await db_batch(client=client)
