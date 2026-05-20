"""BM25 index rebuild-time monitoring benchmark.

Box 3 of issue #391 PERFORMANCE slice: monitor BM25 index rebuild time as
finding count grows. The benchmark produces a build-time CURVE across
N = 1000 / 5000 / 10000 doc slices, not a single point — finding-count
growth is a curve, not a step function (per issue #391, Box 3 statement
"as finding count grows").

**Reference implementation**: a pure-Python BM25 indexer
(``_build_reference_bm25``) that:
1. tokenizes the corpus (whitespace + lowercase),
2. builds the inverted index ``term → posting list``,
3. computes IDF + average document length (BM25 normalization needs both).

This mirrors the conceptual shape of PostgreSQL's
``to_tsvector``/``tsvector_rank_cd`` pipeline used by the production BM25
surface ``hybrid_fulltext_search`` (migration
``database/memory/022_hybrid_search_max_staleness.sql``). The reference
build is a Python-only stand-in so the benchmark runs in CI without
Supabase; once a real-Postgres benchmark surface lands (e.g., via
``REINDEX TABLE triggers``), it can be added as a parallel test that
skips on ``requires_supabase``.

**Baseline strategy (placeholder-first-run-blesses, per PR #379 +
[[feat-377-phase2-benchmark-close-20260519]])**: the first run on a given
environment BLESSES the measured median build time at each slice
(re-write ``tests/benchmarks/baselines/performance.json`` in that PR).
Subsequent runs compare against the blessed value within the documented
tolerance band.

**Why a curve not a point**: a single-point measurement masks
super-linear scaling — if the index-build complexity drifted from
O(N) to O(N log N), a single 10k measurement could fall inside tolerance
while the small-N points are unchanged. Three slices (1k/5k/10k) catch
this.

Marked ``@pytest.mark.benchmark`` so it does NOT run in the default unit-
test sweep.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytestmark = pytest.mark.benchmark

_HERE = Path(__file__).resolve().parent
_CORPUS_FILE = _HERE / "data" / "synthetic_corpus.jsonl"
_BASELINE_FILE = _HERE / "baselines" / "performance.json"

# BM25 algorithm parameters (canonical Okapi defaults, Robertson & Walker
# 1994). Held constant across slices so the build-time curve is comparable.
_BM25_K1 = 1.5
_BM25_B = 0.75


# ---------------------------------------------------------------------------
# Synthetic corpus loader
# ---------------------------------------------------------------------------


def _load_synthetic_corpus(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"synthetic corpus file not found: {path}; re-run "
            "`python scripts/benchmarks/gen_synthetic_corpus.py`"
        )
    docs: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            docs.append({"doc_id": row["doc_id"], "content": row["content"]})
    return docs


def _slice_corpus(corpus: List[Dict[str, str]], target_n: int) -> List[Dict[str, str]]:
    """Slice the corpus to exactly ``target_n`` docs, repeating + reindexing
    when ``target_n`` exceeds the corpus size.

    Reindexing makes each repeated doc a fresh ``doc_id`` so the inverted
    index treats them as distinct postings — matching the real-corpus shape
    where finding count grows by new rows, not duplicates of existing ones.
    """
    out: List[Dict[str, str]] = []
    i = 0
    while len(out) < target_n:
        src = corpus[i % len(corpus)]
        out.append(
            {
                "doc_id": f"{src['doc_id']}-rep{i // len(corpus):04d}",
                "content": src["content"],
            }
        )
        i += 1
    return out


# ---------------------------------------------------------------------------
# Reference BM25 builder (pure-Python; no new deps)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Whitespace + lowercase tokenizer.

    Conceptually equivalent to the first stage of PostgreSQL's
    ``to_tsvector('english', ...)`` pipeline (modulo stemming + stop-word
    removal, which we deliberately skip to keep the reference impl
    dependency-free and reproducible).
    """
    return text.lower().split()


class _ReferenceBM25Index:
    """Reference BM25 index built over a corpus.

    Attributes:
        avgdl: average document length (token count per doc).
        n_docs: number of docs in the corpus.
        idf: term → inverse document frequency.
        postings: term → list[(doc_id, term_freq_in_doc)].
        doc_lengths: doc_id → token count.

    The fields collectively constitute the rebuilt index — they are what
    a BM25 query-time scorer needs to compute a relevance score. We
    measure wall-clock for populating them, which is the conceptual
    rebuild cost.
    """

    __slots__ = ("avgdl", "n_docs", "idf", "postings", "doc_lengths")

    def __init__(self) -> None:
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[str, int]]] = {}
        self.doc_lengths: Dict[str, int] = {}


def _build_reference_bm25(corpus: List[Dict[str, str]]) -> _ReferenceBM25Index:
    """Build the reference BM25 index over ``corpus``.

    This is the measured surface — its wall-clock is the benchmark.

    Time complexity: O(total_tokens) for tokenization + posting append,
    plus O(unique_terms) for IDF computation, plus O(n_docs) for doc-
    length aggregation. Expected linear in N for fixed average doc length.
    """
    index = _ReferenceBM25Index()
    index.n_docs = len(corpus)

    postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    doc_lengths: Dict[str, int] = {}
    term_doc_freq: Counter[str] = Counter()

    total_length = 0
    for doc in corpus:
        doc_id = doc["doc_id"]
        tokens = _tokenize(doc["content"])
        doc_lengths[doc_id] = len(tokens)
        total_length += len(tokens)
        term_freqs: Counter[str] = Counter(tokens)
        for term, freq in term_freqs.items():
            postings[term].append((doc_id, freq))
            term_doc_freq[term] += 1

    # IDF per Robertson & Walker 1994:
    #     idf(term) = ln((N - df + 0.5) / (df + 0.5) + 1)
    # (the "+1" is Lucene/Tantivy's smoothing variant; canonical for
    # non-negative IDF on all terms).
    idf: Dict[str, float] = {}
    n_docs = index.n_docs
    for term, df in term_doc_freq.items():
        idf[term] = math.log(((n_docs - df + 0.5) / (df + 0.5)) + 1)

    index.avgdl = total_length / max(n_docs, 1)
    index.idf = idf
    index.postings = dict(postings)
    index.doc_lengths = doc_lengths

    return index


# ---------------------------------------------------------------------------
# Baseline + tolerance comparison
# ---------------------------------------------------------------------------


def _load_baseline() -> Dict[str, Any]:
    if not _BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"performance baseline file missing: {_BASELINE_FILE}; "
            "seed it before running the harness"
        )
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        baseline: Dict[str, Any] = json.load(fh)
    return baseline


def _within_tolerance(
    observed_ms: float,
    baseline_ms: float,
    tolerance_pct: float,
    tolerance_abs_ms: float,
) -> Tuple[bool, str]:
    """Return (within_band, human_readable_reason). Mirrors the helpers in
    the sibling cascade + hybrid_retriever benchmarks."""
    if observed_ms <= baseline_ms:
        return True, f"improvement: observed={observed_ms:.2f}ms <= baseline={baseline_ms:.2f}ms"
    delta = observed_ms - baseline_ms
    band = max(baseline_ms * (tolerance_pct / 100.0), tolerance_abs_ms)
    if delta <= band:
        return (
            True,
            f"within band: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
            f"delta={delta:.2f}ms, band={band:.2f}ms ({tolerance_pct}% rel OR "
            f"{tolerance_abs_ms}ms abs)",
        )
    return (
        False,
        f"REGRESSION: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
        f"delta={delta:.2f}ms exceeds band={band:.2f}ms "
        f"({tolerance_pct}% rel OR {tolerance_abs_ms}ms abs)",
    )


def _run_build_once(corpus: List[Dict[str, str]]) -> float:
    """Build the BM25 reference index once, return wall-clock ms.

    Sanity-checks the resulting index has the expected shape so a silent
    regression (e.g., index aborts mid-build) produces a louder failure
    than a spuriously fast measurement.
    """
    start = time.perf_counter()
    index = _build_reference_bm25(corpus)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert index.n_docs == len(corpus), (
        f"BM25 index has wrong doc count: {index.n_docs} != {len(corpus)}"
    )
    assert index.idf, "BM25 index has no IDF terms"
    assert index.avgdl > 0, f"BM25 avgdl must be > 0, got {index.avgdl}"
    return elapsed_ms


def _median(values: List[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthetic_corpus_loads() -> None:
    """Smoke test: shipped synthetic JSONL parses without error.

    Falsifiability anchor: this MUST always pass on the shipped corpus.
    """
    corpus = _load_synthetic_corpus(_CORPUS_FILE)
    assert len(corpus) >= 1000, f"Expected >=1000 docs (per CURATION_PERF.md), got {len(corpus)}"


def test_reference_bm25_builds_smoke() -> None:
    """Smoke test: reference BM25 index builds successfully on a small slice.

    Falsifiability anchor: if this fails, the reference impl is broken and
    no slice-level latency measurement is meaningful.
    """
    corpus = _load_synthetic_corpus(_CORPUS_FILE)
    small = _slice_corpus(corpus, 100)
    index = _build_reference_bm25(small)
    assert index.n_docs == 100
    assert index.idf  # at least one term
    # Every IDF must be non-negative under the Lucene/Tantivy smoothing variant.
    for term, idf in index.idf.items():
        assert idf >= 0.0, f"BM25 IDF must be non-negative; got idf({term!r})={idf}"


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "slice_n, baseline_key",
    [
        (1000, "bm25_build_1k"),
        (5000, "bm25_build_5k"),
        (10000, "bm25_build_10k"),
    ],
)
def test_bm25_rebuild_time_against_baseline(slice_n: int, baseline_key: str) -> None:
    """Box 3 of issue #391: monitor BM25 build time at multiple corpus sizes.

    Slices the synthetic corpus to ``slice_n`` docs (via repetition +
    re-indexing — see ``_slice_corpus`` docstring), runs 3 warm builds,
    reports the median against the blessed baseline within the documented
    tolerance band.

    **Re-blessing the baseline**: if the measurement legitimately shifts
    (e.g., after a tokenizer refactor), update
    ``tests/benchmarks/baselines/performance.json`` in the same PR with
    the new ``mean_ms`` at each slice. Do NOT loosen tolerances to mask a
    regression at any slice.
    """
    corpus = _load_synthetic_corpus(_CORPUS_FILE)
    slice_corpus = _slice_corpus(corpus, slice_n)

    runs = 3
    timings: List[float] = []
    for _ in range(runs):
        timings.append(_run_build_once(slice_corpus))
    median_ms = _median(timings)

    baseline = _load_baseline()
    spec = baseline[baseline_key]
    baseline_ms = float(spec["mean_ms"])
    tol_pct = float(spec["tolerance_pct"])
    tol_abs = float(spec["tolerance_abs_ms"])

    print(
        f"\n[issue-#391 box-3] BM25 build-time @ N={slice_n}:"
        f"\n  runs={runs}, timings_ms={[f'{t:.2f}' for t in timings]}"
        f"\n  median_ms={median_ms:.2f}"
        f"\n  baseline_ms={baseline_ms:.2f} (tol: {tol_pct}% rel OR {tol_abs}ms abs)"
        + (
            "\n  NOTE: baseline is 0.0 — this run is the placeholder-blessing "
            "first run; re-write tests/benchmarks/baselines/performance.json "
            f"with the median value for {baseline_key} in this PR."
            if baseline_ms == 0.0
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    if baseline_ms == 0.0:
        return

    within, reason = _within_tolerance(median_ms, baseline_ms, tol_pct, tol_abs)
    assert within, reason
