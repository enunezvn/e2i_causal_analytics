"""Offline A/B: multi-model ensemble vs single-Sonnet over the golden set (#242).

Runs the COMPILED causal-role classifier (what production single-Sonnet uses)
under all three ensemble members (Sonnet 4.6 + Opus 4.7 + GPT-5) on the
``causal_role_golden_set.json`` entries whose ground-truth role is leak-relevant
(``descendant`` = leaks, plus the subtle ``collider`` / ``mediator`` boundaries),
and reports a precision-style comparison plus the #242 AC5 signal:

* AC5 case        — single-Sonnet WRONG and the ensemble {right via majority | escalated via split}
* leak FN         — gt=descendant but single-Sonnet says benign (ancestor/confounder/instrument)
* ensemble regression — single-Sonnet right but the ensemble wrong (harm check)
* correlated failure  — all three models (incl GPT-5) wrong together (multi-vendor independence failed)

This is the reproducible harness behind ``docs/plans/242-p8-ab-findings.md``.
Requires live API + credits for all three providers (preflight fails loudly if a
key is missing). Re-run after an Anthropic credit top-up to get an uncontaminated
full dataset:

    python scripts/measure_ensemble_ab.py --roles descendant,collider,mediator

Spend: ~3 calls/entry (~$0.06/entry at current list prices).

Anti-money-waste fixes (Issue #242):

* Checkpoint/resume: results are written incrementally (after each entry) to
  ``--out``.  On startup, entries already present in the file are SKIPPED unless
  ``--force`` is given.  Contaminated (provider-outage) rows are retried on
  resume (they are not considered done).

* Budget guard: ``--max-cost FLOAT`` caps cumulative spend.  The script prints
  an upfront estimate before the first API call.  It stops cleanly (with a
  resume hint) before a call would push total spend over the cap.

* Graceful quota stop: a credit-balance / quota error from any model triggers
  a clean stop + checkpoint rather than continuing and marking remaining entries
  as ``[CONTAMINATED]`` (which pollutes the dataset and looks like real splits).

* Order control: ``--order {file,reverse,shuffle}`` (default ``file``) lets a
  budget-limited run avoid always starving the same tail entries.  ``--seed``
  makes shuffle reproducible.

De-confound fix (Issue #242):

* ``--prompt-mode {compiled,zeroshot}`` (default ``compiled``): in ``zeroshot``
  mode each vendor's call uses a FRESH (uncompiled) CausalRoleClassifier with
  no Sonnet-optimised few-shot demos.  This eliminates the correlation
  Sonnet-bias imports into Opus and GPT-5 when all three share the same
  compiled artifact.  Each output row records ``prompt_mode`` so runs are
  self-describing.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any, Literal, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)  # find_dotenv walks up to the repo .env (works from a worktree)

from src.data import causal_role_classifier_ensemble as ens  # noqa: E402
from src.data.causal_role_classifier_loader import load_compiled_classifier  # noqa: E402

logger = logging.getLogger(__name__)

GOLDEN = REPO_ROOT / "tests/fixtures/causal_role_golden_set.json"
BENIGN = {"ancestor", "confounder", "instrument"}  # for leak-false-negative detection

# Assumed token size per call for the upfront cost ESTIMATE only (the friendly
# "~$X for N entries" print). Based on observed golden-set entry sizes
# (~300 input, ~50 output tokens). Actual spend varies; the per-entry telemetry
# accumulates the real cost — this estimate never gates the budget guard.
_EST_INPUT_TOKENS_PER_CALL = 300
_EST_OUTPUT_TOKENS_PER_CALL = 50
_EST_USD_PER_CALL = (
    _EST_INPUT_TOKENS_PER_CALL / 1e6 * ens.SONNET_INPUT_USD_PER_MTOK
    + _EST_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.SONNET_OUTPUT_USD_PER_MTOK
    + _EST_INPUT_TOKENS_PER_CALL / 1e6 * ens.OPUS_INPUT_USD_PER_MTOK
    + _EST_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.OPUS_OUTPUT_USD_PER_MTOK
    + _EST_INPUT_TOKENS_PER_CALL / 1e6 * ens.GPT5_INPUT_USD_PER_MTOK
    + _EST_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.GPT5_OUTPUT_USD_PER_MTOK
)

# CONSERVATIVE per-entry upper bound used by the BUDGET GUARD (not the estimate).
# The guard stops BEFORE a call when accumulated-actual + this bound would exceed
# --max-cost, so actual spend can never overshoot the cap (better to stop one
# entry early than overspend). Sized for the Opus-heavy worst case: a long
# reasoning entry at ~3k input + ~1k output tokens across all 3 models. Opus
# dominates ($15/$75 per MTok). Recompute if pricing or expected entry size
# changes; it only needs to be a true upper bound on a single 3-model entry.
_BOUND_INPUT_TOKENS_PER_CALL = 3000
_BOUND_OUTPUT_TOKENS_PER_CALL = 1000
_EST_MAX_USD_PER_ENTRY = (
    _BOUND_INPUT_TOKENS_PER_CALL / 1e6 * ens.SONNET_INPUT_USD_PER_MTOK
    + _BOUND_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.SONNET_OUTPUT_USD_PER_MTOK
    + _BOUND_INPUT_TOKENS_PER_CALL / 1e6 * ens.OPUS_INPUT_USD_PER_MTOK
    + _BOUND_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.OPUS_OUTPUT_USD_PER_MTOK
    + _BOUND_INPUT_TOKENS_PER_CALL / 1e6 * ens.GPT5_INPUT_USD_PER_MTOK
    + _BOUND_OUTPUT_TOKENS_PER_CALL / 1e6 * ens.GPT5_OUTPUT_USD_PER_MTOK
)

# Patterns that identify a credit-balance / quota exhaustion error from ANY
# provider (Anthropic credit balance OR OpenAI insufficient_quota). Stopping on
# a real quota error from either vendor is correct — it is money / hard limits,
# not a transient blip. Transient rate-limit-with-retry messages are
# deliberately NOT matched here (those degrade to a single non-vote and the run
# continues); only an exhaustion signal halts the run. These strings appear in
# HTTP 402 / 429 response bodies.
_QUOTA_PATTERNS: list[str] = [
    r"credit balance is too low",  # Anthropic 402
    r"insufficient.{0,20}credit",  # Anthropic variants
    r"insufficient.{0,20}quota",  # OpenAI insufficient_quota
    r"\bquota\b",  # any "...exceeded your current quota..." (OpenAI 429)
    r"429 insufficient",
    r"rate.limit.exceeded",  # hard rate-limit exhaustion (not a retry hint)
]
_QUOTA_RE = re.compile("|".join(_QUOTA_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no side-effects / no API calls)
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a previously-persisted checkpoint from *path*.

    Returns a dict keyed by ``feature_name`` so callers can do O(1) lookups.
    Returns an empty dict when the file does not exist or contains corrupt JSON
    (treat both as "no progress yet" — a corrupt file is NOT silently extended).
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        rows = data.get("rows", [])
        return {r["feature_name"]: r for r in rows}
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning("_load_checkpoint: corrupt or unreadable file %s — starting fresh.", path)
        return {}


def _remaining(
    entries: list[dict[str, Any]],
    done: dict[str, Any],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    """Return entries that still need to be measured.

    * ``force=True``: all entries (re-measure everything).
    * ``force=False``: skip entries that are in *done* AND NOT contaminated.
      A contaminated row (provider outage / credit exhaustion) is treated as
      NOT-done so the next resume retries it.
    """
    if force:
        return list(entries)
    result = []
    for e in entries:
        name = e["feature_name"]
        prior = done.get(name)
        if prior is None:
            result.append(e)
        elif prior.get("contaminated", False):
            # Previous run had a provider outage for this entry — retry it.
            result.append(e)
        # else: prior is a clean measurement — skip
    return result


def _estimate_cost(n_entries: int, per_call_usd: float) -> float:
    """Upfront spend estimate: n_entries × per_call_usd (ensemble = 3 models/entry,
    so *per_call_usd* should already be the 3-model aggregate)."""
    return n_entries * per_call_usd


def _is_quota_error(exc_or_text: Any) -> bool:
    """Return True iff *exc_or_text* signals credit-balance / quota exhaustion
    from ANY provider (Anthropic credit-balance OR OpenAI insufficient_quota).

    Stopping on a real quota error from either vendor is correct — it is a
    money / hard-limit condition, not a transient blip. Accepts an Exception
    (its string representation is searched) or a plain str. Transient
    rate-limit-with-retry messages, timeouts, and other errors return False so
    the caller handles them differently (record as a non-vote and continue).
    """
    text = str(exc_or_text)
    return bool(_QUOTA_RE.search(text))


def _order_entries(
    entries: list[dict[str, Any]],
    order: str,
    seed: Optional[int],
) -> list[dict[str, Any]]:
    """Return a (possibly reordered) COPY of *entries*.

    ``file``    — original order (reproducible, default).
    ``reverse`` — reversed (start from the tail that previously starved).
    ``shuffle`` — random permutation seeded by *seed* (deterministic when seed
                  is fixed; *seed=None* uses the system random state, which is
                  non-deterministic).

    Does NOT mutate the input list.
    """
    if order == "file":
        return list(entries)
    if order == "reverse":
        return list(reversed(entries))
    if order == "shuffle":
        result = list(entries)
        rng = random.Random(seed)
        rng.shuffle(result)
        return result
    raise ValueError(f"Unknown order={order!r}; expected 'file', 'reverse', or 'shuffle'.")


def _persist(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write (overwrite) the checkpoint file with all *rows* accumulated so far.

    Called after EACH entry during a run so a crash or quota stop loses at
    most one entry.
    """
    path.write_text(json.dumps({"rows": rows}, indent=2))


# ---------------------------------------------------------------------------
# Inner loop helpers (injectable for testing)
# ---------------------------------------------------------------------------


def _run_ensemble_for_entry(
    *,
    feature_name: str,
    derivation_pseudocode: str,
    dataset_context: str,
    models: tuple[str, ...],
    classifier: Any,
    prompt_mode: Literal["compiled", "zeroshot"],
) -> Any:
    """Single-entry dispatch to ``run_ensemble_classification``.

    Thin wrapper kept separate so tests can monkeypatch it without touching
    the real ensemble module.
    """
    return ens.run_ensemble_classification(
        feature_name=feature_name,
        derivation_pseudocode=derivation_pseudocode,
        dataset_context=dataset_context,
        models=models,
        classifier=classifier,
        prompt_mode=prompt_mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser.  Separate from ``main()`` so tests
    can call ``_build_parser().parse_args(...)`` without touching sys.argv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roles",
        default="descendant,collider,mediator",
        help="Comma-separated ground-truth roles to include (default: leak-relevant).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap entries (0 = all).")
    parser.add_argument("--out", type=Path, default=None, help="Write per-entry JSON here.")
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-measure all entries even if --out already has results.",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        dest="max_cost",
        help=(
            "Maximum cumulative spend in USD.  The script stops before making "
            "a call that would push total spend over this cap."
        ),
    )
    parser.add_argument(
        "--order",
        choices=["file", "reverse", "shuffle"],
        default="file",
        help=(
            "Entry ordering.  'file' = golden-set order (default, reproducible); "
            "'reverse' = start from the tail; 'shuffle' = random (use --seed for "
            "reproducibility)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for --order shuffle.  Default: non-deterministic.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["compiled", "zeroshot"],
        default="compiled",
        dest="prompt_mode",
        help=(
            "How to prompt each ensemble member.  'compiled' (default) uses the "
            "Sonnet-optimised few-shot demos from the compiled artifact — same as "
            "production single-Sonnet.  'zeroshot' strips demos so each vendor "
            "reasons from the bare signature only, eliminating Sonnet-bias "
            "correlation across models (#242 de-confound ablation)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Core measurement loop (pure of argparse; injectable for tests)
# ---------------------------------------------------------------------------


def _run_measurement_loop(
    *,
    entries: list[dict[str, Any]],
    done: dict[str, Any],
    models: tuple[str, ...],
    classifier: Any,
    max_cost: Optional[float],
    out: Optional[Path],
    prompt_mode: Literal["compiled", "zeroshot"],
) -> tuple[list[dict[str, Any]], Optional[str]]:
    """Run the measurement loop over *entries* (already filtered by _remaining).

    Returns ``(rows, stop_reason)`` where ``stop_reason`` is:
    * ``None``    — completed all entries normally
    * ``"budget"``— stopped because the next call could exceed ``max_cost``
    * ``"quota"`` — stopped because a quota/credit-exhaustion error was seen

    ``rows`` contains only entries measured in THIS call (not the full done dict).

    Checkpointing (HIGH 3): rows are held in a dict keyed by ``feature_name`` so
    a retried row REPLACES its prior (e.g. contaminated) version in place. The
    deduped set is written on EVERY incremental persist, so a mid-run exit never
    leaves duplicate rows for one feature in the checkpoint.

    NOTE: ``out`` is written after EACH entry (incremental checkpoint).
    """
    # Keyed by feature_name so a retry REPLACES a prior row (e.g. a contaminated
    # one) rather than coexisting with it. Seeded from *done* so the incremental
    # checkpoint always holds ALL prior + current results, deduped.
    rows_by_name: dict[str, dict[str, Any]] = {name: dict(row) for name, row in done.items()}
    new_rows: list[dict[str, Any]] = []
    cumulative_cost = 0.0
    stop_reason: Optional[str] = None

    for e in entries:
        gt = e["ground_truth_role"]

        # --- Budget pre-check (HIGH 2) ---
        # Use a CONSERVATIVE upper bound, not the friendly estimate, so actual
        # spend can never overshoot the cap. Stop one entry early rather than
        # risk an Opus-heavy entry pushing accumulated cost over --max-cost.
        if max_cost is not None:
            if cumulative_cost + _EST_MAX_USD_PER_ENTRY > max_cost:
                print(
                    f"\n[BUDGET] Accumulated ${cumulative_cost:.4f} + conservative "
                    f"per-entry bound ${_EST_MAX_USD_PER_ENTRY:.4f} would exceed cap "
                    f"${max_cost:.4f} — stopping before the next call. "
                    f"Re-run with --out to resume from this point."
                )
                stop_reason = "budget"
                break

        try:
            clf = _run_ensemble_for_entry(
                feature_name=e["feature_name"],
                derivation_pseudocode=e.get("derivation_pseudocode", ""),
                dataset_context=e.get("dataset_context", ""),
                models=models,
                classifier=classifier,
                prompt_mode=prompt_mode,
            )
        except Exception as exc:  # noqa: BLE001
            # A quota error that propagates as a raise (e.g. preflight / non-vote
            # path bypassed) is still handled cleanly; the common case is the
            # vote.error path below (HIGH 1).
            if _is_quota_error(exc):
                print(
                    f"\n[QUOTA] Credit/quota exhaustion detected: {exc}\n"
                    f"Checkpointed {len(new_rows)} new entries this run. "
                    f"Top up credits and re-run with --out to resume."
                )
                stop_reason = "quota"
                break
            # Non-quota transient error on a single run → bubble up (not suppressed)
            raise

        # Accumulate actual cost from telemetry
        if clf.total_cost_usd is not None:
            cumulative_cost += clf.total_cost_usd

        s = next((v.causal_role for v in clf.votes if "sonnet" in v.model), None)
        o = next((v.causal_role for v in clf.votes if "opus" in v.model), None)
        g = next((v.causal_role for v in clf.votes if "gpt" in v.model), None)

        # HIGH 4: a missing GPT-5 vote means this is NOT a valid multi-VENDOR
        # measurement (the whole point of #242), so the row is contaminated and
        # excluded from conclusions — but a GPT-5-only failure does NOT halt the
        # run (only account-wide quota exhaustion does, handled below).
        contaminated = s is None or o is None or g is None

        # HIGH 1: _classify_one SWALLOWS exceptions into vote.error (correct for
        # telemetry / degrade-to-healthy), so a credit-exhaustion error never
        # reaches the except-clause above. Inspect each member's error string;
        # if ANY is a quota/credit-exhaustion error, persist what we have and
        # STOP — do NOT keep iterating remaining entries recording them as
        # contaminated non-votes (the data-pollution failure we are fixing).
        quota_errors = [
            getattr(v, "error", None)
            for v in clf.votes
            if getattr(v, "error", None) and _is_quota_error(getattr(v, "error", ""))
        ]

        if quota_errors:
            # This entry's votes are tainted by an exhaustion error — it is NOT a
            # valid measurement, so it is NOT recorded. Persist what we already
            # have (clean prior entries) and stop.
            print(
                f"{e['feature_name'][:40]:40s} gt={gt:10s} -> [QUOTA] "
                f"member error: {quota_errors[0]}"
            )
            print(
                f"\n[QUOTA] Credit/quota exhaustion detected in a model vote: "
                f"{quota_errors[0]}\n"
                f"Checkpointed {len(new_rows)} new entries this run; remaining "
                f"entries NOT measured. Top up credits and re-run with --out to resume."
            )
            stop_reason = "quota"
            if out is not None:
                _persist(out, list(rows_by_name.values()))
            break

        print(
            f"{e['feature_name'][:40]:40s} gt={gt:10s} S={str(s):10s} O={str(o):10s} "
            f"G={str(g):10s} -> {clf.agreement}/{clf.fused_role}"
            f"{' [CONTAMINATED]' if contaminated else ''}"
        )

        row: dict[str, Any] = {
            "feature_name": e["feature_name"],
            "cohort": e.get("cohort"),
            "gt": gt,
            "sonnet": s,
            "opus": o,
            "gpt5": g,
            "agreement": clf.agreement,
            "fused_role": clf.fused_role,
            "contaminated": contaminated,
            "prompt_mode": prompt_mode,
        }
        new_rows.append(row)
        # Replace any prior row for this feature (HIGH 3 — retry overwrites).
        rows_by_name[e["feature_name"]] = row

        # Incremental checkpoint: write the DEDUPED set after EVERY entry so a
        # crash or quota stop loses at most one entry and never duplicates rows.
        if out is not None:
            _persist(out, list(rows_by_name.values()))

    return new_rows, stop_reason


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """Print the same summary table as the original harness, over *rows*."""
    n = s_ok = e_ok = splits = misses = ac5 = regress = correlated = 0
    leak_fn = leak_fn_caught = 0
    for row in rows:
        gt = row["gt"]
        s = row["sonnet"]
        o = row["opus"]
        g = row["gpt5"]
        fused_role = row["fused_role"]
        agreement = row["agreement"]
        n += 1
        s_correct = s == gt
        e_correct = fused_role == gt
        is_split = agreement == "split"
        s_ok += s_correct
        e_ok += e_correct
        splits += is_split
        s_wrong = s is not None and not s_correct
        misses += s_wrong
        if s_wrong and (e_correct or is_split):
            ac5 += 1
        if s_correct and not e_correct:
            regress += 1
        if all(v is not None and v != gt for v in (s, o, g)):
            correlated += 1
        is_leak_fn = gt == "descendant" and s in BENIGN
        leak_fn += is_leak_fn
        if is_leak_fn and (e_correct or is_split):
            leak_fn_caught += 1

    clean = [r for r in rows if not r.get("contaminated", False)]
    print(f"\n=== A/B (n={n}, clean={len(clean)}) ===")
    print(f"single-Sonnet correct : {s_ok}/{n}")
    print(f"ensemble role-correct : {e_ok}/{n}  (splits/escalated={splits})")
    print(f"sonnet misses={misses}  AC5(caught|escalated)={ac5}  regressions={regress}")
    print(f"correlated all-3-wrong={correlated}  leak-FN={leak_fn} (caught={leak_fn_caught})")
    if len(clean) != n:
        print(
            f"WARNING: {n - len(clean)} entries CONTAMINATED (provider outage) — exclude from conclusions."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    entries = json.loads(GOLDEN.read_text())["entries"]
    sample = [e for e in entries if e.get("ground_truth_role") in roles]
    if args.limit:
        sample = sample[: args.limit]

    # --- Load prior checkpoint ---
    done: dict[str, Any] = {}
    if args.out and args.out.exists() and not args.force:
        done = _load_checkpoint(args.out)
        if done:
            print(
                f"[RESUME] Found {len(done)} previously-measured entries in {args.out}. "
                f"Skipping done (non-contaminated) rows.  Use --force to re-measure all."
            )

    # --- Order entries ---
    sample = _order_entries(sample, order=args.order, seed=args.seed)

    # --- Filter to remaining ---
    remaining = _remaining(sample, done=done, force=args.force)

    print(
        f"golden={len(entries)} roles={sorted(roles)} sample={len(sample)} "
        f"remaining={len(remaining)} (~{len(remaining) * 3} calls)"
    )

    # --- Upfront cost estimate ---
    estimate = _estimate_cost(len(remaining), _EST_USD_PER_CALL)
    print(
        f"[ESTIMATE] ~${estimate:.4f} for {len(remaining)} entries "
        f"(${_EST_USD_PER_CALL:.4f}/entry; documented assumed tokens: "
        f"~{_EST_INPUT_TOKENS_PER_CALL} in / ~{_EST_OUTPUT_TOKENS_PER_CALL} out per model call)"
    )
    if args.max_cost is not None:
        print(f"[BUDGET CAP] ${args.max_cost:.4f}")

    if not remaining:
        print("[DONE] All entries already measured.  Use --force to re-measure.")
        # Still print summary over all done rows
        _print_summary(list(done.values()))
        return 0

    models = ens._resolve_models()
    ens._preflight_models(models)  # loud if any provider key absent
    classifier = load_compiled_classifier()
    print(
        f"models: {models} | compiled classifier: {classifier is not None} "
        f"| prompt-mode: {args.prompt_mode}"
    )
    if args.prompt_mode == "zeroshot":
        print(
            "[ZEROSHOT] Each model will use a FRESH (uncompiled) CausalRoleClassifier "
            "with no Sonnet-optimised demos — Sonnet-bias correlation eliminated."
        )

    new_rows, stop_reason = _run_measurement_loop(
        entries=remaining,
        done=done if not args.force else {},
        models=tuple(models),
        classifier=classifier,
        max_cost=args.max_cost,
        out=args.out,
        prompt_mode=args.prompt_mode,
    )

    # Final write if --out was not given (no incremental writes happened)
    if args.out and not remaining:
        pass  # nothing new to write
    elif args.out and new_rows:
        # Already written incrementally; do a final sync to be sure
        all_rows = list({**done, **{r["feature_name"]: r for r in new_rows}}.values())
        _persist(args.out, all_rows)
        print(f"wrote {args.out} ({len(all_rows)} total entries)")

    # Print summary over all measured rows (done + new)
    all_measured = list({**done, **{r["feature_name"]: r for r in new_rows}}.values())
    _print_summary(all_measured)

    if stop_reason == "budget":
        print(
            f"\n[RESUME] Budget cap reached.  Re-run with the same --out flag "
            f"after topping up funds:\n"
            f"  .venv/bin/python scripts/measure_ensemble_ab.py "
            f"--out {args.out} --roles {args.roles}"
        )
        return 1
    if stop_reason == "quota":
        print(
            f"\n[RESUME] Provider quota / credit exhausted.  Top up the affected "
            f"provider's balance and re-run:\n"
            f"  .venv/bin/python scripts/measure_ensemble_ab.py "
            f"--out {args.out} --roles {args.roles}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
