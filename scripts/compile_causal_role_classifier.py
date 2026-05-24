"""Phase 2.5 — DSPy compile-and-persist for the Layer 4 CausalRoleClassifier.

PREREQUISITE (Codex Gate-2 MED-5, Plan
``.claude/plans/layer4_evaluator_audit_consumer.md``): before producing a
new compiled artifact, run ``make curate-candidates`` and review the
resulting markdown report under ``./candidates/``. Any accepted candidate
must be hand-merged into ``build_compile_set()`` in
``src/data/causal_role_classifier.py`` BEFORE this script is run. Skipping
this step means new evaluator-flagged disagreements never reach the
compile set; the new artifact will be a copy of the old one.

Phase 4.5 enforcement (issue #236): this script now runs a pre-flight
backlog check against ``./candidates/*.json`` (configurable via
``--candidates-dir``) and refuses to recompile when zero accepted
candidates have landed since the existing artifact's mtime. Pass
``--force`` to override (the operator's explicit acknowledgement that
they intend a no-evidence recompile, e.g. for a determinism re-run).
The pre-flight is skipped automatically when the artifact does not yet
exist (cold-start bootstrap).

Compiles ``src.data.causal_role_classifier.CausalRoleClassifier`` via
``BootstrapFewShot`` against the 33-example compile set (21 legacy +
12 Phase-4 S12 Option C paired demos) produced by
``build_compile_set()`` and writes the compiled program JSON to::

    artifacts/dspy/causal_role_classifier.json

Determinism: ``BootstrapFewShot`` is seeded indirectly via the seed argument
passed to ``random.seed`` / ``numpy.random.seed`` at the top of this script.
DSPy's bootstrap doesn't expose a teleprompter-level seed parameter (DSPy 3.1
``BootstrapFewShot.__init__`` signature has none), so the script pins all the
random sources the optimizer touches at the start of the run. With a fixed
seed and a fixed LM endpoint, two runs produce the same bootstrapped demos
(modulo LM nondeterminism — see ``--lm-model`` notes below for the trade-off
between persistence reproducibility and runtime LM behaviour).

Usage::

    # Production compile against the documented Anthropic endpoint:
    python scripts/compile_causal_role_classifier.py \
        --lm-model anthropic/claude-sonnet-4-20250514 \
        --out artifacts/dspy/causal_role_classifier.json

    # CI / no-LM-key path: skips compile, emits a deterministic stub program
    # that ChainOfThought can still load (preserves the schema, no LM calls).
    python scripts/compile_causal_role_classifier.py --no-lm --out <path>

Phase 2.5 / Phase 2.9 Stage 3 unblock: see
``.claude/plans/adaptive_temporal_validity_redesign.md`` line 348.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Allow `python scripts/compile_causal_role_classifier.py` without `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so CLI invocations get ANTHROPIC_API_KEY without manual export.
# Without this the script's --lm-model path fails to authenticate even when
# the key sits in .env (conftest does this for pytest paths; CLI does not).
load_dotenv()

# dspy / classifier imports are deferred to ``compile_and_persist`` so
# the lightweight ``preflight_candidate_check`` helper (and its tests)
# can run in environments without the full DSPy + LangChain dep stack
# installed.
from scripts.check_compile_set_candidate_backlog import (  # noqa: E402
    count_accepted_candidates,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = PROJECT_ROOT / "artifacts" / "dspy" / "causal_role_classifier.json"
# Default location for ``make curate-candidates`` output. The pre-flight
# (issue #236) walks this dir for newer-than-artifact manifests.
DEFAULT_CANDIDATES_DIR = PROJECT_ROOT / "candidates"
DEFAULT_LM_MODEL = "anthropic/claude-sonnet-4-20250514"
DEFAULT_SEED = 7
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 4
# Historical pre-Option-C context (issue #198 codex pass-4 MED-1, 2025
# era when len(build_compile_set()) == 20): the cap was raised
# 16 -> 24 so all 20 labeled compile-set examples survived
# BootstrapFewShot._train's random.sample(demos, max_labeled_demos)
# step. The cap=16 setting from pass-3 dropped 4-8 labeled examples
# randomly per run; the pass-4 audit found this routinely dropped the
# provider IV family (provider_preference_score and
# index_provider_biologic_volume_prior_year) entirely.
#
# Phase-4 S12 Option C recompile (2026-05-19): raised from 24 -> 40 so
# all 33 labeled compile-set examples (21 legacy + 12 new (T, Y)-
# explicit paired-fixture demos per `.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md`)
# survive the random.sample step inside BootstrapFewShot._train.
# Computed as 33 labeled + 4 bootstrapped = 37, +3 slot conservative
# headroom for any future small additions. The §3.5 paired-fixture
# falsifiability gate requires all 12 quadruples to land in the
# persisted artifact, so the cap must be >= len(build_compile_set()).
# Pre-Option-C cap of 24 < 33 would force random.sample(33, 24) to
# drop 9 of 33 demos uniformly per run (~27% per-demo loss probability);
# at cap=40 >= 33 the sample step retains all labeled demos
# deterministically.
DEFAULT_MAX_LABELED_DEMOS = 40

# Plan-239 §5.5: --optimizer miprov2 uses a higher labeled-demo cap so the
# 50-example compile set survives `random.sample(demos, max_labeled_demos)`
# with headroom. Default 60 (= 50 + 10 slot headroom for future expansion).
MIPROV2_DEFAULT_MAX_LABELED_DEMOS = 60

# Plan-239 §5.2: artifact JSON keys whose values are nondeterministic
# across runs (timestamps, cache counters, run IDs). normalize_artifact_json
# strips these before byte-comparing two compiled artifacts for the AC2
# reproducibility test (Tier-2).
VOLATILE_KEY_ALLOWLIST: frozenset[str] = frozenset(
    {
        "compiled_at",
        "cache_hits",
        "cache_misses",
        "lm_request_count",
        "elapsed_seconds",
        "run_id",
    }
)


def normalize_artifact_json(path: Path) -> str:
    """Plan-239 §5.2: canonical comparable string for reproducibility checks.

    Drops volatile keys (timestamps, cache counters, run IDs), sorts keys,
    normalizes whitespace. Two MIPROv2 compiles under the same fixed seed
    should produce byte-identical normalized output.
    """
    obj = json.loads(Path(path).read_text())

    def _strip(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: _strip(v) for k, v in o.items() if k not in VOLATILE_KEY_ALLOWLIST}
        if isinstance(o, list):
            return [_strip(x) for x in o]
        return o

    return json.dumps(_strip(obj), sort_keys=True, indent=2)


def preflight_candidate_check(
    *,
    candidates_dir: Path,
    compiled_artifact_path: Path,
    force: bool,
) -> tuple[bool, str]:
    """Phase 4.5 pre-flight: gate the compile on the curation pipeline.

    Returns ``(proceed, message)``. The compile is allowed when:

      * ``force=True`` (operator's explicit override), OR
      * the compiled artifact does not yet exist (cold-start bootstrap), OR
      * at least one accepted candidate exists in a manifest whose mtime
        is newer than the artifact's mtime.

    A candidate is "accepted" iff every one of the four required
    fill-ins (``expected_causal_role``, ``expected_remediation``,
    ``derivation_pseudocode``, ``dataset_context``) is non-null on its
    JSON manifest row. See
    ``scripts/check_compile_set_candidate_backlog.py`` for the
    canonical definition and the standalone CLI.

    Rationale (issue #236): without this gate, an operator who forgets
    to hand-merge accepted candidates into ``build_compile_set()`` will
    silently produce a new compiled artifact that's a byte-equivalent
    copy of the old one. The gate forces either evidence-driven
    recompiles or an explicit ``--force`` acknowledgement.
    """
    if force:
        return True, "preflight: --force passed; bypassing backlog check"

    if not compiled_artifact_path.exists():
        return True, (
            f"preflight: no prior artifact at {compiled_artifact_path} (bootstrap path); proceeding"
        )

    result = count_accepted_candidates(
        candidates_dir=candidates_dir,
        compiled_artifact_path=compiled_artifact_path,
        logger=logger,
    )
    if result.count == 0:
        return False, (
            "preflight: backlog is zero — no new accepted candidates in "
            f"{candidates_dir} since artifact mtime. Run "
            "`make curate-candidates`, fill in the 4 required fields "
            "per accepted row, or pass --force to override."
        )
    return True, (
        f"preflight: {result.count} accepted candidate(s) in backlog since last compile; proceeding"
    )


def _seed_all(seed: int) -> None:
    """Seed every random source ``BootstrapFewShot`` touches.

    DSPy's bootstrap reshuffles the trainset and picks the bootstrapped demos
    via ``random.sample`` / ``random.shuffle``. Pinning ``random.seed`` and
    ``numpy.random.seed`` makes the demo-selection step reproducible for any
    fixed LM. (LM responses themselves can still vary across runs depending on
    provider-side nondeterminism; see the script docstring's note on the
    persistence vs runtime trade-off.)
    """
    random.seed(seed)
    np.random.seed(seed)
    # DSPy reads the env var ``DSPY_RANDOM_SEED`` in some internal codepaths;
    # set it too so any newly-added randomness picks the pinned seed.
    os.environ.setdefault("DSPY_RANDOM_SEED", str(seed))


def _compile_with_miprov2(
    *,
    program: Any,
    trainset: list[Any],
    seed: int,
    max_labeled_demos: int = MIPROV2_DEFAULT_MAX_LABELED_DEMOS,
    max_bootstrapped_demos: int = DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    auto: str = "light",
) -> Any:
    """Plan-239 §5.1 — compile via MIPROv2 with seed threaded into BOTH
    constructor AND .compile() call (belt-and-suspenders).

    Per §0/V15 + §0/V21 + §9.2 R7: MIPROv2 auto-splits trainset 80/20 if
    no `valset` is passed, producing a train of 10 + val of 40 at n=50 (the
    train shrinks because internal logic flips, but the default
    `minibatch_size=35` would still trip on val=10). We pass `valset`
    explicitly (40 train / 10 val) AND set `minibatch=False` so the
    optimizer runs full-eval on the small val set.

    Returns the compiled program with `metadata['dspy_version']` recorded
    (plan-239 §5.2) for post-upgrade reproducibility debugging.
    """
    from dspy.teleprompt import MIPROv2

    teleprompter = MIPROv2(
        metric=_exact_match_metric,
        seed=seed,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        auto=auto,
    )

    # Explicit train/val split (plan-239 §5.4 R7). 80/20 floor at n=50.
    n_total = len(trainset)
    if n_total >= 5:
        rng = random.Random(seed)
        shuffled = list(trainset)
        rng.shuffle(shuffled)
        val_size = max(1, n_total // 5)
        valset = shuffled[:val_size]
        train_subset = shuffled[val_size:]
    else:
        # Degenerate path used by unit tests with empty/tiny trainsets;
        # let MIPROv2 do its own thing (it will error out on n_total=0
        # which is fine for the wiring-only test).
        valset = None
        train_subset = trainset

    compiled = teleprompter.compile(
        program,
        trainset=train_subset,
        valset=valset,
        seed=seed,
        minibatch=False,
    )

    # Plan-239 §5.2: record DSPy version on the artifact for post-upgrade
    # reproducibility debugging. metadata is a dict-like attribute on
    # dspy.Module subclasses.
    try:
        import dspy

        existing = getattr(compiled, "metadata", None) or {}
        existing.update({"dspy_version": dspy.__version__})
        compiled.metadata = existing
    except Exception as exc:  # pragma: no cover - non-critical
        logger.warning("MIPROv2: could not record dspy_version metadata (%s)", exc)

    return compiled


def _exact_match_metric(example, prediction, _trace=None) -> bool:
    """Bootstrap metric: prediction's ``causal_role`` matches the labeled role.

    BootstrapFewShot uses this to decide whether a teacher-bootstrapped demo
    is kept as a few-shot exemplar. Strict-equality on ``causal_role`` is the
    discriminating signal — ``mechanism`` and ``recommended_remediation`` are
    free-text / dependent-output that vary even when the role is correct.

    ``example`` and ``prediction`` are duck-typed ``dspy.Example`` /
    ``dspy.Prediction`` instances. We intentionally do NOT annotate the
    type so the module imports cleanly without DSPy installed (issue
    #236 pre-flight helper path).
    """
    expected = getattr(example, "causal_role", None)
    actual = getattr(prediction, "causal_role", None)
    return expected is not None and actual is not None and expected == actual


def _configure_lm(model: str, max_tokens: int) -> None:
    """Configure the DSPy default LM endpoint.

    Reads provider credentials from environment variables. We do NOT
    hardcode API keys here; the convention follows ``.env`` / direct env
    population. If no key is present the LM call fails loudly (rather than
    silently degrading to a no-op), which is the desired behaviour for a
    compile script — failure should be visible, not buried.
    """
    import dspy

    lm = dspy.LM(model, max_tokens=max_tokens)
    dspy.configure(lm=lm)


def compile_and_persist(
    *,
    out_path: Path,
    lm_model: str | None,
    max_tokens: int = 1024,
    seed: int = DEFAULT_SEED,
    max_bootstrapped_demos: int = DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    max_labeled_demos: int = DEFAULT_MAX_LABELED_DEMOS,
    optimizer: str = "bootstrap",
) -> Path:
    """Compile the classifier and persist the compiled program JSON.

    Args:
        out_path: File path to write the compiled program JSON to. Parent
            directories are created if missing.
        lm_model: DSPy LM model string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
            If ``None``, no LM is configured and ``BootstrapFewShot`` is
            skipped — the script falls through to ``classifier.save(...)``
            on the un-compiled classifier so the persisted JSON still
            carries the program shape. This is the CI fallback path.
        max_tokens: ``dspy.LM`` max-tokens cap. Plays no role when ``lm_model
            is None``.
        seed: PRNG seed pinned via :func:`_seed_all` for deterministic demo
            selection.
        max_bootstrapped_demos: Cap on teacher-generated demos. BootstrapFewShot
            default is 4; keeping it low so the compile run stays cheap.
        max_labeled_demos: Cap on labeled (compile-set) demos retained as
            few-shot exemplars. Default 40 > ``len(build_compile_set()) == 33``
            so every labeled exemplar (including all 4 collider, 6
            instrument, and 12 Phase-4 S12 Option C paired-fixture
            demos) survives the random.sample step inside
            BootstrapFewShot._train (raised from 8 -> 16 on codex
            pass-3, 16 -> 24 on codex pass-4 after the artifact-pin
            audit found that random.sample at 16 was routinely dropping
            the provider IV exemplars, and 24 -> 40 on Phase-4 S12
            Option C to accommodate the 12 new paired-fixture demos
            whose individual loss would trip the §3.5 falsifiability
            quadruple gate).

    Returns:
        The path the compiled program JSON was written to (mirror of
        ``out_path``).
    """
    # Deferred so the lightweight ``preflight_candidate_check`` helper
    # (issue #236) can run without the full DSPy + classifier dep stack.
    from dspy.teleprompt import BootstrapFewShot

    from src.data.causal_role_classifier import (
        CausalRoleClassifier,
        build_compile_set,
    )

    _seed_all(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    student = CausalRoleClassifier()

    if lm_model is None:
        logger.warning(
            "compile_causal_role_classifier: --no-lm path active; "
            "skipping BootstrapFewShot and persisting un-compiled program. "
            "The persisted JSON carries the signature but the LM-driven "
            "few-shot demos are absent. Production use requires --lm-model."
        )
        student.save(str(out_path))
        return out_path

    _configure_lm(lm_model, max_tokens)

    trainset = build_compile_set()

    if optimizer == "miprov2":
        # Plan-239 AC1+AC2: MIPROv2 path. Default labeled-demos cap raised
        # to MIPROV2_DEFAULT_MAX_LABELED_DEMOS when the caller did not
        # explicitly override DEFAULT_MAX_LABELED_DEMOS.
        effective_labeled = (
            MIPROV2_DEFAULT_MAX_LABELED_DEMOS
            if max_labeled_demos == DEFAULT_MAX_LABELED_DEMOS
            else max_labeled_demos
        )
        compiled = _compile_with_miprov2(
            program=student,
            trainset=trainset,
            seed=seed,
            max_labeled_demos=effective_labeled,
            max_bootstrapped_demos=max_bootstrapped_demos,
        )
    elif optimizer == "bootstrap":
        teleprompter = BootstrapFewShot(
            metric=_exact_match_metric,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            max_rounds=1,
        )
        compiled = teleprompter.compile(student=student, trainset=trainset)
    else:
        raise ValueError(
            f"Unknown --optimizer {optimizer!r}; choose 'bootstrap' or 'miprov2' (plan-239 AC1)."
        )

    compiled.save(str(out_path))
    logger.info("compile_causal_role_classifier: optimizer=%s wrote %s", optimizer, out_path)
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Output path for compiled program JSON (default: {DEFAULT_OUT_PATH}).",
    )
    parser.add_argument(
        "--lm-model",
        type=str,
        default=DEFAULT_LM_MODEL,
        help=(
            "DSPy LM model string (e.g. anthropic/claude-sonnet-4-20250514). "
            f"Default: {DEFAULT_LM_MODEL}."
        ),
    )
    parser.add_argument(
        "--no-lm",
        action="store_true",
        help=(
            "Skip the LM-driven compile entirely; persist the un-compiled "
            "program shape only. CI fallback when no API key is available."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="dspy.LM max-tokens cap per request. Default: 1024.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "PRNG seed pinned via random.seed + numpy.random.seed for "
            "deterministic demo selection (BootstrapFewShot.random.sample). "
            "Also installed into DSPY_RANDOM_SEED via os.environ.setdefault "
            "— if the env var is already set in the environment the "
            "existing value wins, so set the env var explicitly for "
            f"full determinism. Default: {DEFAULT_SEED}."
        ),
    )
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
        help=(
            "Cap on teacher-bootstrapped demos appended to the persisted "
            "few-shot set. The aggregate persisted demo count is "
            "bounded by max_labeled_demos + max_bootstrapped_demos = "
            f"{DEFAULT_MAX_LABELED_DEMOS} + {DEFAULT_MAX_BOOTSTRAPPED_DEMOS} "
            f"= {DEFAULT_MAX_LABELED_DEMOS + DEFAULT_MAX_BOOTSTRAPPED_DEMOS} "
            "(Phase-4 S12 Option C; covers the 33-example compile set + "
            f"slot headroom). Default: {DEFAULT_MAX_BOOTSTRAPPED_DEMOS}."
        ),
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=DEFAULT_MAX_LABELED_DEMOS,
        help=(
            "Cap on labeled (compile-set) demos retained as persisted "
            "few-shot exemplars. BootstrapFewShot._train calls "
            "random.sample(demos, max_labeled_demos); raising this above "
            "len(build_compile_set()) keeps every labeled exemplar. "
            f"Default: {DEFAULT_MAX_LABELED_DEMOS} (set on Phase-4 S12 "
            "Option C recompile to accommodate the 33-example compile "
            "set: 21 legacy + 12 (T, Y)-explicit paired-fixture demos)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging.basicConfig level for this script. Default: INFO.",
    )
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=DEFAULT_CANDIDATES_DIR,
        help=(
            "Directory of curate_compile_set_candidates JSON manifests "
            "for the Phase 4.5 backlog pre-flight (issue #236). "
            f"Default: {DEFAULT_CANDIDATES_DIR}."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the Phase 4.5 backlog pre-flight. Use only when "
            "you explicitly want to recompile without new accepted "
            "candidates (e.g. determinism re-run, hyperparameter "
            "experiment)."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=("bootstrap", "miprov2"),
        default="bootstrap",
        help=(
            "DSPy teleprompter to use (plan-239 AC1). `bootstrap` is "
            "the legacy BootstrapFewShot path; `miprov2` is the MIPROv2 "
            "path with seed threaded into both constructor and compile() "
            "and explicit 80/20 train/val split (plan-239 §5.1-§5.4). "
            "Default: bootstrap (preserves prior behavior on Branch B)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    proceed, preflight_message = preflight_candidate_check(
        candidates_dir=args.candidates_dir,
        compiled_artifact_path=args.out,
        force=args.force,
    )
    if not proceed:
        # Print to stderr with a unique prefix so the refusal is grep-able
        # and falsifiability tests can distinguish "gate fired" from
        # "import failure on a downstream step."
        print(
            f"REFUSED: compile_causal_role_classifier pre-flight blocked: {preflight_message}",
            file=sys.stderr,
        )
        return 1
    logger.info("%s", preflight_message)

    compile_and_persist(
        out_path=args.out,
        lm_model=None if args.no_lm else args.lm_model,
        max_tokens=args.max_tokens,
        seed=args.seed,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        optimizer=args.optimizer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
