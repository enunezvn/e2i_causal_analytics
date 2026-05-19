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
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# Allow `python scripts/compile_causal_role_classifier.py` without `python -m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# Issue #198 codex pass-4 MED-1: raised from 16 -> 24 so all 20 labeled
# compile-set examples survive the random.sample step inside
# BootstrapFewShot._train (which caps `augmented_demos + raw_demos` at
# max_labeled_demos). With max_labeled_demos=24 and 20 examples + 4
# bootstrapped, every labeled exemplar — including both provider IV
# variants (provider_preference_score and
# index_provider_biologic_volume_prior_year) — is preserved in the
# persisted few-shot demos. Pass-3 set this to 16 which dropped 4-8
# labeled examples randomly; pass-4 audit found this routinely dropped
# the provider IV family entirely.
#
# Phase-4 S12 Option C recompile (2026-05-19): raised from 24 -> 40 so
# all 33 labeled compile-set examples (21 legacy + 12 new (T, Y)-
# explicit paired-fixture demos per `.claude/plans/option_c_dspy_recompile_for_s12_FINAL.md`)
# survive the random.sample step inside BootstrapFewShot._train.
# Computed as 33 labeled + 4 bootstrapped = 37, +3 slot conservative
# headroom for any future small additions. The §3.5 paired-fixture
# falsifiability gate requires all 12 quadruples to land in the
# persisted artifact: if any single (T, Y) variant gets dropped by
# random.sample (likelihood ~12/40 = 30% per variant at cap=24 vs ~0
# at cap=40 since 33 < 40), the gate trips. The new ceiling pins this
# to ~0% loss probability.
DEFAULT_MAX_LABELED_DEMOS = 40


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

    teleprompter = BootstrapFewShot(
        metric=_exact_match_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=1,
    )

    compiled = teleprompter.compile(student=student, trainset=trainset)
    compiled.save(str(out_path))
    logger.info("compile_causal_role_classifier: wrote %s", out_path)
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-bootstrapped-demos",
        type=int,
        default=DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    )
    parser.add_argument(
        "--max-labeled-demos",
        type=int,
        default=DEFAULT_MAX_LABELED_DEMOS,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
