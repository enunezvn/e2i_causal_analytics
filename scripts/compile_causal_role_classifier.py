"""Phase 2.5 — DSPy compile-and-persist for the Layer 4 CausalRoleClassifier.

Compiles ``src.data.causal_role_classifier.CausalRoleClassifier`` via
``BootstrapFewShot`` against the 12-example compile set produced by
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

import dspy  # noqa: E402
from dspy.teleprompt import BootstrapFewShot  # noqa: E402

from src.data.causal_role_classifier import (  # noqa: E402
    CausalRoleClassifier,
    build_compile_set,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = PROJECT_ROOT / "artifacts" / "dspy" / "causal_role_classifier.json"
DEFAULT_LM_MODEL = "anthropic/claude-sonnet-4-20250514"
DEFAULT_SEED = 7
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 4
DEFAULT_MAX_LABELED_DEMOS = 8


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


def _exact_match_metric(example: dspy.Example, prediction: dspy.Prediction, _trace=None) -> bool:
    """Bootstrap metric: prediction's ``causal_role`` matches the labeled role.

    BootstrapFewShot uses this to decide whether a teacher-bootstrapped demo
    is kept as a few-shot exemplar. Strict-equality on ``causal_role`` is the
    discriminating signal — ``mechanism`` and ``recommended_remediation`` are
    free-text / dependent-output that vary even when the role is correct.
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
            few-shot exemplars. Default 8 ≤ ``len(build_compile_set()) == 12``.

    Returns:
        The path the compiled program JSON was written to (mirror of
        ``out_path``).
    """
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

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
