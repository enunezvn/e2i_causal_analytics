"""Cohort-identity → feature-manifest-source resolution (Layer-5 opt-in).

Single source of truth for deciding which feature manifest a tier-0 run should
consult. Shared by:

  - the operator step-runner scripts (``run_tier0_test`` / ``run_optum_tier0_test``),
  - the ``MLFoundationPipeline`` scope-definition stage (the programmatic /
    live-ready origin), and
  - the live retraining trigger (``src/tasks/drift_monitoring_tasks.py``).

The strictness contract (originally written for the CSU runner — codex M1/M2/M3,
2026-05-08) is preserved here, but keyed off the canonical ``MANIFEST_SOURCES``
registry instead of a hardcoded tuple so adding a manifest source is a single
edit in ``__init__.py``.

Opt-in safety is load-bearing: when neither an explicit override nor a
recognizable ``data_source`` segment is supplied the resolver returns ``None``,
so synthetic / research regimes never apply a cross-cohort manifest and emit
false-positive Layer-1 verdicts against an unrelated cohort's vocabulary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def known_manifest_sources() -> tuple[str, ...]:
    """The set of registered manifest sources, derived from the registry.

    Imported lazily to avoid any import-time cycle with the package
    ``__init__`` that defines ``MANIFEST_SOURCES`` (and imports the per-cohort
    manifests).
    """
    from src.data.manifests import MANIFEST_SOURCES

    return tuple(MANIFEST_SOURCES.keys())


def autodetect_manifest_source(data_source: str | dict[str, Any] | None) -> set[str]:
    """Return the registered manifest sources that appear as path segments in
    ``data_source`` (case-insensitive).

    Returns the full *set* of matches (not the first) so the caller can detect
    ambiguity rather than silently pick one by iteration order. A ``None`` /
    empty ``data_source``, a non-string ``data_source`` (e.g. the
    ``{"type": "file_dir", "path": ...}`` dict MLFoundationPipeline accepts for
    file batches), or a bare table name with no recognizable segment yields the
    empty set — path autodetection only applies to string paths.
    """
    if not data_source or not isinstance(data_source, str):
        return set()
    known = set(known_manifest_sources())
    parts = {p.lower() for p in Path(data_source).parts}
    # Match a registered source when a path segment EQUALS it OR is an
    # underscore-delimited variant of it (``optum_gap_enriched`` -> ``optum``).
    # A gap-enriched / derived extract is still the same cohort and must consult
    # the same FeatureContract — otherwise Layer 1 declared-safe never fires and
    # the statistical leakage layer over-drops legitimate pre-index predictors
    # (leakage over-drop investigation, 2026-06-03). The ``<source>_`` prefix
    # shape (not a bare ``startswith``) prevents partial-word false positives
    # like ``optumistic`` matching ``optum``, and preserves the M1 ambiguity
    # contract (two distinct sources matching different segments still raises).
    return {s for s in known if any(p == s or p.startswith(f"{s}_") for p in parts)}


def resolve_manifest_source(
    data_source: str | dict[str, Any] | None,
    override: str | None = None,
) -> str | None:
    """Resolve which cohort manifest Layer 5 should consult on this run.

    Priority: explicit ``override`` > auto-detection from ``data_source`` path
    segments > unset (``None``).

    Strictness contract:

    - **M1 ambiguous data_source**: a path containing more than one known
      source segment (e.g. ``data/rwd/csu/optum_baseline``) raises
      ``ValueError`` — the caller must disambiguate via ``override``.
    - **M2 override conflict**: an ``override`` that contradicts what the
      ``data_source`` auto-detects raises ``ValueError`` (prevents silently
      measuring, e.g., Optum data against CSU contracts).
    - **Unknown override**: an ``override`` that names no registered manifest
      source raises ``ValueError`` (fail loud — a caller typo must not silently
      no-op the manifest).
    - **M3 unmatched / unset**: a ``data_source`` with no known segment and no
      override returns ``None`` (with a debug log) so synthetic / ad-hoc runs
      stay frictionless and manifest-free.
    """
    known = set(known_manifest_sources())
    detected = autodetect_manifest_source(data_source)

    if len(detected) > 1:
        raise ValueError(
            f"data_source={data_source!r} contains multiple known manifest "
            f"sources {sorted(detected)} as path segments — auto-detection is "
            f"ambiguous. Pass an explicit feature_manifest_source to disambiguate."
        )

    if override is not None:
        if override not in known:
            raise ValueError(
                f"feature_manifest_source={override!r} is not a registered "
                f"manifest source {sorted(known)}; pass one of those or leave it "
                f"unset to skip the manifest pass."
            )
        if detected and override not in detected:
            raise ValueError(
                f"feature_manifest_source={override!r} conflicts with "
                f"data_source={data_source!r} which auto-detects to "
                f"{sorted(detected)}. The conflict suggests the wrong manifest "
                f"is being applied (e.g. Optum data measured against CSU "
                f"contracts). Rename the data_source to remove the misleading "
                f"segment, or run the data through the matching converter first."
            )
        return override

    if len(detected) == 1:
        return next(iter(detected))

    if data_source:
        logger.debug(
            "data_source=%r contains no known manifest source segment %s; "
            "Layer 5 manifest verdicts will not fire for this run. Pass an "
            "explicit feature_manifest_source if a registry should be consulted.",
            data_source,
            sorted(known),
        )
    return None
