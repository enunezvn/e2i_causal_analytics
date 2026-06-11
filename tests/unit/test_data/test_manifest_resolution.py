"""Phase B: shared cohort-identity → manifest-source resolver.

Single source of truth for resolving which feature manifest (Layer-5 opt-in) a
run should consult, shared by the tier-0 operator scripts, the
``MLFoundationPipeline`` scope stage, and the live retraining trigger. Encodes
the M1 (ambiguous) / M2 (override conflict) / M3 (unmatched → None) strictness
contract originally written for the CSU runner, keyed off the canonical
``MANIFEST_SOURCES`` registry so adding a source is one edit.
"""

from __future__ import annotations

import pytest

from src.data.manifests import MANIFEST_SOURCES
from src.data.manifests.resolution import (
    autodetect_manifest_source,
    known_manifest_sources,
    resolve_manifest_source,
)


def test_known_sources_track_the_registry() -> None:
    """The resolver's known set IS the registry — no hardcoded duplicate."""
    assert set(known_manifest_sources()) == set(MANIFEST_SOURCES)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("data/rwd/optum/initiation", {"optum"}),
        ("data/rwd/csu", {"csu"}),
        ("data/synthetic/foo", {"synthetic"}),
        ("/abs/data/rwd/optum/persistence", {"optum"}),
        ("data/rwd/unknown_cohort", set()),
        ("", set()),
        # Underscore-delimited variants of a registered source resolve to that
        # source: a gap-enriched / derived Optum extract is still Optum and must
        # consult the Optum FeatureContract (else Layer 1 declared-safe never
        # fires and the statistical layer over-drops legitimate pre-index
        # predictors). See the leakage over-drop investigation, 2026-06-03.
        ("data/rwd/optum_gap_enriched/initiation", {"optum"}),
        ("data/rwd/csu_v2/x", {"csu"}),
        # Guard: a partial word that merely STARTS with a source's letters but
        # is not an underscore-delimited variant must NOT match (no false
        # positives — only the `<source>_...` shape counts).
        ("data/rwd/optumistic/x", set()),
    ],
)
def test_autodetect_from_path_segments(path: str, expected: set[str]) -> None:
    assert autodetect_manifest_source(path or None) == expected


def test_autodetect_none_data_source() -> None:
    assert autodetect_manifest_source(None) == set()


def test_autodetect_non_str_data_source_is_safe() -> None:
    # MLFoundationPipeline accepts a dict data_source for file_dir cohorts
    # ({"type": "file_dir", "path": ...}). The resolver must not crash on it
    # (Path(dict) would raise) — autodetect simply yields no segment match.
    assert autodetect_manifest_source({"type": "file_dir", "path": "data/rwd/optum/x"}) == set()


def test_resolve_dict_data_source_with_override() -> None:
    # A file_dir dict data_source + explicit override (the live-trigger path)
    # must resolve to the override without attempting path autodetection.
    ds = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    assert resolve_manifest_source(ds, "optum") == "optum"


def test_resolve_dict_data_source_without_override_is_unset() -> None:
    ds = {"type": "file_dir", "path": "data/rwd/optum/initiation"}
    assert resolve_manifest_source(ds, None) is None


def test_resolve_autodetects_optum() -> None:
    assert resolve_manifest_source("data/rwd/optum/initiation", None) == "optum"


def test_resolve_explicit_override_without_data_source() -> None:
    # The live trigger resolves cohort identity to an explicit override and
    # passes it without a path-shaped data_source.
    assert resolve_manifest_source(None, "optum") == "optum"


def test_resolve_unknown_path_stays_unset() -> None:
    # Opt-in safety: an unrecognised path must NOT apply a manifest (no
    # cross-cohort false positives).
    assert resolve_manifest_source("data/rwd/mystery", None) is None


def test_resolve_no_signal_stays_unset() -> None:
    assert resolve_manifest_source(None, None) is None


def test_resolve_override_agreeing_with_autodetect() -> None:
    assert resolve_manifest_source("data/rwd/optum", "optum") == "optum"


def test_resolve_conflicting_override_fails_fast() -> None:
    # M2: csu override against an optum data_dir must raise, not silently apply
    # the wrong manifest.
    with pytest.raises(ValueError, match="conflicts with"):
        resolve_manifest_source("data/rwd/optum/initiation", "csu")


def test_resolve_ambiguous_data_source_fails_fast() -> None:
    # M1: a path with BOTH csu and optum as full path segments is ambiguous
    # (segment-exact match; "optum_baseline" would NOT match "optum").
    with pytest.raises(ValueError, match="ambiguous|multiple"):
        resolve_manifest_source("data/rwd/csu/optum/baseline", None)


def test_resolve_unknown_override_fails_fast() -> None:
    # An override that names no registered manifest is a caller error — fail
    # loud rather than silently no-op (anti-mocking).
    with pytest.raises(ValueError, match="not a registered|unknown|registered manifest"):
        resolve_manifest_source(None, "bogus")


@pytest.mark.parametrize("source", sorted(MANIFEST_SOURCES))
def test_every_registered_source_round_trips_as_override(source: str) -> None:
    """Wiring guard (Phase C): every manifest source registered in
    MANIFEST_SOURCES must be a valid override — i.e. resolvable. Auto-extends
    with the registry, so adding a source that the resolver would reject (or
    forgetting to register one a runner already passes) trips here."""
    assert resolve_manifest_source(None, source) == source


def test_resolve_ambiguous_path_with_valid_override_defers_to_override() -> None:
    """M1's own error message instructs 'pass an explicit feature_manifest_source
    to disambiguate' — so an ambiguous path WITH a valid in-set override must
    resolve to the override, not raise. Real case: data/rwd/synthetic_CSU/...
    segments match both 'synthetic' (prefix rule) and 'synthetic_csu' (exact)."""
    assert (
        resolve_manifest_source("data/rwd/synthetic_CSU/tier0/initiation", "synthetic_csu")
        == "synthetic_csu"
    )


def test_resolve_ambiguous_path_without_override_still_raises() -> None:
    with pytest.raises(ValueError, match="ambiguous|multiple"):
        resolve_manifest_source("data/rwd/synthetic_CSU/tier0/initiation", None)
