"""Every activated KG cache must actually reach the Docker image (#1607).

The KG Layer-2 activation shipped correct and still did nothing in production:
`.dockerignore` excluded `data/` wholesale, so the committed cache never entered
the build context, and `docker/Dockerfile` never COPYed it. Measured on the
deployed container: `/app/data/kg_cache/` did not exist, `apply_kg_activation`
returned False, and KG stayed off.

That is the #600 shape — a gitignored-by-default artifact that IS committed but
is absent where it is loaded, so the failure presents as "the KG had nothing to
say" rather than as a missing file. The activation code fails LOUD (it logs an
ERROR naming the path), which is what made this findable at all, but a loud
no-op is still a no-op.

These tests are static: they read `.dockerignore` and the Dockerfile rather than
building an image, so they run in the normal unit lane. They exist because
neither the unit suite nor CI could otherwise tell that a shipped feature was
inert.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.data.kg.activation import KG_ACTIVATIONS

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_DOCKERFILE = _REPO_ROOT / "docker" / "Dockerfile"


def _dockerignore_excludes(rel_path: str) -> bool:
    """Resolve `rel_path` against .dockerignore with last-match-wins semantics.

    Docker evaluates every pattern in order and the LAST match decides, which is
    exactly the rule this bug turned on: an un-ignore placed next to `data/`
    reads correctly but is silently undone by the later `*.json` and `*.md`
    lines. A simplified matcher is enough here because we only ask about the
    handful of concrete cache paths, and it keeps the guard dependency-free.
    """
    excluded = False
    for raw in _DOCKERIGNORE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        if _matches(pattern, rel_path):
            excluded = not negate
    return excluded


def _matches(pattern: str, rel_path: str) -> bool:
    """True when a .dockerignore pattern matches `rel_path` or a parent dir."""
    pattern = pattern.rstrip("/")
    if not pattern:
        return False
    # `**` spans separators; `*` does not.
    regex = re.escape(pattern).replace(r"\*\*", "\x00").replace(r"\*", "[^/]*")
    regex = regex.replace("\x00", ".*").replace(r"\?", "[^/]")
    # A directory pattern also covers everything beneath it.
    if re.fullmatch(regex, rel_path):
        return True
    return bool(re.fullmatch(regex + "(/.*)?", rel_path))


@pytest.mark.parametrize("activation_key", sorted(KG_ACTIVATIONS))
def test_activated_cache_is_in_the_docker_build_context(activation_key: str) -> None:
    """Every KG_ACTIVATIONS artifact must survive .dockerignore."""
    filename = KG_ACTIVATIONS[activation_key].cache_filename
    rel = f"data/kg_cache/{filename}"

    assert (_REPO_ROOT / rel).is_file(), f"{rel} is configured but not committed"
    assert not _dockerignore_excludes(rel), (
        f"{rel} is excluded from the Docker build context, so KG Layer 2 will be "
        "inert in the image while looking merely quiet at runtime. Re-include it "
        "AFTER the *.json rule in .dockerignore (last-match-wins)."
    )


def test_dockerfile_copies_the_kg_cache_into_every_app_stage() -> None:
    """Being in the build context is not enough — it must be COPYed.

    Both the development and production stages need it: they are separate
    `FROM`s and production does not inherit the dev stage's layers.
    """
    text = _DOCKERFILE.read_text()
    copies = [ln for ln in text.splitlines() if ln.startswith("COPY") and "kg_cache" in ln]
    assert len(copies) >= 2, (
        "expected a kg_cache COPY in BOTH the development and production stages "
        f"of docker/Dockerfile; found {len(copies)}: {copies}"
    )


def test_the_summary_report_is_not_silently_dropped() -> None:
    """The .summary.md companion must survive the top-of-file `*.md` rule.

    It is how an operator checks what a cache actually contains without parsing
    72 KB of JSON, and `*.md` is excluded near the top of .dockerignore — so it
    only survives because the un-ignore sits later in the file.
    """
    for activation in KG_ACTIVATIONS.values():
        summary = f"data/kg_cache/{Path(activation.cache_filename).stem}.summary.md"
        if not (_REPO_ROOT / summary).is_file():
            continue
        assert not _dockerignore_excludes(summary), (
            f"{summary} is dropped by the `*.md` exclusion — move the kg_cache "
            "un-ignore later in .dockerignore"
        )


# --------------------------------------------------------------- #1623: staleness


def test_configured_cache_filename_still_matches_the_live_manifest() -> None:
    """The cache name must describe the manifest that exists TODAY (#1623).

    The filename embeds fingerprints of both the feature manifest and the target
    entity codes, and ``KG_ACTIVATIONS`` names it as a literal. Edit
    ``OPTUM_FEATURES`` — add a feature, change one ``kg_entity_codes`` entry — and
    the manifest fingerprint changes while the literal does not.

    Nothing else catches that. The packaging guard above proves the artifact reaches
    the image, and ``test_kg_layer2_activation.py`` reads the cache *by its
    configured name*, so both stay green while the name has stopped describing the
    manifest. At runtime the drift IS detected, but in ``shadow`` mode it logs and
    skips — so the symptom is every feature returning ``no_signal``, which is
    exactly the "the KG had nothing to say" ambiguity #1607 exists to eliminate.
    """
    from src.data.kg.cache import (
        compose_cache_filename,
        compute_manifest_fingerprint,
        compute_target_codes_fingerprint,
    )
    from src.data.manifests.optum_feature_manifest import OPTUM_FEATURES

    activation = KG_ACTIVATIONS["optum"]
    expected = compose_cache_filename(
        compute_manifest_fingerprint(OPTUM_FEATURES),
        compute_target_codes_fingerprint(tuple(tuple(t) for t in activation.target_entity_codes)),
    )

    assert expected == activation.cache_filename, (
        f"KG_ACTIVATIONS['optum'] points at {activation.cache_filename!r}, but the "
        f"current OPTUM_FEATURES + target codes fingerprint to {expected!r}. The "
        "manifest or the target changed without the cache being rebuilt, so KG "
        "Layer 2 will go quietly dark (shadow mode logs the mismatch and skips). "
        "Rebuild with scripts/build_kg_cache.py --live (see docs/runbooks/kg_cache.md), "
        "commit the new artifact, add it to the .gitignore un-ignore, and update "
        "KG_ACTIVATIONS."
    )
