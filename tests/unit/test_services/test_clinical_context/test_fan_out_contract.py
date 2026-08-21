"""Network-free contract guard for ``ClinicalContextService._fan_out`` (#1766).

WHY THIS FILE EXISTS — the defect class, not the instance
---------------------------------------------------------
#1763 moved the literature citation out of ``_fan_out`` into ``_citation_for``,
dropping the fan-out from five fragments to four. The only caller that unpacks
``_fan_out`` positionally outside the service is
``tests/integration/test_clinical_context/test_fan_out_degradation_signal.py``
— the #1612 AC4 degradation signal — and that module is ``slow`` +
``requires_network``, so the PR-blocking lane deselects it (``-m "not slow"``)
and only the 07:00 UTC ``slow-tests.yml`` Job A ever executes it. The #1763 train
therefore went green on every PR while leaving a five-way unpack against a
four-tuple, and main was red on the very next nightly (#1766).

Two cheap disproofs were run before writing this guard:

* **"mypy would have caught it."** FALSE. CI type-checks ``src/`` only
  (``backend-tests.yml``: ``mypy --config-file pyproject.toml src/``), and that
  gate is a *ceiling* count, not a zero-error gate. ``tests/`` is never checked.
* **"a stub-based arity test would have caught it."** FALSE on its own. The
  #1763 author changed ``_fan_out``; any arity test they owned would have gone
  red in front of them and been updated in the same commit, while the
  network-gated call site stayed stale. Pinning the signature is necessary but
  not sufficient — the drift is *between a nightly-only call site and
  production*, so the guard has to inspect call sites it does not execute.

Hence both layers below: a live-behaviour pin on the seam (stubs, no network),
and an AST sweep that reads every ``_fan_out`` call site in the repo — including
the ones no PR-lane run will ever execute.

Nothing here touches the network or fakes a live provider result: it asserts a
structural contract. The live degradation signal itself stays live.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, List, Tuple, get_args

import pytest

from src.services.clinical_context.brand_map import resolve_brand_profile
from src.services.clinical_context.clients import CTGovEndpoint, PubMedArticle
from src.services.clinical_context.providers import (
    CitationFragment,
    CompetitorFragment,
    EndpointsFragment,
    IndicationsFragment,
    MechanismFragment,
)
from src.services.clinical_context.service import (
    _BrandFragmentTuple,
    ClinicalContextService,
    reset_caches,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCAN_ROOTS = ("src", "tests", "scripts")
_SEAM = "_fan_out"

# The nightly degradation signal (#1612 AC4) — the call site that drifted. It is
# named explicitly as the POSITIVE CONTROL for the AST sweep: an empty sweep
# would otherwise pass vacuously if the scan roots, the glob or the seam name
# ever stopped matching, which is exactly how a guard rots into decoration.
_DEGRADATION_SIGNAL = (
    _REPO_ROOT
    / "tests"
    / "integration"
    / "test_clinical_context"
    / "test_fan_out_degradation_signal.py"
)


class _StubProvider:
    """Returns a fixed fragment. No network, no plausible-but-fake live values —
    every source label below is one the real code emits for a *stub*."""

    def __init__(self, fragment: object, name: str = "stub") -> None:
        self._fragment = fragment
        self.provider_name = name

    def enrich(self, profile: object) -> object:
        return self._fragment


def _stub_service() -> ClinicalContextService:
    return ClinicalContextService(
        mechanism_provider=_StubProvider(MechanismFragment("CDK4/6 inhibitor", "chembl")),
        endpoints_provider=_StubProvider(
            EndpointsFragment([CTGovEndpoint("PFS")], "clinicaltrials.gov")
        ),
        citation_provider=_StubProvider(
            CitationFragment(
                citation=PubMedArticle(pmid="1", title="stub", journal="j", pubdate="2026"),
                source="pubmed",
            )
        ),
        indications_provider=_StubProvider(
            IndicationsFragment(approved_indications=["HR+ breast cancer"], source="openfda")
        ),
        competitor_provider=_StubProvider(CompetitorFragment(["Ibrance"], 1, "curated")),
    )


@pytest.fixture(autouse=True)
def _clear() -> None:
    reset_caches()


# ============================================================ the seam contract


def test_fan_out_returns_exactly_the_declared_fragment_tuple() -> None:
    """``_fan_out``'s runtime shape must equal ``_BrandFragmentTuple``.

    Derived from the alias, never hardcoded: hardcoding ``4`` here would make
    this test agree with a future drift instead of catching it.
    """
    declared = get_args(_BrandFragmentTuple)
    fragments = _stub_service()._fan_out(resolve_brand_profile("Kisqali"))

    assert isinstance(fragments, tuple)
    assert len(fragments) == len(declared), (
        f"_fan_out returned {len(fragments)} fragments but _BrandFragmentTuple "
        f"declares {len(declared)}. Every positional unpack in the tree — including "
        "the network-gated nightly degradation signal — must move with it."
    )
    assert [type(f) for f in fragments] == list(declared)


def test_fan_out_return_annotation_stays_bound_to_the_declared_alias() -> None:
    """The alias is only a contract while the function is annotated with it.

    Without this, someone could widen ``_fan_out``'s return to a bare ``tuple``
    and the arity guard above would keep passing against a stale alias.
    """
    assert ClinicalContextService._fan_out.__annotations__["return"] == "_BrandFragmentTuple"


def test_the_citation_is_reached_through_its_own_seam_not_the_fan_out() -> None:
    """#1763's split, pinned: the brand fan-out carries no citation, and the
    citation is reachable through ``_citation_for``.

    The nightly degradation signal reads the pubmed source through
    ``_citation_for(profile, profile.rwe_search_term)`` — the curated brand query,
    which attaches no analysis term and so reproduces the pre-#1763 label exactly.
    If the citation is ever folded back into the fan-out, that seam assumption has
    to be re-examined rather than silently inverted.
    """
    service = _stub_service()
    profile = resolve_brand_profile("Kisqali")

    fragments = service._fan_out(profile)
    assert not any(isinstance(f, CitationFragment) for f in fragments), (
        "a CitationFragment is back in the brand-level fan-out — re-check "
        "test_fan_out_degradation_signal.py, which now fetches it separately"
    )

    cite = service._citation_for(profile, profile.rwe_search_term)
    assert isinstance(cite, CitationFragment)


# ====================================================== the drift guard proper


def _python_files_mentioning(token: str) -> Iterator[Tuple[Path, str]]:
    """Every repo ``.py`` whose text contains ``token``. The substring pre-filter
    keeps this to a handful of ``ast.parse`` calls instead of the whole tree."""
    for root in _SCAN_ROOTS:
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):  # pragma: no cover - unreadable file
                continue
            if token in text:
                yield path, text


def _positional_reads(path: Path, text: str) -> List[Tuple[int, int, str]]:
    """``(lineno, positions_read, form)`` for every positional read of a
    ``*._fan_out(...)`` result: tuple unpacks and constant-index subscripts.

    A starred target (``a, *rest = ...``) reads a variable number of positions and
    is reported as unmeasurable rather than silently skipped.
    """
    tree = ast.parse(text, filename=str(path))
    reads: List[Tuple[int, int, str]] = []

    def _is_seam_call(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == _SEAM
        )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_seam_call(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, (ast.Tuple, ast.List)):
                    if any(isinstance(e, ast.Starred) for e in target.elts):
                        reads.append((node.lineno, -1, "starred-unpack"))
                    else:
                        reads.append((node.lineno, len(target.elts), "unpack"))
        elif isinstance(node, ast.Subscript) and _is_seam_call(node.value):
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                position = index.value
                reads.append((node.lineno, position + 1 if position >= 0 else -position, "index"))
    return reads


def test_every_fan_out_call_site_reads_the_declared_arity() -> None:
    """No call site may read more fragments than ``_fan_out`` returns — including
    the ones the PR lane never runs.

    THIS is the guard for #1766. It reads the nightly ``slow`` +
    ``requires_network`` degradation signal statically, so a signature change lands
    red on the PR that makes it rather than on main at 07:00 UTC the next morning.
    """
    arity = len(get_args(_BrandFragmentTuple))
    scanned: List[Path] = []
    offenders: List[str] = []

    for path, text in _python_files_mentioning(_SEAM):
        scanned.append(path)
        for lineno, positions, form in _positional_reads(path, text):
            rel = path.relative_to(_REPO_ROOT)
            if form == "starred-unpack":
                offenders.append(
                    f"{rel}:{lineno} unpacks _fan_out with a starred target; this "
                    "guard cannot verify it — unpack the fragments by name instead"
                )
            elif positions != arity and form == "unpack":
                offenders.append(
                    f"{rel}:{lineno} unpacks {positions} values from _fan_out, "
                    f"which returns {arity}"
                )
            elif positions > arity:
                offenders.append(
                    f"{rel}:{lineno} indexes position {positions - 1} of _fan_out, "
                    f"which returns {arity}"
                )

    # POSITIVE CONTROL: an empty or partial sweep must fail loudly, not pass.
    assert scanned, f"no file under {_SCAN_ROOTS} mentions {_SEAM!r} — the sweep is broken"
    assert _DEGRADATION_SIGNAL in scanned, (
        f"{_DEGRADATION_SIGNAL.relative_to(_REPO_ROOT)} no longer calls {_SEAM!r}. "
        "That module IS the #1612 AC4 degradation signal and the reason this guard "
        "exists; if the seam moved, move this guard with it."
    )
    assert not offenders, "stale _fan_out call site(s):\n  " + "\n  ".join(offenders)
