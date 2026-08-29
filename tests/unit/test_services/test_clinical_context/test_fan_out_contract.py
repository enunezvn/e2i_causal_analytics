"""Network-free contract guard for ``ClinicalContextService._fan_out`` (#1766).

WHY THIS FILE EXISTS — the defect class, not the instance
---------------------------------------------------------
#1763 moved the literature citation out of ``_fan_out`` into ``_citation_for``,
dropping the fan-out from five fragments to four. The only caller that unpacks
``_fan_out`` positionally outside the service is
``tests/integration/test_clinical_context/test_fan_out_degradation_signal.py``
— the #1612 AC4 degradation signal — and that module is ``slow`` +
``requires_network``, so the PR-blocking lane deselects it (``-m "not slow"``)
and only the 05:00 UTC ``slow-tests.yml`` Job A ever executes it. The #1763 train
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
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, get_args

import pytest

from src.services.clinical_context.brand_map import BrandClinicalProfile, resolve_brand_profile
from src.services.clinical_context.clients import CTGovEndpoint, PubMedArticle
from src.services.clinical_context.providers import (
    CitationFragment,
    ClinicalContextProvider,
    CompetitorFragment,
    EndpointsFragment,
    IndicationsFragment,
    MechanismFragment,
)
from src.services.clinical_context.service import (
    ClinicalContextService,
    _BrandFragmentTuple,
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


class _StubProvider(ClinicalContextProvider):
    """Returns a fixed fragment. No network — this file asserts a structural
    contract, so the fragments only need to be the right TYPES; nothing here is
    presented as, or compared against, a live provider result."""

    def __init__(self, fragment: object, name: str = "stub") -> None:
        self._fragment = fragment
        self.provider_name = name

    def enrich(self, profile: BrandClinicalProfile) -> object:
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


# Every construct Python gives its OWN namespace. Class bodies and comprehensions
# belong here just as much as functions do (codex iter-3): a name bound inside one
# does not bind in the enclosing scope, so counting it there would make an
# enclosing read look ambiguous and silently drop it — a false green in the guard
# whose whole job is preventing false greens.
_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _is_seam_call(node: Optional[ast.AST]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == _SEAM
    )


def _nodes_in_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node under ``scope`` that is not inside a NESTED namespace."""
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, _SCOPES):
            yield from _nodes_in_scope(child)


def _child_scopes(scope: ast.AST) -> Iterator[ast.AST]:
    """The namespaces nested DIRECTLY inside ``scope`` (not their own nestings)."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, _SCOPES):
            yield child
        else:
            yield from _child_scopes(child)


def _bound_names(nodes: List[ast.AST]) -> Counter[str]:
    """Every name-binding occurrence in one scope, counted.

    Counting rather than set-membership is what lets the caller ask "is EVERY
    binding of this name a fan-out assignment", which is the question that keeps
    the indirect pass from mis-attributing a shadowed name.

    A ``Name`` in ``Store``/``Del`` context covers assignment, augmented
    assignment, ``for`` targets, ``with ... as`` and the walrus. The rest bind
    through their own fields and are enumerated explicitly. Comprehension and
    class-body targets are NOT seen here: ``_nodes_in_scope`` stops at those
    namespaces, which is exactly where Python puts them.
    """
    bound: Counter[str] = Counter()
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound[node.id] += 1
        elif isinstance(node, ast.arg):
            bound[node.arg] += 1
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound[node.name] += 1
        elif isinstance(node, ast.alias):
            bound[node.asname or node.name.split(".")[0]] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound[node.name] += 1
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            # Bound somewhere this scope cannot see — never followable.
            for name in node.names:
                bound[name] += 1
    return bound


def _positional_reads(path: Path, text: str) -> List[Tuple[int, int, str]]:
    """``(lineno, positions_read, form)`` for every positional read of a
    ``*._fan_out(...)`` result.

    Two forms are swept, because only sweeping the first would let the drift walk
    one line down and escape (codex iter-1 LOW):

    * DIRECT — the call expression is destructured or subscripted in place
      (``a, b, c, d = svc._fan_out(p)``, ``svc._fan_out(p)[3]``).
    * INDIRECT — the call is bound to a name and the NAME is destructured or
      subscripted (``frags = svc._fan_out(p)`` … ``frags[4]``). Followed per
      namespace, inherited into nested ones the way a closure reads an enclosing
      local, and dropped the moment the nested namespace rebinds the name.
      A name is followed ONLY when *every* binding of it in its own namespace is a
      fan-out assignment, counted over every binding form the language has, not
      just plain assignment (codex iter-2 LOW). A missed report is a weaker guard;
      a wrong one is a guard people start ignoring.

    A starred target (``a, *rest = …``) reads a variable number of positions and is
    reported as unmeasurable rather than silently skipped.
    """
    tree = ast.parse(text, filename=str(path))
    reads: List[Tuple[int, int, str]] = []

    def _record_unpack(node: ast.AST, target: ast.expr) -> None:
        if not isinstance(target, (ast.Tuple, ast.List)):
            return
        lineno = getattr(node, "lineno", 0)
        if any(isinstance(e, ast.Starred) for e in target.elts):
            reads.append((lineno, -1, "starred-unpack"))
        else:
            reads.append((lineno, len(target.elts), "unpack"))

    def _record_index(node: ast.Subscript) -> None:
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, int):
            position = index.value
            reads.append((node.lineno, position + 1 if position >= 0 else -position, "index"))

    def _scan(scope: ast.AST, inherited: frozenset) -> None:
        nodes = list(_nodes_in_scope(scope))

        # Pass 1 — DIRECT reads, plus which names in THIS namespace hold a
        # fan-out result. Direct reads are structural and never depend on scope.
        seam_bindings: Counter[str] = Counter()
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_seam_call(node.value):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    _record_unpack(node, target)
                    if isinstance(target, ast.Name):
                        seam_bindings[target.id] += 1
            elif isinstance(node, ast.Subscript) and _is_seam_call(node.value):
                _record_index(node)

        # A name is followed only when EVERY binding of it here is a fan-out
        # assignment. Anything else — a loop target, a with/except alias, an
        # import, a def/class, a parameter, a walrus — makes it ambiguous, and
        # this guard reports nothing rather than accusing the wrong line. An
        # inherited name survives only while this namespace does not rebind it.
        local = _bound_names(nodes)
        visible = frozenset(
            {n for n in inherited if local[n] == 0}
            | {n for n, c in seam_bindings.items() if local[n] == c}
        )

        # Pass 2 — INDIRECT reads through the visible names.
        if visible:
            for node in nodes:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                    if node.value.id in visible:
                        for target in node.targets:
                            _record_unpack(node, target)
                elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                    if node.value.id in visible:
                        _record_index(node)

        for child in _child_scopes(scope):
            _scan(child, visible)

    _scan(tree, frozenset())
    return reads


def test_the_sweep_itself_sees_both_direct_and_indirect_stale_reads() -> None:
    """A guard nobody has seen fail is a decoration. This runs the sweep over
    synthetic source holding each stale form and pins what it reports.

    The indirect case is codex iter-1's LOW: before it was handled, moving the
    stale read one line down (``frags = svc._fan_out(p)`` then ``frags[4]``)
    walked straight past this guard. The shadowing cases are codex iter-2's LOW:
    counting only plain assignments as bindings would have let a ``for`` target, a
    ``with``/``except`` alias, an import or a parameter of the same name get
    accused of a stale read it never made.
    """
    source = (
        "def direct(svc, p):\n"  # 1
        "    a, b, c, d, e = svc._fan_out(p)\n"  # 2
        "    return svc._fan_out(p)[4]\n"  # 3
        "\n"
        "def indirect(svc, p):\n"  # 5
        "    frags = svc._fan_out(p)\n"  # 6
        "    a, b, c, d, e = frags\n"  # 7
        "    return frags[4]\n"  # 8
        "\n"
        "def starred(svc, p):\n"  # 10
        "    head, *rest = svc._fan_out(p)\n"  # 11
        "    return head, rest\n"  # 12
        "\n"
        "def correct(svc, p):\n"  # 14
        "    moa, eps, ind, comp = svc._fan_out(p)\n"  # 15
        "    return moa, eps, ind, comp\n"  # 16
        "\n"
        "def scoped(svc, p):\n"  # 18
        "    frags = svc._fan_out(p)\n"  # 19
        "    def inner(other):\n"  # 20
        "        frags = other\n"  # 21
        "        return frags[9]\n"  # 22
        "    return inner, frags[4]\n"  # 23
        "\n"
        "def rebound(svc, p, other):\n"  # 25
        "    frags = svc._fan_out(p)\n"  # 26
        "    frags = other\n"  # 27
        "    return frags[9]\n"  # 28
        "\n"
        "def loop_shadow(svc, p, rows):\n"  # 30
        "    frags = svc._fan_out(p)\n"  # 31
        "    for frags in rows:\n"  # 32
        "        pass\n"  # 33
        "    return frags[9]\n"  # 34
        "\n"
        "def with_shadow(svc, p, cm):\n"  # 36
        "    frags = svc._fan_out(p)\n"  # 37
        "    with cm as frags:\n"  # 38
        "        return frags[9]\n"  # 39
        "\n"
        "def except_shadow(svc, p):\n"  # 41
        "    frags = svc._fan_out(p)\n"  # 42
        "    try:\n"  # 43
        "        pass\n"  # 44
        "    except ValueError as frags:\n"  # 45
        "        return frags[9]\n"  # 46
        "    return None\n"  # 47
        "\n"
        "def import_shadow(svc, p):\n"  # 49
        "    frags = svc._fan_out(p)\n"  # 50
        "    import frags\n"  # 51
        "    return frags[9]\n"  # 52
        "\n"
        "def param_shadow(svc, p, frags):\n"  # 54
        "    frags = svc._fan_out(p)\n"  # 55
        "    return frags[9]\n"  # 56
        "\n"
        "def class_body(svc, p, other):\n"  # 58
        "    frags = svc._fan_out(p)\n"  # 59
        "    class C:\n"  # 60
        "        frags = other\n"  # 61
        "    return C, frags[4]\n"  # 62
        "\n"
        "def comp_target(svc, p, rows):\n"  # 64
        "    frags = svc._fan_out(p)\n"  # 65
        "    seen = [frags for frags in rows]\n"  # 66
        "    return seen, frags[4]\n"  # 67
        "\n"
        "def closure_read(svc, p):\n"  # 69
        "    frags = svc._fan_out(p)\n"  # 70
        "    def inner():\n"  # 71
        "        return frags[4]\n"  # 72
        "    return inner\n"  # 73
        "\n"
        "def comp_body_read(svc, p, rows):\n"  # 75
        "    frags = svc._fan_out(p)\n"  # 76
        "    return [frags[4] for _ in rows]\n"  # 77
        "\n"
        "def comp_local_read(svc, p, rows):\n"  # 79
        "    frags = svc._fan_out(p)\n"  # 80
        "    return frags, [frags[9] for frags in rows]\n"  # 81
    )
    reads = _positional_reads(Path("<synthetic>"), source)
    by_line = {lineno: (positions, form) for lineno, positions, form in reads}

    assert by_line[2] == (5, "unpack"), "direct 5-way unpack missed"
    assert by_line[3] == (5, "index"), "direct index-4 read missed"
    assert by_line[7] == (5, "unpack"), "INDIRECT 5-way unpack missed (codex iter-1 LOW)"
    assert by_line[8] == (5, "index"), "INDIRECT index-4 read missed (codex iter-1 LOW)"
    assert by_line[11] == (-1, "starred-unpack"), "starred target must be reported, not skipped"
    assert by_line[15] == (4, "unpack"), "the correct call site must still be seen"
    # A nested function is its own scope: the outer read is still followed, and the
    # inner name that merely shares a spelling is not attributed to the outer one.
    assert by_line[23] == (5, "index"), "outer-scope indirect read lost to a nested def"
    assert 22 not in by_line, "a nested scope's unrelated local was attributed to _fan_out"
    # Every ambiguous shadowing form is deliberately NOT reported: a wrong
    # accusation is what teaches people to ignore a guard (codex iter-2 LOW).
    for shadowed in (28, 34, 39, 46, 52, 56):
        assert shadowed not in by_line, (
            f"line {shadowed} is a shadowed name, not a verified _fan_out read — "
            "reporting it would be a false positive"
        )
    # A class body and a comprehension are their OWN namespaces, so a name bound
    # inside one must not make the enclosing read look ambiguous — dropping the
    # enclosing read there would be a silent false green (codex iter-3 LOW).
    assert by_line[62] == (5, "index"), "a class-body binding swallowed an enclosing read"
    assert by_line[67] == (5, "index"), "a comprehension target swallowed an enclosing read"
    # …and reads INSIDE a nested namespace follow the enclosing name the way a
    # closure does, unless that namespace rebinds it.
    assert by_line[72] == (5, "index"), "closure read of the enclosing fan-out result missed"
    assert by_line[77] == (5, "index"), "comprehension-body read of the enclosing result missed"
    assert 81 not in by_line, "a comprehension-LOCAL name was attributed to the enclosing result"


def test_every_fan_out_call_site_reads_the_declared_arity() -> None:
    """No call site may read more fragments than ``_fan_out`` returns — including
    the ones the PR lane never runs.

    THIS is the guard for #1766. It reads the nightly ``slow`` +
    ``requires_network`` degradation signal statically, so a signature change lands
    red on the PR that makes it rather than on main at 05:00 UTC the next morning.
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
