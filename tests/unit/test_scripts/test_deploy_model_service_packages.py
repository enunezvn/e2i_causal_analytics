"""Guard for the dowhy floor in generated causal Bento packages (#869, codex R1 MED).

``scripts/deploy_model.py`` emits a Python package list into every Bento it
builds (``bentofile`` ``python_packages``). For ``service_type == "causal"``
that list declares dowhy — a third resolver surface beside pyproject.toml and
docker/bentoml/requirements-bentoml.txt (those two are guarded by
``tests/test_requirements_lock.py::test_pyproject_dowhy_floor_is_networkx35_compatible``).

A floor below 0.13 lets the Bento image resolver install a dowhy that calls
the removed ``nx.algorithms.d_separated`` (networkx >= 3.5), so every
``CausalModel.identify_effect`` / refuter call inside the served causal model
would raise AttributeError. dowhy >= 0.13 imports ``is_d_separator`` with a
``d_separated`` fallback and works against any modern networkx.

To falsify: lower the dowhy floor in ``service_packages("causal")`` below
0.13 — this test reports the generated spec admits an nx-incompatible dowhy.
"""

from __future__ import annotations

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from scripts.deploy_model import service_packages


def _requirements(service_type: str) -> list[Requirement]:
    return [Requirement(pkg) for pkg in service_packages(service_type)]


def test_causal_service_packages_declare_dowhy() -> None:
    names = {canonicalize_name(req.name) for req in _requirements("causal")}
    assert {"dowhy", "econml"} <= names, (
        f"causal Bento packages must declare dowhy + econml, got {sorted(names)}"
    )


def test_causal_service_dowhy_floor_is_networkx35_compatible() -> None:
    dowhy_reqs = [req for req in _requirements("causal") if canonicalize_name(req.name) == "dowhy"]
    nx_incompatible = [str(r) for r in dowhy_reqs if r.specifier.contains("0.12")]
    assert not nx_incompatible, (
        "generated causal Bento packages admit dowhy releases that call the "
        "removed nx.algorithms.d_separated (broken under networkx >= 3.5, #869); "
        f"raise the floor to >=0.13: {nx_incompatible}"
    )
    assert any(r.specifier.contains("0.14") for r in dowhy_reqs), (
        "generated causal Bento dowhy spec no longer admits the deployed dowhy==0.14 pin"
    )


def test_non_causal_service_packages_unchanged_by_869_fix() -> None:
    """The floor bump must not leak dowhy into non-causal service types."""
    for service_type in ("classification", "regression", "other"):
        names = {canonicalize_name(req.name) for req in _requirements(service_type)}
        assert "dowhy" not in names, f"{service_type} packages unexpectedly declare dowhy"
        assert "bentoml" in names, f"{service_type} packages lost the bentoml base"
