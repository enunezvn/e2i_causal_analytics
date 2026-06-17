"""Unit tests for ``canonical_brand`` — the brand-casing canonicaliser that
collapses duplicate Brand nodes (the demo seed's ``Remibrutinib`` vs the
cohort-constructor agent's lowercase ``remibrutinib``) onto one identity.
"""

import pytest

from src.memory.episodic_memory import E2IBrand, canonical_brand


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("remibrutinib", "Remibrutinib"),
        ("REMIBRUTINIB", "Remibrutinib"),
        ("Remibrutinib", "Remibrutinib"),
        ("  remibrutinib  ", "Remibrutinib"),
        ("kisqali", "Kisqali"),
        ("Fabhalta", "Fabhalta"),
        ("fabhalta", "Fabhalta"),
        ("all", "all"),
        ("ALL", "all"),
    ],
)
def test_canonicalises_known_brands_case_insensitively(raw, expected):
    assert canonical_brand(raw) == expected


def test_every_e2ibrand_value_is_a_fixed_point():
    # The canonical form of a canonical value is itself.
    for brand in E2IBrand:
        assert canonical_brand(brand.value) == brand.value


def test_unknown_brand_is_preserved_not_dropped():
    # An unrecognised brand is surfaced (stripped), never silently erased.
    assert canonical_brand("NovBrandX") == "NovBrandX"
    assert canonical_brand("  spaced brand  ") == "spaced brand"


def test_empty_and_none_pass_through():
    assert canonical_brand("") == ""
    assert canonical_brand(None) is None
