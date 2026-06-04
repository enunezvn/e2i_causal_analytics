"""Unit tests for the shared tenant brand-scoping helpers (H1).

These pure helpers centralize the "which brand may this caller read?" decision
that the memory-search and cognitive-query routes need so a non-admin cannot
read another tenant's PHI-bearing episodic memories.

Memory-review finding H1: ``search_memory`` / ``cognitive.query`` forwarded a
caller-controlled brand straight into ``hybrid_search`` with no grant check,
so omitting the brand (RPC predicate ``filters->>'brand' IS NULL`` => all
brands) or naming another tenant's brand returned cross-tenant rows.

Contract (fail-closed):
- cross-brand admin (ADMIN role or ``'all'`` grant): no restriction; the
  requested brand (including ``None`` => all brands) passes through.
- non-admin with NO grants: denied (cannot tenant-scope a read).
- non-admin, no brand requested: pinned to the first granted brand.
- non-admin, requested brand in grants: allowed for that brand.
- non-admin, requested brand out of grants: denied.
"""

from __future__ import annotations

from typing import Any, Dict

from src.api.dependencies.auth import is_cross_brand_admin, resolve_brand_for_read


def _viewer(*brands: str) -> Dict[str, Any]:
    return {"sub": "u-view", "role": "viewer", "brands": list(brands)}


def _admin() -> Dict[str, Any]:
    return {"sub": "u-admin", "role": "admin", "brands": []}


def _all_grant_viewer() -> Dict[str, Any]:
    return {"sub": "u-all", "role": "viewer", "brands": ["all"]}


# ---------------------------------------------------------------------------
# is_cross_brand_admin
# ---------------------------------------------------------------------------


def test_is_cross_brand_admin_true_for_admin_role() -> None:
    assert is_cross_brand_admin(_admin()) is True


def test_is_cross_brand_admin_true_for_all_grant() -> None:
    assert is_cross_brand_admin(_all_grant_viewer()) is True


def test_is_cross_brand_admin_false_for_scoped_viewer() -> None:
    assert is_cross_brand_admin(_viewer("Brand-X")) is False


# ---------------------------------------------------------------------------
# resolve_brand_for_read -> (allowed: bool, brand: Optional[str])
# ---------------------------------------------------------------------------


def test_admin_passthrough_specific_brand() -> None:
    assert resolve_brand_for_read(_admin(), "Brand-Y") == (True, "Brand-Y")


def test_admin_passthrough_none_means_all_brands() -> None:
    # None => no brand filter => all brands; only admins may do this.
    assert resolve_brand_for_read(_admin(), None) == (True, None)


def test_nonadmin_no_grants_is_denied() -> None:
    allowed, brand = resolve_brand_for_read(_viewer(), None)
    assert allowed is False
    assert brand is None


def test_nonadmin_no_brand_pins_to_first_grant() -> None:
    assert resolve_brand_for_read(_viewer("Brand-X", "Brand-Z"), None) == (True, "Brand-X")


def test_nonadmin_in_grant_brand_allowed() -> None:
    assert resolve_brand_for_read(_viewer("Brand-X", "Brand-Z"), "Brand-Z") == (True, "Brand-Z")


def test_nonadmin_out_of_grant_brand_denied() -> None:
    allowed, brand = resolve_brand_for_read(_viewer("Brand-X"), "Brand-Y")
    assert allowed is False
    assert brand is None
