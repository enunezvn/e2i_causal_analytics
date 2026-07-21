"""data_constraint_context (insights) — constraint-aware grounding for the
home-KPI narrative (constraint-aware insights plan §3, 2026-07-20).

The builder renders ONE deterministic, server-derived block per request:
a per-brand data-constraint profile (disease, prevalence class, claims-lag
band, vendors, CRM source — authored in domain_vocabulary.yaml
``data_constraints``) plus per-KPI classification lines (actionability /
data_plane / measurement_caveat from the KPI registry) for the KPIs actually
present in the grounding.

Contracts under test:
* lag SSOT precedence — the per-brand claims-lag BAND is the only LM-facing
  lag figure; the vocabulary's per-source scalar lags (12d/14d/...) must NOT
  render (two contradictory lag SSOTs invite LM confusion);
* prevalence direction guard — verbatim text: prevalence explains small
  samples and volatility, NOT low engagement/testing/coverage rates;
* claims-plane KPIs present get the lag attachment; CRM/platform-plane KPIs
  present get "current as shown";
* brand='All' renders portfolio-level facts + brand-labeled one-liners;
* loud degradation — any failure returns "" (the route adds the warning chip
  and short TTL); a missing brand profile degrades loudly, not silently;
* completeness — every gold-standard brand has a profile.
"""

from types import SimpleNamespace

import pytest

from src.insights.data_constraint_context import (
    brands_with_profiles,
    build_constraint_context,
)

_BRANDS = ("Remibrutinib", "Kisqali", "Fabhalta")


def _meta(kpi_id, name, data_plane=None, caveat=None, actionability=None, levers=()):
    return SimpleNamespace(
        id=kpi_id,
        name=name,
        data_plane=data_plane,
        measurement_caveat=caveat,
        actionability=actionability,
        actionability_owner="brand_team" if actionability else None,
        levers=list(levers),
        direction=None,
        brand=None,
    )


_METAS = [
    _meta(
        "WS2-TR-001",
        "Trigger Precision",
        data_plane="claims",
        caveat="Definition v2 as of 2026-07-20.",
        actionability="mixed",
    ),
    _meta("WS2-TR-004", "Acceptance Rate", data_plane="crm", actionability="reader_actionable"),
    _meta(
        "WS3-BI-005",
        "Total Prescriptions (TRx)",
        data_plane="claims",
        caveat="Recent windows under-count until claims mature.",
        actionability="reader_actionable",
        levers=["HCP targeting and call-plan coverage"],
    ),
    _meta("WS1-MP-001", "ROC-AUC", data_plane="platform"),
]


@pytest.mark.parametrize("brand", _BRANDS)
def test_per_brand_profile_renders_with_single_lag_claim(brand):
    ctx = build_constraint_context(brand, _METAS)
    assert ctx, f"{brand} must have a constraint profile"
    # the ONLY LM-facing lag figure is the brand band; one claim, no scalars
    assert ctx.lower().count("1-3 month") == 1
    for scalar in ("12d", "14d", "7d", "10d", "IQVIA_APLD", "HealthVerity", "Komodo"):
        assert scalar not in ctx, f"per-source scalar lag {scalar!r} leaked into LM context"


@pytest.mark.parametrize("brand", _BRANDS)
def test_prevalence_direction_guard_is_verbatim(brand):
    ctx = build_constraint_context(brand, _METAS)
    assert "NOT low engagement/testing/coverage rates" in ctx


def test_claims_plane_kpis_get_lag_attachment_and_crm_get_current():
    ctx = build_constraint_context("Remibrutinib", _METAS)
    assert "Trigger Precision" in ctx and "Total Prescriptions (TRx)" in ctx
    assert "Acceptance Rate" in ctx
    # the CRM line must say current/trustworthy as shown
    crm_line = next(line for line in ctx.splitlines() if "Acceptance Rate" in line)
    assert "current as shown" in crm_line
    claims_line = next(line for line in ctx.splitlines() if "Trigger Precision" in line)
    assert "lag" in claims_line.lower()


def test_absent_kpis_render_no_lines():
    ctx = build_constraint_context("Remibrutinib", [_METAS[1]])
    assert "Trigger Precision" not in ctx
    assert "TRx" not in ctx


def test_caveats_render_for_tagged_kpis_only():
    ctx = build_constraint_context("Remibrutinib", _METAS)
    assert "Definition v2 as of 2026-07-20." in ctx
    # untagged KPI gets no invented caveat
    roc_lines = [line for line in ctx.splitlines() if "ROC-AUC" in line]
    for line in roc_lines:
        assert "caveat" not in line.lower()


def test_all_brand_scope_renders_portfolio_profiles():
    ctx = build_constraint_context("All", _METAS)
    assert ctx
    for brand in _BRANDS:
        assert brand in ctx, f"portfolio context must carry a {brand}-labeled line"


def test_unknown_brand_degrades_loudly_to_empty():
    assert build_constraint_context("BrandX", _METAS) == ""


def test_malformed_metas_degrade_to_empty():
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("malformed meta")

    assert build_constraint_context("Remibrutinib", [_Boom()]) == ""


def test_every_gold_standard_brand_has_a_profile():
    """A 4th brand launch must fail THIS test (loudly), not silently render an
    un-profiled brand with no constraint context."""
    assert set(brands_with_profiles()) >= set(_BRANDS)
