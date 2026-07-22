"""data_constraint_context (insights) — constraint-aware grounding for the
home-KPI narrative (constraint-aware insights plan §3, 2026-07-20).

The builder renders ONE deterministic, server-derived block per request:
a per-brand data-constraint profile (disease, prevalence class, claims-lag
band, vendors, CRM source — authored in domain_vocabulary.yaml
``data_constraints``) plus per-KPI classification lines (actionability /
data_plane / measurement_caveat from the KPI registry) for the KPIs actually
present in the grounding.

Contracts under test:
* lag SSOT precedence (reconciled 2026-07-21, plan C0) — the per-brand
  claims-lag BAND is the only LM-facing claims-ADJUDICATION lag figure,
  stated once; the narrative additionally names the DISTINCT 7-14 day
  per-source ingest/feed lag class (as an aggregate class band only); the
  vocabulary's per-source scalar lags (12d/14d/...) must still NOT render
  (two contradictory lag SSOTs invite LM confusion). The former blanket
  vendor-name prohibition is NARROWED (2026-07-22, item 2b, product-owner
  approved): the authored mitigation playbook's illustrative vendors render
  — only inside the playbook block, guarded by home_kpi's vendor allowlist;
* mitigation playbook — authored source classes (SP/hub feeds, open claims,
  lab/EHR, the already-live completion-factor nowcast) render with latency
  bands, coverage caveats, illustrative vendors, and the LM-facing vendor
  validation criteria; the playbook never restates the claims-lag band; a
  missing playbook degrades the WHOLE context loudly ("");
* lag-class reconciliation (C0, reworded for the backlog #45 arrival plane) —
  the band's under-count claim is scoped to REAL-WORLD claims, and the
  narrative states that the claims ARRIVAL plane IS simulated in this
  substrate while the displayed figures are the MATURE values (computed over
  all events regardless of arrival), so they do not under-count (displayed
  figures must never be discounted on the band's account);
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
    build_mitigation_playbook,
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
    # the brand band is the ONLY claims-adjudication lag figure: one claim —
    # the mitigation playbook (2026-07-22) must NOT restate it. Per-source
    # scalar lags stay forbidden — itemized below. (The aggregate "7-14 day"
    # ingest/feed CLASS band is a distinct, permitted figure asserted in
    # test_reconciled_wording_distinguishes_lag_classes; it contains none of
    # the itemized scalar tokens.) The former blanket vendor-name prohibition
    # is narrowed, product-owner approved (item 2b): playbook vendors render,
    # but ONLY inside the playbook block — see
    # test_playbook_vendors_render_only_inside_the_playbook_block.
    assert ctx.lower().count("1-3 month") == 1
    for scalar in ("12d", "14d", "7d", "10d", "IQVIA_APLD"):
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


@pytest.mark.parametrize("brand", _BRANDS + ("All",))
def test_reconciled_wording_distinguishes_lag_classes(brand):
    """Lag-class reconciliation, reworded for the backlog #45 claims ARRIVAL
    plane (design C2, 2026-07-21): the narrative must (a) scope the 1-3-month
    band's under-count claim to REAL-WORLD claims, (b) name the DISTINCT
    7-14 day per-source ingest/feed lag class (aggregate band only — the
    per-source scalar and vendor prohibitions itemized in
    test_per_brand_profile_renders_with_single_lag_claim are unchanged),
    (c) state that the claims arrival plane IS now simulated AND that the
    displayed figures are the MATURE values — computed over all events
    regardless of arrival — so they do not under-count (the no-discount
    guarantee survives the plane landing; the pre-#45 "not simulated" claim
    must be gone), and (d) keep the attribute-don't-recommend instruction
    intact."""
    ctx = build_constraint_context(brand, _METAS)
    assert ctx
    assert "adjudication/runout" in ctx
    assert "In real-world claims" in ctx
    assert "7-14 day" in ctx
    assert "ingest/feed" in ctx
    # (c) the plane is now simulated — the old claim must not survive
    assert "adjudication lag is not simulated" not in ctx
    assert "arrival plane is simulated" in ctx
    assert "mature values" in ctx
    assert "do not under-count" in ctx
    assert "attribute, do not recommend" in ctx


# ---- Mitigation playbook (frontend review 2026-07-22, item 2b) ----------------
_PLAYBOOK_CLASSES = (
    "Specialty pharmacy / hub dispense & status feeds",
    "Open (pre-adjudicated) claims",
    "Lab & EHR feeds",
    "Completion-factor nowcast on closed claims",
)


def test_mitigation_playbook_renders_with_criteria():
    ctx = build_constraint_context("Remibrutinib", _METAS)
    assert "Claims-lag mitigation playbook" in ctx
    # The honest core claim: closed claims cannot be made faster, signal can.
    assert "Faster adjudicated (closed) claims are not achievable" in ctx
    for cls in _PLAYBOOK_CLASSES:
        assert cls in ctx, f"source class {cls!r} missing from playbook"
    # The already-shipped mitigation is credited, not re-recommended blind.
    assert "already live in this platform" in ctx
    # LM-facing vendor validation criteria are itemized with a fail-shut rule.
    assert "Vendor validation criteria" in ctx
    assert "never introduce a vendor from memory" in ctx
    assert "If any check fails, name the source class only." in ctx
    assert "not vetted or contracted suppliers" in ctx


def test_playbook_vendors_render_only_inside_the_playbook_block():
    """The plan-C0 no-vendor rule is superseded ONLY for playbook vendors
    (product-owner approved 2026-07-22), and only inside the playbook block —
    nothing above it may name a vendor."""
    ctx = build_constraint_context("Remibrutinib", _METAS)
    head = ctx[: ctx.index("Claims-lag mitigation playbook")]
    for vendor in ("IQVIA", "Symphony Health", "Komodo Health", "HealthVerity", "AssistRx"):
        assert vendor in ctx, f"illustrative vendor {vendor!r} missing from playbook"
        assert vendor not in head, f"vendor {vendor!r} leaked above the playbook block"


def test_every_playbook_vendor_is_covered_by_the_guard_lexicon():
    """Guard-visibility invariant: every vendor authored into the playbook must
    be detectable by the home-KPI vendor guard's lexicon — otherwise an
    out-of-playbook pairing of that vendor could never be caught."""
    import re

    from src.insights.data_constraint_context import KNOWN_DATA_VENDORS, _constraints

    patterns = [re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE) for n in KNOWN_DATA_VENDORS]
    playbook = _constraints()["mitigation_playbook"]
    for sc in playbook["source_classes"]:
        for vendor in sc.get("illustrative_vendors") or []:
            assert any(p.search(vendor) for p in patterns), (
                f"playbook vendor {vendor!r} invisible to the guard lexicon"
            )


def test_build_mitigation_playbook_structure():
    pb = build_mitigation_playbook()
    assert pb is not None
    assert pb["preamble"].startswith("Faster adjudicated (closed) claims are not achievable")
    assert "not vetted or contracted suppliers" in pb["vendor_note"]
    by_name = {sc["name"]: sc for sc in pb["source_classes"]}
    assert set(by_name) == set(_PLAYBOOK_CLASSES)
    open_claims = by_name["Open (pre-adjudicated) claims"]
    assert "IQVIA" in open_claims["illustrative_vendors"]
    assert "1-7 days" in open_claims["latency"]
    nowcast = by_name["Completion-factor nowcast on closed claims"]
    assert nowcast["status"] and "live" in nowcast["status"]
    assert nowcast["illustrative_vendors"] == []


def test_build_mitigation_playbook_degrades_to_none(monkeypatch):
    import src.insights.data_constraint_context as dcc

    monkeypatch.setattr(dcc, "_constraints", lambda: {})
    assert build_mitigation_playbook() is None


def test_missing_playbook_degrades_constraint_context_loudly(monkeypatch):
    """An authoring regression (playbook dropped from the vocabulary) must take
    the whole context down the loud path ("" -> route warning chip + short
    TTL), never silently revert the narrative to an unactionable lag
    statement."""
    import src.insights.data_constraint_context as dcc

    real = dict(dcc._constraints())
    real.pop("mitigation_playbook", None)
    monkeypatch.setattr(dcc, "_constraints", lambda: real)
    assert dcc.build_constraint_context("Remibrutinib", _METAS) == ""


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
