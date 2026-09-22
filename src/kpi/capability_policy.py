"""The ONE evaluator for "which KPI serves which axis" (#2150).

One fact was hand-copied into 13 places, at least 8 of them LLM-facing, and they
drifted: the system prompts told the model to call TRx/NRx/NBRx for a patient-axis
breakdown, and those names resolve to the CANONICAL ids, which refuse every patient
axis since owner #11 moved the panel to WS3-BI-011..013. Every surface now asks
here instead.

WHAT THIS MODULE OWNS: evaluation and prose. What it does NOT own:

* the DECLARATION — ``config/kpi_definitions.yaml`` + ``KPIMetadata.axis_capability``
  (measured per KPI from the calculators' recorded query binding, not hand-listed);
* the AUTHORED REASONS and BRAND ELIGIBILITY — ``src.kpi.share_axis``. Owner #14
  gave the canonical share and the panel share DIFFERENT reasons because each is
  true of its own substrate; composing one sentence for both would restate for 008
  exactly what owner #14 rejected. This module READS those, never regenerates them.

⚠ A DESTINATION IS NEVER NAMED FROM AN ID MAP. The lane shipped four dead-end
redirects, each one a destination taken from a mapping without asking whether it
could serve the ask. Here a map may PROPOSE (``SHARE_REDIRECTS`` carries owner
#14's ruling, ``CANONICAL_TO_PANEL`` the lane's twin structure) but ``serves()``
DECIDES, and a proposal that cannot serve is discarded rather than offered.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbcSet
from typing import Optional

from src.kpi.registry import get_registry
from src.kpi.share_axis import (
    PATIENT_AXES,
    SHARE_REDIRECTS,
    brand_scoped_axis_refusal,
    share_axis_reason_for,
)
from src.kpi.volume_family import CANONICAL_TO_PANEL

#: The axes this policy governs. ``region`` is included because two KPIs serve it
#: without a window and refuse it under one, which is the case that forced
#: ``axes_under_window`` to exist.
PATIENT_AXIS_NAMES: tuple[str, ...] = tuple(axis for axis, _label in PATIENT_AXES)
AXIS_LABELS: dict[str, str] = dict(PATIENT_AXES)


class _LazySet(AbcSet):
    """A set whose members are computed on EVERY access, not at import.

    ⚠ WHY THIS IS NOT A SNAPSHOT. The derived constants used to be evaluated at
    module import, which bound them to whatever the registry held the first time
    the module was imported. That is harmless while the registry is loaded once
    and never reloaded (MEASURED: nothing in src/ calls KPIRegistry.reset or
    get_registry.cache_clear) — but ``registry._load_definitions`` WARNS instead of
    raising when the YAML cannot be found, so a mis-resolved config path at import
    time would leave the constants silently EMPTY. An empty axis allowlist refuses
    every axis and an empty coverage set never probes, both with nothing but a log
    line. Computing per access removes the snapshot entirely.
    """

    def __init__(self, fn):
        self._fn = fn

    def __contains__(self, item):
        return item in self._fn()

    def __iter__(self):
        return iter(self._fn())

    def __len__(self):
        return len(self._fn())

    def __repr__(self):
        return repr(set(self._fn()))


class _LazyAxisMap(Mapping):
    """``axis -> served KPI ids``, resolved per access for the same reason."""

    def __getitem__(self, axis):
        if axis not in PATIENT_AXIS_NAMES:
            raise KeyError(axis)
        return axis_kpi_ids(axis)

    def __iter__(self):
        return iter(PATIENT_AXIS_NAMES)

    def __len__(self):
        return len(PATIENT_AXIS_NAMES)


def lazy_axis_kpi_ids() -> "_LazyAxisMap":
    return _LazyAxisMap()


def lazy_trailing_coverage_kpi_ids() -> "_LazySet":
    return _LazySet(trailing_coverage_kpi_ids)


class _LazyWindowMap(Mapping):
    """``kpi id -> reporting window``, resolved per access (same snapshot risk)."""

    def __getitem__(self, kpi_id):
        return reporting_windows()[kpi_id]

    def __iter__(self):
        return iter(reporting_windows())

    def __len__(self):
        return len(reporting_windows())


def lazy_reporting_windows() -> "_LazyWindowMap":
    return _LazyWindowMap()


def serves(kpi_id: str, axis: str, *, brand: Optional[str] = None, window: bool = False) -> bool:
    """Does this KPI bind ``axis`` for ``brand`` (optionally under a window)?

    FAILS CLOSED: an undeclared KPI or an undeclared axis is refused, never served.
    A missing declaration must not become a promise.
    """
    kpi = get_registry().get(kpi_id)
    if kpi is None or kpi.axis_capability is None:
        return False
    if not kpi.axis_capability.serves(axis, window=window):
        return False
    # The axis may exist on the KPI and still be unanswerable for this brand:
    # biologic / IgE are populated for one brand only, every other brand is NULL by
    # design. share_axis owns that fact; asking it here is what keeps a redirect
    # from promising an ask that is impossible for the brand in hand (dead end 4).
    if brand is not None and brand_scoped_axis_refusal(axis, brand) is not None:
        return False
    return True


def axis_kpi_ids(axis: str, *, brand: Optional[str] = None, window: bool = False) -> frozenset:
    """Every KPI that serves ``axis`` — the derived form of the hand-kept allowlists."""
    return frozenset(
        kpi.id
        for kpi in get_registry().get_all()
        if serves(kpi.id, axis, brand=brand, window=window)
    )


def trailing_coverage_kpi_ids() -> frozenset:
    """KPIs the trailing-30-day window-coverage probe may run on.

    TWO CONDITIONS, BOTH REQUIRED, and the second cannot be inferred:

    * **event grain** — the measure counts rows in ``treatment_events``, so a
      trailing slice is a sub-period of the same thing;
    * **additive** — DECLARED on the metadata. The probe divides a trailing value
      by a window value and warns when the share is too high; for a ratio, a rate
      or a median that arithmetic is meaningless, which is what the original
      comment at the probe's gate says. ``value_format`` cannot stand in for this:
      MEASURED, five non-additive event-grain KPIs carry ``value_format=None``.

    Undeclared additivity counts as NOT additive (fail closed: a missing advisory
    beats a false warning reaching the user), and the completeness test makes that
    silence loud at test time.
    """
    return frozenset(
        kpi.id
        for kpi in get_registry().get_all()
        if "treatment_events" in (kpi.tables or []) and kpi.additive is True
    )


def event_grain_kpi_ids() -> frozenset:
    """Every KPI measured over ``treatment_events`` rows — the set that must carry
    an explicit additivity declaration."""
    return frozenset(
        kpi.id for kpi in get_registry().get_all() if "treatment_events" in (kpi.tables or [])
    )


def reporting_windows() -> dict[str, str]:
    """KPI id -> the period a DEFAULT (windowless) reading covers.

    Derived from the declared field, so the disclosure cannot drift from the
    substrate the way the hand-kept dict did: it still told the model that the
    canonical WS3-BI-005..008 cover "the most recent 30 days of prescription
    data" after they moved to MONTHLY business_metrics, and it never learned
    about the panel ids that actually do cover 30 days. A KPI with no verified
    window is OMITTED — honest absence over a guessed period.
    """
    return {
        kpi.id: kpi.reporting_window for kpi in get_registry().get_all() if kpi.reporting_window
    }


def redirect_for(kpi_id: str, axis: str, *, brand: Optional[str] = None) -> Optional[str]:
    """Where to send an ask this KPI cannot serve — or ``None`` when NOWHERE can.

    ``None`` is a real answer, not a failure: biologic status for Kisqali is served
    by no KPI at all, and naming one would be the brand-blind redirect of dead end
    4. A proposal is only returned once ``serves()`` has confirmed it.
    """
    if brand is not None and brand_scoped_axis_refusal(axis, brand) is not None:
        return None
    proposals = []
    if kpi_id in SHARE_REDIRECTS:  # owner #14: both shares answer at panel TRx
        proposals.append(SHARE_REDIRECTS[kpi_id][0])
    if kpi_id in CANONICAL_TO_PANEL:  # the lane's canonical -> panel twin
        proposals.append(CANONICAL_TO_PANEL[kpi_id])
    for candidate in proposals:
        if serves(candidate, axis, brand=brand):
            return candidate
    # NO BLANKET FALLBACK. "Any KPI that serves this axis" is not a redirect: it
    # answers a DIFFERENT question. Measured while wiring the caveat — ROC-AUC by
    # segment resolved to "ask CATE by that axis instead", which is a different
    # metric entirely, not a way to get the user's answer. Only a genuine twin or
    # owner #14's share destination is offered; otherwise the case is terminal.
    return None


def reason_for(kpi_id: str, axis: str) -> str:
    """Why this KPI has no breakdown on ``axis``, in words true of ITS substrate.

    The share KPIs keep their AUTHORED sentences (owner #14 ruled they must differ);
    everything else gets the generic statement.
    """
    label = AXIS_LABELS.get(axis, axis)
    if kpi_id in SHARE_REDIRECTS:
        return share_axis_reason_for(kpi_id, axis, label)
    kpi = get_registry().get(kpi_id)
    name = kpi.name if kpi is not None else kpi_id
    return f"{name} does not carry a {label} dimension."


# =============================================================================
# PROSE — generated from the same evaluation, so it cannot drift from the gate
# =============================================================================


def short_names(ids) -> str:
    """Registry names shortened to the parenthesised short form where one exists —
    "Observed Rx Events - Patient Panel TRx (TRx Panel)" reads as "TRx Panel"."""
    registry = get_registry()
    out = []
    for i in sorted(ids):
        kpi = registry.get(i)
        if kpi is None:
            continue
        name = kpi.name
        if "(" in name and name.rstrip().endswith(")"):
            name = name[name.rindex("(") + 1 : -1]
        out.append(name)
    if len(out) <= 1:
        return "".join(out)
    return ", ".join(out[:-1]) + " and " + out[-1]


def axis_served_clause(axis: str) -> str:
    """ "Served ONLY by <the KPIs that actually bind it>" — the sentence that was
    hand-written in four Field descriptions and named the wrong KPIs."""
    return f"Served ONLY by {short_names(axis_kpi_ids(axis))} (#1911)"


def breakdown_kpi_names() -> str:
    """The KPIs a patient-axis breakdown may be asked of, for the prompt blocks."""
    return short_names(axis_kpi_ids("segment"))


def panel_trx_name() -> str:
    """The KPI a "share by tier" ask is actually answered by (owner #14's
    destination), taken from share_axis rather than restated here."""
    from src.kpi.share_axis import TRX_SHARE_KPI_ID

    target = redirect_for(TRX_SHARE_KPI_ID, "segment")
    return short_names([target]) if target else ""


def breakdown_guidance_block() -> str:
    """The ``- BREAKDOWN GUIDANCE:`` block both chat system prompts carry.

    Generated, because the hand-written copies told the model to call TRx / NRx /
    NBRx for a patient-axis breakdown and those names resolve to the CANONICAL ids,
    which refuse it — the prompt instructed the model to make the call the tool
    refuses. Both surfaces substitute this ONE string, so they cannot drift apart
    and neither can drift from the gate.
    """
    from src.kpi.share_axis import BRAND_ONLY_AXIS_BRANDS

    volume = short_names(axis_kpi_ids("biologic"))  # the brand-scoped axes' servers
    every_axis = breakdown_kpi_names()
    trx = panel_trx_name()
    brands = ", ".join(sorted(BRAND_ONLY_AXIS_BRANDS))
    return (
        f"For {every_axis} patient-segment breakdowns, call `kpi_calculate_tool` once per "
        "bucket of ONE axis and present the results as a table. Axes: `segment` ∈ "
        "{low_severity, medium_severity, high_severity}; `therapy_line` ∈ {0,1,2,3}; and "
        f"FOR {brands.upper()} ONLY ({volume}, NOT Conversion Rate) `biologic` ∈ "
        "{naive, experienced} and `ige_tier` ∈ {low, medium, high}. "
        "TRx share is NOT defined on a patient axis (each patient is on one tracked brand, "
        "so a per-bucket portfolio share mixes indications; the tool refuses it): for a "
        '"share by tier / line / biologic status / IgE tier" ask, call '
        f"{trx} once per bucket and present each bucket's {trx} with its % of the brand's "
        f"{trx} total, labelled the within-brand mix. A volume axis's buckets sum to the "
        "head-line KPI, so the breakdown reconciles with the total (rates don't sum — their "
        'numerators/denominators do). When the user names a period ("last year", "Q1 2025"), '
        "ALWAYS pass `window` too — it composes with `segment`/`therapy_line` for "
        # The WINDOW-AWARE set, and keyed to therapy_line: CATE serves segment but
        # not therapy_line, so naming it for BOTH would overstate by one KPI.
        f"{short_names(axis_kpi_ids('therapy_line', window=True))}. "
        "Use exactly one axis per breakdown — they are mutually exclusive."
    )


def axis_rules_prose() -> str:
    """The chat capability catalog's ``AXIS_RULES`` sentence, derived.

    ⚠ ITS CONTENT WAS ALREADY CORRECT — it names the Panel KPIs and says plainly
    that the canonical series carry no patient dimension. That is exactly why it
    belongs here: being right ONCE is not the same as being derived, and this is
    the copy that would go stale next. Generating it changes no wording today; it
    removes the hand-maintenance.
    """
    panel = short_names(axis_kpi_ids("biologic"))
    canonical = short_names(["WS3-BI-005", "WS3-BI-006", "WS3-BI-007"])
    return (
        "Breakdown axes, AT MOST ONE per ask: segment = patient severity tier "
        "(low/medium/high); therapy_line = line of therapy (0-3); region = US census "
        "region (northeast/south/midwest/west); and - Remibrutinib ONLY - biologic "
        "status (naive/experienced) or ige_tier (low/medium/high). "
        f"The patient axes are served for the PATIENT-PANEL KPIs {panel} "
        "(all four axes) and Conversion Rate (segment/therapy_line only), plus CATE by "
        "segment; NO other KPI can be broken down by a patient axis. "
        f"The canonical {canonical} series do NOT support a patient axis: they are "
        "brand x region x calendar month and carry no patient dimension - ask for the "
        "matching Panel KPI by that axis. "
        "NEITHER share supports a patient axis: TRx Share Panel because each patient is "
        "on one tracked brand (a share by tier is TRx Panel by tier as the within-brand "
        "mix), and canonical TRx Share because it carries no patient dimension at all; "
        "both redirect to TRx Panel by that axis. "
        f"The time window composes with any one axis for {panel}; with region for "
        # The share is part of THIS list: 008 serves region under a window like its
        # three volume siblings, so it is one four-item list, not three plus one.
        f"canonical {short_names(['WS3-BI-005', 'WS3-BI-006', 'WS3-BI-007', 'WS3-BI-008'])}; "
        "only with segment/therapy_line for "
        "Conversion Rate; with no axis at all for TRx Share Panel; and only with region "
        "for Trigger Precision, Acceptance Rate, Override Rate and Trigger Funnel "
        "Conversion. "
        "TRx share is share of the tracked 3-brand portfolio, NOT share versus "
        "competitors."
    )


def composed_caveat(kpi_meta) -> Optional[str]:
    """The AUTHORED measurement caveat plus the DERIVED capability sentence.

    Composed, not regenerated: owner #14 ruled the two shares keep DIFFERENT
    authored reasons because each is true of its own substrate, so the authored
    text is carried through verbatim and the derived half is appended. Flattening
    both into one generated sentence is exactly what #14 rejected.

    This reaches the LLM: ``src.insights.data_constraint_context`` renders it into
    the constraint block, which is why the capability half belongs here rather
    than being hand-written into 49 YAML strings.
    """
    authored = kpi_meta.measurement_caveat
    authored = " ".join(str(authored).split()) if authored else ""
    served = sorted(axis for axis in PATIENT_AXIS_NAMES if serves(str(kpi_meta.id), axis))
    # Nothing authored AND no patient axis to describe: stay silent rather than
    # append a sentence to every KPI in the block. The constraint block is put in
    # front of the LLM, so noise there is not free.
    if not authored and not served:
        return None
    if served:
        derived = f"Patient-axis breakdowns available: {', '.join(served)}."
    else:
        target = redirect_for(str(kpi_meta.id), "segment")
        derived = "Carries no patient axis" + (
            f"; ask {short_names([target])} by that axis instead." if target else "."
        )
    return f"{authored} {derived}".strip() or None


def forecast_guidance_block() -> str:
    """Tell the model that the platform can now forecast, and what that forecast is NOT.

    Demo 6.5 ("forecast Kisqali TRx for the next two quarters and tell me the biggest
    risk to that forecast") refused on every run from 2026-07-29 partly because neither
    system prompt named a forecaster, so the model reached for causal_analysis_tool.
    Naming the tool is half the fix; the other half is the boundary around it.

    THE BOUNDARY IS THE POINT. 6.5 asks two questions in one breath, and only one of
    them is forecastable. Every model behind forecast_kpi_tool is univariate: it sees
    the brand's own past volume and nothing else, so a competitor entry, a payer change
    or a regional step inside the window moves nothing in the fit. Without this text the
    model narrates a risk as though the forecast had priced it in — confidently wrong
    about precisely what the user asked for. The metric list is derived from the
    forecaster's own SUPPORTED_METRICS, so the prompt can never offer a KPI the tool
    would refuse.
    """
    from src.kpi.canonical_volume_series import SUPPORTED_METRICS

    metrics = ", ".join(m.upper() for m in SUPPORTED_METRICS)
    return (
        "- FORECASTING: use `forecast_kpi_tool` for any FORWARD-looking volume question "
        f"({metrics} only — these are the KPIs stored as a canonical brand x region "
        "monthly series). It backtests Holt-Winters and, when the forecast worker is "
        "running, TimesFM 2.5 over the same rolling origins, serves the lower-error "
        "model, and returns a per-month prediction band that IS that model's measured "
        "error — present it as measured error, not as a confidence interval, and cite "
        "`data_through` and the champion's backtest MAPE. `kpi_calculate_tool` reports "
        "what a month WAS and never forecasts; `e2i_data_query_tool"
        "(query_type='predictions')` searches stored prediction memories and never "
        "forecasts either — do not substitute them for a forecast ask, and do not "
        "extrapolate a series yourself. THE FORECAST IS UNIVARIATE: it is fitted on the "
        "brand's own past volume alone, so it cannot see a competitor launch, a payer or "
        "formulary change, a label change or a regional step change inside the window, "
        "however large. When the user also asks about RISK, say that plainly and get the "
        "risk half from the gap analyzer (via `orchestrator_tool` or "
        "`tool_composer_tool`) and `causal_analysis_tool` — never narrate a risk as if "
        "the forecast had accounted for it."
    )


def twin_simulation_guidance_block() -> str:
    """Tell the model that the platform simulates interventions, and where (#2211).

    On 2026-09-22, with three brands simulable, the chat answered "use the digital twin to
    simulate an email campaign for Kisqali" with *"the platform doesn't include a digital
    twin simulation capability"* and called causal_analysis_tool — reproduced 4/4 offline
    through the real chat leg, because neither prompt nor any bound tool named a twin, a
    simulation or a counterfactual. Naming the tool is half the fix; the other half is the
    boundary: causal_analysis_tool reports drivers OBSERVED in the registry ("what drives
    Kisqali conversion" stays there), the twin simulates an intervention FORWARD. The
    intervention list is read from the side-effect-free contract module: importing any
    ``src.digital_twin`` module costs 17 s and +548 MB (measured), and this renders at
    import of both route modules.
    """
    from src.data.per_hcp_cohort_columns import INTERVENTION_TREATMENT_MAP

    catalog = ", ".join(INTERVENTION_TREATMENT_MAP)
    return (
        "- DIGITAL TWIN SIMULATION: use `digital_twin_simulate_tool` for any SIMULATION / "
        "COUNTERFACTUAL / WHAT-IF-INTERVENTION ask — 'use the digital twin', 'simulate an "
        "email campaign for Kisqali', 'run a counterfactual', 'what would happen to <brand> "
        "conversion if we increased call frequency'. It runs the Digital Twin engine behind "
        "the Digital Twin page's POST /api/digital-twin/simulate on the brand's per-HCP twin "
        f"cohort (interventions: {catalog}; brands with an active twin model) and returns the "
        "simulated effect on HCP conversion with its 95% interval, per-region effects, a "
        "DEPLOY / REFINE / SKIP recommendation and the per-arm experiment size — present them "
        "as a simulation on a synthetic-gold cohort (the payload says so), never as an "
        "observed result. The platform HAS this capability: never say it has no digital twin, "
        "no simulation or no counterfactual tool; if the tool refuses, report its stated "
        "reason and the interventions it does serve. `causal_analysis_tool` reports modeled "
        "DRIVERS from the causal-path registry (what drives / caused / impacts a KPI) and never "
        "simulates an intervention — do not substitute it for a simulation ask. Do not send a "
        "simulation ask to `orchestrator_tool` (experiment_designer's twin simulation is "
        "disabled by design) and use `tool_composer_tool` only when the ask also needs other "
        "steps (a comparison, a gap, a forecast) — its `counterfactual_simulator` is the same "
        "engine."
    )


def render_blocks(prompt: str) -> str:
    """Substitute every capability-derived block into a system prompt.

    One entry point so a new generated block never costs another line in the two
    ratchet-pinned prompt modules. ``{capability_guidance}`` is the SHARED slot: it
    renders every guidance bullet, so adding the next one costs a function here and
    nothing at all in copilotkit.py or chatbot_graph.py. ``{breakdown_guidance}`` is
    kept as the narrower legacy slot for any surface that wants only that bullet.
    """
    guidance = "\n".join(
        [
            f"- BREAKDOWN GUIDANCE: {breakdown_guidance_block()}",
            forecast_guidance_block(),
            twin_simulation_guidance_block(),
        ]
    )
    return prompt.replace("{capability_guidance}", guidance).replace(
        "{breakdown_guidance}", breakdown_guidance_block()
    )
