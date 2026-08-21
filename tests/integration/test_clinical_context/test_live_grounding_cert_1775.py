"""LIVE certification of #1775 analysis grounding, against the DEPLOYED API.

Why this lives in the repo rather than staying a scratch probe: codex found the
same class of gap in the certification TWICE — first that every check probed only
Kisqali, then that adding Fabhalta still left the third curated brand untested. A
single-brand certification for a multi-brand feature is a green light that means
less than it appears to, and the fix has to be permanent or the next brand added
re-opens it.

So the brand list is derived from the CURATED REGISTRY, not hardcoded. Adding a
brand to ``BRAND_CLINICAL_MAP`` without grounding working for it now fails here
instead of certifying green.

Opt-in twice over. It needs the deployed API plus admin credentials, so it is
``slow`` + ``requires_network`` (deselected on the PR lane) AND gated on
``E2I_RUN_LIVE_CERTS=1`` — without the flag it skips even where credentials
resolve, so a plain ``pytest tests/`` on this box can never hit prod by accident.

    E2I_RUN_LIVE_CERTS=1 pytest tests/integration/test_clinical_context/\
test_live_grounding_cert_1775.py -v

Secrets are read from the environment inside the process and never printed.
"""

from __future__ import annotations

import os
import re

import httpx
import pytest

from src.services.clinical_context.brand_map import BRAND_CLINICAL_MAP
from tests.integration.test_clinical_context._live_gate import requires_network

pytestmark = [pytest.mark.integration, pytest.mark.slow, requires_network]

_OPT_IN = pytest.mark.skipif(
    os.getenv("E2I_RUN_LIVE_CERTS") != "1",
    reason="Live cert against the deployed API; set E2I_RUN_LIVE_CERTS=1 to run (#1775).",
)

API = os.getenv("E2I_CERT_API", "https://eznomics.site/api")
# Every reference we render must LOOK like a PI cross-reference. This is the check
# that would have caught "label 5 or 6" — a prose parenthetical emitted as a
# citation — before an audit did.
_REFERENCE_FORM = re.compile(r"[\d.,\s-]+|Boxed warning")


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _expected_considerations(brand: str, outcome: str, label: dict) -> set[tuple[str, str, str]]:
    """What THIS code makes of that brand's live label, as (title, references, section)."""
    from src.services.clinical_context.analysis_grounding import ground_analysis
    from src.services.clinical_context.brand_map import (
        resolve_brand_profile,
        treatment_context_for,
    )
    from src.services.clinical_context.label_considerations import (
        DOSAGE_SECTION,
        WARNINGS_SECTION,
        boxed_warning_consideration,
        parse_label_considerations,
    )

    items = []
    for field, section in (
        ("warnings_and_cautions", WARNINGS_SECTION),
        ("dosage_and_administration", DOSAGE_SECTION),
    ):
        items.extend(parse_label_considerations(" ".join(label.get(field) or []), section))
    boxed = boxed_warning_consideration(" ".join(label.get("boxed_warning") or []))
    if boxed is not None:
        items.append(boxed)
    grounding = ground_analysis(
        resolve_brand_profile(brand),
        outcome=outcome,
        treatment_context=treatment_context_for(brand, "copay_support"),
        label_considerations=tuple(items),
        label_source="openfda",
    )
    return {(c.title, c.references, c.section) for c in grounding.label_considerations}


@pytest.fixture(scope="module")
def auth_headers() -> dict[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    key = os.getenv("SUPABASE_KEY") or ""
    password = os.getenv("E2I_ADMIN_PASSWORD") or ""
    missing = [
        name
        for name, value in (
            ("SUPABASE_URL", url),
            ("SUPABASE_KEY", key),
            ("E2I_ADMIN_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        pytest.skip(f"missing credentials (names only): {missing}")
    response = httpx.post(
        f"{url}/auth/v1/token",
        params={"grant_type": "password"},
        headers={"apikey": key, "Content-Type": "application/json"},
        json={"email": os.getenv("E2I_ADMIN_EMAIL", "admin@e2i.local"), "password": password},
        timeout=30.0,
    )
    if response.status_code != 200:
        pytest.skip(f"auth failed: HTTP {response.status_code}")
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _context(headers: dict[str, str], **params: str) -> dict:
    response = httpx.get(
        f"{API}/causal/clinical-context", params=params, headers=headers, timeout=120.0
    )
    assert response.status_code == 200, f"HTTP {response.status_code} for {params}"
    return response.json()


@_OPT_IN
@pytest.mark.parametrize("brand", sorted(BRAND_CLINICAL_MAP))
def test_every_curated_brand_is_grounded_on_the_wire(brand: str, auth_headers) -> None:
    """The gap codex found twice: a brand nobody probes can lose grounding silently.

    Parametrised over the REGISTRY, so this cannot go stale when a brand is added.
    """
    grounding = _context(
        auth_headers, brand=brand, outcome="persistent_180d", treatment="copay_support"
    ).get("analysis_grounding")
    assert grounding is not None, f"{brand}: no grounding on the wire (#1775 defect)"
    # POSITIVE CONTROL. `all(...)` over an empty list is True, and a present-but-empty
    # block satisfies `is not None`; require content before asserting anything about it.
    considerations = grounding["label_considerations"]
    assert considerations, f"{brand}: grounding present but empty"
    assert grounding["outcome_theme"] == "persistence", grounding["outcome_theme"]

    for item in considerations:
        assert _REFERENCE_FORM.fullmatch(item["references"]), (
            f"{brand}: {item['references']!r} is not a cross-reference form"
        )
        assert item["source"] == "openfda", item["source"]


@_OPT_IN
@pytest.mark.parametrize("brand", sorted(BRAND_CLINICAL_MAP))
@pytest.mark.parametrize("outcome", ["persistent_180d", "treatment_initiated"])
def test_the_deployed_grounding_matches_the_label_for_each_brand_and_outcome(
    brand: str, outcome: str, auth_headers
) -> None:
    """Two independent checks per (brand, outcome), because one is not enough.

    codex iter-17 found this test's predecessor could pass with INITIATION grounding
    entirely broken: it required non-empty output only for persistence, then verified
    details "if any were rendered", and its `checked >= 1` control could be satisfied
    by the persistence pass alone. Removing the boxed-warning path would have zeroed
    Fabhalta's initiation grounding and still certified green.

    "Require non-empty for every pair" is the wrong repair — Rhapsido genuinely has
    NO initiation factors, and a cert that demands content where the label has none
    would push us to invent some. So the expectation is DERIVED instead: parse and
    select from that brand's own live label here, and require the deployed answer to
    match exactly. Zero is then asserted as zero where zero is real, and Fabhalta's
    single boxed-warning item is asserted as present because it is real.

    The verbatim check stays alongside it. The exactness check compares the deploy to
    THIS code, so it cannot see a defect both share; the verbatim check compares it to
    the LABEL, which no defect of ours can move.
    """
    profile = BRAND_CLINICAL_MAP[brand]
    fda = httpx.get(
        "https://api.fda.gov/drug/label.json",
        params={"search": f'openfda.generic_name:"{profile.drug_name}"', "limit": 1},
        timeout=60.0,
    )
    if fda.status_code != 200:
        pytest.skip(f"openFDA unavailable for {profile.drug_name}: HTTP {fda.status_code}")
    label = (fda.json().get("results") or [{}])[0]

    expected = _expected_considerations(brand, outcome, label)
    grounding = _context(auth_headers, brand=brand, outcome=outcome, treatment="copay_support").get(
        "analysis_grounding"
    )
    assert grounding is not None, f"{brand}/{outcome}: no grounding on the wire"
    served = {
        (item["title"], item["references"], item["section"])
        for item in grounding["label_considerations"]
    }
    assert served == expected, (
        f"{brand}/{outcome}: deployed grounding differs from this label.\n"
        f"  missing from deploy: {sorted(expected - served)}\n"
        f"  extra in deploy:     {sorted(served - expected)}"
    )

    sections = {
        "warnings_and_cautions": "warnings_and_cautions",
        "dosage_and_administration": "dosage_and_administration",
        "contraindications": "contraindications",
        "boxed_warning": "boxed_warning",
    }
    for item in grounding["label_considerations"]:
        field = sections.get(item["section"])
        assert field, f"{brand}: unknown section {item['section']!r}"
        haystack = _norm(" ".join(label.get(field) or []))
        assert haystack, f"{brand}: label has no {field}"
        assert _norm(item["detail"]) in haystack, (
            f"{brand}/{outcome}: detail not verbatim in {field}: {item['detail'][:80]!r}"
        )


@_OPT_IN
def test_the_expectations_are_not_all_empty(auth_headers) -> None:
    """POSITIVE CONTROL for the parametrised test above.

    Every one of its assertions is satisfied by `set() == set()`, so if the local
    parser returned nothing for every brand the whole matrix would pass while the
    feature was dead. Require the expectations themselves to have content — including
    at least one INITIATION pair, which is the case the previous cert could not see.
    """
    totals: dict[str, int] = {}
    for brand, profile in BRAND_CLINICAL_MAP.items():
        fda = httpx.get(
            "https://api.fda.gov/drug/label.json",
            params={"search": f'openfda.generic_name:"{profile.drug_name}"', "limit": 1},
            timeout=60.0,
        )
        if fda.status_code != 200:
            continue
        label = (fda.json().get("results") or [{}])[0]
        for outcome in ("persistent_180d", "treatment_initiated"):
            totals[f"{brand}/{outcome}"] = len(_expected_considerations(brand, outcome, label))
    assert totals, "no expectations computed at all"
    assert sum(totals.values()) >= 3, totals
    assert any(
        count >= 1 for key, count in totals.items() if key.endswith("treatment_initiated")
    ), f"no brand has initiation grounding; the matrix above would be vacuous: {totals}"


@_OPT_IN
def test_the_1763_boundary_survives_for_every_brand(auth_headers) -> None:
    """#1763's honesty boundary must hold everywhere grounding was added: the label
    is silent on a commercial lever, and grounding it must not imply otherwise."""
    for brand in sorted(BRAND_CLINICAL_MAP):
        body = _context(
            auth_headers, brand=brand, outcome="persistent_180d", treatment="copay_support"
        )
        note = (body["analysis_grounding"] or {}).get("note", "").lower()
        assert "commercial access lever" in note, brand
        assert "says nothing about it" in note, brand
        evidence = body.get("causal_evidence") or {}
        assert evidence.get("status") == "commercial_lever", (brand, evidence.get("status"))
        assert not evidence.get("citations"), brand
        # every claim connecting the label to the lever must be negated WHERE IT STANDS
        for match in re.finditer(r"the label (?:speaks to|supports|covers|describes)", note):
            preceding = note[max(0, match.start() - 60) : match.start()]
            assert any(n in preceding for n in ("not ", "none", "nothing", "never", "no ")), (
                brand,
                note[max(0, match.start() - 60) : match.end() + 40],
            )
