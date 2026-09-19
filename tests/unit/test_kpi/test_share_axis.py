"""src.kpi.share_axis: the one statement of why TRx Share has no patient axis."""

import pytest

from src.kpi import share_axis as sa
from src.kpi.registry import get_registry


def test_kpi_ids_and_trx_name_match_the_registry():
    """REWRITTEN to the post-merge contract under OWNER DECISION #11 (#2114), not loosened.

    #2137 wrote this against the pre-lane substrate, where the share KPI was the canonical
    WS3-BI-008 ("TRx Share") and the redirect target was canonical TRx WS3-BI-005. The lane
    makes business_metrics canonical and moves the patient panel to WS3-BI-011..014, so the
    share with no patient-axis breakdown is now PANEL share WS3-BI-014 and the within-brand
    mix lives on PANEL TRx WS3-BI-011. `share.name == "TRx Share"` pinned 008's name and had
    to move with it.

    The assertion is kept as strong as it was: both constants must resolve to real KPIs, and
    the share must be the PANEL share rather than any KPI that happens to exist."""
    share = get_registry().get(sa.TRX_SHARE_KPI_ID)
    trx = get_registry().get(sa.TRX_KPI_ID)
    assert (
        share is not None
        and share.name == "Observed Rx Events - Patient Panel TRx Share (TRx Share Panel)"
    )
    assert trx is not None and trx.name == sa.TRX_NAME


def test_patient_axes_match_the_chat_tool_labels():
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_LABELS

    assert dict(sa.PATIENT_AXES) == _PATIENT_AXIS_LABELS


@pytest.mark.parametrize(
    "context,expected",
    [
        ({}, None),
        ({"region": "northeast"}, None),
        ({"segment": None}, None),
        ({"therapy_line": 0}, ("therapy_line", "line of therapy")),
        ({"ige_tier": "high", "segment": "low_severity"}, ("segment", "severity tier")),
    ],
)
def test_requested_patient_axis(context, expected):
    assert sa.requested_patient_axis(context) == expected


def test_reason_names_the_tautology_only_on_brand_only_axes():
    assert "always 100%" in sa.share_axis_reason("biologic", "biologic status")
    assert "always 100%" in sa.share_axis_reason("ige_tier", "IgE tier")
    assert "always 100%" not in sa.share_axis_reason("segment", "severity tier")
    assert "one tracked brand" in sa.share_axis_reason("segment", "severity tier")
