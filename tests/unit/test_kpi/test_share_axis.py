"""src.kpi.share_axis: the one statement of why TRx Share has no patient axis."""

import pytest

from src.kpi import share_axis as sa
from src.kpi.registry import get_registry


def test_kpi_ids_and_trx_name_match_the_registry():
    share = get_registry().get(sa.TRX_SHARE_KPI_ID)
    trx = get_registry().get(sa.TRX_KPI_ID)
    assert share is not None and share.name == "TRx Share"
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
