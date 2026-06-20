from src.kpi.models import KPIResult


def test_window_fields_default():
    r = KPIResult(kpi_id="WS3-BI-006", value=1.0)
    assert r.window_requested is None
    assert r.window_applied is None
    assert r.window_status == "default"


def test_window_fields_set():
    r = KPIResult(kpi_id="WS3-BI-006", value=1.0,
                  window_requested={"start": "a", "end": "b"},
                  window_applied={"start": "a", "end": "b"},
                  window_status="applied")
    assert r.window_status == "applied"
    assert r.window_applied == {"start": "a", "end": "b"}
