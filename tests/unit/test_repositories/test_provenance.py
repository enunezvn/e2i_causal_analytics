"""SSOT provenance helper: default-exclude predicate + covariate drop-list."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.repositories.provenance import (
    PROVENANCE_DROP_COLS,
    apply_provenance_filter,
    drop_provenance_cols,
)


@pytest.mark.unit
def test_is_synthetic_in_drop_cols():
    assert "is_synthetic" in PROVENANCE_DROP_COLS


@pytest.mark.unit
def test_drop_provenance_cols_removes_tag_only():
    df = pd.DataFrame(
        {"treatment": [0, 1], "outcome": [1.0, 2.0], "is_synthetic": [True, True], "x1": [3, 4]}
    )
    out = drop_provenance_cols(df)
    assert "is_synthetic" not in out.columns
    assert list(out.columns) == ["treatment", "outcome", "x1"]


@pytest.mark.unit
def test_apply_filter_default_excludes():
    q = MagicMock()
    apply_provenance_filter(q, include_synthetic=False)
    q.eq.assert_called_once_with("is_synthetic", False)


@pytest.mark.unit
def test_apply_filter_opt_in_is_noop():
    q = MagicMock()
    out = apply_provenance_filter(q, include_synthetic=True)
    q.eq.assert_not_called()
    assert out is q


# ---------------------------------------------------------------------------
# Issue #883 §4: strict provenance opt-in parser (shared SSOT)
# ---------------------------------------------------------------------------

from src.repositories.provenance import coerce_provenance_flag  # noqa: E402


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, "true", "TRUE", " true ", "1", "yes", "Yes"])
def test_coerce_provenance_flag_opts_in(value):
    assert coerce_provenance_flag(value) is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [False, "false", "False", "0", "no", "", None, 1, 0, 1.0, [], {}, ["true"], {"opt": True}],
)
def test_coerce_provenance_flag_ambiguity_fails_closed(value):
    """Anything that is not an explicit opt-in stays real-mode — the loose
    ``bool()`` this replaces turned a string "false" opt-OUT into True."""
    assert coerce_provenance_flag(value) is False


# ---------------------------------------------------------------------------
# WS-SYNTH: deployment-level synthetic visibility (showcase / synthetic-gold
# instance). ``E2I_INCLUDE_SYNTHETIC`` flips the SSOT read-path chokepoint to
# INCLUDE synthetic rows by default — reversibly (unset restores the strict
# real-mode gate, prod-safe) and honestly (callers still badge the figures via
# ``data_source="synthetic"``). Generalizes the KPI-only
# ``E2I_KPI_INCLUDE_SYNTHETIC`` idiom (synthetic_mode.py) to every reader.
# ---------------------------------------------------------------------------

from src.repositories.provenance import deployment_includes_synthetic  # noqa: E402


@pytest.mark.unit
def test_deployment_includes_synthetic_default_false(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    assert deployment_includes_synthetic() is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["1", "true", "TRUE", " true ", "yes", "Yes"])
def test_deployment_includes_synthetic_truthy(monkeypatch, value):
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", value)
    assert deployment_includes_synthetic() is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "false", "no", "", "off"])
def test_deployment_includes_synthetic_falsey(monkeypatch, value):
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", value)
    assert deployment_includes_synthetic() is False


@pytest.mark.unit
def test_apply_filter_includes_when_deployment_flag_set(monkeypatch):
    """Showcase instance: even a default/explicit real-mode read INCLUDES
    synthetic so the synthetic-gold substrate powers every surface."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    q = MagicMock()
    out = apply_provenance_filter(q, include_synthetic=False)
    q.eq.assert_not_called()
    assert out is q


@pytest.mark.unit
def test_apply_filter_strict_when_deployment_flag_unset(monkeypatch):
    """Real-data prod (flag unset): the strict default-exclude gate is intact."""
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    q = MagicMock()
    apply_provenance_filter(q, include_synthetic=False)
    q.eq.assert_called_once_with("is_synthetic", False)
