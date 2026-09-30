"""#2297: the pipeline's data types live in ``pipeline_types``; ``pipeline`` re-exports them.

``pipeline.py`` sat at the module-size ratchet's LIMIT (1500), so the next added line
would fail CI. The dataclasses and enum carry no behaviour, so they move to their own
module. These tests pin the move itself (the classes are *defined* in ``pipeline_types``,
not merely re-exported the other way round) and that every existing import path still
resolves to the very same objects.
"""

from __future__ import annotations

import pytest

from src.agents import tier_0
from src.agents.tier_0 import pipeline, pipeline_types

NAMES = ("PipelineConfig", "PipelineResult", "PipelineStage")


@pytest.mark.parametrize("name", NAMES)
def test_type_is_defined_in_pipeline_types(name: str) -> None:
    assert getattr(pipeline_types, name).__module__ == "src.agents.tier_0.pipeline_types"


@pytest.mark.parametrize("name", NAMES)
def test_every_import_path_is_the_same_object(name: str) -> None:
    obj = getattr(pipeline_types, name)
    assert getattr(pipeline, name) is obj
    assert getattr(tier_0, name) is obj


def test_pipeline_result_still_defaults_like_before() -> None:
    result = pipeline_types.PipelineResult(
        pipeline_run_id="r",
        status="completed",
        current_stage=pipeline_types.PipelineStage.COMPLETED,
    )
    assert result.stages_completed == [] and result.errors == [] and result.warnings == []
    assert pipeline_types.PipelineConfig().skip_deployment is False
