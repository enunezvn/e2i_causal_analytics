"""OpenAPI /api/docs tag-metadata guards.

Every tag used by a route must appear in an ``x-tagGroups`` group and carry a
description: ReDoc (/api/redoc) HIDES endpoints whose tag is not in any group,
so a new router tag that skips registration silently vanishes from the rendered
docs (happened to Sentinels/Alerts/Executive Insights/Strategic Insights).

Version consistency is pinned separately in
tests/unit/test_api/test_version_single_source.py.
"""

from src.api.main import app


def _openapi_schema() -> dict:
    return app.openapi()


def _tags_used_by_routes(schema: dict) -> set[str]:
    return {
        tag
        for path_item in schema["paths"].values()
        for operation in path_item.values()
        if isinstance(operation, dict)
        for tag in operation.get("tags", [])
    }


def test_every_route_tag_appears_in_a_tag_group():
    schema = _openapi_schema()
    grouped = {tag for group in schema["x-tagGroups"] for tag in group["tags"]}
    ungrouped = _tags_used_by_routes(schema) - grouped
    assert not ungrouped, (
        f"Tags {sorted(ungrouped)} are used by routes but missing from x-tagGroups; "
        "ReDoc hides their endpoints. Add them to a group in src/api/main.py."
    )


def test_every_route_tag_has_a_description():
    schema = _openapi_schema()
    described = {tag["name"] for tag in schema.get("tags", []) if tag.get("description")}
    undescribed = _tags_used_by_routes(schema) - described
    assert not undescribed, (
        f"Tags {sorted(undescribed)} have no description in openapi_tags "
        "(src/api/main.py); they render bare in Swagger UI and ReDoc."
    )
