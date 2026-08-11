#!/usr/bin/env python3
"""
Generate the frontend KPI catalog from config/kpi_definitions.yaml.

The chat's chart action has to route ANY of the 44 registry KPIs the user can
name, so the alias table can't be a hand-maintained subset (it was: 6 of 44,
and the registry-code regex missed the CM-* causal family entirely). This
script is the single source of truth hop: YAML registry -> TypeScript catalog.

Re-run after editing config/kpi_definitions.yaml:

    python3 scripts/gen_kpi_catalog.py

Semantic types are Flint's (see `SemanticTypes` in the flint-chart package) and
drive axis formatting: 'Percentage' gets percent ticks, 'Count' gets
integer-only ticks, 'Duration' gets day/hour units. They are derived from
declared registry facts (value_format, unit, threshold ranges) plus an explicit
per-family override map -- never guessed from the KPI name.

@module scripts/gen_kpi_catalog
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "config" / "kpi_definitions.yaml"
OUT_PATH = REPO_ROOT / "frontend" / "src" / "lib" / "kpi-catalog.generated.ts"
ENUM_LABELS_PATH = REPO_ROOT / "src" / "services" / "enum_labels.py"


def load_region_vocabulary() -> tuple[tuple[str, ...], dict[str, str]]:
    """(labels, folded alias -> label) from the platform SSOT (#1538).

    ``src/services/enum_labels.py`` owns the region_type enum labels and the
    one region synonym table (REGION_ALIASES) every backend surface shares.
    Loaded file-scoped (importlib, not ``import src...``) so this generator
    never triggers the src package init — enum_labels is a pure leaf (re +
    typing only) and stays importable standalone.
    """
    spec = importlib.util.spec_from_file_location("e2i_enum_labels", ENUM_LABELS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.REGION_ENUM_LABELS, dict(module.REGION_LABEL_BY_ALIAS)

WORKSTREAM_KEYS = [
    "ws1_data_quality",
    "ws1_model_performance",
    "ws2_triggers",
    "ws3_business",
    "brand_specific",
    "causal_metrics",
]

# Explicit per-KPI semantic types where the registry carries no unit/value_format
# strong enough to derive one. Each entry is a declared modelling decision, not a
# heuristic: model-quality scores are unitless 0-1 scores (NOT percentages -- an
# ROC-AUC axis labelled "85%" is wrong), Rx volumes and user counts are integer
# counts, ROI and causal effect sizes are signed ratios that must not be forced
# to a zero-based percent axis.
SEMANTIC_OVERRIDES: dict[str, str] = {
    # WS1 model performance: unitless quality scores on their own scales.
    "WS1-MP-001": "Score",  # ROC-AUC
    "WS1-MP-002": "Score",  # PR-AUC
    "WS1-MP-003": "Score",  # F1
    "WS1-MP-004": "Score",  # Recall@Top-K
    "WS1-MP-005": "Score",  # Brier
    "WS1-MP-006": "Score",  # Calibration slope deviation
    "WS1-MP-009": "Score",  # Feature drift (PSI)
    # WS3 volumes / population counts: integers, never fractional.
    "WS3-BI-001": "Count",  # MAU
    "WS3-BI-002": "Count",  # WAU
    "WS3-BI-005": "Count",  # TRx
    "WS3-BI-006": "Count",  # NRx
    "WS3-BI-007": "Count",  # NBRx
    "WS3-BI-010": "Number",  # ROI is a signed multiple, not a percentage
    # Causal metrics: signed effect sizes; a zero-based percent axis would hide
    # negative effects, which are the whole point of reading a CI.
    "CM-001": "Number",
    "CM-002": "Number",
    "CM-003": "Number",
    "CM-004": "Number",
    "CM-005": "Number",
}

UNIT_SEMANTICS: dict[str, str] = {
    "days": "Duration",
    "hours": "Duration",
    "points (1-7 scale)": "Score",
}


def derive_semantic_type(kpi: dict[str, Any]) -> str:
    """Pick a Flint semantic type from declared registry facts."""
    kpi_id = kpi["id"]
    if kpi_id in SEMANTIC_OVERRIDES:
        return SEMANTIC_OVERRIDES[kpi_id]
    if kpi.get("value_format") == "percent":
        return "Percentage"
    unit = kpi.get("unit")
    if unit in UNIT_SEMANTICS:
        return UNIT_SEMANTICS[unit]
    # A threshold band wholly inside 0..1 means the KPI is a rate.
    threshold = kpi.get("threshold") or {}
    bounds = [v for v in threshold.values() if isinstance(v, (int, float))]
    if bounds and all(0 <= v <= 1 for v in bounds):
        return "Percentage"
    return "Number"


def alias_forms(kpi_id: str, yaml_key: str, name: str) -> list[str]:
    """Every string a user or model might plausibly type for this KPI.

    Normalized the same way `normalizeAlias` in kpi-alias.ts normalizes lookups:
    lowercase, runs of space/hyphen/underscore collapsed to a single underscore.
    Parenthesised abbreviations in registry names ("Monthly Active Users (MAU)")
    are additionally emitted bare, since that is what people actually say.
    """

    def norm(value: str) -> str:
        out, prev_sep = [], False
        for ch in value.lower():
            if ch in " -_/":
                prev_sep = True
                continue
            if prev_sep and out:
                out.append("_")
            prev_sep = False
            out.append(ch)
        return "".join(out)

    forms = {norm(kpi_id), norm(yaml_key), norm(name)}
    # "Total Prescriptions (TRx)" -> also "trx"; "Remi - Intent-to-Prescribe Δ"
    # -> also the pre-parenthesis stem.
    if "(" in name and ")" in name:
        inner = name[name.index("(") + 1 : name.rindex(")")]
        stem = name[: name.index("(")]
        forms.add(norm(inner))
        forms.add(norm(stem))
    forms.discard("")
    return sorted(forms)


def main() -> None:
    import yaml

    data = yaml.safe_load(YAML_PATH.read_text())
    entries: list[dict[str, Any]] = []
    for ws_key in WORKSTREAM_KEYS:
        for yaml_key, kpi in (data.get(ws_key) or {}).items():
            threshold = kpi.get("threshold") or {}
            target = threshold.get("target")
            entries.append(
                {
                    "id": kpi["id"],
                    "key": yaml_key,
                    "name": kpi.get("name", yaml_key),
                    "workstream": ws_key,
                    "semanticType": derive_semantic_type(kpi),
                    # Drawn as the goal marker on a KPI Card. Omitted for KPIs
                    # with no declared target (causal metrics, Rx volumes) --
                    # inventing one would put a fake benchmark on a real chart.
                    **({"target": target} if isinstance(target, (int, float)) else {}),
                    "aliases": alias_forms(kpi["id"], yaml_key, kpi.get("name", yaml_key)),
                }
            )

    # An alias that maps to two KPIs would silently route to whichever came
    # last. Fail loudly instead -- a wrong-KPI chart is worse than no chart.
    seen: dict[str, str] = {}
    for entry in entries:
        for alias in entry["aliases"]:
            if alias in seen and seen[alias] != entry["id"]:
                raise SystemExit(
                    f"Ambiguous alias {alias!r}: {seen[alias]} and {entry['id']}"
                )
            seen[alias] = entry["id"]

    region_labels, region_alias_map = load_region_vocabulary()

    body = ",\n".join(
        "  "
        + json.dumps(
            {
                "id": e["id"],
                "key": e["key"],
                "name": e["name"],
                "workstream": e["workstream"],
                "semanticType": e["semanticType"],
                **({"target": e["target"]} if "target" in e else {}),
                "aliases": e["aliases"],
            },
            ensure_ascii=False,
        )
        for e in entries
    )

    OUT_PATH.write_text(
        "/**\n"
        " * KPI catalog (GENERATED -- do not edit by hand)\n"
        " * =============================================\n"
        " *\n"
        " * Every KPI in the registry, with the alias forms a user or model might\n"
        " * type and the Flint semantic type its values should be charted as.\n"
        " *\n"
        " * Source:    config/kpi_definitions.yaml\n"
        " * Generator: scripts/gen_kpi_catalog.py\n"
        " *\n"
        " * Regenerate with `python3 scripts/gen_kpi_catalog.py` after editing the\n"
        " * YAML. `kpi-catalog.test.ts` fails if this file drifts from the YAML.\n"
        " *\n"
        " * @module lib/kpi-catalog.generated\n"
        " */\n\n"
        "/** Flint semantic type governing axis formatting for a KPI's values. */\n"
        "export type KpiSemanticType = 'Percentage' | 'Count' | 'Number' | 'Duration' | 'Score';\n\n"
        "export interface KpiCatalogEntry {\n"
        "  /** Registry code, e.g. 'WS3-BI-005'. */\n"
        "  id: string;\n"
        "  /** YAML key, e.g. 'trx'. */\n"
        "  key: string;\n"
        "  /** Display name, e.g. 'Total Prescriptions (TRx)'. */\n"
        "  name: string;\n"
        "  /** Registry workstream the KPI belongs to. */\n"
        "  workstream: string;\n"
        "  /** Flint semantic type for this KPI's values. */\n"
        "  semanticType: KpiSemanticType;\n"
        "  /** Declared threshold target, drawn as the goal on a KPI Card.\n"
        "   *  Absent for KPIs the registry gives no target. */\n"
        "  target?: number;\n"
        "  /** Normalized strings that resolve to this KPI. */\n"
        "  aliases: string[];\n"
        "}\n\n"
        f"export const KPI_CATALOG: readonly KpiCatalogEntry[] = [\n{body},\n] as const;\n\n"
        "/** region_type enum labels (US census regions) — SSOT: src/services/enum_labels.py (#1538). */\n"
        f"export const REGION_LABELS: readonly string[] = {json.dumps(list(region_labels))} as const;\n\n"
        "/**\n"
        " * Folded region alias -> enum label, mirroring enum_labels.REGION_ALIASES\n"
        " * (the platform's one region synonym table). Keys are folded the way\n"
        " * `fold_region_key` folds: casefolded with space/hyphen/underscore removed —\n"
        " * `resolveRegion` in kpi-alias.ts folds lookups the same way.\n"
        " */\n"
        "export const REGION_ALIAS_MAP: Readonly<Record<string, string>> = "
        f"{json.dumps(dict(sorted(region_alias_map.items())), ensure_ascii=False)} as const;\n"
    )
    print(
        f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(entries)} KPIs, {len(seen)} aliases, "
        f"{len(region_alias_map)} region aliases)"
    )


if __name__ == "__main__":
    main()
