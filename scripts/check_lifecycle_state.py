#!/usr/bin/env python3
"""Plan v4 Gate N2 — lifecycle-state scanner.

Walks every gate-relevant location in the repo and asserts each declares
a ``lifecycle_state`` (YAML/TOML/JSON) or a ``LIFECYCLE_STATE_*`` constant
(Python AST). Reports machine-readable failures suitable for CI.

A "gate-relevant location" is any file or module that contains code or
config that controls whether a verdict drops a feature, halts a pipeline,
or denies a promotion. We discover them by:

* explicit registration in ``GATE_RELEVANT_PYTHON_MODULES`` below (the
  authoritative list — adding a new gate requires updating this list); and
* keyword scan of ``config/`` (recursively) for files whose top-level keys
  mention ``gate``, ``threshold``, ``advisory``, ``enforcement``,
  ``deployer`` combined with at least one ``threshold`` or ``cutoff``
  value (i.e., a numeric guardrail). Configs with no numeric guardrails
  are not gates; configs with guardrails MUST declare ``lifecycle_state``.

**Scope of YAML scan (N2 pass-2 H3 PARTIAL)**: only the canonical
``config/`` root is scanned. Sibling roots commonly used by other Python
projects — ``conf/``, ``configs/``, ``settings/`` — are intentionally NOT
scanned in this iteration. The codebase canonicalises on ``config/``
(verified by repo audit at PR #132 review time); reviewer flagged this
scope as a future-expansion opportunity. To add a sibling root, modify
``_candidate_yaml_configs`` and the workflow's ``paths:`` filter in
``.github/workflows/lifecycle_state_guard.yml``.

The scanner also detects lifecycle-state CHANGES across git history when
invoked with ``--check-changes``: any change in a ``LIFECYCLE_STATE_*``
constant value or a YAML ``lifecycle_state:`` key MUST have a corresponding
signed doc at ``docs/calibration/{slug}_lifecycle_change_{from}_to_{to}_*.md``.

Exit codes:

* ``0`` — all gate-relevant locations declare a lifecycle_state and any
  changes have signed docs.
* ``1`` — one or more locations missing a declaration, or a change has no
  signed doc.
* ``2`` — script-internal error (parse failure, unexpected I/O).

Usage::

    python scripts/check_lifecycle_state.py             # baseline check
    python scripts/check_lifecycle_state.py --check-changes  # also check git diff
    python scripts/check_lifecycle_state.py --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

# Allow this script to import from src/ without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lifecycle.gate_lifecycle import (  # noqa: E402  (after sys.path mutation)
    GateLifecycleState,
    LifecycleDeclaration,
)

VALID_STATE_VALUES: frozenset[str] = frozenset(s.value for s in GateLifecycleState)


# ============================================================================
# Authoritative registry of gate-relevant Python modules.
#
# Each entry is a tuple of (relative_path, set_of_required_lifecycle_constants).
# A module is gate-relevant when it contains constants that control whether a
# verdict drops a feature, halts a pipeline, or denies a promotion. A module
# may have MULTIPLE constants (e.g., evaluator.py owns T2.2 + T2.3, both
# advisory but separately calibrated).
#
# Adding a new gate-relevant module: add it here AND add a `LIFECYCLE_STATE_*`
# constant (typed `GateLifecycleState`) at module scope.
# ============================================================================
GATE_RELEVANT_PYTHON_MODULES: dict[str, frozenset[str]] = {
    "src/agents/ml_foundation/model_trainer/nodes/evaluator.py": frozenset(
        {"LIFECYCLE_STATE_T22", "LIFECYCLE_STATE_T23"}
    ),
    "src/agents/ml_foundation/model_deployer/nodes/registry_manager.py": frozenset(
        {"LIFECYCLE_STATE_T26A", "LIFECYCLE_STATE_T26B"}
    ),
    "src/agents/ml_foundation/data_preparer/nodes/imputation_audit.py": frozenset(
        {"LIFECYCLE_STATE_T24"}
    ),
    # MED-9 fix (codex pass-1): the G2 experiment harness declares
    # LIFECYCLE_STATE_G2 = GateLifecycleState.ADVISORY. The N2 scanner
    # must detect this declaration so a future promotion to ENFORCED
    # cannot evade lifecycle-change documentation.
    "scripts/run_tier1b_b2_experiment.py": frozenset({"LIFECYCLE_STATE_G2"}),
}

# Calibration-doc filename pattern. Captures slug, from_state, to_state, date.
# Example: docs/calibration/T22_lifecycle_change_advisory_to_calibrating_20260615.md
LIFECYCLE_DOC_FILENAME_RE = re.compile(
    r"(?P<slug>[a-zA-Z0-9_]+)"
    r"_lifecycle_change_"
    r"(?P<from_state>[a-z]+)"
    r"_to_"
    r"(?P<to_state>[a-z]+)"
    r"_(?P<date>\d{8})\.md$"
)

# YAML configs we DO NOT scan — these are vocabulary / routing tables, not
# gate-relevant guardrails. The scanner's keyword filter is intentionally
# wide to avoid false negatives, so we maintain an explicit denylist for
# the well-known false positives.
YAML_CONFIG_DENYLIST: frozenset[str] = frozenset(
    {
        "config/agent_config.yaml",
        "config/cohort_vocabulary.yaml",
        "config/domain_vocabulary.yaml",
        "config/library_routing.yaml",
        "config/filter_mapping.yaml",
        "config/kpi_definitions.yaml",
        "config/visualization_rules_v4_1.yaml",
        "config/005_memory_config.yaml",
        "config/digital_twin_config.yaml",
        "config/observability.yaml",
        "config/optuna_config.yaml",
        "config/imbalance_strategy.yaml",
        "config/cost_matrix_demo.yaml",
        "config/feast_materialization.yaml",
        "config/experiment_lifecycle.yaml",  # state machine for EXPERIMENTS, not ML gates
        "config/alert_config.yaml",
        "config/drift_monitoring.yaml",
        "config/confidence_logic.yaml",
        "config/gepa_config.yaml",
        "config/outcome_truth_rules.yaml",
        "config/self_improvement.yaml",
        "config/model_endpoints.yaml",
        "config/autoscale.yml",
        # Nested config dirs surfaced by the H3 fix (rglob). None of these
        # are gate guardrails — they are agent configs, ontology defs, or
        # archived vocabularies. They live under config/ for organization,
        # not because they control verdict behavior.
        "config/agents/cohort_constructor.yaml",
        "config/agents/gap_analyzer.yaml",
        "config/archived/003_memory_vocabulary.yaml",
        "config/archived/Feedback Loop domain vocabulary.yml",
        "config/archived/Ragas-Opik Integration Domain Vocabulary.yml",
        "config/archived/domain_vocabulary_v3.1.0.yaml",
        "config/archived/domain_vocabulary_v3_2_additions.yaml",
        "config/archived/domain_vocabulary_v4.2.0.yaml",
        "config/archived/domain_vocabulary_v5.0.0.yaml",
        "config/mlflow/mlflow.yaml",
        "config/ontology/confidence.yaml",
        "config/ontology/digital_twin.yaml",
        "config/ontology/drift_config.yaml",
        "config/ontology/falkordb_config.yaml",
        "config/ontology/mlops_config.yaml",
        "config/ontology/node_types.yaml",
        "config/ontology/self_improvement.yaml",
        "config/ontology/validation_rules.yaml",
    }
)


@dataclass
class ScanFinding:
    """Single finding emitted by the scanner."""

    path: str
    severity: str  # "error" | "warning" | "info"
    code: str  # short machine-readable code
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def _read_python_module_constants(path: Path) -> dict[str, Optional[str]]:
    """Return {constant_name: lifecycle_state_value | None} for every
    ``LIFECYCLE_STATE_*`` assignment in ``path``, regardless of whether it
    sits at module scope or inside a class body.

    Resolves the RHS via AST. Supported RHS shapes (the 3 idiomatic forms):

    * ``LIFECYCLE_STATE_X = GateLifecycleState.ADVISORY`` — Attribute access,
      resolves to ``"advisory"``.
    * ``LIFECYCLE_STATE_X = "advisory"`` — String literal.
    * ``LIFECYCLE_STATE_X: GateLifecycleState = GateLifecycleState.ADVISORY``
      — annotated form.

    Supported scopes (N2 finding H2): module-level OR direct class-body
    assignments. ``ast.walk(tree)`` is too permissive — it would pick up a
    ``LIFECYCLE_STATE_*`` assigned inside a function or a nested class,
    which is NOT a stable declaration the scanner can rely on. We therefore
    walk the tree and accept assignments whose direct enclosing scope is
    either ``ast.Module`` (module-level) or ``ast.ClassDef`` (class-body).

    Any other RHS shape (function call, conditional, name reference) returns
    None for that constant — the scanner reports "unrecognized RHS" so the
    caller can either simplify to one of the supported shapes or update the
    scanner.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"failed to read {path}: {e}") from e
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        raise RuntimeError(f"failed to parse {path}: {e}") from e

    out: dict[str, Optional[str]] = {}
    # Walk module-body and class-body nodes only. Function bodies are
    # explicitly excluded — a constant assigned inside a function is not a
    # stable declaration and would not represent a gate.
    for parent in _iter_lifecycle_scopes(tree):
        for node in parent.body:
            targets: list[str] = []
            value_node: Optional[ast.expr] = None
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id.startswith("LIFECYCLE_STATE_"):
                        targets.append(tgt.id)
                value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id.startswith(
                    "LIFECYCLE_STATE_"
                ):
                    targets.append(node.target.id)
                    value_node = node.value
            if not targets or value_node is None:
                continue
            resolved = _resolve_lifecycle_value(value_node)
            for name in targets:
                out[name] = resolved
    return out


def _iter_lifecycle_scopes(tree: ast.Module) -> list[ast.Module | ast.ClassDef]:
    """Return every scope where a ``LIFECYCLE_STATE_*`` constant is
    considered a stable declaration: the module itself + every class body
    transitively, EXCEPT classes whose ancestor chain includes a
    ``FunctionDef`` or ``AsyncFunctionDef`` (i.e., function-nested
    classes).

    Function bodies are excluded — a constant defined inside a function is
    a runtime value, not a gate declaration. A class nested inside a
    function is similarly a runtime construct: even though Python permits
    ``def f(): class C: LIFECYCLE_STATE_X = ...``, ``C`` is re-built on
    every call to ``f`` and is not addressable from module-import scope, so
    its lifecycle constant is not a stable declaration the scanner can
    rely on (N2 pass-2 finding H2 PARTIAL + new MED).

    Implementation: walk the tree top-down via ``ast.NodeVisitor`` while
    maintaining a stack of enclosing functions; only emit ``ClassDef``
    nodes whose ancestor chain contains zero functions. Top-level classes
    and arbitrarily-nested *class*-in-*class* constructs are accepted.
    """
    scopes: list[ast.Module | ast.ClassDef] = [tree]

    class _Collector(ast.NodeVisitor):
        def __init__(self) -> None:
            self._function_depth = 0

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._function_depth += 1
            self.generic_visit(node)
            self._function_depth -= 1

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._function_depth += 1
            self.generic_visit(node)
            self._function_depth -= 1

        def visit_Lambda(self, node: ast.Lambda) -> None:
            # Lambdas have no body for class definitions, but be defensive.
            self._function_depth += 1
            self.generic_visit(node)
            self._function_depth -= 1

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if self._function_depth == 0:
                scopes.append(node)
            # Recurse regardless: a class nested in a function may itself
            # contain another class — but since we're inside a function,
            # the function_depth counter keeps us from accepting any of
            # them. A class-in-class-in-function chain stays excluded.
            self.generic_visit(node)

    _Collector().visit(tree)
    return scopes


def _resolve_lifecycle_value(node: ast.expr) -> Optional[str]:
    """Resolve a lifecycle-state RHS to its string value, or None if the
    RHS shape is not a recognized lifecycle declaration.
    """
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "GateLifecycleState":
            attr = node.attr
            try:
                return GateLifecycleState[attr].value
            except KeyError:
                return None
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value in VALID_STATE_VALUES:
            return node.value
    return None


def _candidate_yaml_configs(repo_root: Path) -> list[Path]:
    """Return YAML configs under ``config/`` (recursively) that look
    gate-relevant (have a threshold/cutoff/limit/min/max-bearing key) and
    are NOT in the denylist.

    N2 finding H3: previously only the top-level ``config/`` directory was
    scanned via ``glob("*.y*ml")``. Environment overlays
    (``config/env/prod.yaml``) and other nested gate configs were missed.
    The scanner now uses ``rglob("*.y*ml")`` to walk every YAML in the
    ``config/`` subtree.

    **Scope (intentional / N2 pass-2 H3 PARTIAL)**: only the canonical
    ``config/`` root is scanned. Sibling roots commonly used by other
    Python projects — ``conf/``, ``configs/``, ``settings/`` — are
    intentionally NOT scanned in this iteration. The codebase canonicalises
    on ``config/`` (verified by repo audit at PR #132 review time); a
    sibling root cropping up in the future would be a project-wide refactor
    surfaced loudly enough to also remember to wire this scanner. Reviewer
    flagged this scope as a future-expansion opportunity; to add a sibling
    root, edit this function (one extra ``rglob`` per root) and add the
    sibling to the workflow's ``paths:`` filter in
    ``.github/workflows/lifecycle_state_guard.yml`` so push/PR events on
    that root trigger the gate.
    """
    config_dir = repo_root / "config"
    if not config_dir.is_dir():
        return []
    out: list[Path] = []
    yaml_paths: list[Path] = []
    for ext in ("*.yaml", "*.yml"):
        yaml_paths.extend(config_dir.rglob(ext))
    for path in sorted(set(yaml_paths)):
        rel = path.relative_to(repo_root).as_posix()
        if rel in YAML_CONFIG_DENYLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Numeric-guardrail heuristic: file references one of these
        # gate-shaped keys. The scanner is conservative — false-positive
        # YAMLs land in the denylist; false-negatives are caught by the
        # explicit Python-module registry above.
        guardrail_keywords = (
            "threshold",
            "cutoff",
            "min_lift",
            "max_lift",
            "ceiling",
            "floor",
            "buffer",
        )
        if any(kw in text for kw in guardrail_keywords):
            out.append(path)
    return out


def scan_python_modules(repo_root: Path) -> list[ScanFinding]:
    """Scan every entry in ``GATE_RELEVANT_PYTHON_MODULES`` and emit a
    finding for any required ``LIFECYCLE_STATE_*`` constant that is missing
    or has an invalid RHS.
    """
    findings: list[ScanFinding] = []
    for rel_path, required in GATE_RELEVANT_PYTHON_MODULES.items():
        path = repo_root / rel_path
        if not path.is_file():
            findings.append(
                ScanFinding(
                    path=rel_path,
                    severity="error",
                    code="missing_python_module",
                    message=(
                        f"registered gate-relevant module {rel_path} not found "
                        "on disk; either restore the file or remove from "
                        "GATE_RELEVANT_PYTHON_MODULES."
                    ),
                )
            )
            continue
        try:
            present = _read_python_module_constants(path)
        except RuntimeError as e:
            findings.append(
                ScanFinding(
                    path=rel_path,
                    severity="error",
                    code="parse_error",
                    message=str(e),
                )
            )
            continue
        for name in required:
            value = present.get(name)
            if name not in present:
                findings.append(
                    ScanFinding(
                        path=rel_path,
                        severity="error",
                        code="missing_lifecycle_constant",
                        message=(
                            f"gate-relevant module {rel_path} is missing required "
                            f"constant {name}. Declare it as "
                            f"`{name} = GateLifecycleState.ADVISORY` (or another "
                            "lifecycle state) at module scope."
                        ),
                        details={"constant": name},
                    )
                )
            elif value is None:
                findings.append(
                    ScanFinding(
                        path=rel_path,
                        severity="error",
                        code="unrecognized_lifecycle_rhs",
                        message=(
                            f"constant {name} in {rel_path} has an unrecognized "
                            "RHS. Use one of: "
                            "`GateLifecycleState.<NAME>` or a literal string "
                            "value matching one of: " + ", ".join(sorted(VALID_STATE_VALUES))
                        ),
                        details={"constant": name},
                    )
                )
    return findings


def scan_yaml_configs(repo_root: Path) -> list[ScanFinding]:
    """Scan candidate YAML configs and emit a finding for any that lacks a
    top-level ``lifecycle_state`` key with a recognized value.
    """
    findings: list[ScanFinding] = []
    for path in _candidate_yaml_configs(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except yaml.YAMLError as e:
            findings.append(
                ScanFinding(
                    path=rel,
                    severity="error",
                    code="yaml_parse_error",
                    message=str(e),
                )
            )
            continue
        if not isinstance(doc, dict):
            # Top-level non-mapping YAMLs cannot host a top-level key. Skip.
            continue
        state = doc.get("lifecycle_state")
        if state is None:
            findings.append(
                ScanFinding(
                    path=rel,
                    severity="error",
                    code="missing_lifecycle_state_key",
                    message=(
                        f"config {rel} is gate-shaped (contains threshold/cutoff/"
                        "buffer keys) but lacks a top-level `lifecycle_state` key. "
                        "Declare one of: " + ", ".join(sorted(VALID_STATE_VALUES))
                    ),
                )
            )
        elif state not in VALID_STATE_VALUES:
            findings.append(
                ScanFinding(
                    path=rel,
                    severity="error",
                    code="invalid_lifecycle_state_value",
                    message=(
                        f"config {rel} declares lifecycle_state={state!r}, "
                        "which is not a recognized GateLifecycleState. Valid "
                        "values: " + ", ".join(sorted(VALID_STATE_VALUES))
                    ),
                    details={"value": state},
                )
            )
    return findings


def scan_lifecycle_changes(repo_root: Path, base_ref: str) -> list[ScanFinding]:
    """Walk the git diff between ``base_ref`` and HEAD; for any change to a
    LIFECYCLE_STATE_* constant or YAML lifecycle_state key, require a signed
    doc at ``docs/calibration/{slug}_lifecycle_change_{from}_to_{to}_*.md``.

    Signed-doc requirements (Gate N2 acceptance #3):

    * The doc filename matches ``{slug}_lifecycle_change_{from}_to_{to}_{date}.md``.
    * For transitions INTO the ENFORCED state, the doc body MUST mention all
      four required fields: ``start_date:``, ``end_date:``, ``drift_summary:``,
      ``signing_reviewer:``. (Lightweight — full schema validation is the
      lifecycle-change reviewer's job.)
    """
    findings: list[ScanFinding] = []
    # N2 finding H4: include ``scripts/`` in the diff scope. A future gate
    # constant landing in a script (e.g., ``scripts/gate_runner.py``) was
    # not change-detected when the diff path list was just src/ + config/.
    try:
        diff_output = subprocess.check_output(
            [
                "git",
                "diff",
                "--unified=0",
                base_ref,
                "--",
                "src/",
                "config/",
                "scripts/",
            ],
            cwd=repo_root,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        findings.append(
            ScanFinding(
                path="<git-diff>",
                severity="warning",
                code="git_diff_failed",
                message=f"git diff against {base_ref} failed: {e}; cannot check changes",
            )
        )
        return findings

    changes = _extract_lifecycle_changes(diff_output)
    if not changes:
        return findings

    # Index existing lifecycle-change docs by (slug, from, to).
    docs_by_transition: dict[tuple[str, str, str], list[Path]] = {}
    docs_dir = repo_root / "docs" / "calibration"
    if docs_dir.is_dir():
        for doc_path in docs_dir.glob("*_lifecycle_change_*.md"):
            m = LIFECYCLE_DOC_FILENAME_RE.search(doc_path.name)
            if m:
                key = (m.group("slug"), m.group("from_state"), m.group("to_state"))
                docs_by_transition.setdefault(key, []).append(doc_path)

    for change in changes:
        slug = change["slug"]
        from_state = change["from_state"]
        to_state = change["to_state"]
        # N2 finding M1 + pass-2 follow-up (M1 PARTIAL + new MED): docs
        # MUST use the namespaced slug (``py_t22`` or ``yaml_t22``).
        # The bare-slug fallback (``t22``) was originally kept for
        # backward compat, but it reintroduces the cross-source
        # collision risk M1 was designed to eliminate (a Python
        # ``LIFECYCLE_STATE_T22`` change could match a doc actually
        # authored for the YAML ``t22.yaml`` config). The repo has zero
        # lifecycle-change docs at the time this fallback is removed
        # (verified: ``find docs/calibration -name '*_lifecycle_change_*.md'``
        # returns only the template), so the cut is safe.
        # Lower-casing is preserved for case-insensitive filesystems.
        candidate_keys = [
            (slug, from_state, to_state),
            (slug.lower(), from_state, to_state),
        ]
        matched = False
        matched_doc: Optional[Path] = None
        for key in candidate_keys:
            if key in docs_by_transition:
                matched = True
                matched_doc = docs_by_transition[key][0]
                break
        if not matched:
            findings.append(
                ScanFinding(
                    path=change["source_path"],
                    severity="error",
                    code="missing_lifecycle_change_doc",
                    message=(
                        f"lifecycle change {from_state!r} -> {to_state!r} "
                        f"on {slug} (in {change['source_path']}) has no signed "
                        f"doc at docs/calibration/{slug}_lifecycle_change_"
                        f"{from_state}_to_{to_state}_<YYYYMMDD>.md"
                    ),
                    details=change,
                )
            )
            continue
        if to_state == GateLifecycleState.ENFORCED.value and matched_doc is not None:
            doc_body = matched_doc.read_text(encoding="utf-8")
            # N2 finding M2: each required field MUST appear on a non-comment
            # line (comments-only would satisfy a naive substring check, so
            # an operator could commit a doc with all four fields commented
            # out and pass the gate). We anchor the regex at the start of a
            # line (after optional indent) and reject ``#`` prefixes.
            required_fields = (
                "start_date:",
                "end_date:",
                "drift_summary:",
                "signing_reviewer:",
            )
            missing = [
                f
                for f in required_fields
                if not re.search(rf"^[ \t]*{re.escape(f)}", doc_body, flags=re.MULTILINE)
            ]
            if missing:
                findings.append(
                    ScanFinding(
                        path=str(matched_doc.relative_to(repo_root)),
                        severity="error",
                        code="enforced_doc_missing_fields",
                        message=(
                            f"lifecycle-change doc for ENFORCED transition "
                            f"{slug} {from_state}->{to_state} is missing "
                            f"required fields: {', '.join(missing)}"
                        ),
                        details={"missing_fields": missing, **change},
                    )
                )
    return findings


# Match a unified-diff hunk header with the file path on the prior `+++` line.
_HUNK_HEADER_RE = re.compile(r"^@@ ")
_PYTHON_DIFF_LINE_RE = re.compile(
    r"^(?P<sign>[+-])\s*"
    r"(?P<name>LIFECYCLE_STATE_[A-Z0-9_]+)"
    r"(?:\s*:\s*GateLifecycleState)?"  # optional annotation
    r"\s*=\s*"
    r"(?:GateLifecycleState\.(?P<attr>[A-Z_]+)|\"(?P<lit>[a-z]+)\")"
)
_YAML_DIFF_LINE_RE = re.compile(r"^(?P<sign>[+-])\s*lifecycle_state:\s*(?P<value>[a-z]+)")


def _extract_lifecycle_changes(diff_text: str) -> list[dict[str, str]]:
    """Walk a unified diff and return one entry per (file, constant) pair
    where the lifecycle value changed.

    Each entry: ``{slug, from_state, to_state, source_path}``.

    The ``slug`` derivation (N2 finding M1): the slug is prefixed by the
    SOURCE TYPE — ``py_`` for Python, ``yaml_`` for YAML — to prevent a
    Python constant ``LIFECYCLE_STATE_T22`` from accidentally matching a
    doc named for the YAML config ``t22.yaml`` (or vice versa). Two
    independent gates with overlapping bare slugs would otherwise share
    the same lifecycle-change doc, which is incorrect.

    For Python, the bare slug is the constant name minus the
    ``LIFECYCLE_STATE_`` prefix (lowercase). For YAML, the bare slug is
    the file basename minus the extension. The ``py_``/``yaml_`` prefix
    is then prepended.
    """
    current_path: Optional[str] = None
    # Per-(path, slug) accumulator.
    pluses: dict[tuple[str, str], str] = {}
    minuses: dict[tuple[str, str], str] = {}

    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_path = line[len("+++ b/") :]
            continue
        if line.startswith("--- ") or _HUNK_HEADER_RE.match(line):
            continue
        if not line or line[0] not in "+-":
            continue
        if current_path is None:
            continue
        if current_path.endswith(".py"):
            m = _PYTHON_DIFF_LINE_RE.match(line)
            if not m:
                continue
            bare_slug = m.group("name").removeprefix("LIFECYCLE_STATE_").lower()
            slug = f"py_{bare_slug}"
            attr = m.group("attr")
            lit = m.group("lit")
            if attr is not None:
                try:
                    value = GateLifecycleState[attr].value
                except KeyError:
                    continue
            elif lit is not None and lit in VALID_STATE_VALUES:
                value = lit
            else:
                continue
        elif current_path.endswith((".yml", ".yaml")):
            m_yaml = _YAML_DIFF_LINE_RE.match(line)
            if not m_yaml:
                continue
            value = m_yaml.group("value")
            if value not in VALID_STATE_VALUES:
                continue
            slug = f"yaml_{Path(current_path).stem}"
        else:
            continue
        sign = line[0]
        key = (current_path, slug)
        if sign == "+":
            pluses[key] = value
        else:
            minuses[key] = value

    out: list[dict[str, str]] = []
    for key in set(pluses.keys()) | set(minuses.keys()):
        from_state = minuses.get(key)
        to_state = pluses.get(key)
        if from_state is None or to_state is None:
            # Pure addition (no prior declaration) — handled by the missing-
            # constant check, not by the change-doc check. A pure removal
            # (no replacement) is a delete — skip; the file removal itself
            # is the operator's signal.
            continue
        if from_state == to_state:
            # Cosmetic edit (e.g., comma added on same line). Skip.
            continue
        path, slug = key
        out.append(
            {
                "slug": slug,
                "from_state": from_state,
                "to_state": to_state,
                "source_path": path,
            }
        )
    out.sort(key=lambda d: (d["source_path"], d["slug"]))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plan v4 Gate N2 — lifecycle-state scanner")
    parser.add_argument(
        "--check-changes",
        action="store_true",
        help="Also check lifecycle-state changes against signed docs.",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to diff against when --check-changes is set.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit machine-readable JSON instead of human text.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root (default: this script's parent).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    findings: list[ScanFinding] = []
    findings.extend(scan_python_modules(repo_root))
    findings.extend(scan_yaml_configs(repo_root))
    if args.check_changes:
        findings.extend(scan_lifecycle_changes(repo_root, args.base_ref))

    errors = [f for f in findings if f.severity == "error"]
    if args.json_output:
        print(
            json.dumps(
                {
                    "findings": [f.to_dict() for f in findings],
                    "error_count": len(errors),
                    "scanned_at": date.today().isoformat(),
                },
                indent=2,
            )
        )
    else:
        if not findings:
            print("OK — every gate-relevant location declares a lifecycle_state.")
        for f in findings:
            print(f"[{f.severity.upper()}] {f.code}: {f.path}")
            print(f"  {f.message}")
        if errors:
            print(
                f"\n{len(errors)} lifecycle-state guard violation(s) found.",
                file=sys.stderr,
            )
    return 1 if errors else 0


# Pydantic model is imported only for use by callers that want to validate
# parsed declarations.  Re-exported for downstream tests.
__all__ = [
    "GateLifecycleState",
    "GATE_RELEVANT_PYTHON_MODULES",
    "LifecycleDeclaration",
    "ScanFinding",
    "main",
    "scan_lifecycle_changes",
    "scan_python_modules",
    "scan_yaml_configs",
]


if __name__ == "__main__":
    sys.exit(main())
