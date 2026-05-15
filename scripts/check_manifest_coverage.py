#!/usr/bin/env python3
"""Layer 1 manifest-coverage CI guard — Phase 1.5.

Plan reference: ``.claude/plans/adaptive_temporal_validity_redesign.md``
Layer 1 (Declarative Temporal Contracts). The plan requires that every
column emitted into ``patient_journeys`` output by the CSU and Optum
converters has a corresponding :class:`FeatureContract` entry in the
appropriate manifest, so that
``src/agents/ml_foundation/data_preparer/nodes/adaptive_validity_check.py::
lookup_feature_contract`` returns a non-None contract and Layer 1 emits a
verdict for every column. Without this guard, a converter PR can silently
introduce a new column whose manifest entry is missing; ``lookup_feature_contract``
returns ``None`` for that column and Layer 1 no-ops on it. Reviewer
vigilance is the only line of defense today; this script makes the
invariant machine-enforceable.

Discovery mechanism — AST parse with loop-iterable expansion
==============================================================
The task brief lists three candidate discovery strategies:

  (a) AST parse the converter scripts to extract column names.
  (b) Run a smoke conversion against a synthetic fixture and read the
      output frame's columns.
  (c) Maintain a checked-in golden column-list per cohort.

This script chooses (a). Rationale:

  - **Determinism & no data dependency.** AST parsing is a pure function
    of the source file. Option (b) requires shipping a synthetic input
    fixture that exercises every code path that produces a column —
    Optum's ``_compute_features`` has branches keyed on the presence of
    inpatient / lab / procedure tables, and a fixture light enough to
    keep CI fast risks missing branches. Option (c) requires a separate
    checked-in golden file per cohort; a PR that adds a column WITHOUT
    a manifest entry could trivially update the golden file in the same
    commit, and the guard would silently pass.
  - **No external compute.** The CSU/Optum converters depend on pandas,
    openpyxl, and the rwd_common package; option (b) would require
    installing the full converter dependency stack in the CI job. AST
    parsing uses only the stdlib.
  - **Loop-iterable expansion is tractable.** The Optum
    ``_compute_features`` body produces ~80 columns, most via f-strings
    inside ``for X in Y:`` loops where ``Y`` is a module-level dict or
    tuple literal (``COMORBIDITY_CODES``, ``CSU_LABS_LOINC``,
    ``NON_TARGET_DRUG_CLASSES``). The AST walker resolves these
    iterables by re-parsing the converter's own module-level
    assignments and constant-folding the loop variable into each
    f-string subscript expression.

**Acknowledged limitation.** The CSU converter emits ``extra_demo`` keys
(``demo_<col>`` runtime pass-through of unrecognised input demo
columns); these are not statically discoverable. They are covered by an
explicit ALLOWED_PREFIXES entry. Any other dynamic column-generation
mechanism that is NOT a literal or loop-over-module-constant will be
invisible to this guard — that is the documented gap.

Allowlist
=========
The script tolerates a small allowlist of non-feature columns that are
real outputs of the converter but are NOT model inputs. Each entry is
justified inline in :data:`AUDIT_COLUMN_ALLOWLIST`:

  - **Identifier columns** (``patient_journey_id``, ``patient_id``,
    ``patient_hash``, ``_patid``): keys used to join the journey record
    to other tables (``treatment_events``, ``hcp_profiles``). The
    cohort-builder gate at ``_drop_forbidden_columns`` strips
    ``OPTUM_FORBIDDEN_NON_TARGET`` only — these IDs survive and are
    used for downstream joins. They are not features.
  - **Audit timestamps** (``created_at``, ``updated_at``,
    ``ingestion_timestamp``, ``source_timestamp``): operational
    metadata recording when the row was materialised. Not features.
  - **Data-source provenance** (``data_source``, ``data_sources_matched``,
    ``source_match_confidence``, ``source_stacking_flag``,
    ``source_combination_method``, ``data_lag_hours``, ``data_split``,
    ``split_config_id``): records WHERE the row's evidence originated
    and the split decision. Audit-only.
  - **Diagnosis-narrative columns** (``primary_diagnosis_desc``,
    ``secondary_diagnosis_codes``, ``comorbidities``): human-readable
    text or list-shaped columns used by the journey-viewer UI, not by
    any ML pipeline.
  - **Quality scores** (``data_quality_score``, ``risk_score``,
    ``state``): operational scores attached for downstream audit / UI
    display.
  - **CSU-specific runtime pass-through prefix** ``demo_*`` (handled
    via ALLOWED_PREFIXES): the CSU converter's runtime pass-through of
    unrecognised demo columns into ``demo_<col_name>`` keys. These are
    by construction unmappable at static-analysis time; the CSU
    converter's ``_ALREADY_EXTRACTED_DEMO_COLS`` set documents which
    demo columns are mapped to canonical feature names.

If a column appears in the manifest AND the allowlist, the allowlist
entry is silently ignored (the manifest is authoritative). If a column
is in NEITHER, the guard fails with a non-zero exit code naming the
column + cohort.

Usage
=====

.. code-block:: bash

    # Run all cohorts (default in CI).
    python scripts/check_manifest_coverage.py

    # Lint a single cohort:
    python scripts/check_manifest_coverage.py --only-cohort csu
    python scripts/check_manifest_coverage.py --only-cohort optum-initiation

    # Custom repo root (defaults to the cwd-or-script-parent's parent).
    python scripts/check_manifest_coverage.py --repo-root /path/to/repo

Exit codes
==========

  * ``0`` — every column in every cohort's discovered output is covered
    by manifest or allowlist.
  * ``1`` — at least one column is unmapped. stderr lists each
    ``<cohort>:<column>`` pair.
  * ``2`` — script invocation error (e.g. converter file not found,
    AST parse failure, unresolvable loop iterable).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Config — cohort registry
# ---------------------------------------------------------------------------

# Each cohort declares (a) the converter script that builds its
# ``patient_journeys`` output, (b) the function name whose body produces
# the journey record's column set, and (c) the manifest module-attribute
# pair to consult for coverage.
#
# Optum runs three cohort directories (``initiation``, ``discontinuation``,
# ``persistence``); all three share the same ``_build_journey_record``
# code path, but the discontinuation / persistence cohorts set additional
# targets (``discontinued_180d``, ``persistent_at_180d``) explicitly via
# the record's dict literal. We declare each cohort separately so a
# failure message names the specific cohort directory affected.


@dataclass(frozen=True)
class DiscoveryFunc:
    """One AST-discovery target inside a converter.

    ``func_name`` is the function whose body holds the column-producing
    assignments. ``output_dict_names`` enumerates the local variable
    names that the function uses as the output / spread dict — these
    are the ONLY variables whose subscript assignments and dict-literal
    assignments are collected as columns. Any other dict (e.g., a
    helper aggregation like ``type_counts`` or ``l50_counts``) is
    ignored.

    This whitelisting prevents false positives from incidental dict
    literals + subscript assignments unrelated to the journey output.
    A converter refactor that renames the output dict will be caught
    by the test ``test_<cohort>_discovery_finds_known_column``.
    """

    func_name: str
    output_dict_names: tuple[str, ...]


@dataclass(frozen=True)
class CohortConfig:
    """One cohort's discovery + reconciliation configuration."""

    name: str  # display name used in stderr + tests
    converter_rel_path: str  # repo-relative converter script path
    # Functions whose body emits the journey-record dict. We discover
    # column names by AST-scanning these functions. CSU has only
    # ``_build_patient_journeys``. Optum has ``_build_journey_record``
    # AND a sibling ``_compute_features`` whose output is spread into
    # the record via ``record.update(feats)`` — so the discovery must
    # follow that spread.
    discovery_funcs: tuple[DiscoveryFunc, ...]
    # Module path (dotted) for the manifest registry. The script imports
    # ``MODULE.ATTR`` and reads names off the resulting list.
    manifest_module: str
    manifest_attr: str


COHORTS: tuple[CohortConfig, ...] = (
    CohortConfig(
        name="csu",
        converter_rel_path="scripts/convert_csu_rwd.py",
        discovery_funcs=(
            DiscoveryFunc(
                func_name="_build_patient_journeys",
                # ``journey_dict`` is the canonical output dict assembled
                # per-patient and appended to ``journeys``. ``extra_demo``
                # is the runtime pass-through dict that ``journey_dict``
                # ingests via ``.update(extra_demo)`` — its keys
                # (``demo_<col>``) become real output columns. Both must
                # be tracked so the static scan sees the full surface.
                output_dict_names=("journey_dict", "extra_demo"),
            ),
        ),
        manifest_module="src.data.manifests.csu_feature_manifest",
        manifest_attr="CSU_FEATURES",
    ),
    CohortConfig(
        name="optum-initiation",
        converter_rel_path="scripts/convert_optum_rwd.py",
        discovery_funcs=(
            DiscoveryFunc(
                func_name="_build_journey_record",
                # ``record`` is the journey dict; it ingests
                # ``feats`` via ``record.update(feats)`` — the spread
                # happens at runtime, so we scan ``_compute_features``
                # separately and union.
                output_dict_names=("record",),
            ),
            DiscoveryFunc(
                func_name="_compute_features",
                # ``feats`` is the per-patient feature dict assembled in
                # ``_compute_features`` and spread into ``record`` at the
                # caller. All subscript assignments to ``feats`` are
                # journey-output columns.
                output_dict_names=("feats",),
            ),
        ),
        manifest_module="src.data.manifests.optum_feature_manifest",
        manifest_attr="OPTUM_FEATURES",
    ),
    CohortConfig(
        name="optum-discontinuation",
        converter_rel_path="scripts/convert_optum_rwd.py",
        discovery_funcs=(
            DiscoveryFunc(func_name="_build_journey_record", output_dict_names=("record",)),
            DiscoveryFunc(func_name="_compute_features", output_dict_names=("feats",)),
        ),
        manifest_module="src.data.manifests.optum_feature_manifest",
        manifest_attr="OPTUM_FEATURES",
    ),
    CohortConfig(
        name="optum-persistence",
        converter_rel_path="scripts/convert_optum_rwd.py",
        discovery_funcs=(
            DiscoveryFunc(func_name="_build_journey_record", output_dict_names=("record",)),
            DiscoveryFunc(func_name="_compute_features", output_dict_names=("feats",)),
        ),
        manifest_module="src.data.manifests.optum_feature_manifest",
        manifest_attr="OPTUM_FEATURES",
    ),
)


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

# Audit / identifier / provenance columns that are real outputs of the
# converter but are NOT model features. The Layer 1 manifest deliberately
# does not catalog these — they have no temporal-validity claim to audit.
# Each entry is justified in the module docstring (see "Allowlist"
# section) so a reviewer adding a new entry MUST explain WHY the column
# is not a feature.
#
# Maintenance: if you find yourself adding a new entry here that COULD
# be a feature (anything with a numeric value derived from clinical
# events), STOP — that column belongs in the manifest as
# ``FeatureContract`` with the appropriate ``knowable_at`` claim.
AUDIT_COLUMN_ALLOWLIST: frozenset[str] = frozenset(
    {
        # ------------------------------------------------------------------
        # Identifier columns — join keys to other tables, not features.
        # ------------------------------------------------------------------
        "patient_journey_id",
        "patient_id",
        "patient_hash",
        "_patid",  # internal — stripped before parquet write
        # ------------------------------------------------------------------
        # Audit timestamps — operational metadata.
        # ------------------------------------------------------------------
        "created_at",
        "updated_at",
        "ingestion_timestamp",
        "source_timestamp",
        # ------------------------------------------------------------------
        # Data-source provenance — records where the row's evidence
        # originated and how the split was assigned.
        # ------------------------------------------------------------------
        "data_source",
        "data_sources_matched",
        "source_match_confidence",
        "source_stacking_flag",
        "source_combination_method",
        "data_lag_hours",
        "data_split",
        "split_config_id",
        # ------------------------------------------------------------------
        # Diagnosis-narrative columns — UI / human-readable, not ML inputs.
        # ------------------------------------------------------------------
        "primary_diagnosis_desc",
        "secondary_diagnosis_codes",
        "comorbidities",
        # ------------------------------------------------------------------
        # Operational scores + free-text state — not Layer 1 features.
        # ``data_quality_score`` is itself a downstream gate (issue #156),
        # not a model input; ``risk_score`` is a placeholder for an
        # external risk model not yet wired; ``state`` is a free-text
        # location label never used as a feature.
        # ------------------------------------------------------------------
        "data_quality_score",
        "risk_score",
        "state",
        # ------------------------------------------------------------------
        # Optum payer-raw columns — these are documented in
        # ``convert_optum_rwd.py`` as audit-trail raw values that
        # preserve the source fields used to derive ``payer_category``
        # (which IS in the manifest). Issue #156 item 6 explicitly
        # treats them as audit not feature. The downstream pipeline
        # does not consume ``payer_*_raw`` as a model input.
        # ------------------------------------------------------------------
        "payer_bus_raw",
        "payer_product_raw",
        "payer_health_exch_raw",
        "payer_lis_dual_raw",
    }
)


# Allowed column-name prefixes. The CSU converter pass-through writes
# ``demo_<col>`` for any unrecognised raw-demo column. The runtime
# extension is bounded by the converter's ``_ALREADY_EXTRACTED_DEMO_COLS``
# exclusion set, but the resulting names are unknowable at static-analysis
# time. Treating the prefix as allow-listed means we accept "this column
# is dynamically generated runtime metadata" — analogous to the audit
# columns above.
ALLOWED_PREFIXES: tuple[str, ...] = ("demo_",)


# Optional manifest-side audit. The named manifest entry MUST appear in
# the discovered column set; if not, the manifest has drifted ahead of
# the converter (a feature that was declared but never produced).
# Empty by default; populated lazily if/when a regression motivates it.
MANIFEST_MUST_BE_PRODUCED: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _extract_module_iterables(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Return module-level ``name = {...}`` / ``name = (...)`` constant-string
    iterables. Used to constant-fold ``for X in NAME:`` loops in the
    discovery walker.

    Supported shapes:

      * ``NAME: dict[str, ...] = {"k1": ..., "k2": ...}``  →  ("k1", "k2")
      * ``NAME: tuple[str, ...] = ("a", "b", "c")``       →  ("a", "b", "c")
      * ``NAME = ("a", "b", "c")``                         →  ("a", "b", "c")

    Annotated assignments and bare assignments are both accepted. Only
    the iteration-key set is collected — values are ignored.
    """
    iterables: dict[str, tuple[str, ...]] = {}

    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue
            target = node.targets[0]
            value = node.value
        else:
            continue

        if not isinstance(target, ast.Name) or value is None:
            continue

        name = target.id
        names_in_iterable: list[str] = []

        if isinstance(value, ast.Dict):
            # Dict literal — collect string keys (used by ``for K, V in NAME.items():``
            # AND by bare ``for K in NAME:`` (since iter over dict yields keys).
            ok = True
            for k in value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    names_in_iterable.append(k.value)
                else:
                    ok = False
                    break
            if ok:
                iterables[name] = tuple(names_in_iterable)
        elif isinstance(value, (ast.Tuple, ast.List)):
            ok = True
            for elt in value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names_in_iterable.append(elt.value)
                else:
                    ok = False
                    break
            if ok:
                iterables[name] = tuple(names_in_iterable)

    return iterables


class _ColumnDiscoveryVisitor(ast.NodeVisitor):
    """Walk a function body and collect every string-valued subscript key
    written to a NAMED output dict, plus the dict-literal keys of an
    assignment whose target Name matches the output-dict whitelist.

    Three patterns are handled:

      1. ``<output>[<literal_str>] = ...``       → literal key (added)
      2. ``<output>[f"{prefix}_{var}"] = ...``   → for-loop f-string;
         resolved by walking enclosing ``for`` nodes whose iterable is
         a module-level constant (resolved via
         ``_extract_module_iterables``). If the f-string starts with
         an ``ALLOWED_PREFIXES`` literal AND the variable can NOT be
         resolved, the f-string is marked as "covered by ALLOWED_PREFIXES"
         and contributes a synthetic key with that prefix; the
         orchestrator's allowlist check then accepts it.
      3. ``<output> = {<literal_str>: ..., ...}`` → dict-literal keys
         (added). Annotated and unannotated forms both handled.

    The ``<output>`` whitelist comes from the cohort config's
    ``DiscoveryFunc.output_dict_names``. Any subscript or dict-literal
    assignment to a NON-whitelisted target is ignored — this prevents
    helper-aggregate dicts (``type_counts``, ``l50_counts``) from
    leaking into the discovered set.

    The visitor does NOT recurse into nested ``def``/``async def`` /
    ``lambda`` — those are handled separately by the orchestrator
    (``discover_columns_for_cohort``) which scans the named entry
    functions.
    """

    # Sentinel key inserted by the visitor when an f-string subscript
    # assignment matches ALLOWED_PREFIXES but the variable couldn't be
    # statically resolved. The orchestrator's ``_is_allowed`` check
    # accepts these without further work; using a distinct sentinel
    # makes the coverage-failure report unambiguous if a future change
    # demotes a prefix from the allowlist.
    _ALLOWED_PREFIX_SENTINEL: str = "<allowed-prefix>"

    def __init__(
        self,
        module_iterables: dict[str, tuple[str, ...]],
        output_dict_names: tuple[str, ...],
    ):
        self.module_iterables = module_iterables
        self.output_dict_names = output_dict_names
        self.discovered: set[str] = set()
        # Track the loop variable bindings active at any given point
        # during traversal so f-string resolution can constant-fold.
        # Stack of dicts: each dict maps var name → tuple of possible
        # values for the iterations of the enclosing ``for`` loop.
        self._loop_scopes: list[dict[str, tuple[str, ...]]] = []
        # Resolved f-string expressions where the iterable couldn't be
        # constant-folded. These trigger a hard error in the discovery
        # orchestrator: an unresolved f-string column means the static
        # analyser cannot enumerate the column set, so the guard must
        # fail-closed rather than silently miss columns.
        self.unresolved_f_strings: list[str] = []

    # ------------------------------------------------------------------ #
    # Loop tracking                                                       #
    # ------------------------------------------------------------------ #

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._handle_for(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._handle_for(node)

    def _handle_for(self, node: ast.For | ast.AsyncFor) -> None:
        # Resolve the iterable; bind loop variable(s) to the resolved
        # value set; recurse; pop binding.
        iter_values = self._resolve_iter(node.iter)
        bindings: dict[str, tuple[str, ...]] = {}
        target = node.target

        if iter_values is not None:
            if isinstance(target, ast.Name):
                bindings[target.id] = iter_values
            elif isinstance(target, ast.Tuple):
                # ``for k, v in d.items():`` — bind k to keys, leave v
                # unresolved (values aren't column-relevant for the
                # current code patterns). Only the first tuple element
                # is the iteration key when the source is a dict.
                if target.elts and isinstance(target.elts[0], ast.Name):
                    bindings[target.elts[0].id] = iter_values

        self._loop_scopes.append(bindings)
        try:
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)
        finally:
            self._loop_scopes.pop()

    def _resolve_iter(self, expr: ast.expr) -> tuple[str, ...] | None:
        """Try to resolve a ``for X in EXPR:`` iterable to a static
        tuple of strings.

        Supports:

          * ``NAME`` where NAME is a module-level dict/tuple/list
          * ``NAME.items()`` / ``NAME.keys()`` / ``NAME.values()``
            where NAME is a module-level dict (we use keys for any
            of these because the loop typically iterates keys; .values()
            is rare enough that we leave it unresolved if encountered
            as the sole binding).
        """
        if isinstance(expr, ast.Name):
            return self.module_iterables.get(expr.id)
        if isinstance(expr, ast.Call):
            # Pattern: NAME.items() / NAME.keys() — the iterable is the
            # tuple of dict keys.
            if (
                isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.attr in ("items", "keys")
            ):
                return self.module_iterables.get(expr.func.value.id)
        return None

    # ------------------------------------------------------------------ #
    # Subscript assignments: ``<output>[<KEY>] = ...``                    #
    # ------------------------------------------------------------------ #

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Pattern A: <output>[<str>] = ... — accept ONLY when the
        # subscript target's Name is in ``output_dict_names``.
        for tgt in node.targets:
            if isinstance(tgt, ast.Subscript) and self._is_output_target(tgt):
                self._handle_subscript_assign(tgt)
        # Pattern B: <output> = { "literal_key": ..., ... } — collect
        # dict-literal keys only when the assignment target is a Name
        # in ``output_dict_names``.
        if isinstance(node.value, ast.Dict):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in self.output_dict_names:
                    self._collect_dict_literal_keys(node.value)
                    break
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        # Annotated dict-literal assignment: ``<output>: <T> = {...}``.
        # The converters use this for output-record annotations like
        # ``record: dict[str, Any] = {...}``.
        if isinstance(node.value, ast.Dict) and isinstance(node.target, ast.Name):
            if node.target.id in self.output_dict_names:
                self._collect_dict_literal_keys(node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        # No-op for column discovery; included for completeness.
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Pattern: ``<output>.update({...})`` — collect dict-literal
        # keys in the first positional argument when the .update receiver
        # is one of the output dicts.
        if isinstance(node.func, ast.Attribute) and node.func.attr == "update":
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.output_dict_names
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                self._collect_dict_literal_keys(node.args[0])
        self.generic_visit(node)

    def _is_output_target(self, sub: ast.Subscript) -> bool:
        """True iff the subscript expression's ``value`` (the dict
        being subscripted) is a Name whose id is in
        ``output_dict_names``.
        """
        if isinstance(sub.value, ast.Name) and sub.value.id in self.output_dict_names:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _handle_subscript_assign(self, sub: ast.Subscript) -> None:
        # Extract the key. ast.Subscript.slice is the key expression
        # (Index-wrapped in 3.8; bare expr in 3.9+; we target 3.12).
        key = sub.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            self.discovered.add(key.value)
            return
        if isinstance(key, ast.JoinedStr):
            resolved = self._resolve_fstring(key)
            if resolved is not None:
                self.discovered.update(resolved)
            return

    def _collect_dict_literal_keys(self, d: ast.Dict) -> None:
        # We only collect string-constant keys. F-string keys inside a
        # dict literal are rare in the journey-record idiom; if one
        # appears, log it as unresolved so the orchestrator fails.
        for k in d.keys:
            if k is None:
                # Dict-unpack (``**other_dict``) — out of static scope.
                continue
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                self.discovered.add(k.value)
            elif isinstance(k, ast.JoinedStr):
                resolved = self._resolve_fstring(k)
                if resolved is not None:
                    self.discovered.update(resolved)

    def _resolve_fstring(self, fstr: ast.JoinedStr) -> tuple[str, ...] | None:
        """Resolve an f-string to a set of concrete strings by enumerating
        every active loop binding the f-string references.

        For an f-string like ``f"has_{name}"`` inside ``for name in
        COMORBIDITY_CODES.items():``, the visitor's ``_loop_scopes``
        contains ``{"name": ("atopic_dermatitis", "asthma", ...)}``;
        each substitution yields one resolved column name.

        Multiple references → cartesian product.

        Special case — **allowed-prefix tolerance**: if the f-string
        starts with a literal that matches an entry in
        ``ALLOWED_PREFIXES`` (e.g. ``f"demo_{col_name}"``) AND at least
        one var reference is unbindable, the whole expression is
        accepted as "covered by ALLOWED_PREFIXES" and ``None`` is
        returned with no entry added to ``unresolved_f_strings``. The
        caller treats a ``None`` return as "no concrete keys to add" —
        which is exactly the behaviour we want, because the allowlist
        check at coverage-reconciliation time accepts any column with
        a covered prefix. We do, however, push a sentinel into
        ``discovered`` so the per-function discovery surface reflects
        that the function emits at-least-one prefix-keyed column.

        Unbound references (no allowed prefix) → log in
        ``unresolved_f_strings`` and return ``None``; the orchestrator
        fails the cohort.
        """
        # Flatten the joined-str into a sequence of ("literal", str) /
        # ("var", name) parts.
        parts: list[tuple[str, str]] = []
        for v in fstr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(("literal", v.value))
            elif isinstance(v, ast.FormattedValue):
                # Only support ``{name}`` and ``{name.attr}`` / Name
                # references; complex expressions → unresolved.
                if isinstance(v.value, ast.Name):
                    parts.append(("var", v.value.id))
                else:
                    if not self._matches_allowed_prefix(parts):
                        self.unresolved_f_strings.append(ast.unparse(fstr))
                    return None
            else:
                if not self._matches_allowed_prefix(parts):
                    self.unresolved_f_strings.append(ast.unparse(fstr))
                return None

        # Collect all var names and their possible values.
        var_values: dict[str, tuple[str, ...]] = {}
        for kind, val in parts:
            if kind != "var":
                continue
            resolved = self._lookup_loop_var(val)
            if resolved is None:
                if self._matches_allowed_prefix(parts):
                    # Accepted via the prefix allowlist — no concrete
                    # keys to add, but mark the function as emitting
                    # an allowed-prefix column for downstream auditing.
                    self.discovered.add(self._ALLOWED_PREFIX_SENTINEL)
                else:
                    self.unresolved_f_strings.append(ast.unparse(fstr))
                return None
            var_values[val] = resolved

        # Cartesian product over var bindings.
        var_names = list(var_values.keys())
        if not var_names:
            # Constant f-string with no var references — flatten literals.
            joined = "".join(val for kind, val in parts if kind == "literal")
            return (joined,)

        # Build all combinations. For the journey-record idiom we
        # typically have a single var, but support N variables for
        # generality (the cartesian product over module-constant
        # iterables is finite and small).
        def _expand(idx: int, partial: dict[str, str]) -> list[str]:
            if idx == len(var_names):
                out_parts: list[str] = []
                for kind, val in parts:
                    if kind == "literal":
                        out_parts.append(val)
                    else:
                        out_parts.append(partial[val])
                return ["".join(out_parts)]
            results: list[str] = []
            var = var_names[idx]
            for v in var_values[var]:
                partial[var] = v
                results.extend(_expand(idx + 1, partial))
            return results

        return tuple(_expand(0, {}))

    def _matches_allowed_prefix(self, parts: list[tuple[str, str]]) -> bool:
        """True iff the parts list starts with a literal that equals
        one of ``ALLOWED_PREFIXES`` (e.g. ``"demo_"``). Used to gate
        ``unresolved_f_strings`` reporting: an f-string with an allowed
        prefix is treated as "dynamic but allowlisted" rather than as
        a hard discovery error.
        """
        if not parts:
            return False
        kind, val = parts[0]
        if kind != "literal":
            return False
        return any(val == p for p in ALLOWED_PREFIXES)

    def _lookup_loop_var(self, var: str) -> tuple[str, ...] | None:
        # Search loop scopes top→bottom (innermost first).
        for scope in reversed(self._loop_scopes):
            if var in scope:
                return scope[var]
        return None


# ---------------------------------------------------------------------------
# Discovery orchestration
# ---------------------------------------------------------------------------


def _load_function_nodes(tree: ast.Module, names: Sequence[str]) -> dict[str, ast.FunctionDef]:
    """Find the named function/async-function nodes anywhere in the tree
    (top-level OR nested inside classes — converters define their
    builders as instance methods on a converter class).
    """
    found: dict[str, ast.FunctionDef] = {}

    class _Finder(ast.NodeVisitor):
        def visit_FunctionDef(  # noqa: N802
            self, node: ast.FunctionDef
        ) -> None:
            if node.name in names and node.name not in found:
                found[node.name] = node
            self.generic_visit(node)

        def visit_AsyncFunctionDef(  # noqa: N802
            self, node: ast.AsyncFunctionDef
        ) -> None:
            # Treated identically for column discovery purposes.
            if node.name in names and node.name not in found:
                # Cast through a structural-equivalence: FunctionDef and
                # AsyncFunctionDef share body/args/etc.
                found[node.name] = node  # type: ignore[assignment]
            self.generic_visit(node)

    _Finder().visit(tree)
    return found


def discover_columns_for_cohort(
    repo_root: Path, cohort: CohortConfig
) -> tuple[frozenset[str], list[str]]:
    """Return (discovered_columns, errors).

    On success, ``errors`` is empty and ``discovered_columns`` is the
    full set of statically-discoverable column names. On failure,
    ``discovered_columns`` is whatever was discovered before the failure
    and ``errors`` lists each blocker.
    """
    errors: list[str] = []
    converter_path = repo_root / cohort.converter_rel_path
    if not converter_path.is_file():
        return (
            frozenset(),
            [f"{cohort.name}: converter not found at {converter_path}"],
        )

    src = converter_path.read_text()
    try:
        tree = ast.parse(src, filename=str(converter_path))
    except SyntaxError as exc:
        return (
            frozenset(),
            [f"{cohort.name}: AST parse failed: {exc}"],
        )

    module_iterables = _extract_module_iterables(tree)
    func_names = tuple(df.func_name for df in cohort.discovery_funcs)
    func_nodes = _load_function_nodes(tree, func_names)
    missing_funcs = [n for n in func_names if n not in func_nodes]
    if missing_funcs:
        errors.append(
            f"{cohort.name}: discovery functions not found in "
            f"{cohort.converter_rel_path}: {sorted(missing_funcs)}. "
            f"(The converter was refactored without updating this guard.)"
        )
        # No point continuing — fall through to return the partial set.

    discovered: set[str] = set()
    for df in cohort.discovery_funcs:
        fn = func_nodes.get(df.func_name)
        if fn is None:
            continue
        visitor = _ColumnDiscoveryVisitor(
            module_iterables, df.output_dict_names
        )
        # Walk the function body. We intentionally skip the function
        # parameter signature — discovery is body-only.
        for stmt in fn.body:
            visitor.visit(stmt)
        discovered.update(visitor.discovered)
        if visitor.unresolved_f_strings:
            # De-duplicate while preserving order.
            seen_unresolved: list[str] = []
            for f in visitor.unresolved_f_strings:
                if f not in seen_unresolved:
                    seen_unresolved.append(f)
            errors.append(
                f"{cohort.name}: unresolved f-string subscript keys in "
                f"{df.func_name} (column enumeration incomplete; fix the "
                f"discovery walker or pin the loop iterable to a "
                f"module-level constant):\n  "
                + "\n  ".join(seen_unresolved)
            )

    return frozenset(discovered), errors


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest_names(
    repo_root: Path, cohort: CohortConfig
) -> tuple[frozenset[str], list[str]]:
    """Import the cohort's manifest module and return the registered
    feature names. Failure modes:

      * Import error (e.g., a circular import or a refactor that
        renamed the manifest module) → returned as a single error.
      * Manifest attribute missing → returned as a single error.
    """
    errors: list[str] = []
    # We need ``repo_root`` on ``sys.path`` so ``src.data.manifests.*``
    # resolves. Prepend if not present.
    rr_str = str(repo_root.resolve())
    inserted = False
    if rr_str not in sys.path:
        sys.path.insert(0, rr_str)
        inserted = True
    try:
        import importlib

        module = importlib.import_module(cohort.manifest_module)
    except Exception as exc:  # pragma: no cover — defensive
        return (
            frozenset(),
            [f"{cohort.name}: manifest import failed ({cohort.manifest_module}): {exc!r}"],
        )
    finally:
        if inserted:
            try:
                sys.path.remove(rr_str)
            except ValueError:  # pragma: no cover
                pass

    if not hasattr(module, cohort.manifest_attr):
        errors.append(
            f"{cohort.name}: manifest module {cohort.manifest_module} "
            f"has no attribute {cohort.manifest_attr!r}"
        )
        return frozenset(), errors

    contracts = getattr(module, cohort.manifest_attr)
    names: list[str] = []
    for c in contracts:
        if not hasattr(c, "name"):
            errors.append(
                f"{cohort.name}: manifest entry {c!r} has no .name attribute"
            )
            continue
        names.append(c.name)

    return frozenset(names), errors


# ---------------------------------------------------------------------------
# Coverage reconciliation
# ---------------------------------------------------------------------------


def _is_allowed(column: str) -> bool:
    if column == _ColumnDiscoveryVisitor._ALLOWED_PREFIX_SENTINEL:
        # Sentinel emitted by the visitor when an f-string subscript
        # is dynamically generated but its literal prefix matches an
        # entry in ALLOWED_PREFIXES. The discovery walker emits this
        # so the per-cohort discovered-count reflects "at least one
        # prefix-keyed column is dynamically produced"; coverage
        # accepts it because the allowlist would accept any concrete
        # key matching the prefix.
        return True
    if column in AUDIT_COLUMN_ALLOWLIST:
        return True
    return any(column.startswith(p) for p in ALLOWED_PREFIXES)


@dataclass(frozen=True)
class CohortReport:
    """One cohort's reconciliation result."""

    cohort: str
    discovered: frozenset[str]
    manifest: frozenset[str]
    unmapped: tuple[str, ...]  # in discovered but NOT in (manifest ∪ allowlist)
    manifest_unproduced: tuple[str, ...]  # in MANIFEST_MUST_BE_PRODUCED but absent from discovered

    @property
    def passed(self) -> bool:
        return not self.unmapped and not self.manifest_unproduced


def reconcile_cohort(
    discovered: frozenset[str], manifest: frozenset[str], cohort_name: str
) -> CohortReport:
    unmapped = sorted(c for c in discovered if c not in manifest and not _is_allowed(c))
    # The optional reverse-check: manifest entries that the static
    # analyser failed to find in the converter. Empty by default.
    manifest_unproduced = sorted(
        c for c in MANIFEST_MUST_BE_PRODUCED if c not in discovered
    )
    return CohortReport(
        cohort=cohort_name,
        discovered=discovered,
        manifest=manifest,
        unmapped=tuple(unmapped),
        manifest_unproduced=tuple(manifest_unproduced),
    )


# ---------------------------------------------------------------------------
# Public orchestration entrypoint
# ---------------------------------------------------------------------------


def check_all(
    repo_root: Path, only_cohorts: Iterable[str] | None = None
) -> tuple[int, list[CohortReport], list[str]]:
    """Run the guard against every cohort (or a subset).

    Returns ``(exit_code, reports, errors)``:

      * ``exit_code`` — 0 on full pass, 1 on coverage failure, 2 on
        invocation / discovery error.
      * ``reports`` — one CohortReport per cohort (always populated for
        every requested cohort, even on partial failure).
      * ``errors`` — diagnostic messages from discovery / manifest load
        failures. Empty on success.
    """
    cohorts = [c for c in COHORTS if only_cohorts is None or c.name in only_cohorts]
    if only_cohorts is not None:
        requested = set(only_cohorts)
        found = {c.name for c in cohorts}
        missing = sorted(requested - found)
        if missing:
            return (
                2,
                [],
                [
                    f"unknown cohort(s) in --only-cohort: {missing}. "
                    f"valid choices: {[c.name for c in COHORTS]}"
                ],
            )

    reports: list[CohortReport] = []
    errors: list[str] = []

    for cohort in cohorts:
        discovered, disc_errs = discover_columns_for_cohort(repo_root, cohort)
        errors.extend(disc_errs)
        manifest_names, man_errs = load_manifest_names(repo_root, cohort)
        errors.extend(man_errs)
        report = reconcile_cohort(discovered, manifest_names, cohort.name)
        reports.append(report)

    # Any discovery / manifest-load error is a hard fail (exit 2):
    # we cannot make a coverage judgement when the inputs are broken.
    if errors:
        return 2, reports, errors

    any_failed = any(not r.passed for r in reports)
    return (1 if any_failed else 0), reports, []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="check_manifest_coverage",
        description=(
            "Layer 1 manifest-coverage CI guard (Phase 1.5). AST-parses the "
            "CSU and Optum converters to enumerate the column set of each "
            "patient_journeys output, then reconciles each column against "
            "its cohort's FeatureContract manifest registry. Exits non-zero "
            "if any column is unmapped (neither in the manifest nor in the "
            "audit-column allowlist)."
        ),
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root. Defaults to the script's parent directory "
            "(i.e. the repo root when this script is at scripts/...)"
        ),
    )
    p.add_argument(
        "--only-cohort",
        action="append",
        default=None,
        choices=[c.name for c in COHORTS],
        help=(
            "Restrict to one or more cohorts (repeatable). Defaults to all "
            "cohorts."
        ),
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = (args.repo_root or Path(__file__).resolve().parent.parent).resolve()

    exit_code, reports, errors = check_all(
        repo_root, only_cohorts=args.only_cohort
    )

    # Summary output. Always print the discovered column counts so the
    # CI log shows the audit even on PASS.
    print(f"Layer 1 manifest-coverage guard — repo_root={repo_root}", file=sys.stderr)
    for r in reports:
        suffix = ""
        if r.unmapped:
            suffix = f"  UNMAPPED={len(r.unmapped)}"
        print(
            f"  cohort={r.cohort}: discovered={len(r.discovered)} "
            f"manifest={len(r.manifest)}{suffix}",
            file=sys.stderr,
        )

    if errors:
        print("\nGUARD ERRORS (exit 2):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    failing = [r for r in reports if not r.passed]
    if failing:
        print(
            "\nMANIFEST COVERAGE FAILED — at least one converter column "
            "has no FeatureContract entry and is not on the audit allowlist.",
            file=sys.stderr,
        )
        for r in failing:
            if r.unmapped:
                print(f"\n  [{r.cohort}] unmapped columns:", file=sys.stderr)
                for col in r.unmapped:
                    print(f"    - {col}", file=sys.stderr)
            if r.manifest_unproduced:
                print(
                    f"\n  [{r.cohort}] manifest entries not produced by converter:",
                    file=sys.stderr,
                )
                for col in r.manifest_unproduced:
                    print(f"    - {col}", file=sys.stderr)
        print(
            "\nTo fix: add the missing column(s) to "
            "src/data/manifests/<cohort>_feature_manifest.py as "
            "FeatureContract entries, OR if the column is a non-feature "
            "audit/identifier column, add it to "
            "AUDIT_COLUMN_ALLOWLIST in this guard (with justification "
            "in the docstring).",
            file=sys.stderr,
        )
        return 1

    print("\nLayer 1 manifest coverage: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
