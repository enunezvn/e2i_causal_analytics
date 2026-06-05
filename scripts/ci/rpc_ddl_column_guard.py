#!/usr/bin/env python3
"""RPC-vs-DDL column guard (memory-system audit 2026-06-05, Rec 5 / F1).

Static check that diffs every `CREATE FUNCTION`'s `<alias>.<column>` references
against the actual `CREATE TABLE` / `ALTER TABLE ... ADD COLUMN` DDL for the
table the alias is bound to (via `FROM`/`JOIN`). Flags references to columns the
table does not define — exactly the defect the broken `016` RPCs carry
(`cognitive_cycles cc` referencing `cc.agent_response`, `cc.feedback_*`, ...).

stdlib-only, no DB, no third-party deps — runs anywhere (mirrors the
`scripts/check_manifest_coverage.py` guard convention). The faithful environment
for a "does this column exist?" question is the schema source, which this reads
directly; mocked unit tests would not catch the drift, which is why `016`
survived.

Exit codes:
  0 — no phantom references (or `--report-only`).
  1 — phantom references found (blocking).
  2 — invocation/discovery error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"

# Tokens that must never be treated as a table alias after FROM/JOIN.
_ALIAS_STOPWORDS = {
    "where", "on", "using", "group", "order", "limit", "offset", "left", "right",
    "inner", "outer", "full", "cross", "join", "union", "and", "or", "set",
    "returning", "loop", "return", "select", "as", "lateral", "having", "window",
    "for", "into", "values", "when", "then", "else", "end", "natural",
}
# Column-def lines inside CREATE TABLE that are constraints, not columns.
_CONSTRAINT_KEYWORDS = {
    "constraint", "primary", "foreign", "unique", "check", "exclude", "like",
}


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def _balanced_parens(text: str, open_idx: int) -> Tuple[str, int]:
    """Return (inner, end_idx) for the parenthesised group starting at `open_idx`
    (which must point at '('). Respects nesting and single-quoted strings."""
    depth = 0
    i = open_idx
    in_str = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "'":
                in_str = False
        elif ch == "'":
            in_str = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : i], i
        i += 1
    return text[open_idx + 1 :], len(text)


def _split_top_level_commas(s: str) -> List[str]:
    parts, depth, buf = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _norm_table(name: str) -> str:
    """Strip schema qualifier and quotes: public."Foo" -> foo."""
    name = name.split(".")[-1].strip().strip('"')
    return name.lower()


def parse_table_columns(sql: str) -> Dict[str, Set[str]]:
    """Map table -> set(columns) from CREATE TABLE + ALTER TABLE ADD COLUMN."""
    tables: Dict[str, Set[str]] = {}
    # CREATE TABLE [IF NOT EXISTS] [schema.]name ( ... )
    for m in re.finditer(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w.\"]*)\s*\(",
        sql,
        flags=re.IGNORECASE,
    ):
        table = _norm_table(m.group(1))
        inner, _ = _balanced_parens(sql, m.end() - 1)
        cols = tables.setdefault(table, set())
        for entry in _split_top_level_commas(inner):
            entry = entry.strip()
            if not entry:
                continue
            first = entry.split(None, 1)[0].strip().strip('"')
            if first.lower() in _CONSTRAINT_KEYWORDS:
                continue
            if re.fullmatch(_IDENT, first):
                cols.add(first.lower())
    # ALTER TABLE [schema.]name ... — capture EVERY `ADD COLUMN` in the
    # statement (a single ALTER can add many comma-separated columns; matching
    # only the first is the parser gap that false-positives on multi-column
    # ALTERs like database/ml/011_realtime_shap_audit.sql).
    for m in re.finditer(
        r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?([A-Za-z_][\w.\"]*)\b(.*?);",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        table = _norm_table(m.group(1))
        for cm in re.finditer(
            rf"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\"?{_IDENT}\"?)",
            m.group(2),
            flags=re.IGNORECASE,
        ):
            tables.setdefault(table, set()).add(cm.group(1).strip('"').lower())
    return tables


def _function_bodies(sql: str) -> List[Tuple[str, str]]:
    """Return (function_name, body) for each CREATE FUNCTION (dollar-quoted body)."""
    out: List[Tuple[str, str]] = []
    for m in re.finditer(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+([A-Za-z_][\w.\"]*)\s*\(",
        sql,
        flags=re.IGNORECASE,
    ):
        name = _norm_table(m.group(1))
        # Body is the first dollar-quoted block after the signature.
        dq = re.search(r"\$(\w*)\$", sql[m.end() :])
        if not dq:
            continue
        tag = dq.group(0)
        start = m.end() + dq.end()
        end = sql.find(tag, start)
        body = sql[start:end] if end != -1 else sql[start:]
        out.append((name, body))
    return out


def _alias_table_map(body: str) -> Dict[str, str]:
    amap: Dict[str, str] = {}
    for m in re.finditer(
        rf"\b(?:FROM|JOIN)\s+(?:ONLY\s+)?([A-Za-z_][\w.\"]*)\s+(?:AS\s+)?({_IDENT})",
        body,
        flags=re.IGNORECASE,
    ):
        table = _norm_table(m.group(1))
        alias = m.group(2)
        if alias.lower() in _ALIAS_STOPWORDS:
            continue
        amap[alias.lower()] = table
    return amap


def find_phantom_column_references(db_dir: Path) -> List[dict]:
    """Return a list of phantom references: {file, function, alias, column, table}."""
    sql_files = sorted(db_dir.rglob("*.sql"))
    if not sql_files:
        raise FileNotFoundError(f"no .sql files under {db_dir}")
    # Global table->columns (merge across all files; tables may be altered later).
    table_columns: Dict[str, Set[str]] = {}
    raw: Dict[Path, str] = {}
    for f in sql_files:
        text = _strip_sql_comments(f.read_text(encoding="utf-8", errors="ignore"))
        raw[f] = text
        for tbl, cols in parse_table_columns(text).items():
            table_columns.setdefault(tbl, set()).update(cols)

    findings: List[dict] = []
    for f, text in raw.items():
        for fn_name, body in _function_bodies(text):
            amap = _alias_table_map(body)
            if not amap:
                continue
            # SCOPE (deliberate, to stay trustworthy): only analyze unambiguous
            # SINGLE-SOURCE functions — exactly one alias bound to exactly one
            # KNOWN table (the `016` shape: `FROM cognitive_cycles cc`). Multi-
            # table joins / CTEs need per-statement alias scoping to validate
            # without false positives (an alias can bind to different tables in
            # different sub-queries of one function body); flagging there would
            # produce noise, so we conservatively skip. This still catches the
            # entire defect CLASS the audit's Rec 5 names (an RPC referencing
            # phantom columns on its one backing table). Documented limitation:
            # column drift inside multi-table joins is NOT covered.
            known = {a: t for a, t in amap.items() if t in table_columns}
            if len(known) != 1:
                continue
            sole_alias, sole_table = next(iter(known.items()))
            for ref in re.finditer(rf"\b({_IDENT})\.({_IDENT})\b", body):
                alias, col = ref.group(1).lower(), ref.group(2).lower()
                if alias != sole_alias:
                    continue
                table = sole_table
                cols = table_columns.get(table)
                # Only flag when we actually know the table's columns AND the
                # column is genuinely absent (conservative: never flag a table
                # whose DDL we could not parse).
                if cols and col not in cols:
                    findings.append(
                        {
                            "file": str(f.relative_to(db_dir.parent)),
                            "function": fn_name,
                            "alias": alias,
                            "column": col,
                            "table": table,
                        }
                    )
    return findings


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "database",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="print findings but exit 0 (used to baseline before a known fix lands)",
    )
    args = ap.parse_args(argv)
    try:
        findings = find_phantom_column_references(args.db_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"rpc_ddl_column_guard: discovery error: {exc}", file=sys.stderr)
        return 2
    if not findings:
        print("rpc_ddl_column_guard: OK — no phantom RPC column references.")
        return 0
    print(
        f"rpc_ddl_column_guard: found {len(findings)} phantom RPC column "
        f"reference(s) (column referenced by a function but absent from the "
        f"table's DDL):",
        file=sys.stderr,
    )
    for f in sorted(findings, key=lambda d: (d["file"], d["function"], d["column"])):
        print(
            f"  {f['file']}: {f['function']}() references {f['alias']}.{f['column']} "
            f"-> table '{f['table']}' has no column '{f['column']}'",
            file=sys.stderr,
        )
    if args.report_only:
        print("rpc_ddl_column_guard: --report-only set, exiting 0 (non-blocking).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
