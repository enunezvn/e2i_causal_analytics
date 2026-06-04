#!/usr/bin/env python3
"""
SQL-statement-aware idempotency transformer for migration files.

Splits a .sql file into top-level statements (respecting $$-dollar-quoting,
'..' strings, -- line comments and /* */ block comments) and rewrites the
non-idempotent DDL so the migration is safe to re-run:

  CREATE TYPE x AS ENUM (...)   ->  DO $$ BEGIN <stmt>; EXCEPTION
                                       WHEN duplicate_object THEN null; END $$
  CREATE TABLE x (...)          ->  CREATE TABLE IF NOT EXISTS x (...)
  CREATE [UNIQUE] INDEX x ...    ->  CREATE [UNIQUE] INDEX IF NOT EXISTS x ...
                                    (CONCURRENTLY left alone — handled by runner)
  CREATE TRIGGER x ... ON t ...  ->  DROP TRIGGER IF EXISTS x ON t; CREATE TRIGGER ...
  CREATE POLICY x ON t ...       ->  DROP POLICY IF EXISTS x ON t; CREATE POLICY ...

Statements already carrying IF NOT EXISTS / wrapped in a DO block / using
CREATE OR REPLACE are left untouched. Verified by the faithful prod
BEGIN..ROLLBACK dry-run after transform — this only makes the change; the
dry-run proves it.

Usage: _idempotent_migration.py <file.sql> [<file.sql> ...]
"""
import re
import sys


def split_statements(sql: str):
    """Yield (statement_text_including_trailing_semicolon) at top level."""
    out, buf = [], []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        two = sql[i:i + 2]
        if two == "--":
            j = sql.find("\n", i)
            j = n if j == -1 else j + 1
            buf.append(sql[i:j]); i = j; continue
        if two == "/*":
            j = sql.find("*/", i)
            j = n if j == -1 else j + 2
            buf.append(sql[i:j]); i = j; continue
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'" and sql[j + 1:j + 2] == "'":
                    j += 2; continue
                if sql[j] == "'":
                    j += 1; break
                j += 1
            buf.append(sql[i:j]); i = j; continue
        m = re.match(r"\$([A-Za-z0-9_]*)\$", sql[i:])
        if m:
            tag = m.group(0)
            j = sql.find(tag, i + len(tag))
            j = n if j == -1 else j + len(tag)
            buf.append(sql[i:j]); i = j; continue
        if ch == ";":
            buf.append(";"); out.append("".join(buf)); buf = []; i += 1; continue
        buf.append(ch); i += 1
    if "".join(buf).strip():
        out.append("".join(buf))
    return out


def code_prefix(stmt: str) -> str:
    """The statement with comments/whitespace stripped from the front, for matching."""
    s = stmt
    while True:
        s2 = s.lstrip()
        if s2.startswith("--"):
            s = s2.split("\n", 1)[1] if "\n" in s2 else ""
        elif s2.startswith("/*"):
            s = s2.split("*/", 1)[1] if "*/" in s2 else ""
        else:
            return s2


def transform_statement(stmt: str) -> str:
    head = code_prefix(stmt)
    up = head.upper()

    # CREATE TYPE ... AS ENUM  -> wrap in DO/EXCEPTION (idempotent)
    if re.match(r"CREATE\s+TYPE\s+", up) and " AS ENUM" in up:
        body = stmt.strip()
        if body.endswith(";"):
            body = body[:-1]
        return ("DO $idem$ BEGIN\n" + body +
                ";\nEXCEPTION WHEN duplicate_object THEN null; END $idem$;")

    # CREATE TABLE name -> IF NOT EXISTS
    m = re.match(r"(CREATE\s+TABLE\s+)(?!IF\s+NOT\s+EXISTS)", up)
    if m:
        return re.sub(r"(CREATE\s+TABLE\s+)(?!IF\s+NOT\s+EXISTS)",
                      r"\1IF NOT EXISTS ", stmt, count=1, flags=re.IGNORECASE)

    # CREATE [UNIQUE] INDEX name (not CONCURRENTLY, not IF NOT EXISTS)
    if re.match(r"CREATE\s+(UNIQUE\s+)?INDEX\s+", up) and "CONCURRENTLY" not in up \
            and not re.match(r"CREATE\s+(UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS", up):
        return re.sub(r"(CREATE\s+(?:UNIQUE\s+)?INDEX\s+)(?!IF\s+NOT\s+EXISTS)",
                      r"\1IF NOT EXISTS ", stmt, count=1, flags=re.IGNORECASE)

    # CREATE VIEW name -> CREATE OR REPLACE VIEW
    if re.match(r"CREATE\s+VIEW\s+", up):
        return re.sub(r"CREATE\s+VIEW\s+", "CREATE OR REPLACE VIEW ",
                      stmt, count=1, flags=re.IGNORECASE)

    # identifier: bare word OR a "double-quoted name" (may contain spaces)
    IDENT = r'("(?:[^"]|"")*"|[A-Za-z0-9_]+)'
    QUAL = r'("(?:[^"]|"")*"|[A-Za-z0-9_.]+)'

    # CREATE TRIGGER name ... ON table  -> DROP TRIGGER IF EXISTS first
    mt = re.match(r"CREATE\s+TRIGGER\s+" + IDENT, head, re.IGNORECASE)
    if mt and not up.startswith("CREATE OR REPLACE"):
        on = re.search(r"\bON\s+" + QUAL, head, re.IGNORECASE)
        if on:
            name, tbl = mt.group(1), on.group(1)
            lead = stmt[:len(stmt) - len(stmt.lstrip())]
            return f"{lead}DROP TRIGGER IF EXISTS {name} ON {tbl};\n{stmt.lstrip()}"

    # CREATE POLICY name ON table  -> DROP POLICY IF EXISTS first
    mp = re.match(r"CREATE\s+POLICY\s+" + IDENT + r"\s+ON\s+" + QUAL, head, re.IGNORECASE)
    if mp:
        name, tbl = mp.group(1), mp.group(2)
        lead = stmt[:len(stmt) - len(stmt.lstrip())]
        return f"{lead}DROP POLICY IF EXISTS {name} ON {tbl};\n{stmt.lstrip()}"

    return stmt


def transform_file(path: str) -> bool:
    src = open(path).read()
    stmts = split_statements(src)
    new = "".join(transform_statement(s) for s in stmts)
    if new != src:
        open(path, "w").write(new)
        return True
    return False


if __name__ == "__main__":
    changed = [f for f in sys.argv[1:] if transform_file(f)]
    print(f"idempotency-transformed {len(changed)} file(s):")
    for f in changed:
        print("  ", f)
