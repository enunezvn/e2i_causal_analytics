"""A cert cites its evidence by filename; that evidence must be IN the repo.

#2115 shipped a `cert.md` whose evidence table cited `live_cert.log` and
`timesfm_image_suite.log`. Both existed on the machine that ran the
certification and neither was committed: `.gitignore` carries a blanket
`*.log`, so `git add <dir>` skipped them silently and the cert's own table
pointed at files no reviewer could open. The lane worktree was one
`git worktree remove` away from destroying them.

This is the second time a header-cited evidence file turned out to be
untracked, so it is a guard rather than a note.

Scope and limits, stated plainly: this can only catch a cited file that is
still present in the working tree. Once the evidence is deleted there is
nothing left to distinguish "cited file was never committed" from "cited
file lives somewhere else on purpose", which is exactly why the check has
to run while the lane is still open.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "docs" / "demos" / "results"

# Backtick-quoted filenames with an evidence-ish extension, e.g. `live_cert.log`.
_CITATION = re.compile(r"`([A-Za-z0-9_.\-/]+\.(?:log|json|jsonl|csv|txt|py|sql))`")


def _tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {line for line in out.split("\n") if line}


def test_every_cert_cites_evidence_that_is_committed() -> None:
    """A file a cert names, that sits next to it on disk, must be tracked."""
    certs = sorted(RESULTS_DIR.glob("*/cert.md"))
    assert certs, f"no cert.md found under {RESULTS_DIR} — has the layout moved?"

    tracked = _tracked_paths()
    offenders: list[str] = []

    for cert in certs:
        lane = cert.parent
        for name in sorted(set(_CITATION.findall(cert.read_text(errors="replace")))):
            candidate = lane / name
            if not candidate.exists():
                # Cited but absent here: it may legitimately live elsewhere
                # (a script under scripts/, another lane's cert). Not this
                # guard's business — it can only speak about what it can see.
                continue
            rel = candidate.relative_to(REPO_ROOT).as_posix()
            if rel not in tracked:
                offenders.append(f"{cert.relative_to(REPO_ROOT)} cites untracked {rel}")

    assert not offenders, (
        "cert.md files cite evidence that exists on disk but is not committed, so "
        "the evidence dies with the worktree:\n  " + "\n  ".join(offenders) + "\n"
        "Commit it (a blanket .gitignore rule such as `*.log` is the usual cause; "
        "add a negation for docs/demos/results/)."
    )
