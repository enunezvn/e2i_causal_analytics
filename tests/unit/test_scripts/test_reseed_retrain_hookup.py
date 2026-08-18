"""Guard the weekly gold-standard retrain hookup in the reseed cron wrapper.

The Monday-3AM cron runs ``scripts/reseed_synthetic.sh`` (crontab line frozen —
see the wrapper header). The goldstd retrain stage rides that wrapper rather
than owning a crontab entry, so if the hookup were dropped in a refactor the 12
staging models would silently go stale against the growing substrate (the exact
condition the 2026-07-04 probe found: registry rows trained on 2,853 rows vs
6,651 live). These text-level assertions pin the contract.
"""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESEED = REPO_ROOT / "scripts" / "reseed_synthetic.sh"
RETRAIN = REPO_ROOT / "scripts" / "retrain_goldstd.sh"


def test_retrain_script_runs_both_cohort_clis() -> None:
    """retrain_goldstd.sh must cover all 12 slots: 9 patient + 3 HCP."""
    text = RETRAIN.read_text()
    assert text.startswith("#!/bin/bash")
    assert "set -euo pipefail" in text
    assert "src.mlops.gold_standard_eval.run_patient_cohorts" in text
    assert "src.mlops.gold_standard_eval.run_hcp_cohorts" in text
    # The dotenv/PYTHONPATH/LOKY invocation gotchas the wrapper headers warn
    # about — losing either silently breaks the cron run on the droplet.
    assert ".venv/bin/dotenv -f .env run --" in text
    assert "LOKY_MAX_CPU_COUNT=1" in text


def test_retrain_script_repromotes_hcp_adoption_champions() -> None:
    """The HCP-slot UPSERT resets the three hcp_adoption rows to staging,
    demoting the production champions the chat propensity path serves from —
    the weekly demotion loop of #1690. The wrapper must re-promote through the
    #1384 calibration gate with ``--execute`` (dry-run writes nothing), AFTER
    the registration that demotes, and failure-tolerantly (a bare invocation
    under ``set -e`` would abort the wrapper and kill downstream reseed
    stages the retrain does not own)."""
    text = RETRAIN.read_text()
    assert "scripts/promote_hcp_adoption_champions.py --execute" in text
    assert text.index("src.mlops.gold_standard_eval.run_hcp_cohorts") < text.index(
        "promote_hcp_adoption_champions.py"
    )
    # Failure-tolerant wiring: the promotion invocation is an `if !` guard
    # (warn + continue), never a bare command under set -euo pipefail.
    assert "if ! PYTHONPATH=" in text
    assert "WARNING: hcp_adoption champion re-promotion FAILED" in text


def test_retrain_script_is_executable_in_git_index() -> None:
    """The INDEX mode is what CI/cron checkouts honor (core.fileMode=false on
    the droplet hides a 100644 locally — the #1128/#1134 lesson)."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "scripts/retrain_goldstd.sh"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert out.stdout.startswith("100755"), (
        f"scripts/retrain_goldstd.sh must be 100755 in the git index, got: "
        f"{out.stdout!r} (fix: git update-index --chmod=+x)"
    )


def test_reseed_wrapper_invokes_retrain_with_opt_out() -> None:
    """reseed_synthetic.sh must call the retrain stage and honor --skip-retrain."""
    text = RESEED.read_text()
    assert "retrain_goldstd.sh" in text
    assert "--skip-retrain" in text
    # The opt-out flag must be consumed by the wrapper, not forwarded to
    # load_synthetic_data.py (which would die on the unknown argument).
    assert 'if [[ "$arg" == "--skip-retrain" ]]' in text
