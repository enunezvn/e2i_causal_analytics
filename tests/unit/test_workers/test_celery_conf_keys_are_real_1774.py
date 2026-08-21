"""CI guard: every setting we pass to ``celery_app.conf`` must be a real one (#1774).

The defect
----------
``celery_app.conf.update()`` accepts anything. Unknown keys are stored as inert
custom config and Celery never looks at them, so a typo — or a setting that does
not exist at app level at all — lands silently and *reads* as configured
behaviour. #1774 was four of them::

    task_autoretry_for=(Exception,)
    task_retry_kwargs={"max_retries": 3}
    task_retry_backoff=True
    task_retry_backoff_max=600

``autoretry_for`` / ``retry_kwargs`` / ``retry_backoff`` / ``retry_backoff_max``
are **per-task decorator arguments**. ``celery/app/autoretry.py`` reads them from
exactly two places — the decorator's ``options`` and an attribute on the ``Task``
class — never from ``app.conf``. The strings ``task_autoretry_for`` &c. appear
nowhere in the celery or kombu source. The config therefore documented a
platform-wide retry policy the platform has never had, which is a false claim
about production behaviour, not a style nit.

Measured before removing them (all 54 registered ``src.*`` tasks, against a
positive control that asks for autoretry explicitly):

    autoretry-WRAPPED       : 0 []
    POSITIVE CONTROL wrapped: True

Two guards, because they catch different things
-----------------------------------------------
* the source guard reads what ``celery_app.py`` *passes*, so a phantom key is
  caught even if something later overwrites it;
* the runtime guard reads what the app ends up *configured with*, so a phantom
  key set from any other module is caught too.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

from celery.app.defaults import NAMESPACES, flatten

from src.workers.celery_app import celery_app

# Parse the module's own __file__ so the guard always reads the file that was
# actually imported (worktree vs editable install), never a guessed path.
# importlib.import_module rather than `import src.workers.celery_app as ...`:
# src/workers/__init__.py does `from .celery_app import celery_app`, which rebinds
# the name `src.workers.celery_app` to the Celery *instance* and shadows the
# submodule, so the plain import form hands back an object with no __file__.
_SOURCE_PATH = pathlib.Path(importlib.import_module("src.workers.celery_app").__file__)

# Every setting name Celery actually understands.
_REAL_SETTINGS = {key.lower() for key, _ in flatten(NAMESPACES)}

# Keys that appear in ``conf.changes`` but are NOT passed by our code: Celery's
# own ``detect_settings()`` injects ``deprecated_settings`` (celery/app/utils.py)
# when it builds the Settings object. Verified by grep: nothing in this repo
# writes it. This is the only such key — it is not a general allow-list, and a
# new entry here needs the same evidence.
_CELERY_INJECTED = {"deprecated_settings"}

# Settings we know are real, used as positive controls so a green run cannot come
# from an extractor that found nothing.
_CONTROLS = {"task_acks_late", "task_default_queue", "task_time_limit"}


def _settings_passed_in_source() -> set[str]:
    """Names given to ``*.conf.update(**kwargs)`` and ``*.conf.<name> = ...``."""
    tree = ast.parse(_SOURCE_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()

    for node in ast.walk(tree):
        # celery_app.conf.update(key=value, ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "conf"
        ):
            names.update(kw.arg for kw in node.keywords if kw.arg is not None)

        # celery_app.conf.<name> = ...
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "conf"
                ):
                    names.add(target.attr)

    return names


def test_extractor_is_not_blind() -> None:
    """Positive control for the source guard — a null result needs one."""
    assert _SOURCE_PATH.is_file(), f"celery_app source not found at {_SOURCE_PATH}"

    passed = _settings_passed_in_source()
    assert len(passed) >= 20, (
        f"only {len(passed)} settings extracted from {_SOURCE_PATH.name} — the AST "
        "walk has stopped matching how the config is written, so the guard below "
        f"would pass vacuously. Extracted: {sorted(passed)}"
    )
    missing_controls = sorted(_CONTROLS - passed)
    assert not missing_controls, (
        f"known-real settings not seen by the extractor: {missing_controls}. Fix the "
        "extractor before trusting a clean run."
    )
    assert len(_REAL_SETTINGS) >= 100, (
        f"celery.app.defaults yielded only {len(_REAL_SETTINGS)} names — the "
        "membership test below would reject everything or accept everything."
    )


def test_every_setting_passed_in_source_is_a_real_celery_setting() -> None:
    """No phantom key may be passed to ``conf`` from ``celery_app.py``."""
    phantom = sorted(name for name in _settings_passed_in_source() if name not in _REAL_SETTINGS)

    assert not phantom, (
        f"{_SOURCE_PATH.name} passes config keys that are not Celery settings: "
        f"{phantom}. conf.update() accepts anything and stores unknown keys as inert "
        "custom config, so these configure nothing while reading as though they do "
        "(#1774). If the behaviour is genuinely wanted, apply it where Celery reads "
        "it — usually a per-task decorator argument — not here."
    )


def test_app_conf_has_no_unknown_settings() -> None:
    """Same invariant at runtime, so a phantom key set from any module is caught."""
    changes = set(celery_app.conf.changes)
    assert _CONTROLS <= changes, (
        f"known-real settings missing from conf.changes: {sorted(_CONTROLS - changes)} — "
        "conf.changes is no longer reporting what our code set, so this guard would "
        "pass vacuously."
    )

    phantom = sorted(
        key for key in changes if key not in _REAL_SETTINGS and key not in _CELERY_INJECTED
    )
    assert not phantom, (
        f"celery_app.conf carries keys that are not Celery settings: {phantom}. Celery "
        "never reads them; they document behaviour the platform does not have (#1774)."
    )


def test_retry_is_a_per_task_decision_not_app_config() -> None:
    """Pin the #1774 decision: there is no app-level autoretry, by design.

    ``src.tasks.graph_emptiness_sentinel`` (#1761) sets ``max_retries=0`` and raises
    ``GraphReseedError`` on a partial reseed *because* nothing retries it globally —
    an app-wide autoretry would turn one failed heal into a burst of ``CREATE``-based
    reseeds. The absence is load-bearing, so assert it rather than leaving it to
    whoever next reads the config.
    """
    for phantom in (
        "task_autoretry_for",
        "task_retry_kwargs",
        "task_retry_backoff",
        "task_retry_backoff_max",
    ):
        assert phantom not in set(celery_app.conf.changes), (
            f"{phantom} is back in celery_app.conf. It is not a Celery setting and "
            "configures nothing (#1774). Retries are declared per task, on the "
            "decorator, where Celery reads them."
        )

    # Stock Task base: no class-level autoretry attributes either, which is the
    # only other place celery/app/autoretry.py looks.
    assert celery_app.conf.get("task_cls") is None
    for attr in ("autoretry_for", "retry_backoff", "retry_backoff_max", "retry_kwargs"):
        assert not hasattr(celery_app.Task, attr), (
            f"celery_app.Task now carries {attr}, which applies autoretry to EVERY "
            "task on the platform — including src.tasks.graph_emptiness_sentinel, "
            "whose max_retries=0 assumes it is never retried automatically (#1761)."
        )
