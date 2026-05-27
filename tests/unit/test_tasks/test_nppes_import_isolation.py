"""Regression: celery autodiscover must not crash when ``scripts/`` is absent.

Prod incident 2026-05-26: ``src/tasks/nppes_tasks.py`` imported
``scripts.rwd_common`` at module top level. The deployed container ships only
``src/`` (not ``scripts/``), so celery ``autodiscover_tasks(["src.tasks", ...])``
raised ``ModuleNotFoundError: No module named 'scripts'`` at boot and every
worker + the beat scheduler crash-looped.

This pins the boot path: importing the task module must succeed even when
``scripts`` is unimportable (mirroring the container's sys.path).
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys

import pytest


class _BlockScripts(importlib.abc.MetaPathFinder):
    """Make ``import scripts[...]`` fail, simulating a container with no
    ``scripts/`` dir on sys.path."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by test)")
        return None


@pytest.fixture
def scripts_unimportable(monkeypatch):
    """Block ``scripts`` imports and purge cached modules so the task module's
    top-level imports re-execute under the block."""
    for name in list(sys.modules):
        if name == "scripts" or name.startswith("scripts.") or name == "src.tasks.nppes_tasks":
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockScripts(), *sys.meta_path])
    # sanity: the block is actually active
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scripts.rwd_common")


def test_nppes_tasks_imports_without_scripts(scripts_unimportable):
    """``src.tasks.nppes_tasks`` must import with ``scripts/`` absent (celery boot path)."""
    mod = importlib.import_module("src.tasks.nppes_tasks")
    assert hasattr(mod, "refresh_npi_taxonomy_cache")
