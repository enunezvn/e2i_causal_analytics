"""GEPA Module Versioning for E2I Agents.

This module provides version management for GEPA-optimized DSPy modules,
enabling save, load, and rollback of optimized agent prompts.

Integrates with:
- optimized_instructions table (database/ml/023_gepa_optimization_tables.sql;
  persistence wired via src/repositories/prompt_optimization.py, migration 035)
- MLflow for experiment tracking
- Local file system for module persistence
"""

import contextlib
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Optional

# Saved artifact names start ``gepa_v{n}_...`` (see generate_version_id).
_GEPA_VERSION_RE = re.compile(r"^gepa_v(\d+)")

# How many version increments an auto-named save attempts when losing the
# creation race to concurrent saves before failing loudly (#1500).
_VERSION_COLLISION_RETRIES = 16


def gepa_artifact_sort_key(path: Path) -> tuple[int, str]:
    """Ordering key for saved GEPA artifact files, oldest first.

    Primary key: the integer after the leading ``gepa_v``, compared
    numerically — the highest lineage version wins "newest", regardless of the
    timestamps embedded in the names. Lexicographic name order inverts at v10
    (``"gepa_v10..." < "gepa_v2..."``, with ``"gepa_v9..."`` greatest of all),
    which pinned newest-artifact resolution to a stale version forever after
    the 10th save (#1496).

    Tie-break, only within one version: the full file name. Within one agent's
    directory the name embeds a zero-padded ``YYYYMMDD_HHMMSS`` timestamp, so
    equal lineage versions resolve by save time — which is what every artifact
    saved before #1500 (when generate_version_id hardcoded ``v1``) relies on.

    A name with no parseable ``gepa_v<n>`` prefix sorts below every well-formed
    version (as version ``-1``): a file the saver could not have produced must
    never win "newest" (lexicographically, e.g. ``gepa_zzz...`` used to outrank
    every real version), but its presence must not crash resolution either.

    Every site that resolves "newest artifact" MUST order with this key — the
    cognitive-RAG module-reload probe (``_artifact_signature`` in
    src/rag/cognitive_rag_dspy.py) keys its cache on the newest artifact's path
    and desynchronizes from this loader if the two ever rank names differently.
    Both call :func:`newest_saved_artifact` so they cannot drift.
    """
    match = _GEPA_VERSION_RE.match(path.name)
    version = int(match.group(1)) if match else -1
    return (version, path.name)


def newest_saved_artifact(directory: Path) -> Optional[Path]:
    """The newest ``gepa_*.json`` artifact in ``directory``, or None if none.

    "Newest" means the highest lineage version (the ``gepa_v<n>`` number); the
    timestamp embedded in the name is only the tie-break within a version —
    see :func:`gepa_artifact_sort_key`.

    The single shared resolver for "which saved version is current":
    :func:`load_optimized_module` below and the cognitive-RAG reload probe both
    call this, so they cannot disagree on which artifact is newest (#1496).
    """
    versions = list(directory.glob("gepa_*.json"))
    if not versions:
        return None
    return max(versions, key=gepa_artifact_sort_key)


def next_artifact_version(directory: Path) -> int:
    """The version number the next artifact saved into ``directory`` should get.

    One more than the highest parseable ``gepa_v<n>`` among the ``gepa_*.json``
    files already there; 1 when the directory does not exist, is empty, or
    holds only names without a version (which :func:`gepa_artifact_sort_key`
    ranks as -1 — they must not drag the next real version to 0).

    ``v<n>`` records optimization lineage (the DB schema's ab_test_variant enum
    and the domain vocabulary both define ``gepa_v2`` as "Second GEPA
    iteration"), so every save of a re-optimized module advances it (#1500).
    Before #1500 generate_version_id hardcoded ``v1``, which also meant two
    saves within one second collided on the same file name.
    """
    if not directory.exists():
        return 1
    versions = [
        int(match.group(1))
        for path in directory.glob("gepa_*.json")
        if (match := _GEPA_VERSION_RE.match(path.name))
    ]
    return max(versions) + 1 if versions else 1


def generate_version_id(
    agent_name: str,
    timestamp: Optional[datetime] = None,
    version: int = 1,
) -> str:
    """Generate a unique version ID for an optimized module.

    Format: gepa_v{n}_{agent}_{timestamp}

    Args:
        agent_name: Name of the agent
        timestamp: Optional timestamp (defaults to now)
        version: The ``v<n>`` lineage number (defaults to 1 — correct when no
            prior artifact exists; save_optimized_module passes
            :func:`next_artifact_version` so re-optimizations increment, #1500)

    Returns:
        Version ID string
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    return f"gepa_v{version}_{agent_name}_{ts_str}"


def compute_instruction_hash(instruction: str) -> str:
    """Compute SHA256 hash of an instruction for deduplication.

    Args:
        instruction: The instruction/prompt text

    Returns:
        Hex-encoded SHA256 hash
    """
    return hashlib.sha256(instruction.encode()).hexdigest()


def save_optimized_module(
    module,
    agent_name: str,
    version_id: Optional[str] = None,
    output_dir: str = "./optimized_modules",
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Save an optimized DSPy module to disk.

    Saves:
    - Module state as JSON
    - Metadata including version, agent, and optimization info
    - Hash for deduplication

    Args:
        module: Optimized DSPy module
        agent_name: Name of the agent
        version_id: Optional version ID. When None, one is minted with the
            next lineage version for this agent's directory and the file is
            created exclusively — concurrent saves that mint the same name
            advance to the next version instead of overwriting (#1500). When
            given, re-saving the same id replaces the existing artifact —
            atomically, so a failed replace leaves the prior artifact intact.
        output_dir: Directory to save modules
        metadata: Additional metadata to save

    Returns:
        Dict with save info (path, version_id, instruction_hash)
    """
    # Create output directory
    output_path = Path(output_dir) / agent_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Get module state
    module_state = module.dump_state() if hasattr(module, "dump_state") else {}

    # Extract instructions for hashing. Older dspy exposed `extended_signature`;
    # dspy 3.x predictors expose `signature`. Fall back so the optimized
    # instructions (and thus the dedup hash) are actually captured — otherwise
    # every saved version hashes to the empty string and dedup silently breaks.
    instructions = []
    if hasattr(module, "predictors"):
        for predictor in module.predictors():
            sig = getattr(predictor, "extended_signature", None) or getattr(
                predictor, "signature", None
            )
            if sig is not None and getattr(sig, "instructions", None):
                instructions.append(sig.instructions)

    instruction_text = "\n---\n".join(instructions)
    instruction_hash = compute_instruction_hash(instruction_text)

    def _build_save_data(vid: str) -> dict[str, Any]:
        return {
            "version_id": vid,
            "agent_name": agent_name,
            "created_at": datetime.now().isoformat(),
            "instruction_hash": instruction_hash,
            "instructions": instructions,
            "module_state": module_state,
            "metadata": metadata or {},
        }

    if version_id is not None:
        # An explicit version_id is the caller's namespace: the caller owns
        # collision semantics, and re-saving the same id replaces the file —
        # atomically. open(path, "w") would truncate on open, so a dump dying
        # mid-stream destroyed the prior artifact AND left invalid JSON that
        # newest_saved_artifact (filename-only) served as "newest". Dump to a
        # same-directory temp (os.replace atomicity requires one filesystem)
        # and replace only on success; a failed replace leaves the prior
        # artifact intact byte-for-byte. The temp name does not match the
        # resolver's ``gepa_*.json`` glob, so even a crash-orphaned temp is
        # inert to resolution. It must also be unique PER CALL, not per PID:
        # two concurrent same-process savers sharing one temp inode would let
        # the loser's still-open fd keep writing into the file the winner's
        # os.replace already published — corrupt JSON behind an ok return.
        save_path = output_path / f"{version_id}.json"
        tmp_path = output_path / f"{version_id}.json.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "w") as f:
                json.dump(_build_save_data(version_id), f, indent=2, default=str)
            os.replace(tmp_path, save_path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_path)
            raise
    else:
        # Auto-generated names must be created exclusively (O_CREAT | O_EXCL):
        # next_artifact_version's directory scan and the write are not one
        # atomic step, so a concurrent save for the same agent can observe the
        # same on-disk max and mint the identical ``gepa_v{n}_{agent}_{ts}``
        # name in the same second. Losing that creation race advances to the
        # next version — it never overwrites (#1500).
        version = next_artifact_version(output_path)
        ts = datetime.now()
        for offset in range(_VERSION_COLLISION_RETRIES):
            candidate = generate_version_id(agent_name, timestamp=ts, version=version + offset)
            candidate_path = output_path / f"{candidate}.json"
            try:
                fd = os.open(candidate_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                # The retry signal — no file was created on this failure. A
                # concurrent save (or a file the scan predates) holds this
                # name: advance the lineage version and try again.
                continue
            # From here the reservation exists on disk. ANY failure before a
            # complete dump — fdopen, payload construction, serialization —
            # must unlink it: newest_saved_artifact resolves by filename
            # alone, so a leftover empty/partial file would become the
            # permanent "newest" (JSONDecodeError on direct loads; the
            # cognitive-RAG fail-soft wrapper silently pinned to the base
            # prompt on every retry).
            try:
                try:
                    artifact_file: IO[str] = os.fdopen(fd, "w")
                except BaseException:
                    # fdopen failed, so the raw fd is still ours to close.
                    with contextlib.suppress(OSError):
                        os.close(fd)
                    raise
                with artifact_file:
                    json.dump(_build_save_data(candidate), artifact_file, indent=2, default=str)
            except BaseException:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(candidate_path)
                raise
            version_id = candidate
            save_path = candidate_path
            break
        else:
            raise FileExistsError(
                f"Could not reserve an artifact name for agent {agent_name!r}: versions "
                f"v{version}..v{version + _VERSION_COLLISION_RETRIES - 1} were all taken "
                "by concurrent saves"
            )

    return {
        "path": str(save_path),
        "version_id": version_id,
        "instruction_hash": instruction_hash,
    }


def load_optimized_module(
    module_cls,
    agent_name: str,
    version_id: Optional[str] = None,
    input_dir: str = "./optimized_modules",
) -> tuple[Any, dict[str, Any]]:
    """Load an optimized DSPy module from disk.

    Args:
        module_cls: The DSPy module class to instantiate
        agent_name: Name of the agent
        version_id: Version to load (loads latest if None)
        input_dir: Directory containing saved modules

    Returns:
        Tuple of (loaded_module, metadata_dict)

    Raises:
        FileNotFoundError: If no saved modules found
    """
    input_path = Path(input_dir) / agent_name

    if not input_path.exists():
        raise FileNotFoundError(f"No saved modules for agent: {agent_name}")

    # Find version to load
    if version_id is None:
        # Load latest = highest lineage version, timestamp tie-break within a
        # version (#1496): a plain name sort inverts at v10 and resolves a
        # stale artifact forever.
        load_path_or_none = newest_saved_artifact(input_path)
        if load_path_or_none is None:
            raise FileNotFoundError(f"No saved versions for agent: {agent_name}")
        load_path = load_path_or_none
    else:
        load_path = input_path / f"{version_id}.json"
        if not load_path.exists():
            raise FileNotFoundError(f"Version not found: {version_id}")

    # Load data
    with open(load_path) as f:
        save_data = json.load(f)

    # Instantiate module
    module = module_cls()

    # Load state if available
    if save_data.get("module_state") and hasattr(module, "load_state"):
        module.load_state(save_data["module_state"])

    metadata = {
        "version_id": save_data["version_id"],
        "created_at": save_data["created_at"],
        "instruction_hash": save_data["instruction_hash"],
        "source_path": str(load_path),
        **save_data.get("metadata", {}),
    }

    return module, metadata


def list_versions(
    agent_name: str,
    input_dir: str = "./optimized_modules",
) -> list[dict[str, Any]]:
    """List all saved versions for an agent.

    Args:
        agent_name: Name of the agent
        input_dir: Directory containing saved modules

    Returns:
        List of version info dicts, sorted by creation date (newest first)
    """
    input_path = Path(input_dir) / agent_name

    if not input_path.exists():
        return []

    versions = []
    for version_file in input_path.glob("gepa_*.json"):
        try:
            with open(version_file) as f:
                data = json.load(f)
            versions.append(
                {
                    "version_id": data["version_id"],
                    "created_at": data["created_at"],
                    "instruction_hash": data["instruction_hash"],
                    "path": str(version_file),
                }
            )
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by creation date, newest first
    versions.sort(key=lambda v: v["created_at"], reverse=True)
    return versions


def rollback_to_version(
    module_cls,
    agent_name: str,
    version_id: str,
    input_dir: str = "./optimized_modules",
) -> tuple[Any, dict[str, Any]]:
    """Rollback to a specific version of an optimized module.

    Args:
        module_cls: The DSPy module class to instantiate
        agent_name: Name of the agent
        version_id: Version to rollback to
        input_dir: Directory containing saved modules

    Returns:
        Tuple of (loaded_module, metadata_dict)
    """
    return load_optimized_module(
        module_cls=module_cls,
        agent_name=agent_name,
        version_id=version_id,
        input_dir=input_dir,
    )


def compare_versions(
    agent_name: str,
    version_id_a: str,
    version_id_b: str,
    input_dir: str = "./optimized_modules",
) -> dict[str, Any]:
    """Compare two versions of an optimized module.

    Args:
        agent_name: Name of the agent
        version_id_a: First version to compare
        version_id_b: Second version to compare
        input_dir: Directory containing saved modules

    Returns:
        Comparison dict with differences
    """
    input_path = Path(input_dir) / agent_name

    # Load both versions
    path_a = input_path / f"{version_id_a}.json"
    path_b = input_path / f"{version_id_b}.json"

    with open(path_a) as f:
        data_a = json.load(f)
    with open(path_b) as f:
        data_b = json.load(f)

    # Compare instructions
    instructions_a = data_a.get("instructions", [])
    instructions_b = data_b.get("instructions", [])

    return {
        "version_a": version_id_a,
        "version_b": version_id_b,
        "hash_match": data_a["instruction_hash"] == data_b["instruction_hash"],
        "instruction_count_a": len(instructions_a),
        "instruction_count_b": len(instructions_b),
        "created_a": data_a["created_at"],
        "created_b": data_b["created_at"],
        "metadata_a": data_a.get("metadata", {}),
        "metadata_b": data_b.get("metadata", {}),
    }


__all__ = [
    "gepa_artifact_sort_key",
    "newest_saved_artifact",
    "next_artifact_version",
    "generate_version_id",
    "compute_instruction_hash",
    "save_optimized_module",
    "load_optimized_module",
    "list_versions",
    "rollback_to_version",
    "compare_versions",
]
