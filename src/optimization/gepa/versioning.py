"""GEPA Module Versioning for E2I Agents.

This module provides version management for GEPA-optimized DSPy modules,
enabling save, load, and rollback of optimized agent prompts.

Integrates with:
- optimized_instructions table (database/ml/023_gepa_optimization_tables.sql)
- MLflow for experiment tracking
- Local file system for module persistence
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dspy import Example


def generate_version_id(agent_name: str, timestamp: Optional[datetime] = None) -> str:
    """Generate a unique version ID for an optimized module.

    Format: gepa_v{n}_{agent}_{timestamp}

    Args:
        agent_name: Name of the agent
        timestamp: Optional timestamp (defaults to now)

    Returns:
        Version ID string
    """
    ts = timestamp or datetime.now()
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    return f"gepa_v1_{agent_name}_{ts_str}"


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
        version_id: Optional version ID (auto-generated if None)
        output_dir: Directory to save modules
        metadata: Additional metadata to save

    Returns:
        Dict with save info (path, version_id, instruction_hash)
    """
    # Generate version ID if not provided
    if version_id is None:
        version_id = generate_version_id(agent_name)

    # Create output directory
    output_path = Path(output_dir) / agent_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Get module state
    module_state = module.dump_state() if hasattr(module, "dump_state") else {}

    # Extract instructions for hashing
    instructions = []
    if hasattr(module, "predictors"):
        for predictor in module.predictors():
            if hasattr(predictor, "extended_signature"):
                sig = predictor.extended_signature
                if hasattr(sig, "instructions"):
                    instructions.append(sig.instructions)

    instruction_text = "\n---\n".join(instructions)
    instruction_hash = compute_instruction_hash(instruction_text)

    # Build save data
    save_data = {
        "version_id": version_id,
        "agent_name": agent_name,
        "created_at": datetime.now().isoformat(),
        "instruction_hash": instruction_hash,
        "instructions": instructions,
        "module_state": module_state,
        "metadata": metadata or {},
    }

    # Save to file
    save_path = output_path / f"{version_id}.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

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
        # Load latest version
        versions = sorted(input_path.glob("gepa_*.json"), reverse=True)
        if not versions:
            raise FileNotFoundError(f"No saved versions for agent: {agent_name}")
        load_path = versions[0]
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
            versions.append({
                "version_id": data["version_id"],
                "created_at": data["created_at"],
                "instruction_hash": data["instruction_hash"],
                "path": str(version_file),
            })
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
    "generate_version_id",
    "compute_instruction_hash",
    "save_optimized_module",
    "load_optimized_module",
    "list_versions",
    "rollback_to_version",
    "compare_versions",
]
