"""
Utilities for content-addressable artifact caching with metadata validation.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
from src.config import Config


def get_config_hash(cfg: Config, artifact_type: str) -> str:
    """
    Generate a deterministic hash from config parameters relevant to the artifact.

    Args:
        cfg: Configuration object
        artifact_type: One of 'tokens', 'llm_activations', 'sae_activations', 'embeddings'

    Returns:
        8-character hex hash string
    """
    hash_params = {
        "data_name": cfg.data.name,
        "data_elements_filename": cfg.data.elements_filename,
        "data_trajectories_filename": cfg.data.trajectories_filename,
        "data_fixed_context_length": cfg.data.fixed_context_length,
        "llm_hf_name": cfg.llm.hf_name,
        "llm_layer_idx": cfg.llm.layer_idx,
    }

    # Add aggregation method for embeddings (they are saved post-aggregation)
    if artifact_type == "embeddings":
        hash_params["sequence_aggregation_method"] = cfg.exp.sequence_aggregation_method

    # Add SAE-specific params if this is an SAE artifact
    if artifact_type == "sae_activations" and cfg.sae is not None:
        hash_params.update({
            "sae_llm_name": cfg.sae.llm_name,
            "sae_llm_layer_idx": cfg.sae.llm_layer_idx,
            "sae_arch": cfg.sae.arch,
            "sae_act_scaling_factor": cfg.sae.act_scaling_factor,
        })

    # Create deterministic JSON string (sorted keys)
    hash_str = json.dumps(hash_params, sort_keys=True)

    # Generate hash
    return hashlib.sha256(hash_str.encode()).hexdigest()[:8]


def get_artifact_metadata(cfg: Config, artifact_type: str) -> Dict[str, Any]:
    """
    Create metadata dictionary for an artifact.

    Args:
        cfg: Configuration object
        artifact_type: One of 'tokens', 'llm_activations', 'sae_activations'

    Returns:
        Metadata dictionary with config snapshot and artifact info
    """
    import datetime

    metadata = {
        "artifact_type": artifact_type,
        "created_at": datetime.datetime.now().isoformat(),
        "config_hash": get_config_hash(cfg, artifact_type),
        "config_snapshot": {
            "data": asdict(cfg.data),
            "llm": asdict(cfg.llm),
            "env": {
                "dtype": str(cfg.env.dtype),
                "device": cfg.env.device,
            }
        }
    }

    # Add SAE config if relevant
    if cfg.sae is not None and artifact_type == "sae_activations":
        metadata["config_snapshot"]["sae"] = asdict(cfg.sae)

    # Add experiment config for embeddings (aggregation method affects cached values)
    if artifact_type == "embeddings":
        metadata["config_snapshot"]["exp"] = {
            "sequence_aggregation_method": cfg.exp.sequence_aggregation_method,
        }

    return metadata


def save_artifact_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """Save metadata to JSON file."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_artifact_metadata(metadata_path: Path) -> Optional[Dict[str, Any]]:
    """Load metadata from JSON file, return None if doesn't exist."""
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r") as f:
        return json.load(f)


def validate_artifact_metadata(
    cfg: Config,
    artifact_type: str,
    metadata_path: Path,
    strict: bool = True
) -> tuple[bool, Optional[str]]:
    """
    Validate that artifact metadata matches current config.

    Args:
        cfg: Current configuration
        artifact_type: Type of artifact being validated
        metadata_path: Path to metadata JSON file
        strict: If True, raise error on mismatch. If False, return validation result.

    Returns:
        (is_valid, error_message) tuple
    """
    metadata = load_artifact_metadata(metadata_path)

    if metadata is None:
        msg = f"Metadata file not found: {metadata_path}"
        if strict:
            raise FileNotFoundError(msg)
        return False, msg

    # Check hash matches
    current_hash = get_config_hash(cfg, artifact_type)
    stored_hash = metadata.get("config_hash")

    if current_hash != stored_hash:
        msg = (
            f"Config hash mismatch for {artifact_type}!\n"
            f"Current hash: {current_hash}\n"
            f"Stored hash:  {stored_hash}\n"
            f"This likely means the config changed after the artifact was cached.\n"
            f"Set force_recompute=True to regenerate the artifact."
        )
        if strict:
            raise ValueError(msg)
        return False, msg

    return True, None


def get_token_artifact_path(cfg: Config) -> tuple[Path, Path]:
    """
    Get paths for token artifact and its metadata.

    Returns:
        (artifact_path, metadata_path) tuple
    """
    config_hash = get_config_hash(cfg, "tokens")
    tokens_dir = Path(cfg.env.tokens_dir)

    artifact_path = tokens_dir / f"{cfg.data.name}_{config_hash}.safetensors"
    metadata_path = tokens_dir / f"{cfg.data.name}_{config_hash}.json"

    return artifact_path, metadata_path


def get_llm_activation_artifact_path(cfg: Config) -> tuple[Path, Path]:
    """
    Get paths for LLM activation artifact and its metadata.

    Returns:
        (artifact_path, metadata_path) tuple
    """
    config_hash = get_config_hash(cfg, "llm_activations")
    activations_dir = Path(cfg.env.activations_dir)

    base_name = f"{cfg.data.name}_{cfg.llm.name}_layer{cfg.llm.layer_idx}_{config_hash}"
    artifact_path = activations_dir / f"{base_name}_llm.safetensors"
    metadata_path = activations_dir / f"{base_name}_llm.json"

    return artifact_path, metadata_path


def get_embedding_artifact_path(cfg: Config) -> tuple[Path, Path]:
    """
    Get paths for embedding artifact and its metadata.

    Returns:
        (artifact_path, metadata_path) tuple
    """
    config_hash = get_config_hash(cfg, "embeddings")
    activations_dir = Path(cfg.env.activations_dir)

    base_name = f"{cfg.data.name}_{cfg.llm.name}_embedding_{config_hash}"
    artifact_path = activations_dir / f"{base_name}.safetensors"
    metadata_path = activations_dir / f"{base_name}.json"

    return artifact_path, metadata_path


def get_sae_activation_artifact_paths(cfg: Config) -> Dict[str, tuple[Path, Path]]:
    """
    Get paths for SAE activation artifacts and their metadata.

    Returns:
        Dictionary mapping artifact names to (artifact_path, metadata_path) tuples
        For temporal SAE: {"recons": (...), "pred": (...), "novel": (...)}
        For standard SAE: {"recons": (...), "codes": (...)}
    """
    if cfg.sae is None:
        raise ValueError("No SAE configured")

    config_hash = get_config_hash(cfg, "sae_activations")
    activations_dir = Path(cfg.env.activations_dir)

    base_name = f"{cfg.data.name}_{cfg.llm.name}_layer{cfg.llm.layer_idx}_{cfg.sae.arch}_{config_hash}"

    paths = {}
    if cfg.sae.arch == "temporal":
        for suffix in ["recons", "pred", "novel"]:
            artifact_path = activations_dir / f"{base_name}_{suffix}.safetensors"
            metadata_path = activations_dir / f"{base_name}_{suffix}.json"
            paths[suffix] = (artifact_path, metadata_path)
    else:
        for suffix in ["recons", "codes"]:
            artifact_path = activations_dir / f"{base_name}_{suffix}.safetensors"
            metadata_path = activations_dir / f"{base_name}_{suffix}.json"
            paths[suffix] = (artifact_path, metadata_path)

    return paths
