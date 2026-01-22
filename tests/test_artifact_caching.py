"""
Tests for the content-addressable artifact caching system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.config import Config, DataConfig, LLMConfig, SAEConfig, EnvironmentConfig, ExperimentConfig
from src.artifact_utils import (
    get_config_hash,
    get_artifact_metadata,
    save_artifact_metadata,
    load_artifact_metadata,
    validate_artifact_metadata,
    get_token_artifact_path,
    get_llm_activation_artifact_path,
    get_sae_activation_artifact_paths,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration for testing."""
    return Config(
        env=EnvironmentConfig(
            dtype="float32",
            device="cpu",
            hf_cache_dir=None,
            texts_dir="data/texts",
            tokens_dir=str(temp_dir / "tokens"),
            activations_dir=str(temp_dir / "activations"),
            sae_dir="data/saes",
            debug=False,
        ),
        data=DataConfig(
            name="test_data",
            num_elements=7,
            elements_filename="test_elements.txt",
            trajectories_filename="test_trajectories.json",
            fixed_context_length=128,
        ),
        llm=LLMConfig(
            name="test_llm",
            hf_name="test/model",
            layer_idx=6,
            batch_size=32,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test_llm",
            llm_layer_idx=6,
            arch="standard",
            batch_size=32,
            act_scaling_factor=1.0,
        ),
        filter=None,
        exp=ExperimentConfig(
            sequence_aggregation_method="final",
            num_pca_components=10,
        ),
    )


def test_config_hash_deterministic(sample_config):
    """Test that config hashes are deterministic."""
    hash1 = get_config_hash(sample_config, "tokens")
    hash2 = get_config_hash(sample_config, "tokens")
    assert hash1 == hash2, "Hashes should be deterministic"


def test_config_hash_changes_with_params(sample_config):
    """Test that config hash changes when relevant parameters change."""
    hash1 = get_config_hash(sample_config, "tokens")

    # Change a relevant parameter
    sample_config.data.fixed_context_length = 256
    hash2 = get_config_hash(sample_config, "tokens")

    assert hash1 != hash2, "Hash should change when config changes"


def test_config_hash_same_for_irrelevant_changes(sample_config):
    """Test that config hash doesn't change for irrelevant parameters."""
    hash1 = get_config_hash(sample_config, "tokens")

    # Change an irrelevant parameter (batch_size doesn't affect tokenization)
    sample_config.llm.batch_size = 64
    hash2 = get_config_hash(sample_config, "tokens")

    # Note: batch_size is NOT included in hash, but llm.hf_name and layer_idx are
    # So this test checks that we're using the right subset of params
    assert hash1 == hash2, "Hash should not change for irrelevant config changes"


def test_sae_hash_includes_sae_params(sample_config):
    """Test that SAE hash includes SAE-specific parameters."""
    hash1 = get_config_hash(sample_config, "sae_activations")

    # Change SAE parameter
    sample_config.sae.act_scaling_factor = 2.0
    hash2 = get_config_hash(sample_config, "sae_activations")

    assert hash1 != hash2, "SAE hash should change when SAE params change"


def test_metadata_creation(sample_config):
    """Test that metadata is created with correct structure."""
    metadata = get_artifact_metadata(sample_config, "tokens")

    assert "artifact_type" in metadata
    assert "created_at" in metadata
    assert "config_hash" in metadata
    assert "config_snapshot" in metadata
    assert metadata["artifact_type"] == "tokens"


def test_metadata_save_and_load(sample_config, temp_dir):
    """Test saving and loading metadata."""
    metadata = get_artifact_metadata(sample_config, "tokens")
    metadata_path = temp_dir / "test_metadata.json"

    save_artifact_metadata(metadata, metadata_path)
    assert metadata_path.exists(), "Metadata file should be created"

    loaded_metadata = load_artifact_metadata(metadata_path)
    assert loaded_metadata is not None
    assert loaded_metadata["config_hash"] == metadata["config_hash"]


def test_metadata_validation_success(sample_config, temp_dir):
    """Test that metadata validation succeeds for matching config."""
    metadata = get_artifact_metadata(sample_config, "tokens")
    metadata_path = temp_dir / "test_metadata.json"
    save_artifact_metadata(metadata, metadata_path)

    # Validation should succeed
    is_valid, error = validate_artifact_metadata(sample_config, "tokens", metadata_path, strict=False)
    assert is_valid, f"Validation should succeed, got error: {error}"


def test_metadata_validation_failure(sample_config, temp_dir):
    """Test that metadata validation fails for mismatched config."""
    metadata = get_artifact_metadata(sample_config, "tokens")
    metadata_path = temp_dir / "test_metadata.json"
    save_artifact_metadata(metadata, metadata_path)

    # Change config
    sample_config.data.fixed_context_length = 256

    # Validation should fail
    is_valid, error = validate_artifact_metadata(sample_config, "tokens", metadata_path, strict=False)
    assert not is_valid, "Validation should fail for mismatched config"
    assert "hash mismatch" in error.lower()


def test_metadata_validation_strict_mode(sample_config, temp_dir):
    """Test that strict mode raises exceptions."""
    metadata = get_artifact_metadata(sample_config, "tokens")
    metadata_path = temp_dir / "test_metadata.json"
    save_artifact_metadata(metadata, metadata_path)

    # Change config
    sample_config.data.fixed_context_length = 256

    # Strict mode should raise ValueError
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact_metadata(sample_config, "tokens", metadata_path, strict=True)


def test_token_artifact_paths(sample_config):
    """Test token artifact path generation."""
    artifact_path, metadata_path = get_token_artifact_path(sample_config)

    assert artifact_path.suffix == ".safetensors"
    assert metadata_path.suffix == ".json"
    assert artifact_path.stem == metadata_path.stem
    assert "test_data" in str(artifact_path)


def test_llm_activation_artifact_paths(sample_config):
    """Test LLM activation artifact path generation."""
    artifact_path, metadata_path = get_llm_activation_artifact_path(sample_config)

    assert artifact_path.suffix == ".safetensors"
    assert metadata_path.suffix == ".json"
    assert artifact_path.stem == metadata_path.stem
    assert "test_data" in str(artifact_path)
    assert "test_llm" in str(artifact_path)
    assert "layer6" in str(artifact_path)
    assert "_llm" in str(artifact_path)


def test_sae_activation_artifact_paths_standard(sample_config):
    """Test SAE activation artifact path generation for standard SAE."""
    sample_config.sae.arch = "standard"
    paths = get_sae_activation_artifact_paths(sample_config)

    assert "recons" in paths
    assert "codes" in paths
    assert len(paths) == 2

    for name, (artifact_path, metadata_path) in paths.items():
        assert artifact_path.suffix == ".safetensors"
        assert metadata_path.suffix == ".json"
        assert name in str(artifact_path)


def test_sae_activation_artifact_paths_temporal(sample_config):
    """Test SAE activation artifact path generation for temporal SAE."""
    sample_config.sae.arch = "temporal"
    paths = get_sae_activation_artifact_paths(sample_config)

    assert "recons" in paths
    assert "pred" in paths
    assert "novel" in paths
    assert len(paths) == 3

    for name, (artifact_path, metadata_path) in paths.items():
        assert artifact_path.suffix == ".safetensors"
        assert metadata_path.suffix == ".json"
        assert name in str(artifact_path)


def test_different_data_different_hash(sample_config):
    """Test that different data configs produce different hashes."""
    hash1 = get_config_hash(sample_config, "tokens")

    sample_config.data.name = "different_data"
    hash2 = get_config_hash(sample_config, "tokens")

    assert hash1 != hash2


def test_different_llm_different_hash(sample_config):
    """Test that different LLM configs produce different hashes."""
    hash1 = get_config_hash(sample_config, "llm_activations")

    sample_config.llm.hf_name = "different/model"
    hash2 = get_config_hash(sample_config, "llm_activations")

    assert hash1 != hash2


def test_different_layer_different_hash(sample_config):
    """Test that different layer indices produce different hashes."""
    hash1 = get_config_hash(sample_config, "llm_activations")

    sample_config.llm.layer_idx = 12
    hash2 = get_config_hash(sample_config, "llm_activations")

    assert hash1 != hash2
