import pytest
import tempfile
from pathlib import Path
import torch as th
from unittest.mock import Mock, MagicMock, patch

from src.config import load_config
from src.activation_cache import (
    get_submodule,
    batch_activation_cache,
    save_cached_activations,
    get_cached_activations,
)


def test_get_submodule():
    """Test getting submodule from model."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))

    # Mock LLM model
    mock_layer = Mock()
    mock_llm = Mock()
    mock_llm.model.layers = [Mock() for _ in range(20)]
    mock_llm.model.layers[cfg.llm.layer_idx] = mock_layer

    # Call get_submodule
    result = get_submodule(cfg, mock_llm)

    # Verify correct layer is returned
    assert result == mock_layer


def test_batch_activation_cache(tmp_path):
    """Test batch activation caching."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))
    cfg.env.device = "cpu"  # Use CPU for testing

    # Mock submodule
    mock_submodule = Mock()
    mock_output = th.randn(2, 5, 10)  # (batch, seq_len, hidden_dim)
    mock_submodule.register_forward_hook = Mock()

    # Mock LLM
    mock_llm = Mock()
    mock_llm.model.layers = [Mock() for _ in range(20)]
    mock_llm.model.layers[cfg.llm.layer_idx] = mock_submodule

    # Track hook calls
    hook_outputs = []

    def mock_hook(module, input, output):
        hook_outputs.append(output.detach())

    mock_handle = Mock()
    mock_submodule.register_forward_hook.return_value = mock_handle

    # Mock the forward pass
    with patch("src.activation_cache.get_submodule", return_value=mock_submodule):
        # Create encoded data
        encoded = {
            "input_ids": th.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            "attention_mask": th.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        }

        # Mock the forward hook to capture outputs
        def register_hook_side_effect(hook_fn):
            # Simulate hook being called during forward pass
            hook_fn(mock_submodule, None, mock_output)
            return mock_handle

        mock_submodule.register_forward_hook.side_effect = register_hook_side_effect

        # Mock the forward pass
        mock_llm.return_value = None

        # Call batch_activation_cache
        result = batch_activation_cache(cfg, encoded, mock_llm)

        # Verify hook was registered and removed
        assert mock_submodule.register_forward_hook.called
        assert mock_handle.remove.called


def test_save_cached_activations(tmp_path):
    """Test saving cached activations."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))
    cfg.env.texts_dir = str(tmp_path)
    cfg.env.device = "cpu"

    # Mock activations
    mock_activations = th.randn(2, 5, 10)

    # Mock batch_activation_cache
    with patch(
        "src.activation_cache.batch_activation_cache", return_value=mock_activations
    ):
        # Mock LLM
        mock_llm = Mock()
        mock_llm.model.layers = [Mock() for _ in range(20)]

        encoded = {
            "input_ids": th.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": th.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        # Call save_cached_activations
        result = save_cached_activations(cfg, encoded, mock_llm)

        # Verify result
        assert th.allclose(result, mock_activations)

        # Verify file was created
        output_path = tmp_path / f"{cfg.data.name}_layer{cfg.llm.layer_idx}.safetensors"
        assert output_path.exists()


def test_get_cached_activations_loads_existing(tmp_path):
    """Test get_cached_activations loads existing file."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))
    cfg.env.texts_dir = str(tmp_path)
    cfg.env.device = "cpu"

    # Create existing safetensors file
    from safetensors.torch import save_file

    output_path = tmp_path / f"{cfg.data.name}_layer{cfg.llm.layer_idx}.safetensors"
    mock_activations = th.randn(2, 5, 10)
    save_file({"activations": mock_activations}, str(output_path))

    # Call get_cached_activations without encoded/llm (should load existing)
    result = get_cached_activations(cfg, encoded=None, llm=None)

    # Verify loaded data
    assert result.shape == mock_activations.shape


def test_get_cached_activations_computes_when_missing(tmp_path):
    """Test get_cached_activations computes when file doesn't exist."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))
    cfg.env.texts_dir = str(tmp_path)
    cfg.env.device = "cpu"

    # Mock activations
    mock_activations = th.randn(2, 5, 10)

    # Mock save_cached_activations
    with patch(
        "src.activation_cache.save_cached_activations", return_value=mock_activations
    ):
        encoded = {
            "input_ids": th.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": th.tensor([[1, 1, 1], [1, 1, 1]]),
        }
        mock_llm = Mock()

        # Call get_cached_activations with encoded/llm (should compute)
        result = get_cached_activations(cfg, encoded=encoded, llm=mock_llm)

        # Verify result
        assert th.allclose(result, mock_activations)
