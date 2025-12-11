import pytest
import tempfile
from pathlib import Path
import torch as th
from unittest.mock import Mock, MagicMock, patch

from src.config import (
    load_config,
    Config,
    EnvironmentConfig,
    LLMConfig,
    DataConfig,
    FilterConfig,
)
from src.activation_cache import (
    get_submodule,
    batch_activation_cache,
    save_cached_activations,
    aggregate_activations,
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
    cfg.env.activations_dir = str(tmp_path)
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


def test_aggregate_activations_max():
    """Test aggregate_activations with max aggregation method."""
    # Create mock config
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            debug=False,
        ),
        llm=LLMConfig(
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            sequence_aggregation_method="max",
        ),
        data=DataConfig(name="test"),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
        ),
    )

    # Create mock data: batch=2, sequence=4, dimension=3
    act_BTD = th.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [2.0, 1.0, 4.0],
                [3.0, 2.0, 1.0],
            ],  # batch 0
            [
                [7.0, 8.0, 9.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
                [1.0, 1.0, 1.0],
            ],  # batch 1
        ]
    )

    # Mask: batch=2, sequence=4 (True means valid token)
    mask_BT = th.tensor(
        [
            [True, True, True, False],  # batch 0: first 3 tokens valid
            [True, True, True, True],  # batch 1: all 4 tokens valid
        ]
    )

    # Call aggregate_activations
    result = aggregate_activations(cfg, act_BTD, mask_BT)

    # The function now returns per-batch results: max over sequence dimension for each batch
    # Shape should be (B, D) = (2, 3)
    # Batch 0: max over valid tokens [0:3] -> max([1,2,3], [4,5,6], [2,1,4]) = [4,5,6]
    # Batch 1: max over all tokens [0:4] -> max([7,8,9], [5,6,7], [8,9,10], [1,1,1]) = [8,9,10]
    expected = th.tensor(
        [
            [4.0, 5.0, 6.0],  # batch 0: max over sequence
            [8.0, 9.0, 10.0],  # batch 1: max over sequence
        ]
    )

    assert result.shape == (2, 3)  # (batch, dimension)
    assert th.allclose(result, expected)


def test_aggregate_activations_final():
    """Test aggregate_activations with final aggregation method."""
    # Create mock config
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            debug=False,
        ),
        llm=LLMConfig(
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            sequence_aggregation_method="final",
        ),
        data=DataConfig(name="test"),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
        ),
    )

    # Create mock data: batch=2, sequence=4, dimension=3
    act_BTD = th.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [2.0, 1.0, 4.0],
                [3.0, 2.0, 1.0],
            ],  # batch 0
            [
                [7.0, 8.0, 9.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
                [1.0, 1.0, 1.0],
            ],  # batch 1
        ]
    )

    # Mask: batch=2, sequence=4
    mask_BT = th.tensor(
        [
            [
                True,
                True,
                True,
                False,
            ],  # batch 0: first 3 tokens valid, last valid at index 2
            [
                True,
                True,
                True,
                True,
            ],  # batch 1: all 4 tokens valid, last valid at index 3
        ]
    )

    # Call aggregate_activations
    result = aggregate_activations(cfg, act_BTD, mask_BT)

    # The function returns the last valid token for each batch
    # Shape should be (B, D) = (2, 3)
    # Batch 0: last valid token at index 2 -> [2.0, 1.0, 4.0]
    # Batch 1: last valid token at index 3 -> [1.0, 1.0, 1.0]
    expected = th.tensor(
        [
            [2.0, 1.0, 4.0],  # batch 0: last valid token
            [1.0, 1.0, 1.0],  # batch 1: last valid token
        ]
    )

    assert result.shape == (2, 3)  # (batch, dimension)
    assert th.allclose(result, expected)


def test_aggregate_activations_sum():
    """Test aggregate_activations with sum aggregation method."""
    # Create mock config
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            debug=False,
        ),
        llm=LLMConfig(
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            sequence_aggregation_method="sum",
        ),
        data=DataConfig(name="test"),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
        ),
    )

    # Create mock data: batch=2, sequence=4, dimension=3
    act_BTD = th.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [2.0, 1.0, 4.0],
                [3.0, 2.0, 1.0],
            ],  # batch 0
            [
                [7.0, 8.0, 9.0],
                [5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0],
                [1.0, 1.0, 1.0],
            ],  # batch 1
        ]
    )

    # Mask: batch=2, sequence=4
    mask_BT = th.tensor(
        [
            [True, True, True, False],  # batch 0: first 3 tokens valid
            [True, True, True, True],  # batch 1: all 4 tokens valid
        ]
    )

    # Call aggregate_activations
    result = aggregate_activations(cfg, act_BTD, mask_BT)

    # The function now returns per-batch results: sum over sequence dimension for each batch
    # Shape should be (B, D) = (2, 3)
    # Batch 0: sum over valid tokens [0:3] -> [1+4+2, 2+5+1, 3+6+4] = [7, 8, 13]
    # Batch 1: sum over all tokens [0:4] -> [7+5+8+1, 8+6+9+1, 9+7+10+1] = [21, 24, 27]
    expected = th.tensor(
        [
            [7.0, 8.0, 13.0],  # batch 0: sum over sequence
            [21.0, 24.0, 27.0],  # batch 1: sum over sequence
        ]
    )

    assert result.shape == (2, 3)  # (batch, dimension)
    assert th.allclose(result, expected)


def test_aggregate_activations_unknown_method():
    """Test aggregate_activations raises error for unknown aggregation method."""
    # Create mock config
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            debug=False,
        ),
        llm=LLMConfig(
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            sequence_aggregation_method="unknown_method",
        ),
        data=DataConfig(name="test"),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
        ),
    )

    # Create mock data
    act_BTD = th.randn(2, 4, 3)
    mask_BT = th.ones(2, 4, dtype=th.bool)

    # Call aggregate_activations and expect ValueError
    with pytest.raises(ValueError, match="Unknown llm_sequence_aggregation_method"):
        aggregate_activations(cfg, act_BTD, mask_BT)
