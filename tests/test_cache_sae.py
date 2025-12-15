import pytest
import tempfile
from pathlib import Path
import torch as th
from unittest.mock import Mock, MagicMock, patch

from src.config import (
    Config,
    EnvironmentConfig,
    LLMConfig,
    DataConfig,
    FilterConfig,
    ExperimentConfig,
    SAEConfig,
)
from src.cache_sae import (
    batch_sae_standard_cache,
    batch_sae_temporal_cache,
    batch_sae_cache,
    save_sae_cache,
)


def test_batch_sae_standard_cache():
    """Test batch_sae_standard_cache with standard SAE."""
    # Create mock config
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="standard",
            batch_size=2,
            act_scaling_factor=2.0,
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    # Create mock activations: batch=4, sequence=3, dimension=5
    act_BTD = th.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0, 15.0]],
            [[16.0, 17.0, 18.0, 19.0, 20.0], [21.0, 22.0, 23.0, 24.0, 25.0], [26.0, 27.0, 28.0, 29.0, 30.0]],
            [[31.0, 32.0, 33.0, 34.0, 35.0], [36.0, 37.0, 38.0, 39.0, 40.0], [41.0, 42.0, 43.0, 44.0, 45.0]],
            [[46.0, 47.0, 48.0, 49.0, 50.0], [51.0, 52.0, 53.0, 54.0, 55.0], [56.0, 57.0, 58.0, 59.0, 60.0]],
        ]
    )

    # Mock SAE that returns (reconstruction, hidden) tuple
    mock_sae = Mock()
    # Mock return values for each batch call
    # Each call should return (reconstruction, hidden) where hidden is what we want
    # Shape should be (batch_size, sequence, dimension) = (2, 3, 5) for each call
    mock_sae.side_effect = [
        (th.randn(2, 3, 5), th.tensor([[[10.0, 20.0, 30.0, 40.0, 50.0], [60.0, 70.0, 80.0, 90.0, 100.0], [110.0, 120.0, 130.0, 140.0, 150.0]], [[160.0, 170.0, 180.0, 190.0, 200.0], [210.0, 220.0, 230.0, 240.0, 250.0], [260.0, 270.0, 280.0, 290.0, 300.0]]])),
        (th.randn(2, 3, 5), th.tensor([[[310.0, 320.0, 330.0, 340.0, 350.0], [360.0, 370.0, 380.0, 390.0, 400.0], [410.0, 420.0, 430.0, 440.0, 450.0]], [[460.0, 470.0, 480.0, 490.0, 500.0], [510.0, 520.0, 530.0, 540.0, 550.0], [560.0, 570.0, 580.0, 590.0, 600.0]]])),
    ]

    # Call batch_sae_standard_cache
    result = batch_sae_standard_cache(cfg, mock_sae, act_BTD)

    # Verify SAE was called with scaled activations and return_hidden=True
    assert mock_sae.call_count == 2  # 4 batches / 2 batch_size = 2 calls
    for call in mock_sae.call_args_list:
        args, kwargs = call
        assert kwargs.get("return_hidden") == True
        # Verify scaling factor was applied (input should be scaled)
        assert args[0].shape[0] <= cfg.sae.batch_size

    # Verify result shape: should be (B, T, D) = (4, 3, 5)
    assert result.shape == (4, 3, 5)


def test_batch_sae_standard_cache_single_batch():
    """Test batch_sae_standard_cache with batch_size larger than input."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="standard",
            batch_size=10,  # Larger than input batch size
            act_scaling_factor=1.5,
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    # Create small batch: batch=2, sequence=3, dimension=4
    act_BTD = th.randn(2, 3, 4)

    mock_sae = Mock()
    mock_hidden = th.randn(2, 3, 4)
    mock_sae.return_value = (th.randn(2, 3, 4), mock_hidden)

    result = batch_sae_standard_cache(cfg, mock_sae, act_BTD)

    # Should be called once since batch_size > input size
    assert mock_sae.call_count == 1
    assert result.shape == (2, 3, 4)
    assert th.allclose(result, mock_hidden)


def test_batch_sae_temporal_cache():
    """Test batch_sae_temporal_cache with temporal SAE."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="temporal",
            batch_size=2,
            act_scaling_factor=1.0,
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    # Create mock activations: batch=4, sequence=3, dimension=5
    act_BTD = th.randn(4, 3, 5)

    # Mock SAE that returns dict with pred_codes and novel_codes
    mock_sae = Mock()
    mock_pred = th.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])
    mock_novel = th.tensor([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], [[70.0, 80.0], [90.0, 100.0], [110.0, 120.0]]])
    mock_sae.side_effect = [
        {"pred_codes": mock_pred, "novel_codes": mock_novel},
        {"pred_codes": mock_pred + 100, "novel_codes": mock_novel + 100},
    ]

    # Call batch_sae_temporal_cache
    pred_result, novel_result = batch_sae_temporal_cache(cfg, mock_sae, act_BTD)

    # Verify SAE was called without return_hidden
    assert mock_sae.call_count == 2
    for call in mock_sae.call_args_list:
        args, kwargs = call
        # Should not have return_hidden keyword
        assert "return_hidden" not in kwargs
        assert args[0].shape[0] <= cfg.sae.batch_size

    # Verify result shapes
    assert pred_result.shape == (4, 3, 2)
    assert novel_result.shape == (4, 3, 2)


def test_batch_sae_cache_standard():
    """Test batch_sae_cache dispatches to standard cache."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="standard",
            batch_size=2,
            act_scaling_factor=1.0,
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    act_BTD = th.randn(2, 3, 4)
    mock_sae = Mock()
    mock_hidden = th.randn(2, 3, 4)
    mock_sae.return_value = (th.randn(2, 3, 4), mock_hidden)

    with patch("src.cache_sae.batch_sae_standard_cache", return_value=mock_hidden) as mock_standard:
        result = batch_sae_cache(cfg, mock_sae, act_BTD)
        mock_standard.assert_called_once_with(cfg, mock_sae, act_BTD)
        assert th.allclose(result, mock_hidden)


def test_batch_sae_cache_temporal():
    """Test batch_sae_cache dispatches to temporal cache."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="temporal",
            batch_size=2,
            act_scaling_factor=1.0,
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    act_BTD = th.randn(2, 3, 4)
    mock_sae = Mock()
    mock_pred = th.randn(2, 3, 4)
    mock_novel = th.randn(2, 3, 4)

    with patch("src.cache_sae.batch_sae_temporal_cache", return_value=(mock_pred, mock_novel)) as mock_temporal:
        result = batch_sae_cache(cfg, mock_sae, act_BTD)
        mock_temporal.assert_called_once_with(cfg, mock_sae, act_BTD)
        assert result == (mock_pred, mock_novel)


def test_save_sae_cache_standard(tmp_path):
    """Test save_sae_cache with standard SAE saves file correctly."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir=str(tmp_path),
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="standard",
            batch_size=2,
            act_scaling_factor=1.0,
        ),
        data=DataConfig(name="test-data", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    act_BTD = th.randn(2, 3, 4)
    mock_sae = Mock()
    mock_codes = th.randn(2, 3, 4)
    mock_sae.return_value = (th.randn(2, 3, 4), mock_codes)

    # Call save_sae_cache
    result = save_sae_cache(cfg, mock_sae, act_BTD)

    # Verify file was created
    expected_fname = f"{cfg.data.name}_{cfg.llm.name}_layer{cfg.llm.layer_idx}_{cfg.sae.arch}.safetensors"
    output_path = tmp_path / expected_fname
    assert output_path.exists()

    # Verify result matches expected codes
    assert th.allclose(result, mock_codes)

    # Verify file can be loaded and contains correct data
    from safetensors.torch import load_file
    loaded = load_file(str(output_path))
    assert "activations" in loaded
    assert th.allclose(loaded["activations"], mock_codes)


def test_save_sae_cache_temporal(tmp_path):
    """Test save_sae_cache with temporal SAE saves both files correctly."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir=str(tmp_path),
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="temporal",
            batch_size=2,
            act_scaling_factor=1.0,
        ),
        data=DataConfig(name="test-data", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    act_BTD = th.randn(2, 3, 4)
    mock_sae = Mock()
    mock_pred = th.randn(2, 3, 4)
    mock_novel = th.randn(2, 3, 4)
    mock_sae.side_effect = [{"pred_codes": mock_pred, "novel_codes": mock_novel}]

    # Call save_sae_cache
    pred_result, novel_result = save_sae_cache(cfg, mock_sae, act_BTD)

    # Verify both files were created
    base_fname = f"{cfg.data.name}_{cfg.llm.name}_layer{cfg.llm.layer_idx}_{cfg.sae.arch}"
    pred_path = tmp_path / f"{base_fname}_pred.safetensors"
    novel_path = tmp_path / f"{base_fname}_novel.safetensors"
    assert pred_path.exists()
    assert novel_path.exists()

    # Verify results match expected codes
    assert th.allclose(pred_result, mock_pred)
    assert th.allclose(novel_result, mock_novel)

    # Verify files can be loaded and contain correct data
    from safetensors.torch import load_file
    loaded_pred = load_file(str(pred_path))
    loaded_novel = load_file(str(novel_path))
    assert "activations" in loaded_pred
    assert "activations" in loaded_novel
    assert th.allclose(loaded_pred["activations"], mock_pred)
    assert th.allclose(loaded_novel["activations"], mock_novel)


def test_batch_sae_standard_cache_act_scaling():
    """Test that act_scaling_factor is applied correctly."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="standard",
            batch_size=10,
            act_scaling_factor=3.0,  # Scale by 3
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    # Create activations with known values
    act_BTD = th.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    original_act_BTD = act_BTD.clone()  # Save original before in-place modification

    mock_sae = Mock()
    mock_hidden = th.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    mock_sae.return_value = (th.randn(1, 2, 2), mock_hidden)

    # Capture the input to SAE to verify scaling
    captured_input = None

    def capture_input(*args, **kwargs):
        nonlocal captured_input
        captured_input = args[0].clone()
        return mock_sae.return_value

    mock_sae.side_effect = capture_input

    result = batch_sae_standard_cache(cfg, mock_sae, act_BTD)

    # Verify scaling was applied (input should be scaled by 3.0)
    # Note: act_BTD is modified in place, so we compare against original
    assert captured_input is not None
    expected_scaled = original_act_BTD * cfg.sae.act_scaling_factor
    assert th.allclose(captured_input, expected_scaled)


def test_batch_sae_temporal_cache_act_scaling():
    """Test that act_scaling_factor is applied correctly for temporal SAE."""
    cfg = Config(
        env=EnvironmentConfig(
            dtype="bfloat16",
            device="cpu",
            hf_cache_dir="models",
            texts_dir="data/texts",
            tokens_dir="data/tokens",
            activations_dir="data/activations",
            sae_dir="data/trained_saes",
            debug=False,
        ),
        llm=LLMConfig(
            name="test-model",
            hf_name="test-model",
            layer_idx=15,
            batch_size=100,
            quantization_bits=None,
        ),
        sae=SAEConfig(
            llm_name="test-model",
            llm_layer_idx=15,
            arch="temporal",
            batch_size=10,
            act_scaling_factor=2.5,  # Scale by 2.5
        ),
        data=DataConfig(name="test", fixed_context_length=None),
        filter=FilterConfig(
            corpus="test-corpus",
            regex_file="test-regex",
            num_occurences=1,
            min_char_count=None,
        ),
        exp=ExperimentConfig(sequence_aggregation_method="max"),
    )

    # Create activations with known values
    act_BTD = th.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    original_act_BTD = act_BTD.clone()  # Save original before in-place modification

    mock_sae = Mock()
    mock_pred = th.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    mock_novel = th.tensor([[[100.0, 200.0], [300.0, 400.0]]])

    # Capture the input to SAE to verify scaling
    captured_input = None

    def capture_input(*args, **kwargs):
        nonlocal captured_input
        captured_input = args[0].clone()
        return {"pred_codes": mock_pred, "novel_codes": mock_novel}

    mock_sae.side_effect = capture_input

    result = batch_sae_temporal_cache(cfg, mock_sae, act_BTD)

    # Verify scaling was applied (input should be scaled by 2.5)
    # Note: act_BTD is modified in place, so we compare against original
    assert captured_input is not None
    expected_scaled = original_act_BTD * cfg.sae.act_scaling_factor
    assert th.allclose(captured_input, expected_scaled)

