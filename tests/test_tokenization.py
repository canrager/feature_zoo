import pytest
import torch as th
from unittest.mock import Mock

from src.config import load_config
from src.tokenization import save_tokenized


def test_save_tokenized_varying_ctx_length(tmp_path):
    """Test saving tokenized data to safetensors."""
    # Load test config from configs directory
    cfg = load_config("test")

    # Override for varying ctx length
    cfg.data.fixed_context_length = None
    # Override tokens_dir to use temp directory (where the function saves)
    cfg.env.tokens_dir = str(tmp_path)

    # Create a dummy CSV file
    csv_path = tmp_path / f"{cfg.data.name}.txt"
    csv_path.write_text("Monday\nTuesday\nWednesday")

    # Mock tokenizer (it's callable)
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": th.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": th.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }
    # Make it callable
    mock_tokenizer.__call__ = Mock(return_value=mock_tokenizer.return_value)

    # Prepare texts list
    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    # Call save_tokenized
    result = save_tokenized(cfg, texts, mock_tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 3

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()


def test_save_tokenized_fixed_ctx_length(tmp_path):
    """Test saving tokenized data to safetensors."""
    # Load test config from configs directory
    cfg = load_config("test")

    # Override tokens_dir to use temp directory (where the function saves)
    cfg.env.tokens_dir = str(tmp_path)

    # Create a dummy CSV file
    csv_path = tmp_path / f"{cfg.data.name}.txt"
    csv_path.write_text("Monday\nTuesday\nWednesday")

    # Mock tokenizer (it's callable)
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": th.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": th.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }
    # Make it callable
    mock_tokenizer.__call__ = Mock(return_value=mock_tokenizer.return_value)

    # Prepare texts list
    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    # Call save_tokenized
    result = save_tokenized(cfg, texts, mock_tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 3
    assert result["input_ids"].shape[1] == cfg.data.fixed_context_length

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()
