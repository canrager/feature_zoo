import pytest
import tempfile
import os
from pathlib import Path
import torch as th
from unittest.mock import Mock, patch, MagicMock

from src.config import load_config
from src.tokenization import save_tokenized, get_tokenized


def test_save_tokenized(tmp_path):
    """Test saving tokenized data to safetensors."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))

    # Override texts_dir to use temp directory
    cfg.env.texts_dir = str(tmp_path)

    # Create a dummy CSV file
    csv_path = tmp_path / f"{cfg.data.name}.csv"
    csv_path.write_text("Monday\nTuesday\nWednesday")

    # Mock tokenizer (it's callable)
    mock_tokenizer = Mock()
    mock_tokenizer.return_value = {
        "input_ids": th.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "attention_mask": th.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }
    # Make it callable
    mock_tokenizer.__call__ = Mock(return_value=mock_tokenizer.return_value)

    # Call save_tokenized
    result = save_tokenized(cfg, mock_tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 3

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()


def test_get_tokenized_loads_existing(tmp_path):
    """Test get_tokenized loads existing safetensors file."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))

    # Override texts_dir to use temp directory
    cfg.env.texts_dir = str(tmp_path)

    # Create existing safetensors file
    from safetensors.torch import save_file

    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    save_file(
        {"input_ids": th.tensor([[1, 2, 3]]), "attention_mask": th.tensor([[1, 1, 1]])},
        str(output_path),
    )

    # Call get_tokenized without tokenizer (should load existing)
    result = get_tokenized(cfg, tokenizer=None)

    # Verify loaded data
    assert "input_ids" in result
    assert "attention_mask" in result


def test_get_tokenized_saves_when_missing(tmp_path):
    """Test get_tokenized saves when file doesn't exist."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))

    # Override texts_dir to use temp directory
    cfg.env.texts_dir = str(tmp_path)

    # Create a dummy CSV file
    csv_path = tmp_path / f"{cfg.data.name}.csv"
    csv_path.write_text("Monday\nTuesday")

    # Mock tokenizer (it's callable)
    mock_tokenizer = Mock(
        return_value={
            "input_ids": th.tensor([[1, 2], [3, 4]]),
            "attention_mask": th.tensor([[1, 1], [1, 1]]),
        }
    )

    # Call get_tokenized with tokenizer (should save)
    result = get_tokenized(cfg, tokenizer=mock_tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()
