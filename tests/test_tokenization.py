import pytest
import tempfile
import os
from pathlib import Path
import torch as th
from unittest.mock import Mock, patch, MagicMock

from src.config import load_config
from src.tokenization import save_tokenized


def test_save_tokenized(tmp_path):
    """Test saving tokenized data to safetensors."""
    # Load dummy config
    config_path = Path(__file__).parent / "dummy_config.yaml"
    cfg = load_config(str(config_path))

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
    texts = ["Monday", "Tuesday", "Wednesday"]

    # Call save_tokenized
    result = save_tokenized(cfg, texts, mock_tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 3

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()
