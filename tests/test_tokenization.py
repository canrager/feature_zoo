import pytest
import torch as th

from src.config import load_config
from src.tokenization import save_tokenized
from src.loading import load_tokenizer


def test_save_tokenized_varying_ctx_length(tmp_path):
    """Test saving tokenized data to safetensors."""
    # Load test config from configs directory
    cfg = load_config("test")

    # Override for varying ctx length
    cfg.data.fixed_context_length = None
    # Override tokens_dir to use temp directory (where the function saves)
    cfg.env.tokens_dir = str(tmp_path)

    # Load actual tokenizer
    tokenizer = load_tokenizer(cfg)

    # Prepare texts list
    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    # Call save_tokenized
    result = save_tokenized(cfg, texts, tokenizer)

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
    cfg.sae = None

    # Override tokens_dir to use temp directory (where the function saves)
    cfg.env.tokens_dir = str(tmp_path)

    # Load actual tokenizer
    tokenizer = load_tokenizer(cfg)

    # Prepare texts list
    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    # Call save_tokenized
    result = save_tokenized(cfg, texts, tokenizer)

    # Verify result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert result["input_ids"].shape[0] == 3
    assert (
        result["input_ids"].shape[1] == cfg.data.fixed_context_length + 1
    )  # +1 for BOS token

    # Verify file was created
    output_path = tmp_path / f"{cfg.data.name}.safetensors"
    assert output_path.exists()


def test_bos_token_at_position_0_varying_ctx_length(tmp_path):
    """Test that BOS token is always at position 0 after tokenization with varying context length."""
    cfg = load_config("test", overrides=["sae=llama3.1-8b-base_temporal"])
    cfg.sae = None
    cfg.data.fixed_context_length = None
    cfg.env.tokens_dir = str(tmp_path)

    # Load actual tokenizer
    tokenizer = load_tokenizer(cfg)

    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    result = save_tokenized(cfg, texts, tokenizer)

    # Verify BOS token is at position 0 for all sequences
    assert result["input_ids"].shape[0] == 3
    for i in range(result["input_ids"].shape[0]):
        assert (
            result["input_ids"][i, 0].item() == tokenizer.bos_token_id
        ), f"Sequence {i} does not have BOS token at position 0"


def test_bos_token_at_position_0_fixed_ctx_length(tmp_path):
    """Test that BOS token is always at position 0 after tokenization with fixed context length."""
    cfg = load_config("test")
    cfg.sae = None
    cfg.env.tokens_dir = str(tmp_path)

    # Load actual tokenizer
    tokenizer = load_tokenizer(cfg)

    texts = [
        "How about Monday",
        "Yesterday was Tuesday",
        " I will go shopping on Wednesday",
    ]

    result = save_tokenized(cfg, texts, tokenizer)

    # Verify BOS token is at position 0 for all sequences
    assert result["input_ids"].shape[0] == 3
    assert (
        result["input_ids"].shape[1] == cfg.data.fixed_context_length + 1
    )  # +1 for BOS token
    for i in range(result["input_ids"].shape[0]):
        assert (
            result["input_ids"][i, 0].item() == tokenizer.bos_token_id
        ), f"Sequence {i} does not have BOS token at position 0"
