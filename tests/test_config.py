import pytest
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf
import torch as th

from src.config import load_config, Config, EnvironmentConfig, LLMConfig


def test_load_config_valid():
    """Test loading a valid config file."""
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    result = load_config(str(config_path))

    # Verify it returns a MainConfig instance
    assert isinstance(result, Config)

    # Verify nested configs are properly loaded
    assert isinstance(result.env, EnvironmentConfig)
    assert isinstance(result.llm, LLMConfig)

    # Verify values
    assert result.env.dtype == th.bfloat16
    assert result.env.device == "cuda"
    assert result.llm.hf_name == "allenai/Olmo-3-1025-7B"
