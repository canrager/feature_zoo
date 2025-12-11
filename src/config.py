from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass
import torch as th

DTYPE_STR_TO_TH = {"bfloat16": th.bfloat16, "float32": th.float32}


def get_root_parent_subdir(path: str | None) -> Path | None:
    """Return a path to a subdirectory of the parent directory of the project root."""
    if path is None:
        return None

    root_dir = Path(__file__).parent.parent
    return root_dir.parent / path


@dataclass
class EnvironmentConfig:
    dtype: str
    device: str
    hf_cache_dir: str
    texts_dir: str
    tokens_dir: str 
    activations_dir: str

    def __post_init__(self):
        self.dtype = DTYPE_STR_TO_TH[self.dtype]
        self.hf_cache_dir = get_root_parent_subdir(self.hf_cache_dir)


@dataclass
class LLMConfig:
    hf_name: str
    layer_idx: int
    batch_size: int
    sequence_aggregation_method: str


@dataclass
class DataConfig:
    name: str


@dataclass
class Config:
    env: EnvironmentConfig
    llm: LLMConfig
    data: DataConfig


def load_config(path: str = "configs/config.yaml") -> Config:
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(Config)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # Returns actual dataclass instance
