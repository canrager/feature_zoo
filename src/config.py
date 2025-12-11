from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, fields, is_dataclass
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
    debug: bool

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
class FilterConfig:
    corpus: str
    regex_file: str
    num_occurences: int


@dataclass
class Config:
    env: EnvironmentConfig
    llm: LLMConfig
    data: DataConfig
    filter: FilterConfig

    def __repr__(self):
        return pretty_print_dataclass(self)


def pretty_print_dataclass(obj, indent: int = 0) -> str:
    """Pretty print a dataclass with indentation for nested dataclasses."""
    if not is_dataclass(obj):
        return repr(obj)

    indent_str = "  " * indent
    lines = [f"{obj.__class__.__name__}("]

    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            nested_str = pretty_print_dataclass(value, indent + 1)
            lines.append(f"{indent_str}  {field.name}={nested_str}")
        else:
            lines.append(f"{indent_str}  {field.name}={repr(value)}")

    lines.append(f"{indent_str})")
    return "\n".join(lines)


def load_config(path: str = "config") -> Config:
    path = f"configs/{path}.yaml"
    raw = OmegaConf.load(path)
    schema = OmegaConf.structured(Config)
    merged = OmegaConf.merge(schema, raw)
    return OmegaConf.to_object(merged)  # Returns actual dataclass instance
