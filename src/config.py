from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, fields, is_dataclass
import torch as th
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

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
    sae_dir: str
    debug: bool

    def __post_init__(self):
        self.dtype = DTYPE_STR_TO_TH[self.dtype]
        self.hf_cache_dir = get_root_parent_subdir(self.hf_cache_dir)

@dataclass
class DataConfig:
    name: str
    fixed_context_length: int | None


@dataclass
class LLMConfig:
    name: str
    hf_name: str
    layer_idx: int
    batch_size: int
    quantization_bits: int | None


@dataclass
class SAEConfig:
    llm_name: str
    llm_layer_idx: int
    arch: str
    batch_size: int
    act_scaling_factor: float


@dataclass
class FilterConfig:
    corpus: str
    regex_file: str
    num_occurences: int
    min_char_count: int | None


@dataclass
class ExperimentConfig:
    sequence_aggregation_method: str
    num_pca_components: int


@dataclass
class Config:
    env: EnvironmentConfig
    data: DataConfig
    llm: LLMConfig
    sae: SAEConfig | None
    filter: FilterConfig
    exp: ExperimentConfig

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


def load_config(
    config_name: str = "config", overrides: list[str] | None = None
) -> Config:
    """Load config using Hydra's compose API with defaults composition.

    Args:
        config_name: Name of the config file (without .yaml extension)
        overrides: Optional list of Hydra overrides, e.g. ["llm=olmo3-7b-base", "env.device=cpu"]

    Returns:
        Config dataclass instance with all defaults composed and post-processing applied.
    """
    # Clear any previous Hydra state (important for notebooks)
    GlobalHydra.instance().clear()

    config_dir = str(Path(__file__).parent.parent / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
        return OmegaConf.to_object(merged)
