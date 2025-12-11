"""
Utils for caching activations
"""

import torch as th
from typing import Dict
from pathlib import Path
from safetensors.torch import save_file, load_file

from src.config import Config
from src.loading import load_llm
from src.tokenization import get_tokenized
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_submodule(cfg: Config, llm: AutoModelForCausalLM) -> th.nn.Module:
    if "olmo" in cfg.llm.hf_name.lower():
        return llm.model.layers[cfg.llm.layer_idx]


def batch_activation_cache(
    cfg: Config, encoded: Dict, llm: AutoModelForCausalLM
) -> th.Tensor:

    submodule = get_submodule(cfg, llm)
    acts_BTD = []

    def hook(module, input, output):
        acts_BTD.append(output.detach())

    handle = submodule.register_forward_hook(hook)

    input_ids = encoded["input_ids"].to(cfg.env.device)
    attention_mask = encoded["attention_mask"].to(cfg.env.device)

    num_samples = input_ids.shape[0]
    for batch_start in range(0, num_samples, cfg.llm.batch_size):
        batch_end = min(batch_start + cfg.llm.batch_size, num_samples)
        batch_input_ids = input_ids[batch_start:batch_end]
        batch_encoded = {"input_ids": batch_input_ids}
        if attention_mask is not None:
            batch_encoded["attention_mask"] = attention_mask[batch_start:batch_end]

        with th.inference_mode():
            llm(**batch_encoded)

    handle.remove()

    return th.cat(acts_BTD, dim=0)


def save_cached_activations(
    cfg: Config, encoded: Dict, llm: AutoModelForCausalLM
) -> th.Tensor:
    "Compute activations and save to safetensors"

    # Compute activations
    activations = batch_activation_cache(cfg, encoded, llm)

    # Save as safetensors
    activations_dir = Path(cfg.env.activations_dir)
    activations_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        activations_dir / f"{cfg.data.name}_layer{cfg.llm.layer_idx}.safetensors"
    )
    activations_dict = {
        "activations": activations.to("cpu")
    }
    save_file(activations_dict, output_path)

    return activations


def get_cached_activations(
    cfg: Config, encoded: Dict | None = None, llm: AutoModelForCausalLM | None = None
) -> th.Tensor:
    "Attempt to load cached activations, otherwise compute and save"

    output_path = Path(
        f"{cfg.env.activations_dir}/{cfg.data.name}_layer{cfg.llm.layer_idx}.safetensors"
    )

    if output_path.exists():
        # Load from safetensors
        activation_dict = load_file(str(output_path))
        return activation_dict["activations"].to(cfg.env.device)
    else:
        # Compute and save
        assert encoded is not None, "Need to pass encoded data."
        assert llm is not None, "Need to pass llm."
        return save_cached_activations(cfg, encoded, llm)


if __name__ == "__main__":
    from src.config import load_config
    from src.loading import load_tokenizer

    cfg = load_config()
    tokenizer = load_tokenizer(cfg)
    encoded = get_tokenized(cfg, tokenizer)

    llm = load_llm(cfg)

    activations = get_cached_activations(cfg, encoded, llm)
    print(f"Loaded activations with shape: {activations.shape}")
