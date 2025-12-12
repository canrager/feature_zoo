"""
Utils for caching activations
"""

import torch as th
from typing import Dict
from pathlib import Path
from safetensors.torch import save_file, load_file
from tqdm import trange

from src.config import Config
from src.loading import load_llm, load_tokenizer, load_texts
from src.tokenization import save_tokenized
from transformers import AutoModelForCausalLM


def get_submodule(cfg: Config, llm: AutoModelForCausalLM) -> th.nn.Module:
    if any([name in cfg.llm.hf_name.lower() for name in ["olmo"]]):
        return llm.model.layers[cfg.llm.layer_idx]
    elif "gpt2" in cfg.llm.hf_name.lower():
        return llm.transformer.h[cfg.llm.layer_idx]
    else:
        raise ValueError("Unknown model.")


def batch_activation_cache(
    cfg: Config, encoded: Dict, llm: AutoModelForCausalLM
) -> th.Tensor:

    submodule = get_submodule(cfg, llm)
    acts_BTD = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        acts_BTD.append(output.detach())

    handle = submodule.register_forward_hook(hook)

    input_ids = encoded["input_ids"].to(cfg.env.device)
    attention_mask = encoded["attention_mask"].to(cfg.env.device)

    num_samples = input_ids.shape[0]
    for batch_start in trange(
        0, num_samples, cfg.llm.batch_size, desc="Caching Activations"
    ):
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
    activations_dict = {"activations": activations.to("cpu")}
    save_file(activations_dict, output_path)

    return activations


def aggregate_activations(
    cfg: Config, act_BTD: th.Tensor, mask_BT: th.Tensor
) -> th.Tensor:

    # Expand mask to (B, T, 1) to broadcast across D dimension
    mask_BTD = mask_BT.unsqueeze(-1).bool().to(cfg.env.device)

    match cfg.exp.sequence_aggregation_method:
        case "max":
            # Mask invalid positions with -inf so they don't affect max
            masked_act = act_BTD.masked_fill(~mask_BTD, -float("inf"))
            return masked_act.max(dim=1).values
        case "final":
            # Find the last valid position for each batch item
            # Get the index of the last True value in each row
            seq_lengths = mask_BT.sum(dim=1) - 1  # -1 because indices are 0-based
            batch_indices = th.arange(act_BTD.shape[0], device=act_BTD.device)
            return act_BTD[batch_indices, seq_lengths, :]
        case "sum":
            # Mask invalid positions with 0 so they don't affect sum
            masked_act = act_BTD.masked_fill(~mask_BTD, 0.0)
            return masked_act.sum(dim=1)
        case _:
            raise ValueError(
                "Unknown sequence_aggregation_method declared in the config.yaml!"
            )


def load_labeled_acts(cfg: Config, force_recompute=False):

    # Load texts
    labels, full_texts = load_texts(cfg)

    # Load or compute tokens
    token_path = Path(f"{cfg.env.texts_dir}/{cfg.data.name}.safetensors")

    if token_path.exists() and not force_recompute:
        encoded = load_file(str(token_path))
    else:
        if cfg.env.debug:
            print(f"Re-tokenizing...")
        tokenizer = load_tokenizer(cfg)
        encoded = save_tokenized(cfg, full_texts, tokenizer)

    # Get the actual string representations of the tokens, might be truncated versions of full_text
    tokens_list = [t[m] for t, m in zip(encoded["input_ids"], encoded["attention_mask"].bool())]
    tokenized_texts = [tokenizer.decode(t) for t in tokens_list] 

    # Load or compute activations
    act_path = Path(
        f"{cfg.env.activations_dir}/{cfg.data.name}_layer{cfg.llm.layer_idx}.safetensors"
    )

    if act_path.exists() and not force_recompute:
        activation_dict = load_file(str(act_path))
        act_BTD = activation_dict["activations"].to(cfg.env.device)

        # Validate batch size matches between activations and tokens
        act_batch_size = act_BTD.shape[0]
        token_batch_size = encoded["attention_mask"].shape[0]
        if act_batch_size != token_batch_size:
            raise ValueError(
                f"Batch size mismatch: activations have {act_batch_size} samples, "
                f"but tokens have {token_batch_size} samples. "
                f"This likely means the dataset changed after activations were cached. "
                f"Set force_recompute=True to recompute activations."
            )
    else:
        llm = load_llm(cfg)
        act_BTD = save_cached_activations(cfg, encoded, llm)

    act_BD = aggregate_activations(cfg, act_BTD, encoded["attention_mask"])

    return labels, tokenized_texts, act_BD


if __name__ == "__main__":
    from src.config import load_config
    from src.loading import load_tokenizer, load_llm
    from src.loading import load_texts

    cfg = load_config()
    labels, texts = load_texts(cfg)

    tokenizer = load_tokenizer(cfg)
    llm = load_llm(cfg)

    encoded = save_tokenized(cfg, texts, tokenizer)
    activations = save_cached_activations(cfg, encoded, llm)
    print(f"Loaded activations with shape: {activations.shape}")

    labels, texts, act_BD = load_labeled_acts(cfg)
    print(f"Aggregated activations with shape: {act_BD.shape}")
