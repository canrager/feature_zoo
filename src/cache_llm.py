"""
Utils for caching activations
"""

import torch as th
from typing import Dict
from pathlib import Path
from safetensors.torch import save_file, load_file
from tqdm import trange
import einops

from src.cache_sae import save_sae_cache
from src.config import Config
from src.loading import load_llm, load_tokenizer, load_texts, load_sae, load_trajectory_texts, load_elements
from src.tokenization import save_tokenized, get_tokenized_texts
from transformers import AutoModelForCausalLM


def get_submodule(cfg: Config, llm: AutoModelForCausalLM) -> th.nn.Module:
    if any([name in cfg.llm.hf_name.lower() for name in ["olmo", "llama"]]):
        return llm.model.layers[cfg.llm.layer_idx]
    elif "gpt2" in cfg.llm.hf_name.lower():
        return llm.transformer.h[cfg.llm.layer_idx]
    else:
        raise ValueError("Unknown model.")


def batch_llm_cache(cfg: Config, encoded: Dict, llm: AutoModelForCausalLM) -> th.Tensor:

    submodule = get_submodule(cfg, llm)
    acts_BTD = []

    def input_hook(module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        acts_BTD.append(input.detach())

    def output_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        acts_BTD.append(output.detach())

    handle = submodule.register_forward_hook(output_hook)

    input_ids = encoded["input_ids"].to(cfg.env.device)
    attention_mask = encoded["attention_mask"].to(cfg.env.device)

    num_samples = input_ids.shape[0]
    for batch_start in trange(0, num_samples, cfg.llm.batch_size, desc="LLM Cache"):
        batch_end = min(batch_start + cfg.llm.batch_size, num_samples)
        batch_input_ids = input_ids[batch_start:batch_end]
        batch_encoded = {"input_ids": batch_input_ids}
        if attention_mask is not None:
            batch_encoded["attention_mask"] = attention_mask[batch_start:batch_end]

        with th.inference_mode():
            llm(**batch_encoded)

    handle.remove()

    return th.cat(acts_BTD, dim=0)


def save_cached_activations(cfg: Config, encoded: Dict, llm: AutoModelForCausalLM) -> th.Tensor:
    "Compute activations and save to safetensors"
    from src.artifact_utils import (
        get_llm_activation_artifact_path,
        get_artifact_metadata,
        save_artifact_metadata
    )

    # Compute activations
    activations = batch_llm_cache(cfg, encoded, llm)

    # Get hash-based paths
    output_path, metadata_path = get_llm_activation_artifact_path(cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as safetensors
    activations_dict = {"activations": activations.to("cpu")}
    save_file(activations_dict, output_path)

    # Save metadata
    metadata = get_artifact_metadata(cfg, "llm_activations")
    metadata["activation_shape"] = list(activations.shape)
    save_artifact_metadata(metadata, metadata_path)

    if cfg.env.debug:
        print(f"Saved LLM activations to {output_path}")
        print(f"Saved metadata to {metadata_path}")

    return activations


def aggregate_activations(cfg: Config, act_BTD: th.Tensor, mask_BT: th.Tensor) -> th.Tensor:

    # Expand mask to (B, T, 1) to broadcast across D dimension
    mask_BTD = mask_BT.unsqueeze(-1).bool().to(cfg.env.device)

    match cfg.exp.sequence_aggregation_method:
        case "max":
            # mask BOS token
            mask_BT[:, 0] = 0
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
            # mask BOS token
            mask_BT[:, 0] = 0
            # Mask invalid positions with 0 so they don't affect sum
            masked_act = act_BTD.masked_fill(~mask_BTD, 0.0)
            return masked_act.sum(dim=1)
        case _:
            raise ValueError("Unknown sequence_aggregation_method declared in the config.yaml!")


def load_labeled_acts(cfg: Config, force_recompute=False, compute_if_missing=True):
    from src.artifact_utils import (
        get_token_artifact_path,
        validate_artifact_metadata
    )

    # Load texts
    labels, full_texts = load_texts(cfg)

    # Load or compute tokens
    if cfg.env.debug:
        print(f"Tokenize...")

    token_path, token_metadata_path = get_token_artifact_path(cfg)
    tokenizer = load_tokenizer(cfg)

    if token_path.exists() and not force_recompute:
        # Validate metadata before loading
        validate_artifact_metadata(cfg, "tokens", token_metadata_path, strict=True)
        encoded = load_file(str(token_path))
        if cfg.env.debug:
            print(f"Loaded tokens from {token_path}")
    elif compute_if_missing:
        if cfg.env.debug:
            print(f"Re-tokenizing...")
        encoded = save_tokenized(cfg, full_texts, tokenizer)
    else:
        raise FileNotFoundError(f"Token artifact not found: {token_path}")

    # Get the actual string representations of the tokens, might be truncated versions of full_text
    tokenized_texts = get_tokenized_texts(encoded, tokenizer)

    # Load or compute activations
    if cfg.env.debug:
        print(f"LLM activations...")

    from src.artifact_utils import (
        get_llm_activation_artifact_path,
    )
    act_path, act_metadata_path = get_llm_activation_artifact_path(cfg)

    if act_path.exists() and not force_recompute:
        # Validate metadata before loading
        validate_artifact_metadata(cfg, "llm_activations", act_metadata_path, strict=True)
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

        if cfg.env.debug:
            print(f"Loaded LLM activations from {act_path}")
    elif compute_if_missing:
        llm = load_llm(cfg)
        act_BTD = save_cached_activations(cfg, encoded, llm)

        del llm
        th.cuda.empty_cache()
    else:
        raise FileNotFoundError(f"LLM activation artifact not found: {act_path}")

    return_dict = {
        "labels": labels,
        "texts": tokenized_texts,
        "input_ids_BT": encoded["input_ids"],
        "mask_BT": encoded["attention_mask"],
        "llm_BTD": act_BTD,
        "llm_BD": aggregate_activations(cfg, act_BTD, encoded["attention_mask"]),
    }

    if cfg.sae is not None:
        if cfg.env.debug:
            print(f"SAE activations...")

        from src.artifact_utils import get_sae_activation_artifact_paths
        sae_paths = get_sae_activation_artifact_paths(cfg)

        if cfg.sae.arch == "temporal":
            sae_recons_path, sae_recons_metadata = sae_paths["recons"]
            sae_pred_path, sae_pred_metadata = sae_paths["pred"]
            sae_novel_path, sae_novel_metadata = sae_paths["novel"]

            if sae_pred_path.exists() and not force_recompute:
                # Validate all metadata
                validate_artifact_metadata(cfg, "sae_activations", sae_recons_metadata, strict=True)
                validate_artifact_metadata(cfg, "sae_activations", sae_pred_metadata, strict=True)
                validate_artifact_metadata(cfg, "sae_activations", sae_novel_metadata, strict=True)

                recons_BTD = load_file(str(sae_recons_path))["activations"].to(cfg.env.device)
                pred_codes_BTD = load_file(str(sae_pred_path))["activations"].to(cfg.env.device)
                novel_codes_BTD = load_file(str(sae_novel_path))["activations"].to(cfg.env.device)

                if cfg.env.debug:
                    print(f"Loaded SAE activations from {sae_recons_path.parent}")
            elif compute_if_missing:
                sae = load_sae(cfg)
                recons_BTD, pred_codes_BTD, novel_codes_BTD = save_sae_cache(cfg, sae, act_BTD)
                del sae
                th.cuda.empty_cache()
            else:
                raise FileNotFoundError(f"SAE activation artifacts not found: {sae_pred_path}")

            return_dict["recons_BTD"] = recons_BTD
            return_dict["recons_BD"] = aggregate_activations(cfg, pred_codes_BTD, encoded["attention_mask"])
            return_dict["pred_codes_BTD"] = pred_codes_BTD
            return_dict["pred_codes_BD"] = aggregate_activations(cfg, pred_codes_BTD, encoded["attention_mask"])
            return_dict["novel_codes_BTD"] = novel_codes_BTD
            return_dict["novel_codes_BD"] = aggregate_activations(cfg, novel_codes_BTD, encoded["attention_mask"])

        else:
            sae_recons_path, sae_recons_metadata = sae_paths["recons"]
            sae_codes_path, sae_codes_metadata = sae_paths["codes"]

            if sae_recons_path.exists() and not force_recompute:
                # Validate metadata
                validate_artifact_metadata(cfg, "sae_activations", sae_recons_metadata, strict=True)
                validate_artifact_metadata(cfg, "sae_activations", sae_codes_metadata, strict=True)

                recons_BTD = load_file(str(sae_recons_path))["activations"].to(cfg.env.device)
                codes_BTD = load_file(str(sae_codes_path))["activations"].to(cfg.env.device)

                if cfg.env.debug:
                    print(f"Loaded SAE activations from {sae_recons_path.parent}")
            elif compute_if_missing:
                sae = load_sae(cfg)
                recons_BTD, codes_BTD = save_sae_cache(cfg, sae, act_BTD)
                del sae
                th.cuda.empty_cache()
            else:
                raise FileNotFoundError(f"SAE activation artifacts not found: {sae_recons_path}")

            return_dict["recons_BTD"] = recons_BTD
            return_dict["recons_BD"] = aggregate_activations(cfg, recons_BTD, encoded["attention_mask"])
            return_dict["codes_BTD"] = codes_BTD
            return_dict["codes_BD"] = aggregate_activations(cfg, codes_BTD, encoded["attention_mask"])

    return return_dict


def load_short_trajectory_acts(
    cfg: Config, force_recompute: bool = False, compute_if_missing: bool = True
) -> Dict:
    from src.artifact_utils import (
        get_token_artifact_path,
        get_llm_activation_artifact_path,
        validate_artifact_metadata
    )

    # Load data
    labels, texts, indices = load_trajectory_texts(cfg)

    elements = load_elements(cfg)

    # Load or compute tokens
    if cfg.env.debug:
        print(f"Tokenize...")

    token_path, token_metadata_path = get_token_artifact_path(cfg)
    tokenizer = load_tokenizer(cfg)

    if token_path.exists() and not force_recompute:
        # Validate metadata before loading
        validate_artifact_metadata(cfg, "tokens", token_metadata_path, strict=True)
        encoded = load_file(str(token_path))
        if cfg.env.debug:
            print(f"Loaded tokens from {token_path}")
    elif compute_if_missing:
        if cfg.env.debug:
            print(f"Re-tokenizing...")
        encoded = save_tokenized(cfg, texts, tokenizer)
    else:
        raise FileNotFoundError(f"Token artifact not found: {token_path}")

    # Get the actual string representations of the tokens
    tokenized_texts = get_tokenized_texts(encoded, tokenizer)

    # Load or compute activations
    if cfg.env.debug:
        print(f"LLM activations...")

    act_path, act_metadata_path = get_llm_activation_artifact_path(cfg)

    if act_path.exists() and not force_recompute:
        # Validate metadata before loading
        validate_artifact_metadata(cfg, "llm_activations", act_metadata_path, strict=True)
        activation_dict = load_file(str(act_path))
        acts_BTD = activation_dict["activations"].to(cfg.env.device)

        # Validate batch size matches between activations and tokens
        act_batch_size = acts_BTD.shape[0]
        token_batch_size = encoded["attention_mask"].shape[0]
        if act_batch_size != token_batch_size:
            raise ValueError(
                f"Batch size mismatch: activations have {act_batch_size} samples, "
                f"but tokens have {token_batch_size} samples. "
                f"This likely means the dataset changed after activations were cached. "
                f"Set force_recompute=True to recompute activations."
            )

        if cfg.env.debug:
            print(f"Loaded LLM activations from {act_path}")
    elif compute_if_missing:
        llm = load_llm(cfg)
        acts_BTD = save_cached_activations(cfg, encoded, llm)

        del llm
        th.cuda.empty_cache()
    else:
        raise FileNotFoundError(f"LLM activation artifact not found: {act_path}")

    acts_aggregated_BD = aggregate_activations(cfg, acts_BTD, encoded["attention_mask"])
    
    acts_BCTD = einops.rearrange(acts_BTD, "(B C) T D -> B C T D", C=max(indices) + 1)
    acts_aggregated_BCD = einops.rearrange(acts_aggregated_BD, "(B C) D -> B C D", C=max(indices) + 1)

    return {
        "elements_C": elements,
        "labels_B": labels,
        "texts_B": tokenized_texts,
        "input_ids_BT": encoded["input_ids"],
        "mask_BT": encoded["attention_mask"],
        "llm_BCTD": acts_BCTD,
        "llm_BCD": acts_aggregated_BCD,
    }


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

    results_dict = load_labeled_acts(cfg)
    print(f"Aggregated activations with shape: {results_dict["llm_BD"].shape}")
