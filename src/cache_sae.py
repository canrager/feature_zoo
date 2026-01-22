from typing import Any
from safetensors.torch import save_file, load_file
from sae import SAEStandard, TemporalSAE
from tqdm import trange
from src.config import Config
import torch as th


def batch_sae_standard_cache(cfg: Config, sae: SAEStandard, act_BTD: th.Tensor) -> th.tensor:
    B, T, D = act_BTD.shape
    act_BTD *= cfg.sae.act_scaling_factor

    recons_act_BTD = []
    sae_act_BTD = []

    with th.inference_mode():
        for batch_start in trange(0, B, cfg.sae.batch_size, desc="SAE Cache"):
            batch_end = batch_start + cfg.sae.batch_size
            act_bTD = act_BTD[batch_start : batch_end]
            x_recons, sae_act_bTD = sae(act_bTD, return_hidden=True)
            recons_act_BTD.append(x_recons)
            sae_act_BTD.append(sae_act_bTD)

    recons_act_BTD = th.cat(recons_act_BTD, dim=0)
    sae_act_BTD = th.cat(sae_act_BTD, dim=0)
    return recons_act_BTD, sae_act_BTD


def batch_sae_temporal_cache(cfg: Config, sae: TemporalSAE, act_BTD: th.Tensor) -> th.tensor:
    B, T, D = act_BTD.shape
    act_BTD *= cfg.sae.act_scaling_factor

    recons_act_BTD = []
    pred_act_BTD = []
    novel_act_BTD = []

    with th.inference_mode():
        for batch_start in trange(0, B, cfg.sae.batch_size, desc="SAE Cache"):
            batch_end = batch_start + cfg.sae.batch_size
            act_bTD = act_BTD[batch_start : batch_end]
            x_recons, result_dict = sae(act_bTD)
            recons_act_BTD.append(x_recons)
            pred_act_BTD.append(result_dict["pred_codes"])
            novel_act_BTD.append(result_dict["novel_codes"])

    recons_act_BTD = th.cat(recons_act_BTD, dim=0)
    pred_act_BTD = th.cat(pred_act_BTD, dim=0)
    novel_act_BTD = th.cat(novel_act_BTD, dim=0)
    return recons_act_BTD, pred_act_BTD, novel_act_BTD


def batch_sae_cache(cfg: Config, sae: Any, act_BTD: th.Tensor) -> th.tensor:
    if cfg.sae.arch == "temporal":
        return batch_sae_temporal_cache(cfg, sae, act_BTD)
    else:
        return batch_sae_standard_cache(cfg, sae, act_BTD)


def save_sae_cache(cfg: Config, sae: Any, act_BTD: th.Tensor) -> None:
    from src.artifact_utils import (
        get_sae_activation_artifact_paths,
        get_artifact_metadata,
        save_artifact_metadata
    )

    # Get hash-based paths
    paths = get_sae_activation_artifact_paths(cfg)

    if cfg.sae.arch == "temporal":
        recons, pred_codes, novel_codes = batch_sae_temporal_cache(cfg, sae, act_BTD)

        # Save artifacts and metadata
        save_file({"activations": recons.to("cpu")}, paths["recons"][0])
        save_file({"activations": pred_codes.to("cpu")}, paths["pred"][0])
        save_file({"activations": novel_codes.to("cpu")}, paths["novel"][0])

        # Save metadata for each artifact
        for name in ["recons", "pred", "novel"]:
            metadata = get_artifact_metadata(cfg, "sae_activations")
            metadata["sae_output_type"] = name
            metadata["activation_shape"] = list(locals()[name if name == "recons" else f"{name}_codes"].shape)
            save_artifact_metadata(metadata, paths[name][1])

        if cfg.env.debug:
            print(f"Saved SAE activations to {paths['recons'][0].parent}")

        return recons, pred_codes, novel_codes
    else:
        recons, codes = batch_sae_standard_cache(cfg, sae, act_BTD)

        # Save artifacts and metadata
        save_file({"activations": recons.to("cpu")}, paths["recons"][0])
        save_file({"activations": codes.to("cpu")}, paths["codes"][0])

        # Save metadata for each artifact
        for name in ["recons", "codes"]:
            metadata = get_artifact_metadata(cfg, "sae_activations")
            metadata["sae_output_type"] = name
            metadata["activation_shape"] = list(locals()[name].shape)
            save_artifact_metadata(metadata, paths[name][1])

        if cfg.env.debug:
            print(f"Saved SAE activations to {paths['recons'][0].parent}")

        return recons, codes