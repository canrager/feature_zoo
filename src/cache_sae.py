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
    fname = f"{cfg.env.activations_dir}/{cfg.data.name}_{cfg.llm.name}_layer{cfg.llm.layer_idx}_{cfg.sae.arch}"

    if cfg.sae.arch == "temporal":
        recons, pred_codes, novel_codes = batch_sae_temporal_cache(cfg, sae, act_BTD)
        save_file({"activations": recons}, f"{fname}_recons.safetensors")
        save_file({"activations": pred_codes}, f"{fname}_pred.safetensors")
        save_file({"activations": novel_codes}, f"{fname}_novel.safetensors")
        return recons, pred_codes, novel_codes
    else:
        recons, codes = batch_sae_standard_cache(cfg, sae, act_BTD)
        save_file({"activations": recons}, f"{fname}_recons.safetensors")
        save_file({"activations": codes}, f"{fname}_codes.safetensors")
        return recons, codes