#!/usr/bin/env python3
"""
Convert GPT2 SAE from SAELens format to the standard format used by SAEStandard.

This script converts SAE artifacts from:
  - Source: /data/trained_saes/gpt2/layer_6_saelens/ (SAELens format)
  - Target: /data/trained_saes/gpt2/layer_6/relu/ (SAEStandard format)

Key differences:
  - SAELens uses safetensors, SAEStandard uses PyTorch .pt files
  - SAELens naming: W_enc, b_enc, W_dec, b_dec
  - SAEStandard naming: Ae, be, Ad, bd, lambda_pre
  - Weight matrix shapes need to be transposed
  - Bias shapes need to be reshaped from [n] to [1, n]
"""

import torch
import json
import yaml
from pathlib import Path
from safetensors import safe_open
from omegaconf import OmegaConf

# Configuration
SOURCE_DIR = Path("/home/can/feature_zoo/data/trained_saes/gpt2/layer_6_saelens")
TARGET_DIR = Path("/home/can/feature_zoo/data/trained_saes/gpt2/layer_6/relu")

print("=" * 80)
print("GPT2 SAE CONVERSION: SAELens -> SAEStandard Format")
print("=" * 80)

# Step 1: Load and inspect GPT2 SAE config
print("\n[1/4] Loading GPT2 SAE config...")
with open(SOURCE_DIR / "cfg.json", "r") as f:
    gpt2_cfg = json.load(f)

print(f"  Model: {gpt2_cfg['model_name']}")
print(f"  Hook point: {gpt2_cfg['hook_point']}")
print(f"  Layer: {gpt2_cfg['hook_point_layer']}")
print(f"  d_in: {gpt2_cfg['d_in']}")
print(f"  d_sae: {gpt2_cfg['d_sae']}")
print(f"  Expansion factor: {gpt2_cfg['expansion_factor']}")

# Step 2: Load and inspect GPT2 SAE weights
print("\n[2/4] Loading GPT2 SAE weights from safetensors...")
weights = {}
with safe_open(SOURCE_DIR / "sae_weights.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)
        print(f"  {key}: {weights[key].shape} ({weights[key].dtype})")

# Step 3: Convert to SAEStandard format
print("\n[3/4] Converting weights to SAEStandard format...")

# SAELens format:
#   W_enc: [d_in, d_sae]  -> transpose to Ae: [d_sae, d_in]
#   b_enc: [d_sae]        -> reshape to be: [1, d_sae]
#   W_dec: [d_sae, d_in]  -> transpose to Ad: [d_in, d_sae]
#   b_dec: [d_in]         -> reshape to bd: [1, d_in]

sae_state_dict = {
    "Ae": weights["W_enc"].T.contiguous(),      # [d_in, d_sae] -> [d_sae, d_in]
    "be": weights["b_enc"].unsqueeze(0),        # [d_sae] -> [1, d_sae]
    "Ad": weights["W_dec"].T.contiguous(),      # [d_sae, d_in] -> [d_in, d_sae]
    "bd": weights["b_dec"].unsqueeze(0),        # [d_in] -> [1, d_in]
    "lambda_pre": torch.tensor([1.0]),          # Initialize scaling parameter
}

print("  Converted weights:")
for key, tensor in sae_state_dict.items():
    print(f"    {key}: {tensor.shape} ({tensor.dtype})")

# Step 4: Create config in OmegaConf format
print("\n[4/4] Creating config file...")

# Create a config similar to the llama SAE format
config = {
    "deploy": True,
    "tag": "saelens_converted",
    "seed": gpt2_cfg["seed"],
    "device_id": "cuda:0",
    "data": {
        "epochs": 1,
        "num_total_steps": gpt2_cfg["total_training_tokens"] // gpt2_cfg["train_batch_size"],
        "context_length": gpt2_cfg["context_size"],
        "batch_size": gpt2_cfg["store_batch_size"],
        "dtype": "float32",
        "hf_name": gpt2_cfg["dataset_path"],
        "num_workers": 2,
        "cache_dir": gpt2_cfg.get("cached_activations_path", ""),
    },
    "llm": {
        "model_hf_name": gpt2_cfg["model_name"],
        "tokenizer_hf_name": gpt2_cfg["model_name"],
        "dimin": gpt2_cfg["d_in"],
    },
    "sae": {
        "sae_type": "relu",
        "block_id": gpt2_cfg["hook_point_layer"],
        "exp_factor": gpt2_cfg["expansion_factor"],
        "kval_topk": 256,  # Default value
        "mp_kval": 256,    # Default value
        "branchingFactor": 10,  # Default value
        "gamma_reg": 8,    # Default value
        "encoder_reg": True,
        "scaling_factor": 0.085,  # Default value
    },
    "optimizer": {
        "learning_rate": gpt2_cfg["lr"],
        "weight_decay": 0.0001,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "decay_lr": True,
        "warmup_iters": gpt2_cfg.get("lr_warm_up_steps", 200),
        "min_lr": gpt2_cfg["lr"] * 0.9,
    },
    "eval": {
        "save_tables": False,
    },
    "log": {
        "save_multiple": False,
        "log_interval": gpt2_cfg["wandb_log_frequency"],
        "save_interval": 20000,
        "wandb_project_name": gpt2_cfg.get("wandb_project", ""),
    },
}

# Step 5: Create checkpoint
print("\n[5/5] Creating checkpoint...")
checkpoint = {
    "sae": sae_state_dict,
    "optimizer": {},  # Empty optimizer state
    "iter": gpt2_cfg["total_training_tokens"] // gpt2_cfg["train_batch_size"],
    "config": OmegaConf.create(config),
}

# Step 6: Save to target directory
print(f"\n[6/6] Saving to {TARGET_DIR}...")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Save config as YAML
conf_path = TARGET_DIR / "conf.yaml"
with open(conf_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"  ✓ Saved config to {conf_path}")

# Save checkpoint
ckpt_path = TARGET_DIR / "latest_ckpt.pt"
torch.save(checkpoint, ckpt_path)
print(f"  ✓ Saved checkpoint to {ckpt_path}")

# Verification
print("\n" + "=" * 80)
print("VERIFICATION")
print("=" * 80)
print(f"\nLoading checkpoint from {ckpt_path}...")
loaded_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("\nCheckpoint keys:", list(loaded_ckpt.keys()))
print("\nSAE state_dict:")
for key, tensor in loaded_ckpt["sae"].items():
    print(f"  {key}: {tensor.shape} ({tensor.dtype})")

print("\n" + "=" * 80)
print("CONVERSION COMPLETE!")
print("=" * 80)
print(f"\nConverted SAE saved to: {TARGET_DIR}")
print("\nYou can now load it using:")
print(f"  sae = SAEStandard.from_pretrained('{TARGET_DIR}', dtype=..., device=...)")
