"""
Utils for tokenizing and saving datasets
"""

import os
import torch as th
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file

from src.config import Config
from src.loading import load_tokenizer, load_texts


def save_tokenized(
    cfg: Config, texts: List[str], tokenizer: AutoTokenizer
) -> Dict[str, th.Tensor]:
    "Load texts separated by newlines, tokenize and save"
    # Tokenize
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        padding_side="right",
    )

    # Save as safetensors
    tokens_dir = Path(cfg.env.tokens_dir)
    tokens_dir.mkdir(parents=True, exist_ok=True)
    output_path = tokens_dir / f"{cfg.data.name}.safetensors"

    # Prepare tensors for saving
    # input_ids = th.cat(input_ids_list, dim=0)
    # attention_mask = th.cat(attention_mask_list, dim=0)
    token_dict = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    save_file(token_dict, output_path)

    return token_dict


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    labels, texts = load_texts(cfg)
    tokenizer = load_tokenizer(cfg)
    tokenized = save_tokenized(cfg, texts, tokenizer)
    print(f"Loaded {len(tokenized['input_ids'])} tokenized texts")
    if labels:
        print(f"Labels available: {len(labels)} labels")
    print(f"First input_ids shape: {tokenized['input_ids'][0].shape}")
    print(f"First attention_mask shape: {tokenized['attention_mask'][0].shape}")
