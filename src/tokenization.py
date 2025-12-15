"""
Utils for tokenizing and saving datasets
"""

import os
import torch as th
from typing import List, Dict
from pathlib import Path
from transformers import AutoTokenizer
from safetensors.torch import save_file, load_file
from collections import defaultdict

from src.config import Config
from src.loading import load_tokenizer, load_texts


def batch_tokenize(
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
        add_special_tokens=True,
    )

    # For models like GPT-2 that don't add BOS tokens automatically, prepend them
    if encoded["input_ids"][0, 0].item() != tokenizer.bos_token_id:
        bos_tokens = th.full(
            (encoded["input_ids"].shape[0], 1),
            tokenizer.bos_token_id,
            dtype=encoded["input_ids"].dtype,
            device=encoded["input_ids"].device,
        )
        ones = th.ones_like(bos_tokens)
        encoded["input_ids"] = th.cat((bos_tokens, encoded["input_ids"]), dim=-1)
        encoded["attention_mask"] = th.cat((ones, encoded["attention_mask"]), dim=-1)

    # Optionally truncate to final cfg.data.fixed_context_length
    if cfg.data.fixed_context_length is not None:
        ctx_len = cfg.data.fixed_context_length

        # assert that every sequence starts with a BOS token
        unique_first_tokens = set(encoded["input_ids"][:, 0].tolist())
        assert unique_first_tokens == {tokenizer.bos_token_id}

        # Remove BOS tokens (to be added back later)
        encoded["input_ids"] = encoded["input_ids"][:, 1:]
        encoded["attention_mask"] = encoded["attention_mask"][:, 1:]

        seq_lengths = encoded["attention_mask"].sum(dim=-1)  # (B,)

        # Check that all sequences are long enough
        if (seq_lengths < ctx_len).any():
            min_len = seq_lengths.min().item()
            raise ValueError(
                f"Some sequences are shorter than fixed_context_length={ctx_len}. "
                f"Minimum sequence length is {min_len}."
            )

        # Start index for each batch: seq_len - ctx_len
        start_indices = seq_lengths - ctx_len  # (B,)

        # Create indices: for each batch, [start, start+1, ..., start+ctx_len-1]
        indices = start_indices.unsqueeze(1) + th.arange(ctx_len).unsqueeze(
            0
        )  # (B, ctx_len)

        encoded = {
            "input_ids": th.gather(encoded["input_ids"], dim=1, index=indices),
            "attention_mask": th.gather(
                encoded["attention_mask"], dim=1, index=indices
            ),
        }

        # Add BOS tokens back in
        bos_tokens_B = th.full(
            (encoded["input_ids"].shape[0], 1),
            tokenizer.bos_token_id,
            dtype=encoded["input_ids"].dtype,
            device=encoded["input_ids"].device,
        )
        ones_B = th.ones_like(bos_tokens_B)

        encoded["input_ids"] = th.cat((bos_tokens_B, encoded["input_ids"]), dim=-1)
        encoded["attention_mask"] = th.cat((ones_B, encoded["attention_mask"]), dim=-1)

    return encoded


def save_tokenized(
    cfg: Config, texts: List[str], tokenizer: AutoTokenizer
) -> Dict[str, th.Tensor]:
    "Load texts separated by newlines, tokenize and save"

    encoded = batch_tokenize(cfg, texts, tokenizer)

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
