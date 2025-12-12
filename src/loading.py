"""
Utils for loading language models from huggingface
"""

from numpy import True_
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from src.config import Config
from typing import List, Tuple, Optional


def load_llm(cfg: Config) -> AutoModelForCausalLM:
    kwargs = {}

    if cfg.llm.quantization_bits is not None:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm.hf_name,
        dtype=cfg.env.dtype,
        device_map=cfg.env.device,
        cache_dir=cfg.env.hf_cache_dir,
        **kwargs,
    )
    return llm


def load_tokenizer(cfg: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.hf_name, cache_dir=cfg.env.hf_cache_dir
    )
    if "gpt2" in cfg.llm.hf_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_texts(
    cfg: Config, filename: str | None = None
) -> Tuple[List[str], Optional[List[str]]]:
    """
    Load texts from CSV file with 'label' and 'text' columns.

    Returns:
        Tuple[List[str], Optional[List[str]]]: (texts, labels).
        If 'label' column is missing or empty, labels will be None.
    """
    if filename is None:
        filename = cfg.data.name

    csv_path = Path(cfg.env.texts_dir) / f"{filename}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file {csv_path} does not exist")

    # Load CSV using pandas
    df = pd.read_csv(csv_path)
    labels = df["label"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    return labels, texts


def load_corpus(cfg: Config) -> Dataset:
    return load_dataset(cfg.env.corpus, split="train", streaming=True)


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config("olmo3-32b-base")
    print(load_tokenizer(cfg))
    print(load_llm(cfg))
