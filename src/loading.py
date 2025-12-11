"""
Utils for loading language models from huggingface
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import Config


def load_llm(cfg: Config) -> AutoModelForCausalLM:
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.llm.hf_name,
        dtype=cfg.env.dtype,
        device_map=cfg.env.device,
        cache_dir=cfg.env.hf_cache_dir,
    )
    return llm


def load_tokenizer(cfg: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.hf_name, cache_dir=cfg.env.hf_cache_dir
    )
    return tokenizer


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config()
    print(load_tokenizer(cfg))
    print(load_llm(cfg))
