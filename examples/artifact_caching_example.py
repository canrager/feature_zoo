#!/usr/bin/env python3
"""
Example demonstrating the content-addressable artifact caching system.

This shows how the system automatically handles caching and validation.
"""

from src.config import load_config
from src.cache_llm import load_short_trajectory_acts

def main():
    print("=" * 80)
    print("ARTIFACT CACHING EXAMPLE")
    print("=" * 80)

    # Load default config
    cfg = load_config()
    print(f"\nLoaded config:")
    print(f"  Data: {cfg.data.name}")
    print(f"  LLM: {cfg.llm.name} (layer {cfg.llm.layer_idx})")
    print(f"  Fixed context length: {cfg.data.fixed_context_length}")

    # Show what artifact names will be used
    from src.artifact_utils import (
        get_config_hash,
        get_token_artifact_path,
        get_llm_activation_artifact_path
    )

    token_hash = get_config_hash(cfg, "tokens")
    llm_hash = get_config_hash(cfg, "llm_activations")

    print(f"\nArtifact hashes:")
    print(f"  Tokens: {token_hash}")
    print(f"  LLM activations: {llm_hash}")

    token_path, token_meta = get_token_artifact_path(cfg)
    llm_path, llm_meta = get_llm_activation_artifact_path(cfg)

    print(f"\nArtifact paths:")
    print(f"  Tokens: {token_path.name}")
    print(f"  Token metadata: {token_meta.name}")
    print(f"  LLM activations: {llm_path.name}")
    print(f"  LLM metadata: {llm_meta.name}")

    # Demonstrate validation
    print(f"\n" + "=" * 80)
    print("LOADING WITH VALIDATION")
    print("=" * 80)

    print("\nAttempting to load artifacts...")
    print("(This will recompute if not cached, or validate if cached)")

    # This will:
    # 1. Check if artifacts with matching hash exist
    # 2. Validate metadata matches current config
    # 3. Load if valid, or recompute if missing/invalid
    try:
        cfg.env.debug = True  # Show debug output
        result = load_short_trajectory_acts(cfg, force_recompute=False)
        print(f"\n✓ Successfully loaded activations!")
        print(f"  Shape: {result['llm_BCD'].shape}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"\nTip: If config changed, use force_recompute=True to regenerate")

    # Example of forcing recomputation
    print(f"\n" + "=" * 80)
    print("FORCE RECOMPUTE EXAMPLE")
    print("=" * 80)
    print("\nTo force recomputation (ignoring cache):")
    print("  result = load_short_trajectory_acts(cfg, force_recompute=True)")

    # Example of config change detection
    print(f"\n" + "=" * 80)
    print("CONFIG CHANGE DETECTION")
    print("=" * 80)

    print("\nIf you change config after caching:")
    print("  cfg.data.fixed_context_length = 256")
    print("\nThe system will detect the mismatch and raise:")
    print("  ValueError: Config hash mismatch for tokens!")
    print("  Current hash: abc12345")
    print("  Stored hash:  def67890")
    print("  Set force_recompute=True to regenerate the artifact.")

    print(f"\n" + "=" * 80)
    print("ARTIFACT MANAGEMENT")
    print("=" * 80)
    print("\nUse the management script to inspect artifacts:")
    print("  python scripts/manage_artifacts.py list      # Show all artifacts")
    print("  python scripts/manage_artifacts.py orphans   # Show old artifacts")
    print("  python scripts/manage_artifacts.py clean     # Remove old artifacts")


if __name__ == "__main__":
    main()
