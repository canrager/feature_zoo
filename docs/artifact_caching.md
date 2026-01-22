# Artifact Caching System

This document describes the content-addressable artifact caching system used in the feature_zoo project.

## Overview

The caching system ensures that artifacts (tokenized data, LLM activations, SAE activations) are always correctly matched to their configuration. It prevents bugs from loading outdated or mismatched cached data.

## How It Works

### Content-Addressable Hashing

Each artifact is named using a **config hash** that uniquely identifies the configuration parameters used to create it:

```
{data_name}_{config_hash}.safetensors              # Tokens
{data_name}_{llm_name}_layer{idx}_{config_hash}_llm.safetensors  # LLM activations
{data_name}_{llm_name}_layer{idx}_{sae_arch}_{config_hash}_{type}.safetensors  # SAE activations
```

### Hash Includes

The hash is computed from:
- **For all artifacts:**
  - `data.name`
  - `data.elements_filename`
  - `data.trajectories_filename`
  - `data.fixed_context_length`
  - `llm.hf_name`
  - `llm.layer_idx`

- **Additionally for SAE artifacts:**
  - `sae.llm_name`
  - `sae.llm_layer_idx`
  - `sae.arch`
  - `sae.act_scaling_factor`

### Metadata Files

Each `.safetensors` artifact has a companion `.json` metadata file with:
- Config hash
- Creation timestamp
- Full config snapshot
- Artifact shape information

## Using the System

### Normal Usage

The system works automatically. When you call `load_labeled_acts()` or `load_short_trajectory_acts()`:

1. **If `force_recompute=False` (default):**
   - Checks if artifact with matching hash exists
   - Validates metadata matches current config
   - Loads the artifact if valid
   - Raises clear error if metadata doesn't match

2. **If artifact doesn't exist:**
   - Computes and saves with hash-based filename
   - Saves metadata alongside

3. **If `force_recompute=True`:**
   - Always recomputes, regardless of cached artifacts
   - Overwrites existing artifacts with same hash

### Example Error Message

If you change config after artifacts were cached:

```
ValueError: Config hash mismatch for tokens!
Current hash: abc12345
Stored hash:  def67890
This likely means the config changed after the artifact was cached.
Set force_recompute=True to regenerate the artifact.
```

This tells you **exactly** what's wrong and how to fix it.

## Managing Artifacts

Use the `scripts/manage_artifacts.py` utility:

```bash
# List all artifacts with metadata
python scripts/manage_artifacts.py list

# List orphaned artifacts (old format without metadata)
python scripts/manage_artifacts.py orphans

# Preview cleanup (dry run)
python scripts/manage_artifacts.py clean --dry-run

# Actually delete orphaned artifacts
python scripts/manage_artifacts.py clean

# Show detailed info about specific artifact
python scripts/manage_artifacts.py info my_artifact.safetensors
```

## Migration from Old System

Old artifacts (without config hashes) are automatically detected as "orphaned" because they lack metadata files.

**Options:**

1. **Keep old artifacts:** They won't interfere. New artifacts use different filenames.

2. **Clean old artifacts:** Use `python scripts/manage_artifacts.py clean` to remove them and free disk space.

3. **Regenerate everything:** Use `force_recompute=True` to create new hash-based artifacts.

## Benefits

✅ **Correctness:** Impossible to load wrong cached artifact for current config

✅ **Debugging:** Clear error messages explain mismatches

✅ **Transparency:** Metadata files show exactly what config created each artifact

✅ **Storage efficiency:** Multiple configs can coexist without overwriting each other

✅ **Reproducibility:** Config hash ensures deterministic artifact naming

## Technical Details

### Hash Algorithm

- Uses SHA256 for cryptographic quality
- Truncated to 8 hex characters (32 bits)
- Collision probability: ~1 in 4 billion for random configs

### Metadata Schema

```json
{
  "artifact_type": "tokens",
  "created_at": "2026-01-20T12:34:56.789",
  "config_hash": "abc12345",
  "num_samples": 1000,
  "max_seq_length": 128,
  "config_snapshot": {
    "data": { ... },
    "llm": { ... },
    "env": { ... }
  }
}
```

## Troubleshooting

**Q: I get "Config hash mismatch" but I didn't change anything**

A: Some config change occurred (possibly in a default file). Use `force_recompute=True` or check the metadata file to see what differs.

**Q: Can I manually edit config hashes?**

A: No, hashes are computed deterministically. Manual edits will cause validation failures.

**Q: How do I share artifacts between machines?**

A: Copy both `.safetensors` and `.json` files. The hash ensures they match the config.

**Q: What if I want to keep multiple versions?**

A: Just keep different configs! Each generates its own hash, so artifacts won't overwrite.
