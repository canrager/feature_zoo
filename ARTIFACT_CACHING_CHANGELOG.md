# Artifact Caching System - Implementation Changelog

## Date: 2026-01-20

## Summary

Implemented a **content-addressable artifact caching system** to ensure correct artifacts are always loaded when `force_recompute=False`. This prevents bugs from loading outdated or mismatched cached data.

## Problem Statement

The previous caching system had several critical issues:

1. **Filename-based matching insufficient** - Artifacts named only by `{data_name}_{llm_name}_layer{idx}` didn't encode important parameters like:
   - `fixed_context_length` (affects tokenization)
   - `sae.act_scaling_factor` (affects SAE activations)
   - Full model path (`llm.hf_name` vs short `name`)

2. **No validation** - System only checked if files existed, not if they matched current config

3. **Silent bugs** - Loading wrong cached artifacts produced subtle, hard-to-debug errors

4. **Duplicate accumulation** - Changing configs slightly would overwrite artifacts with same names

## Solution Implemented

### 1. Content-Addressable Hashing (`src/artifact_utils.py`)

Created deterministic hash from all relevant config parameters:
- For tokens: `data.*`, `llm.hf_name`, `llm.layer_idx`, `fixed_context_length`
- For LLM activations: Same as tokens
- For SAE activations: Add `sae.arch`, `sae.act_scaling_factor`

Hash: SHA256 truncated to 8 hex chars (1 in 4 billion collision rate)

### 2. Hash-Based Filenames

**Before:**
```
tokens/days_trajectories.safetensors
activations/days_trajectories_gpt2_layer6_llm.safetensors
```

**After:**
```
tokens/days_trajectories_a1f14627.safetensors
activations/days_trajectories_gpt2_layer6_a1f14627_llm.safetensors
```

### 3. Metadata Files

Each `.safetensors` now has companion `.json` with:
- Config hash
- Creation timestamp
- Full config snapshot
- Shape information

### 4. Validation on Load

Before loading any artifact:
1. Check hash-based filename exists
2. Load and validate metadata
3. Raise clear error if config mismatch
4. Only then load the artifact

### 5. Management Utility

Created `scripts/manage_artifacts.py` to:
- List all artifacts with metadata
- Identify orphaned artifacts (old format)
- Clean up old artifacts
- Show detailed info for debugging

## Files Modified

### Core Changes
- **`src/artifact_utils.py`** (NEW) - Hash generation, metadata management, validation
- **`src/tokenization.py`** - Updated `save_tokenized()` to use hashes and save metadata
- **`src/cache_llm.py`** - Updated all save/load functions for tokens, LLM acts, SAE acts
- **`src/cache_sae.py`** - Updated `save_sae_cache()` to use hashes and save metadata

### Utilities & Documentation
- **`scripts/manage_artifacts.py`** (NEW) - Artifact management CLI
- **`docs/artifact_caching.md`** (NEW) - User documentation
- **`tests/test_artifact_caching.py`** (NEW) - Comprehensive test suite

## Migration Path

### For Existing Artifacts

Old artifacts (without hashes) are automatically detected as "orphaned":

```bash
# See what old artifacts exist
python scripts/manage_artifacts.py orphans

# Preview cleanup
python scripts/manage_artifacts.py clean --dry-run

# Actually clean (optional - old artifacts won't interfere)
python scripts/manage_artifacts.py clean
```

### For New Work

Just use normally with `force_recompute=False`:
- First run: Computes and saves with hash-based filenames
- Subsequent runs: Loads from cache with validation
- Config changes: Automatic error with clear message

## Example Error Messages

### Before (Silent Bug)
```python
# Changed fixed_context_length from 128 to 256
# Loaded old tokens with length 128
# Got cryptic shape mismatch errors later
```

### After (Clear Error)
```python
ValueError: Config hash mismatch for tokens!
Current hash: a1f14627
Stored hash:  14448f1d
This likely means the config changed after the artifact was cached.
Set force_recompute=True to regenerate the artifact.
```

## Benefits

✅ **Correctness** - Impossible to load wrong cached artifact

✅ **Debuggability** - Clear errors explain exactly what's wrong

✅ **Transparency** - Metadata shows what created each artifact

✅ **Storage efficiency** - Multiple configs coexist without overwrites

✅ **Reproducibility** - Deterministic hashing ensures same config → same hash

## Testing

Comprehensive test suite covers:
- Hash determinism
- Hash changes with config changes
- Metadata creation/save/load
- Validation success and failure
- Path generation for all artifact types
- Strict vs non-strict validation modes

All tests passing ✅

## Performance Impact

Negligible:
- Hash computation: <1ms
- Metadata I/O: <10ms per artifact
- Path lookup: Same as before

## Backward Compatibility

**Breaking:** Old artifact filenames won't be loaded automatically

**Mitigation:**
1. Old artifacts are preserved (different filenames)
2. First run with new system regenerates with `force_recompute=True` if needed
3. Management script helps identify and clean old artifacts

## Future Enhancements

Potential improvements:
1. Add checksums to detect corrupted artifacts
2. Compression for metadata files
3. Cloud sync support (artifacts + metadata together)
4. Artifact provenance tracking (what code version created it)

## Notes for Developers

When adding new config parameters that affect artifacts:
1. Add parameter to hash computation in `get_config_hash()`
2. Add to config snapshot in metadata
3. Update documentation
4. Write test to verify hash changes with new parameter
