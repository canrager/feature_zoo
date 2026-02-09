### Tensor Dimensionality Annotations

All tensors (`torch.Tensor`) and NumPy arrays (`np.ndarray`) **must** be annotated with their dimensionality by appending dimension symbols as a suffix to the variable name, separated by `_`. Example:

```python
llm_og_BCD = experiment_data["original"]["llm_BCD"]
llm_og_bD = einops.rearrange(llm_og_BCD, "B C D -> (B C) D")
U_og_bD, S_og_D, Vh_og_DD = np.linalg.svd(llm_og_bD, full_matrices=False)
```

**Rules:**
- Each uppercase letter represents a distinct dimension (e.g., `B`, `C`, `D`).
- A lowercase letter represents a flattened/combined dimension (e.g., `b = B x C`).
- Scalar results (0-d) don't need a suffix.
- When introducing a **new** dimension symbol not listed below, **ask the user** before using it.
- The dimension symbol registry in `README.md` **must** be kept in sync with the table below. When a new symbol is approved, update both this file and `README.md`.

**Dimension symbol registry:**

| Symbol | Meaning |
|--------|---------|
| `B` | Batch |
| `C` | Condition / Class |
| `D` | Embedding dimension |
| `T` | Time / Sequence position / context length |
| `b` | Flattened batch (`B x C`) |

### HuggingFace CLI Commands
```bash
# Create repository
hf repo create [repo_id] --repo-type [model|dataset|space]

# Upload files/folders
hf upload [repo_id] [local_path] [path_in_repo] --repo-type [type]

# Download with filters
hf download [repo_id] --include [pattern] --local-dir [dir] --repo-type [type]
```
