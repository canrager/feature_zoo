# feature_zoo
A collection of multidimensional features in the embedding space of language models.

- [ ] Days of the week (Engels et al)
    - Data at `/data/texts/days.txt` # days in uppercase, lowercase, lowercase + s
    - Filtering for 21k weekday occurences in huggingface fineweb takes about 3h



# Data

Each dataset lives in `data/texts/` as a plain-text file of labels (one per line) and is configured via a YAML file in `configs/data/`.

### Ground truth pairwise distances

Pre-computed pairwise distance matrices are stored as labeled square CSVs in `data/gram/`. Each CSV has row and column headers matching the dataset labels, with entry `(i, j)` giving the ground truth distance between elements `i` and `j`. A dataset config can reference one via the `gram` field (filename relative to `data/gram/`).

Generate all gram matrices:
```bash
python exp/generate_gram.py
```


# Code Conventions

We denote tensor shapes via suffixes (e.g. `llm_og_BCD`, `llm_og_bD`):

| Symbol | Meaning |
|--------|---------|
| `B` | Batch |
| `C` | Condition / Class |
| `D` | Embedding dimension |
| `T` | Time / Sequence position / context length |
| `R` | PCA truncation rank |
| `b` | Flattened batch (`B x C`) |
| `H` | Homology dimension |


### Upload artifacts to HuggingFace
```bash
./scripts/hf_upload_artifacts.sh
```
Uploads `data/activations`, `data/texts`, and `data/tokens` to `canrager/feature_zoo`

### Download artifacts from HuggingFace
```bash
./scripts/hf_download_artifacts.sh
```
Downloads artifacts from `canrager/feature_zoo` to local `data/` directory
