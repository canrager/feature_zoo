# feature_zoo
A collection of multidimensional features in the embedding space of language models.

- [ ] Days of the week (Engels et al)
    - Data at `/data/texts/days.txt` # days in uppercase, lowercase, lowercase + s
    - Filtering for 21k weekday occurences in huggingface fineweb takes about 3h



# Code Conventions

We'll denote tensor shapes via suffixes:
- B: Batch
- T: Time / Sequence position / context length
- D: Model embedding dimension


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
