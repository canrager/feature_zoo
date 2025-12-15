### HuggingFace CLI Commands
```bash
# Create repository
hf repo create [repo_id] --repo-type [model|dataset|space]

# Upload files/folders
hf upload [repo_id] [local_path] [path_in_repo] --repo-type [type]

# Download with filters
hf download [repo_id] --include [pattern] --local-dir [dir] --repo-type [type]
```
