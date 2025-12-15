#!/bin/bash
set -e

REPO_ID="canrager/feature_zoo"
REPO_TYPE="model"

# Define artifacts to download
artifacts=(
    "data/activations"
    "data/texts"
    "data/tokens"
)

echo "Starting download from HuggingFace repository: $REPO_ID"

# Create data directory if it doesn't exist
mkdir -p data

# Download each artifact folder
for artifact in "${artifacts[@]}"; do
    echo "Downloading $artifact from $REPO_ID..."
    hf download $REPO_ID --include "$artifact/*" --local-dir . --repo-type $REPO_TYPE
done

echo "Download complete!"
