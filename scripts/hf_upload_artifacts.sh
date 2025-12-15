#!/bin/bash
set -e

REPO_ID="canrager/feature_zoo"
REPO_TYPE="model"

# Define artifacts to upload
artifacts=(
    "data/activations"
    "data/texts"
    "data/tokens"
)

echo "Starting upload to HuggingFace repository: $REPO_ID"

# Create repo if it doesn't exist (will skip if exists)
echo "Ensuring repository exists: $REPO_ID"
hf repo create $REPO_ID --repo-type $REPO_TYPE 2>/dev/null || echo "Repository already exists or created successfully"

# Upload each artifact folder
for artifact in "${artifacts[@]}"; do
    if [ -d "$artifact" ]; then
        echo "Uploading $artifact to $REPO_ID..."
        hf upload $REPO_ID "$artifact" "$artifact" --repo-type $REPO_TYPE
    else
        echo "Warning: Directory $artifact does not exist, skipping..."
    fi
done

echo "Upload complete!"
