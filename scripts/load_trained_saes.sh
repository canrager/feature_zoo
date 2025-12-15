#!/bin/bash
set -e

# Configuration
# REPO_ID="ekdeepslubana/temporalSAEs_llama"
# SUBDIR="layer_16"
# OUTPUT_DIR="data/trained_saes"

REPO_ID="jbloom/GPT2-Small-SAEs-Reformatted"
SUBDIR="blocks.7.hook_resid_pre"
OUTPUT_DIR="data/trained_saes"

# Navigate to project root
cd "$(dirname "$0")/.."

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download specific subfolder from HuggingFace
hf download "$REPO_ID" \
    --include "${SUBDIR}/*" \
    --local-dir "$OUTPUT_DIR"

echo "Files downloaded to ${OUTPUT_DIR}"
