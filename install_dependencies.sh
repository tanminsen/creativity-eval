#!/usr/bin/env bash
# Linux / macOS setup script.
# Installs Python dependencies from requirements.txt and reports CUDA status.
# The cd / HF_HOME / HF_HUB_CACHE lines below are specific to the NSCC
# (National Supercomputing Centre, Singapore) environment used during the
# paper experiments. Adjust or remove them for other hosts.

set -e

# NSCC-specific cache configuration. Comment out or edit on other hosts.
cd ~/scratch/macgyversemanticprobing
export HF_HOME=~/scratch/macgyversemanticprobing/.cache/huggingface
export HF_HUB_CACHE=~/scratch/macgyversemanticprobing/.cache/huggingface/hub
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"

# Install pinned dependencies.
pip install --upgrade pip
pip install -r requirements.txt

# Reduce fragmentation on large-model CUDA allocations.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA sanity check (non-fatal).
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi || true
nvcc --version || true

echo "Done installing dependencies."
