#!/bin/bash
# CLXAI Environment Setup Script for Leonardo HPC

echo "=========================================="
echo "CLXAI Environment Setup"
echo "=========================================="

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Set paths
PROJECT_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"
ENV_DIR="${PROJECT_DIR}/clxai_env"

# Create virtual environment if it doesn't exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment..."
    python -m venv "$ENV_DIR" --system-site-packages
fi

# Activate environment
source "${ENV_DIR}/bin/activate"

# Install additional packages
echo "Installing additional packages..."
pip install --upgrade pip
pip install captum>=0.7.0
pip install pytorch-metric-learning>=2.3.0
pip install umap-learn>=0.5.4
pip install omegaconf>=2.3.0

# Create directories
echo "Creating directories..."
mkdir -p "${PROJECT_DIR}/data"
mkdir -p "${PROJECT_DIR}/logs/slurm"
mkdir -p "${PROJECT_DIR}/results/models/ce_baseline"
mkdir -p "${PROJECT_DIR}/results/models/scl"
mkdir -p "${PROJECT_DIR}/results/evaluation"
mkdir -p "${PROJECT_DIR}/results/figures"
mkdir -p "${PROJECT_DIR}/results/embeddings"

# Make scripts executable
chmod +x "${PROJECT_DIR}/scripts/"*.sh
chmod +x "${PROJECT_DIR}/scripts/"*.py

# Verify setup
echo ""
echo "Running verification..."
cd "$PROJECT_DIR"
python scripts/verify_setup.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To activate the environment:"
echo "  module load profile/deeplrn"
echo "  module load cineca-ai/4.3.0"
echo "  source ${ENV_DIR}/bin/activate"
echo "  cd ${PROJECT_DIR}"
echo ""
echo "To train models:"
echo "  sbatch scripts/train_ce.sh    # CE baseline"
echo "  sbatch scripts/train_scl.sh   # SCL model"
echo ""
echo "To run evaluation:"
echo "  sbatch scripts/run_xai_eval.sh"
echo "=========================================="
