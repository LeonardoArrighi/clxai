#!/bin/bash
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --job-name=clxai_ce
#SBATCH --output=logs/slurm/ce_baseline_%j.out
#SBATCH --error=logs/slurm/ce_baseline_%j.err

echo "=========================================="
echo "CLXAI: Cross-Entropy Baseline Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/wandb

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate 2>/dev/null || {
    echo "Creating virtual environment..."
    python -m venv /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env --system-site-packages
    source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
    pip install captum pytorch-metric-learning umap-learn
}

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm results/models/ce_baseline data wandb

# Run training
echo "Starting CE training..."
python src/training/train_ce.py \
    --config configs/ce_baseline.yaml \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --data_dir ./data \
    --output_dir results/models/ce_baseline \
    --run_name ce_baseline

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
