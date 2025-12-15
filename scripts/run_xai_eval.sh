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
#SBATCH --job-name=clxai_eval
#SBATCH --output=logs/slurm/xai_eval_%j.out
#SBATCH --error=logs/slurm/xai_eval_%j.err

echo "=========================================="
echo "CLXAI: XAI Faithfulness Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# wandb offline mode (compute nodes have no internet)
export WANDB_MODE=offline
export WANDB_DIR=/leonardo_scratch/fast/CNHPC_1905882/clxai/wandb

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p results/evaluation results/figures wandb

# Run evaluation
echo "Running XAI evaluation..."
python scripts/run_evaluation.py \
    --config configs/evaluation.yaml \
    --n_samples 1000

echo ""
echo "Evaluation completed at: $(date)"
echo "=========================================="
