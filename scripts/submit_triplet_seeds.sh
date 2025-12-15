#!/bin/bash
# ===========================================
# Submit 5 triplet loss training jobs (seeds 0-4)
# ===========================================

BASE_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"
cd ${BASE_DIR}

# Seeds to train
SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "Submitting 5 Triplet Loss Training Jobs"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="
echo ""

# Create log directory
mkdir -p logs/slurm

# Submit Triplet jobs
echo "Submitting 5 Triplet jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_tri_s${seed}" \
           --output="logs/slurm/triplet_seed${seed}_%j.out" \
           --error="logs/slurm/triplet_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/triplet_seed${seed}",RUN_NAME="triplet_seed${seed}" \
           scripts/train_triplet_seed.sh
    echo "  ✓ Triplet seed ${seed}"
done

echo ""
echo "=========================================="
echo "Submitted 5 Triplet jobs"
echo ""
echo "Check status: squeue -u \$USER"
echo "=========================================="
