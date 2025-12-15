#!/bin/bash
# ===========================================
# Submit all remaining training jobs
# CE seeds 0-4, SCL seeds 0-4, Triplet seeds 0-4
# (excluding seed 42 which is already done/running)
# ===========================================

BASE_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"
cd ${BASE_DIR}

# Seeds to train
SEEDS=(0 1 2 3 4)

echo "=========================================="
echo "CLXAI: Submit All Training Jobs"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo ""

# Create log directory
mkdir -p logs/slurm

# Submit CE jobs (5 seeds × ~22 min each)
echo "Submitting 5 CE jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_ce_s${seed}" \
           --output="logs/slurm/ce_seed${seed}_%j.out" \
           --error="logs/slurm/ce_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/ce_seed${seed}",RUN_NAME="ce_seed${seed}" \
           scripts/train_ce_seed.sh
    echo "  ✓ CE seed ${seed}"
    sleep 0.5
done
echo ""

# Submit SCL jobs (5 seeds × ~2h each)
echo "Submitting 5 SCL jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_scl_s${seed}" \
           --output="logs/slurm/scl_seed${seed}_%j.out" \
           --error="logs/slurm/scl_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/scl_seed${seed}",RUN_NAME="scl_seed${seed}" \
           scripts/train_scl_seed.sh
    echo "  ✓ SCL seed ${seed}"
    sleep 0.5
done
echo ""

# Submit Triplet jobs (5 seeds × ~2h each)
echo "Submitting 5 Triplet jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_tri_s${seed}" \
           --output="logs/slurm/triplet_seed${seed}_%j.out" \
           --error="logs/slurm/triplet_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/triplet_seed${seed}",RUN_NAME="triplet_seed${seed}" \
           scripts/train_triplet_seed.sh
    echo "  ✓ Triplet seed ${seed}"
    sleep 0.5
done

echo ""
echo "=========================================="
echo "Submitted 15 jobs total:"
echo "  - 5 CE baseline (seeds 0-4)"
echo "  - 5 SCL SupCon (seeds 0-4)"
echo "  - 5 Triplet (seeds 0-4)"
echo ""
echo "Check status: squeue -u \$USER"
echo "=========================================="
