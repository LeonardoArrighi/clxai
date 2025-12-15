#!/bin/bash
# ===========================================
# Submit remaining 4 seeds (0, 1, 2, 3)
# Seed 42 is already complete (CE) / running (SCL)
# ===========================================

BASE_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"
cd ${BASE_DIR}

# Seeds to train (seed 42 already done)
SEEDS=(0 1 2 3)

echo "=========================================="
echo "Submitting remaining 4 seeds for CE and SCL"
echo "Seeds: ${SEEDS[@]}"
echo "(Seed 42 already complete/running)"
echo "=========================================="
echo ""

# Create log directory
mkdir -p logs/slurm

# Submit CE jobs (4 seeds × ~22 min each = ~1.5h total if parallel)
echo "Submitting 4 CE jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_ce_s${seed}" \
           --output="logs/slurm/ce_seed${seed}_%j.out" \
           --error="logs/slurm/ce_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/ce_seed${seed}",RUN_NAME="ce_seed${seed}" \
           scripts/train_ce_seed.sh
    echo "  ✓ CE seed ${seed}"
done

echo ""

# Submit SCL jobs (4 seeds × ~2h each = ~8h total if parallel)
echo "Submitting 4 SCL jobs..."
for seed in "${SEEDS[@]}"; do
    sbatch --job-name="clxai_scl_s${seed}" \
           --output="logs/slurm/scl_seed${seed}_%j.out" \
           --error="logs/slurm/scl_seed${seed}_%j.err" \
           --export=ALL,SEED=${seed},OUTPUT_DIR="results/models/scl_seed${seed}",RUN_NAME="scl_seed${seed}" \
           scripts/train_scl_seed.sh
    echo "  ✓ SCL seed ${seed}"
done

echo ""
echo "=========================================="
echo "Submitted 8 jobs total (4 CE + 4 SCL)"
echo ""
echo "Combined with existing seed 42 models:"
echo "  CE:  5 total (seed42 + 0,1,2,3)"
echo "  SCL: 5 total (seed42 + 0,1,2,3)"
echo ""
echo "Check status: squeue -u \$USER"
echo "=========================================="
