#!/bin/bash
# ===========================================
# Multi-seed training submission script
# Submits 5 seeds for each model type (CE and SCL)
# ===========================================

# Seeds to use (seed 42 was used for initial run, now use 0-4 for consistency)
SEEDS=(0 1 2 3 4)

# Base directory
BASE_DIR="/leonardo_scratch/fast/CNHPC_1905882/clxai"

echo "=========================================="
echo "CLXAI Multi-Seed Training Submission"
echo "=========================================="
echo "Seeds: ${SEEDS[@]}"
echo ""

# Parse arguments
TRAIN_CE=false
TRAIN_SCL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ce)
            TRAIN_CE=true
            shift
            ;;
        --scl)
            TRAIN_SCL=true
            shift
            ;;
        --all)
            TRAIN_CE=true
            TRAIN_SCL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--ce] [--scl] [--all]"
            exit 1
            ;;
    esac
done

# If no arguments, show usage
if [ "$TRAIN_CE" = false ] && [ "$TRAIN_SCL" = false ]; then
    echo "Usage: $0 [--ce] [--scl] [--all]"
    echo "  --ce   : Train CE baseline models (5 seeds)"
    echo "  --scl  : Train SCL models (5 seeds)"
    echo "  --all  : Train both CE and SCL models"
    exit 0
fi

# Submit CE jobs
if [ "$TRAIN_CE" = true ]; then
    echo "Submitting CE baseline jobs..."
    for seed in "${SEEDS[@]}"; do
        JOB_NAME="ce_seed${seed}"
        OUTPUT_DIR="results/models/ce_seed${seed}"
        
        sbatch --job-name="clxai_${JOB_NAME}" \
               --output="logs/slurm/${JOB_NAME}_%j.out" \
               --error="logs/slurm/${JOB_NAME}_%j.err" \
               --export=ALL,SEED=${seed},OUTPUT_DIR=${OUTPUT_DIR},RUN_NAME=${JOB_NAME} \
               ${BASE_DIR}/scripts/train_ce_seed.sh
        
        echo "  Submitted: ${JOB_NAME}"
        sleep 1  # Small delay between submissions
    done
    echo ""
fi

# Submit SCL jobs
if [ "$TRAIN_SCL" = true ]; then
    echo "Submitting SCL jobs..."
    for seed in "${SEEDS[@]}"; do
        JOB_NAME="scl_seed${seed}"
        OUTPUT_DIR="results/models/scl_seed${seed}"
        
        sbatch --job-name="clxai_${JOB_NAME}" \
               --output="logs/slurm/${JOB_NAME}_%j.out" \
               --error="logs/slurm/${JOB_NAME}_%j.err" \
               --export=ALL,SEED=${seed},OUTPUT_DIR=${OUTPUT_DIR},RUN_NAME=${JOB_NAME} \
               ${BASE_DIR}/scripts/train_scl_seed.sh
        
        echo "  Submitted: ${JOB_NAME}"
        sleep 1  # Small delay between submissions
    done
    echo ""
fi

echo "=========================================="
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "=========================================="
