#!/bin/bash
#SBATCH --job-name=clxai_debug
#SBATCH --account=CNHPC_1905882
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --output=logs/slurm/debug_test_%j.out
#SBATCH --error=logs/slurm/debug_test_%j.err

# =============================================================================
# CLXAI Debug Test Suite - Full Node
# =============================================================================
# Runs unit tests and parallel training on 4 GPUs
# QoS: boost_qos_dbg (30 min max, priority scheduling)
# =============================================================================

echo "=========================================="
echo "CLXAI Debug Test Suite - Full Node"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Exit on error
set -e

# Disable wandb (no internet on compute nodes)
export WANDB_MODE=disabled

# Load modules
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate environment
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate 2>/dev/null || {
    echo "ERROR: Environment not found. Please run setup_env.sh first."
    exit 1
}

# Navigate to project
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Create directories
mkdir -p logs/slurm logs/debug results/models/debug/{ce,supcon,triplet_hard,triplet_semi} data

# =============================================================================
# Phase 1: Unit Tests
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 1: Running Unit Tests"
echo "=========================================="

python scripts/test_components.py
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Unit tests failed! Aborting parallel training."
    exit 1
fi

echo ""
echo "Unit tests PASSED! Proceeding to parallel training..."

# =============================================================================
# Phase 2: Parallel Training (4 GPUs)
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 2: Parallel Training (4 GPUs)"
echo "=========================================="
echo "Starting 4 training jobs in parallel..."
echo ""

# Training function
run_training() {
    local GPU_ID=$1
    local MODEL_TYPE=$2
    local SCRIPT=$3
    local CONFIG=$4
    local OUTPUT_DIR=$5
    local LOG_FILE="logs/debug/${MODEL_TYPE}_gpu${GPU_ID}.log"
    
    echo "  [GPU $GPU_ID] Starting $MODEL_TYPE training..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
        --config $CONFIG \
        --output_dir $OUTPUT_DIR \
        --no_wandb \
        > $LOG_FILE 2>&1 &
    
    echo $!  # Return PID
}

# Start all training jobs in parallel
echo "Launching training jobs:"

# GPU 0: CE Baseline
CUDA_VISIBLE_DEVICES=0 python src/training/train_ce.py \
    --config configs/debug_ce.yaml \
    --output_dir results/models/debug/ce \
    --no_wandb \
    > logs/debug/ce_gpu0.log 2>&1 &
PID_CE=$!
echo "  [GPU 0] CE Baseline (PID: $PID_CE)"

# GPU 1: SupCon
CUDA_VISIBLE_DEVICES=1 python src/training/train_scl.py \
    --config configs/debug_supcon.yaml \
    --output_dir results/models/debug/supcon \
    --no_wandb \
    > logs/debug/supcon_gpu1.log 2>&1 &
PID_SUPCON=$!
echo "  [GPU 1] SupCon (PID: $PID_SUPCON)"

# GPU 2: Triplet (Hard Mining)
CUDA_VISIBLE_DEVICES=2 python src/training/train_scl.py \
    --config configs/debug_triplet.yaml \
    --output_dir results/models/debug/triplet_hard \
    --no_wandb \
    > logs/debug/triplet_hard_gpu2.log 2>&1 &
PID_TRIPLET_HARD=$!
echo "  [GPU 2] Triplet Hard (PID: $PID_TRIPLET_HARD)"

# GPU 3: Triplet (Semi-Hard Mining) - modify mining strategy
CUDA_VISIBLE_DEVICES=3 python src/training/train_scl.py \
    --config configs/debug_triplet.yaml \
    --output_dir results/models/debug/triplet_semi \
    --no_wandb \
    > logs/debug/triplet_semi_gpu3.log 2>&1 &
PID_TRIPLET_SEMI=$!
echo "  [GPU 3] Triplet Semi-Hard (PID: $PID_TRIPLET_SEMI)"

echo ""
echo "All jobs launched. Waiting for completion..."
echo ""

# Wait for all jobs and capture exit codes
wait $PID_CE
EXIT_CE=$?

wait $PID_SUPCON
EXIT_SUPCON=$?

wait $PID_TRIPLET_HARD
EXIT_TRIPLET_HARD=$?

wait $PID_TRIPLET_SEMI
EXIT_TRIPLET_SEMI=$?

# =============================================================================
# Phase 3: Results Summary
# =============================================================================
echo ""
echo "=========================================="
echo "Phase 3: Results Summary"
echo "=========================================="

# Print exit codes
echo ""
echo "Exit Codes:"
echo "  CE Baseline:     $EXIT_CE"
echo "  SupCon:          $EXIT_SUPCON"
echo "  Triplet Hard:    $EXIT_TRIPLET_HARD"
echo "  Triplet Semi:    $EXIT_TRIPLET_SEMI"

# Function to extract final metrics from log
extract_metrics() {
    local LOG_FILE=$1
    local MODEL_NAME=$2
    
    if [ -f "$LOG_FILE" ]; then
        # Try to extract last accuracy line
        LAST_ACC=$(grep -E "(test_acc|kNN_acc)" "$LOG_FILE" | tail -1 || echo "N/A")
        echo "  $MODEL_NAME: $LAST_ACC"
    else
        echo "  $MODEL_NAME: Log file not found"
    fi
}

echo ""
echo "Training Metrics (from logs):"
extract_metrics "logs/debug/ce_gpu0.log" "CE Baseline"
extract_metrics "logs/debug/supcon_gpu1.log" "SupCon"
extract_metrics "logs/debug/triplet_hard_gpu2.log" "Triplet Hard"
extract_metrics "logs/debug/triplet_semi_gpu3.log" "Triplet Semi"

# Check saved models
echo ""
echo "Saved Models:"
for dir in results/models/debug/*/; do
    MODEL_COUNT=$(find "$dir" -name "*.pt" 2>/dev/null | wc -l)
    echo "  $dir: $MODEL_COUNT checkpoint(s)"
done

# Overall status
echo ""
echo "=========================================="
TOTAL_ERRORS=$((EXIT_CE + EXIT_SUPCON + EXIT_TRIPLET_HARD + EXIT_TRIPLET_SEMI))
if [ $TOTAL_ERRORS -eq 0 ]; then
    echo "STATUS: ALL TESTS PASSED"
else
    echo "STATUS: SOME TESTS FAILED (errors: $TOTAL_ERRORS)"
fi
echo "End time: $(date)"
echo "=========================================="

# Return overall status
exit $TOTAL_ERRORS
