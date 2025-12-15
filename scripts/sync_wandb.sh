#!/bin/bash
# Sync all offline wandb runs to the cloud
# Run this from the LOGIN NODE (which has internet access)

cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# Load environment
module load profile/deeplrn
module load cineca-ai/4.3.0
source clxai_env/bin/activate

echo "=================================================="
echo "Syncing wandb offline runs..."
echo "=================================================="

# Count pending runs (check both possible locations)
pending=$(find wandb -name "offline-run-*" -type d 2>/dev/null | wc -l)
echo "Found $pending offline run(s) to sync"
echo ""

if [ "$pending" -gt 0 ]; then
    # Sync all offline runs (handle both wandb/ and wandb/wandb/ structures)
    find wandb -name "offline-run-*" -type d -exec wandb sync {} \;
    
    echo ""
    echo "Sync complete!"
else
    echo "No offline runs to sync."
fi

echo "=================================================="
