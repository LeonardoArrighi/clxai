# CLXAI: Contrastive Learning for XAI Faithfulness Evaluation

## Research Question

> Are neural networks trained using Contrastive Learning offering a natural way of evaluating XAI faithfulness via pixel flipping, without necessitating additional data augmentation or fine-tuning steps?

## Hypothesis

SCL-trained models produce embeddings where perturbed inputs move predictably and continuously away from their original cluster position, enabling reliable pixel-flipping faithfulness evaluation without fine-tuning.

## Quick Start (Leonardo HPC)

```bash
# 1. Load modules and activate environment
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo_scratch/fast/CNHPC_1905882/clxai/clxai_env/bin/activate
cd /leonardo_scratch/fast/CNHPC_1905882/clxai

# 2. Train CE baseline
sbatch scripts/train_ce.sh

# 3. Train SCL model
sbatch scripts/train_scl.sh

# 4. Run XAI evaluation
sbatch scripts/run_xai_eval.sh
```

## Project Structure

```
clxai/
├── src/
│   ├── models/          # ResNet-18, classifiers
│   ├── training/        # CE and SCL training
│   ├── xai/             # Saliency, pixel flipping
│   ├── analysis/        # Embedding analysis, metrics
│   └── utils/           # Data loading, utilities
├── configs/             # YAML configurations
├── scripts/             # SLURM job scripts
├── notebooks/           # Analysis notebooks
└── results/             # Models, embeddings, figures
```

## Models Compared

| Model | Training Loss | Classification Method |
|-------|---------------|----------------------|
| CE-Baseline | Cross-Entropy | Softmax output |
| SCL-kNN | SupCon Loss | k-NN on embeddings |
| SCL-Linear | SupCon Loss | Linear probe |
| SCL-Mahalanobis | SupCon Loss | Mahalanobis distance |

## XAI Methods

- Vanilla Gradients
- Integrated Gradients
- GradCAM / Grad-CAM++
- LRP (Layer-wise Relevance Propagation)

## Metrics

- **AUC-Deletion**: Lower = more faithful
- **AUC-Insertion**: Higher = more faithful
- **Monotonicity**: Spearman correlation
- **Embedding Drift**: L2 distance from original
- **Trajectory Smoothness**: Second derivative

## References

- SupCon: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
- F-Fidelity: Zheng et al. "A Robust Framework for Faithfulness Evaluation" (2024)
- Pixel Flipping: Montavon et al. "Deep Taylor Decomposition" (2017)
