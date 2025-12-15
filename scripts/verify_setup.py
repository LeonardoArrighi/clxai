#!/usr/bin/env python3
"""
Verify CLXAI setup and dependencies.
"""

import sys
from pathlib import Path

print("=" * 50)
print("CLXAI Setup Verification")
print("=" * 50)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check imports
print("\nChecking dependencies...")

dependencies = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("sklearn", "Scikit-learn"),
    ("matplotlib", "Matplotlib"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
]

optional_deps = [
    ("captum", "Captum (XAI)"),
    ("pytorch_metric_learning", "PyTorch Metric Learning"),
    ("umap", "UMAP"),
    ("wandb", "Weights & Biases"),
]

all_ok = True

for module, name in dependencies:
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [MISSING] {name}")
        all_ok = False

print("\nOptional dependencies:")
for module, name in optional_deps:
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [MISSING] {name} (optional)")

# Check CUDA
print("\nCUDA availability:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available")
        print(f"       Device: {torch.cuda.get_device_name(0)}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [WARNING] CUDA not available")
except Exception as e:
    print(f"  [ERROR] {e}")

# Check project structure
print("\nProject structure:")
project_root = Path(__file__).parent.parent
dirs_to_check = [
    "src/models",
    "src/training",
    "src/xai",
    "src/analysis",
    "src/utils",
    "configs",
    "scripts",
]

for d in dirs_to_check:
    path = project_root / d
    if path.exists():
        print(f"  [OK] {d}/")
    else:
        print(f"  [MISSING] {d}/")
        all_ok = False

# Test imports from project
print("\nProject modules:")
sys.path.insert(0, str(project_root))

modules_to_test = [
    "src.models.resnet",
    "src.models.classifiers",
    "src.training.losses",
    "src.utils.data",
    "src.utils.metrics",
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"  [OK] {module}")
    except Exception as e:
        print(f"  [ERROR] {module}: {e}")
        all_ok = False

# Summary
print("\n" + "=" * 50)
if all_ok:
    print("Setup verification PASSED")
else:
    print("Setup verification FAILED - check missing items above")
print("=" * 50)
